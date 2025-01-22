import json
import typing as t
from tqdm import tqdm 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from langchain_openai import OpenAIEmbeddings
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import ParameterGrid
from sklearn.manifold import TSNE
import umap

from models.models import TripletLinearClassifierNLayers, PairwiseLinearClassifier
from models.loss import CustomRankingLoss

from metrics import pairs_accuracy, pairs_errors_decomposed, analyze_triplet_errors

from data_loading import (load_pairs_data, 
                         load_pairs_data_stratified, 
                         load_triplets_data,
                         load_pre_splited_pairs,
                         load_raw_qa_data,
                         load_time_grouped_pairs,
                         load_introductions_years_pairs_ranking)

import time 
import logging

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


openai_key = "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"

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    openai_api_key = openai_key
)

def truncate_pairs(pairs, truncate_size):
    return [(pair[0][truncate_size[0]:truncate_size[1]], pair[1][truncate_size[0]:truncate_size[1]]) for pair in pairs]

class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        hawkish, neutral, dovish = self.triplets[idx]

        return (torch.tensor(hawkish, dtype=torch.float32),
                torch.tensor(neutral, dtype=torch.float32),
                torch.tensor(dovish, dtype=torch.float32))
    
class PairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        first, second = self.pairs[idx]
        
        label = self.labels[idx]
        return (torch.tensor(first, dtype=torch.float32),
                torch.tensor(second, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))
    

def create_dataloader(pairs, labels):
    dataset = PairDataset(pairs, labels)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return loader

def train_model_with_early_stopping(train_loader, test_loader,
                                    input_dim,
                                    epochs=100,
                                    lr=1e-3,
                                    patience=300,
                                    n_layers=1,
                                    loss_weights=[1, 1],
                                    model_type: t.Literal['default', 'difference', 'pairwise'] = 'default'):
    
    
    if model_type == 'pairwise':
        model = PairwiseLinearClassifier(input_dim=input_dim, n_layers=n_layers)
    else:
        model = TripletLinearClassifierNLayers(input_dim=input_dim, num_layers=n_layers)

    criterion = CustomRankingLoss(loss_weights=loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            first, second, label = batch # dataset of pair and label

            if model_type == 'difference':
                doc_diff = first - second
                diff = model(doc_diff)

            if model_type == 'default':
                f_first = model(first)
                f_second = model(second)
                diff = f_first - f_second

            elif model_type == 'pairwise':
                diff = model(torch.cat((first, second), dim=1))

            loss = criterion(diff, label)
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in test_loader:
                first, second, label = batch
                if model_type == 'default':
                    f_first = model(first)
                    f_second = model(second)
                    diff = f_first - f_second

                elif model_type == 'pairwise':
                    diff = model(torch.cat((first, second), dim=1))

                elif model_type == 'difference':
                    doc_diff = first - second
                    diff = model(doc_diff)

                val_loss += criterion(diff, label)

            # Log the losses
            logging.info(f"Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break

    model.load_state_dict(best_model)
    return model

def compute_metrics(model, 
                    data_loader,
                     model_type: t.Literal['default', 'pairwise', 'difference'] = 'default',
                     metrics_fns: t.List[t.Callable] = []):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            first, second, label = batch
            if model_type == 'default':
                f_first = model(first)
                f_second = model(second)
                diff = f_first - f_second

            elif model_type == 'pairwise':
                diff = model(torch.cat((first, second), dim=1))

            elif model_type == 'difference':
                doc_diff = first - second
                diff = model(doc_diff)

            predictions = diff
            all_predictions.append(predictions)
            all_labels.append(label)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels)

    metrics = {}
    for metric_fn in metrics_fns:
        metrics[metric_fn.__name__] = metric_fn(all_predictions, all_labels)

    return metrics


def train_reducer(train_embeddings, 
                  train_labels,
                    method, truncate_size, **params):
    """
        Train a dimensionality reduction model.
        train_labels are required for UMAP with labels.
    """
    if method == 'pca':
        pca = PCA(n_components=truncate_size[1])
        pca.fit(train_embeddings)
        return pca
    
    elif method == 'umap':
        umap_compresser = umap.UMAP(n_components=truncate_size[1], **params)
        umap_compresser.fit(train_embeddings)
        return umap_compresser

def reduce_pairs(pairs, reducer, truncate_size):
    embeddings = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
    results = reducer.transform(embeddings)[:, truncate_size[0]:truncate_size[1]]

    first_half = results[:len(pairs)]
    second_half = results[len(pairs):]

    return [(first_half[i], second_half[i]) for i in range(len(pairs))]


def pipeline(
        train_pairs: List[tuple], 
        train_labels: List[int],
        test_pairs: List[tuple],
        test_labels: List[int],
        model_type: t.Literal['default', 'difference', 'pairwise'] = 'default',
        reducer: t.Optional[t.Any] = None,
        reduction_truncate_size: tuple = (0, 50),
        truncate_size: tuple = (0, 2048),
        epochs=2500, 
        patience=100,
        lr=3e-2,
        n_layers=1,
        loss_weights=[1, 1],
        **kwargs):
    
    train_pairs = truncate_pairs(train_pairs, truncate_size)
    test_pairs = truncate_pairs(test_pairs, truncate_size)
    input_dim = truncate_size[1] - truncate_size[0]

    
    if reducer is not None:
        train_pairs = reduce_pairs(train_pairs, reducer, reduction_truncate_size)
        test_pairs = reduce_pairs(test_pairs, reducer, reduction_truncate_size)

        input_dim = reduction_truncate_size[1] - reduction_truncate_size[0]

    train_loader = create_dataloader(train_pairs, train_labels)
    test_loader = create_dataloader(test_pairs, test_labels)

    model = train_model_with_early_stopping(train_loader, test_loader, 
                                            input_dim=input_dim,
                                            epochs=epochs, 
                                            lr=lr,
                                            patience=patience, 
                                            n_layers=n_layers,
                                            loss_weights=loss_weights,
                                            model_type=model_type)

    test_metrics = compute_metrics(model, test_loader, model_type=model_type, metrics_fns=[pairs_accuracy, pairs_errors_decomposed])
    logging.info(f"Test Accuracy: {test_metrics['pairs_accuracy']:.4f}")
    logging.info(f"Test Hard Errors: {test_metrics['pairs_errors_decomposed']['hard_05']}, {test_metrics['pairs_errors_decomposed']['hard_1']}, {test_metrics['pairs_errors_decomposed']['hard_0']}")
    logging.info(f"Test Soft Errors: {test_metrics['pairs_errors_decomposed']['soft_05']}, {test_metrics['pairs_errors_decomposed']['soft_1']}, {test_metrics['pairs_errors_decomposed']['soft_0']}")

    return model, test_metrics
    
    

def predict_triplets(triplets, model):
    hawkish = [triplet[0] for triplet in triplets]
    dovish = [triplet[1] for triplet in triplets]
    anchor = [triplet[2] for triplet in triplets]

    hawkish = torch.tensor(hawkish, dtype=torch.float32)
    dovish = torch.tensor(dovish, dtype=torch.float32)
    anchor = torch.tensor(anchor, dtype=torch.float32)

    hawkish_scores = model(hawkish)
    dovish_scores = model(dovish)
    anchor_scores = model(anchor)

    results = torch.stack((hawkish_scores, anchor_scores, dovish_scores), dim=0)

    return results

base_config = {
    "pca_components": None,
    "epochs": 2500,
    "patience": 200,
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training.log', filemode='w')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it for the console handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the root logger
logging.getLogger().addHandler(console_handler)

base_parameters = {
    "epochs": 3000,
    "patience": 200,
    "loss_weights": [1, 1],
}

parameters_grid = {
    'pca_components': [None, 300, 600],
    'truncate_size': [1024, 2048],
    'lr': [1e-2, 3e-2],
    'n_layers': [1, 2],
}


training_config = {
    "model_type": "pairwise",
    "reduction_method": None,
    "reduction_truncate_size": (0, 50),
    "truncate_size": [0, 3072],
    "epochs": 1500,
    "patience": 50,
    "lr": 1e-2,
    "n_layers": 2,
    "loss_weights": [1, 1] #weigth of 0.5 loss and 1.0 loss
}

grid = ParameterGrid(parameters_grid)

if __name__ == "__main__":
    # Load data
    # data = load_time_grouped_pairs()
    
    # metrics = {}
    # base_index = 5 #2011-2015 is the base training set
    # train_data = data[0:11]
    # test_data = data[11:]

    # train_pairs = sum([pairs for pairs, _ in train_data], [])
    # train_labels = sum([labels for _, labels in train_data], [])

    # test_pairs = sum([pairs for pairs, _ in test_data], [])
    # test_labels = sum([labels for _, labels in test_data], [])

    # train_embeddings = [pair[0] for pair in train_pairs] + [pair[1] for pair in train_pairs]
    # reducer = None
    # if training_config['reduction_method'] is not None:
    #     reducer = train_reducer(train_embeddings, train_labels, training_config['reduction_method'], training_config['reduction_truncate_size'])
    
    # model, test_metrics = pipeline(train_pairs, train_labels, test_pairs, test_labels, reducer=reducer, **training_config)
    # logging.info(f"Test Metrics: {test_metrics}")

    # ___________
    
    # for i in range(0, len(data)-base_index):
    #     train_data = data[0+i:base_index+i]
    #     test_data = data[base_index+i]

    #     train_pairs = sum([pairs for pairs, _ in train_data], [])
    #     train_labels = sum([labels for _, labels in train_data], [])

    #     train_embeddings = [pair[0] for pair in train_pairs] + [pair[1] for pair in train_pairs]
    #     reducer = None
    #     if training_config['reduction_method'] is not None:
    #         reducer = train_reducer(train_embeddings, train_labels, training_config['reduction_method'], training_config['reduction_truncate_size'])

    #     test_pairs = test_data[0]
    #     test_labels = test_data[1]

    #     logging.info(f"Start Training for {2011+i} - {2011+base_index+i} years")
    #     model, test_metrics = pipeline(train_pairs, train_labels, test_pairs, test_labels, reducer=reducer, **training_config)
    #     metrics[f"{2011+i}"] = test_metrics

    # logging.info(f"Metrics: {metrics}")

    #train_idx, test_idx = qa_df[qa_df['date'] <= '2020-01-01'].index, qa_df[qa_df['date'] > '2020-01-01'].index

    # reducer = None
    # if training_config['reduction_method'] is not None:
    #     reducer = train_reducer(train_embeddings, train_labels, training_config['reduction_method'], training_config['reduction_truncate_size'])

    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = load_pre_splited_pairs(validation_size=0.05)

    triplets = load_triplets_data(date_from='2020-01-01', date_to='2025-01-01')

    reducer = None
    train_embeddings = [pair[0] for pair in train_pairs] + [pair[1] for pair in train_pairs]
    if training_config['reduction_method'] is not None:
        reducer = train_reducer(train_embeddings, train_labels, training_config['reduction_method'], training_config['reduction_truncate_size'])

    model, test_metrics = pipeline(train_pairs, train_labels, test_pairs, test_labels, reducer=reducer, **training_config)

    predictions = predict_triplets(triplets, model)
    triplet_errors = analyze_triplet_errors(predictions)
    logging.info(f"Triplet Errors: {triplet_errors}")

    # results = {}
    # for params in tqdm(grid):
    #     model, val_metrics, test_metrics = pipeline(train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, **params, **base_parameters)
    #     logging.info(f"Params: {params}")
    #     logging.info(f"Validation Metrics: {val_metrics}")
    #     logging.info(f"Test Metrics: {test_metrics}")

    #     results[json.dumps(params)] = {
    #         "val_metrics": val_metrics,
    #         "test_metrics": test_metrics
    #     }

    #     with open("grid_search.json", "w") as f:
    #         json.dump(results, f)






# 0.55, 38/153
# 0.6, 26/153
# 0.575, 34/153
#0.58, 30/153
#  0.5850, 27/153
#  0.59, 32/153
# 0.6 , 28/153
# 0.585, 27/153