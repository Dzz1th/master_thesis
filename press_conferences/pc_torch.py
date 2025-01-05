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

from models.models import TripletLinearClassifier, PairwiseLinearClassifier
from models.loss import CustomRankingLoss

from data_loading import (load_pairs_data, 
                         load_pairs_data_stratified, 
                         load_triplets_data,
                         load_pre_splited_pairs)

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
    return [(pair[0][:truncate_size], pair[1][:truncate_size]) for pair in pairs]

def truncate_triplets(triplets, truncate_size):
    return [(triplet[0][:truncate_size], triplet[1][:truncate_size], triplet[2][:truncate_size]) for triplet in triplets]


def pca_pairs(pairs: List[tuple], n_components: int):
    all_pairs = np.array([pair[0] for pair in pairs] + [pair[1] for pair in pairs])
    pca = PCA(n_components=n_components)
    pca.fit(all_pairs)
    return pca

def pca_transform_pairs(pairs: List[tuple], pca: PCA):
    all_pairs = np.array([pair[0] for pair in pairs] + [pair[1] for pair in pairs])
    transformed_pairs = pca.transform(all_pairs)
    first_half = transformed_pairs[:len(pairs)]
    second_half = transformed_pairs[len(pairs):]
    transformed_pairs = [(first_half[i], second_half[i]) for i in range(len(pairs))]

    return transformed_pairs

def pca_transform_triplets(triplets, pca):
    all_embeddings = np.array([triplet[0] for triplet in triplets] + [triplet[1] for triplet in triplets] + [triplet[2] for triplet in triplets])
    transformed_triplets = pca.transform(all_embeddings)
    hawkish, neutral, dovish = transformed_triplets[:len(triplets)], transformed_triplets[len(triplets):len(triplets)*2], transformed_triplets[len(triplets)*2:]
    transformed_triplets = [(hawkish[i], neutral[i], dovish[i]) for i in range(len(triplets))]

    return transformed_triplets

def ranking_loss(f_hawkish, f_dovish, f_anchor, margin=1.0):
    loss = torch.relu(margin - (f_hawkish - f_anchor)) + torch.relu(margin - (f_dovish - f_anchor))
    return loss.sum()

def accuracy(predictions, labels):
    """
        Predictions are a tensor of shape (3, n) of 3 predictions scores:
            hawkish, neutral, dovish.
        If label is 1, we check prediction for hawkish and compare it to the label, 
        if label is 0, we check prediction for dovish and compare it to the label.

    """
    results = []
    for i in range(len(labels)):
        if labels[i] == 1:
            results.append(predictions[0, i] > 0)
        if labels[i] == 0:
            results.append(predictions[2, i] < 0)
    return results

def main_triplet_accuracy(predictions):
    """
        Predictions are a tensor of shape (3, n) of 3 predictions scores:
            hawkish, neutral, dovish.
        We check whether hawkish > dovish.

        We check whether hawkish > dovish for an entire triplet.
    """
    accuracy_tensor = (predictions[0] > predictions[2])
    accuracy_tensor = accuracy_tensor.float()
    return accuracy_tensor.mean()

def triplet_accuracy(predictions):
    """
        Predictions are a tensor of shape (3, n) of 3 predictions scores:
            hawkish, neutral, dovish.
        We check whether hawkish > neutral > dovish.

        We check exact order for an entire triplet.
        
    """
    accuracy_tensor = (predictions[0] > predictions[1]) & (predictions[1] > predictions[2])
    accuracy_tensor = accuracy_tensor.float()
    return accuracy_tensor.mean()

def analyze_triplet_errors(predictions):
    hawkish = predictions[0]
    neutral = predictions[1]
    dovish = predictions[2]
    
    # Count different types of ordering violations
    hawkish_neutral = (hawkish <= neutral).sum().item()
    neutral_dovish = (neutral <= dovish).sum().item()
    hawkish_dovish = (hawkish <= dovish).sum().item()
    
    total_triplets = len(hawkish)
    
    logging.info(f"Error Analysis:")
    logging.info(f"Hawkish <= Neutral: {hawkish_neutral}/{total_triplets} ({hawkish_neutral/total_triplets:.2%})")
    logging.info(f"Neutral <= Dovish: {neutral_dovish}/{total_triplets} ({neutral_dovish/total_triplets:.2%})")
    logging.info(f"Hawkish <= Dovish: {hawkish_dovish}/{total_triplets} ({hawkish_dovish/total_triplets:.2%})")

def score_pairs(scores, labels):
    """
        Scores - tensor of shape (n) of the difference between the first and second texts in pairs
        Labels - tensor of shape (n) of labels for the pairs
    """
    diff = scores
    accurate = torch.zeros_like(labels, dtype=torch.bool)
    
    # Check conditions for each label
    # For label == 0.5 or -0.5
    mask_05 = (labels == 0.5) | (labels == -0.5)
    accurate[mask_05] = ((diff * labels.sign())[mask_05] >= 0) & ((diff * labels.sign())[mask_05] <= 1.0)
    
    # For label == 1 or -1
    mask_1 = (labels == 1) | (labels == -1)
    accurate[mask_1] = (diff * labels.sign())[mask_1] >= 1.0
    
    # For label == 0
    mask_0 = (labels == 0)
    accurate[mask_0] = (diff[mask_0] >= -0.5) & (diff[mask_0] <= 0.5)
    
    # Compute accuracy as the mean of accurate pairs
    accuracy = accurate.float().mean().item()
    
    return accuracy

def analyze_pair_errors(scores, labels, padding=0.25):
    """
    Analyze errors in pair scoring and print misclassified types.
    
    Args:
        scores - tensor of shape (n) of the difference between the first and second texts in pairs
        labels - tensor of shape (n) of the labels for the pairs
    """
    # Calculate the difference
    diff = scores

    #Hard Errors
    hard_05 = 0
    hard_1 = 0
    hard_0 = 0
    
    # Initialize counters for misclassified types
    soft_05 = 0
    soft_1 = 0
    soft_0 = 0
    
    # Check conditions for each label
    for i in range(len(labels)):
        if labels[i] in [0.5, -0.5]:
            if diff[i] * labels[i].sign() < 0 or diff[i] * labels[i].sign() > 1:
                hard_05 += 1
                #logging.info(f"Pair {i} with label {labels[i]} hard error: diff = {diff[i].item()}")
            else:
                if not (0.5 - padding <= diff[i] * labels[i].sign() <= 0.5 + padding):
                    soft_05 += 1
                    #logging.info(f"Pair {i} with label {labels[i]} soft error: diff = {diff[i].item()}")
        
        elif labels[i] in [1, -1]:
            if diff[i] * labels[i].sign() < 0.0:
                hard_1 += 1
                #logging.info(f"Pair {i} with label {labels[i]} hard error: diff = {diff[i].item()}")
            else:
                if not (diff[i] * labels[i].sign() > 1.0):
                    soft_1 += 1
                    #logging.info(f"Pair {i} with label {labels[i]} soft error: diff = {diff[i].item()}")
        
        elif labels[i] == 0:
            if diff[i] * labels[i].sign() > 0.5:
                hard_0 += 1
                #logging.info(f"Pair {i} with label {labels[i]} hard error: diff = {diff[i].item()}")
            if not (-padding <= diff[i] <= padding):
                soft_0 += 1
                #logging.info(f"Pair {i} with label {labels[i]} soft error: diff = {diff[i].item()}")
    
    total_05 = len([label for label in labels if label in [0.5, -0.5]])
    total_1 = len([label for label in labels if label in [1, -1]])
    total_0 = len([label for label in labels if label == 0])
    # Print summary of misclassifications
    logging.info(f"Total pairs: {len(labels)}")
    logging.info(f"Total hard errors with label 0.5 or -0.5: {hard_05} Out of {total_05}")
    logging.info(f"Total hard errors with label 1 or -1: {hard_1} Out of {total_1}")
    logging.info(f"Total hard errors with label 0: {hard_0} Out of {total_0}")

    logging.info(f"Total soft errors with label 0.5 or -0.5: {soft_05} Out of {total_05}")
    logging.info(f"Total soft errors with label 1 or -1: {soft_1} Out of {total_1}")
    logging.info(f"Total soft errors with label 0: {soft_0} Out of {total_0}")

    return {
        'hard_05': hard_05,
        'hard_1': hard_1,
        'hard_0': hard_0,
        'soft_05': soft_05,
        'soft_1': soft_1,
        'soft_0': soft_0
    }



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
                                    model_type: t.Literal['default', 'pairwise'] = 'default'):
    
    if model_type == 'default':
        model = TripletLinearClassifier(input_dim=input_dim, num_layers=n_layers)
    elif model_type == 'pairwise':
        model = PairwiseLinearClassifier(input_dim=input_dim, n_layers=n_layers)

    criterion = CustomRankingLoss(loss_weights=loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            first, second, label = batch # dataset of pair and label

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

def compute_metrics(model, data_loader, model_type: t.Literal['default', 'pairwise'] = 'default'):
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

            predictions = diff
            all_predictions.append(predictions)
            all_labels.append(label)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels)

    score_acc = score_pairs(all_predictions, all_labels)

    errors_analysis = analyze_pair_errors(all_predictions, all_labels)

    return {
        'score_acc': score_acc,
        **errors_analysis
    }


def pipeline(
        train_pairs: List[tuple], 
        train_labels: List[int],
        val_pairs: List[tuple], 
        val_labels: List[int],
        test_pairs: List[tuple],
        test_labels: List[int],
        model_type: t.Literal['default', 'pairwise'] = 'default',
        pca_components: int = 500,
        truncate_size: int = 2048,
        epochs=2500, 
        patience=100,
        lr=3e-2,
        n_layers=1,
        loss_weights=[1, 1],):
    
    train_pairs = truncate_pairs(train_pairs, truncate_size)
    val_pairs = truncate_pairs(val_pairs, truncate_size)
    test_pairs = truncate_pairs(test_pairs, truncate_size)
    
    input_dim = truncate_size

    if pca_components is not None:
        pca = pca_pairs(train_pairs, n_components=pca_components)
        train_pairs = pca_transform_pairs(train_pairs, pca)
        val_pairs = pca_transform_pairs(val_pairs, pca)
        test_pairs = pca_transform_pairs(test_pairs, pca)
        input_dim = pca_components

    train_loader = create_dataloader(train_pairs, train_labels)
    val_loader = create_dataloader(val_pairs, val_labels)
    test_loader = create_dataloader(test_pairs, test_labels)

    model = train_model_with_early_stopping(train_loader, val_loader, 
                                            input_dim=input_dim,
                                            epochs=epochs, 
                                            lr=lr,
                                            patience=patience, 
                                            n_layers=n_layers,
                                            loss_weights=loss_weights,
                                            model_type=model_type)
    
    val_metrics = compute_metrics(model, val_loader, model_type=model_type)

    logging.info(f"Validation Accuracy: {val_metrics['score_acc']:.4f}")
    logging.info(f"Validation Hard Errors: {val_metrics['hard_05']}, {val_metrics['hard_1']}, {val_metrics['hard_0']}")
    logging.info(f"Validation Soft Errors: {val_metrics['soft_05']}, {val_metrics['soft_1']}, {val_metrics['soft_0']}")

    test_metrics = compute_metrics(model, test_loader, model_type=model_type)
    logging.info(f"Test Accuracy: {test_metrics['score_acc']:.4f}")
    logging.info(f"Test Hard Errors: {test_metrics['hard_05']}, {test_metrics['hard_1']}, {test_metrics['hard_0']}")
    logging.info(f"Test Soft Errors: {test_metrics['soft_05']}, {test_metrics['soft_1']}, {test_metrics['soft_0']}")


    return model, val_metrics,test_metrics

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
    'model_type': 'pairwise',
    "pca_components": None, 
    "truncate_size": 3072,
    "epochs": 1500,
    "patience": 100,
    "lr": 1e-2,
    "n_layers": 2,
    "loss_weights": [1, 1] #weigth of 0.5 loss and 1.0 loss
}

grid = ParameterGrid(parameters_grid)

if __name__ == "__main__":
    # Load data
    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = load_pre_splited_pairs(validation_size=0.25)

    model, val_metrics, test_metrics = pipeline(train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels, **training_config)
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