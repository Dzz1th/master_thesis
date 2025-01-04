import json
from tqdm import tqdm 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from langchain_openai import OpenAIEmbeddings
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import ParameterGrid

from models.models import TripletLinearClassifier, TripletLinearClassifier2Layers

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

def get_label_distribution(labels):
    labels_ = np.array(labels)
    return {
        "1's": labels_[labels_ == 1].shape[0] / labels_.shape[0],
        "0.5's": labels_[labels_ == 0.5].shape[0] / labels_.shape[0],
        "0's": labels_[labels_ == 0].shape[0] / labels_.shape[0],
        "-1's": labels_[labels_ == -1].shape[0] / labels_.shape[0],
        "-0.5's": labels_[labels_ == -0.5].shape[0] / labels_.shape[0]
    }

def truncate_pairs(pairs, truncate_size):
    return [(pair[0][:truncate_size], pair[1][:truncate_size]) for pair in pairs]

def load_pairs_data_stratified(test_size: float = 0.25, random_seed = None):
    """
        This version of data loading read all the pairs and then splits them into train and test with stratification.

        Potential issue with this approach is that we will have same objects (but not same pairs) in train and test
            (altough we can have x1 < x2 < x3, and x1<x3 and x2<x3 in train, and x1<x3 in test).
        This could lead to the leakage in data and overfitting.
    """
    
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/text_embeddings.json", "r") as f:
        data = json.load(f)

    embeddings = data['embeddings']
    embeddings = {int(key): value for key, value in embeddings.items()}

    text_idxs = data['text_idxs']
    text_idxs = {key: int(value) for key, value in text_idxs.items()} 
    
    pairs_ranking = data['pairs_ranking']
    pairs_ranking = {tuple(json.loads(key)): value for key, value in pairs_ranking.items()}

    pairs = list(pairs_ranking.keys())
    pairs = [(embeddings[pair[0]], embeddings[pair[1]]) for pair in pairs]
    labels = list(pairs_ranking.values())

    train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=test_size, stratify=labels, random_state=random_seed)

    train_label_distribution = get_label_distribution(train_labels)
    test_label_distribution = get_label_distribution(test_labels)

    logging.info(f"Train label distribution: {train_label_distribution}")
    logging.info(f"Test label distribution: {test_label_distribution}")

    return train_pairs, train_labels, test_pairs, test_labels


def load_pairs_data(test_size: float = 0.25):
    """
        Load ranked pairs of texts and split into train and test.
        
        Returns pairs of the embeddings and their respective rankings.
    """
    with open("/Users/dzz1th/Job/mgi/Soroka/data/qa_data/text_embeddings.json", "r") as f:
        data = json.load(f)

    embeddings = data['embeddings']
    embeddings = {int(key): value for key, value in embeddings.items()}

    text_idxs = data['text_idxs']
    text_idxs = {key: int(value) for key, value in text_idxs.items()} 
    
    pairs_ranking = data['pairs_ranking']

    pairs_ranking = {tuple(json.loads(key)): value for key, value in pairs_ranking.items()}

    train_idxs, test_idxs = train_test_split(list(text_idxs.values()), test_size=test_size)

    train_pairs = {key: value for key, value in pairs_ranking.items() if key[0] in train_idxs and key[1] in train_idxs}
    test_pairs = {key: value for key, value in pairs_ranking.items() if key[0] in test_idxs and key[1] in test_idxs}

    logging.info(f"Train pairs: {len(train_pairs)}")
    logging.info(f"Test pairs: {len(test_pairs)}")

    train_embeddings = [(embeddings[key[0]], embeddings[key[1]]) for key in train_pairs.keys()]
    test_embeddings = [(embeddings[key[0]], embeddings[key[1]]) for key in test_pairs.keys()]
    train_labels = [value for value in train_pairs.values()]   
    test_labels = [value for value in test_pairs.values()]

    train_label_distribution = get_label_distribution(train_labels)
    test_label_distribution = get_label_distribution(test_labels)
    
    logging.info(f"Train label distribution: {train_label_distribution}")
    logging.info(f"Test label distribution: {test_label_distribution}")
    
    return train_embeddings, train_labels, test_embeddings, test_labels


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

def score_pairs(scores, labels, padding=0.2):
    """
        Scores - tensor of shape (2, n) of 2 scores of the first and second texts in pairs
        Labels - tensor of shape (n) of labels for the pairs
    """
    diff = scores[0] - scores[1]
    accurate = torch.zeros_like(labels, dtype=torch.bool)
    
    # Check conditions for each label
    # For label == 0.5 or -0.5
    mask_05 = (labels == 0.5) | (labels == -0.5)
    accurate[mask_05] = ((diff * labels.sign())[mask_05] >= 0.5 - padding) & ((diff * labels.sign())[mask_05] <= 1.0)
    
    # For label == 1 or -1
    mask_1 = (labels == 1) | (labels == -1)
    accurate[mask_1] = (diff * labels.sign())[mask_1] > 1.0
    
    # For label == 0
    mask_0 = (labels == 0)
    accurate[mask_0] = (diff[mask_0] >= -0.5+padding) & (diff[mask_0] <= 0.5-padding)
    
    # Compute accuracy as the mean of accurate pairs
    accuracy = accurate.float().mean().item()
    
    return accuracy

def analyze_pair_errors(scores, labels, padding=0.25):
    """
    Analyze errors in pair scoring and print misclassified types.
    
    Args:
    - x1: Tensor of scores for the first element in each pair.
    - x2: Tensor of scores for the second element in each pair.
    - labels: Tensor of labels indicating the desired relationship between x1 and x2.
    """
    # Calculate the difference
    diff = scores[0] - scores[1]

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
            if diff[i] * labels[i].sign() < 0:
                hard_05 += 1
                #logging.info(f"Pair {i} with label {labels[i]} hard error: diff = {diff[i].item()}")
            else:
                if not (0.5 - padding <= diff[i] * labels[i].sign() <= 0.5 + padding):
                    soft_05 += 1
                    #logging.info(f"Pair {i} with label {labels[i]} soft error: diff = {diff[i].item()}")
        
        elif labels[i] in [1, -1]:
            if diff[i] * labels[i].sign() < 0.5:
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
    def __init__(self, triplets, labels, truncate_size=2048):
        self.triplets = triplets
        self.labels = labels
        self.truncate_size = truncate_size

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        hawkish, neutral, dovish = self.triplets[idx]
        if self.truncate_size != -1:
            hawkish = hawkish[:self.truncate_size]
            neutral = neutral[:self.truncate_size]
            dovish = dovish[:self.truncate_size]
        label = self.labels[idx]
        return (torch.tensor(hawkish, dtype=torch.float32),
                torch.tensor(neutral, dtype=torch.float32),
                torch.tensor(dovish, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))
    
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
    
class CustomRankingLoss(nn.Module):
    """
        This loss works as the following:
            1. For labels 1 and -1, it computes the standart margin ranking loss:
                loss = max(0, margin - (f_1 - f_2)) - means we penalize if the difference is lower than 1.
            2. For labels 0.5 and -0.5, we penalize if the difference is outside of the range 0.5 +- self.threshold :
                loss = max(0, diff - (0.5 + self.padding)) + max(0, (0.5 - self.padding) - diff)
            3. If the label is 0, we penalize if the difference is bigger than threshold:
                loss = max(0, (-0.5 + self.padding) - diff) + max(0, diff - (0.5 - self.padding))

            2. If the label is 0, it means that we want this pair to be close to each other, and we penalize it if the difference is bigger than threshold.
    """

    def __init__(self, padding=0.2, loss_weights=[1, 1]):
        """
            Args:
                padding - padding from the 0.5 when we start to penalize incorrect scoring
                loss_weights - weights for the loss of the different types of labels
        """

        super(CustomRankingLoss, self).__init__()
        self.padding = padding
        self.loss_weights = loss_weights

    def forward(self, f_first, f_second, labels):
        """
            f_first, f_second - tensors of scores of the first and second texts in pairs
            labels - margins for the first and second texts that obtained through LLM model

        """
        diff = f_first - f_second
        loss = torch.zeros_like(labels)
        diff = diff * labels.sign()


        mask_05 = (labels == 0.5) | (labels == -0.5)
        loss_05 = torch.relu(diff - 1) + torch.relu((0.5 - self.padding) - diff)

        mask_1 = (labels == 1) | (labels == -1)
        loss_1 = torch.relu(1 - diff)

        mask_0 = (labels == 0)
        loss_0 = torch.relu((-0.5 + self.padding) - diff) + torch.relu(diff - (0.5 - self.padding))

        loss = self.loss_weights[0] * loss_05[mask_05].sum() + self.loss_weights[1] * loss_1[mask_1].sum() + loss_0[mask_0].sum()

        return loss


def create_dataloaders(train_pairs, train_labels, test_pairs, test_labels):
    train_dataset = PairDataset(train_pairs, train_labels)
    test_dataset = PairDataset(test_pairs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, test_loader

def train_model_with_early_stopping(train_loader, test_loader,
                                    input_dim,
                                    epochs=100,
                                    lr=1e-3,
                                    patience=300,
                                    n_layers=1,
                                    loss_padding=0.25,
                                    loss_weights=[1, 1]):
    model = TripletLinearClassifier(input_dim=input_dim, num_layers=n_layers)
    criterion = CustomRankingLoss(padding=loss_padding, loss_weights=loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            first, second, label = batch # dataset of pair and label

            f_first = model(first)
            f_second = model(second)

            loss = criterion(f_first, f_second, label)
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
                f_first = model(first)
                f_second = model(second)

                val_loss += criterion(f_first, f_second, label)

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

def compute_metrics(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            first, second, label = batch
            f_first = model(first)
            f_second = model(second)

            predictions = torch.stack((f_first, f_second), dim=0)
            all_predictions.append(predictions)
            all_labels.append(label)

    all_predictions = torch.cat(all_predictions, dim=1)
    all_labels = torch.cat(all_labels)

    score_acc = score_pairs(all_predictions, all_labels)

    logging.info(f"Score Accuracy: {score_acc:.4f}")

    errors_analysis = analyze_pair_errors(all_predictions, all_labels)

    return {
        'score_acc': score_acc,
        **errors_analysis
    }

def pipeline(
        train_pairs: List[tuple], 
        train_labels: List[int],
        test_pairs: List[tuple], 
        test_labels: List[int],
        pca_components: int = 500,
        truncate_size: int = 2048,
        epochs=2500, 
        patience=100,
        lr=3e-2,
        n_layers=1,
        loss_padding=0.25,
        loss_weights=[1, 1]):
    
    
    train_pairs = truncate_pairs(train_pairs, truncate_size)
    test_pairs = truncate_pairs(test_pairs, truncate_size)

    if pca_components is not None:
        pca = pca_pairs(train_pairs, n_components=pca_components)
        train_pairs = pca_transform_pairs(train_pairs, pca)
        test_pairs = pca_transform_pairs(test_pairs, pca)
        input_dim = pca_components

    train_loader, test_loader = create_dataloaders(train_pairs, train_labels, test_pairs, test_labels)
    model = train_model_with_early_stopping(train_loader, test_loader, 
                                            input_dim=input_dim,
                                            epochs=epochs, 
                                            lr=lr,
                                            patience=patience, 
                                            n_layers=n_layers, 
                                            loss_padding=loss_padding,
                                            loss_weights=loss_weights)
    metrics = compute_metrics(model, test_loader)

    return metrics

training_config = {
    "pca_components": 800, 
    "truncate_size": 1600,
    "epochs": 2500,
    "patience": 200,
    "lr": 1e-2,
    "n_layers": 1, 
    "loss_padding": 0.0,
    "loss_weights": [1, 1]
}

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


parameters_grid = {
    'truncate_size': [512, 1024, 2048, 3072],
    'lr': [1e-3, 5e-3, 1e-2, 3e-2],
    'n_layers': [1, 2],
    'loss_padding': [0.2, 0.25, 0.4],
}

grid = ParameterGrid(parameters_grid)

if __name__ == "__main__":
    # Load data
    for seed in range(10):
        train_pairs, train_labels, test_pairs, test_labels = load_pairs_data_stratified(test_size=0.2, random_seed=seed)
        metrics = pipeline(train_pairs, train_labels, test_pairs, test_labels, **training_config)


# 0.55, 38/153
# 0.6, 26/153
# 0.575, 34/153
#0.58, 30/153
#  0.5850, 27/153
#  0.59, 32/153
# 0.6 , 28/153
# 0.585, 27/153