from engine.logger import make_logger
from engine.src.models.pairwise_classifier import LinearRankNet
from engine.config import Config

import mlflow 
import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any

from imblearn.under_sampling import RandomUnderSampler

from engine.src.utils.graph_utils import ensure_transitivity, reconstruct_pairs


mlflow.set_tracking_uri(uri="http://localhost:5001")

logger = make_logger(__name__)


def load_data(config, task, subtask):
    ranked_pairs_dir = os.path.join(config.output_dir, "ranked_pairs")
    pairs_file = os.path.join(ranked_pairs_dir, f"{task}_{subtask}_ranked_pairs.json")

    if not os.path.exists(pairs_file):
        logger.error(f"Ranked pairs file not found: {pairs_file}")
        return None, None
    
    with open(pairs_file, 'r') as f:
        pair_data = json.load(f)

    ## Download the original data with statement texts
    try:
        df = pd.read_csv(os.path.join(config.output_dir, "extracted_statements.csv"))
    except FileNotFoundError:
        logger.error("Extracted statements not found")
        return None, None
    
    return pair_data, df 

def truncate_by_window(pairs, 
                        rankings,
                        doc_to_date: Dict[str, str], 
                        window_size: int = 3) -> List[Tuple[str, str]]:
    """Create pairs of documents that are chronologically close to each other.
    
    Args:
        doc_to_date: Dictionary mapping documents to their dates
        max_distance: Maximum number of positions between documents in the sorted sequence
        
    Returns:
        List of document pairs (doc1, doc2) where doc1 is chronologically before doc2
        and they are at most max_distance positions apart in the sorted sequence
    """
    doc_dates = [(doc, datetime.strptime(date.split()[0], '%Y-%m-%d')) for doc, date in doc_to_date.items()]
    
    sorted_docs = sorted(doc_dates, key=lambda x: x[1])
    docs_idxs = {doc: i for i, (doc, _) in enumerate(sorted_docs)}
    
    final_pairs = []
    final_rankings = []

    for i in range(len(pairs)):
        doc1, doc2 = pairs[i]
        if abs(docs_idxs[doc1] - docs_idxs[doc2]) <= window_size:
            final_pairs.append((doc1, doc2))
            final_rankings.append(rankings[i])
        
    return final_pairs, final_rankings

def compute_class_distribution(rankings):
    train_positives = len([item for item in rankings if item == 1]) / len(rankings)
    train_negatives = len([item for item in rankings if item == -1]) / len(rankings)
    train_zeros = len([item for item in rankings if item == 0]) / len(rankings)
    return train_positives, train_negatives, train_zeros

def undersample_data(train_pairs, train_rankings, has_neutral_class=True):
    X_train = np.arange(len(train_pairs)).reshape(-1, 1)
    y_train = np.array(train_rankings)

    # Calculate target distribution
    # Configure undersampler to have equal number of samples per class
    if has_neutral_class:
        class_counts = np.bincount(y_train + 1)  # Shift -1,0,1 to 0,1,2
        min_class_count = min(class_counts)
        sampling_strategy = {-1: min_class_count, 0: min_class_count, 1: min_class_count}
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    else:
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        sampling_strategy = {1: class_dist[1], -1: class_dist[-1]}
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    
    # Apply undersampling
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    train_pairs = [train_pairs[i] for i in X_resampled.ravel()]
    train_rankings = y_resampled.tolist()

    resampled_positives = len([item for item in train_rankings if item == 1]) / len(train_rankings)
    resampled_negatives = len([item for item in train_rankings if item == -1]) / len(train_rankings)
    resampled_zeros = len([item for item in train_rankings if item == 0]) / len(train_rankings)
    logger.info(f"After undersampling - Train classes distribution Positives/Negatives/Zeros: {resampled_positives}, {resampled_negatives}, {resampled_zeros}")
    logger.info(f"Training data size reduced from {len(y_train)} to {len(y_resampled)} pairs")

    return train_pairs, train_rankings


def data_pipeline(pair_data, df, task, 
                  last_year:int,
                  base_year:int,
                  purge_cycles: bool=True):

    column_mapping = {
        'employment': 'employment_summary',
        'inflation': 'inflation_summary',  # Inflation data is in employment summary
        'forward_guidance': 'forward_guidance_summary',
        'interest_rate': 'interest_rate_summary',
        'balance_sheet': 'balance_sheet_summary'
    }

    if task not in column_mapping:
        logger.error(f"Unknown task: {task}")
        return None
    
    text_column = column_mapping[task]
    
    # Split data into train and validation sets
    # Use chronological split based on dates
    cutoff_year = last_year
    starting_year = base_year
    
    # Create a mapping from documents to their dates
    doc_to_date = {}
    for _, row in df.iterrows():
        doc_to_date[row[text_column]] = row["date"]

    train_indices, val_indices, test_indices = [], [], []

    ## Purge pairs data to eliminate cycles
    pair_data = [(item["doc_1"], item["doc_2"], item["ranking"]) for item in pair_data]
    if purge_cycles:
        result = ensure_transitivity(pair_data)
        pair_data = reconstruct_pairs(result["final_graph"], result["removed_nodes"], result["union_find"])

    pairs = [(item[0], item[1]) for item in pair_data]
    rankings = [item[2] for item in pair_data]
    
    train_year_indices = [[] for _ in range(cutoff_year - starting_year + 1)]
    for i, (doc1, doc2) in enumerate(pairs):
        date1 = doc_to_date.get(doc1, "")
        date2 = doc_to_date.get(doc2, "")

        year_1 = int(date1.split('-')[0])
        year_2 = int(date2.split('-')[0])

        # Ensure no sample contamination from future data
        if min(year_1, year_2) == cutoff_year and max(year_1, year_2) == cutoff_year:
            val_indices.append(i)
        elif max(year_1, year_2)  >= starting_year and max(year_1, year_2) < cutoff_year:
            train_indices.append(i)
            train_year_indices[year_1 - starting_year].append(i)
            if year_1 != year_2:
                train_year_indices[year_2 - starting_year].append(i)
        else:
            test_indices.append(i)
    
    # Create train and validation sets
    train_pairs = [pairs[i] for i in train_indices]
    train_rankings = [rankings[i] for i in train_indices]

    ## Group train pairs by year to evaluate later yearly distribution of errors
    train_year_pairs = {}
    train_year_rankings = {}
    for year in range(starting_year, cutoff_year):
        year_pairs = [pairs[i] for i in train_year_indices[year - starting_year]]
        year_rankings = [rankings[i] for i in train_year_indices[year - starting_year]]
        train_year_pairs[year] = year_pairs
        train_year_rankings[year] = year_rankings


    val_pairs = [pairs[i] for i in val_indices]
    val_rankings = [rankings[i] for i in val_indices]
    test_pairs = [pairs[i] for i in test_indices]
    test_rankings = [rankings[i] for i in test_indices]

    train_positives, train_negatives, train_zeros = compute_class_distribution(train_rankings)
    val_positives, val_negatives, val_zeros = compute_class_distribution(val_rankings)
    test_positives, test_negatives, test_zeros = compute_class_distribution(test_rankings)

    logger.info(f"Split data into {len(train_pairs)} train and {len(val_pairs)} validation pairs")
    logger.info(f"Train classes distribution Positives/Negatives/Zeros: {train_positives}, {train_negatives}, {train_zeros}")
    logger.info(f"Val classes distribution Positives/Negatives/Zeros: {val_positives}, {val_negatives}, {val_zeros}")
    logger.info(f"Test classes distribution Positives/Negatives/Zeros: {test_positives}, {test_negatives}, {test_zeros}")

    return (train_pairs, train_rankings), (val_pairs, val_rankings), (test_pairs, test_rankings)

async def train_pairwise_classifier(config: Config, task: str, subtask: str, llm_client: Any):
    """Train a pairwise classifier for a given task and subtask.
    
    Args:
        config: Configuration object.
        task: The main task (e.g., 'employment').
        subtask: The subtask (e.g., 'level').
        llm_client: The LLM client (unused, for compatibility).
    """
    logger.info(f"Training pairwise classifier for task: {task}, subtask: {subtask}")

    # Load data
    data_path = os.path.join(config.output_dir, "ranked_statements.csv")
    if not os.path.exists(data_path):
        logger.error(f"Ranked data not found at {data_path}. Please run ranking first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Filter for the specific task and subtask
    subtask_df = df[(df['task'] == task) & (df['subtask'] == subtask)].copy()
    
    if subtask_df.empty:
        logger.warning(f"No data found for task '{task}' and subtask '{subtask}'. Skipping training.")
        return

    # Get embeddings for all unique statements
    statements = pd.unique(subtask_df[['statement_a', 'statement_b']].values.ravel('K'))
    if llm_client is None:
        from engine.src.llms.llm_client import LLMClient
        llm_client = LLMClient(config)
        close_client = True
    else:
        close_client = False
        
    try:
        embeddings = await llm_client.get_embeddings(statements.tolist())
        embedding_map = dict(zip(statements, embeddings))
    finally:
        if close_client:
            await llm_client.aclose()

    # Create feature vectors for pairs
    X1 = np.array([embedding_map[s] for s in subtask_df['statement_a']])
    X2 = np.array([embedding_map[s] for s in subtask_df['statement_b']])
    y = subtask_df['ranking'].values
    
    # Train model
    model = LinearRankNet()
    model.fit(X1, X2, y)
    
    # Save model
    model_dir = os.path.join(config.output_dir, "models", task, subtask)
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "pairwise_model.pkl"))
    logger.info(f"Saved pairwise model for {task}/{subtask} to {model_dir}")
