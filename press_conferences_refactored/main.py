#!/usr/bin/env python
"""
FED Statement Sentiment Analysis
Main script to orchestrate the sentiment analysis pipeline
"""
import argparse
import logging
import os
import json
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import make_scorer

from graph_utils import ensure_transitivity, reconstruct_pairs
from llm_cache import set_cache
from data_processor import StatementExtractor, StatementClassifier, StatementRanker
from single_classifier import SingleObjectClassifier
from pairwise_classifier import LinearRankNet
from evaluator import ModelEvaluator
from config import Config

from prompts.summarization_prompts import (
    sentiment_prompt,
    employment_prompt,
    inflation_prompt,
    forward_guidance_prompt,
    interest_rate_prompt,
    balance_sheet_prompt
)

from prompts.classification_prompts import (
    GUIDANCE_PROMPT,
    SENTIMENT_PROMPT,
    EMPLOYMENT_INFLATION_LEVELS_PROMPT,
    EMPLOYMENT_INFLATION_DYNAMICS_PROMPT,
    EMPLOYMENT_INFLATION_CONCERN_PROMPT,
    INTEREST_RATE_TRAJECTORY_PROMPT,
    BALANCE_SHEET_TRAJECTORY_PROMPT
)

from prompts.classification_prompts import (
    Guidance,
    Sentiment,
    EmploymentInflationLevels,
    EmploymentInflationDynamics,
    EmploymentInflationConcern,
    InterestRateTrajectory,
    BalanceSheetTrajectory
)

from prompts.ranking_prompts import (
    EMPLOYMENT_LEVEL_RANKING_PROMPT,
    EMPLOYMENT_DYNAMICS_RANKING_PROMPT,
    INFLATION_LEVEL_RANKING_PROMPT,
    INFLATION_DYNAMICS_RANKING_PROMPT,
    INTEREST_RATE_PROJECTION_RANKING_PROMPT,
    BALANCE_SHEET_PROJECTION_RANKING_PROMPT,
    FORWARD_GUIDANCE_RANKING_PROMPT
)

from prompts.ranking_prompts import (
    EmploymentLevelRankingResponse,
    EmploymentDynamicsRankingResponse,
    InflationLevelRankingResponse,
    InflationDynamicsRankingResponse,
    InterestRateProjectionRankingResponse,
    BalanceSheetProjectionRankingResponse,
    GuidanceRankingResponse
)

#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "master-thesis"

set_cache()

def setup_logging():
    """Configure logging for the application"""
    # Suppress HTTPX logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sentiment_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FED Statement Sentiment Analysis")
    
    # General arguments
    parser.add_argument(
        '--data_path', 
        type=str, 
        default="/Users/dzz1th/Job/mgi/Soroka/data/pc_data/summarized_data.csv",
        help='Path to the summarized data CSV file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="/Users/dzz1th/Job/mgi/Soroka/data/pc_data/",
        help='Directory to save output files'
    )
    parser.add_argument(
        '--last_year', 
        type=int, 
        default=2023,
        help='Last year to include in training data'
    )
    parser.add_argument(
        '--base_year', 
        type=int, 
        default=2016,
        help='Base year for dataset splitting'
    )
    
    # Pipeline mode arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['extract', 'classify', 'rank', 'train_classifier', 'train_ranker', 'predict', 'evaluate', 'clear_cache', 'all'],
        default='all',
        help='Pipeline mode'
    )
    
    # Task arguments
    parser.add_argument(
        '--task',
        type=str,
        choices=['sentiment', 'employment', 'inflation', 'forward_guidance', 'interest_rate', 'balance_sheet', 'all'],
        default='all',
        help='Task to process'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['logreg', 'svc', 'rf', 'three_way_ranker'],
        default='logreg',
        help='Type of model to train'
    )
    
    # Three-way ranker specific arguments
    parser.add_argument(
        '--ranker_margin',
        type=float,
        default=0.5,
        help='Margin for three-way ranker equal class'
    )
    parser.add_argument(
        '--ranker_learning_rate',
        type=float,
        default=0.01,
        help='Learning rate for three-way ranker'
    )
    parser.add_argument(
        '--ranker_epochs',
        type=int,
        help='Number of epochs for three-way ranker'
    )
    
    # Text processing arguments
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=5000,
        help='Maximum chunk size for statement extraction'
    )
    
    # Cache arguments
    parser.add_argument(
        '--cache_embeddings', 
        action='store_true',
        help='Cache embeddings to disk'
    )
    parser.add_argument(
        '--clear_cache_name',
        type=str,
        default=None,
        help='Name of specific cache to clear (used with --mode clear_cache)'
    )
    
    # API key arguments
    parser.add_argument(
        '--openai_key', 
        type=str,
        default=None,
        help='OpenAI API key (optional, will use environment var if not provided)'
    )
    
    return parser.parse_args()

def extract_statements(config, task):
    """Extract statements from raw texts
    
    Args:
        config: Configuration object
        task: Task to process ('sentiment', 'employment', 'forward_guidance', 'all')
        
    Returns:
        Dataframe with extracted statements
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting statements for task: {task}")
    
    # Load data
    df = pd.read_csv(config.data_path)
    
    logger.info(f"Extracting statements for task: {task}")
    if task in ['employment', 'all']:
        employment_extractor = StatementExtractor(
            config=config,
            extraction_prompt=employment_prompt,
            chunk_size=config.chunk_size
        )
        df = employment_extractor.process_statements_from_df(df, 'text', 'employment_summary')
        logger.info(f"Extracted employment statements")
    
    if task in ['inflation', 'all']:
        inflation_extractor = StatementExtractor(
            config=config,
            extraction_prompt=inflation_prompt,
            chunk_size=config.chunk_size
        )
        df = inflation_extractor.process_statements_from_df(df, 'text', 'inflation_summary')
        logger.info(f"Extracted inflation statements")
    if task in ['forward_guidance', 'all']:
        guidance_extractor = StatementExtractor(
            config=config,
            extraction_prompt=forward_guidance_prompt,
            chunk_size=config.chunk_size
        )
        df = guidance_extractor.process_statements_from_df(df, 'text', 'forward_guidance_summary')
        logger.info(f"Extracted forward guidance statements")
    if task in ['interest_rate', 'all']:
        interest_rate_extractor = StatementExtractor(
            config=config,
            extraction_prompt=interest_rate_prompt,
            chunk_size=config.chunk_size
        )
        df = interest_rate_extractor.process_statements_from_df(df, 'text', 'interest_rate_summary')
        logger.info(f"Extracted interest rate statements")

    if task in ['balance_sheet', 'all']:
        balance_sheet_extractor = StatementExtractor(
            config=config,
            extraction_prompt=balance_sheet_prompt,
            chunk_size=config.chunk_size
        )
        df = balance_sheet_extractor.process_statements_from_df(df, 'text', 'balance_sheet_summary')
        logger.info(f"Extracted balance sheet statements")

    # Save processed data
    output_path = os.path.join(config.output_dir, "extracted_statements.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved extracted statements to {output_path}")
    
    return df

def classify_statements(config, task):
    """Classify extracted statements
    
    Args:
        config: Configuration object
        task: Task to process ('sentiment', 'employment', 'forward_guidance', 'all')
        
    Returns:
        Dataframe with classified statements
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Classifying statements for task: {task}")
    
    # Load data
    try:
        df = pd.read_csv(os.path.join(config.output_dir, "extracted_statements.csv"))
    except FileNotFoundError:
        logger.info("Extracted statements not found, loading original data")
        df = pd.read_csv(config.data_path)
    
    # Process different tasks
    if task in ['sentiment', 'all']:
        sentiment_classifier = StatementClassifier(
            config=config,
            classification_prompt=SENTIMENT_PROMPT,
            output_schema=Sentiment
        )
        df = sentiment_classifier.process_statements_from_df(df, 'sentiment_summary', 'sentiment_class')
    
    if task in ['employment', 'all']:
        employment_levels_classifier = StatementClassifier(
            config=config,
            classification_prompt=EMPLOYMENT_INFLATION_LEVELS_PROMPT,
            output_schema=EmploymentInflationLevels
        )
        df = employment_levels_classifier.process_statements_from_df(df, 'employment_summary', 'employment_levels_class')
        employment_dynamics_classifier = StatementClassifier(
            config=config,
            classification_prompt=EMPLOYMENT_INFLATION_DYNAMICS_PROMPT,
            output_schema=EmploymentInflationDynamics
        )
        df = employment_dynamics_classifier.process_statements_from_df(df, 'employment_summary', 'employment_dynamics_class')

        employment_concern_classifier = StatementClassifier(
            config=config,
            classification_prompt=EMPLOYMENT_INFLATION_CONCERN_PROMPT,
            output_schema=EmploymentInflationConcern
        )
        df = employment_concern_classifier.process_statements_from_df(df, 'employment_summary', 'employment_concern_class')
        

    if task in ['forward_guidance', 'all']:
        guidance_classifier = StatementClassifier(
            config=config,
            classification_prompt=GUIDANCE_PROMPT,
            output_schema=Guidance
        )
        df = guidance_classifier.process_statements_from_df(df, 'forward_guidance_summary', 'guidance_class')

    if task in ['interest_rate', 'all']:
        interest_rate_classifier = StatementClassifier(
            config=config,
            classification_prompt=INTEREST_RATE_TRAJECTORY_PROMPT,
            output_schema=InterestRateTrajectory
        )
        df = interest_rate_classifier.process_statements_from_df(df, 'interest_rate_summary', 'interest_rate_class')

    if task in ['balance_sheet', 'all']:
        balance_sheet_classifier = StatementClassifier(
            config=config,
            classification_prompt=BALANCE_SHEET_TRAJECTORY_PROMPT,
            output_schema=BalanceSheetTrajectory
        )
        df = balance_sheet_classifier.process_statements_from_df(df, 'balance_sheet_summary', 'balance_sheet_class')
    
    # Add chairman feature
    df['chairman'] = df['date'].apply(lambda x: 0 if x < '2014-01-01' else 1 if x < '2018-01-01' else 2)
    
    # Save processed data
    output_path = os.path.join(config.output_dir, "classified_statements.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved classified statements to {output_path}")
    
    return df

def rank_statements(config, task):
    """Rank statements
    
    Args:
        config: Configuration object
        task: Task to rank ('employment', 'inflation', 'forward_guidance', 'interest_rate', 'balance_sheet')
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Ranking statements for task: {task}")

    # Load data
    df = pd.read_csv(os.path.join(config.output_dir, "extracted_statements.csv"))

    # Create output directory for ranked pairs
    ranked_pairs_dir = os.path.join(config.output_dir, "ranked_pairs")
    os.makedirs(ranked_pairs_dir, exist_ok=True)

    ranking_configs = {
        'employment': {
            'level': {
                'prompt': EMPLOYMENT_LEVEL_RANKING_PROMPT,
                'schema': EmploymentLevelRankingResponse,
                'column': 'employment_summary'
            },
            'dynamics': {
                'prompt': EMPLOYMENT_DYNAMICS_RANKING_PROMPT,
                'schema': EmploymentDynamicsRankingResponse,
                'column': 'employment_summary'
            }
        },
        'inflation': {
            'level': {
                'prompt': INFLATION_LEVEL_RANKING_PROMPT,
                'schema': InflationLevelRankingResponse,
                'column': 'inflation_summary' 
            },
            'dynamics': {
                'prompt': INFLATION_DYNAMICS_RANKING_PROMPT,
                'schema': InflationDynamicsRankingResponse,
                'column': 'inflation_summary' 
            }
        },
        'forward_guidance': {
            'guidance': {
                'prompt': FORWARD_GUIDANCE_RANKING_PROMPT,
                'schema': GuidanceRankingResponse,
                'column': 'forward_guidance_summary'
            }
        },
        'interest_rate': {
            'trajectory': {
                'prompt': INTEREST_RATE_PROJECTION_RANKING_PROMPT,
                'schema': InterestRateProjectionRankingResponse,
                'column': 'interest_rate_summary'
            }
        },
        'balance_sheet': {
            'trajectory': {
                'prompt': BALANCE_SHEET_PROJECTION_RANKING_PROMPT,
                'schema': BalanceSheetProjectionRankingResponse,
                'column': 'balance_sheet_summary'
            }
        }
    }

    if task == 'all':
        tasks_to_process = list(ranking_configs.keys())
    else:
        if task not in ranking_configs:
            logger.error(f"Unknown task: {task}")
            return
        tasks_to_process = [task]

    for current_task in tasks_to_process:
        logger.info(f"Processing rankings for task: {current_task}")
        task_configs = ranking_configs[current_task]
        
        for subtask, subtask_config in task_configs.items():
            logger.info(f"Processing {current_task}.{subtask} rankings")
            
            # Create ranker
            ranker = StatementRanker(
                config=config,
                ranking_prompt=subtask_config['prompt'],
                output_schema=subtask_config['schema'],
                window_size=5
            )

            # Process pairs
            pairs, rankings, reasonings = ranker.process_pairs_from_df(df, subtask_config['column'])
            pairs_data = []
            for i, ((doc1, doc2), ranking) in enumerate(zip(pairs, rankings)):
                # Find corresponding rows in dataframe
                row1 = df[df[subtask_config['column']] == doc1].iloc[0]
                row2 = df[df[subtask_config['column']] == doc2].iloc[0]

                pairs_data.append({
                    'doc_1': row1[subtask_config['column']],
                    'doc_2': row2[subtask_config['column']],
                    'date_1': row1['date'],
                    'date_2': row2['date'],
                    'ranking': ranking,
                    'ranking_reasoning': reasonings[i]
                })


            output_file = os.path.join(ranked_pairs_dir, f"{current_task}_{subtask}_ranked_pairs.json")
            with open(output_file, 'w') as f:
                json.dump(pairs_data, f, indent=2)
            
            logger.info(f"Saved {len(pairs)} ranked pairs to {output_file}")

            csv_data = pd.DataFrame({
                "pair_id": range(len(pairs)),
                "doc1_date": [df[df[subtask_config['column']] == p[0]].iloc[0]["date"] for p in pairs],
                "doc2_date": [df[df[subtask_config['column']] == p[1]].iloc[0]["date"] for p in pairs],
                "ranking": rankings,
                "ranking_reasoning": reasonings
            })
            csv_output_file = os.path.join(ranked_pairs_dir, f"{current_task}_{subtask}_ranked_pairs.csv")
            csv_data.to_csv(csv_output_file, index=False)
            
            logger.info(f"Processing complete for {current_task}.{subtask}")

def train_pairwise_classifier(config, task, subtask, model_params=None):
    """Train pairwise classifier

    Args:
        config: Configuration object
        task: Task to train on ('sentiment', 'employment', 'forward_guidance')

    Pairwise classifier is a linear model that is trying to predict scores for each document 
        such that sigmoid of the difference corresponds to the ranking labels.
        It supports binary and 3-class (1, 0, -1) ranking labels.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training pairwise classifier for task: {task}")

    logger = logging.getLogger(__name__)
    logger.info(f"Training pairwise model for {task}.{subtask}")
    
    # Load the ranked pairs
    ranked_pairs_dir = os.path.join(config.output_dir, "ranked_pairs")
    pairs_file = os.path.join(ranked_pairs_dir, f"{task}_{subtask}_ranked_pairs.json")

    if not os.path.exists(pairs_file):
        logger.error(f"Ranked pairs file not found: {pairs_file}")
        return None
    
    with open(pairs_file, 'r') as f:
        pair_data = json.load(f)


    # Load the original data with statement texts
    try:
        df = pd.read_csv(os.path.join(config.output_dir, "extracted_statements.csv"))
    except FileNotFoundError:
        logger.error("Extracted statements not found")
        return None

    logger.info(f"Loaded {len(pair_data)} ranked pairs from {pairs_file}")

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
    cutoff_year = config.last_year
    starting_year = config.base_year
    
    # Create a mapping from documents to their dates
    doc_to_date = {}
    for _, row in df.iterrows():
        doc_to_date[row[text_column]] = row["date"]
    
    # Split pairs based on document dates
    train_indices = []
    val_indices = []

    #purge pairs data to eliminate cycles
    pair_data = [(item["doc_1"], item["doc_2"], item["ranking"]) for item in pair_data]
    #result = ensure_transitivity(pair_data)
    #pair_data = reconstruct_pairs(result["final_graph"], result["removed_nodes"], result["union_find"])

    pairs = [(item[0], item[1]) for item in pair_data]
    rankings = [item[2] for item in pair_data]
    
    for i, (doc1, doc2) in enumerate(pairs):
        date1 = doc_to_date.get(doc1, "")
        date2 = doc_to_date.get(doc2, "")

        year_1 = int(date1.split('-')[0])
        year_2 = int(date2.split('-')[0])

        # Ensure no sample contamination from future data
        if min(year_1, year_2) > cutoff_year:
            val_indices.append(i)
        elif max(year_1, year_2)  > starting_year:
            train_indices.append(i)
    
    # Create train and validation sets

    ##Purge only train pairs
    train_pairs = [pairs[i] for i in train_indices]
    train_rankings = [rankings[i] for i in train_indices]
    # train_pair_data = [(train_pairs[i][0], train_pairs[i][1], train_rankings[i]) for i in range(len(train_pairs))]
    # result = ensure_transitivity(train_pair_data)
    # train_pair_data = reconstruct_pairs(result["final_graph"], result["removed_nodes"], result["union_find"])
    # train_pairs = [(item[0], item[1]) for item in train_pair_data]
    # train_rankings = [item[2] for item in train_pair_data]

    val_pairs = [pairs[i] for i in val_indices]
    val_rankings = [rankings[i] for i in val_indices]

    train_positives = len([item for item in train_rankings if item == 1]) / len(train_rankings)
    train_negatives = len([item for item in train_rankings if item == -1]) / len(train_rankings)
    train_zeros = len([item for item in train_rankings if item == 0]) / len(train_rankings)

    val_positives = len([item for item in val_rankings if item == 1]) / len(val_rankings)
    val_negatives = len([item for item in val_rankings if item == -1]) / len(val_rankings)
    val_zeros = len([item for item in val_rankings if item == 0]) / len(val_rankings)

    logger.info(f"Train classes distribution Positives/Negatives/Zeros: {train_positives}, {train_negatives}, {train_zeros}")
    logger.info(f"Val classes distribution Positives/Negatives/Zeros: {val_positives}, {val_negatives}, {val_zeros}")
    
    # Add chairman feature if it exists
    feature_columns = []
    if "chairman" in df.columns:
        feature_columns.append("chairman")
    
    logger.info(f"Split data into {len(train_pairs)} train and {len(val_pairs)} validation pairs")
    
    # Set default model parameters if not provided
    if model_params is None:
        model_params = {
            'sigma': 10.0,
            'error_weight_1_vs_minus1': 3.0,
            'error_weight_with_0': 1.0,
            'learning_rate': 0.01,
            'regularization': 0.001,
            'num_epochs': 200,
            'margin': 0.01,
            'batch_size': 250,
            'initialization_scale': 0.01
        }

    model = LinearRankNet(
        config=config,
        model_params=model_params,
        text_column=text_column,
        feature_columns=feature_columns,
        cache_prefix=f"{task}_{subtask}_ranker"
    )

    model.logger.setLevel(logging.INFO)

    param_grid = {
        'pca_components': [0, 10, 20, 30, 50],
        'sigma': [1, 10.0, 20.0, 30.0],
        'error_weight_1_vs_minus1': [1.2],
        'learning_rate': [0.01],
        'regularization': [0.01],
        'num_epochs': [100],
        'margin': [0.01, 0.03, 0.05],
        'batch_size': [100],
        'initialization_scale': [0.01]
    }

    best_params, best_metrics = model.hyperparameter_search(
        pairs=train_pairs,
        rankings=train_rankings,
        df=df,
        val_pairs=val_pairs,
        val_rankings=val_rankings,
        param_grid=param_grid,
        n_trials=150
    )

    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best metrics: {best_metrics}")

    # Save model
    models_dir = os.path.join(config.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{task}_{subtask}_pairwise_model.pkl")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Evaluate on validation set
    if val_pairs:
        metrics = model.evaluate_pairs(val_pairs, val_rankings, df)
        logger.info(f"Validation metrics: {metrics}")

        json_safe_metrics = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }
        
        # Save evaluation results
        metrics_path = os.path.join(config.output_dir, f"{task}_{subtask}_pairwise_evaluation.json")
        with open(metrics_path, 'w') as f:
            json.dump(json_safe_metrics, f, indent=2)
    
    # Generate document scores
    scores = model.predict(df, return_probs=False)
    
    # Add scores to the dataframe
    score_column = f"{task}_{subtask}_score"
    df[score_column] = scores

    # Save results
    results_path = os.path.join(config.output_dir, f"{task}_{subtask}_pairwise_scores.csv")
    df.to_csv(results_path, index=False)
    
    logger.info(f"Generated scores and saved to {results_path}")
    
    return model

def train_classifier(config, task, model_type):
    """Train classification model
    
    Args:
        config: Configuration object
        task: Task to train on ('sentiment', 'employment', 'forward_guidance')
        model_type: Type of model to train ('logreg', 'svc', 'rf')
        
    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training {model_type} classifier for task: {task}")
    
    # Load data
    try:
        df = pd.read_csv(os.path.join(config.output_dir, "classified_statements.csv"))
    except FileNotFoundError:
        logger.error("Classified statements not found. Run extraction and classification first.")
        return None
    
    # Split data
    train_df = df[df['date'] < f'{config.last_year}-01-01']
    test_df = df[(df['date'] >= f'{config.last_year}-01-01') & 
                  (df['date'] < f'{config.last_year+1}-01-01')]
    
    logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test samples")

    tasks_mapping = {
        'sentiment': ('sentiment_class', 'sentiment_summary'),
        'employment-levels': ('employment_levels_class', 'employment_summary'),
        'employment-dynamics': ('employment_dynamics_class', 'employment_summary'),
        'employment-concern': ('employment_concern_class', 'employment_summary'),
        'forward_guidance': ('guidance_class', 'forward_guidance_summary'),
        'interest_rate': ('interest_rate_class', 'interest_rate_summary'),
        'balance_sheet': ('balance_sheet_class', 'balance_sheet_summary')
    }

    if task not in tasks_mapping:
        logger.error(f"Unknown task: {task}")
        return None

    target_column, embedding_column = tasks_mapping[task]
    
    # Create and train model
    model = SingleObjectClassifier(
        config=config,
        model_type=model_type,
        target_column=target_column,
        embedding_column=embedding_column,
        stratify_column='chairman',
        cache_prefix=f"{task}_{model_type}"
    )

    model.logger.setLevel(logging.INFO)
    
    # Train model
    cv_results, best_params = model.fit(train_df, param_search=True)
    
    logger.info(f"Model trained successfully. Best parameters: {best_params}")
    logger.info(f"Cross-validation results: {cv_results}")
    
    # Evaluate on test set
    test_metrics = model.evaluate(test_df)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save model
    models_dir = os.path.join(config.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{task}_{model_type}_model.pkl")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Add predictions to test data
    test_df[f"{task}_pred"] = model.predict(test_df)
    test_df[f"{task}_prob"] = model.predict_proba(test_df)[:, 1] if model.label_encoder.classes_.shape[0] == 2 else None
    
    # Save results
    results_path = os.path.join(config.output_dir, f"{task}_classification_results.csv")
    test_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    return model


def main():
    """Main execution function"""
    args = parse_args()
    logger = setup_logging()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize configuration
    config = Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        last_year=args.last_year,
        base_year=args.base_year,
        cache_embeddings=args.cache_embeddings,
        openai_key=args.openai_key or os.environ.get("OPENAI_API_KEY")
    )
    config.chunk_size = args.chunk_size
    
    # Handle clear_cache mode
    if args.mode == 'clear_cache':
        logger.info(f"Clearing embedding cache: {args.clear_cache_name or 'all'}")
        config.clear_embedding_cache(args.clear_cache_name)
        return
    
    # Execute pipeline based on mode
    if args.mode == 'extract':
        # Extract statements
        df = extract_statements(config, args.task)
    
    elif args.mode == 'classify':
        # Classify statements
        df = classify_statements(config, args.task)

    elif args.mode == 'rank':
        # Rank statements
        df = rank_statements(config, args.task)
    
    elif args.mode == 'train_classifier':
        # Train classifier
        if args.task == 'all':
            for task in ['sentiment', 'employment', 'forward_guidance']:
                train_classifier(config, task, args.model_type)
        else:
            train_classifier(config, args.task, args.model_type)
    
    elif args.mode == 'train_ranker':

        subtasks_mapping = {
            'employment': ['level', 'dynamics'],
            'inflation': ['level', 'dynamics'],
            'forward_guidance': ['guidance'],
            'interest_rate': ['trajectory'],
            'balance_sheet': ['trajectory']
        }

        if args.task == 'all':
            for task in subtasks_mapping.keys():
                for subtask in subtasks_mapping[task]:
                    train_pairwise_classifier(config, task, subtask)
        else:
            subtasks = subtasks_mapping[args.task]
            for subtask in subtasks:
                train_pairwise_classifier(config, args.task, subtask)
    
    elif args.mode == 'predict':
        # Generate predictions
        if args.task == 'all':
            for task in ['sentiment', 'employment', 'forward_guidance']:
                predict(config, task, args.model_type)
        else:
            predict(config, args.task, args.model_type)
    
    elif args.mode == 'evaluate':
        # Evaluate model
        if args.task == 'all':
            for task in ['sentiment', 'employment', 'forward_guidance']:
                evaluate(config, task, args.model_type)
        else:
            evaluate(config, args.task, args.model_type)
    
    elif args.mode == 'all':
        # Run entire pipeline
        
        # 1. Extract statements
        logger.info("Step 1: Extracting statements")
        df = extract_statements(config, args.task)
        
        # 2. Classify statements
        logger.info("Step 2: Classifying statements")
        df = classify_statements(config, args.task)
        
        # 3. Train models
        logger.info("Step 3: Training models")
        if args.task == 'all':
            for task in ['sentiment', 'employment', 'forward_guidance']:
                train_classifier(config, task, 'logreg')
            
            # Train ranker for sentiment
            ranker_params = {
                'learning_rate': args.ranker_learning_rate,
                'margin': args.ranker_margin,
                'num_epochs': args.ranker_epochs
            }
            train_ranker(config, 'three_way_ranker', args.max_pairs, ranker_params)
        else:
            train_classifier(config, args.task, args.model_type)
            
            if args.task == 'sentiment' and args.model_type == 'three_way_ranker':
                ranker_params = {
                    'learning_rate': args.ranker_learning_rate,
                    'margin': args.ranker_margin,
                    'num_epochs': args.ranker_epochs
                }
                train_ranker(config, 'three_way_ranker', args.max_pairs, ranker_params)
        
        # 4. Evaluate models
        logger.info("Step 4: Evaluating models")
        if args.task == 'all':
            for task in ['sentiment', 'employment', 'forward_guidance']:
                evaluate(config, task, 'logreg')
            
            # Evaluate ranker for sentiment
            evaluate(config, 'sentiment', 'three_way_ranker')
        else:
            evaluate(config, args.task, args.model_type)
    
    logger.info("Pipeline complete")

if __name__ == "__main__":
    main()