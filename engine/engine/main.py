#!/usr/bin/env python
"""
FED Statement Sentiment Analysis
Main script to orchestrate the sentiment analysis pipeline
"""
import argparse
import logging
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from imblearn.under_sampling import RandomUnderSampler

from engine.src.utils.graph_utils import ensure_transitivity, reconstruct_pairs
from engine.src.llms.llm_cache import set_cache

from engine.src.processing.extractor import StatementExtractor
from engine.src.processing.classifier import StatementClassifier
from engine.src.processing.ranker import StatementRanker
from engine.src.models.single_classifier import SingleObjectClassifier
from engine.src.models.pairwise_classifier import LinearRankNet
from engine.src.models.evaluator import ModelEvaluator
from engine.config import Config

set_cache()

from engine.src.tasks.extract_statements import extract_statements
from engine.src.tasks.classify_statements import classify_statements
from engine.src.tasks.rank_statements import rank_statements
from engine.src.tasks.train_single import train_classifier
from engine.src.tasks.train_pairwise import train_pairwise_classifier
from engine.src.tasks.tag_statements import tag_statements



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
        choices=['extract', 'classify', 'rank', 'tag', 'train_classifier', 'train_ranker', 'predict', 'evaluate', 'clear_cache', 'all'],
        default='all',
        help='Pipeline mode'
    )
    
    # Task arguments
    parser.add_argument(
        '--task',
        type=str,
        choices=['sentiment', 'employment', 'inflation', 'forward_guidance', 'interest_rate', 'balance_sheet', 'economic_outlook', 'all'],
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
        default=10000,
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

    elif args.mode == 'tag':
        # Tag statements
        df = tag_statements(config, args.task)

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
            'balance_sheet': ['trajectory'],
            'economic_outlook': ['outlook']
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

    
    logger.info("Pipeline complete")

if __name__ == "__main__":
    main()