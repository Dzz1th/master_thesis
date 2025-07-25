"""
Data processing module for sentiment analysis pipeline with a flexible,
generalizable structure for different types of statement processing
"""
import logging
import asyncio
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from langchain_core.messages import SystemMessage, HumanMessage
import tiktoken
from tqdm.asyncio import tqdm as tqdm_async
from pydantic import BaseModel, Field

from typing import Type

from engine.config import Config
from engine.src.processing.base import StatementProcessor

class DataProcessor:
    """Main data processor coordinating different statement processors"""
    
    def __init__(self, config: Config):
        """Initialize data processor
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sentiment_processor = SentimentStatementProcessor(config)
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare and preprocess data
        
        Returns:
            Tuple containing train and test dataframes
        """
        # Load data
        df = pd.read_csv(self.config.data_path)
        self.logger.info(f"Loaded data with {len(df)} rows")
        
        # Process data with sentiment processor
        processed_df = self.sentiment_processor.process_dataframe(df)
        
        # Split into train and test
        train_df = processed_df[processed_df['date'] < f'{self.config.last_year}-01-01']
        test_df = processed_df[(processed_df['date'] >= f'{self.config.last_year}-01-01') & 
                              (processed_df['date'] < f'{self.config.last_year+1}-01-01')]
        
        self.logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test rows")
        
        return train_df, test_df