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

from config import Config
from graph_utils import ensure_transitivity, reconstruct_pairs
from llm_client import LLMClient


class StatementProcessor:
    """Base class for processing statements using LLM"""
    
    def __init__(self, config: Config, processing_prompt: str):
        """Initialize statement processor
        
        Args:
            config: Configuration object
            processing_prompt: Prompt template for processing statements
        """
        self.config = config
        self.processing_prompt = processing_prompt
        self.logger = logging.getLogger(__name__)
        self.llm_client = LLMClient(config)
        self.tokenizer = tiktoken.encoding_for_model(config.llm_model)
    
    async def process_statement(self, statement: str) -> str:
        """Process a single statement using the prompt template
        
        Args:
            statement: Input statement to process
            
        Returns:
            Processed statement
        """
        response = await self.llm_client.generate_text(self.processing_prompt, statement)
        return response
    
    async def process_statements(self, statements: List[str]) -> List[str]:
        """Process multiple statements
        
        Args:
            statements: List of statements to process
            
        Returns:
            List of processed statements
        """
        self.logger.info(f"Processing {len(statements)} statements")
        tasks = [self.process_statement(statement) for statement in statements]
        results = await tqdm_async.gather(*tasks, desc="Processing Statements")
        return results
    
    def process_statements_from_df(self, df: pd.DataFrame, input_column: str, output_column: str) -> pd.DataFrame:
        """Process statements from a dataframe column
        
        Args:
            df: Input dataframe
            input_column: Column containing input statements
            output_column: Column to store processed statements
            
        Returns:
            Dataframe with processed statements
        """
        statements = df[input_column].tolist()
        results = asyncio.run(self.process_statements(statements))
        df[output_column] = results
        return df


class StatementExtractor(StatementProcessor):
    """Extract relevant statements from raw text"""

    def __init__(self, config: Config, extraction_prompt: str, chunk_size: int = 5000):
        super().__init__(config, extraction_prompt)
        self.chunk_size = chunk_size

    async def process_chunk(self, chunk: str) -> str:
        """Process a single chunk of text using the prompt template
        """
        user_query = "Part of the transcript: " + chunk
        response = await self.llm_client.generate_text(self.processing_prompt, user_query)
        return response
    
    def generate_chunks(self, text: str) -> List[str]:
        """Generate chunks of text from a single string
        """
        chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return chunks
    
    async def process_statement(self, statement: str) -> str:
        """Process multiple statements
        
        Args:
            statements: List of statements to process
            
        Returns:
            List of processed statements
        """
        chunks = self.generate_chunks(statement)
        results = await asyncio.gather(*[self.process_chunk(chunk) for chunk in chunks])
        results = '\n'.join(results)
        return results
    
    async def process_statements(self, statements: List[str]) -> List[str]:
        """Extract statements from multiple raw texts
        
        Args:
            statements: List of raw texts to extract statements from
            
        Returns:
            List of extracted statements
        """
        self.logger.info(f"Extracting statements from {len(statements)} texts")
        tasks = [self.process_statement(statement) for statement in statements]
        results = await tqdm_async.gather(*tasks, desc="Extracting Statements")
        return results


class StatementClassifier(StatementProcessor):
    """Classify statements using structured output with Pydantic models"""
    
    def __init__(self, config: Config, classification_prompt: str, output_schema: Type[BaseModel]):
        """Initialize statement classifier
        
        Args:
            config: Configuration object
            classification_prompt: Prompt template for classification
            output_schema: Pydantic model to structure the output
        """
        super().__init__(config, classification_prompt)
        self.output_schema = output_schema
        self.structured_chat = self.llm_client.chat.with_structured_output(self.output_schema)
    
    async def process_statement(self, statement: str) -> Dict[str, Any]:
        """Classify a statement using structured output
        
        Args:
            statement: Statement to classify
            
        Returns:
            Dictionary with classification results
        """
        system_message = SystemMessage(content=self.processing_prompt)
        user_message = HumanMessage(content=statement)
        
        tokens = self.tokenizer.encode(self.processing_prompt + statement)
        await self.llm_client.rate_limiter.wait_for_token_availability(len(tokens))
        
        try:
            response = await self.structured_chat.ainvoke([system_message, user_message])
            
            # Extract results using the model's method if available
            if hasattr(response, 'extract_results'):
                return response.extract_results()
            else:
                return response.dict()
                
        except Exception as e:
            self.logger.error(f"Error generating structured output: {e}")
            return {}
    
    async def process_statements(self, statements: List[str]) -> List[Dict[str, Any]]:
        """Process multiple statements with structured output
        
        Args:
            statements: List of statements to classify
            
        Returns:
            List of classification results as dictionaries
        """
        self.logger.info(f"Classifying {len(statements)} statements")
        tasks = [self.process_statement(statement) for statement in statements]
        results = await tqdm_async.gather(*tasks, desc="Classifying Statements")
        return results


class StatementAugmenter(StatementProcessor):
    """Augment statements with variations"""
    
    async def process_statement(self, statement: str) -> str:
        """Generate augmented version of a statement
        
        Args:
            statement: Statement to augment
            
        Returns:
            Augmented statement
        """
        system_message = SystemMessage(content=self.processing_prompt)
        user_message = HumanMessage(content=statement)
        
        tokens = self.tokenizer.encode(self.processing_prompt + statement)
        await self.llm_client.rate_limiter.wait_for_token_availability(len(tokens))
        
        response = await self.llm_client.generate_text(system_message, user_message)
        return response

class StatementRanker(StatementProcessor):
    """Rank statements using structured output with Pydantic models"""

    def __init__(self, config: Config, ranking_prompt: str, output_schema: Type[BaseModel], window_size: int = 10):
        super().__init__(config, ranking_prompt)
        self.output_schema = output_schema
        self.window_size = window_size

    async def rank_statements(self, statement_1, statement_2) -> int:
        """Rank two statements

        Args:
            statement_1: First statement
            statement_2: Second statement
            
        Returns:
            Integer ranking result (1, -1, or 0)
        """
        user_message = f"Statement 1: {statement_1}\nStatement 2: {statement_2}"
        
        response = await self.llm_client.generate_structured_output(self.processing_prompt, user_message, self.output_schema)

        ## Each ranking response will have a field with results which consist of -1, 0, 1 score
        return response.result, response.ranking_reasoning
    
    async def process_pairs(self, pairs: List[Tuple[str, str]]) -> List[int]:
        """Process pairs of statements
        
        Args:
            pairs: List of pairs of statements
            
        """
        results = await asyncio.gather(*[self.rank_statements(pair[0], pair[1]) for pair in pairs])
        return results
    
    def process_pairs_from_df(self, df: pd.DataFrame, input_column: str) -> Tuple[List[Tuple[str, str]], List[int], List[str]]:
        """Generate pairs of documents for ranking
        
        Args:
            df: Dataframe containing documents
            
        Returns:
            List of document pairs
        """
        
        # Sort dataframe by date
        df_sorted = df.sort_values(by='date').reset_index(drop=True)
        texts = df_sorted[input_column].tolist()
            
            # Generate pairs
        pairs = []
        for i in range(self.window_size, len(texts)):
            # Define window of previous documents
            start_idx = i - self.window_size
            previous_texts = texts[start_idx:i]
            
            # Create pairs
            for prev_text in previous_texts:
                # Create pair in canonical order (alphabetical)
                current_text = texts[i]
                pairs.append((prev_text, current_text))

        rankings = asyncio.run(self.process_pairs(pairs))
        reasonings = [res[1] for res in rankings]
        rankings = [res[0] for res in rankings]
        ranked_pairs = [(pairs[i][0], pairs[i][1], rankings[i]) for i in range(len(pairs))]
        #result = ensure_transitivity(ranked_pairs)
        #ranked_pairs = reconstruct_pairs(result["final_graph"], result["removed_nodes"], result["union_find"])
        pairs = [(pair[0], pair[1]) for pair in ranked_pairs]
        rankings = [pair[2] for pair in ranked_pairs]

        return pairs, rankings, reasonings

    def _get_chairman_feature(self, date_str: str) -> int:
        """Get chairman feature from date
        
        Args:
            date_str: Date string in format YYYY-MM-DD
            
        Returns:
            Integer chairman identifier (0, 1, or 2)
        """
        if date_str < '2014-01-01':
            return 0
        elif date_str < '2018-01-01':
            return 1
        else:
            return 2


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