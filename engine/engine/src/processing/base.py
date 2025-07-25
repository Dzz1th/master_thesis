from engine.config import Config
from engine.logger import make_logger
from engine.src.llms.llm_client import LLMClient

import tiktoken
import asyncio
import pandas as pd
from typing import List
from tqdm.asyncio import tqdm as tqdm_async

logger = make_logger(__name__)

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
        self.logger = logger
        self.llm_client = LLMClient(config)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
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