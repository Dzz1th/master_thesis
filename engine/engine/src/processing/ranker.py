from engine.logger import make_logger
from engine.src.processing.base import StatementProcessor
from engine.config import Config

from typing import Type, List, Tuple, Dict, Any
from pydantic import BaseModel
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm
import itertools

from engine.src.llms.llm_client import LLMClient

logger = make_logger(__name__)

class StatementRanker(StatementProcessor):
    """Rank statements using structured output with Pydantic models"""

    def __init__(self, config: Config, ranking_prompt: str, output_schema: Type[BaseModel], window_size: int = 10):
        super().__init__(config, ranking_prompt)
        self.output_schema = output_schema
        self.window_size = window_size
    
    async def process_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[int, str]]:
        """Process pairs of statements"""
        results = await self.llm_client.generate_rankings(pairs, self.processing_prompt, self.output_schema)
        return results
    
    async def _process_and_cleanup(self, pairs: List[Tuple[str, str]]):
        """Helper to run async processing and ensure client cleanup."""
        try:
            results = await self.process_pairs(pairs)
        finally:
            await self.llm_client.client.close()
        return results
    
    def process_pairs_from_df(self, df: pd.DataFrame, input_column: str) -> Tuple[List[Tuple[str, str]], List[int], List[str]]:
        """
        Generate pairs of documents for ranking and process them.
        This is the synchronous entry point that will cause the event loop error.
        """
        df_sorted = df.sort_values(by='date').reset_index(drop=True)
        texts = df_sorted[input_column].dropna().tolist()
        
        pairs = []
        for i in range(self.window_size, len(texts)):
            start_idx = max(0, i - self.window_size)
            previous_texts = texts[start_idx:i]
            current_text = texts[i]
            for prev_text in previous_texts:
                pairs.append((prev_text, current_text))

        if not pairs:
            return [], [], []
            
        # This call is the source of the "event loop is closed" error
        results = asyncio.run(self._process_and_cleanup(pairs))
        
        reasonings = [res[1] for res in results]
        rankings = [res[0] for res in results]
        
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
        

class RankerSummary(StatementProcessor):
    """Ranker summary"""

    def __init__(self, config: Config, summary_prompt: str, output_schema: Type[BaseModel], window_size: int = 10):
        super().__init__(config, summary_prompt)
        self.output_schema = output_schema
        self.window_size = window_size
        
    async def process_summary(self, summary: str) -> str:
        """Process summary
        
        Args:
            summary: Summary string
            
        Returns:
            Summary string
        """
        return await self.llm_client.generate_structured_output(self.processing_prompt, summary, self.output_schema)
