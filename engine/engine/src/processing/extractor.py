from pandas import DataFrame
from engine.config import Config
from engine.logger import make_logger
from engine.src.llms.llm_client import LLMClient
from engine.src.processing.base import StatementProcessor

import asyncio
from typing import List
from tqdm.asyncio import tqdm as tqdm_async

logger = make_logger(__name__)

def get_chunks(text: str, chunk_size: int) -> List[str]:
    """Simple text chunking function."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

class StatementExtractor(StatementProcessor):
    """Extract relevant statements from raw text"""

    def __init__(self, config: Config, extraction_prompt: str, chunk_size: int = 5000):
        super().__init__(config, extraction_prompt)
        self.chunk_size = chunk_size

    async def process_statement(self, statement: str) -> str:
        """Process a single statement by chunking."""
        chunks = get_chunks(statement, self.chunk_size)
        results = await asyncio.gather(*[self.process_chunk(chunk) for chunk in chunks])
        return '\n'.join(results)
    