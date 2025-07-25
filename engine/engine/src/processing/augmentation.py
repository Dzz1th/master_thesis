from engine.logger import make_logger
from engine.src.processing.base import StatementProcessor
from engine.engine.config import Config

from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any
from tqdm.asyncio import tqdm as tqdm_async

logger = make_logger(__name__)


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