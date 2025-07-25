from typing import List
import pandas as pd
from tqdm.asyncio import tqdm
import asyncio

from engine.config import Config
from engine.src.llms.llm_client import LLMClient
from engine.logger import make_logger
from engine.src.processing.base import StatementProcessor
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Type, Dict, Any

logger = make_logger(__name__)

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
    
    async def process_statement(self, statement: str) -> BaseModel:
        """Classify a statement using structured output
        
        Args:
            statement: Statement to classify
            
        Returns:
            Dictionary with classification results
        """

        system_message = "You are an expert in macroeconomics and central bank policy."
        user_prompt = self.processing_prompt + "\n\nStatement: " + statement
        
        tokens = self.tokenizer.encode(system_message + user_prompt)
        await self.llm_client.rate_limiter.wait_for_token_availability(len(tokens))
        response = await self.llm_client.generate_structured_output(system_message, user_prompt, self.output_schema)
        return response.extract_results()
    
    def process_statements_from_df(self, df: pd.DataFrame, text_column: str, output_column: str) -> pd.DataFrame:
        """Process statements from a DataFrame by running the async processor."""
        statements = df[text_column].dropna().tolist()
        self.logger.info(f"Classifying {len(statements)} statements")

        async def run_all():
            tasks = [self.process_statement(statement) for statement in statements]
            return await tqdm.gather(*tasks, desc="Classifying Statements")

        results = asyncio.run(run_all())
        df[output_column] = pd.Series(results, index=df.index[df[text_column].notna()])
        return df