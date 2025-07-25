"""
LLM client for handling interactions with the language model
"""
import logging
import asyncio
import typing as t
import numpy as np
import tiktoken
from typing import List, Optional, Union, Dict, Any, Tuple

from openai import AsyncOpenAI
from openai import DefaultAioHttpClient

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from engine.config import Config
from engine.src.utils.utils import TokenRateLimiter
from engine.env import OPENAI_API_KEY

class LLMClient:
    """Client for interacting with language models"""
    
    def __init__(self, config: Config):
        """Initialize LLM client
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiter
        self.rate_limiter = TokenRateLimiter(
            config.rate_limit_requests,
            config.rate_limit_tokens
        )
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        # Initialize LLM
        self.client: Optional[AsyncOpenAI] = None

        # self.chat = ChatOpenAI(
        #     api_key=OPENAI_API_KEY,
        #     model=config.llm_model,
        #     temperature=config.llm_temperature
        # )
        
        # Initialize embeddings engine
        self.embeddings_engine = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=OPENAI_API_KEY
        )

    async def _ensure_client_initialized(self):
        """Initializes the AsyncOpenAI client if it hasn't been already."""
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                base_url="https://api.openai.com/v1"
            )
    
    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from prompt
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """

        await self._ensure_client_initialized()
        
        # Estimate token count
        system_message = SystemMessage(content=system_prompt)
        user_message = HumanMessage(content=user_prompt)
        
        # Estimate token count
        tokens = self.tokenizer.encode(system_prompt + user_prompt)
        
        # Wait for token availability
        await self.rate_limiter.wait_for_token_availability(len(tokens))
        
        # Generate text
        response = await self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[system_message, user_message]
        )
        return response.choices[0].message.content
    
    async def generate_structured_output(self, 
                                         system_prompt: str, 
                                         user_prompt: str, 
                                         output_schema: Any) -> Any:
        """Generate structured output using a specific schema
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            output_schema: Pydantic model to structure the output
            
        Returns:
            Structured output according to the provided schema
        """

        await self._ensure_client_initialized()
        
        # Estimate token count
        tokens = self.tokenizer.encode(system_prompt + user_prompt)

        # Wait for token availability
        await self.rate_limiter.wait_for_token_availability(len(tokens))
        
        # Generate text

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = await self.client.beta.chat.completions.parse(
            model=self.config.llm_model,
            messages=messages,
            response_format=output_schema
        )
        return response.choices[0].message.parsed
    
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for text
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """

        await self._ensure_client_initialized()
        
        # Get embeddings
        embeddings = await asyncio.to_thread(
            self.embeddings_engine.embed_documents, 
            texts
        )
        return np.array(embeddings)
    
    async def rank_pair(self, doc1, doc2, user_prompt, output_schema):
        await self._ensure_client_initialized()
        
        system_prompt = "You are an expert in macroeconomics and central bank policy."
        user_prompt += f"Statements A: {doc1}\nStatements B: {doc2}"
        tokens_needed = len(self.tokenizer.encode(system_prompt + user_prompt))
        await self.rate_limiter.wait_for_token_availability(tokens_needed)
        response = await self.generate_structured_output(system_prompt, user_prompt, output_schema)
        return response.ranking
    
    async def generate_rankings(self, 
                             document_pairs: List[Tuple[str, str]], 
                             user_prompt: str,
                             output_schema: t.Optional[Any]) -> List[int]:
        """Rank pairs of documents
        
        Args:
            document_pairs: List of document pairs to rank
            user_prompt: Prompt for ranking. It is not a system prompt - system prompt is a small guidance for the style and role.
                We do not put big prompt with the logic in the system prompt as it hurts the performance generally.
            output_schema: Pydantic model to structure the output.
                            Pydantic schema is optional, but if provided - should contain field 'ranking'
                            which is an integer value (1, 0, -1)
        Returns:
            List of rankings (1 if first document ranks higher, -1 if second, 0 if equal)
        """
        await self._ensure_client_initialized()

        tasks = [self.rank_pair(pair[0], pair[1], user_prompt, output_schema) for pair in document_pairs]
        results = await asyncio.gather(*tasks)
        return results