"""
LLM client for handling interactions with the language model
"""
import logging
import asyncio
import typing as t
import numpy as np
import tiktoken
from typing import List, Optional, Union, Dict, Any, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from config import Config
from utils import TokenRateLimiter

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
        self.tokenizer = tiktoken.encoding_for_model(config.llm_model)
        
        # Initialize LLM
        self.chat = ChatOpenAI(
            api_key=config.openai_key,
            model=config.llm_model,
            temperature=config.llm_temperature
        )
        
        # Initialize embeddings engine
        self.embeddings_engine = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_key
        )
    
    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from prompt
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        # Estimate token count
        system_message = SystemMessage(content=system_prompt)
        user_message = HumanMessage(content=user_prompt)
        
        # Estimate token count
        tokens = self.tokenizer.encode(system_prompt + user_prompt)
        
        # Wait for token availability
        await self.rate_limiter.wait_for_token_availability(len(tokens))
        
        # Generate text
        response = await self.chat.ainvoke([system_message, user_message])
        return response.content
    
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
        structured_chat = self.chat.with_structured_output(output_schema)
        
        # Estimate token count
        system_message = SystemMessage(content=system_prompt)
        user_message = HumanMessage(content=user_prompt)
        
        # Estimate token count
        tokens = self.tokenizer.encode(system_prompt + user_prompt)
        
        # Wait for token availability
        await self.rate_limiter.wait_for_token_availability(len(tokens))
        
        # Generate text
        response = await structured_chat.ainvoke([system_message, user_message])
        return response
    
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for text
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        # Get embeddings
        embeddings = await asyncio.to_thread(
            self.embeddings_engine.embed_documents, 
            texts
        )
        return np.array(embeddings)
    
    async def generate_rankings(self, 
                             document_pairs: List[Tuple[str, str]], 
                             system_prompt: str,
                             output_schema: t.Optional[Any]) -> List[int]:
        """Rank pairs of documents
        
        Args:
            document_pairs: List of document pairs to rank
            system_prompt: System prompt for ranking
            output_schema: Pydantic model to structure the output.
                            Pydantic schema is optional, but if provided - should contain field 'ranking'
                            which is an integer value (1, 0, -1)
        Returns:
            List of rankings (1 if first document ranks higher, -1 if second, 0 if equal)
        """
        async def rank_pair(doc1, doc2, chat: ChatOpenAI):
            user_message = HumanMessage(content="Statements 1: {statements1}\nStatements 2: {statements2}")
            system_message = SystemMessage(content=system_prompt)
            tokens_needed = len(self.tokenizer.encode(system_prompt + user_message.content))
            await self.rate_limiter.wait_for_token_availability(tokens_needed)
            try:
                response = await chat.ainvoke([system_message, user_message])
                if output_schema is not None:
                    result = response.ranking
                else:
                    result = int(response.content.strip())
                return result
            except Exception as e:
                self.logger.error(f"Error ranking pair: {e}")
                return 0 # Default ranking if error occurs
        
        if output_schema is not None:
            chat = self.chat.with_structured_output(output_schema)
        else:
            chat = self.chat
        
        tasks = [rank_pair(pair[0], pair[1], chat) for pair in document_pairs]
        results = await asyncio.gather(*tasks)
        return results