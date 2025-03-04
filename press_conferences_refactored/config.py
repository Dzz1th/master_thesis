"""
Configuration module for sentiment analysis pipeline
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Config:
    """Configuration class for sentiment analysis pipeline"""
    
    # Data paths
    data_path: str
    output_dir: str
    
    # Year parameters
    last_year: int = 2023
    base_year: int = 2016
    
    # API keys
    openai_key: str = None
    
    # Model configuration
    first_stage_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 3,
        'eval_metric': 'error',
        'objective': 'rank:pairwise'
    })
    
    second_stage_learning_rate: float = 0.01
    second_stage_epochs: int = 10
    
    # Embedding configuration
    embedding_model: str = "text-embedding-3-large"
    cache_embeddings: bool = True
    embeddings_dir: str = field(init=False)
    
    # LLM configuration
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    
    # Token rate limiter configuration
    rate_limit_tokens: int = 1500000
    rate_limit_requests: int = 150
    
    def __post_init__(self):
        """Initialize derived attributes"""
        self.embeddings_dir = os.path.join(self.output_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)
    
    def get_embeddings_path(self, name: str) -> str:
        """Get path for cached embeddings"""
        return os.path.join(self.embeddings_dir, f"{name}_embeddings.pkl")