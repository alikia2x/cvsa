"""
Embedding service for generating embeddings using OpenAI-compatible API
"""
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
import logging
from openai import AsyncOpenAI
import os
from config_loader import config_loader
from dotenv import load_dotenv

load_dotenv() 

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        # Get configuration from config loader
        self.embedding_models = config_loader.get_embedding_models()
        
        # Initialize OpenAI client (will be configured per model)
        self.clients: Dict[str, AsyncOpenAI] = {}
        self._initialize_clients()
        
        # Rate limiting
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "100"))
        self.request_interval = 60.0 / self.max_requests_per_minute
    
    def _initialize_clients(self):
        """Initialize OpenAI clients for different models/endpoints"""
        for model_name, model_config in self.embedding_models.items():
            if model_config.type == "openai-compatible":
                # Get API key from environment variable specified in config
                api_key = os.getenv(model_config.api_key_env)
                
                self.clients[model_name] = AsyncOpenAI(
                    api_key=api_key,
                    base_url=model_config.api_endpoint
                )
                logger.info(f"Initialized client for model {model_name}")
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        model: str,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        
        # Get model configuration
        if model not in self.embedding_models:
            raise ValueError(f"Model '{model}' not found in configuration")
        
        model_config = self.embedding_models[model]
        
        # Use model's max_batch_size if not specified
        if batch_size is None:
            batch_size = model_config.max_batch_size
        
        # Validate model and get expected dimensions
        expected_dims = model_config.dimensions
        
        if model not in self.clients:
            raise ValueError(f"No client configured for model '{model}'")
        
        client = self.clients[model]
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Rate limiting
                if i > 0:
                    await asyncio.sleep(self.request_interval)
                
                # Generate embeddings
                response = await client.embeddings.create(
                    model=model_config.name,
                    input=batch,
                    dimensions=expected_dims
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # For now, fill with zeros as fallback (could implement retry logic)
                zero_embedding = [0.0] * expected_dims
                all_embeddings.extend([zero_embedding] * len(batch))
        
        return all_embeddings
    
    def create_text_checksum(self, text: str) -> str:
        """Create MD5 checksum for text deduplication"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def combine_video_text(self, title: str, description: str, tags: str) -> str:
        """Combine video metadata into a single text for embedding"""
        parts = [
            title.strip() if "标题："+title else "",
            description.strip() if "简介："+description else "",
            tags.strip() if "标签："+tags else ""
        ]
        
        # Filter out empty parts and join
        combined = '\n'.join(filter(None, parts))
        return combined
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if embedding service is healthy"""
        try:
            # Test with a simple embedding using the first available model
            model_name = list(self.embedding_models.keys())[0]
            
            test_embedding = await self.generate_embeddings_batch(
                ["health check"], 
                model_name,
                batch_size=1
            )
            
            return {
                "status": "healthy",
                "service": "embedding_service",
                "model": model_name,
                "dimensions": len(test_embedding[0]) if test_embedding else 0,
                "available_models": list(self.embedding_models.keys())
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "service": "embedding_service",
                "error": str(e)
            }

# Global embedding service instance
embedding_service = EmbeddingService()