"""
Embedding service for generating embeddings using OpenAI-compatible API and legacy methods
"""
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import os
from config_loader import config_loader
from dotenv import load_dotenv
import torch
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from logger_config import get_logger

load_dotenv()

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        # Get configuration from config loader
        self.embedding_models = config_loader.get_embedding_models()
        
        # Initialize OpenAI client (will be configured per model)
        self.clients: Dict[str, AsyncOpenAI] = {}
        self.legacy_models: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_clients()
        self._initialize_legacy_models()
        
        # Rate limiting
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "100"))
        self.request_interval = 60.0 / self.max_requests_per_minute
    
    def _initialize_clients(self):
        """Initialize OpenAI clients for different models/endpoints"""
        for model_name, model_config in self.embedding_models.items():
            if model_config.type != "openai-compatible":
                continue
            
            # Get API key from environment variable specified in config
            api_key = os.getenv(model_config.api_key_env)
            
            self.clients[model_name] = AsyncOpenAI(
                api_key=api_key,
                base_url=model_config.api_endpoint
            )
            logger.info(f"Initialized client for model {model_name}")
    
    def _initialize_legacy_models(self):
        """Initialize legacy ONNX models for embedding generation"""
        for model_name, model_config in self.embedding_models.items():
            if model_config.type != "legacy":
                continue
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
                
                
                
                # Load ONNX model
                session = ort.InferenceSession(model_config.model_path)
                
                self.legacy_models[model_name] = {
                    "tokenizer": tokenizer,
                    "session": session,
                    "config": model_config
                }
                logger.info(f"Initialized legacy model {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize legacy model {model_name}: {e}")
            
    def get_jina_embeddings_1024(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings using legacy Jina method (same as ml/api/main.py)"""
        if model_name not in self.legacy_models:
            raise ValueError(f"Legacy model '{model_name}' not initialized")
        
        legacy_model = self.legacy_models[model_name]
        tokenizer = legacy_model["tokenizer"]
        session = legacy_model["session"]
        
        # Encode inputs using tokenizer
        encoded_inputs = tokenizer(
            texts,
            add_special_tokens=False,  # Don't add special tokens (consistent with JS)
            return_attention_mask=False,
            return_tensors=None  # Return native Python lists for easier processing
        )
        input_ids = encoded_inputs["input_ids"]  # Shape: [batch_size, seq_len_i] (variable length per sample)
        
        # Calculate offsets (consistent with JS cumsum logic)
        # Get token length for each sample first
        lengths = [len(ids) for ids in input_ids]
        # Calculate cumulative sum (exclude last sample)
        cumsum = []
        current_sum = 0
        for l in lengths[:-1]:  # Only accumulate first n-1 samples
            current_sum += l
            cumsum.append(current_sum)
        # Build offsets: start with 0, followed by cumulative sums
        offsets = [0] + cumsum  # Shape: [batch_size]
        
        # Flatten input_ids to 1D array
        flattened_input_ids = []
        for ids in input_ids:
            flattened_input_ids.extend(ids)  # Directly concatenate all token ids
        flattened_input_ids = np.array(flattened_input_ids, dtype=np.int64)
        
        # Prepare ONNX inputs (consistent tensor shapes with JS)
        inputs = {
            "input_ids": ort.OrtValue.ortvalue_from_numpy(flattened_input_ids),
            "offsets": ort.OrtValue.ortvalue_from_numpy(np.array(offsets, dtype=np.int64))
        }
        
        # Run model inference
        outputs = session.run(None, inputs)
        embeddings = outputs[0]  # Assume first output is embeddings, shape: [batch_size, embedding_dim]
        
        return torch.tensor(embeddings, dtype=torch.float32).numpy()
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: str,
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        
        # Get model configuration
        if model not in self.embedding_models:
            raise ValueError(f"Model '{model}' not found in configuration")
        
        model_config = self.embedding_models[model]
        
        # Handle different model types
        if model_config.type == "legacy":
            return self._generate_legacy_embeddings_batch(texts, model, batch_size, progress_callback)
        elif model_config.type == "openai-compatible":
            return await self._generate_openai_embeddings_batch(texts, model, batch_size, progress_callback=progress_callback)
        else:
            raise ValueError(f"Unsupported model type: {model_config.type}")
    
    def _generate_legacy_embeddings_batch(
        self,
        texts: List[str],
        model: str,
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[List[float]]:
        """Generate embeddings using legacy ONNX model"""
        if model not in self.legacy_models:
            raise ValueError(f"Legacy model '{model}' not initialized")
        
        model_config = self.embedding_models[model]
        
        # Use model's max_batch_size if not specified
        if batch_size is None:
            batch_size = model_config.max_batch_size
        
        expected_dims = model_config.dimensions
        all_embeddings = []
        
        # Calculate total batches for progress tracking
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_idx = i // batch_size + 1
            
            try:
                # Generate embeddings using legacy method
                embeddings = self.get_jina_embeddings_1024(batch, model)
                
                # Convert to list of lists (expected format)
                batch_embeddings = embeddings.tolist()
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated legacy embeddings for batch {batch_idx}/{total_batches}")
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(batch_idx, total_batches)
                
            except Exception as e:
                logger.error(f"Error generating legacy embeddings for batch {batch_idx}: {e}")
                # Fill with zeros as fallback
                zero_embedding = [0.0] * expected_dims
                all_embeddings.extend([zero_embedding] * len(batch))
        
        # Final progress update
        if progress_callback:
            progress_callback(total_batches, total_batches)
        
        return all_embeddings
    
    async def _generate_openai_embeddings_batch(
        self,
        texts: List[str],
        model: str,
        batch_size: Optional[int] = None,
        max_concurrent: int = 6,
        progress_callback: Optional[callable] = None
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI-compatible API with parallel requests"""
        model_config = self.embedding_models[model]
        
        # Use model's max_batch_size if not specified
        if batch_size is None:
            batch_size = model_config.max_batch_size
        
        # Validate model and get expected dimensions
        expected_dims = model_config.dimensions
        
        if model not in self.clients:
            raise ValueError(f"No client configured for model '{model}'")
        
        client = self.clients[model]
        
        # Split texts into batches
        batches = [(i, texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        total_batches = len(batches)
        
        # Track completed batches for progress reporting
        completed_batches = 0
        completed_batches_lock = asyncio.Lock()
        
        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch_idx: int, batch: List[str]) -> tuple[int, List[List[float]]]:
            """Process a single batch with semaphore-controlled concurrency"""
            nonlocal completed_batches
            async with semaphore:
                try:
                    # Rate limiting
                    if batch_idx > 0:
                        await asyncio.sleep(self.request_interval)
                    
                    # Generate embeddings
                    response = await client.embeddings.create(
                        model=model_config.name,
                        input=batch,
                        dimensions=expected_dims
                    )
                    
                    batch_embeddings = [data.embedding for data in response.data]
                    logger.info(f"Generated embeddings for batch {batch_idx//batch_size + 1}/{total_batches}")
                    
                    # Update completed batch count and progress
                    async with completed_batches_lock:
                        completed_batches += 1
                        if progress_callback:
                            progress_callback(completed_batches, total_batches)
                    
                    return (batch_idx, batch_embeddings)
                    
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {batch_idx//batch_size + 1}: {e}")
                    # Fill with zeros as fallback
                    zero_embedding = [0.0] * expected_dims
                    
                    # Update completed batch count and progress even for failed batches
                    async with completed_batches_lock:
                        completed_batches += 1
                        if progress_callback:
                            progress_callback(completed_batches, total_batches)
                    
                    return (batch_idx, [zero_embedding] * len(batch))
        
        # Process all batches concurrently with semaphore limit
        tasks = [process_batch(i, batch) for i, batch in batches]
        results = await asyncio.gather(*tasks)
        
        # Final progress update
        if progress_callback:
            progress_callback(total_batches, total_batches)
        
        # Sort results by batch index and flatten
        results.sort(key=lambda x: x[0])
        all_embeddings = []
        for _, embeddings in results:
            all_embeddings.extend(embeddings)
        
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
            if not self.embedding_models:
                return {
                    "status": "unhealthy",
                    "service": "embedding_service",
                    "error": "No embedding models configured"
                }
            
            # Test with a simple embedding using the first available model
            model_name = config_loader.get_selected_model()
            model_config = self.embedding_models[model_name]
            
            test_embedding = await self.generate_embeddings_batch(
                ["health check"],
                model_name,
                batch_size=1,
                progress_callback=None
            )
            
            return {
                "status": "healthy",
                "service": "embedding_service",
                "model": model_name,
                "model_type": model_config.type,
                "dimensions": len(test_embedding[0]) if test_embedding else 0,
                "available_models": list(self.embedding_models.keys()),
                "legacy_models": list(self.legacy_models.keys()),
                "openai_clients": list(self.clients.keys())
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "embedding_service",
                "error": str(e)
            }

# Global embedding service instance
embedding_service = EmbeddingService()