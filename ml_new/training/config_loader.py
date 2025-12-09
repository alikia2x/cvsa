"""
Configuration loader for embedding models and other settings
"""

import toml
import os
from typing import Dict
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class EmbeddingModelConfig(BaseModel):
    name: str
    dimensions: int
    type: str
    api_endpoint: str = "https://api.openai.com/v1"
    max_tokens: int = 8191
    max_batch_size: int = 8
    api_key_env: str = "OPENAI_API_KEY"


class ConfigLoader:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to the embedding_models.toml file we created
            config_path = os.path.join(
                os.path.dirname(__file__), "embedding_models.toml"
            )

        self.config_path = config_path
        self.embedding_models: Dict[str, EmbeddingModelConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from TOML file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}")
                return

            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = toml.load(f)

            # Load embedding models
            if "models" not in config_data:
                return

            for model_key, model_data in config_data["models"].items():
                self.embedding_models[model_key] = EmbeddingModelConfig(
                    **model_data
                )

            logger.info(
                f"Loaded {len(self.embedding_models)} embedding models from {self.config_path}"
            )

        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")

    def get_embedding_models(self) -> Dict[str, EmbeddingModelConfig]:
        """Get all available embedding models"""
        return self.embedding_models.copy()

    def get_embedding_model(self, model_name: str) -> EmbeddingModelConfig:
        """Get specific embedding model config"""
        if model_name not in self.embedding_models:
            raise ValueError(
                f"Embedding model '{model_name}' not found in configuration"
            )
        return self.embedding_models[model_name]

    def list_model_names(self) -> list:
        """Get list of available model names"""
        return list(self.embedding_models.keys())

    def reload_config(self):
        """Reload configuration from file"""
        self.embedding_models = {}
        self._load_config()


# Global config loader instance
config_loader = ConfigLoader()
