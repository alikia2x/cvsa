"""
Data models for dataset building functionality
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DatasetBuildRequest(BaseModel):
    """Request model for dataset building"""
    aid_list: List[int] = Field(..., description="List of video AIDs")
    embedding_model: str = Field(..., description="Embedding model name")
    force_regenerate: bool = Field(False, description="Whether to force regenerate embeddings")


class DatasetBuildResponse(BaseModel):
    """Response model for dataset building"""
    dataset_id: str
    total_records: int
    status: str
    message: str
    created_at: Optional[datetime] = None


class DatasetRecord(BaseModel):
    """Model for a single dataset record"""
    aid: int
    embedding: List[float]
    label: bool
    metadata: Dict[str, Any]
    user_labels: List[Dict[str, Any]]
    inconsistent: bool
    text_checksum: str


class DatasetInfo(BaseModel):
    """Model for dataset information"""
    dataset_id: str
    dataset: List[DatasetRecord]
    stats: Dict[str, Any]
    created_at: datetime


class DatasetBuildStats(BaseModel):
    """Statistics for dataset building process"""
    total_records: int
    new_embeddings: int
    reused_embeddings: int
    inconsistent_labels: int
    embedding_model: str
    processing_time: Optional[float] = None


class EmbeddingModelInfo(BaseModel):
    """Information about embedding models"""
    name: str
    dimensions: int
    type: str
    api_endpoint: Optional[str] = None
    max_tokens: Optional[int] = None
    max_batch_size: Optional[int] = None