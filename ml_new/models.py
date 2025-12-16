"""
Data models for dataset building functionality
"""

from typing import List, Optional, Dict, Any, Literal
import uuid
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SamplingStrategy(str, Enum):
    """Sampling strategy enumeration"""
    ALL = "all"  # All labeled AIDs
    RANDOM = "random"  # Random sampling from labeled data


class TaskProgress(BaseModel):
    """Progress information for a task"""
    current_step: str
    total_steps: int
    completed_steps: int
    percentage: float
    message: Optional[str] = None
    estimated_time_remaining: Optional[float] = None


class DatasetBuildTaskStatus(BaseModel):
    """Status model for dataset building task"""
    task_id: str
    status: TaskStatus
    dataset_id: Optional[str] = None
    aid_list: List[int]
    embedding_model: str
    force_regenerate: bool
    progress: Optional[TaskProgress] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


class DatasetBuildRequest(BaseModel):
    """Request model for dataset building"""
    id: Optional[str] = Field(str(uuid.uuid4()), description="Dataset ID")
    aid_list: List[int] = Field(..., description="List of video AIDs")
    embedding_model: str = Field(..., description="Embedding model name")
    force_regenerate: bool = Field(False, description="Whether to force regenerate embeddings")
    description: Optional[str] = Field(None, description="Optional description for the dataset")


class DatasetBuildResponse(BaseModel):
    """Response model for dataset building"""
    dataset_id: str
    total_records: int
    status: str
    message: str
    description: Optional[str] = None
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


# Sampling and Dataset Selection Models

class SamplingRequest(BaseModel):
    """Request model for dataset sampling"""
    strategy: SamplingStrategy = Field(..., description="Sampling strategy to use")
    limit: Optional[int] = Field(None, description="Maximum number of AIDs to sample (for random sampling)")


class SamplingResponse(BaseModel):
    """Response model for dataset sampling"""
    strategy: SamplingStrategy
    total_available: int
    sampled_count: int
    aid_list: List[int]
    filters_applied: Optional[Dict[str, Any]] = None
    sampling_info: Dict[str, Any]


class DatasetCreateRequest(BaseModel):
    """Request model for creating dataset with sampling"""
    sampling: SamplingRequest = Field(..., description="Sampling configuration")
    embedding_model: str = Field(..., description="Embedding model name")
    force_regenerate: bool = Field(False, description="Whether to force regenerate embeddings")
    description: Optional[str] = Field(None, description="Optional description for the dataset")


class DatasetCreateResponse(BaseModel):
    """Response model for dataset creation"""
    dataset_id: str
    sampling_response: SamplingResponse
    task_id: str
    total_records: int
    status: str
    message: str
    description: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Response model for task status endpoint"""
    task_id: str
    status: TaskStatus
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskListResponse(BaseModel):
    """Response model for listing tasks"""
    tasks: List[TaskStatusResponse]
    total_count: int
    pending_count: int
    running_count: int
    completed_count: int
    failed_count: int