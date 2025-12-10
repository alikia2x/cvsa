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


from typing import List, Optional, Dict, Any, Literal
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