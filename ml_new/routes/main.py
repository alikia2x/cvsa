"""
API routes for the ML training service
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ml_new.config.config_loader import config_loader
from models import (
    DatasetBuildRequest, 
    DatasetBuildResponse, 
    TaskStatus, 
    TaskStatusResponse, 
    TaskListResponse,
    SamplingRequest,
    SamplingResponse,
    DatasetCreateRequest,
    DatasetCreateResponse
)
from ml_new.data.dataset_service import DatasetBuilder
from ml_new.config.logger_config import get_logger


logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/v1")

# Global dataset builder instance (will be set by main.py)
dataset_builder: DatasetBuilder = None


def set_dataset_builder(builder: DatasetBuilder):
    """Set the global dataset builder instance"""
    global dataset_builder
    dataset_builder = builder


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    if not dataset_builder:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "message": "Dataset builder not initialized"}
        )
    
    try:
        # Check embedding service health
        embedding_health = await dataset_builder.embedding_service.health_check()
    except Exception as e:
        embedding_health = {"status": "unhealthy", "error": str(e)}
    
    # Check database connection (pool should already be initialized)
    db_status = "disconnected"
    if dataset_builder.db_manager.is_connected:
        try:
            response = await dataset_builder.db_manager.pool.fetch("SELECT 1 FROM information_schema.tables")
            db_status = "connected" if response else "disconnected"
        except Exception as e:
            db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy", 
        "service": "ml-training-api",
        "embedding_service": embedding_health,
        "database": db_status,
        "available_models": list(config_loader.get_embedding_models().keys())
    }


@router.get("/models/embedding")
async def get_embedding_models():
    """Get available embedding models"""
    return {
        "models": {
            name: {
                "name": config.name,
                "dimensions": config.dimensions,
                "type": config.type,
                "api_endpoint": config.api_endpoint,
                "max_tokens": config.max_tokens,
                "max_batch_size": config.max_batch_size
            }
            for name, config in config_loader.get_embedding_models().items()
        }
    }


@router.post("/dataset/build", response_model=DatasetBuildResponse)
async def build_dataset_endpoint(request: DatasetBuildRequest):
    """Build dataset endpoint with task tracking"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    # Validate embedding model
    if request.embedding_model not in config_loader.get_embedding_models():
        raise HTTPException(status_code=400, detail=f"Invalid embedding model: {request.embedding_model}")
    
    dataset_id = str(uuid.uuid4())
    
    # Start task-based dataset building
    task_id = await dataset_builder.start_dataset_build_task(
        dataset_id,
        request.aid_list,
        request.embedding_model,
        request.force_regenerate,
        request.description
    )
    
    return DatasetBuildResponse(
        dataset_id=dataset_id,
        total_records=len(request.aid_list),
        status="started",
        message=f"Dataset building started with task ID: {task_id}",
        description=request.description
    )


@router.get("/dataset/{dataset_id}")
async def get_dataset_endpoint(dataset_id: str):
    """Get built dataset by ID"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    if not dataset_builder.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_info = dataset_builder.get_dataset(dataset_id)
    
    if "error" in dataset_info:
        raise HTTPException(status_code=500, detail=dataset_info["error"])
    
    return {
        "dataset_id": dataset_id,
        "dataset": dataset_info["dataset"],
        "description": dataset_info.get("description"),
        "stats": dataset_info["stats"],
        "created_at": dataset_info["created_at"]
    }


@router.get("/datasets")
async def list_datasets_endpoint():
    """List all built datasets"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    datasets = dataset_builder.list_datasets()
    return {"datasets": datasets}


@router.delete("/dataset/{dataset_id}")
async def delete_dataset_endpoint(dataset_id: str):
    """Delete a built dataset"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    if dataset_builder.delete_dataset(dataset_id):
        return {"message": f"Dataset {dataset_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")


@router.get("/datasets/stats")
async def get_dataset_stats_endpoint():
    """Get overall statistics about stored datasets"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    stats = dataset_builder.get_dataset_stats()
    return stats

# Task Status Endpoints

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status_endpoint(task_id: str):
    """Get status of a specific task"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    task_status = dataset_builder.get_task_status(task_id)
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Convert to response model
    progress_dict = None
    if task_status.progress:
        progress_dict = {
            "current_step": task_status.progress.current_step,
            "total_steps": task_status.progress.total_steps,
            "completed_steps": task_status.progress.completed_steps,
            "percentage": task_status.progress.percentage,
            "message": task_status.progress.message
        }
    
    return TaskStatusResponse(
        task_id=task_status.task_id,
        status=task_status.status,
        progress=progress_dict,
        result=task_status.result,
        error=task_status.error_message,
        created_at=task_status.created_at,
        started_at=task_status.started_at,
        completed_at=task_status.completed_at
    )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks_endpoint(status: Optional[TaskStatus] = None, limit: int = 50):
    """List all tasks, optionally filtered by status"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    tasks = dataset_builder.list_tasks(status_filter=status)
    
    # Limit results
    if limit > 0:
        tasks = tasks[:limit]
    
    # Convert to response models
    task_responses = []
    for task_status in tasks:
        progress_dict = None
        if task_status.progress:
            progress_dict = {
                "current_step": task_status.progress.current_step,
                "total_steps": task_status.progress.total_steps,
                "completed_steps": task_status.progress.completed_steps,
                "percentage": task_status.progress.percentage,
                "message": task_status.progress.message
            }
        
        task_responses.append(TaskStatusResponse(
            task_id=task_status.task_id,
            status=task_status.status,
            progress=progress_dict,
            result=task_status.result,
            error=task_status.error_message,
            created_at=task_status.created_at,
            started_at=task_status.started_at,
            completed_at=task_status.completed_at
        ))
    
    # Get statistics
    stats = dataset_builder.get_task_statistics()
    
    return TaskListResponse(
        tasks=task_responses,
        total_count=stats["total_tasks"],
        pending_count=stats["status_counts"][TaskStatus.PENDING],
        running_count=stats["status_counts"][TaskStatus.RUNNING],
        completed_count=stats["status_counts"][TaskStatus.COMPLETED],
        failed_count=stats["status_counts"][TaskStatus.FAILED]
    )


@router.get("/tasks/stats")
async def get_task_statistics_endpoint():
    """Get statistics about all tasks"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    return dataset_builder.get_task_statistics()


@router.post("/tasks/cleanup")
async def cleanup_tasks_endpoint(max_age_hours: int = 24):
    """Clean up completed/failed tasks older than specified hours"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    cleaned_count = await dataset_builder.cleanup_completed_tasks(max_age_hours)
    return {"message": f"Cleaned up {cleaned_count} tasks older than {max_age_hours} hours"}


# Sampling Endpoints

@router.post("/dataset/sample", response_model=SamplingResponse)
async def sample_dataset_endpoint(request: SamplingRequest):
    """Sample AIDs based on strategy"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    try:
        # Get AIDs based on strategy
        aid_list = await dataset_builder.db_manager.get_aids_by_strategy(
            strategy=request.strategy,
            limit=request.limit,
        )
        
        # Get statistics
        total_available = await dataset_builder.db_manager.get_all_aids_count()
        
        return SamplingResponse(
            strategy=request.strategy,
            total_available=total_available,
            sampled_count=len(aid_list),
            aid_list=aid_list,
            filters_applied={
                "limit": request.limit
            },
            sampling_info={
                "strategy_description": _get_strategy_description(request.strategy),
                "sample_ratio": len(aid_list) / total_available if total_available > 0 else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Sampling failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Sampling failed: {str(e)}")

@router.post("/dataset/create-with-sampling", response_model=DatasetCreateResponse)
async def create_dataset_with_sampling_endpoint(request: DatasetCreateRequest):
    """Create dataset using sampling strategy"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    # Validate embedding model
    if request.embedding_model not in config_loader.get_embedding_models():
        raise HTTPException(status_code=400, detail=f"Invalid embedding model: {request.embedding_model}")
    
    import uuid
    dataset_id = str(uuid.uuid4())
    
    try:
        # First sample the AIDs
        sampling_response = await sample_dataset_endpoint(request.sampling)
        aid_list = sampling_response.aid_list
        
        if not aid_list:
            raise HTTPException(status_code=400, detail="No AIDs found matching the sampling criteria")
        
        # Start task-based dataset building with sampled AIDs
        task_id = await dataset_builder.start_dataset_build_task(
            dataset_id,
            aid_list,
            request.embedding_model,
            request.force_regenerate,
            request.description
        )
        
        return DatasetCreateResponse(
            dataset_id=dataset_id,
            sampling_response=sampling_response,
            task_id=task_id,
            total_records=len(aid_list),
            status="started",
            message=f"Dataset building started with task ID: {task_id}",
            description=request.description
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset creation with sampling failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dataset creation failed: {str(e)}")


def _get_strategy_description(strategy: str) -> str:
    """Get description for sampling strategy"""
    descriptions = {
        "all": "All labeled videos in the database",
        "random": "Randomly sampled labeled videos"
    }
    return descriptions.get(strategy, "Unknown sampling strategy")