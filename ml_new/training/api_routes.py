"""
API routes for the ML training service
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from config_loader import config_loader
from models import DatasetBuildRequest, DatasetBuildResponse
from dataset_service import DatasetBuilder


logger = logging.getLogger(__name__)

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
async def build_dataset_endpoint(request: DatasetBuildRequest, background_tasks: BackgroundTasks):
    """Build dataset endpoint"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    # Validate embedding model
    if request.embedding_model not in config_loader.get_embedding_models():
        raise HTTPException(status_code=400, detail=f"Invalid embedding model: {request.embedding_model}")
    
    dataset_id = str(uuid.uuid4())
    # Start background task for dataset building
    background_tasks.add_task(
        dataset_builder.build_dataset,
        dataset_id,
        request.aid_list,
        request.embedding_model,
        request.force_regenerate
    )
    
    return DatasetBuildResponse(
        dataset_id=dataset_id,
        total_records=len(request.aid_list),
        status="started",
        message="Dataset building started"
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
        "stats": dataset_info["stats"],
        "created_at": dataset_info["created_at"]
    }


@router.get("/datasets")
async def list_datasets():
    """List all built datasets"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    datasets = []
    for dataset_id, dataset_info in dataset_builder.dataset_storage.items():
        if "error" not in dataset_info:
            datasets.append({
                "dataset_id": dataset_id,
                "stats": dataset_info["stats"],
                "created_at": dataset_info["created_at"]
            })
    
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


@router.get("/datasets")
async def list_datasets_endpoint():
    """List all built datasets"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    datasets = dataset_builder.list_datasets()
    return {"datasets": datasets}


@router.get("/datasets/stats")
async def get_dataset_stats_endpoint():
    """Get overall statistics about stored datasets"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    stats = dataset_builder.get_dataset_stats()
    return stats


@router.post("/datasets/cleanup")
async def cleanup_datasets_endpoint(max_age_days: int = 30):
    """Remove datasets older than specified days"""
    
    if not dataset_builder:
        raise HTTPException(status_code=503, detail="Dataset builder not available")
    
    await dataset_builder.cleanup_old_datasets(max_age_days)
    return {"message": f"Cleanup completed for datasets older than {max_age_days} days"}