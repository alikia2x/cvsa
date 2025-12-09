from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import uuid
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CVSA ML Training API", 
    version="1.0.0",
    description="ML training service for video classification"
)

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Hyperparameter(BaseModel):
    name: str
    type: str  # 'number', 'boolean', 'select'
    value: Any
    range: Optional[tuple] = None
    options: Optional[List[str]] = None
    description: Optional[str] = None

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    early_stop: bool = True
    patience: int = 3
    embedding_model: str = "text-embedding-3-small"

class TrainingRequest(BaseModel):
    experiment_name: str
    config: TrainingConfig
    dataset: Dict[str, Any]

class TrainingStatus(BaseModel):
    experiment_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class ExperimentResult(BaseModel):
    experiment_id: str
    experiment_name: str
    config: TrainingConfig
    metrics: Dict[str, float]
    created_at: str
    status: str

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: str

# In-memory storage for experiments (in production, use database)
training_sessions: Dict[str, Dict] = {}
experiments: Dict[str, ExperimentResult] = {}

# Default hyperparameters that will be dynamically discovered
DEFAULT_HYPERPARAMETERS = [
    Hyperparameter(
        name="learning_rate",
        type="number",
        value=1e-4,
        range=(1e-6, 1e-2),
        description="Learning rate for optimizer"
    ),
    Hyperparameter(
        name="batch_size",
        type="number", 
        value=32,
        range=(8, 256),
        description="Training batch size"
    ),
    Hyperparameter(
        name="epochs",
        type="number",
        value=10,
        range=(1, 100),
        description="Number of training epochs"
    ),
    Hyperparameter(
        name="early_stop",
        type="boolean",
        value=True,
        description="Enable early stopping"
    ),
    Hyperparameter(
        name="patience",
        type="number",
        value=3,
        range=(1, 20),
        description="Early stopping patience"
    ),
    Hyperparameter(
        name="embedding_model",
        type="select",
        value="text-embedding-3-small",
        options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        description="Embedding model to use"
    )
]

@app.get("/")
async def root():
    return {"message": "CVSA ML Training API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-training-api"}

@app.get("/hyperparameters", response_model=List[Hyperparameter])
async def get_hyperparameters():
    """Get all available hyperparameters for the current model"""
    return DEFAULT_HYPERPARAMETERS

@app.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new training experiment"""
    experiment_id = str(uuid.uuid4())
    
    # Store training session
    training_sessions[experiment_id] = {
        "experiment_id": experiment_id,
        "experiment_name": request.experiment_name,
        "config": request.config.dict(),
        "dataset": request.dataset,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    # Start background training task
    background_tasks.add_task(run_training, experiment_id, request)
    
    return {"experiment_id": experiment_id}

@app.get("/train/{experiment_id}/status", response_model=TrainingStatus)
async def get_training_status(experiment_id: str):
    """Get training status for an experiment"""
    if experiment_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    session = training_sessions[experiment_id]
    
    return TrainingStatus(
        experiment_id=experiment_id,
        status=session.get("status", "unknown"),
        progress=session.get("progress"),
        current_epoch=session.get("current_epoch"),
        total_epochs=session.get("total_epochs"),
        metrics=session.get("metrics"),
        error=session.get("error")
    )

@app.get("/experiments", response_model=List[ExperimentResult])
async def list_experiments():
    """List all experiments"""
    return list(experiments.values())

@app.get("/experiments/{experiment_id}", response_model=ExperimentResult)
async def get_experiment(experiment_id: str):
    """Get experiment details"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return experiments[experiment_id]

@app.post("/embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings using OpenAI-compatible API"""
    # This is a placeholder implementation
    # In production, this would call actual embedding API
    embeddings = []
    for text in request.texts:
        # Mock embedding generation
        embedding = [0.1] * 1536  # Mock 1536-dimensional embedding
        embeddings.append(embedding)
    
    return embeddings

async def run_training(experiment_id: str, request: TrainingRequest):
    """Background task to run training"""
    try:
        session = training_sessions[experiment_id]
        session["status"] = "running"
        session["total_epochs"] = request.config.epochs
        
        # Simulate training process
        for epoch in range(request.config.epochs):
            session["current_epoch"] = epoch + 1
            session["progress"] = (epoch + 1) / request.config.epochs
            
            # Simulate training metrics
            session["metrics"] = {
                "loss": max(0.0, 1.0 - (epoch + 1) * 0.1),
                "accuracy": min(0.95, 0.5 + (epoch + 1) * 0.05),
                "val_loss": max(0.0, 0.8 - (epoch + 1) * 0.08),
                "val_accuracy": min(0.92, 0.45 + (epoch + 1) * 0.04)
            }
            
            logger.info(f"Training epoch {epoch + 1}/{request.config.epochs}")
            await asyncio.sleep(1)  # Simulate training time
        
        # Training completed
        session["status"] = "completed"
        final_metrics = session["metrics"]
        
        # Store final experiment result
        experiments[experiment_id] = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=request.experiment_name,
            config=request.config,
            metrics=final_metrics,
            created_at=session["created_at"],
            status="completed"
        )
        
        logger.info(f"Training completed for experiment {experiment_id}")
        
    except Exception as e:
        session["status"] = "failed"
        session["error"] = str(e)
        logger.error(f"Training failed for experiment {experiment_id}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)