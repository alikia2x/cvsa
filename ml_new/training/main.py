"""
Main FastAPI application for ML training service
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import DatabaseManager
from embedding_service import EmbeddingService
from dataset_service import DatasetBuilder
from api_routes import router, set_dataset_builder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global service instances
db_manager = None
embedding_service = None
dataset_builder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    global db_manager, embedding_service, dataset_builder
    
    # Startup
    logger.info("Initializing services...")
    
    try:
        # Database manager
        db_manager = DatabaseManager()
        await db_manager.connect()  # Initialize database connection pool
        logger.info("Database manager initialized and connected")
        
        # Embedding service
        embedding_service = EmbeddingService()
        logger.info("Embedding service initialized")
        
        # Dataset builder
        dataset_builder = DatasetBuilder(db_manager, embedding_service)
        logger.info("Dataset builder initialized")
        
        # Set global dataset builder instance
        set_dataset_builder(dataset_builder)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    # Yield control to the application
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    
    try:
        if db_manager:
            await db_manager.close()
            logger.info("Database connection pool closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Create FastAPI app with lifespan manager
    app = FastAPI(
        title="ML Training Service",
        description="ML training, dataset building, and experiment management service",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router)
    
    return app


def main():
    """Main entry point"""
    app = create_app()
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8322,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()