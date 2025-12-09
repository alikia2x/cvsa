# ML Training Service

A FastAPI-based ML training service for dataset building, embedding generation, and experiment management.

## Architecture

The service is organized into modular components:

```
ml_new/training/
├── main.py              # FastAPI application entry point
├── models.py            # Pydantic data models
├── config_loader.py     # Configuration loading from TOML
├── database.py          # Database connection and operations
├── embedding_service.py # Embedding generation service
├── dataset_service.py   # Dataset building logic
├── api_routes.py        # API endpoint definitions
├── embedding_models.toml # Embedding model configurations
└── requirements.txt     # Python dependencies
```

## Key Components

### 1. Main Application (`main.py`)
- FastAPI app initialization
- CORS middleware configuration
- Service dependency injection
- Startup/shutdown event handlers

### 2. Data Models (`models.py`)
- `DatasetBuildRequest`: Request model for dataset building
- `DatasetBuildResponse`: Response model for dataset building
- `DatasetRecord`: Individual dataset record structure
- `EmbeddingModelInfo`: Embedding model configuration

### 3. Configuration (`config_loader.py`)
- Loads embedding model configurations from TOML
- Manages model parameters (dimensions, API endpoints, etc.)

### 4. Database Layer (`database.py`)
- PostgreSQL connection management
- CRUD operations for video metadata, user labels, and embeddings
- Optimized batch queries to avoid N+1 problems

### 5. Embedding Service (`embedding_service.py`)
- Integration with OpenAI-compatible embedding APIs
- Text preprocessing and checksum generation
- Batch embedding generation with rate limiting

### 6. Dataset Building (`dataset_service.py`)
- Complete dataset construction workflow:
  1. Pull raw text from database
  2. Text preprocessing (placeholder)
  3. Batch embedding generation with deduplication
  4. Embedding storage and caching
  5. Final dataset compilation with labels

### 7. API Routes (`api_routes.py`)
- `/api/v1/health`: Health check
- `/api/v1/models/embedding`: List available embedding models
- `/api/v1/dataset/build`: Build new dataset
- `/api/v1/dataset/{id}`: Retrieve built dataset
- `/api/v1/datasets`: List all datasets
- `/api/v1/dataset/{id}`: Delete dataset

## Dataset Building Flow

1. **Model Selection**: Choose embedding model from TOML configuration
2. **Data Retrieval**: Pull video metadata and user labels from PostgreSQL
3. **Text Processing**: Combine title, description, and tags
4. **Deduplication**: Generate checksums to avoid duplicate embeddings
5. **Batch Processing**: Generate embeddings for new texts only
6. **Storage**: Store embeddings in database with caching
7. **Final Assembly**: Combine embeddings with labels using consensus mechanism

## Configuration

### Embedding Models (`embedding_models.toml`)
```toml
[text-embedding-3-large]
name = "text-embedding-3-large"
dimensions = 3072
type = "openai"
api_endpoint = "https://api.openai.com/v1/embeddings"
max_tokens = 8192
max_batch_size = 100
```

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_KEY`: OpenAI API key for embedding generation

## Usage

### Start the Service
```bash
cd ml_new/training
python main.py
```

### Build a Dataset
```bash
curl -X POST "http://localhost:8322/v1/dataset/build" \
  -H "Content-Type: application/json" \
  -d '{
    "aid_list": [170001, 170002, 170003],
    "embedding_model": "text-embedding-3-large",
    "force_regenerate": false
  }'
```

### Check Health
```bash
curl "http://localhost:8322/v1/health"
```

### List Embedding Models
```bash
curl "http://localhost:8322/v1/models/embedding"
```

## Features

- **High Performance**: Optimized database queries with batch operations
- **Deduplication**: Text-level deduplication using MD5 checksums
- **Consensus Labels**: Majority vote mechanism for user annotations
- **Batch Processing**: Efficient embedding generation and storage
- **Error Handling**: Comprehensive error handling and logging
- **Async Support**: Fully asynchronous operations for scalability
- **CORS Enabled**: Ready for frontend integration

## Production Considerations

- Replace in-memory dataset storage with database
- Add authentication and authorization
- Implement rate limiting for API endpoints
- Add monitoring and metrics collection
- Configure proper logging levels
- Set up database connection pooling
- Add API documentation with OpenAPI/Swagger