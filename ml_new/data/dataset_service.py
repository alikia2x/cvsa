"""
Dataset building service - handles the complete dataset construction flow
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading

from ml_new.data.database import DatabaseManager
from ml_new.data.embedding_service import EmbeddingService
from ml_new.config.config_loader import config_loader
from ml_new.config.logger_config import get_logger
from ml_new.models import TaskStatus, DatasetBuildTaskStatus, TaskProgress
from ml_new.data.dataset_storage_parquet import ParquetDatasetStorage


logger = get_logger(__name__)


class DatasetBuilder:
    """Service for building datasets with the specified flow"""
    
    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService, storage_dir: str = "datasets"):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.storage = ParquetDatasetStorage(str(self.storage_dir))
        
        # Task status tracking
        self._task_status_lock = threading.Lock()
        self.task_statuses: Dict[str, DatasetBuildTaskStatus] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}

    
    def _create_task_status(self, task_id: str, dataset_id: str, aid_list: List[int],
                          embedding_model: str, force_regenerate: bool) -> DatasetBuildTaskStatus:
        """Create initial task status"""
        with self._task_status_lock:
            task_status = DatasetBuildTaskStatus(
                task_id=task_id,
                status=TaskStatus.PENDING,
                dataset_id=dataset_id,
                aid_list=aid_list,
                embedding_model=embedding_model,
                force_regenerate=force_regenerate,
                created_at=datetime.now(),
                progress=TaskProgress(
                    current_step="initialized",
                    total_steps=8,
                    completed_steps=0,
                    percentage=0.0,
                    message="Task initialized"
                )
            )
            self.task_statuses[task_id] = task_status
            return task_status
    
    
    def _update_task_status(self, task_id: str, **kwargs):
        """Update task status with new values"""
        with self._task_status_lock:
            if task_id in self.task_statuses:
                task_status = self.task_statuses[task_id]
                for key, value in kwargs.items():
                    if hasattr(task_status, key):
                        setattr(task_status, key, value)
                self.task_statuses[task_id] = task_status
    
    
    def _update_task_progress(self, task_id: str, current_step: str, completed_steps: int,
                            message: str = None, percentage: float = None):
        """Update task progress"""
        with self._task_status_lock:
            if task_id in self.task_statuses:
                task_status = self.task_statuses[task_id]
                if percentage is not None:
                    progress_percentage = percentage
                else:
                    progress_percentage = (completed_steps / task_status.progress.total_steps) * 100 if task_status.progress else 0.0
                
                task_status.progress = TaskProgress(
                    current_step=current_step,
                    total_steps=task_status.progress.total_steps if task_status.progress else 7,
                    completed_steps=completed_steps,
                    percentage=progress_percentage,
                    message=message
                )
                self.task_statuses[task_id] = task_status
    
    
    def get_task_status(self, task_id: str) -> Optional[DatasetBuildTaskStatus]:
        """Get task status by task ID"""
        with self._task_status_lock:
            return self.task_statuses.get(task_id)
    
    
    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[DatasetBuildTaskStatus]:
        """List all tasks, optionally filtered by status"""
        with self._task_status_lock:
            tasks = list(self.task_statuses.values())
            if status_filter:
                tasks = [task for task in tasks if task.status == status_filter]
            # Sort by creation time (newest first)
            tasks.sort(key=lambda x: x.created_at, reverse=True)
            return tasks
    
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about all tasks"""
        with self._task_status_lock:
            total_tasks = len(self.task_statuses)
            status_counts = {
                TaskStatus.PENDING: 0,
                TaskStatus.RUNNING: 0,
                TaskStatus.COMPLETED: 0,
                TaskStatus.FAILED: 0,
                TaskStatus.CANCELLED: 0
            }
            
            for task_status in self.task_statuses.values():
                status_counts[task_status.status] += 1
            
            return {
                "total_tasks": total_tasks,
                "status_counts": status_counts,
                "running_tasks": status_counts[TaskStatus.RUNNING]
            }
    
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up completed/failed tasks older than specified hours"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        with self._task_status_lock:
            tasks_to_remove = []
            
            for task_id, task_status in self.task_statuses.items():
                if task_status.completed_at:
                    try:
                        completed_time = task_status.completed_at.timestamp()
                        if completed_time < cutoff_time:
                            tasks_to_remove.append(task_id)
                    except Exception as e:
                        logger.warning(f"Failed to check completion time for task {task_id}: {e}")
            
            for task_id in tasks_to_remove:
                # Remove from task statuses
                del self.task_statuses[task_id]
                
                # Remove from running tasks if still there
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
                
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old tasks")
        
        return cleaned_count
    
    
    async def build_dataset_with_task_tracking(self, task_id: str, dataset_id: str, aid_list: List[int],
                                            embedding_model: str, force_regenerate: bool = False,
                                            description: Optional[str] = None) -> str:
        """
        Build dataset with task status tracking
        
        Steps:
        1. Initialize task status
        2. Select embedding model (from TOML config)
        3. Pull raw text from database
        4. Preprocess (placeholder for now)
        5. Batch get embeddings (deduplicate by hash, skip if already in embeddings table)
        6. Write to embeddings table
        7. Pull all needed embeddings to create dataset with format: embeddings, label
        """
        
        # Update task status to running
        self._update_task_status(task_id, status=TaskStatus.RUNNING, started_at=datetime.now())
        
        try:
            logger.info(f"Starting dataset building task {dataset_id} (task_id: {task_id})")
            
            # Step 1: Get model configuration
            EMBEDDING_MODELS = config_loader.get_embedding_models()
            if embedding_model not in EMBEDDING_MODELS:
                raise ValueError(f"Invalid embedding model: {embedding_model}")
            
            model_config = EMBEDDING_MODELS[embedding_model]
            self._update_task_progress(task_id, "getting_metadata", 1, "Retrieving video metadata from database", 3)
            
            # Step 2: Get video metadata from database
            metadata = await self.db_manager.get_video_metadata(aid_list)
            self._update_task_progress(task_id, "getting_labels", 2, "Retrieving user labels from database", 15)
            
            # Step 3: Get user labels
            labels = await self.db_manager.get_user_labels(aid_list)
            self._update_task_progress(task_id, "preparing_text", 3, "Preparing text data and checksums", 30)
            
            # Step 4: Prepare text data and checksums
            text_data = []
            total_aids = len(aid_list)
            
            for i, aid in enumerate(aid_list):
                if aid in metadata:
                    # Combine title, description, tags
                    combined_text = self.embedding_service.combine_video_text(
                        metadata[aid]['title'],
                        metadata[aid]['description'],
                        metadata[aid]['tags']
                    )
                    
                    # Create checksum for deduplication
                    checksum = self.embedding_service.create_text_checksum(combined_text)
                    
                    text_data.append({
                        'aid': aid,
                        'text': combined_text,
                        'checksum': checksum
                    })
                
                # Update progress for text preparation
                if i % 10 == 0 or i == total_aids - 1:  # Update every 10 items or at the end
                    progress_pct = 3 + (i + 1) / total_aids
                    self._update_task_progress(
                        task_id,
                        "preparing_text",
                        min(3, int(progress_pct)),
                        f"Prepared {i + 1}/{total_aids} text entries"
                    )
            
            self._update_task_progress(task_id, "checking_embeddings", 4, "Checking existing embeddings", 33)
            
            # Step 5: Check existing embeddings
            checksums = [item['checksum'] for item in text_data]
            existing_embeddings = await self.db_manager.get_existing_embeddings(checksums, embedding_model)
            
            # Step 6: Generate new embeddings for texts that don't have them
            new_embeddings_needed = []
            for item in text_data:
                if item['checksum'] not in existing_embeddings or force_regenerate:
                    new_embeddings_needed.append(item['text'])
            
            new_embeddings_count = 0
            if new_embeddings_needed:
                self._update_task_progress(
                    task_id,
                    "generating_embeddings",
                    5,
                    f"Generating {len(new_embeddings_needed)} new embeddings",
                    50
                )
                
                # Create progress callback for embedding generation
                def embedding_progress_callback(current: int, total: int):
                    percentage = (current / total) * 25 + 50   # Map to progress range 5-6
                    self._update_task_progress(
                        task_id,
                        "generating_embeddings",
                        5,
                        f"Generated {current}/{total} embeddings",
                        percentage
                    )
                
                logger.info(f"Generating {len(new_embeddings_needed)} new embeddings")
                generated_embeddings = await self.embedding_service.generate_embeddings_batch(
                    new_embeddings_needed,
                    embedding_model,
                    progress_callback=embedding_progress_callback
                )
                
                # Step 7: Store new embeddings in database
                embeddings_to_store = []
                for i, (text, embedding) in enumerate(zip(new_embeddings_needed, generated_embeddings)):
                    checksum = self.embedding_service.create_text_checksum(text)
                    embeddings_to_store.append({
                        'model_name': embedding_model,
                        'checksum': checksum,
                        'dimensions': model_config.dimensions,
                        'vector': embedding
                    })
                
                self._update_task_progress(
                    task_id,
                    "inserting_embeddings",
                    6,
                    f"Inserting {len(embeddings_to_store)} embeddings into database",
                    75
                )
                
                await self.db_manager.insert_embeddings(embeddings_to_store)
                new_embeddings_count = len(embeddings_to_store)
                
                # Update existing embeddings cache
                for emb_data in embeddings_to_store:
                    existing_embeddings[emb_data['checksum']] = {
                        'checksum': emb_data['checksum'],
                        f'vec_{model_config.dimensions}': emb_data['vector']
                    }
            
            self._update_task_progress(task_id, "building_dataset", 7, "Building final dataset", 95)
            
            # Step 8: Build final dataset
            dataset = []
            inconsistent_count = 0
            
            for i, item in enumerate(text_data):
                aid = item['aid']
                checksum = item['checksum']
                
                # Get embedding vector
                embedding_vector = None
                if checksum in existing_embeddings:
                    vec_key = f'vec_{model_config.dimensions}'
                    if vec_key in existing_embeddings[checksum]:
                        embedding_vector = existing_embeddings[checksum][vec_key]
                
                # Get labels for this aid
                aid_labels = labels.get(aid, [])
                
                # Determine final label using consensus (majority vote)
                final_label = None
                if aid_labels:
                    positive_votes = sum(1 for lbl in aid_labels if lbl['label'])
                    final_label = positive_votes > len(aid_labels) / 2
                
                # Check for inconsistent labels
                inconsistent = len(aid_labels) > 1 and (
                    sum(1 for lbl in aid_labels if lbl['label']) != 0 and
                    sum(1 for lbl in aid_labels if lbl['label']) != len(aid_labels)
                )
                
                if inconsistent:
                    inconsistent_count += 1
                
                if embedding_vector and final_label is not None:
                    dataset.append({
                        'aid': aid,
                        'embedding': embedding_vector,
                        'label': final_label,
                        'metadata': metadata.get(aid, {}),
                        'user_labels': aid_labels,
                        'inconsistent': inconsistent,
                        'text_checksum': checksum
                    })
                
                # Update progress for dataset building
                if i % 10 == 0 or i == len(text_data) - 1:  # Update every 10 items or at the end
                    progress_pct = 7 + (i + 1) / len(text_data)
                    self._update_task_progress(
                        task_id,
                        "building_dataset",
                        min(7, int(progress_pct)),
                        f"Built {i + 1}/{len(text_data)} dataset records",
                        100
                    )
            
            reused_count = len(dataset) - new_embeddings_count
            
            logger.info(f"Dataset building completed: {len(dataset)} records, {new_embeddings_count} new, {reused_count} reused, {inconsistent_count} inconsistent")
            
            # Prepare dataset data
            dataset_data = {
                'dataset': dataset,
                'description': description,
                'stats': {
                    'total_records': len(dataset),
                    'new_embeddings': new_embeddings_count,
                    'reused_embeddings': reused_count,
                    'inconsistent_labels': inconsistent_count,
                    'embedding_model': embedding_model
                },
                'created_at': datetime.now().isoformat()
            }
            
            self._update_task_progress(task_id, "saving_dataset", 8, "Saving dataset to storage")
            
            # Save to file using efficient storage (Parquet format)
            if self.storage.save_dataset(dataset_id, dataset_data['dataset'], description, dataset_data['stats']):
                logger.info(f"Dataset {dataset_id} saved to efficient storage (Parquet format)")
            else:
                logger.warning(f"Failed to save dataset {dataset_id} to storage")
            
            # Update task status to completed
            result = {
                'dataset_id': dataset_id,
                'stats': dataset_data['stats']
            }
            
            self._update_task_status(
                task_id,
                status=TaskStatus.COMPLETED,
                completed_at=datetime.now(),
                result=result,
                progress=TaskProgress(
                    current_step="completed",
                    total_steps=8,
                    completed_steps=8,
                    percentage=100.0,
                    message="Dataset building completed successfully"
                )
            )
            
            return dataset_id
            
        except Exception as e:
            logger.error(f"Dataset building failed for {dataset_id}: {str(e)}")
            
            # Update task status to failed
            self._update_task_status(
                task_id,
                status=TaskStatus.FAILED,
                completed_at=datetime.now(),
                error_message=str(e),
                progress=TaskProgress(
                    current_step="failed",
                    total_steps=8,
                    completed_steps=0,
                    percentage=0.0,
                    message=f"Task failed: {str(e)}"
                )
            )
            
            # Store error information
            error_data = {
                'error': str(e),
                'created_at': datetime.now().isoformat()
            }
            
            # Try to save error using new storage
            self.storage.save_dataset(dataset_id, [], description=str(e), stats={'error': True})
            raise
    
    
    async def start_dataset_build_task(self, dataset_id: str, aid_list: List[int],
                                     embedding_model: str, force_regenerate: bool = False,
                                     description: Optional[str] = None) -> str:
        """
        Start a dataset building task and return task ID for status tracking
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        # Create task status
        task_status = self._create_task_status(task_id, dataset_id, aid_list, embedding_model, force_regenerate)
        
        # Start the actual task
        task = asyncio.create_task(
            self.build_dataset_with_task_tracking(task_id, dataset_id, aid_list, embedding_model, force_regenerate, description)
        )
        
        # Store the running task
        with self._task_status_lock:
            self.running_tasks[task_id] = task
        
        return task_id
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get built dataset by ID"""
        # Use pure Parquet storage interface
        return self.storage.load_dataset_full(dataset_id)
    
    def dataset_exists(self, dataset_id: str) -> bool:
        """Check if dataset exists"""
        return self.storage.dataset_exists(dataset_id)
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset from both memory and file system"""
        return self.storage.delete_dataset(dataset_id)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets with their basic information"""
        return self.storage.list_datasets()
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get overall statistics about stored datasets"""
        return self.storage.get_dataset_stats()
    
    def load_dataset_metadata_fast(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.load_dataset_metadata(dataset_id)
    
    def load_dataset_partial(self, dataset_id: str, columns: Optional[List[str]] = None,
                           filters: Optional[Dict[str, Any]] = None):
        return self.storage.load_dataset_partial(dataset_id, columns, filters)