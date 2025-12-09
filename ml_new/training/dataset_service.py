"""
Dataset building service - handles the complete dataset construction flow
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from database import DatabaseManager
from embedding_service import EmbeddingService
from config_loader import config_loader


logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Service for building datasets with the specified flow"""
    
    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService, storage_dir: str = "datasets"):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Load existing datasets from file system
        self.dataset_storage: Dict[str, Dict] = self._load_all_datasets()


    def _get_dataset_file_path(self, dataset_id: str) -> Path:
        """Get file path for dataset"""
        return self.storage_dir / f"{dataset_id}.json"
    
    
    def _load_dataset_from_file(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Load dataset from file"""
        file_path = self._get_dataset_file_path(dataset_id)
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id} from file: {e}")
            return None
    
    
    def _save_dataset_to_file(self, dataset_id: str, dataset_data: Dict[str, Any]) -> bool:
        """Save dataset to file"""
        file_path = self._get_dataset_file_path(dataset_id)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save dataset {dataset_id} to file: {e}")
            return False
    
    
    def _load_all_datasets(self) -> Dict[str, Dict]:
        """Load all datasets from file system"""
        datasets = {}
        
        try:
            for file_path in self.storage_dir.glob("*.json"):
                dataset_id = file_path.stem
                dataset_data = self._load_dataset_from_file(dataset_id)
                if dataset_data:
                    datasets[dataset_id] = dataset_data
            logger.info(f"Loaded {len(datasets)} datasets from file system")
        except Exception as e:
            logger.error(f"Failed to load datasets from file system: {e}")
        
        return datasets
    
    
    async def cleanup_old_datasets(self, max_age_days: int = 30):
        """Remove datasets older than specified days"""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            removed_count = 0
            
            for dataset_id in list(self.dataset_storage.keys()):
                dataset_info = self.dataset_storage[dataset_id]
                if "created_at" in dataset_info:
                    try:
                        created_time = datetime.fromisoformat(dataset_info["created_at"]).timestamp()
                        if created_time < cutoff_time:
                            # Remove from memory
                            del self.dataset_storage[dataset_id]
                            
                            # Remove file
                            file_path = self._get_dataset_file_path(dataset_id)
                            if file_path.exists():
                                file_path.unlink()
                            
                            removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to process dataset {dataset_id} for cleanup: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old datasets")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old datasets: {e}")
    
    async def build_dataset(self, dataset_id: str, aid_list: List[int], embedding_model: str, force_regenerate: bool = False) -> str:
        """
        Build dataset with the specified flow:
        1. Select embedding model (from TOML config)
        2. Pull raw text from database
        3. Preprocess (placeholder for now)
        4. Batch get embeddings (deduplicate by hash, skip if already in embeddings table)
        5. Write to embeddings table
        6. Pull all needed embeddings to create dataset with format: embeddings, label
        """
        
        try:
            logger.info(f"Starting dataset building task {dataset_id}")
            
            EMBEDDING_MODELS = config_loader.get_embedding_models()
            
            # Get model configuration
            if embedding_model not in EMBEDDING_MODELS:
                raise ValueError(f"Invalid embedding model: {embedding_model}")
            
            model_config = EMBEDDING_MODELS[embedding_model]
            
            # Step 1: Get video metadata from database
            metadata = await self.db_manager.get_video_metadata(aid_list)
            
            # Step 2: Get user labels
            labels = await self.db_manager.get_user_labels(aid_list)
            
            # Step 3: Prepare text data and checksums
            text_data = []
            
            for aid in aid_list:
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
            
            # Step 4: Check existing embeddings
            checksums = [item['checksum'] for item in text_data]
            existing_embeddings = await self.db_manager.get_existing_embeddings(checksums, embedding_model)
            
            # Step 5: Generate new embeddings for texts that don't have them
            new_embeddings_needed = []
            for item in text_data:
                if item['checksum'] not in existing_embeddings or force_regenerate:
                    new_embeddings_needed.append(item['text'])
            
            new_embeddings_count = 0
            if new_embeddings_needed:
                logger.info(f"Generating {len(new_embeddings_needed)} new embeddings")
                generated_embeddings = await self.embedding_service.generate_embeddings_batch(
                    new_embeddings_needed, 
                    embedding_model
                )
                
                # Step 6: Store new embeddings in database
                embeddings_to_store = []
                for i, (text, embedding) in enumerate(zip(new_embeddings_needed, generated_embeddings)):
                    checksum = self.embedding_service.create_text_checksum(text)
                    embeddings_to_store.append({
                        'model_name': embedding_model,
                        'checksum': checksum,
                        'dimensions': model_config.dimensions,
                        'vector': embedding
                    })
                
                await self.db_manager.insert_embeddings(embeddings_to_store)
                new_embeddings_count = len(embeddings_to_store)
                
                # Update existing embeddings cache
                for emb_data in embeddings_to_store:
                    existing_embeddings[emb_data['checksum']] = {
                        'checksum': emb_data['checksum'],
                        f'vec_{model_config.dimensions}': emb_data['vector']
                    }
            
            # Step 7: Build final dataset
            dataset = []
            inconsistent_count = 0
            
            for item in text_data:
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
            
            reused_count = len(dataset) - new_embeddings_count
            
            logger.info(f"Dataset building completed: {len(dataset)} records, {new_embeddings_count} new, {reused_count} reused, {inconsistent_count} inconsistent")
            
            # Prepare dataset data
            dataset_data = {
                'dataset': dataset,
                'stats': {
                    'total_records': len(dataset),
                    'new_embeddings': new_embeddings_count,
                    'reused_embeddings': reused_count,
                    'inconsistent_labels': inconsistent_count,
                    'embedding_model': embedding_model
                },
                'created_at': datetime.now().isoformat()
            }
            
            # Save to file and memory cache
            if self._save_dataset_to_file(dataset_id, dataset_data):
                self.dataset_storage[dataset_id] = dataset_data
                logger.info(f"Dataset {dataset_id} saved to file system")
            else:
                logger.warning(f"Failed to save dataset {dataset_id} to file, keeping in memory only")
                self.dataset_storage[dataset_id] = dataset_data
            
            return dataset_id
            
        except Exception as e:
            logger.error(f"Dataset building failed for {dataset_id}: {str(e)}")
            
            # Store error information
            error_data = {
                'error': str(e),
                'created_at': datetime.now().isoformat()
            }
            
            # Try to save error to file as well
            self._save_dataset_to_file(dataset_id, error_data)
            self.dataset_storage[dataset_id] = error_data
            raise
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get built dataset by ID"""
        # First check memory cache
        if dataset_id in self.dataset_storage:
            return self.dataset_storage[dataset_id]
        
        # If not in memory, try to load from file
        dataset_data = self._load_dataset_from_file(dataset_id)
        if dataset_data:
            # Add to memory cache
            self.dataset_storage[dataset_id] = dataset_data
            return dataset_data
        
        return None
    
    def dataset_exists(self, dataset_id: str) -> bool:
        """Check if dataset exists"""
        # Check memory cache first
        if dataset_id in self.dataset_storage:
            return True
        
        # Check file system
        return self._get_dataset_file_path(dataset_id).exists()
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset from both memory and file system"""
        try:
            # Remove from memory
            if dataset_id in self.dataset_storage:
                del self.dataset_storage[dataset_id]
            
            # Remove file
            file_path = self._get_dataset_file_path(dataset_id)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Dataset {dataset_id} deleted from file system")
                return True
            else:
                logger.warning(f"Dataset file {dataset_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            return False
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets with their basic information"""
        datasets = []
        
        for dataset_id, dataset_info in self.dataset_storage.items():
            if "error" not in dataset_info:
                datasets.append({
                    "dataset_id": dataset_id,
                    "stats": dataset_info["stats"],
                    "created_at": dataset_info["created_at"]
                })
        
        # Sort by creation time (newest first)
        datasets.sort(key=lambda x: x["created_at"], reverse=True)
        
        return datasets
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get overall statistics about stored datasets"""
        total_datasets = len(self.dataset_storage)
        error_datasets = sum(1 for data in self.dataset_storage.values() if "error" in data)
        valid_datasets = total_datasets - error_datasets
        
        total_records = 0
        total_new_embeddings = 0
        total_reused_embeddings = 0
        
        for dataset_info in self.dataset_storage.values():
            if "stats" in dataset_info:
                stats = dataset_info["stats"]
                total_records += stats.get("total_records", 0)
                total_new_embeddings += stats.get("new_embeddings", 0)
                total_reused_embeddings += stats.get("reused_embeddings", 0)
        
        return {
            "total_datasets": total_datasets,
            "valid_datasets": valid_datasets,
            "error_datasets": error_datasets,
            "total_records": total_records,
            "total_new_embeddings": total_new_embeddings,
            "total_reused_embeddings": total_reused_embeddings,
            "storage_directory": str(self.storage_dir)
        }