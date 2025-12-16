"""
Efficient dataset storage using Parquet format for better space utilization and loading performance
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from ml_new.config.logger_config import get_logger

logger = get_logger(__name__)

class ParquetDatasetStorage:    
    def __init__(self, storage_dir: str = "datasets"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Parquet file extension
        self.parquet_ext = ".parquet"
        self.metadata_ext = ".metadata.json"
        
        # In-memory cache: only cache metadata to avoid large file memory usage
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._load_metadata_cache()
    
    def _get_dataset_files(self, dataset_id: str) -> tuple[Path, Path]:
        """Get file paths for the dataset"""
        base_path = self.storage_dir / dataset_id
        data_file = base_path.with_suffix(self.parquet_ext)
        metadata_file = base_path.with_suffix(self.metadata_ext)
        return data_file, metadata_file
    
    def _load_metadata_cache(self):
        """Load metadata cache"""
        try:
            for metadata_file in self.storage_dir.glob("*.metadata.json"):
                try:
                    # Remove ".metadata" suffix
                    dataset_id = metadata_file.stem[:-9]  
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    self.metadata_cache[dataset_id] = metadata
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {metadata_file}: {e}")
            
            logger.info(f"Loaded metadata for {len(self.metadata_cache)} datasets")
            
        except Exception as e:
            logger.error(f"Failed to load metadata cache: {e}")
    
    def save_dataset(self, dataset_id: str, dataset: List[Dict[str, Any]], 
                    description: Optional[str] = None, stats: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save dataset using Parquet format
        
        Args:
            dataset_id: Dataset ID
            dataset: Dataset content
            description: Dataset description
            stats: Dataset statistics
            
        Returns:
            bool: Whether the save was successful
        """
        try:
            data_file, metadata_file = self._get_dataset_files(dataset_id)
            
            # Ensure directory exists
            data_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data: convert embedding vectors to numpy arrays
            if not dataset:
                logger.warning(f"Empty dataset for {dataset_id}")
                return False
            
            # Analyze data structure
            first_item = dataset[0]
            embedding_dim = len(first_item.get('embedding', []))
            
            # Build DataFrame
            records = []
            for item in dataset:
                record = {
                    'aid': item.get('aid'),
                    'label': item.get('label'),
                    'inconsistent': item.get('inconsistent', False),
                    'text_checksum': item.get('text_checksum'),
                    # Store embedding as a separate column
                    'embedding': item.get('embedding', []),
                    # Store metadata as JSON string
                    'metadata_json': json.dumps(item.get('metadata', {}), ensure_ascii=False),
                    'user_labels_json': json.dumps(item.get('user_labels', []), ensure_ascii=False)
                }
                records.append(record)
            
            # Create DataFrame
            df = pd.DataFrame(records)
            
            # Convert embedding column to numpy arrays
            df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32) if x else np.array([], dtype=np.float32))
            
            # Use PyArrow Schema for type safety
            schema = pa.schema([
                ('aid', pa.int64()),
                ('label', pa.bool_()),
                ('inconsistent', pa.bool_()),
                ('text_checksum', pa.string()),
                ('embedding', pa.list_(pa.float32())),
                ('metadata_json', pa.string()),
                ('user_labels_json', pa.string())
            ])
            
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df, schema=schema)
            
            # Write Parquet file with efficient compression settings
            pq.write_table(
                table, 
                data_file,
                compression='zstd',  # Better compression ratio
                compression_level=6,  # Balance compression ratio and speed
                use_dictionary=True,  # Enable dictionary encoding
                write_page_index=True,  # Support fast metadata access
                write_statistics=True  # Enable statistics
            )
            
            # Save metadata
            metadata = {
                'dataset_id': dataset_id,
                'description': description,
                'stats': stats or {},
                'created_at': datetime.now().isoformat(),
                'file_format': 'parquet_v1',
                'embedding_dimension': embedding_dim,
                'total_records': len(dataset),
                'columns': list(df.columns),
                'file_size_bytes': data_file.stat().st_size,
                'compression': 'zstd'
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Update cache
            self.metadata_cache[dataset_id] = metadata
            
            logger.info(f"Saved dataset {dataset_id} to Parquet: {len(dataset)} records, {data_file.stat().st_size} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dataset {dataset_id}: {e}")
            return False
    
    def _regenerate_metadata_from_parquet(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Regenerate metadata from parquet file when metadata JSON is missing or corrupted
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dict: Regenerated metadata, or None if failed
        """
        try:
            data_file, metadata_file = self._get_dataset_files(dataset_id)
            if not data_file.exists():
                logger.warning(f"Parquet file not found for dataset {dataset_id}, cannot regenerate metadata")
                return None
            
            # Read parquet file to extract basic information
            table = pq.read_table(data_file, columns=['aid', 'label', 'embedding', 'metadata_json', 'user_labels_json'])
            
            # Extract embedding dimension from first row
            first_embedding = table.column('embedding')[0]
            embedding_dim = len(first_embedding) if first_embedding else 0
            
            # Get total records count
            total_records = len(table)
            
            # Generate regenerated metadata
            regenerated_metadata = {
                'dataset_id': dataset_id,
                'description': f'Auto-regenerated metadata for dataset {dataset_id}',
                'stats': {
                    'total_records': total_records,
                    'regenerated': True,
                    'regeneration_reason': 'missing_or_corrupted_metadata_file'
                },
                'created_at': datetime.fromtimestamp(data_file.stat().st_mtime).isoformat(),
                'file_format': 'parquet_v1',
                'embedding_dimension': embedding_dim,
                'total_records': total_records,
                'columns': ['aid', 'label', 'inconsistent', 'text_checksum', 'embedding', 'metadata_json', 'user_labels_json'],
                'file_size_bytes': data_file.stat().st_size,
                'compression': 'zstd'
            }
            
            # Save regenerated metadata to file
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(regenerated_metadata, f, ensure_ascii=False, indent=2)
            
            # Update cache
            self.metadata_cache[dataset_id] = regenerated_metadata
            
            logger.info(f"Successfully regenerated metadata for dataset {dataset_id} from parquet file")
            return regenerated_metadata
            
        except Exception as e:
            logger.error(f"Failed to regenerate metadata for dataset {dataset_id}: {e}")
            return None
    
    def load_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Quickly load dataset metadata (without loading the entire file)
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dict: Metadata, or None if not found
        """
        # Check cache
        if dataset_id in self.metadata_cache:
            return self.metadata_cache[dataset_id]
        
        # Load from file
        _, metadata_file = self._get_dataset_files(dataset_id)
        if not metadata_file.exists():
            # Try to regenerate metadata from parquet file
            logger.warning(f"Metadata file not found for dataset {dataset_id}, attempting to regenerate from parquet")
            return self._regenerate_metadata_from_parquet(dataset_id)
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Update cache
            self.metadata_cache[dataset_id] = metadata
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata for {dataset_id}: {e}")
            # Try to regenerate metadata from parquet file
            logger.warning(f"Metadata file corrupted for dataset {dataset_id}, attempting to regenerate from parquet")
            return self._regenerate_metadata_from_parquet(dataset_id)
    
    def load_dataset_partial(self, dataset_id: str, columns: Optional[List[str]] = None, 
                           filters: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Partially load the dataset (only reading specified columns or rows meeting criteria)
        
        Args:
            dataset_id: Dataset ID
            columns: Columns to read, None to read all
            filters: Filtering conditions, format {column: value}
            
        Returns:
            pd.DataFrame: Loaded data, or None if failed
        """
        data_file, _ = self._get_dataset_files(dataset_id)
        if not data_file.exists():
            return None
        
        try:
            # Read Parquet file, supporting column selection and filtering
            if columns:
                # Ensure necessary columns exist
                all_columns = ['aid', 'label', 'inconsistent', 'text_checksum', 'embedding', 'metadata_json', 'user_labels_json']
                required_cols = ['aid', 'label', 'embedding']  # These are fundamentally needed
                columns = list(set(columns + required_cols))
                
                # Filter out non-existent columns
                columns = [col for col in columns if col in all_columns]
            
            # Use pyarrow to read, supporting filters
            if filters:
                # Build filter expressions
                expressions = []
                for col, value in filters.items():
                    if col == 'label':
                        expressions.append(pa.compute.equal(pa.field(col), value))
                    elif col == 'aid':
                        expressions.append(pa.compute.equal(pa.field(col), value))
                
                if expressions:
                    filter_expr = expressions[0]
                    for expr in expressions[1:]:
                        filter_expr = pa.compute.and_(filter_expr, expr)
                else:
                    filter_expr = None
            else:
                filter_expr = None
            
            # Read data
            if columns and filter_expr:
                table = pq.read_table(data_file, columns=columns, filter=filter_expr)
            elif columns:
                table = pq.read_table(data_file, columns=columns)
            elif filter_expr:
                table = pq.read_table(data_file, filter=filter_expr)
            else:
                table = pq.read_table(data_file)
            
            # Convert to DataFrame
            df = table.to_pandas()
            
            # Handle embedding column
            if 'embedding' in df.columns:
                df['embedding'] = df['embedding'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else list(x))
            
            logger.info(f"Loaded partial dataset {dataset_id}: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load partial dataset {dataset_id}: {e}")
            return None
    
    def load_dataset_full(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Fully load the dataset (maintaining backward compatibility format)
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dict: Full dataset data, or None if failed
        """
        data_file, _ = self._get_dataset_files(dataset_id)
        if not data_file.exists():
            return None
        
        try:
            # Load metadata
            metadata = self.load_dataset_metadata(dataset_id)
            if not metadata:
                return None
            
            # Load data
            df = self.load_dataset_partial(dataset_id)
            if df is None:
                return None
            
            # Convert to original format
            dataset = []
            for _, row in df.iterrows():
                record = {
                    'aid': int(row['aid']),
                    'embedding': row['embedding'],
                    'label': bool(row['label']),
                    'metadata': json.loads(row['metadata_json']) if row['metadata_json'] else {},
                    'user_labels': json.loads(row['user_labels_json']) if row['user_labels_json'] else [],
                    'inconsistent': bool(row['inconsistent']),
                    'text_checksum': row['text_checksum']
                }
                dataset.append(record)
            
            return {
                'dataset': dataset,
                'description': metadata.get('description'),
                'stats': metadata.get('stats', {}),
                'created_at': metadata.get('created_at')
            }
            
        except Exception as e:
            logger.error(f"Failed to load full dataset {dataset_id}: {e}")
            return None
    
    def dataset_exists(self, dataset_id: str) -> bool:
        """Check if the dataset exists"""
        data_file, _ = self._get_dataset_files(dataset_id)
        return data_file.exists()
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset"""
        try:
            data_file, metadata_file = self._get_dataset_files(dataset_id)
            
            # Delete files
            if data_file.exists():
                data_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from cache
            if dataset_id in self.metadata_cache:
                del self.metadata_cache[dataset_id]
            
            logger.info(f"Deleted dataset {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            return False
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List metadata for all datasets"""
        datasets = []
        
        # Scan for all parquet files and try to load their metadata
        for parquet_file in self.storage_dir.glob("*.parquet"):
            try:
                # Extract dataset_id from filename (remove .parquet extension)
                dataset_id = parquet_file.stem
                
                # Try to load metadata, this will automatically regenerate if missing
                metadata = self.load_dataset_metadata(dataset_id)
                if metadata:
                    datasets.append({
                        "dataset_id": dataset_id,
                        "description": metadata.get("description"),
                        "stats": metadata.get("stats", {}),
                        "created_at": metadata.get("created_at"),
                        "total_records": metadata.get("total_records", 0),
                        "file_size_mb": round(metadata.get("file_size_bytes", 0) / (1024 * 1024), 2),
                        "embedding_dimension": metadata.get("embedding_dimension"),
                        "file_format": metadata.get("file_format")
                    })
            except Exception as e:
                logger.warning(f"Failed to process dataset {parquet_file.stem}: {e}")
                continue
        
        # Sort by creation time descending
        datasets.sort(key=lambda x: x["created_at"], reverse=True)
        return datasets
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        total_datasets = len(self.metadata_cache)
        total_records = sum(m.get("total_records", 0) for m in self.metadata_cache.values())
        total_size_bytes = sum(m.get("file_size_bytes", 0) for m in self.metadata_cache.values())
        
        return {
            "total_datasets": total_datasets,
            "total_records": total_records,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "average_size_mb": round(total_size_bytes / total_datasets / (1024 * 1024), 2) if total_datasets > 0 else 0,
            "storage_directory": str(self.storage_dir),
            "storage_format": "parquet_v1"
        }
    
    def migrate_from_json(self, dataset_id: str, json_data: Dict[str, Any]) -> bool:
        """
        Migrate a dataset from JSON format to Parquet format
        
        Args:
            dataset_id: Dataset ID
            json_data: Data in JSON format
            
        Returns:
            bool: Migration success status
        """
        try:
            dataset = json_data.get('dataset', [])
            description = json_data.get('description')
            stats = json_data.get('stats')
            
            return self.save_dataset(dataset_id, dataset, description, stats)
            
        except Exception as e:
            logger.error(f"Failed to migrate dataset {dataset_id} from JSON: {e}")
            return False