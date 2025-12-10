"""
Data loader for embedding datasets
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_new.config.logger_config import get_logger

logger = get_logger(__name__)


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for embedding-based classification
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        transform: Optional[callable] = None,
        normalize: bool = True
    ):
        """
        Initialize embedding dataset
        
        Args:
            embeddings: Array of embedding vectors (n_samples, embedding_dim)
            labels: Array of binary labels (n_samples,)
            metadata: Optional list of metadata dictionaries
            transform: Optional transformation function
            normalize: Whether to normalize embeddings
        """
        assert len(embeddings) == len(labels), "Embeddings and labels must have same length"
        
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.metadata = metadata or []
        self.transform = transform
        
        # Normalize embeddings if requested
        if normalize and len(embeddings) > 0:
            self.scaler = StandardScaler()
            self.embeddings = self.scaler.fit_transform(self.embeddings)
        else:
            self.scaler = None
        
        # Calculate class weights for balanced sampling
        self._calculate_class_weights()
    
    def _calculate_class_weights(self):
        """Calculate weights for each class for balanced sampling"""
        unique, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        
        self.class_weights = {}
        for class_label, count in zip(unique, counts):
            # Inverse frequency weighting
            weight = total_samples / (2 * count)
            self.class_weights[class_label] = weight
        
        logger.info(f"Class distribution: {dict(zip(unique, counts))}")
        logger.info(f"Class weights: {self.class_weights}")
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample from the dataset
        
        Returns:
            tuple: (embedding, label, metadata)
        """
        embedding = torch.from_numpy(self.embeddings[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        metadata = {}
        if self.metadata and idx < len(self.metadata):
            metadata = self.metadata[idx]
        
        if self.transform:
            embedding = self.transform(embedding)
        
        return embedding, label, metadata


class DatasetLoader:
    """
    Loader for embedding datasets stored in Parquet format
    """
    
    def __init__(self, datasets_dir: str = "training/datasets"):
        """
        Initialize dataset loader
        
        Args:
            datasets_dir: Directory containing dataset files
        """
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_id: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Load a dataset by ID from Parquet files
        
        Args:
            dataset_id: Unique identifier for the dataset
            
        Returns:
            tuple: (embeddings, labels, metadata_list)
        """
        dataset_file = self.datasets_dir / f"{dataset_id}.parquet"
        metadata_file = self.datasets_dir / f"{dataset_id}.metadata.json"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        # Load metadata
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Load data from Parquet
        logger.info(f"Loading dataset {dataset_id} from {dataset_file}")
        df = pd.read_parquet(dataset_file)
        
        # Extract embeddings (they might be stored as list or numpy array)
        embeddings = self._extract_embeddings(df)
        
        # Extract labels
        labels = df['label'].values.astype(np.int64)
        
        # Extract metadata
        metadata_list = []
        if 'metadata_json' in df.columns:
            for _, row in df.iterrows():
                meta = {}
                if pd.notna(row.get('metadata_json')):
                    try:
                        meta = json.loads(row['metadata_json'])
                    except (json.JSONDecodeError, TypeError):
                        meta = {}
                
                # Add other fields
                meta.update({
                    'aid': row.get('aid'),
                    'inconsistent': row.get('inconsistent', False),
                    'text_checksum': row.get('text_checksum')
                })
                metadata_list.append(meta)
        else:
            # Create basic metadata
            metadata_list = [{
                'aid': aid,
                'inconsistent': inconsistent,
                'text_checksum': checksum
            } for aid, inconsistent, checksum in zip(
                df.get('aid', []),
                df.get('inconsistent', [False] * len(df)),
                df.get('text_checksum', [''] * len(df))
            )]
        
        logger.info(f"Loaded dataset with {len(embeddings)} samples, {embeddings.shape[1]} embedding dimensions")
        
        return embeddings, labels, metadata_list
    
    def _extract_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Extract embeddings from DataFrame, handling different storage formats"""
        embedding_col = None
        for col in ['embedding', 'embeddings', 'vec_2048', 'vec_1024']:
            if col in df.columns:
                embedding_col = col
                break
        
        if embedding_col is None:
            raise ValueError("No embedding column found in dataset")
        
        embeddings_data = df[embedding_col]
        
        # Handle different embedding storage formats
        if embeddings_data.dtype == 'object':
            # Likely stored as lists or numpy arrays
            embeddings = np.array([
                np.array(emb) if isinstance(emb, (list, np.ndarray)) else np.zeros(2048)
                for emb in embeddings_data
            ])
        else:
            # Already numpy array
            embeddings = embeddings_data.values
        
        # Ensure 2D array
        if embeddings.ndim == 1:
            # If embeddings are flattened, reshape
            embedding_dim = len(embeddings) // len(df)
            embeddings = embeddings.reshape(len(df), embedding_dim)
        
        return embeddings.astype(np.float32)
    
    def create_data_loaders(
        self,
        dataset_id: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        random_state: int = 42,
        normalize: bool = True,
        use_weighted_sampler: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
        """
        Create train, validation, and test data loaders
        
        Args:
            dataset_id: Dataset identifier
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            random_state: Random seed for reproducibility
            normalize: Whether to normalize embeddings
            use_weighted_sampler: Whether to use weighted random sampling
            
        Returns:
            tuple: (train_loader, val_loader, test_loader, dataset_info)
        """
        # Load dataset
        embeddings, labels, metadata = self.load_dataset(dataset_id)
        
        # Split data
        (
            train_emb, test_emb,
            train_lbl, test_lbl,
            train_meta, test_meta
        ) = train_test_split(
            embeddings, labels, metadata,
            test_size=1 - train_ratio,
            stratify=labels,
            random_state=random_state
        )
        
        # Split test into val and test
        val_size = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
        (
            val_emb, test_emb,
            val_lbl, test_lbl,
            val_meta, test_meta
        ) = train_test_split(
            test_emb, test_lbl, test_meta,
            test_size=1 - val_size,
            stratify=test_lbl,
            random_state=random_state
        )
        
        # Create datasets
        train_dataset = EmbeddingDataset(train_emb, train_lbl, train_meta, normalize=normalize)
        val_dataset = EmbeddingDataset(val_emb, val_lbl, val_meta, normalize=False)  # Don't re-normalize
        test_dataset = EmbeddingDataset(test_emb, test_lbl, test_meta, normalize=False)
        
        # Create samplers
        train_sampler = None
        if use_weighted_sampler and hasattr(train_dataset, 'class_weights'):
            # Create weighted sampler for balanced training
            sample_weights = [train_dataset.class_weights[label] for label in train_dataset.labels]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Dataset info
        dataset_info = {
            'dataset_id': dataset_id,
            'total_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'train_ratio': len(train_dataset) / len(embeddings),
            'val_ratio': len(val_dataset) / len(embeddings),
            'test_ratio': len(test_dataset) / len(embeddings),
            'class_distribution': {
                'train': dict(zip(*np.unique(train_dataset.labels, return_counts=True))),
                'val': dict(zip(*np.unique(val_dataset.labels, return_counts=True))),
                'test': dict(zip(*np.unique(test_dataset.labels, return_counts=True)))
            },
            'normalize': normalize,
            'use_weighted_sampler': use_weighted_sampler
        }
        
        logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, dataset_info
    
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        parquet_files = list(self.datasets_dir.glob("*.parquet"))
        return [f.stem for f in parquet_files]
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a dataset"""
        metadata_file = self.datasets_dir / f"{dataset_id}.metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        # Fallback: load dataset and return basic info
        embeddings, labels, metadata = self.load_dataset(dataset_id)
        return {
            'dataset_id': dataset_id,
            'total_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'file_format': 'parquet',
            'created_at': 'unknown'
        }


if __name__ == "__main__":
    # Test dataset loading
    loader = DatasetLoader()
    
    # List available datasets
    datasets = loader.list_datasets()
    print(f"Available datasets: {datasets}")
    
    if datasets:
        # Test loading first dataset
        dataset_id = datasets[0]
        print(f"\nTesting dataset: {dataset_id}")
        
        info = loader.get_dataset_info(dataset_id)
        print("Dataset info:", info)
        
        # Test creating data loaders
        try:
            train_loader, val_loader, test_loader, data_info = loader.create_data_loaders(
                dataset_id,
                batch_size=8,
                normalize=True
            )
            
            print("\nData loader info:")
            for key, value in data_info.items():
                print(f"  {key}: {value}")
            
            # Test single batch
            for batch_embeddings, batch_labels, batch_metadata in train_loader:
                print(f"\nBatch test:")
                print(f"  Embeddings shape: {batch_embeddings.shape}")
                print(f"  Labels shape: {batch_labels.shape}")
                print(f"  Sample labels: {batch_labels[:5].tolist()}")
                break
                
        except Exception as e:
            print(f"Error creating data loaders: {e}")