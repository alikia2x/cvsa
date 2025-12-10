#!/usr/bin/env python3
"""
Test script for evaluating trained models on a dataset
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import aiohttp
import asyncio
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from typing import Optional, List, Dict, Any

# Add the parent directory to the path to import ml_new modules
sys.path.append(str(Path(__file__).parent.parent))

from ml_new.training.models import create_model
from ml_new.training.data_loader import DatasetLoader, EmbeddingDataset
from ml_new.config.logger_config import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test embedding classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="ID of the dataset to use for testing"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        help="Name of the experiment to load model from"
    )
    
    # Optional arguments
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="training/datasets",
        help="Directory containing dataset files"
    )
    
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="training/checkpoints",
        help="Directory containing model checkpoints"
    )
    
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="best_model.pth",
        help="Checkpoint file to load (relative to experiment dir)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for testing"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize embeddings during testing"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )
    
    parser.add_argument(
        "--use-api",
        action="store_true",
        default=False,
        help="Use API model instead of local model"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8544",
        help="API base URL"
    )
    
    parser.add_argument(
        "--misclassified-output",
        type=str,
        default=None,
        help="Output file for misclassified samples (FN and FP aids)"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str):
    """Setup device"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    return device


def load_model_from_experiment(
    checkpoints_dir: str,
    experiment_name: str,
    checkpoint_file: str,
    device: torch.device
):
    """
    Load a trained model from an experiment checkpoint
    
    Args:
        checkpoints_dir: Directory containing checkpoints
        experiment_name: Name of the experiment
        checkpoint_file: Checkpoint file name
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    checkpoint_path = Path(checkpoints_dir) / experiment_name / checkpoint_file
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    model_config = checkpoint.get('model_config', {})
    
    # Create model with saved config
    model = create_model(
        input_dim=model_config.get('input_dim', 2048),
        hidden_dims=tuple(model_config.get('hidden_dims', [512, 256, 128])),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        batch_norm=model_config.get('batch_norm', True)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Model config: {model_config}")
    
    return model, model_config


def safe_extract_aid(metadata_entry):
    """Safely extract aid from metadata entry"""
    if isinstance(metadata_entry, dict) and 'aid' in metadata_entry:
        return metadata_entry['aid']
    return None

def normalize_batch_metadata(metadata, expected_batch_size):
    """
    Normalize batch metadata to ensure consistent structure
    
    Args:
        metadata: Raw metadata from DataLoader (could be various formats)
        expected_batch_size: Expected number of metadata entries
        
    Returns:
        List of metadata dictionaries
    """
    # Handle different metadata structures
    if metadata is None:
        return [{}] * expected_batch_size
    
    if isinstance(metadata, dict):
        # Single metadata object - duplicate for entire batch
        return [metadata] * expected_batch_size
    
    if isinstance(metadata, (list, tuple)):
        if len(metadata) == expected_batch_size:
            return list(metadata)
        elif len(metadata) < expected_batch_size:
            # Pad with empty dicts
            padded = list(metadata) + [{}] * (expected_batch_size - len(metadata))
            return padded
        else:
            # Truncate to expected size
            return list(metadata[:expected_batch_size])
    
    # Unknown format - return empty dicts
    logger.warning(f"Unknown metadata format: {type(metadata)}")
    return [{}] * expected_batch_size

def evaluate_model(
    model,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
):
    """
    Evaluate model on test set
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to use
        threshold: Classification threshold
        
    Returns:
        Tuple of (metrics, predictions, probabilities, true_labels, fn_aids, fp_aids)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_metadata = []
    fn_aids = []
    fp_aids = []
    
    with torch.no_grad():
        for batch_idx, (embeddings, labels, metadata) in enumerate(test_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device).float()
            
            # Forward pass
            outputs = model(embeddings)
            
            # Get predictions and probabilities
            probabilities = torch.sigmoid(outputs).squeeze()
            predictions = (probabilities > threshold).long()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Collect metadata and track FN/FP
            batch_size = len(labels)
            batch_metadata = normalize_batch_metadata(metadata, batch_size)
            all_metadata.extend(batch_metadata)
            
            # Track FN and FP aids for this batch
            logger.debug(f"Batch {batch_idx}: labels shape {labels.shape}, predictions shape {predictions.shape}, metadata structure: {type(batch_metadata)}")
            if len(batch_metadata) != len(labels):
                logger.warning(f"Metadata length mismatch: {len(batch_metadata)} metadata entries vs {len(labels)} samples")
            
            for i, (true_label, pred_label) in enumerate(zip(labels.cpu().numpy(), predictions.cpu().numpy())):
                try:
                    # Safely get metadata entry with bounds checking
                    if i >= len(batch_metadata):
                        logger.warning(f"Index {i} out of range for batch_metadata (length: {len(batch_metadata)})")
                        continue
                    
                    meta_entry = batch_metadata[i]
                    if not isinstance(meta_entry, dict):
                        logger.warning(f"Metadata entry {i} is not a dict: {type(meta_entry)}")
                        continue
                    
                    if 'aid' not in meta_entry:
                        logger.debug(f"No 'aid' key in metadata entry {i}")
                        continue
                    
                    aid = safe_extract_aid(meta_entry)
                    if aid is not None:
                        if true_label == 1 and pred_label == 0:  # False Negative
                            fn_aids.append(aid)
                        elif true_label == 0 and pred_label == 1:  # False Positive
                            fp_aids.append(aid)
                        
                except Exception as e:
                    logger.warning(f"Error processing metadata entry {i}: {e}")
                    continue
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    test_accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    try:
        test_auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        test_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics = {
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': test_auc,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'total_samples': len(all_labels),
        'threshold': threshold
    }
    
    # Add class distribution
    unique, counts = np.unique(all_labels, return_counts=True)
    metrics['class_distribution'] = {int(k): int(v) for k, v in zip(unique, counts)}
    
    return metrics, all_predictions, all_probabilities, all_labels, fn_aids, fp_aids


async def call_api_batch(session: aiohttp.ClientSession, api_url: str, requests: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Call the classification API for batch requests"""
    try:
        url = f"{api_url}/classify_batch"
        async with session.post(url, json=requests) as response:
            if response.status == 200:
                result = await response.json()
                return result.get('results', [])
            else:
                logger.warning(f"Batch API request failed with status {response.status}")
                return None
    except Exception as e:
        logger.warning(f"Batch API request failed: {e}")
        return None


def convert_api_label_to_bool(api_label: int) -> int:
    """Convert API label to boolean (non-zero = true)"""
    return 1 if api_label != 0 else 0


async def evaluate_with_api(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    api_url: str,
    batch_size: int = 32
):
    """
    Evaluate using the API instead of local model
    
    Args:
        embeddings: Array of embeddings (not used for API calls)
        labels: Ground truth labels
        metadata: Metadata containing title, description, tags, aid
        api_url: API base URL
        batch_size: Number of requests per API batch call
        
    Returns:
        Tuple of (metrics, predictions, probabilities, true_labels, fn_aids, fp_aids)
    """
    logger.info(f"Using API at {api_url} for evaluation")
    
    # Prepare API requests
    requests = []
    for i, meta in enumerate(metadata):
        # Extract metadata fields for API
        title = meta.get('title', '')
        description = meta.get('description', '')
        tags = meta.get('tags', '')
        aid = meta.get('aid', i)
        
        # Handle missing or empty fields
        if not title:
            title = f"Video {aid}"
        if not description:
            description = ""
        if not tags:
            tags = ""
            
        request_data = {
            "title": title,
            "description": description,
            "tags": tags,
            "aid": aid
        }
        requests.append(request_data)
    
    # Split requests into batches
    num_batches = (len(requests) + batch_size - 1) // batch_size
    logger.info(f"Making {num_batches} batch API requests with batch_size={batch_size} for {len(requests)} total requests")
    
    # Process all batches
    all_predictions = []
    all_probabilities = []
    all_labels = labels.tolist()
    all_aids = [meta.get('aid', i) for i, meta in enumerate(metadata)]
    failed_requests = 0
    fn_aids = []
    fp_aids = []
    
    async with aiohttp.ClientSession() as session:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(requests))
            batch_requests = requests[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_requests)} requests)")
            
            results = await call_api_batch(session, api_url, batch_requests)
            
            if results is None:
                logger.error(f"Batch {batch_idx + 1} API request failed completely")
                # Create dummy results for this batch
                all_predictions.extend([0] * len(batch_requests))
                all_probabilities.extend([0.0] * len(batch_requests))
                failed_requests += len(batch_requests)
                continue
            
            for i, result in enumerate(results):
                global_idx = start_idx + i
                if not isinstance(result, dict) or 'error' in result:
                    error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Invalid result'
                    logger.warning(f"Failed to get API prediction for request {global_idx}: {error_msg}")
                    failed_requests += 1
                    all_predictions.append(0)
                    all_probabilities.append(0.0)
                    continue
                    
                # Convert API response to our format
                api_label = result.get('label', -1)
                probabilities = result.get('probabilities')
                
                # Convert to boolean (non-zero = true)
                prediction = convert_api_label_to_bool(api_label)
                # Use the probability of the positive class
                if probabilities and len(probabilities) > 0:
                    positive_prob = 1 - probabilities[0]
                else:
                    logger.warning(f"No probabilities for request {global_idx}")
                    failed_requests += 1
                    all_predictions.append(0)
                    all_probabilities.append(0.0)
                    continue
                    
                all_predictions.append(prediction)
                all_probabilities.append(positive_prob)
    
    if failed_requests > 0:
        logger.warning(f"Failed to get API predictions for {failed_requests} requests")
    
    # Collect FN and FP aids
    for i, (true_label, pred_label) in enumerate(zip(all_labels, all_predictions)):
        aid = all_aids[i]
        if true_label == 1 and pred_label == 0:  # False Negative
            fn_aids.append(aid)
        elif true_label == 0 and pred_label == 1:  # False Positive
            fp_aids.append(aid)
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    try:
        test_auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        test_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics = {
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': test_auc,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'total_samples': len(all_labels),
        'failed_requests': failed_requests
    }
    
    # Add class distribution
    unique, counts = np.unique(all_labels, return_counts=True)
    metrics['class_distribution'] = {int(k): int(v) for k, v in zip(unique, counts)}
    
    return metrics, all_predictions, all_probabilities, all_labels, fn_aids, fp_aids


def main():
    """Main test function"""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Check if dataset exists
    loader = DatasetLoader(args.datasets_dir)
    datasets = loader.list_datasets()
    
    if args.dataset_id not in datasets:
        logger.error(f"Dataset '{args.dataset_id}' not found in {args.datasets_dir}")
        logger.info(f"Available datasets: {datasets}")
        sys.exit(1)
    
    # Load dataset (use entire dataset as test set)
    try:
        logger.info(f"Loading dataset {args.dataset_id}...")
        embeddings, labels, metadata = loader.load_dataset(args.dataset_id)
        
        logger.info(f"Dataset loaded: {len(embeddings)} samples")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Choose evaluation method
    if args.use_api:
        # Use API for evaluation
        logger.info("Using API-based evaluation")
        
        # Run async evaluation
        metrics, predictions, probabilities, true_labels, fn_aids, fp_aids = asyncio.run(
            evaluate_with_api(
                embeddings, labels, metadata,
                args.api_url,
                args.batch_size
            )
        )
        
        # For API mode, we don't have model_config
        model_config = {"type": "api", "api_url": args.api_url}
        
    else:
        # Use local model for evaluation
        # Check if experiment exists
        experiment_dir = Path(args.checkpoints_dir) / args.experiment
        if not experiment_dir.exists():
            logger.error(f"Experiment '{args.experiment}' not found in {args.checkpoints_dir}")
            available = [d.name for d in Path(args.checkpoints_dir).iterdir() if d.is_dir()]
            logger.info(f"Available experiments: {available}")
            sys.exit(1)
        
        # Load model
        try:
            model, model_config = load_model_from_experiment(
                args.checkpoints_dir,
                args.experiment,
                args.checkpoint_file,
                device
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
        
        # Create test dataset and loader
        test_dataset = EmbeddingDataset(
            embeddings, labels, metadata,
            normalize=args.normalize
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Evaluate model
        logger.info("Starting local model evaluation...")
        metrics, predictions, probabilities, true_labels, fn_aids, fp_aids = evaluate_model(
            model, test_loader, device, args.threshold
        )
    
    # Print results
    logger.info("=" * 50)
    logger.info("Test Results")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.dataset_id}")
    if args.use_api:
        logger.info(f"Method: API ({args.api_url})")
    else:
        logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Class distribution: {metrics['class_distribution']}")
    if 'failed_requests' in metrics:
        logger.info(f"Failed API requests: {metrics['failed_requests']}")
    logger.info("-" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info("-" * 50)
    logger.info(f"True Positives: {metrics['true_positives']}")
    logger.info(f"True Negatives: {metrics['true_negatives']}")
    logger.info(f"False Positives: {metrics['false_positives']}")
    logger.info(f"False Negatives: {metrics['false_negatives']}")
    logger.info("=" * 50)
    
    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'dataset_id': args.dataset_id,
            'experiment': args.experiment,
            'checkpoint': args.checkpoint_file,
            'model_config': model_config,
            'metrics': metrics,
            'predictions': [int(p) for p in predictions],
            'probabilities': [float(p) for p in probabilities],
            'labels': [int(l) for l in true_labels]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to {output_path}")
    
    # Save misclassified samples (FN and FP aids) if requested
    if args.misclassified_output:
        misclassified_path = Path(args.misclassified_output)
        misclassified_path.parent.mkdir(parents=True, exist_ok=True)
        
        misclassified_data = {
            'false_negatives': fn_aids,
            'false_positives': fp_aids,
            'fn_count': len(fn_aids),
            'fp_count': len(fp_aids),
            'total_misclassified': len(fn_aids) + len(fp_aids)
        }
        
        with open(misclassified_path, 'w') as f:
            json.dump(misclassified_data, f, indent=2)
        
        logger.info(f"Misclassified samples (FN: {len(fn_aids)}, FP: {len(fp_aids)}) saved to {misclassified_path}")


if __name__ == "__main__":
    main()
