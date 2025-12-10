#!/usr/bin/env python3
"""
Main training script for embedding classification models
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Add the parent directory to the path to import ml_new modules
sys.path.append(str(Path(__file__).parent.parent))

from ml_new.training.models import create_model
from ml_new.training.data_loader import DatasetLoader
from ml_new.training.trainer import create_trainer
from ml_new.config.logger_config import get_logger


def json_safe_convert(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {str(k): json_safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe_convert(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train embedding classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="ID of the dataset to use for training"
    )
    
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="training/datasets",
        help="Directory containing dataset files"
    )
    
    parser.add_argument(
        "--input-dim",
        type=int,
        default=2048,
        help="Input embedding dimension"
    )
    
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer dimensions"
    )
    
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
        help="Dropout rate for regularization"
    )
    
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        default=True,
        help="Use batch normalization"
    )
    
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "gelu", "tanh", "leaky_relu", "elu"],
        default="relu",
        help="Activation function"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 regularization weight decay"
    )
    
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["plateau", "step", "none"],
        default="plateau",
        help="Learning rate scheduler"
    )
    
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Patience for early stopping"
    )
    
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Minimum improvement for early stopping"
    )
    
    parser.add_argument(
        "--validation-metric",
        type=str,
        choices=["f1", "auc", "accuracy"],
        default="f1",
        help="Metric for model selection and early stopping"
    )
    
    # Data arguments
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize embeddings during training"
    )
    
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        default=False,
        help="Use weighted random sampling for balanced training"
    )
    
    # Output arguments
    parser.add_argument(
        "--save-dir",
        type=str,
        default="training/checkpoints",
        help="Directory to save model checkpoints"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for the experiment (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    
    # Other arguments
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
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test data loading without training"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str):
    """Setup training device"""
    import torch
    
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device


def list_available_datasets(datasets_dir: str):
    """List and display information about available datasets"""
    loader = DatasetLoader(datasets_dir)
    datasets = loader.list_datasets()
    
    print(f"\nAvailable datasets in {datasets_dir}:")
    print("-" * 50)
    
    if not datasets:
        print("No datasets found.")
        return []
    
    for i, dataset_id in enumerate(datasets, 1):
        try:
            info = loader.get_dataset_info(dataset_id)
            print(f"{i}. {dataset_id}")
            print(f"   Samples: {info.get('total_samples', 'N/A')}")
            print(f"   Embedding dim: {info.get('embedding_dim', 'N/A')}")
            print(f"   Classes: {info.get('class_distribution', 'N/A')}")
            print(f"   Created: {info.get('created_at', 'N/A')}")
            print()
        except Exception as e:
            print(f"{i}. {dataset_id} (Error loading info: {e})")
            print()
    
    return datasets


def validate_arguments(args):
    """Validate command line arguments"""
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    if args.dropout_rate < 0 or args.dropout_rate >= 1.0:
        raise ValueError("dropout_rate must be between 0 and 1")


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seeds for reproducibility
    import torch
    import numpy as np
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # List datasets if requested
    if args.list_datasets:
        list_available_datasets(args.datasets_dir)
        return
    
    # Validate arguments
    try:
        validate_arguments(args)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        sys.exit(1)
    
    # Check if dataset exists
    loader = DatasetLoader(args.datasets_dir)
    datasets = loader.list_datasets()
    
    if args.dataset_id not in datasets:
        logger.error(f"Dataset '{args.dataset_id}' not found in {args.datasets_dir}")
        logger.info(f"Available datasets: {datasets}")
        sys.exit(1)
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{timestamp}_{args.dataset_id}"
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Model: hidden dims {args.hidden_dims}")
    
    # Load dataset and create data loaders
    try:
        logger.info("Loading dataset and creating data loaders...")
        train_loader, val_loader, test_loader, data_info = loader.create_data_loaders(
            dataset_id=args.dataset_id,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_state=args.random_seed,
            normalize=args.normalize,
            use_weighted_sampler=args.use_weighted_sampler
        )
        
        logger.info("Data loaders created successfully")
        logger.info(f"  Training samples: {data_info['train_samples']}")
        logger.info(f"  Validation samples: {data_info['val_samples']}")
        logger.info(f"  Test samples: {data_info['test_samples']}")
        logger.info(f"  Training class distribution: {data_info['class_distribution']['train']}")
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        sys.exit(1)
    
    # Create model
    try:
        logger.info("Creating model...")
        model = create_model(
            input_dim=args.input_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout_rate=args.dropout_rate,
            batch_norm=args.batch_norm,
            activation=args.activation
        )
        
        model_info = model.get_model_info()
        logger.info("Model created successfully")
        logger.info(f"  Parameters: {model_info['total_parameters']:,}")
        logger.info(f"  Model size: {model_info['model_size_mb']:.1f} MB")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)
    
    # Test data loading (dry run)
    if args.dry_run:
        logger.info("Dry run: Testing data loading...")
        
        # Test one batch from each loader
        for name, loader_obj in [("train", train_loader), ("val", val_loader)]:
            for batch_embeddings, batch_labels, batch_metadata in loader_obj:
                logger.info(f"  {name} batch - Embeddings: {batch_embeddings.shape}, Labels: {batch_labels.shape}")
                logger.info(f"  Sample labels: {batch_labels[:5].tolist()}")
                break
        
        logger.info("Dry run completed successfully")
        return
    
    # Create trainer
    try:
        logger.info("Creating trainer...")
        
        # Determine scheduler type
        scheduler_type = None if args.scheduler == "none" else args.scheduler
        
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_type=scheduler_type,
            device=device,
            save_dir=args.save_dir,
            experiment_name=args.experiment_name
        )
        
        logger.info("Trainer created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        sys.exit(1)
    
    # Save configuration
    config_path = Path(args.save_dir) / args.experiment_name / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "experiment_name": args.experiment_name,
        "dataset_id": args.dataset_id,
        "model_config": {
            "input_dim": args.input_dim,
            "hidden_dims": args.hidden_dims,
            "dropout_rate": args.dropout_rate,
            "batch_norm": args.batch_norm,
            "activation": args.activation
        },
        "training_config": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "early_stopping_patience": args.early_stopping_patience,
            "validation_metric": args.validation_metric
        },
        "data_config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "normalize": args.normalize,
            "use_weighted_sampler": args.use_weighted_sampler,
            "random_seed": args.random_seed
        },
        "dataset_info": json_safe_convert(data_info)
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
    
    # Start training
    try:
        logger.info("Starting training...")
        results = trainer.train(
            num_epochs=args.num_epochs,
            save_every=args.save_every,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            validation_score=args.validation_metric
        )
        
        # Save results
        results_path = Path(args.save_dir) / args.experiment_name / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Best validation {args.validation_metric}: {results['best_val_score']:.4f}")
        
        if results.get('test_metrics'):
            logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
            logger.info(f"Test F1: {results['test_metrics']['f1']:.4f}")
            logger.info(f"Test AUC: {results['test_metrics']['auc']:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()