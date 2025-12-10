"""
Model trainer for embedding classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np
from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path
from ml_new.training.models import EmbeddingClassifier
from ml_new.config.logger_config import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trainer for embedding classification models
    """
    
    def __init__(
        self,
        model: EmbeddingClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_dir: str = "training/checkpoints",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize model trainer
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            optimizer: Optimizer instance (default: Adam)
            criterion: Loss function (default: BCEWithLogitsLoss)
            scheduler: Learning rate scheduler (optional)
            device: Device to use (default: auto-detect)
            save_dir: Directory to save checkpoints
            experiment_name: Name for the experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler
        
        # Training configuration
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = self.save_dir / self.experiment_name
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "logs"))
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model info: {self.model.get_model_info()}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (embeddings, labels, metadata) in enumerate(self.train_loader):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device).float()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(embeddings)
            loss = self.criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Collect statistics
            total_loss += loss.item()
            
            # Get predictions for metrics
            with torch.no_grad():
                predictions = torch.sigmoid(outputs).squeeze()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Log progress
            if batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, (np.array(all_predictions) > 0.5).astype(int))
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for embeddings, labels, metadata in self.val_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass
                outputs = self.model(embeddings)
                loss = self.criterion(outputs.squeeze(), labels)
                
                # Collect statistics
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.sigmoid(outputs).squeeze()
                predictions = (probabilities > 0.5).long()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        )
        
        try:
            val_auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            val_auc = 0.0  # AUC not defined for single class
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            'loss': val_loss,
            'accuracy': val_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': val_auc,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
    
    def train(
        self,
        num_epochs: int = 100,
        save_every: int = 10,
        early_stopping_patience: int = 15,
        min_delta: float = 0.001,
        validation_score: str = 'f1'
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            validation_score: Metric to use for early stopping ('f1', 'auc', 'accuracy')
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
            self.writer.add_scalar('AUC/Validation', val_metrics['auc'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_auc'].append(val_metrics['auc'])
            self.training_history['learning_rates'].append(current_lr)
            
            # Print progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}"
            )
            
            # Check for best model
            val_score = val_metrics[validation_score]
            if val_score > self.best_val_score + min_delta:
                self.best_val_score = val_score
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(is_best=True)
                logger.info(f"New best {validation_score}: {val_score:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final evaluation on test set if available
        test_metrics = None
        if self.test_loader:
            test_metrics = self.evaluate_test_set()
        
        # Save final training state
        self.save_training_state()
        
        # Close TensorBoard writer
        self.writer.close()
        
        results = {
            'experiment_name': self.experiment_name,
            'best_val_score': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'training_history': self.training_history,
            'final_val_metrics': self.validate(),
            'test_metrics': test_metrics,
            'model_info': self.model.get_model_info()
        }
        
        logger.info(f"Training completed. Best validation {validation_score}: {self.best_val_score:.4f}")
        return results
    
    def evaluate_test_set(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for embeddings, labels, metadata in self.test_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass
                outputs = self.model(embeddings)
                loss = self.criterion(outputs.squeeze(), labels)
                
                # Collect statistics
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.sigmoid(outputs).squeeze()
                predictions = (probabilities > 0.5).long()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        test_loss = total_loss / len(self.test_loader)
        test_accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        )
        
        try:
            test_auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            test_auc = 0.0
        
        logger.info(f"Test Set Results - Loss: {test_loss:.4f}, Acc: {test_accuracy:.4f}, F1: {f1:.4f}, AUC: {test_auc:.4f}")
        
        return {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': test_auc
        }
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'training_history': self.training_history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'dropout_rate': self.model.dropout_rate,
                'batch_norm': self.model.batch_norm
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_model_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
    
    def save_training_state(self) -> None:
        """Save final training state"""
        state = {
            'experiment_name': self.experiment_name,
            'best_val_score': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'training_history': self.training_history,
            'model_info': self.model.get_model_info(),
            'training_config': {
                'optimizer': self.optimizer.__class__.__name__,
                'criterion': self.criterion.__class__.__name__,
                'device': str(self.device)
            }
        }
        
        state_path = self.checkpoint_dir / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved training state to {state_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> None:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def create_trainer(
    model: EmbeddingClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    scheduler_type: Optional[str] = 'plateau',
    **kwargs
) -> ModelTrainer:
    """
    Factory function to create a configured trainer
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        scheduler_type: Learning rate scheduler type ('plateau', 'step', None)
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured ModelTrainer instance
    """
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    scheduler = None
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.1
        )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        **kwargs
    )
    
    return trainer


if __name__ == "__main__":
    # Test trainer creation
    from ml_new.training.models import create_model
    from ml_new.training.data_loader import DatasetLoader
    
    # Create dummy model and data
    model = create_model(
        model_type="standard",
        input_dim=2048,
        hidden_dims=(512, 256, 128)
    )
    
    loader = DatasetLoader()
    datasets = loader.list_datasets()
    
    if datasets:
        train_loader, val_loader, test_loader, data_info = loader.create_data_loaders(
            datasets[0],
            batch_size=8,
            normalize=True
        )
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            experiment_name="test_experiment"
        )
        
        # Test one epoch
        print("Testing trainer...")
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.validate()
        
        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_metrics}")