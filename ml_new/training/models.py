"""
Neural network model for binary classification of video embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ml_new.config.logger_config import get_logger

logger = get_logger(__name__)


class EmbeddingClassifier(nn.Module):
    """
    Neural network model for binary classification of video embeddings
    
    Architecture:
    - Embedding layer (configurable input dimension)
    - Hidden layers with dropout and batch normalization
    - Binary classification head
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: Optional[Tuple[int, ...]] = None,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu"
    ):
        """
        Initialize the embedding classifier model
        
        Args:
            input_dim: Dimension of input embeddings (default: 2048 for qwen3-embedding)
            hidden_dims: Tuple of hidden layer dimensions (default: (512, 256, 128))
            dropout_rate: Dropout probability for regularization
            batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super(EmbeddingClassifier, self).__init__()
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = (512, 256, 128)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input dimension to first hidden layer
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear_layer = nn.Linear(prev_dim, hidden_dim)
            self.layers.append(linear_layer)
            
            # Batch normalization (optional)
            if batch_norm:
                bn_layer = nn.BatchNorm1d(hidden_dim)
                self.layers.append(bn_layer)
            
            # Activation function
            activation_layer = self._get_activation(activation)
            self.layers.append(activation_layer)
            
            # Dropout
            dropout_layer = nn.Dropout(dropout_rate)
            self.layers.append(dropout_layer)
            
            prev_dim = hidden_dim
        
        # Binary classification head
        self.classifier = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized EmbeddingClassifier: input_dim={input_dim}, hidden_dims={hidden_dims}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        
        if activation not in activations:
            logger.warning(f"Unknown activation '{activation}', using ReLU")
            return nn.ReLU()
        
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input embeddings of shape (batch_size, input_dim)
            
        Returns:
            logits of shape (batch_size, 1)
        """
        # Ensure input is float tensor
        if not x.dtype == torch.float32:
            x = x.float()
        
        # Flatten input if it's multi-dimensional (shouldn't be for embeddings)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x)
        
        # Final classification layer
        logits = self.classifier(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities
        
        Args:
            x: Input embeddings
            
        Returns:
            Class probabilities of shape (batch_size, 2)
        """
        logits = self.forward(x)
        probabilities = torch.sigmoid(logits)
        
        # Convert to [negative_prob, positive_prob] format
        prob_0 = 1 - probabilities
        prob_1 = probabilities
        
        return torch.cat([prob_0, prob_1], dim=1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict class labels
        
        Args:
            x: Input embeddings
            threshold: Classification threshold
            
        Returns:
            Binary predictions of shape (batch_size,)
        """
        probabilities = self.predict_proba(x)
        predictions = (probabilities[:, 1] > threshold).long()
        return predictions
    
    def get_model_info(self) -> dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_class': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class AttentionEmbeddingClassifier(EmbeddingClassifier):
    """
    Enhanced classifier with self-attention mechanism
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: Optional[Tuple[int, ...]] = None,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        attention_dim: int = 512
    ):
        super().__init__(input_dim, hidden_dims, dropout_rate, batch_norm, activation)
        
        # Self-attention mechanism
        self.attention_dim = attention_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Attention projection layer
        self.attention_projection = nn.Linear(input_dim, attention_dim)
        
        # Re-initialize attention weights
        self._initialize_attention_weights()
        
        logger.info(f"Initialized AttentionEmbeddingClassifier with attention_dim={attention_dim}")
    
    def _initialize_attention_weights(self):
        """Initialize attention mechanism weights"""
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism
        
        Args:
            x: Input embeddings of shape (batch_size, input_dim)
            
        Returns:
            logits of shape (batch_size, 1)
        """
        # Ensure input is float tensor
        if not x.dtype == torch.float32:
            x = x.float()
        
        # Add sequence dimension for attention (batch_size, 1, input_dim)
        x_expanded = x.unsqueeze(1)
        
        # Apply self-attention
        attended, attention_weights = self.attention(x_expanded, x_expanded, x_expanded)
        
        # Remove sequence dimension (batch_size, input_dim)
        attended = attended.squeeze(1)
        
        # Project to attention dimension
        attended = self.attention_projection(attended)
        
        # Process through original classification layers
        for layer in self.layers:
            attended = layer(attended)
        
        # Final classification layer
        logits = self.classifier(attended)
        
        return logits


def create_model(
    model_type: str = "standard",
    input_dim: int = 2048,
    hidden_dims: Optional[Tuple[int, ...]] = None,
    **kwargs
) -> EmbeddingClassifier:
    """
    Factory function to create embedding classifier models
    
    Args:
        model_type: Type of model ('standard', 'attention')
        input_dim: Input embedding dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    if model_type == "standard":
        return EmbeddingClassifier(input_dim=input_dim, hidden_dims=hidden_dims, **kwargs)
    elif model_type == "attention":
        return AttentionEmbeddingClassifier(input_dim=input_dim, hidden_dims=hidden_dims, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model(
        model_type="standard",
        input_dim=2048,
        hidden_dims=(512, 256, 128),
        dropout_rate=0.3
    )
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 8
    dummy_input = torch.randn(batch_size, 2048)
    
    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)
        predictions = model.predict(dummy_input)
    
    print(f"\nTest Results:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probabilities shape: {probabilities.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions.tolist()}")