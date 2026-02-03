"""Multi-Layer Perceptron implementation."""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture.

    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output
        activation: Activation function to use
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate (0 = no dropout)

    Example:
        >>> model = MLP(input_dim=2, hidden_dims=[64, 64, 64], output_dim=1)
        >>> x = torch.randn(100, 2)
        >>> y = model(x)  # Shape: (100, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Activation function
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if activation.lower() not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_fn = activation_map[activation.lower()]

        # Build layers
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            layers.append(self.activation_fn())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer (no activation)
        layers.append(nn.Linear(dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)