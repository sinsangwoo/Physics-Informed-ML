"""Physics-Informed Neural Network (PINN) implementation."""

from typing import Callable, List

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from physics_informed_ml.models.base import PhysicsInformedModel


class PINNConfig(BaseModel):
    """Configuration for PINN model.

    Attributes:
        input_dim: Dimensionality of input (e.g., time + space dimensions)
        output_dim: Dimensionality of output (e.g., state variables)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name ('tanh', 'relu', 'gelu', 'silu')
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate (0.0 means no dropout)
    """

    input_dim: int = Field(gt=0, description="Input dimensionality")
    output_dim: int = Field(gt=0, description="Output dimensionality")
    hidden_dims: List[int] = Field(
        default=[64, 64, 64], description="Hidden layer dimensions"
    )
    activation: str = Field(
        default="tanh", description="Activation function"
    )
    use_batch_norm: bool = Field(
        default=False, description="Use batch normalization"
    )
    dropout_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Dropout rate"
    )

    class Config:
        """Pydantic config."""

        frozen = True


class PINN(PhysicsInformedModel):
    """Physics-Informed Neural Network.

    A fully-connected neural network that embeds physical laws (PDEs)
    directly into the loss function through automatic differentiation.

    Example:
        >>> config = PINNConfig(input_dim=2, output_dim=1)
        >>> model = PINN(config, pde_residual_fn=my_pde)
        >>> x = torch.randn(100, 2, requires_grad=True)
        >>> y = model(x)
        >>> losses = model.compute_total_loss(x)
    """

    def __init__(
        self,
        config: PINNConfig,
        pde_residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize PINN.

        Args:
            config: PINN configuration
            pde_residual_fn: Function that computes PDE residual given (x, u)
                           Returns residual tensor that should be zero when PDE is satisfied
        """
        super().__init__()
        self.config = config
        self.pde_residual_fn = pde_residual_fn

        # Build network
        layers: List[nn.Module] = []
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add activation except for output layer
            if i < len(dims) - 2:
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

                layers.append(self._get_activation(config.activation))

                if config.dropout_rate > 0:
                    layers.append(nn.Dropout(config.dropout_rate))

        self.network = nn.Sequential(*layers)

        # Initialize weights with Xavier initialization
        self.apply(self._init_weights)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.

        Args:
            name: Activation function name

        Returns:
            Activation module

        Raises:
            ValueError: If activation name is not recognized
        """
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
        }
        if name not in activations:
            raise ValueError(
                f"Unknown activation: {name}. Choose from {list(activations.keys())}"
            )
        return activations[name]

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def compute_physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute physics-based loss using automatic differentiation.

        Args:
            x: Input tensor with requires_grad=True

        Returns:
            Physics loss (MSE of PDE residual)

        Raises:
            RuntimeError: If pde_residual_fn is not provided
        """
        if self.pde_residual_fn is None:
            raise RuntimeError(
                "PDE residual function not provided. "
                "Set pde_residual_fn in constructor or override compute_physics_loss."
            )

        # Ensure gradient tracking
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # Forward pass
        u = self.forward(x)

        # Compute PDE residual
        residual = self.pde_residual_fn(x, u)

        # Physics loss is MSE of residual (should be zero)
        return torch.mean(residual**2)

    def compute_data_loss(
        self, x: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute data fitting loss.

        Args:
            x: Input tensor
            y_true: Target tensor

        Returns:
            MSE data loss
        """
        y_pred = self.forward(x)
        return torch.mean((y_pred - y_true) ** 2)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions (no gradient computation).

        Args:
            x: Input tensor

        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_config(self) -> dict:
        """Get model configuration.

        Returns:
            Configuration dictionary
        """
        base_config = super().get_config()
        base_config.update({
            "pinn_config": self.config.model_dump(),
            "has_pde_residual": self.pde_residual_fn is not None,
        })
        return base_config