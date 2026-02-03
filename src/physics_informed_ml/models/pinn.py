"""Physics-Informed Neural Network (PINN) implementation."""

import torch
import torch.nn as nn
from typing import Dict, Callable, Optional, Tuple
from pydantic import BaseModel, Field
from physics_informed_ml.models.mlp import MLP


class PINNConfig(BaseModel):
    """Configuration for Physics-Informed Neural Network.

    Attributes:
        input_dim: Dimension of input (e.g., 2 for (x, t))
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output (e.g., 1 for u(x,t))
        activation: Activation function name
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate
        lambda_data: Weight for data loss
        lambda_physics: Weight for physics loss
        lambda_ic: Weight for initial condition loss
        lambda_bc: Weight for boundary condition loss
    """

    input_dim: int = Field(gt=0)
    hidden_dims: list[int] = Field(default=[64, 64, 64])
    output_dim: int = Field(default=1, gt=0)
    activation: str = Field(default="tanh")
    use_batch_norm: bool = Field(default=False)
    dropout_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Loss weights
    lambda_data: float = Field(default=1.0, ge=0.0)
    lambda_physics: float = Field(default=1.0, ge=0.0)
    lambda_ic: float = Field(default=1.0, ge=0.0)
    lambda_bc: float = Field(default=1.0, ge=0.0)

    class Config:
        """Pydantic config."""

        frozen = True


class PINN(nn.Module):
    """Physics-Informed Neural Network.

    A neural network that embeds physics laws (PDEs) directly into the loss function.
    This enables learning solutions to PDEs with limited or no training data by
    enforcing physical constraints during training.

    Args:
        config: PINN configuration
        pde_residual: Function computing PDE residual given (model, x)

    Example:
        >>> # Define PDE residual for simple pendulum
        >>> def pendulum_residual(model, x):
        ...     # x = [theta, t], output = theta(t)
        ...     theta = model(x)
        ...     # Compute d²theta/dt² using automatic differentiation
        ...     theta_t = torch.autograd.grad(theta, x, ...)[0][:, 1:2]
        ...     theta_tt = torch.autograd.grad(theta_t, x, ...)[0][:, 1:2]
        ...     # Residual: d²theta/dt² + (g/L)*sin(theta) = 0
        ...     return theta_tt + (9.81/1.0) * torch.sin(theta)
        >>>
        >>> config = PINNConfig(input_dim=2, output_dim=1)
        >>> model = PINN(config, pde_residual=pendulum_residual)
    """

    def __init__(
        self,
        config: PINNConfig,
        pde_residual: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.pde_residual = pde_residual

        # Build neural network backbone
        self.network = MLP(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            activation=config.activation,
            use_batch_norm=config.use_batch_norm,
            dropout_rate=config.dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def compute_data_loss(
        self, x_data: torch.Tensor, y_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute data fitting loss.

        Args:
            x_data: Input data points
            y_data: Target values

        Returns:
            Mean squared error between predictions and targets
        """
        y_pred = self.forward(x_data)
        return torch.mean((y_pred - y_data) ** 2)

    def compute_physics_loss(self, x_physics: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss using PDE residual.

        Args:
            x_physics: Collocation points where PDE should be satisfied

        Returns:
            Mean squared PDE residual

        Raises:
            ValueError: If pde_residual function is not provided
        """
        if self.pde_residual is None:
            raise ValueError("PDE residual function not provided")

        # Ensure gradients can be computed
        x_physics = x_physics.requires_grad_(True)

        # Compute PDE residual
        residual = self.pde_residual(self, x_physics)

        return torch.mean(residual**2)

    def compute_ic_loss(
        self, x_ic: torch.Tensor, y_ic: torch.Tensor
    ) -> torch.Tensor:
        """Compute initial condition loss.

        Args:
            x_ic: Initial condition points
            y_ic: Initial condition values

        Returns:
            Mean squared error for initial conditions
        """
        y_pred = self.forward(x_ic)
        return torch.mean((y_pred - y_ic) ** 2)

    def compute_bc_loss(
        self, x_bc: torch.Tensor, y_bc: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary condition loss.

        Args:
            x_bc: Boundary condition points
            y_bc: Boundary condition values

        Returns:
            Mean squared error for boundary conditions
        """
        y_pred = self.forward(x_bc)
        return torch.mean((y_pred - y_bc) ** 2)

    def compute_total_loss(
        self,
        x_data: Optional[torch.Tensor] = None,
        y_data: Optional[torch.Tensor] = None,
        x_physics: Optional[torch.Tensor] = None,
        x_ic: Optional[torch.Tensor] = None,
        y_ic: Optional[torch.Tensor] = None,
        x_bc: Optional[torch.Tensor] = None,
        y_bc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total weighted loss.

        Args:
            x_data: Data input points (optional)
            y_data: Data target values (optional)
            x_physics: Physics collocation points (optional)
            x_ic: Initial condition points (optional)
            y_ic: Initial condition values (optional)
            x_bc: Boundary condition points (optional)
            y_bc: Boundary condition values (optional)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)

        # Data loss
        if x_data is not None and y_data is not None:
            loss_data = self.compute_data_loss(x_data, y_data)
            losses["data"] = loss_data.item()
            total_loss = total_loss + self.config.lambda_data * loss_data

        # Physics loss
        if x_physics is not None:
            loss_physics = self.compute_physics_loss(x_physics)
            losses["physics"] = loss_physics.item()
            total_loss = total_loss + self.config.lambda_physics * loss_physics

        # Initial condition loss
        if x_ic is not None and y_ic is not None:
            loss_ic = self.compute_ic_loss(x_ic, y_ic)
            losses["ic"] = loss_ic.item()
            total_loss = total_loss + self.config.lambda_ic * loss_ic

        # Boundary condition loss
        if x_bc is not None and y_bc is not None:
            loss_bc = self.compute_bc_loss(x_bc, y_bc)
            losses["bc"] = loss_bc.item()
            total_loss = total_loss + self.config.lambda_bc * loss_bc

        losses["total"] = total_loss.item()

        return total_loss, losses