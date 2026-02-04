"""Base class for physics-informed models."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class PhysicsInformedModel(nn.Module, ABC):
    """Abstract base class for physics-informed neural networks.

    All physics-informed models should inherit from this class and implement
    the required abstract methods for computing physics residuals and losses.
    """

    def __init__(self) -> None:
        """Initialize the physics-informed model."""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        pass

    @abstractmethod
    def compute_physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the physics-based loss (PDE residual).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Physics loss scalar tensor
        """
        pass

    @abstractmethod
    def compute_data_loss(
        self, x: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the data fitting loss.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            y_true: Target tensor of shape (batch_size, output_dim)

        Returns:
            Data loss scalar tensor
        """
        pass

    def compute_total_loss(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor | None = None,
        lambda_physics: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss combining physics and data losses.

        Args:
            x: Input tensor
            y_true: Target tensor (optional)
            lambda_physics: Weight for physics loss

        Returns:
            Dictionary containing all loss components and total loss
        """
        physics_loss = self.compute_physics_loss(x)

        losses = {
            "physics_loss": physics_loss,
            "total_loss": lambda_physics * physics_loss,
        }

        if y_true is not None:
            data_loss = self.compute_data_loss(x, y_true)
            losses["data_loss"] = data_loss
            losses["total_loss"] = losses["total_loss"] + data_loss

        return losses

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Dictionary containing model configuration
        """
        return {
            "model_type": self.__class__.__name__,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }