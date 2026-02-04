"""Loss functions for physics-informed learning."""

import torch
import torch.nn as nn
from typing import Callable


class PhysicsLoss(nn.Module):
    """Physics-based loss using PDE residuals.

    This loss enforces physical laws by penalizing violations of PDEs.
    The residual is computed using automatic differentiation.
    """

    def __init__(
        self,
        pde_residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        weight: float = 1.0,
    ) -> None:
        """Initialize physics loss.

        Args:
            pde_residual_fn: Function computing PDE residual from (x, u)
            weight: Weight for this loss component
        """
        super().__init__()
        self.pde_residual_fn = pde_residual_fn
        self.weight = weight

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute physics loss.

        Args:
            x: Input coordinates (requires_grad=True for derivatives)
            u: Network output

        Returns:
            Weighted MSE of PDE residual
        """
        residual = self.pde_residual_fn(x, u)
        return self.weight * torch.mean(residual**2)


class DataLoss(nn.Module):
    """Data fitting loss (supervised learning component)."""

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize data loss.

        Args:
            weight: Weight for this loss component
        """
        super().__init__()
        self.weight = weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MSE data loss.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Weighted MSE
        """
        return self.weight * torch.mean((y_pred - y_true) ** 2)


class BoundaryLoss(nn.Module):
    """Loss for enforcing boundary conditions."""

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize boundary loss.

        Args:
            weight: Weight for this loss component
        """
        super().__init__()
        self.weight = weight

    def forward(
        self, x_boundary: torch.Tensor, u_boundary: torch.Tensor, u_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary condition loss.

        Args:
            x_boundary: Boundary points
            u_boundary: Predicted values at boundary
            u_true: True boundary values

        Returns:
            Weighted MSE at boundary
        """
        return self.weight * torch.mean((u_boundary - u_true) ** 2)


class InitialConditionLoss(nn.Module):
    """Loss for enforcing initial conditions (time-dependent problems)."""

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize initial condition loss.

        Args:
            weight: Weight for this loss component
        """
        super().__init__()
        self.weight = weight

    def forward(
        self, x_initial: torch.Tensor, u_initial: torch.Tensor, u_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute initial condition loss.

        Args:
            x_initial: Initial time points
            u_initial: Predicted values at t=0
            u_true: True initial values

        Returns:
            Weighted MSE at initial time
        """
        return self.weight * torch.mean((u_initial - u_true) ** 2)


class TotalLoss(nn.Module):
    """Combined loss for physics-informed learning.

    Total loss = data_loss + λ_physics * physics_loss + λ_boundary * boundary_loss
    """

    def __init__(
        self,
        lambda_physics: float = 1.0,
        lambda_boundary: float = 1.0,
        lambda_initial: float = 1.0,
    ) -> None:
        """Initialize total loss.

        Args:
            lambda_physics: Weight for physics loss
            lambda_boundary: Weight for boundary loss
            lambda_initial: Weight for initial condition loss
        """
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_boundary = lambda_boundary
        self.lambda_initial = lambda_initial

    def forward(
        self,
        data_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        boundary_loss: torch.Tensor | None = None,
        initial_loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute total weighted loss.

        Args:
            data_loss: Data fitting loss
            physics_loss: Physics (PDE) loss
            boundary_loss: Boundary condition loss (optional)
            initial_loss: Initial condition loss (optional)

        Returns:
            Total weighted loss
        """
        total = data_loss + self.lambda_physics * physics_loss

        if boundary_loss is not None:
            total = total + self.lambda_boundary * boundary_loss

        if initial_loss is not None:
            total = total + self.lambda_initial * initial_loss

        return total