"""Neural network models for physics-informed learning."""

from physics_informed_ml.models.base import PhysicsInformedModel
from physics_informed_ml.models.pinn import PINN, PINNConfig
from physics_informed_ml.models.losses import PhysicsLoss, DataLoss, TotalLoss

# Neural Operators
from physics_informed_ml.models.operators.fno import (
    FNO1d,
    FNO2d,
    FNO3d,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
)

__all__ = [
    # Base models
    "PhysicsInformedModel",
    "PINN",
    "PINNConfig",
    # Losses
    "PhysicsLoss",
    "DataLoss",
    "TotalLoss",
    # Neural Operators
    "FNO1d",
    "FNO2d",
    "FNO3d",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConv3d",
]
