"""Neural network models for physics-informed learning."""

from physics_informed_ml.models.base import PhysicsInformedModel
from physics_informed_ml.models.pinn import PINN, PINNConfig
from physics_informed_ml.models.losses import PhysicsLoss, DataLoss, TotalLoss

__all__ = [
    "PhysicsInformedModel",
    "PINN",
    "PINNConfig",
    "PhysicsLoss",
    "DataLoss",
    "TotalLoss",
]