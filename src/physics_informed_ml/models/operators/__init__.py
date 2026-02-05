"""Neural Operator models for physics-informed machine learning.

This module implements various neural operator architectures that learn
mappings between function spaces, enabling resolution-invariant learning
of partial differential equations.
"""

from physics_informed_ml.models.operators.fno import (
    FNO1d,
    FNO2d,
    FNO3d,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
)

__all__ = [
    "FNO1d",
    "FNO2d",
    "FNO3d",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConv3d",
]
