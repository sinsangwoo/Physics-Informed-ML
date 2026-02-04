"""Configuration management for simulations."""

from pydantic import BaseModel, Field
from typing import Literal


class SimulationConfig(BaseModel):
    """Base configuration for physics simulations.

    Attributes:
        dt: Time step for numerical integration
        t_max: Maximum simulation time
        gravity: Gravitational acceleration (m/s²)
        integrator: Numerical integration method
        precision: Numerical precision for computations
    """

    dt: float = Field(default=0.001, gt=0, description="Time step for integration")
    t_max: float = Field(default=10.0, gt=0, description="Maximum simulation time")
    gravity: float = Field(default=9.81, gt=0, description="Gravitational acceleration")
    integrator: Literal["euler", "rk4", "verlet"] = Field(
        default="rk4", description="Numerical integration method"
    )
    precision: Literal["float32", "float64"] = Field(
        default="float64", description="Numerical precision"
    )

    class Config:
        """Pydantic config."""

        frozen = True  # Make config immutable