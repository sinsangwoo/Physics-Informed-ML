"""Pendulum physics simulator and PINN implementation."""

from typing import Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field

from physics_informed_ml.core.config import SimulationConfig
from physics_informed_ml.models.pinn import PINN, PINNConfig
from physics_informed_ml.solvers.integrators import RK4Integrator, Integrator


class PendulumConfig(BaseModel):
    """Configuration for pendulum simulation.

    Attributes:
        length: Pendulum length (meters)
        mass: Bob mass (kg)
        gravity: Gravitational acceleration (m/s²)
        damping: Damping coefficient (optional)
    """

    length: float = Field(gt=0, description="Pendulum length")
    mass: float = Field(default=1.0, gt=0, description="Bob mass")
    gravity: float = Field(default=9.81, gt=0, description="Gravity")
    damping: float = Field(default=0.0, ge=0, description="Damping coefficient")

    class Config:
        """Pydantic config."""

        frozen = True


class PendulumSimulator:
    """High-fidelity pendulum physics simulator.

    Simulates pendulum dynamics using numerical integration of the equation:
    d²θ/dt² = -(g/L)sin(θ) - (c/m)dθ/dt

    where:
    - θ is the angle from vertical
    - g is gravitational acceleration
    - L is pendulum length
    - c is damping coefficient
    - m is bob mass
    """

    def __init__(
        self,
        config: PendulumConfig,
        sim_config: SimulationConfig | None = None,
        integrator: Integrator | None = None,
    ) -> None:
        """Initialize pendulum simulator.

        Args:
            config: Pendulum physical parameters
            sim_config: Simulation parameters
            integrator: Numerical integrator (default: RK4)
        """
        self.config = config
        self.sim_config = sim_config or SimulationConfig()
        self.integrator = integrator or RK4Integrator()

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for pendulum ODE.

        Args:
            state: [theta, omega] where omega = dθ/dt

        Returns:
            [dθ/dt, d²θ/dt²]
        """
        theta, omega = state
        g = self.config.gravity
        L = self.config.length
        c = self.config.damping
        m = self.config.mass

        # dθ/dt = ω
        dtheta = omega

        # d²θ/dt² = -(g/L)sin(θ) - (c/m)ω
        domega = -(g / L) * np.sin(theta) - (c / m) * omega

        return np.array([dtheta, domega])

    def simulate(
        self, theta0: float, omega0: float = 0.0, t_max: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pendulum motion.

        Args:
            theta0: Initial angle (radians)
            omega0: Initial angular velocity (rad/s)
            t_max: Maximum simulation time (uses sim_config.t_max if None)

        Returns:
            Tuple of (times, angles, angular_velocities, energies)
        """
        t_max = t_max or self.sim_config.t_max
        dt = self.sim_config.dt
        n_steps = int(t_max / dt)

        # Initialize arrays
        times = np.zeros(n_steps)
        angles = np.zeros(n_steps)
        omegas = np.zeros(n_steps)
        energies = np.zeros(n_steps)

        # Initial state
        state = np.array([theta0, omega0])
        times[0] = 0.0
        angles[0] = theta0
        omegas[0] = omega0
        energies[0] = self._compute_energy(state)

        # Time integration
        for i in range(1, n_steps):
            state = self.integrator.step(state, self._derivatives, dt)
            times[i] = i * dt
            angles[i] = state[0]
            omegas[i] = state[1]
            energies[i] = self._compute_energy(state)

        return times, angles, omegas, energies

    def _compute_energy(self, state: np.ndarray) -> float:
        """Compute total mechanical energy.

        Args:
            state: [theta, omega]

        Returns:
            Total energy (kinetic + potential)
        """
        theta, omega = state
        m = self.config.mass
        L = self.config.length
        g = self.config.gravity

        # Kinetic energy: (1/2) * m * L² * ω²
        kinetic = 0.5 * m * (L * omega) ** 2

        # Potential energy: m * g * L * (1 - cos(θ))
        # Reference: potential is zero at lowest point
        potential = m * g * L * (1 - np.cos(theta))

        return kinetic + potential

    def compute_period(self, theta0: float) -> float:
        """Compute period for given initial angle.

        Args:
            theta0: Initial angle (radians)

        Returns:
            Period (seconds)
        """
        times, angles, _, _ = self.simulate(theta0, t_max=20.0)

        # Find zero crossings with positive velocity
        zero_crossings = []
        for i in range(1, len(angles) - 1):
            if angles[i - 1] < 0 and angles[i] >= 0:
                # Linear interpolation for accurate crossing time
                t_cross = times[i - 1] + (times[i] - times[i - 1]) * (
                    -angles[i - 1] / (angles[i] - angles[i - 1])
                )
                zero_crossings.append(t_cross)

        if len(zero_crossings) >= 2:
            return 2 * (zero_crossings[1] - zero_crossings[0])
        else:
            return np.nan


class PendulumPINN(PINN):
    """Physics-Informed Neural Network for pendulum dynamics.

    Learns to predict pendulum motion while satisfying the equation of motion:
    d²θ/dt² + (g/L)sin(θ) = 0
    """

    def __init__(self, pinn_config: PINNConfig, pendulum_config: PendulumConfig) -> None:
        """Initialize pendulum PINN.

        Args:
            pinn_config: Neural network configuration
            pendulum_config: Pendulum physical parameters
        """
        # Create PDE residual function
        def pendulum_pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            """Compute pendulum PDE residual.

            Args:
                x: Time coordinates (batch_size, 1)
                u: Predicted angle (batch_size, 1)

            Returns:
                PDE residual (should be zero when satisfied)
            """
            # Compute first derivative: dθ/dt
            du_dt = torch.autograd.grad(
                u,
                x,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Compute second derivative: d²θ/dt²
            d2u_dt2 = torch.autograd.grad(
                du_dt,
                x,
                grad_outputs=torch.ones_like(du_dt),
                create_graph=True,
                retain_graph=True,
            )[0]

            # PDE: d²θ/dt² + (g/L)sin(θ) = 0
            g = pendulum_config.gravity
            L = pendulum_config.length
            residual = d2u_dt2 + (g / L) * torch.sin(u)

            return residual

        super().__init__(pinn_config, pde_residual_fn=pendulum_pde)
        self.pendulum_config = pendulum_config

    def get_config(self) -> dict:
        """Get model configuration including pendulum parameters.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config["pendulum_config"] = self.pendulum_config.model_dump()
        return config