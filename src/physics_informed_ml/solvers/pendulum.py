"""Pendulum physics simulator with multiple integration schemes."""

import numpy as np
import torch
from typing import Tuple, Optional
from pydantic import BaseModel, Field
from numpy.typing import NDArray


class PendulumConfig(BaseModel):
    """Configuration for pendulum simulation.

    Attributes:
        length: Pendulum length in meters
        gravity: Gravitational acceleration in m/s²
        damping: Damping coefficient (0 = no damping)
        dt: Time step for numerical integration
        t_max: Maximum simulation time
        integrator: Integration method ('rk4', 'euler', 'verlet')
    """

    length: float = Field(default=1.0, gt=0, description="Pendulum length (m)")
    gravity: float = Field(default=9.81, gt=0, description="Gravity (m/s²)")
    damping: float = Field(default=0.0, ge=0, description="Damping coefficient")
    dt: float = Field(default=0.01, gt=0, description="Time step (s)")
    t_max: float = Field(default=10.0, gt=0, description="Max simulation time (s)")
    integrator: str = Field(default="rk4", description="Integration method")

    class Config:
        """Pydantic config."""

        frozen = True


class PendulumSimulator:
    """High-fidelity pendulum simulator using various integration schemes.

    Simulates the equation of motion:
        d²θ/dt² = -(g/L)sin(θ) - c(dθ/dt)

    where:
        θ = angle from vertical
        L = pendulum length
        g = gravitational acceleration
        c = damping coefficient

    Example:
        >>> config = PendulumConfig(length=1.0, gravity=9.81)
        >>> sim = PendulumSimulator(config)
        >>> times, angles, velocities = sim.simulate(initial_angle=30.0)
        >>> print(f"Period: {sim.compute_period(angles, times):.3f}s")
    """

    def __init__(self, config: PendulumConfig) -> None:
        self.config = config

    def _derivatives(
        self, theta: float, omega: float
    ) -> Tuple[float, float]:
        """Compute derivatives for pendulum equation of motion.

        Args:
            theta: Current angle (radians)
            omega: Current angular velocity (rad/s)

        Returns:
            (dθ/dt, dω/dt)
        """
        g, L, c = self.config.gravity, self.config.length, self.config.damping

        dtheta_dt = omega
        domega_dt = -(g / L) * np.sin(theta) - c * omega

        return dtheta_dt, domega_dt

    def _step_euler(
        self, theta: float, omega: float, dt: float
    ) -> Tuple[float, float]:
        """Single Euler integration step."""
        dtheta, domega = self._derivatives(theta, omega)
        return theta + dt * dtheta, omega + dt * domega

    def _step_rk4(
        self, theta: float, omega: float, dt: float
    ) -> Tuple[float, float]:
        """Single Runge-Kutta 4th order integration step."""
        # k1
        k1_theta, k1_omega = self._derivatives(theta, omega)

        # k2
        k2_theta, k2_omega = self._derivatives(
            theta + 0.5 * dt * k1_theta, omega + 0.5 * dt * k1_omega
        )

        # k3
        k3_theta, k3_omega = self._derivatives(
            theta + 0.5 * dt * k2_theta, omega + 0.5 * dt * k2_omega
        )

        # k4
        k4_theta, k4_omega = self._derivatives(
            theta + dt * k3_theta, omega + dt * k3_omega
        )

        # Weighted average
        theta_new = theta + (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
        omega_new = omega + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

        return theta_new, omega_new

    def simulate(
        self, initial_angle: float, initial_velocity: float = 0.0
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Simulate pendulum motion.

        Args:
            initial_angle: Initial angle in degrees
            initial_velocity: Initial angular velocity in deg/s

        Returns:
            Tuple of (times, angles, velocities) as numpy arrays
        """
        # Convert to radians
        theta = np.radians(initial_angle)
        omega = np.radians(initial_velocity)

        # Choose integration method
        if self.config.integrator == "rk4":
            step_fn = self._step_rk4
        elif self.config.integrator == "euler":
            step_fn = self._step_euler
        else:
            raise ValueError(f"Unknown integrator: {self.config.integrator}")

        # Simulate
        n_steps = int(self.config.t_max / self.config.dt)
        times = np.zeros(n_steps)
        angles = np.zeros(n_steps)
        velocities = np.zeros(n_steps)

        for i in range(n_steps):
            times[i] = i * self.config.dt
            angles[i] = theta
            velocities[i] = omega

            theta, omega = step_fn(theta, omega, self.config.dt)

        return times, angles, velocities

    def compute_period(
        self, angles: NDArray[np.float64], times: NDArray[np.float64]
    ) -> float:
        """Compute period from simulation data using zero-crossing detection.

        Args:
            angles: Array of angles
            times: Array of times

        Returns:
            Estimated period in seconds
        """
        # Find zero crossings
        crossings = []
        for i in range(1, len(angles)):
            if angles[i - 1] * angles[i] < 0 and angles[i] > angles[i - 1]:
                # Linear interpolation for exact crossing time
                t_cross = times[i - 1] + (times[i] - times[i - 1]) * (
                    -angles[i - 1] / (angles[i] - angles[i - 1])
                )
                crossings.append(t_cross)

        if len(crossings) < 2:
            return np.nan

        # Period is 2 * (time between consecutive crossings)
        return 2.0 * (crossings[1] - crossings[0])

    def generate_dataset(
        self,
        n_lengths: int = 30,
        n_angles: int = 10,
        length_range: Tuple[float, float] = (0.1, 2.0),
        angle_range: Tuple[float, float] = (5.0, 60.0),
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate dataset of (length, angle) -> period mappings.

        Args:
            n_lengths: Number of length values to sample
            n_angles: Number of angle values to sample
            length_range: (min, max) length in meters
            angle_range: (min, max) angle in degrees

        Returns:
            Tuple of (X, y) where X is (n_samples, 2) and y is (n_samples,)
        """
        lengths = np.linspace(*length_range, n_lengths)
        angles = np.linspace(*angle_range, n_angles)

        X = []
        y = []

        for L in lengths:
            for theta in angles:
                # Create temporary config with this length
                config = PendulumConfig(
                    length=L,
                    gravity=self.config.gravity,
                    damping=self.config.damping,
                    dt=self.config.dt,
                    t_max=self.config.t_max,
                    integrator=self.config.integrator,
                )
                temp_sim = PendulumSimulator(config)

                # Simulate and compute period
                times, angles_sim, _ = temp_sim.simulate(theta)
                period = temp_sim.compute_period(angles_sim, times)

                if not np.isnan(period):
                    X.append([L, theta])
                    y.append(period)

        return np.array(X), np.array(y)


def simulate_pendulum(
    length: float,
    initial_angle: float,
    gravity: float = 9.81,
    dt: float = 0.01,
    t_max: float = 10.0,
    integrator: str = "rk4",
) -> float:
    """Convenience function to simulate pendulum and return period.

    Args:
        length: Pendulum length in meters
        initial_angle: Initial angle in degrees
        gravity: Gravitational acceleration
        dt: Time step
        t_max: Max simulation time
        integrator: Integration method

    Returns:
        Period in seconds
    """
    config = PendulumConfig(
        length=length,
        gravity=gravity,
        dt=dt,
        t_max=t_max,
        integrator=integrator,
    )
    sim = PendulumSimulator(config)
    times, angles, _ = sim.simulate(initial_angle)
    return sim.compute_period(angles, times)


def pendulum_pde_residual(
    model: torch.nn.Module, x: torch.Tensor, length: float = 1.0, gravity: float = 9.81
) -> torch.Tensor:
    """Compute PDE residual for pendulum equation.

    The pendulum equation is:
        d²θ/dt² + (g/L)sin(θ) = 0

    Args:
        model: Neural network model
        x: Input tensor (batch_size, 2) where x[:, 0] is initial angle,
           x[:, 1] is time
        length: Pendulum length
        gravity: Gravitational acceleration

    Returns:
        PDE residual tensor
    """
    x = x.requires_grad_(True)
    theta = model(x)

    # First derivative: dθ/dt
    theta_t = torch.autograd.grad(
        theta,
        x,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]

    # Second derivative: d²θ/dt²
    theta_tt = torch.autograd.grad(
        theta_t,
        x,
        grad_outputs=torch.ones_like(theta_t),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]

    # PDE residual: d²θ/dt² + (g/L)sin(θ)
    residual = theta_tt + (gravity / length) * torch.sin(theta)

    return residual