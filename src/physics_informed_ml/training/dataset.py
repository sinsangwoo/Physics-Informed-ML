"""Dataset utilities for physics-informed learning."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from physics_informed_ml.solvers.pendulum import PendulumSimulator, PendulumConfig
from physics_informed_ml.core.config import SimulationConfig


class PhysicsDataset(Dataset):
    """Dataset for physics-informed learning.

    Stores both labeled data points and collocation points for physics loss.
    """

    def __init__(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        x_physics: torch.Tensor | None = None,
    ) -> None:
        """Initialize physics dataset.

        Args:
            x_data: Input data points for supervised learning
            y_data: Target values
            x_physics: Collocation points for physics loss (optional)
        """
        self.x_data = x_data
        self.y_data = y_data
        self.x_physics = x_physics if x_physics is not None else x_data

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single data point.

        Args:
            idx: Index

        Returns:
            Tuple of (input, target)
        """
        return self.x_data[idx], self.y_data[idx]

    def get_physics_points(self, n_points: int | None = None) -> torch.Tensor:
        """Get physics collocation points.

        Args:
            n_points: Number of points to sample (None = all points)

        Returns:
            Physics collocation points
        """
        if n_points is None or n_points >= len(self.x_physics):
            return self.x_physics

        # Random sampling
        indices = torch.randperm(len(self.x_physics))[:n_points]
        return self.x_physics[indices]


def create_pendulum_dataset(
    pendulum_config: PendulumConfig,
    sim_config: SimulationConfig | None = None,
    n_trajectories: int = 10,
    angle_range: Tuple[float, float] = (5.0, 45.0),
    n_physics_points: int = 1000,
) -> PhysicsDataset:
    """Create dataset for pendulum PINN training.

    Args:
        pendulum_config: Pendulum configuration
        sim_config: Simulation configuration
        n_trajectories: Number of pendulum trajectories to simulate
        angle_range: Range of initial angles in degrees (min, max)
        n_physics_points: Number of collocation points for physics loss

    Returns:
        PhysicsDataset with trajectory data and physics points
    """
    sim_config = sim_config or SimulationConfig()
    simulator = PendulumSimulator(pendulum_config, sim_config)

    # Generate training data from multiple trajectories
    x_data_list = []
    y_data_list = []

    angles_deg = np.linspace(angle_range[0], angle_range[1], n_trajectories)

    for angle_deg in angles_deg:
        theta0 = np.radians(angle_deg)
        times, angles, omegas, _ = simulator.simulate(theta0)

        # Sample subset of points from each trajectory
        sample_rate = max(1, len(times) // 100)  # Keep ~100 points per trajectory
        sampled_indices = np.arange(0, len(times), sample_rate)

        for idx in sampled_indices:
            x_data_list.append([times[idx]])
            y_data_list.append([angles[idx]])

    # Convert to tensors
    x_data = torch.tensor(x_data_list, dtype=torch.float32)
    y_data = torch.tensor(y_data_list, dtype=torch.float32)

    # Generate physics collocation points (uniform sampling in time)
    t_physics = torch.linspace(0, sim_config.t_max, n_physics_points).reshape(-1, 1)

    return PhysicsDataset(x_data, y_data, t_physics)