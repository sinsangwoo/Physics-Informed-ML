"""Tests for pendulum simulator."""

import pytest
import numpy as np
import torch

from physics_informed_ml.solvers.pendulum import (
    PendulumSimulator,
    PendulumConfig,
    PendulumPINN,
)
from physics_informed_ml.models.pinn import PINNConfig
from physics_informed_ml.core.config import SimulationConfig


class TestPendulumConfig:
    """Test suite for PendulumConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = PendulumConfig(length=1.0)
        assert config.length == 1.0
        assert config.mass == 1.0
        assert config.gravity == 9.81
        assert config.damping == 0.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PendulumConfig(length=2.0, mass=0.5, gravity=10.0, damping=0.1)
        assert config.length == 2.0
        assert config.mass == 0.5
        assert config.gravity == 10.0
        assert config.damping == 0.1

    def test_invalid_length(self) -> None:
        """Test that negative length raises error."""
        with pytest.raises(ValueError):
            PendulumConfig(length=-1.0)


class TestPendulumSimulator:
    """Test suite for PendulumSimulator."""

    def test_initialization(self) -> None:
        """Test simulator initialization."""
        config = PendulumConfig(length=1.0)
        sim = PendulumSimulator(config)

        assert sim.config == config
        assert sim.sim_config is not None

    def test_simulate_shape(self) -> None:
        """Test that simulation returns correct array shapes."""
        config = PendulumConfig(length=1.0)
        sim_config = SimulationConfig(dt=0.01, t_max=1.0)
        sim = PendulumSimulator(config, sim_config)

        times, angles, omegas, energies = sim.simulate(theta0=0.1)

        assert len(times) == len(angles) == len(omegas) == len(energies)
        assert len(times) > 0

    def test_small_angle_approximation(self) -> None:
        """Test small angle approximation for period."""
        config = PendulumConfig(length=1.0, gravity=9.81)
        sim = PendulumSimulator(config)

        # Small angle: period ≈ 2π√(L/g)
        expected_period = 2 * np.pi * np.sqrt(config.length / config.gravity)
        measured_period = sim.compute_period(theta0=0.1)  # ~5.7 degrees

        # Should be within 1% for small angles
        relative_error = abs(measured_period - expected_period) / expected_period
        assert relative_error < 0.01

    def test_energy_conservation(self) -> None:
        """Test energy conservation for undamped pendulum."""
        config = PendulumConfig(length=1.0, damping=0.0)  # No damping
        sim_config = SimulationConfig(dt=0.001, t_max=5.0)
        sim = PendulumSimulator(config, sim_config)

        times, angles, omegas, energies = sim.simulate(theta0=0.5)

        # Energy should be conserved (within numerical error)
        energy_variation = np.std(energies) / np.mean(energies)
        assert energy_variation < 0.01  # Less than 1% variation

    def test_damping_decreases_energy(self) -> None:
        """Test that damping decreases energy over time."""
        config = PendulumConfig(length=1.0, damping=0.1)  # With damping
        sim_config = SimulationConfig(dt=0.01, t_max=10.0)
        sim = PendulumSimulator(config, sim_config)

        times, angles, omegas, energies = sim.simulate(theta0=0.5)

        # Energy should decrease monotonically
        assert energies[-1] < energies[0]

    def test_different_initial_conditions(self) -> None:
        """Test simulation with different initial conditions."""
        config = PendulumConfig(length=1.0)
        sim = PendulumSimulator(config)

        # Different initial angles
        for theta0 in [0.1, 0.5, 1.0]:
            times, angles, omegas, energies = sim.simulate(theta0)
            assert len(times) > 0
            assert np.abs(angles[0] - theta0) < 1e-6

        # Non-zero initial velocity
        times, angles, omegas, energies = sim.simulate(theta0=0.0, omega0=1.0)
        assert len(times) > 0
        assert np.abs(omegas[0] - 1.0) < 1e-6


class TestPendulumPINN:
    """Test suite for PendulumPINN."""

    def test_initialization(self) -> None:
        """Test PendulumPINN initialization."""
        pinn_config = PINNConfig(input_dim=1, output_dim=1, hidden_dims=[32])
        pendulum_config = PendulumConfig(length=1.0)

        model = PendulumPINN(pinn_config, pendulum_config)

        assert model.config == pinn_config
        assert model.pendulum_config == pendulum_config
        assert model.pde_residual_fn is not None

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        pinn_config = PINNConfig(input_dim=1, output_dim=1, hidden_dims=[32])
        pendulum_config = PendulumConfig(length=1.0)
        model = PendulumPINN(pinn_config, pendulum_config)

        t = torch.linspace(0, 1, 10).reshape(-1, 1)
        theta = model(t)

        assert theta.shape == (10, 1)

    def test_physics_loss_computation(self) -> None:
        """Test that physics loss can be computed."""
        pinn_config = PINNConfig(input_dim=1, output_dim=1, hidden_dims=[32])
        pendulum_config = PendulumConfig(length=1.0)
        model = PendulumPINN(pinn_config, pendulum_config)

        t = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
        physics_loss = model.compute_physics_loss(t)

        assert physics_loss.item() >= 0
        assert physics_loss.requires_grad

    def test_get_config(self) -> None:
        """Test configuration retrieval."""
        pinn_config = PINNConfig(input_dim=1, output_dim=1)
        pendulum_config = PendulumConfig(length=1.0)
        model = PendulumPINN(pinn_config, pendulum_config)

        config = model.get_config()

        assert "pendulum_config" in config
        assert config["pendulum_config"]["length"] == 1.0