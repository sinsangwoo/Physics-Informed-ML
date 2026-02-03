"""Tests for pendulum simulator."""

import pytest
import numpy as np
import torch
from physics_informed_ml.solvers import (
    PendulumSimulator,
    PendulumConfig,
    simulate_pendulum,
    pendulum_pde_residual,
)


class TestPendulumSimulator:
    """Test suite for pendulum simulator."""

    def test_simulator_initialization(self) -> None:
        """Test simulator initialization."""
        config = PendulumConfig(length=1.0, gravity=9.81)
        sim = PendulumSimulator(config)

        assert sim.config.length == 1.0
        assert sim.config.gravity == 9.81

    def test_simulation_output_shape(self) -> None:
        """Test that simulation returns correct shapes."""
        config = PendulumConfig(length=1.0, dt=0.01, t_max=10.0)
        sim = PendulumSimulator(config)

        times, angles, velocities = sim.simulate(initial_angle=30.0)

        expected_steps = int(config.t_max / config.dt)
        assert len(times) == expected_steps
        assert len(angles) == expected_steps
        assert len(velocities) == expected_steps

    def test_small_angle_period(self) -> None:
        """Test period calculation for small angles."""
        L = 1.0
        g = 9.81
        config = PendulumConfig(length=L, gravity=g, dt=0.001, t_max=20.0)
        sim = PendulumSimulator(config)

        times, angles, _ = sim.simulate(initial_angle=5.0)  # Small angle
        period = sim.compute_period(angles, times)

        # Theoretical period for small angles: T = 2π√(L/g)
        theoretical_period = 2 * np.pi * np.sqrt(L / g)

        # Should be within 1% for small angles
        assert abs(period - theoretical_period) / theoretical_period < 0.01

    def test_energy_conservation_no_damping(self) -> None:
        """Test energy conservation for undamped pendulum."""
        config = PendulumConfig(
            length=1.0, gravity=9.81, damping=0.0, dt=0.001, t_max=10.0
        )
        sim = PendulumSimulator(config)

        times, angles, velocities = sim.simulate(initial_angle=30.0)

        # Compute total energy at each timestep
        # E = (1/2)mL²ω² + mgL(1-cos(θ))
        # Normalized: E = (1/2)ω² + g/L(1-cos(θ))
        kinetic = 0.5 * velocities**2
        potential = (config.gravity / config.length) * (1 - np.cos(angles))
        total_energy = kinetic + potential

        # Energy should be conserved (within numerical error)
        energy_variation = np.std(total_energy) / np.mean(total_energy)
        assert energy_variation < 0.01  # Less than 1% variation

    def test_dataset_generation(self) -> None:
        """Test dataset generation."""
        config = PendulumConfig(length=1.0, dt=0.01, t_max=10.0)
        sim = PendulumSimulator(config)

        X, y = sim.generate_dataset(n_lengths=5, n_angles=5)

        assert X.shape[1] == 2  # (length, angle)
        assert len(y) == len(X)
        assert y.shape == (len(X),)
        assert np.all(y > 0)  # All periods should be positive

    def test_rk4_vs_euler(self) -> None:
        """Test that RK4 is more accurate than Euler."""
        L = 1.0
        theta0 = 30.0

        # RK4 simulation
        config_rk4 = PendulumConfig(
            length=L, dt=0.01, t_max=10.0, integrator="rk4"
        )
        sim_rk4 = PendulumSimulator(config_rk4)
        times_rk4, angles_rk4, _ = sim_rk4.simulate(theta0)
        period_rk4 = sim_rk4.compute_period(angles_rk4, times_rk4)

        # Euler simulation with same dt
        config_euler = PendulumConfig(
            length=L, dt=0.01, t_max=10.0, integrator="euler"
        )
        sim_euler = PendulumSimulator(config_euler)
        times_euler, angles_euler, _ = sim_euler.simulate(theta0)
        period_euler = sim_euler.compute_period(angles_euler, times_euler)

        # Theoretical period
        theoretical = 2 * np.pi * np.sqrt(L / 9.81)

        # RK4 should be more accurate
        error_rk4 = abs(period_rk4 - theoretical) / theoretical
        error_euler = abs(period_euler - theoretical) / theoretical

        assert error_rk4 < error_euler

    def test_convenience_function(self) -> None:
        """Test convenience function."""
        period = simulate_pendulum(length=1.0, initial_angle=30.0)

        assert isinstance(period, float)
        assert period > 0
        assert not np.isnan(period)


class TestPendulumPDEResidual:
    """Test suite for pendulum PDE residual."""

    def test_pde_residual_shape(self) -> None:
        """Test PDE residual computation shape."""
        from physics_informed_ml.models import PINN, PINNConfig

        config = PINNConfig(input_dim=2, output_dim=1)
        model = PINN(config)

        x = torch.randn(10, 2, requires_grad=True)
        residual = pendulum_pde_residual(model, x)

        assert residual.shape == (10, 1)

    def test_pde_residual_gradient_flow(self) -> None:
        """Test that gradients flow through PDE residual."""
        from physics_informed_ml.models import PINN, PINNConfig

        config = PINNConfig(input_dim=2, output_dim=1)
        model = PINN(config)

        x = torch.randn(5, 2, requires_grad=True)
        residual = pendulum_pde_residual(model, x)
        loss = torch.mean(residual**2)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None