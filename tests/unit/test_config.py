"""Tests for configuration management."""

import pytest
from physics_informed_ml.core.config import SimulationConfig
from pydantic import ValidationError


class TestSimulationConfig:
    """Test suite for SimulationConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.dt == 0.001
        assert config.t_max == 10.0
        assert config.gravity == 9.81
        assert config.integrator == "rk4"
        assert config.precision == "float64"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SimulationConfig(
            dt=0.01,
            t_max=20.0,
            gravity=10.0,
            integrator="euler",
            precision="float32",
        )
        assert config.dt == 0.01
        assert config.t_max == 20.0
        assert config.gravity == 10.0
        assert config.integrator == "euler"
        assert config.precision == "float32"

    def test_invalid_dt(self) -> None:
        """Test that negative dt raises validation error."""
        with pytest.raises(ValidationError):
            SimulationConfig(dt=-0.001)

    def test_invalid_t_max(self) -> None:
        """Test that negative t_max raises validation error."""
        with pytest.raises(ValidationError):
            SimulationConfig(t_max=-10.0)

    def test_invalid_gravity(self) -> None:
        """Test that negative gravity raises validation error."""
        with pytest.raises(ValidationError):
            SimulationConfig(gravity=-9.81)

    def test_invalid_integrator(self) -> None:
        """Test that invalid integrator raises validation error."""
        with pytest.raises(ValidationError):
            SimulationConfig(integrator="invalid")  # type: ignore

    def test_invalid_precision(self) -> None:
        """Test that invalid precision raises validation error."""
        with pytest.raises(ValidationError):
            SimulationConfig(precision="float16")  # type: ignore

    def test_config_immutability(self) -> None:
        """Test that config is immutable."""
        config = SimulationConfig()
        with pytest.raises(ValidationError):
            config.dt = 0.01  # type: ignore