"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def random_seed() -> int:
    """Provide a random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed: int) -> None:
    """Set random seed for all tests."""
    np.random.seed(random_seed)


@pytest.fixture
def sample_simulation_params() -> dict:
    """Provide sample simulation parameters."""
    return {
        "length": 1.0,
        "initial_angle": 30.0,
        "gravity": 9.81,
        "dt": 0.01,
        "t_max": 10.0,
    }