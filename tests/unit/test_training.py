"""Tests for training utilities."""

import pytest
import torch
from pathlib import Path

from physics_informed_ml.training.trainer import PINNTrainer, TrainingConfig
from physics_informed_ml.training.dataset import PhysicsDataset, create_pendulum_dataset
from physics_informed_ml.models.pinn import PINN, PINNConfig
from physics_informed_ml.solvers.pendulum import PendulumConfig


class TestPhysicsDataset:
    """Test suite for PhysicsDataset."""

    def test_initialization(self) -> None:
        """Test dataset initialization."""
        x_data = torch.randn(100, 2)
        y_data = torch.randn(100, 1)
        dataset = PhysicsDataset(x_data, y_data)

        assert len(dataset) == 100
        assert dataset.x_data.shape == (100, 2)
        assert dataset.y_data.shape == (100, 1)

    def test_getitem(self) -> None:
        """Test dataset indexing."""
        x_data = torch.randn(100, 2)
        y_data = torch.randn(100, 1)
        dataset = PhysicsDataset(x_data, y_data)

        x, y = dataset[0]
        assert x.shape == (2,)
        assert y.shape == (1,)

    def test_physics_points(self) -> None:
        """Test physics points retrieval."""
        x_data = torch.randn(100, 2)
        y_data = torch.randn(100, 1)
        x_physics = torch.randn(200, 2)
        dataset = PhysicsDataset(x_data, y_data, x_physics)

        # Get all physics points
        points = dataset.get_physics_points()
        assert points.shape == (200, 2)

        # Get subset
        points = dataset.get_physics_points(50)
        assert points.shape == (50, 2)


class TestCreatePendulumDataset:
    """Test suite for create_pendulum_dataset."""

    def test_dataset_creation(self) -> None:
        """Test pendulum dataset creation."""
        config = PendulumConfig(length=1.0)
        dataset = create_pendulum_dataset(
            config, n_trajectories=5, n_physics_points=100
        )

        assert isinstance(dataset, PhysicsDataset)
        assert len(dataset) > 0
        assert dataset.x_physics.shape[0] == 100


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = TrainingConfig()
        assert config.epochs == 1000
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.lambda_physics == 1.0
        assert config.device == "cpu"


class TestPINNTrainer:
    """Test suite for PINNTrainer."""

    def test_initialization(self) -> None:
        """Test trainer initialization."""

        def dummy_pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(u)

        pinn_config = PINNConfig(input_dim=1, output_dim=1, hidden_dims=[16])
        model = PINN(pinn_config, pde_residual_fn=dummy_pde)
        train_config = TrainingConfig(epochs=10, batch_size=8)

        trainer = PINNTrainer(model, train_config)

        assert trainer.model == model
        assert trainer.config == train_config
        assert trainer.optimizer is not None

    def test_training_loop(self) -> None:
        """Test that training loop runs without errors."""

        def dummy_pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(u)

        # Create small model and dataset
        pinn_config = PINNConfig(input_dim=1, output_dim=1, hidden_dims=[8])
        model = PINN(pinn_config, pde_residual_fn=dummy_pde)

        x_data = torch.randn(50, 1)
        y_data = torch.randn(50, 1)
        dataset = PhysicsDataset(x_data, y_data)

        # Train for just a few epochs
        train_config = TrainingConfig(
            epochs=5, batch_size=10, log_interval=5
        )
        trainer = PINNTrainer(model, train_config)

        history = trainer.train(dataset)

        assert "total_loss" in history
        assert "data_loss" in history
        assert "physics_loss" in history
        assert len(history["total_loss"]) == 5

    def test_checkpoint_save_load(self, tmp_path: Path) -> None:
        """Test checkpoint saving and loading."""

        def dummy_pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(u)

        pinn_config = PINNConfig(input_dim=1, output_dim=1, hidden_dims=[8])
        model = PINN(pinn_config, pde_residual_fn=dummy_pde)

        train_config = TrainingConfig(
            epochs=2, checkpoint_dir=str(tmp_path)
        )
        trainer = PINNTrainer(model, train_config)

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(epoch=1, filename="test.pth")
        assert checkpoint_path.exists()

        # Load checkpoint
        epoch = trainer.load_checkpoint(checkpoint_path)
        assert epoch == 1