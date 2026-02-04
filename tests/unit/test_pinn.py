"""Tests for PINN models."""

import pytest
import torch
import numpy as np

from physics_informed_ml.models.pinn import PINN, PINNConfig
from physics_informed_ml.models.base import PhysicsInformedModel


class TestPINNConfig:
    """Test suite for PINNConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PINNConfig(input_dim=2, output_dim=1)
        assert config.input_dim == 2
        assert config.output_dim == 1
        assert config.hidden_dims == [64, 64, 64]
        assert config.activation == "tanh"
        assert config.use_batch_norm is False
        assert config.dropout_rate == 0.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PINNConfig(
            input_dim=3,
            output_dim=2,
            hidden_dims=[128, 128],
            activation="relu",
            use_batch_norm=True,
            dropout_rate=0.1,
        )
        assert config.input_dim == 3
        assert config.output_dim == 2
        assert config.hidden_dims == [128, 128]
        assert config.activation == "relu"
        assert config.use_batch_norm is True
        assert config.dropout_rate == 0.1

    def test_invalid_dimensions(self) -> None:
        """Test that invalid dimensions raise validation errors."""
        with pytest.raises(ValueError):
            PINNConfig(input_dim=0, output_dim=1)

        with pytest.raises(ValueError):
            PINNConfig(input_dim=1, output_dim=-1)


class TestPINN:
    """Test suite for PINN model."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        config = PINNConfig(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        model = PINN(config)

        assert isinstance(model, PhysicsInformedModel)
        assert model.config == config

    def test_forward_pass(self) -> None:
        """Test forward pass produces correct output shape."""
        config = PINNConfig(input_dim=2, output_dim=1, hidden_dims=[32])
        model = PINN(config)

        x = torch.randn(10, 2)
        y = model(x)

        assert y.shape == (10, 1)

    def test_different_activations(self) -> None:
        """Test different activation functions."""
        activations = ["tanh", "relu", "gelu", "silu", "elu"]

        for act in activations:
            config = PINNConfig(
                input_dim=2, output_dim=1, hidden_dims=[16], activation=act
            )
            model = PINN(config)
            x = torch.randn(5, 2)
            y = model(x)
            assert y.shape == (5, 1)

    def test_batch_norm(self) -> None:
        """Test batch normalization integration."""
        config = PINNConfig(
            input_dim=2, output_dim=1, hidden_dims=[32], use_batch_norm=True
        )
        model = PINN(config)
        x = torch.randn(10, 2)
        y = model(x)
        assert y.shape == (10, 1)

    def test_dropout(self) -> None:
        """Test dropout integration."""
        config = PINNConfig(
            input_dim=2, output_dim=1, hidden_dims=[32], dropout_rate=0.5
        )
        model = PINN(config)

        x = torch.randn(10, 2)

        # Training mode (dropout active)
        model.train()
        y1 = model(x)

        # Eval mode (dropout inactive)
        model.eval()
        y2 = model(x)

        assert y1.shape == y2.shape == (10, 1)

    def test_data_loss(self) -> None:
        """Test data loss computation."""
        config = PINNConfig(input_dim=2, output_dim=1)
        model = PINN(config)

        x = torch.randn(10, 2)
        y_true = torch.randn(10, 1)

        loss = model.compute_data_loss(x, y_true)
        assert loss.item() >= 0  # MSE is non-negative

    def test_physics_loss_with_function(self) -> None:
        """Test physics loss with provided PDE residual function."""

        def simple_pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            """Simple PDE: du/dx = 0."""
            du_dx = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            return du_dx

        config = PINNConfig(input_dim=1, output_dim=1)
        model = PINN(config, pde_residual_fn=simple_pde)

        x = torch.randn(10, 1, requires_grad=True)
        loss = model.compute_physics_loss(x)

        assert loss.item() >= 0

    def test_physics_loss_without_function_raises(self) -> None:
        """Test that physics loss raises error without PDE function."""
        config = PINNConfig(input_dim=1, output_dim=1)
        model = PINN(config)  # No pde_residual_fn

        x = torch.randn(10, 1, requires_grad=True)

        with pytest.raises(RuntimeError):
            model.compute_physics_loss(x)

    def test_total_loss(self) -> None:
        """Test total loss computation."""

        def dummy_pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(u)

        config = PINNConfig(input_dim=1, output_dim=1)
        model = PINN(config, pde_residual_fn=dummy_pde)

        x = torch.randn(10, 1, requires_grad=True)
        y_true = torch.randn(10, 1)

        losses = model.compute_total_loss(x, y_true, lambda_physics=0.5)

        assert "total_loss" in losses
        assert "data_loss" in losses
        assert "physics_loss" in losses
        assert all(v.item() >= 0 for v in losses.values())

    def test_predict(self) -> None:
        """Test prediction mode (no gradients)."""
        config = PINNConfig(input_dim=2, output_dim=1)
        model = PINN(config)

        x = torch.randn(10, 2)
        y = model.predict(x)

        assert y.shape == (10, 1)
        assert not y.requires_grad

    def test_get_config(self) -> None:
        """Test configuration retrieval."""
        config = PINNConfig(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        model = PINN(config)

        model_config = model.get_config()

        assert "model_type" in model_config
        assert "num_parameters" in model_config
        assert "pinn_config" in model_config
        assert model_config["num_parameters"] > 0