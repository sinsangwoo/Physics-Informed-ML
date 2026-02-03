"""Tests for Physics-Informed Neural Network."""

import pytest
import torch
from physics_informed_ml.models import PINN, PINNConfig


class TestPINN:
    """Test suite for PINN model."""

    def test_pinn_initialization(self) -> None:
        """Test PINN model initialization."""
        config = PINNConfig(input_dim=2, hidden_dims=[32, 32], output_dim=1)
        model = PINN(config)

        assert model.config.input_dim == 2
        assert model.config.output_dim == 1
        assert len(model.config.hidden_dims) == 2

    def test_pinn_forward_pass(self) -> None:
        """Test PINN forward pass."""
        config = PINNConfig(input_dim=2, output_dim=1)
        model = PINN(config)

        x = torch.randn(10, 2)
        y = model(x)

        assert y.shape == (10, 1)

    def test_pinn_data_loss(self) -> None:
        """Test data loss computation."""
        config = PINNConfig(input_dim=2, output_dim=1)
        model = PINN(config)

        x_data = torch.randn(10, 2)
        y_data = torch.randn(10, 1)

        loss = model.compute_data_loss(x_data, y_data)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() >= 0

    def test_pinn_physics_loss(self) -> None:
        """Test physics loss computation."""

        def simple_pde(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Simple PDE: du/dx = 0."""
            x = x.requires_grad_(True)
            u = model(x)
            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0][:, 0:1]
            return u_x

        config = PINNConfig(input_dim=2, output_dim=1)
        model = PINN(config, pde_residual=simple_pde)

        x_physics = torch.randn(20, 2)
        loss = model.compute_physics_loss(x_physics)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() >= 0

    def test_pinn_total_loss(self) -> None:
        """Test total loss computation."""

        def dummy_pde(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1)

        config = PINNConfig(
            input_dim=2,
            output_dim=1,
            lambda_data=1.0,
            lambda_physics=0.5,
        )
        model = PINN(config, pde_residual=dummy_pde)

        x_data = torch.randn(10, 2)
        y_data = torch.randn(10, 1)
        x_physics = torch.randn(20, 2)

        total_loss, loss_dict = model.compute_total_loss(
            x_data=x_data, y_data=y_data, x_physics=x_physics
        )

        assert "total" in loss_dict
        assert "data" in loss_dict
        assert "physics" in loss_dict
        assert total_loss.item() >= 0

    def test_pinn_parameter_count(self) -> None:
        """Test parameter counting."""
        config = PINNConfig(input_dim=2, hidden_dims=[64, 64], output_dim=1)
        model = PINN(config)

        n_params = model.network.count_parameters()
        assert n_params > 0

    def test_pinn_gradient_flow(self) -> None:
        """Test that gradients flow properly."""

        def simple_pde(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            x = x.requires_grad_(True)
            u = model(x)
            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0][:, 0:1]
            return u_x

        config = PINNConfig(input_dim=2, output_dim=1, lambda_physics=1.0)
        model = PINN(config, pde_residual=simple_pde)

        x_physics = torch.randn(10, 2, requires_grad=True)
        loss = model.compute_physics_loss(x_physics)
        loss.backward()

        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None