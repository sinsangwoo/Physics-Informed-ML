"""Tests for Multi-Layer Perceptron."""

import pytest
import torch
from physics_informed_ml.models.mlp import MLP


class TestMLP:
    """Test suite for MLP model."""

    def test_mlp_initialization(self) -> None:
        """Test MLP initialization."""
        model = MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1)

        assert model.input_dim == 2
        assert model.hidden_dims == [32, 32]
        assert model.output_dim == 1

    def test_mlp_forward_pass(self) -> None:
        """Test forward pass."""
        model = MLP(input_dim=3, hidden_dims=[64, 64], output_dim=2)

        x = torch.randn(10, 3)
        y = model(x)

        assert y.shape == (10, 2)

    def test_mlp_different_activations(self) -> None:
        """Test different activation functions."""
        activations = ["relu", "tanh", "sigmoid", "gelu", "silu"]

        for act in activations:
            model = MLP(
                input_dim=2, hidden_dims=[16], output_dim=1, activation=act
            )
            x = torch.randn(5, 2)
            y = model(x)
            assert y.shape == (5, 1)

    def test_mlp_invalid_activation(self) -> None:
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError):
            MLP(
                input_dim=2,
                hidden_dims=[16],
                output_dim=1,
                activation="invalid",
            )

    def test_mlp_batch_norm(self) -> None:
        """Test with batch normalization."""
        model = MLP(
            input_dim=2,
            hidden_dims=[32, 32],
            output_dim=1,
            use_batch_norm=True,
        )

        x = torch.randn(10, 2)
        y = model(x)
        assert y.shape == (10, 1)

    def test_mlp_dropout(self) -> None:
        """Test with dropout."""
        model = MLP(
            input_dim=2, hidden_dims=[32, 32], output_dim=1, dropout_rate=0.2
        )

        x = torch.randn(10, 2)

        # Training mode
        model.train()
        y_train = model(x)

        # Eval mode
        model.eval()
        y_eval = model(x)

        assert y_train.shape == (10, 1)
        assert y_eval.shape == (10, 1)

    def test_mlp_parameter_count(self) -> None:
        """Test parameter counting."""
        model = MLP(input_dim=2, hidden_dims=[64, 64], output_dim=1)

        n_params = model.count_parameters()
        assert n_params > 0

        # Manual calculation for verification
        # Layer 1: (2 * 64) + 64 = 192
        # Layer 2: (64 * 64) + 64 = 4160
        # Layer 3: (64 * 1) + 1 = 65
        # Total: 192 + 4160 + 65 = 4417
        assert n_params == 4417

    def test_mlp_gradient_flow(self) -> None:
        """Test that gradients flow properly."""
        model = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)

        x = torch.randn(10, 2)
        y = model(x)
        loss = y.mean()
        loss.backward()

        # Check all parameters have gradients
        for param in model.parameters():
            assert param.grad is not None