"""Tests for Neural Operator models."""

import pytest
import torch
import torch.nn as nn

from physics_informed_ml.models.operators.fno import (
    FNO1d,
    FNO2d,
    FNO3d,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
)


class TestSpectralConv:
    """Test spectral convolution layers."""
    
    def test_spectral_conv_1d_shape(self):
        """Test 1D spectral conv output shape."""
        batch_size = 4
        in_channels = 3
        out_channels = 5
        spatial_dim = 64
        modes = 12
        
        layer = SpectralConv1d(in_channels, out_channels, modes)
        x = torch.randn(batch_size, in_channels, spatial_dim)
        
        output = layer(x)
        
        assert output.shape == (batch_size, out_channels, spatial_dim)
    
    def test_spectral_conv_2d_shape(self):
        """Test 2D spectral conv output shape."""
        batch_size = 4
        in_channels = 3
        out_channels = 5
        height, width = 32, 32
        modes1, modes2 = 12, 12
        
        layer = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        x = torch.randn(batch_size, in_channels, height, width)
        
        output = layer(x)
        
        assert output.shape == (batch_size, out_channels, height, width)
    
    def test_spectral_conv_3d_shape(self):
        """Test 3D spectral conv output shape."""
        batch_size = 2
        in_channels = 2
        out_channels = 4
        depth, height, width = 16, 16, 16
        modes1, modes2, modes3 = 8, 8, 8
        
        layer = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        x = torch.randn(batch_size, in_channels, depth, height, width)
        
        output = layer(x)
        
        assert output.shape == (batch_size, out_channels, depth, height, width)
    
    def test_spectral_conv_backward(self):
        """Test gradient flow through spectral conv."""
        layer = SpectralConv1d(2, 2, 8)
        x = torch.randn(1, 2, 32, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.weights.grad is not None


class TestFNO:
    """Test Fourier Neural Operator models."""
    
    def test_fno1d_forward(self):
        """Test FNO1d forward pass."""
        batch_size = 4
        spatial_dim = 64
        in_channels = 1
        out_channels = 1
        
        model = FNO1d(
            modes=12,
            width=32,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=4,
        )
        
        # Input shape: (batch, spatial, channels)
        x = torch.randn(batch_size, spatial_dim, in_channels)
        output = model(x)
        
        assert output.shape == (batch_size, spatial_dim, out_channels)
    
    def test_fno2d_forward(self):
        """Test FNO2d forward pass."""
        batch_size = 2
        height, width = 32, 32
        in_channels = 1
        out_channels = 1
        
        model = FNO2d(
            modes1=12,
            modes2=12,
            width=32,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=4,
        )
        
        # Input shape: (batch, height, width, channels)
        x = torch.randn(batch_size, height, width, in_channels)
        output = model(x)
        
        assert output.shape == (batch_size, height, width, out_channels)
    
    def test_fno3d_forward(self):
        """Test FNO3d forward pass."""
        batch_size = 1
        depth, height, width = 16, 16, 16
        in_channels = 1
        out_channels = 1
        
        model = FNO3d(
            modes1=8,
            modes2=8,
            modes3=8,
            width=16,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=2,
        )
        
        # Input shape: (batch, depth, height, width, channels)
        x = torch.randn(batch_size, depth, height, width, in_channels)
        output = model(x)
        
        assert output.shape == (batch_size, depth, height, width, out_channels)
    
    def test_fno_resolution_invariance(self):
        """Test that FNO works on different resolutions."""
        model = FNO1d(modes=8, width=16, n_layers=2)
        
        # Test on different resolutions
        for resolution in [32, 64, 128]:
            x = torch.randn(1, resolution, 1)
            output = model(x)
            assert output.shape == (1, resolution, 1)
    
    def test_fno_gradient_flow(self):
        """Test gradient flow through FNO."""
        model = FNO1d(modes=8, width=16, n_layers=2)
        x = torch.randn(2, 32, 1, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None
    
    def test_fno_parameter_count(self):
        """Test FNO has reasonable number of parameters."""
        model = FNO1d(modes=12, width=32, n_layers=4)
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should have tens of thousands of parameters, not millions
        assert 10_000 < n_params < 500_000
    
    @pytest.mark.parametrize("modes,width,n_layers", [
        (8, 16, 2),
        (12, 32, 4),
        (16, 64, 6),
    ])
    def test_fno_different_configs(self, modes, width, n_layers):
        """Test FNO with different hyperparameter configurations."""
        model = FNO1d(modes=modes, width=width, n_layers=n_layers)
        x = torch.randn(2, 64, 1)
        
        output = model(x)
        assert output.shape == (2, 64, 1)
    
    def test_fno_multi_channel(self):
        """Test FNO with multiple input/output channels."""
        model = FNO1d(
            modes=12,
            width=32,
            in_channels=3,  # e.g., [u, v, p]
            out_channels=2,  # e.g., [u, v]
            n_layers=4,
        )
        
        x = torch.randn(2, 64, 3)
        output = model(x)
        
        assert output.shape == (2, 64, 2)


class TestFNOTraining:
    """Test FNO training capabilities."""
    
    def test_simple_overfitting(self):
        """Test that FNO can overfit a simple pattern."""
        # Simple task: learn to double the input
        model = FNO1d(modes=8, width=16, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Generate simple data
        x_train = torch.randn(10, 32, 1)
        y_train = 2 * x_train
        
        # Train for a few epochs
        initial_loss = None
        for epoch in range(50):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = nn.MSELoss()(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss.item()
        
        # Loss should decrease significantly
        assert loss.item() < 0.1 * initial_loss
    
    def test_deterministic_forward(self):
        """Test that forward pass is deterministic."""
        torch.manual_seed(42)
        model = FNO1d(modes=8, width=16, n_layers=2)
        x = torch.randn(2, 32, 1)
        
        # Two forward passes with same input
        output1 = model(x)
        output2 = model(x)
        
        # Should be identical
        assert torch.allclose(output1, output2)
