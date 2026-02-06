"""Benchmark tests for Fourier Neural Operators."""

import pytest
import torch
from physics_informed_ml.models import FNO1d, FNO2d


class TestFNOBenchmarks:
    """Benchmark FNO performance."""

    @pytest.mark.benchmark(group="fno1d")
    def test_fno1d_forward_small(self, benchmark):
        """Benchmark small FNO1d forward pass."""
        model = FNO1d(modes=8, width=16, n_layers=2)
        model.eval()
        x = torch.randn(1, 32, 1)

        result = benchmark(model, x)
        assert result.shape == (1, 32, 1)

    @pytest.mark.benchmark(group="fno1d")
    def test_fno1d_forward_medium(self, benchmark):
        """Benchmark medium FNO1d forward pass."""
        model = FNO1d(modes=16, width=32, n_layers=4)
        model.eval()
        x = torch.randn(1, 64, 1)

        result = benchmark(model, x)
        assert result.shape == (1, 64, 1)

    @pytest.mark.benchmark(group="fno1d")
    def test_fno1d_forward_large(self, benchmark):
        """Benchmark large FNO1d forward pass."""
        model = FNO1d(modes=16, width=64, n_layers=4)
        model.eval()
        x = torch.randn(1, 128, 1)

        result = benchmark(model, x)
        assert result.shape == (1, 128, 1)

    @pytest.mark.benchmark(group="fno2d")
    def test_fno2d_forward_small(self, benchmark):
        """Benchmark small FNO2d forward pass."""
        model = FNO2d(modes=(8, 8), width=16, n_layers=2)
        model.eval()
        x = torch.randn(1, 32, 32, 1)

        result = benchmark(model, x)
        assert result.shape == (1, 32, 32, 1)

    @pytest.mark.benchmark(group="fno2d")
    def test_fno2d_forward_medium(self, benchmark):
        """Benchmark medium FNO2d forward pass."""
        model = FNO2d(modes=(12, 12), width=32, n_layers=4)
        model.eval()
        x = torch.randn(1, 64, 64, 1)

        result = benchmark(model, x)
        assert result.shape == (1, 64, 64, 1)

    @pytest.mark.benchmark(group="resolution-invariance")
    def test_fno1d_resolution_64(self, benchmark):
        """Test FNO at resolution 64."""
        model = FNO1d(modes=16, width=32, n_layers=4)
        model.eval()
        x = torch.randn(1, 64, 1)

        result = benchmark(model, x)
        assert result.shape == (1, 64, 1)

    @pytest.mark.benchmark(group="resolution-invariance")
    def test_fno1d_resolution_128(self, benchmark):
        """Test FNO at resolution 128."""
        model = FNO1d(modes=16, width=32, n_layers=4)
        model.eval()
        x = torch.randn(1, 128, 1)

        result = benchmark(model, x)
        assert result.shape == (1, 128, 1)

    @pytest.mark.benchmark(group="resolution-invariance")
    def test_fno1d_resolution_256(self, benchmark):
        """Test FNO at resolution 256."""
        model = FNO1d(modes=16, width=32, n_layers=4)
        model.eval()
        x = torch.randn(1, 256, 1)

        result = benchmark(model, x)
        assert result.shape == (1, 256, 1)
