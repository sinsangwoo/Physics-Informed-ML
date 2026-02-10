"""Benchmark tests for FNO models.

These tests measure performance characteristics:
- Inference latency
- Training throughput
- Memory usage
- Resolution scaling
"""

import pytest
import torch
from physics_informed_ml.models import FNO1d
from physics_informed_ml.benchmarks import HeatEquation1D, BenchmarkRunner


@pytest.mark.benchmark
class TestFNOBenchmark:
    """Benchmark FNO performance."""

    @pytest.fixture
    def model(self):
        """Create FNO model."""
        return FNO1d(modes=8, width=32, n_layers=4)

    @pytest.fixture
    def problem(self):
        """Create benchmark problem."""
        # Fixed: HeatEquation1D takes alpha and L, not n_points
        return HeatEquation1D(alpha=0.01, L=1.0)

    def test_fno_inference_latency(self, benchmark, model):
        """Benchmark FNO inference latency."""
        model.eval()
        x = torch.randn(1, 64, 1)

        def inference():
            with torch.no_grad():
                return model(x)
        
        benchmark(inference)

    def test_fno_batch_inference(self, benchmark, model):
        """Benchmark batch inference."""
        model.eval()
        x = torch.randn(32, 64, 1)  # Batch of 32

        def batch_inference():
            with torch.no_grad():
                return model(x)
        
        benchmark(batch_inference)

    def test_fno_training_step(self, benchmark, model):
        """Benchmark single training step."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(16, 64, 1)
        y = torch.randn(16, 64, 1)

        def train_step():
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            return loss.item()
        
        benchmark(train_step)

    @pytest.mark.parametrize("resolution", [32, 64, 128, 256])
    def test_fno_resolution_scaling(self, benchmark, resolution):
        """Benchmark FNO at different resolutions."""
        model = FNO1d(modes=8, width=32, n_layers=4)
        model.eval()
        x = torch.randn(1, resolution, 1)

        def inference():
            with torch.no_grad():
                return model(x)
        
        benchmark(inference)

    def test_benchmark_runner(self, model, problem):
        """Test benchmark runner without pytest-benchmark fixture."""
        # Fixed: Don't use benchmark fixture, just test functionality
        runner = BenchmarkRunner()
        
        # Quick test with minimal epochs
        results = runner.run(
            model=model,
            problem=problem,
            n_epochs=2,  # Minimal for testing
            batch_size=8,
            test_resolutions=[64],
        )
        
        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)
