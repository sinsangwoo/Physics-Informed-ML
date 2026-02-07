"""Benchmark tests for FNO models.

These tests measure performance characteristics:
- Inference latency
- Training throughput
- Memory usage
- Resolution scaling
"""

import pytest
import torch
import time
import numpy as np
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
        return HeatEquation1D(n_points=64, n_samples=100)

    def test_fno_inference_latency(self, benchmark, model):
        """Benchmark FNO inference latency."""
        model.eval()
        x = torch.randn(1, 64, 1)

        @benchmark
        def inference():
            with torch.no_grad():
                return model(x)

    def test_fno_batch_inference(self, benchmark, model):
        """Benchmark batch inference."""
        model.eval()
        x = torch.randn(32, 64, 1)  # Batch of 32

        @benchmark
        def batch_inference():
            with torch.no_grad():
                return model(x)

    def test_fno_training_step(self, benchmark, model):
        """Benchmark single training step."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(16, 64, 1)
        y = torch.randn(16, 64, 1)

        @benchmark
        def train_step():
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            return loss.item()

    @pytest.mark.parametrize("resolution", [32, 64, 128, 256])
    def test_fno_resolution_scaling(self, benchmark, resolution):
        """Benchmark FNO at different resolutions."""
        model = FNO1d(modes=8, width=32, n_layers=4)
        model.eval()
        x = torch.randn(1, resolution, 1)

        @benchmark
        def inference():
            with torch.no_grad():
                return model(x)

    def test_benchmark_runner(self, benchmark, model, problem):
        """Benchmark the benchmark runner itself."""
        runner = BenchmarkRunner()

        @benchmark
        def run_benchmark():
            return runner.run(
                model=model,
                problem=problem,
                n_epochs=10,
                batch_size=16,
                test_resolutions=[64],
            )
