"""Tests for benchmark suite."""

import pytest
import torch

from physics_informed_ml.benchmarks.problems import (
    HeatEquation1D,
    WaveEquation1D,
    BurgersEquation1D,
)
from physics_informed_ml.benchmarks.metrics import (
    BenchmarkMetrics,
    measure_inference_time,
    count_parameters,
)
from physics_informed_ml.models.operators.fno import FNO1d


class TestBenchmarkProblems:
    """Test benchmark PDE problems."""
    
    def test_heat_equation_data_generation(self):
        """Test heat equation generates correct data shapes."""
        problem = HeatEquation1D(alpha=0.01, L=1.0)
        
        n_samples = 10
        resolution = 64
        
        X, y = problem.generate_data(n_samples, resolution)
        
        assert X.shape == (n_samples, resolution, 1)
        assert y.shape == (n_samples, resolution, 1)
    
    def test_heat_equation_analytical_solution(self):
        """Test heat equation analytical solution."""
        problem = HeatEquation1D(alpha=0.01, L=1.0)
        
        x = torch.linspace(0, 1.0, 100)
        t = torch.tensor(0.1)
        
        u = problem.analytical_solution(x, t, k=1)
        
        assert u.shape == x.shape
        # Solution should decay with time
        assert torch.all(torch.abs(u) <= 1.0)
    
    def test_wave_equation_data_generation(self):
        """Test wave equation generates correct data shapes."""
        problem = WaveEquation1D(c=1.0, L=1.0)
        
        n_samples = 5
        resolution = 64
        
        X, y = problem.generate_data(n_samples, resolution)
        
        # Input has 2 channels: [displacement, velocity]
        assert X.shape == (n_samples, resolution, 2)
        assert y.shape == (n_samples, resolution, 1)
    
    def test_burgers_equation_data_generation(self):
        """Test Burgers equation data generation."""
        problem = BurgersEquation1D(nu=0.01, L=1.0)
        
        n_samples = 5
        resolution = 64
        
        X, y = problem.generate_data(n_samples, resolution)
        
        assert X.shape == (n_samples, resolution, 1)
        assert y.shape == (n_samples, resolution, 1)


class TestBenchmarkMetrics:
    """Test benchmark metrics."""
    
    def test_accuracy_metrics(self):
        """Test accuracy metric computation."""
        metrics = BenchmarkMetrics()
        
        # Perfect prediction
        pred = torch.randn(10, 64, 1)
        true = pred.clone()
        
        metrics.compute_accuracy_metrics(pred, true)
        
        assert metrics.l2_relative_error < 1e-6
        assert metrics.l_inf_error < 1e-6
        assert metrics.mse < 1e-6
        assert abs(metrics.r2_score - 1.0) < 1e-5
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = BenchmarkMetrics(
            l2_relative_error=0.01,
            problem_name="HeatEquation",
            model_name="FNO1d",
        )
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data["problem"] == "HeatEquation"
        assert data["model"] == "FNO1d"
        assert "accuracy" in data
        assert "performance" in data
    
    def test_measure_inference_time(self):
        """Test inference time measurement."""
        model = FNO1d(modes=8, width=16, n_layers=2)
        x = torch.randn(1, 32, 1)
        
        time_ms = measure_inference_time(model, x, n_runs=10, warmup=2)
        
        assert time_ms > 0
        assert time_ms < 1000  # Should be fast
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = FNO1d(modes=8, width=16, n_layers=2)
        
        n_params = count_parameters(model)
        
        assert n_params > 0
        # Manual verification for this config
        expected_range = (5_000, 50_000)
        assert expected_range[0] < n_params < expected_range[1]


class TestResolutionInvariance:
    """Test resolution-invariance property of neural operators."""
    
    def test_fno_different_resolutions(self):
        """Test FNO works on multiple resolutions."""
        problem = HeatEquation1D()
        model = FNO1d(modes=12, width=32, n_layers=4)
        model.eval()
        
        resolutions = [32, 64, 128, 256]
        errors = []
        
        for res in resolutions:
            X, y = problem.generate_data(n_samples=10, resolution=res)
            
            with torch.no_grad():
                y_pred = model(X)
            
            # Compute error
            error = torch.norm(y_pred - y) / torch.norm(y)
            errors.append(error.item())
        
        # Errors should be in same ballpark (not increasing drastically)
        # This is a weak test since model is untrained
        assert all(e < 10.0 for e in errors)  # Sanity check
