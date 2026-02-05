"""Benchmark suite for physics-informed machine learning models.

Provides standardized test cases for various PDE problems with:
- Ground truth data generation
- Performance metrics (accuracy, speed, memory)
- Comparison against traditional solvers
"""

from physics_informed_ml.benchmarks.problems import (
    BenchmarkProblem,
    HeatEquation1D,
    WaveEquation1D,
    BurgersEquation1D,
    NavierStokes2D,
)
from physics_informed_ml.benchmarks.runner import BenchmarkRunner
from physics_informed_ml.benchmarks.metrics import BenchmarkMetrics

__all__ = [
    "BenchmarkProblem",
    "HeatEquation1D",
    "WaveEquation1D",
    "BurgersEquation1D",
    "NavierStokes2D",
    "BenchmarkRunner",
    "BenchmarkMetrics",
]
