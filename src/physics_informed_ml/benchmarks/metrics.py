"""Performance metrics for benchmarking.

Provides comprehensive evaluation metrics for comparing:
- Neural operators vs traditional solvers
- Different model architectures
- Accuracy-speed trade-offs
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class BenchmarkMetrics:
    """Container for benchmark results.
    
    Tracks both accuracy and computational efficiency metrics.
    """
    
    # Accuracy metrics
    l2_relative_error: float = 0.0  # ||pred - true||_2 / ||true||_2
    l_inf_error: float = 0.0  # max|pred - true|
    mse: float = 0.0  # Mean squared error
    r2_score: float = 0.0  # Coefficient of determination
    
    # Computational metrics
    inference_time_ms: float = 0.0  # Forward pass time
    training_time_s: float = 0.0  # Total training time
    memory_mb: float = 0.0  # Peak memory usage
    
    # Model complexity
    n_parameters: int = 0  # Total trainable parameters
    flops: Optional[int] = None  # Floating point operations
    
    # Resolution metrics (for operator learning)
    train_resolution: int = 0  # Grid resolution during training
    test_resolution: int = 0  # Grid resolution during testing
    
    # Additional info
    problem_name: str = ""
    model_name: str = ""
    
    # History tracking
    loss_history: List[float] = field(default_factory=list)
    
    def compute_accuracy_metrics(
        self, pred: torch.Tensor, true: torch.Tensor
    ) -> None:
        """Compute all accuracy metrics.
        
        Args:
            pred: Predicted values
            true: Ground truth values
        """
        with torch.no_grad():
            # L2 relative error (normalized by solution magnitude)
            l2_diff = torch.norm(pred - true, p=2)
            l2_true = torch.norm(true, p=2)
            self.l2_relative_error = (l2_diff / l2_true).item()
            
            # L-infinity error (maximum pointwise error)
            self.l_inf_error = torch.max(torch.abs(pred - true)).item()
            
            # Mean squared error
            self.mse = torch.mean((pred - true) ** 2).item()
            
            # R² score (1 = perfect, 0 = mean baseline, <0 = worse than mean)
            ss_res = torch.sum((true - pred) ** 2)
            ss_tot = torch.sum((true - true.mean()) ** 2)
            self.r2_score = (1 - ss_res / ss_tot).item()
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for logging/saving."""
        return {
            "problem": self.problem_name,
            "model": self.model_name,
            "accuracy": {
                "l2_relative_error": self.l2_relative_error,
                "l_inf_error": self.l_inf_error,
                "mse": self.mse,
                "r2_score": self.r2_score,
            },
            "performance": {
                "inference_time_ms": self.inference_time_ms,
                "training_time_s": self.training_time_s,
                "memory_mb": self.memory_mb,
            },
            "model_complexity": {
                "n_parameters": self.n_parameters,
                "flops": self.flops,
            },
            "resolution": {
                "train": self.train_resolution,
                "test": self.test_resolution,
            },
        }
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║ Benchmark Results: {self.problem_name} - {self.model_name}
╠══════════════════════════════════════════════════════════════╣
║ Accuracy Metrics:
║   L2 Relative Error:  {self.l2_relative_error:.6f}
║   L∞ Error:           {self.l_inf_error:.6f}
║   MSE:                {self.mse:.6e}
║   R² Score:           {self.r2_score:.6f}
╠══════════════════════════════════════════════════════════════╣
║ Performance Metrics:
║   Inference Time:     {self.inference_time_ms:.2f} ms
║   Training Time:      {self.training_time_s:.2f} s
║   Memory Usage:       {self.memory_mb:.2f} MB
╠══════════════════════════════════════════════════════════════╣
║ Model Complexity:
║   Parameters:         {self.n_parameters:,}
║   Resolution:         {self.train_resolution} → {self.test_resolution}
╚══════════════════════════════════════════════════════════════╝
        """


def measure_inference_time(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    n_runs: int = 100,
    warmup: int = 10,
) -> float:
    """Measure average inference time.
    
    Args:
        model: Model to benchmark
        input_data: Sample input
        n_runs: Number of timing runs
        warmup: Number of warmup runs (excluded from timing)
        
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    # Warmup (allows GPU to optimize)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
    
    # Synchronize if using CUDA (important for accurate timing!)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Actual timing
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(input_data)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
    
    # Return average in milliseconds
    return np.mean(times) * 1000


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory_usage(model: torch.nn.Module) -> float:
    """Measure peak memory usage.
    
    Args:
        model: PyTorch model
        
    Returns:
        Memory usage in MB
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # Dummy forward pass to allocate memory
        device = next(model.parameters()).device
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1024**2
    
    # CPU memory estimation (parameter size + activations)
    param_memory = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1024**2
    
    return param_memory
