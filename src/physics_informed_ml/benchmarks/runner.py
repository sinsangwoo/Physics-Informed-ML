"""Benchmark runner for evaluating neural operator performance.

Provides:
- Standardized training loop
- Multi-resolution evaluation
- Performance metrics collection
- Result serialization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import numpy as np

from physics_informed_ml.benchmarks.problems import BenchmarkProblem
from physics_informed_ml.benchmarks.metrics import BenchmarkMetrics


class BenchmarkRunner:
    """Runs benchmarks on neural operator models.
    
    Features:
    - Automated training with progress tracking
    - Multi-resolution evaluation
    - Performance profiling (time, memory)
    - Result export (JSON, plots)
    
    Example:
        >>> runner = BenchmarkRunner(device="cuda")
        >>> problem = HeatEquation1D(alpha=0.01)
        >>> model = FNO1d(modes=12, width=32)
        >>> results = runner.run(
        ...     model=model,
        ...     problem=problem,
        ...     train_resolution=64,
        ...     test_resolutions=[64, 128, 256],
        ... )
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize benchmark runner.
        
        Args:
            device: Device to use ("cpu", "cuda", "cuda:0", etc.)
                   If None, automatically selects GPU if available
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    def run(
        self,
        model: nn.Module,
        problem: BenchmarkProblem,
        train_resolution: int = 64,
        test_resolutions: Optional[List[int]] = None,
        n_samples: int = 1000,
        batch_size: int = 32,
        n_epochs: int = 100,
        lr: float = 1e-3,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run complete benchmark.
        
        Args:
            model: Neural operator model to benchmark
            problem: PDE problem to solve
            train_resolution: Spatial resolution for training
            test_resolutions: Resolutions to test on (default: [64, 128, 256])
            n_samples: Number of training samples
            batch_size: Training batch size
            n_epochs: Number of training epochs
            lr: Learning rate
            save_dir: Directory to save results
            
        Returns:
            Dictionary with benchmark results
        """
        if test_resolutions is None:
            test_resolutions = [64, 128, 256]
        
        model = model.to(self.device)
        
        print(f"Running benchmark on {self.device}")
        print(f"Problem: {problem.__class__.__name__}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Training resolution: {train_resolution}")
        print(f"Test resolutions: {test_resolutions}")
        
        # Step 1: Generate training data
        print(f"\nGenerating {n_samples} training samples...")
        X_train, y_train = problem.generate_data(n_samples, train_resolution)
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Step 2: Train model
        print(f"Training for {n_epochs} epochs...")
        # Fixed: Remove unused variable
        self._train(
            model, train_loader, n_epochs, lr, problem.__class__.__name__
        )
        
        # Step 3: Evaluate on multiple resolutions
        print("\nEvaluating on multiple resolutions...")
        results = {}
        
        for resolution in test_resolutions:
            print(f"\nTesting on resolution {resolution}...")
            
            # Generate test data
            X_test, y_test = problem.generate_data(100, resolution)
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            
            # Evaluate
            metrics = self._evaluate(
                model, X_test, y_test, resolution
            )
            
            results[resolution] = metrics
            
            print(f"  L2 Error: {metrics.l2_relative_error:.6f}")
            print(f"  Inference: {metrics.inference_time_ms:.2f}ms")
        
        # Step 4: Save results
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(results, save_dir)
        
        return results
    
    def _train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
        problem_name: str,
    ) -> Dict[str, List[float]]:
        """Training loop.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            epochs: Number of epochs
            lr: Learning rate
            problem_name: Name of problem (for logging)
            
        Returns:
            Training metrics
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        return {"loss": losses}
    
    def _evaluate(
        self,
        model: nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        resolution: int,
    ) -> BenchmarkMetrics:
        """Evaluate model on test data.
        
        Args:
            model: Model to evaluate
            X_test: Test inputs
            y_test: Test targets
            resolution: Spatial resolution
            
        Returns:
            Benchmark metrics
        """
        model.eval()
        
        # Measure inference time
        with torch.no_grad():
            # Warmup
            _ = model(X_test[:1])
            
            # Measure
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            predictions = model(X_test)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Compute metrics
        metrics = BenchmarkMetrics(
            problem_name="benchmark",
            model_name=model.__class__.__name__,
            train_resolution=resolution,
            test_resolution=resolution,
        )
        
        # Update with actual values
        metrics.compute_accuracy(predictions, y_test)
        metrics.inference_time_ms = inference_time / len(X_test)  # per sample
        metrics.n_parameters = sum(p.numel() for p in model.parameters())
        
        return metrics
    
    def _save_results(
        self, results: Dict[int, BenchmarkMetrics], save_dir: Path
    ) -> None:
        """Save benchmark results.
        
        Args:
            results: Benchmark results by resolution
            save_dir: Directory to save results
        """
        # Convert to JSON-serializable format
        results_dict = {}
        for resolution, metrics in results.items():
            results_dict[str(resolution)] = {
                "l2_error": float(metrics.l2_relative_error),
                "l_inf_error": float(metrics.l_inf_error),
                "mse": float(metrics.mse),
                "r2_score": float(metrics.r2_score),
                "inference_time_ms": float(metrics.inference_time_ms),
                "n_parameters": int(metrics.n_parameters),
            }
        
        # Save JSON
        json_path = save_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {json_path}")
