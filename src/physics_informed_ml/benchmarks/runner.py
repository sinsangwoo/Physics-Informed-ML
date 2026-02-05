"""Benchmark runner for systematic evaluation.

Provides automated workflow for:
1. Training models on benchmark problems
2. Evaluating on multiple resolutions
3. Comparing against baselines
4. Generating reports
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from typing import Dict, List, Optional, Type
from pathlib import Path
import json

from physics_informed_ml.benchmarks.problems import BenchmarkProblem
from physics_informed_ml.benchmarks.metrics import (
    BenchmarkMetrics,
    measure_inference_time,
    count_parameters,
    measure_memory_usage,
)


class BenchmarkRunner:
    """Orchestrates benchmark experiments.
    
    Usage:
        runner = BenchmarkRunner(device="cuda")
        metrics = runner.run(
            model=fno_model,
            problem=HeatEquation1D(),
            train_resolution=64,
            test_resolutions=[64, 128, 256],
        )
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        results_dir: Optional[Path] = None,
    ):
        """Initialize benchmark runner.
        
        Args:
            device: Device for computation
            results_dir: Directory to save results
        """
        self.device = torch.device(device)
        self.results_dir = results_dir or Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"BenchmarkRunner initialized on {self.device}")
    
    def run(
        self,
        model: nn.Module,
        problem: BenchmarkProblem,
        train_resolution: int = 64,
        test_resolutions: List[int] = [64, 128, 256],
        n_train_samples: int = 1000,
        n_test_samples: int = 100,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> Dict[int, BenchmarkMetrics]:
        """Run complete benchmark.
        
        Args:
            model: Neural operator model to benchmark
            problem: PDE problem
            train_resolution: Grid resolution for training
            test_resolutions: Grid resolutions for testing (generalization)
            n_train_samples: Number of training samples
            n_test_samples: Number of test samples
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Dictionary mapping resolution -> metrics
        """
        print(f"\n{'='*70}")
        print(f"Running benchmark: {problem.__class__.__name__}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Training resolution: {train_resolution}")
        print(f"Test resolutions: {test_resolutions}")
        print(f"{'='*70}\n")
        
        # Move model to device
        model = model.to(self.device)
        
        # Step 1: Generate training data
        print("Generating training data...")
        X_train, y_train = problem.generate_data(n_train_samples, train_resolution)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Step 2: Train model
        print(f"Training for {epochs} epochs...")
        metrics_dict = self._train(
            model, train_loader, epochs, lr, problem.__class__.__name__
        )
        
        # Step 3: Evaluate on multiple resolutions
        results = {}
        for test_res in test_resolutions:
            print(f"\nEvaluating at resolution {test_res}...")
            
            # Generate test data at this resolution
            X_test, y_test = problem.generate_data(n_test_samples, test_res)
            
            # Evaluate
            metrics = self._evaluate(
                model, X_test, y_test, train_resolution, test_res
            )
            metrics.problem_name = problem.__class__.__name__
            metrics.model_name = model.__class__.__name__
            
            results[test_res] = metrics
            print(metrics)
        
        # Step 4: Save results
        self._save_results(results, problem.__class__.__name__, model.__class__.__name__)
        
        return results
    
    def _train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
        problem_name: str,
    ) -> BenchmarkMetrics:
        """Train the model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            epochs: Number of epochs
            lr: Learning rate
            problem_name: Name of the problem
            
        Returns:
            Training metrics
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        model.train()
        start_time = time.time()
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f}s")
        
        # Create metrics object
        metrics = BenchmarkMetrics(
            training_time_s=training_time,
            n_parameters=count_parameters(model),
            loss_history=loss_history,
        )
        
        return metrics
    
    def _evaluate(
        self,
        model: nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        train_resolution: int,
        test_resolution: int,
    ) -> BenchmarkMetrics:
        """Evaluate model on test data.
        
        Args:
            model: Trained model
            X_test: Test inputs
            y_test: Test targets
            train_resolution: Resolution used for training
            test_resolution: Current test resolution
            
        Returns:
            Evaluation metrics
        """
        model.eval()
        
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Measure inference time
        inf_time = measure_inference_time(model, X_test[:1])
        
        # Get predictions
        with torch.no_grad():
            y_pred = model(X_test)
        
        # Compute accuracy metrics
        metrics = BenchmarkMetrics(
            train_resolution=train_resolution,
            test_resolution=test_resolution,
            inference_time_ms=inf_time,
            n_parameters=count_parameters(model),
            memory_mb=measure_memory_usage(model),
        )
        metrics.compute_accuracy_metrics(y_pred, y_test)
        
        return metrics
    
    def _save_results(
        self, results: Dict[int, BenchmarkMetrics], problem: str, model: str
    ) -> None:
        """Save benchmark results to JSON.
        
        Args:
            results: Dictionary of metrics per resolution
            problem: Problem name
            model: Model name
        """
        filename = self.results_dir / f"{problem}_{model}_results.json"
        
        # Convert to serializable format
        results_dict = {
            str(res): metrics.to_dict() for res, metrics in results.items()
        }
        
        with open(filename, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def compare_models(
        self,
        models: List[nn.Module],
        problem: BenchmarkProblem,
        **kwargs,
    ) -> Dict[str, Dict[int, BenchmarkMetrics]]:
        """Compare multiple models on the same problem.
        
        Args:
            models: List of models to compare
            problem: Benchmark problem
            **kwargs: Additional arguments for run()
            
        Returns:
            Dictionary mapping model name -> results
        """
        all_results = {}
        
        for model in models:
            model_name = model.__class__.__name__
            print(f"\n{'-'*70}")
            print(f"Benchmarking {model_name}")
            print(f"{'-'*70}")
            
            results = self.run(model, problem, **kwargs)
            all_results[model_name] = results
        
        # Print comparison summary
        self._print_comparison(all_results)
        
        return all_results
    
    def _print_comparison(self, all_results: Dict) -> None:
        """Print comparison table of all models."""
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        for model_name, results in all_results.items():
            print(f"\n{model_name}:")
            for res, metrics in results.items():
                print(
                    f"  Resolution {res}: "
                    f"L2 Error={metrics.l2_relative_error:.6f}, "
                    f"Time={metrics.inference_time_ms:.2f}ms"
                )
