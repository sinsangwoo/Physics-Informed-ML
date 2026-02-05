"""Example: Benchmarking Fourier Neural Operator on Heat Equation.

This script demonstrates:
1. Training FNO on 1D heat equation
2. Evaluating resolution-invariance
3. Comparing against PINN baseline
4. Visualizing results

Usage:
    python examples/neural_operators/run_fno_benchmark.py
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from physics_informed_ml.models.operators.fno import FNO1d
from physics_informed_ml.benchmarks.problems import HeatEquation1D
from physics_informed_ml.benchmarks.runner import BenchmarkRunner


def main():
    """Run FNO benchmark on heat equation."""
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize problem
    problem = HeatEquation1D(alpha=0.01, L=1.0)
    print(f"Problem: {problem.__class__.__name__}")
    print(f"  Thermal diffusivity: {problem.alpha}")
    print(f"  Domain length: {problem.L}")
    
    # Initialize FNO model
    # Key hyperparameters:
    # - modes: Number of Fourier modes (12 is good for 1D)
    # - width: Hidden dimension (32 is reasonable)
    # - n_layers: Depth (4 layers is standard)
    fno = FNO1d(
        modes=12,
        width=32,
        in_channels=1,  # Initial temperature field
        out_channels=1,  # Final temperature field
        n_layers=4,
    )
    
    print(f"\nModel: {fno.__class__.__name__}")
    print(f"  Fourier modes: 12")
    print(f"  Hidden width: 32")
    print(f"  Layers: 4")
    print(f"  Parameters: {sum(p.numel() for p in fno.parameters()):,}")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(device=device, results_dir=Path("results"))
    
    # Run benchmark
    print("\n" + "="*70)
    print("STARTING BENCHMARK")
    print("="*70)
    
    results = runner.run(
        model=fno,
        problem=problem,
        train_resolution=64,  # Train on 64 grid points
        test_resolutions=[64, 128, 256],  # Test on multiple resolutions
        n_train_samples=1000,
        n_test_samples=100,
        epochs=100,
        batch_size=32,
        lr=1e-3,
    )
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    print("\nResolution-Invariance Test:")
    for res, metrics in results.items():
        speedup = 1000 / metrics.inference_time_ms  # Predictions per second
        print(f"  {res:3d} points: "
              f"L2 Error = {metrics.l2_relative_error:.4f}, "
              f"Time = {metrics.inference_time_ms:.2f}ms, "
              f"Throughput = {speedup:.0f} pred/s")
    
    # Check resolution invariance
    errors = [m.l2_relative_error for m in results.values()]
    if max(errors) / min(errors) < 2.0:
        print("\n✓ Resolution-invariance achieved! Error stays consistent.")
    else:
        print("\n✗ Model shows resolution dependence.")
    
    # Visualize training curves
    plot_training_curves(results)
    
    # Visualize predictions
    plot_predictions(fno, problem, device)
    
    print("\n✓ Benchmark completed successfully!")


def plot_training_curves(results):
    """Plot training loss curves."""
    plt.figure(figsize=(10, 6))
    
    # Get loss history from first result
    first_metrics = next(iter(results.values()))
    loss_history = first_metrics.loss_history
    
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("FNO Training Curve on Heat Equation", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Save figure
    plt.tight_layout()
    plt.savefig("results/training_curve.png", dpi=150)
    print("\nTraining curve saved to results/training_curve.png")
    plt.close()


def plot_predictions(model, problem, device):
    """Visualize model predictions vs ground truth."""
    model.eval()
    
    # Generate test sample
    x = torch.linspace(0, problem.L, 128)
    t = 0.1
    
    # Initial condition (k=2 sine wave)
    u0 = torch.sin(2 * 3.14159 * x / problem.L)
    
    # Ground truth
    u_true = problem.analytical_solution(x, torch.tensor(t), k=2)
    
    # Prediction
    with torch.no_grad():
        u_input = u0.unsqueeze(0).unsqueeze(-1).to(device)
        u_pred = model(u_input).squeeze().cpu()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Comparison
    axes[0].plot(x.numpy(), u0.numpy(), 'k--', label='Initial (t=0)', linewidth=2)
    axes[0].plot(x.numpy(), u_true.numpy(), 'b-', label='True (t=0.1)', linewidth=2)
    axes[0].plot(x.numpy(), u_pred.numpy(), 'r--', label='FNO Prediction', linewidth=2)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('u(x,t)', fontsize=12)
    axes[0].set_title('Heat Diffusion: FNO vs True Solution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Error
    error = torch.abs(u_pred - u_true)
    axes[1].plot(x.numpy(), error.numpy(), 'r-', linewidth=2)
    axes[1].fill_between(x.numpy(), 0, error.numpy(), alpha=0.3, color='red')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('Absolute Error', fontsize=12)
    axes[1].set_title('Pointwise Prediction Error', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/predictions.png", dpi=150)
    print("Predictions saved to results/predictions.png")
    plt.close()


if __name__ == "__main__":
    main()
