"""Example: Bayesian Neural Networks for Uncertainty Quantification.

Demonstrates:
- Training Bayesian PINN
- Prediction with uncertainty
- Comparison with deterministic model
- Calibration analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from physics_informed_ml.uncertainty import (
    BayesianPINN,
    VariationalInference,
    CalibrationMetrics,
    plot_calibration_curve,
    plot_prediction_intervals,
)


def generate_noisy_data(n_samples=100):
    """Generate noisy training data from sine function."""
    x = torch.linspace(0, 2 * np.pi, n_samples).unsqueeze(-1)
    y = torch.sin(x) + torch.randn_like(x) * 0.1  # Add noise
    return x, y


def example_bayesian_training():
    """Example 1: Train Bayesian PINN."""
    print("="*70)
    print("Example 1: Bayesian PINN Training")
    print("="*70)
    
    # Generate data
    X_train, y_train = generate_noisy_data(n_samples=100)
    X_test = torch.linspace(0, 2 * np.pi, 200).unsqueeze(-1)
    y_test = torch.sin(X_test)
    
    # Create Bayesian model
    model = BayesianPINN(
        input_dim=1,
        hidden_dims=[32, 32],
        output_dim=1,
        prior_std=1.0,
    )
    
    # Variational inference
    vi = VariationalInference(model, n_data=len(X_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    print("\nTraining...")
    n_epochs = 500
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions, kl_divergence = model(X_train)
        
        # ELBO loss
        total_loss, nll, kl = vi.elbo_loss(predictions, y_train, kl_divergence)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}:")
            print(f"  Total Loss: {total_loss.item():.4f}")
            print(f"  NLL: {nll.item():.4f}")
            print(f"  KL: {kl.item():.4f}")
    
    # Predict with uncertainty
    print("\nPredicting with uncertainty...")
    mean, std, samples = model.predict_with_uncertainty(X_test, n_samples=100)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Predictions with uncertainty
    plt.subplot(1, 2, 1)
    plt.plot(X_test.numpy(), y_test.numpy(), 'r--', label='True', linewidth=2)
    plt.plot(X_test.numpy(), mean.numpy(), 'b-', label='Predicted', linewidth=2)
    plt.fill_between(
        X_test.numpy().flatten(),
        (mean - 2*std).numpy().flatten(),
        (mean + 2*std).numpy().flatten(),
        alpha=0.3,
        label='95% CI'
    )
    plt.scatter(X_train.numpy(), y_train.numpy(), c='gray', alpha=0.5, label='Training')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bayesian PINN Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty
    plt.subplot(1, 2, 2)
    plt.plot(X_test.numpy(), std.numpy(), 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Uncertainty (σ)')
    plt.title('Epistemic Uncertainty')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_pinn_results.png', dpi=300, bbox_inches='tight')
    print("\nResults saved to 'bayesian_pinn_results.png'")
    plt.show()
    
    return model, X_test, mean, std, y_test


def example_calibration_analysis():
    """Example 2: Calibration analysis."""
    print("\n" + "="*70)
    print("Example 2: Calibration Analysis")
    print("="*70)
    
    # Train model (reuse from example 1 or train new one)
    print("\nTraining model for calibration analysis...")
    X_train, y_train = generate_noisy_data(n_samples=100)
    X_test = torch.linspace(0, 2 * np.pi, 200).unsqueeze(-1)
    y_test = torch.sin(X_test)
    
    model = BayesianPINN(input_dim=1, hidden_dims=[32, 32], output_dim=1)
    vi = VariationalInference(model, n_data=len(X_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Quick training
    for epoch in range(300):
        optimizer.zero_grad()
        predictions, kl_divergence = model(X_train)
        total_loss, _, _ = vi.elbo_loss(predictions, y_train, kl_divergence)
        total_loss.backward()
        optimizer.step()
    
    # Predict
    mean, std, samples = model.predict_with_uncertainty(X_test, n_samples=100)
    
    # Compute calibration metrics
    print("\nComputing calibration metrics...")
    metrics = CalibrationMetrics(n_bins=10)
    
    ece = metrics.expected_calibration_error(
        mean.flatten(), std.flatten(), y_test.flatten()
    )
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    
    coverage, below, above = metrics.prediction_interval_coverage(
        mean.flatten(), std.flatten(), y_test.flatten(), confidence=0.95
    )
    print(f"\nPrediction Interval Coverage:")
    print(f"  95% CI Coverage: {coverage:.2%}")
    print(f"  Below: {below:.2%}")
    print(f"  Above: {above:.2%}")
    
    sharpness = metrics.sharpness(std.flatten())
    print(f"\nSharpness (avg uncertainty): {sharpness:.4f}")
    
    crps = metrics.continuous_ranked_probability_score(
        samples.squeeze(-1), y_test.flatten()
    )
    print(f"CRPS: {crps:.4f}")
    
    # Plot calibration curve
    print("\nPlotting calibration curve...")
    plot_calibration_curve(
        mean.flatten(),
        std.flatten(),
        y_test.flatten(),
        n_bins=10,
        save_path='calibration_curve.png'
    )
    
    # Plot prediction intervals
    print("Plotting prediction intervals...")
    plot_prediction_intervals(
        X_test.numpy().flatten(),
        mean.flatten(),
        std.flatten(),
        targets=y_test.flatten(),
        confidence=0.95,
        save_path='prediction_intervals.png'
    )


def example_comparison():
    """Example 3: Compare Bayesian vs Deterministic."""
    print("\n" + "="*70)
    print("Example 3: Bayesian vs Deterministic Comparison")
    print("="*70)
    
    # Generate data with outliers
    X_train, y_train = generate_noisy_data(n_samples=80)
    # Add outliers
    X_outliers = torch.tensor([[1.0], [4.0], [5.5]])
    y_outliers = torch.tensor([[2.5], [-2.5], [2.0]])
    X_train = torch.cat([X_train, X_outliers])
    y_train = torch.cat([y_train, y_outliers])
    
    X_test = torch.linspace(0, 2 * np.pi, 200).unsqueeze(-1)
    y_test = torch.sin(X_test)
    
    # Train both models
    print("\nTraining Bayesian model...")
    bayesian_model = BayesianPINN(input_dim=1, hidden_dims=[32, 32], output_dim=1)
    vi = VariationalInference(bayesian_model, n_data=len(X_train))
    optimizer_bayesian = torch.optim.Adam(bayesian_model.parameters(), lr=1e-3)
    
    for epoch in range(300):
        optimizer_bayesian.zero_grad()
        predictions, kl = bayesian_model(X_train)
        loss, _, _ = vi.elbo_loss(predictions, y_train, kl)
        loss.backward()
        optimizer_bayesian.step()
    
    print("Training deterministic model...")
    from torch import nn
    deterministic_model = nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )
    optimizer_det = torch.optim.Adam(deterministic_model.parameters(), lr=1e-3)
    
    for epoch in range(300):
        optimizer_det.zero_grad()
        predictions = deterministic_model(X_train)
        loss = nn.functional.mse_loss(predictions, y_train)
        loss.backward()
        optimizer_det.step()
    
    # Compare predictions
    mean_bayesian, std_bayesian, _ = bayesian_model.predict_with_uncertainty(
        X_test, n_samples=100
    )
    
    deterministic_model.eval()
    with torch.no_grad():
        pred_deterministic = deterministic_model(X_test)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(X_test.numpy(), y_test.numpy(), 'r--', label='True', linewidth=2)
    plt.plot(X_test.numpy(), mean_bayesian.numpy(), 'b-', label='Bayesian', linewidth=2)
    plt.fill_between(
        X_test.numpy().flatten(),
        (mean_bayesian - 2*std_bayesian).numpy().flatten(),
        (mean_bayesian + 2*std_bayesian).numpy().flatten(),
        alpha=0.3
    )
    plt.scatter(X_train[:80].numpy(), y_train[:80].numpy(), c='gray', alpha=0.5, label='Normal')
    plt.scatter(X_train[80:].numpy(), y_train[80:].numpy(), c='red', s=100, marker='x', label='Outliers')
    plt.title('Bayesian Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(X_test.numpy(), y_test.numpy(), 'r--', label='True', linewidth=2)
    plt.plot(X_test.numpy(), pred_deterministic.numpy(), 'g-', label='Deterministic', linewidth=2)
    plt.scatter(X_train[:80].numpy(), y_train[:80].numpy(), c='gray', alpha=0.5, label='Normal')
    plt.scatter(X_train[80:].numpy(), y_train[80:].numpy(), c='red', s=100, marker='x', label='Outliers')
    plt.title('Deterministic Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(X_test.numpy(), std_bayesian.numpy(), 'b-', linewidth=2)
    plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Outlier regions')
    plt.axvline(x=4.0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=5.5, color='r', linestyle='--', alpha=0.5)
    plt.title('Bayesian Uncertainty')
    plt.xlabel('x')
    plt.ylabel('σ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_vs_deterministic.png', dpi=300, bbox_inches='tight')
    print("\nComparison saved to 'bayesian_vs_deterministic.png'")
    plt.show()


if __name__ == "__main__":
    print("\nBayesian Neural Networks - Uncertainty Quantification Examples")
    print("="*70)
    
    # Run examples
    example_bayesian_training()
    example_calibration_analysis()
    example_comparison()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
