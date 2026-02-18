"""Example: Deep Ensemble for Uncertainty Quantification.

Demonstrates:
- Training ensemble of neural networks
- Ensemble prediction
- Uncertainty decomposition
- Outlier detection
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from physics_informed_ml.uncertainty import DeepEnsemble, EnsemblePredictor


def create_simple_model():
    """Factory function for ensemble members."""
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )


def generate_data(n_samples=100):
    """Generate training data."""
    x = torch.linspace(0, 2 * np.pi, n_samples).unsqueeze(-1)
    y = torch.sin(x) + torch.randn_like(x) * 0.1
    return x, y


def example_ensemble_training():
    """Example 1: Train deep ensemble."""
    print("="*70)
    print("Example 1: Deep Ensemble Training")
    print("="*70)
    
    # Generate data
    X_train, y_train = generate_data(n_samples=100)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Create ensemble
    print("\nCreating ensemble with 5 members...")
    ensemble = DeepEnsemble(
        base_model=create_simple_model,
        n_models=5,
        device="cpu"
    )
    
    # Train ensemble
    print("\nTraining ensemble...")
    ensemble.train(
        train_loader=train_loader,
        epochs=200,
        lr=1e-3,
        verbose=True
    )
    
    # Test data
    X_test = torch.linspace(0, 2 * np.pi, 200).unsqueeze(-1)
    y_test = torch.sin(X_test)
    
    # Predict
    print("\nPredicting with ensemble...")
    mean, std, individuals = ensemble.predict(X_test, return_individuals=True)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Individual models
    plt.subplot(1, 3, 1)
    for i, pred in enumerate(individuals):
        plt.plot(X_test.numpy(), pred.numpy(), alpha=0.3, label=f'Model {i+1}')
    plt.plot(X_test.numpy(), y_test.numpy(), 'r--', linewidth=2, label='True')
    plt.scatter(X_train.numpy(), y_train.numpy(), c='gray', alpha=0.5, s=10)
    plt.title('Individual Ensemble Members')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Ensemble mean
    plt.subplot(1, 3, 2)
    plt.plot(X_test.numpy(), y_test.numpy(), 'r--', linewidth=2, label='True')
    plt.plot(X_test.numpy(), mean.numpy(), 'b-', linewidth=2, label='Ensemble Mean')
    plt.fill_between(
        X_test.numpy().flatten(),
        (mean - 2*std).numpy().flatten(),
        (mean + 2*std).numpy().flatten(),
        alpha=0.3,
        label='95% CI'
    )
    plt.scatter(X_train.numpy(), y_train.numpy(), c='gray', alpha=0.5, s=10)
    plt.title('Ensemble Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty
    plt.subplot(1, 3, 3)
    plt.plot(X_test.numpy(), std.numpy(), 'g-', linewidth=2)
    plt.title('Ensemble Disagreement')
    plt.xlabel('x')
    plt.ylabel('Uncertainty (σ)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deep_ensemble_results.png', dpi=300, bbox_inches='tight')
    print("\nResults saved to 'deep_ensemble_results.png'")
    plt.show()
    
    return ensemble


def example_confidence_intervals():
    """Example 2: Confidence intervals."""
    print("\n" + "="*70)
    print("Example 2: Confidence Intervals")
    print("="*70)
    
    # Quick ensemble training
    X_train, y_train = generate_data(n_samples=100)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    ensemble = DeepEnsemble(base_model=create_simple_model, n_models=5)
    ensemble.train(train_loader, epochs=200, verbose=False)
    
    # Create predictor
    predictor = EnsemblePredictor(ensemble)
    
    # Test data
    X_test = torch.linspace(0, 2 * np.pi, 200).unsqueeze(-1)
    y_test = torch.sin(X_test)
    
    # Different confidence levels
    confidence_levels = [0.68, 0.95, 0.99]
    
    plt.figure(figsize=(15, 5))
    
    for i, confidence in enumerate(confidence_levels):
        plt.subplot(1, 3, i+1)
        
        mean, lower, upper = predictor.predict_with_confidence(
            X_test, confidence=confidence
        )
        
        plt.plot(X_test.numpy(), y_test.numpy(), 'r--', linewidth=2, label='True')
        plt.plot(X_test.numpy(), mean.numpy(), 'b-', linewidth=2, label='Mean')
        plt.fill_between(
            X_test.numpy().flatten(),
            lower.numpy().flatten(),
            upper.numpy().flatten(),
            alpha=0.3,
            label=f'{int(confidence*100)}% CI'
        )
        
        plt.title(f'{int(confidence*100)}% Confidence Interval')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confidence_intervals.png', dpi=300, bbox_inches='tight')
    print("\nConfidence intervals saved to 'confidence_intervals.png'")
    plt.show()


def example_outlier_detection():
    """Example 3: Outlier detection."""
    print("\n" + "="*70)
    print("Example 3: Outlier Detection")
    print("="*70)
    
    # Training data (normal)
    X_train, y_train = generate_data(n_samples=100)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Train ensemble
    ensemble = DeepEnsemble(base_model=create_simple_model, n_models=5)
    ensemble.train(train_loader, epochs=200, verbose=False)
    
    predictor = EnsemblePredictor(ensemble)
    
    # Test data with outliers
    X_normal = torch.linspace(0, 2 * np.pi, 100).unsqueeze(-1)
    X_outliers = torch.tensor([[0.5], [3.0], [5.5]])  # Out-of-distribution
    X_test = torch.cat([X_normal, X_outliers])
    
    # Detect outliers
    print("\nDetecting outliers...")
    outlier_mask = predictor.detect_outliers(X_test, threshold=2.0)
    
    n_outliers = outlier_mask.sum().item()
    print(f"Detected {n_outliers} outliers out of {len(X_test)} samples")
    
    # Predict
    mean, std, _ = ensemble.predict(X_test)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(
        X_test[~outlier_mask].numpy(),
        mean[~outlier_mask].numpy(),
        c='blue',
        label='Normal',
        alpha=0.6
    )
    plt.scatter(
        X_test[outlier_mask].numpy(),
        mean[outlier_mask].numpy(),
        c='red',
        s=100,
        marker='x',
        label='Outlier',
        linewidths=3
    )
    plt.plot(X_normal.numpy(), torch.sin(X_normal).numpy(), 'k--', alpha=0.3)
    plt.title('Outlier Detection')
    plt.xlabel('x')
    plt.ylabel('Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(
        X_test[~outlier_mask].numpy(),
        std[~outlier_mask].numpy(),
        c='blue',
        label='Normal',
        alpha=0.6
    )
    plt.scatter(
        X_test[outlier_mask].numpy(),
        std[outlier_mask].numpy(),
        c='red',
        s=100,
        marker='x',
        label='Outlier',
        linewidths=3
    )
    plt.title('Uncertainty (Higher for Outliers)')
    plt.xlabel('x')
    plt.ylabel('Uncertainty (σ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
    print("\nOutlier detection saved to 'outlier_detection.png'")
    plt.show()


if __name__ == "__main__":
    print("\nDeep Ensemble - Uncertainty Quantification Examples")
    print("="*70)
    
    # Run examples
    example_ensemble_training()
    example_confidence_intervals()
    example_outlier_detection()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
