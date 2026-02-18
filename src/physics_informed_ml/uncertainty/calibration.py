"""Calibration metrics for uncertainty quantification.

Implements:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagrams
- Sharpness metrics
- Prediction intervals
"""

import torch
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class CalibrationMetrics:
    """Compute calibration metrics for uncertainty estimates.
    
    A well-calibrated model should have:
    - Low ECE (< 0.05)
    - Prediction intervals that contain true values at expected rate
    - Sharp (small) prediction intervals
    
    Example:
        >>> metrics = CalibrationMetrics()
        >>> ece = metrics.expected_calibration_error(
        ...     predictions=pred_mean,
        ...     uncertainties=pred_std,
        ...     targets=y_true
        ... )
        >>> print(f"ECE: {ece:.4f}")
    """
    
    def __init__(self, n_bins: int = 10):
        """Initialize calibration metrics.
        
        Args:
            n_bins: Number of bins for calibration curve
        """
        self.n_bins = n_bins
    
    def expected_calibration_error(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute Expected Calibration Error (ECE).
        
        Measures average difference between confidence and accuracy.
        
        Args:
            predictions: Model predictions [N]
            uncertainties: Prediction uncertainties (std) [N]
            targets: Ground truth [N]
            
        Returns:
            ECE value (0 = perfect calibration)
        """
        # Compute errors
        errors = torch.abs(predictions - targets)
        
        # Compute confidence (inverse of uncertainty)
        confidences = 1.0 / (uncertainties + 1e-8)
        confidences = (confidences - confidences.min()) / (
            confidences.max() - confidences.min() + 1e-8
        )
        
        # Bin by confidence
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        
        ece = 0.0
        n_total = len(predictions)
        
        for i in range(self.n_bins):
            # Find samples in this bin
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > lower) & (confidences <= upper)
            n_bin = in_bin.sum().item()
            
            if n_bin > 0:
                # Average confidence in bin
                avg_confidence = confidences[in_bin].mean().item()
                
                # Average accuracy in bin (1 - normalized error)
                avg_error = errors[in_bin].mean().item()
                max_error = errors.max().item()
                avg_accuracy = 1.0 - (avg_error / (max_error + 1e-8))
                
                # Weighted contribution to ECE
                ece += (n_bin / n_total) * abs(avg_confidence - avg_accuracy)
        
        return ece
    
    def prediction_interval_coverage(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Compute prediction interval coverage.
        
        A well-calibrated model should have coverage ≈ confidence level.
        
        Args:
            predictions: Mean predictions [N]
            uncertainties: Standard deviations [N]
            targets: Ground truth [N]
            confidence: Desired confidence level (e.g., 0.95)
            
        Returns:
            coverage: Actual coverage rate
            lower_coverage: Coverage below interval
            upper_coverage: Coverage above interval
        """
        # Compute confidence interval (assuming normal distribution)
        z_score = torch.distributions.Normal(0, 1).icdf(
            torch.tensor(1 - (1 - confidence) / 2)
        )
        
        lower = predictions - z_score * uncertainties
        upper = predictions + z_score * uncertainties
        
        # Check coverage
        in_interval = (targets >= lower) & (targets <= upper)
        coverage = in_interval.float().mean().item()
        
        # Asymmetric coverage
        below = (targets < lower).float().mean().item()
        above = (targets > upper).float().mean().item()
        
        return coverage, below, above
    
    def sharpness(
        self,
        uncertainties: torch.Tensor,
    ) -> float:
        """Compute sharpness (average uncertainty).
        
        Lower is better (sharper predictions).
        
        Args:
            uncertainties: Prediction uncertainties [N]
            
        Returns:
            Average uncertainty
        """
        return uncertainties.mean().item()
    
    def continuous_ranked_probability_score(
        self,
        samples: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute Continuous Ranked Probability Score (CRPS).
        
        Proper scoring rule for probabilistic predictions.
        
        Args:
            samples: Prediction samples [n_samples, N]
            targets: Ground truth [N]
            
        Returns:
            CRPS (lower is better)
        """
        n_samples = samples.shape[0]
        
        # Term 1: Average distance to target
        term1 = torch.abs(samples - targets.unsqueeze(0)).mean()
        
        # Term 2: Average pairwise distance (measures spread)
        # Only compute for subset to save memory
        if n_samples > 100:
            idx = torch.randperm(n_samples)[:100]
            samples_subset = samples[idx]
        else:
            samples_subset = samples
        
        pairwise = torch.abs(
            samples_subset.unsqueeze(0) - samples_subset.unsqueeze(1)
        )
        term2 = pairwise.mean() / 2
        
        crps = term1 - term2
        return crps.item()


def plot_calibration_curve(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
    save_path: Optional[str] = None,
):
    """Plot reliability diagram (calibration curve).
    
    Args:
        predictions: Model predictions
        uncertainties: Prediction uncertainties
        targets: Ground truth
        n_bins: Number of bins
        save_path: Path to save figure
    """
    metrics = CalibrationMetrics(n_bins=n_bins)
    
    # Compute ECE
    ece = metrics.expected_calibration_error(predictions, uncertainties, targets)
    
    # Compute bin statistics
    errors = torch.abs(predictions - targets)
    confidences = 1.0 / (uncertainties + 1e-8)
    confidences = (confidences - confidences.min()) / (
        confidences.max() - confidences.min() + 1e-8
    )
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > lower) & (confidences <= upper)
        n_bin = in_bin.sum().item()
        
        if n_bin > 0:
            avg_conf = confidences[in_bin].mean().item()
            avg_error = errors[in_bin].mean().item()
            max_error = errors.max().item()
            avg_acc = 1.0 - (avg_error / (max_error + 1e-8))
            
            bin_confidences.append(avg_conf)
            bin_accuracies.append(avg_acc)
            bin_counts.append(n_bin)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(bin_confidences, bin_accuracies, 'o-', label=f'Model (ECE={ece:.4f})')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.bar(range(len(bin_counts)), bin_counts, alpha=0.7)
    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('Count')
    ax2.set_title('Sample Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to {save_path}")
    else:
        plt.show()


def plot_prediction_intervals(
    x: np.ndarray,
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    confidence: float = 0.95,
    save_path: Optional[str] = None,
):
    """Plot predictions with uncertainty intervals.
    
    Args:
        x: Input coordinates (for x-axis)
        predictions: Mean predictions
        uncertainties: Standard deviations
        targets: Ground truth (optional)
        confidence: Confidence level
        save_path: Path to save figure
    """
    predictions = predictions.cpu().numpy()
    uncertainties = uncertainties.cpu().numpy()
    
    # Compute confidence interval
    from scipy.stats import norm
    z_score = norm.ppf(1 - (1 - confidence) / 2)
    
    lower = predictions - z_score * uncertainties
    upper = predictions + z_score * uncertainties
    
    plt.figure(figsize=(12, 6))
    
    # Prediction with uncertainty
    plt.plot(x, predictions, 'b-', label='Prediction', linewidth=2)
    plt.fill_between(
        x, lower.flatten(), upper.flatten(),
        alpha=0.3, label=f'{int(confidence*100)}% CI'
    )
    
    # Ground truth
    if targets is not None:
        targets = targets.cpu().numpy()
        plt.plot(x, targets, 'r--', label='Ground Truth', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Predictions with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction intervals saved to {save_path}")
    else:
        plt.show()
