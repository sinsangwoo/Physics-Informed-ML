"""Deep Ensemble for Uncertainty Quantification.

Implements:
- Training multiple independent models
- Ensemble prediction aggregation
- Disagreement-based uncertainty
- Adversarial training for diversity
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Callable
import copy


class DeepEnsemble:
    """Deep Ensemble of neural networks.
    
    Trains N independent models with different initializations.
    Uncertainty is estimated from ensemble disagreement.
    
    Benefits:
    - Simple to implement
    - No architecture changes needed
    - Often outperforms Bayesian methods
    - Parallelizable training
    
    Args:
        base_model: Model class or factory function
        n_models: Number of ensemble members
        device: Device for training
    
    Example:
        >>> from physics_informed_ml.models import FNO1d
        >>> ensemble = DeepEnsemble(
        ...     base_model=lambda: FNO1d(modes=12, width=32),
        ...     n_models=5
        ... )
        >>> ensemble.train(train_loader, epochs=100)
        >>> mean, std = ensemble.predict(x_test)
    """
    
    def __init__(
        self,
        base_model: Callable[[], nn.Module],
        n_models: int = 5,
        device: Optional[str] = None,
    ):
        self.base_model = base_model
        self.n_models = n_models
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize ensemble members
        self.models = []
        for i in range(n_models):
            model = base_model()
            model.to(self.device)
            self.models.append(model)
            print(f"Initialized ensemble member {i+1}/{n_models}")
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        lr: float = 1e-3,
        loss_fn: Optional[Callable] = None,
        verbose: bool = True,
    ):
        """Train all ensemble members.
        
        Args:
            train_loader: Training data loader
            epochs: Number of epochs per model
            lr: Learning rate
            loss_fn: Loss function (default: MSE)
            verbose: Print training progress
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        for i, model in enumerate(self.models):
            if verbose:
                print(f"\nTraining ensemble member {i+1}/{self.n_models}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                n_batches = 0
                
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pred = model(X_batch)
                    loss = loss_fn(pred, y_batch)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / n_batches
                    print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    def predict(
        self,
        x: torch.Tensor,
        return_individuals: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Predict with uncertainty estimation.
        
        Args:
            x: Input tensor [batch, ...]
            return_individuals: Return individual model predictions
            
        Returns:
            mean: Ensemble mean [batch, ...]
            std: Ensemble standard deviation [batch, ...]
            individuals: Individual predictions (if requested) [n_models, batch, ...]
        """
        x = x.to(self.device)
        
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_models, batch, ...]
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        if return_individuals:
            return mean, std, predictions
        else:
            return mean, std, None
    
    def save(self, path: str):
        """Save ensemble models."""
        torch.save({
            'n_models': self.n_models,
            'models': [model.state_dict() for model in self.models],
        }, path)
        print(f"Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load ensemble models."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if checkpoint['n_models'] != self.n_models:
            raise ValueError(
                f"Checkpoint has {checkpoint['n_models']} models, "
                f"but ensemble has {self.n_models}"
            )
        
        for model, state_dict in zip(self.models, checkpoint['models']):
            model.load_state_dict(state_dict)
        
        print(f"Ensemble loaded from {path}")


class EnsemblePredictor:
    """Utility for making predictions with trained ensemble.
    
    Provides:
    - Batch prediction
    - Confidence intervals
    - Outlier detection
    - Calibrated uncertainty
    """
    
    def __init__(self, ensemble: DeepEnsemble):
        self.ensemble = ensemble
    
    def predict_with_confidence(
        self,
        x: torch.Tensor,
        confidence: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with confidence intervals.
        
        Args:
            x: Input tensor
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            mean: Mean prediction
            lower: Lower confidence bound
            upper: Upper confidence bound
        """
        mean, std, samples = self.ensemble.predict(x, return_individuals=True)
        
        # Compute percentiles for confidence interval
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = torch.quantile(samples, lower_percentile / 100, dim=0)
        upper = torch.quantile(samples, upper_percentile / 100, dim=0)
        
        return mean, lower, upper
    
    def detect_outliers(
        self,
        x: torch.Tensor,
        threshold: float = 3.0,
    ) -> torch.Tensor:
        """Detect outliers based on prediction uncertainty.
        
        Args:
            x: Input tensor
            threshold: Standard deviation threshold
            
        Returns:
            outlier_mask: Boolean mask [batch]
        """
        mean, std, _ = self.ensemble.predict(x, return_individuals=False)
        
        # Normalized uncertainty
        normalized_std = std / (mean.abs() + 1e-8)
        
        # Points with high relative uncertainty are outliers
        outlier_mask = normalized_std.mean(dim=-1) > threshold
        
        return outlier_mask
