"""Bayesian uncertainty quantification for PINNs.

Implements:
- Monte Carlo Dropout
- Variational Inference
- Ensemble methods
- Epistemic and aleatoric uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class MCDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.
    
    Keeps dropout active during inference to sample from
    approximate posterior distribution.
    
    Args:
        model: Base neural network
        dropout_rate: Dropout probability
        n_samples: Number of MC samples for prediction
        
    Example:
        >>> base_model = FNO1d(modes=12, width=32)
        >>> mc_model = MCDropout(base_model, dropout_rate=0.1, n_samples=50)
        >>> mean, std = mc_model.predict_with_uncertainty(x_test)
    """
    
    def __init__(self, model: nn.Module, dropout_rate: float = 0.1, n_samples: int = 50):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        
        # Add dropout layers
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Add dropout after each linear/conv layer."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Wrap with dropout
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                    setattr(parent, name.split('.')[-1], 
                           nn.Sequential(module, nn.Dropout(self.dropout_rate)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass."""
        return self.model(x)
    
    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            mean: Predictive mean
            std: Predictive standard deviation (epistemic uncertainty)
        """
        self.train()  # Keep dropout active
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
    
    def predict_quantiles(
        self, x: torch.Tensor, quantiles: List[float] = [0.05, 0.95]
    ) -> List[torch.Tensor]:
        """Predict with credible intervals.
        
        Args:
            x: Input tensor
            quantiles: Quantile levels (e.g., [0.05, 0.95] for 90% CI)
            
        Returns:
            List of quantile predictions
        """
        self.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Compute quantiles
        quantile_preds = []
        for q in quantiles:
            quantile_preds.append(torch.quantile(predictions, q, dim=0))
        
        return quantile_preds


class BayesianPINN(nn.Module):
    """Bayesian Physics-Informed Neural Network.
    
    Uses variational inference to learn posterior distribution
    over network weights.
    
    Benefits:
    - Principled uncertainty quantification
    - Regularization through prior
    - Calibrated confidence intervals
    
    Args:
        input_dim: Input dimensionality
        hidden_dims: List of hidden layer sizes
        output_dim: Output dimensionality
        prior_std: Prior standard deviation for weights
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_std: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        
        # Build variational layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_std))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(BayesianLinear(prev_dim, output_dim, prior_std))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        kl = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        return kl
    
    def elbo_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics_loss: torch.Tensor,
        n_samples: int = 1,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Evidence Lower Bound (ELBO) loss.
        
        Args:
            x: Input
            y: Target
            physics_loss: PDE residual loss
            n_samples: Number of MC samples
            beta: KL divergence weight
            
        Returns:
            Negative ELBO (to minimize)
        """
        # Data likelihood
        pred = self.forward(x)
        data_loss = F.mse_loss(pred, y)
        
        # KL divergence
        kl = self.kl_divergence() / n_samples
        
        # ELBO = -log p(y|x) + β * KL[q(w)||p(w)] + physics_loss
        return data_loss + beta * kl + physics_loss


class BayesianLinear(nn.Module):
    """Bayesian linear layer with Gaussian weights.
    
    Learns mean and log-variance of weight distribution.
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log-variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 3)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features) - 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling."""
        # Sample weights from posterior
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """KL divergence between posterior q(w) and prior p(w)."""
        # KL[q(w)||p(w)] for Gaussian distributions
        kl_weight = 0.5 * (
            self.weight_logvar.exp() / (self.prior_std ** 2)
            + (self.weight_mu ** 2) / (self.prior_std ** 2)
            - 1
            - self.weight_logvar
            + 2 * np.log(self.prior_std)
        ).sum()
        
        kl_bias = 0.5 * (
            self.bias_logvar.exp() / (self.prior_std ** 2)
            + (self.bias_mu ** 2) / (self.prior_std ** 2)
            - 1
            - self.bias_logvar
            + 2 * np.log(self.prior_std)
        ).sum()
        
        return kl_weight + kl_bias
