"""Bayesian Neural Networks for Uncertainty Quantification.

Implements:
- Variational Inference for weight uncertainty
- Bayes by Backprop algorithm
- KL divergence regularization
- Posterior sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional
import math


class BayesianLinear(nn.Module):
    """Bayesian Linear Layer with weight uncertainty.
    
    Uses variational inference to learn weight distributions:
    - Prior: N(0, σ²)
    - Posterior: N(μ, σ²) learned via backprop
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        prior_std: Prior standard deviation
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize means with Xavier
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)
        
        # Initialize log sigmas (small values)
        self.weight_log_sigma.data.fill_(-5.0)
        self.bias_log_sigma.data.fill_(-5.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with reparameterization trick.
        
        Args:
            x: Input tensor [batch, in_features]
            
        Returns:
            output: Linear transformation [batch, out_features]
            kl_divergence: KL divergence for this layer
        """
        # Sample weights using reparameterization trick
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_eps
        
        # Sample bias
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias_eps = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * bias_eps
        
        # Linear transformation
        output = F.linear(x, weight, bias)
        
        # Compute KL divergence
        kl = self._kl_divergence()
        
        return output, kl
    
    def _kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior.
        
        KL(q(w|θ) || p(w)) where:
        - q(w|θ): Posterior N(μ, σ²)
        - p(w): Prior N(0, prior_std²)
        """
        # Weight KL
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_sigma ** 2) / (self.prior_std ** 2)
            - 1.0
            - 2 * self.weight_log_sigma
            + 2 * math.log(self.prior_std)
        )
        
        # Bias KL
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_sigma ** 2) / (self.prior_std ** 2)
            - 1.0
            - 2 * self.bias_log_sigma
            + 2 * math.log(self.prior_std)
        )
        
        return weight_kl + bias_kl


class BayesianPINN(nn.Module):
    """Bayesian Physics-Informed Neural Network.
    
    Extends PINN with Bayesian layers for uncertainty quantification.
    
    Features:
    - Weight uncertainty via variational inference
    - Aleatoric uncertainty (data noise)
    - Epistemic uncertainty (model uncertainty)
    - Prediction intervals
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        prior_std: Prior standard deviation for weights
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        prior_std: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build Bayesian layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(BayesianLinear(dims[i], dims[i + 1], prior_std))
        
        self.layers = nn.ModuleList(layers)
        self.activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with KL divergence.
        
        Args:
            x: Input tensor
            
        Returns:
            output: Predictions
            kl_divergence: Total KL divergence
        """
        total_kl = 0.0
        
        for i, layer in enumerate(self.layers):
            x, kl = layer(x)
            total_kl += kl
            
            # Activation except last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x, total_kl
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation.
        
        Args:
            x: Input tensor [batch, input_dim]
            n_samples: Number of Monte Carlo samples
            
        Returns:
            mean: Mean prediction [batch, output_dim]
            std: Standard deviation [batch, output_dim]
            samples: All samples [n_samples, batch, output_dim]
        """
        self.train()  # Enable sampling
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred, _ = self.forward(x)
                samples.append(pred)
        
        samples = torch.stack(samples)  # [n_samples, batch, output_dim]
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        return mean, std, samples


class BayesianFNO(nn.Module):
    """Bayesian Fourier Neural Operator.
    
    Combines FNO with Bayesian inference for uncertainty quantification.
    
    Args:
        modes: Number of Fourier modes
        width: Channel width
        n_layers: Number of FNO layers
        prior_std: Prior standard deviation
    """
    
    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        prior_std: float = 1.0,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        
        # Lifting layer (deterministic)
        self.lift = nn.Linear(1, width)
        
        # Bayesian spectral layers
        self.spectral_layers = nn.ModuleList([
            BayesianLinear(width, width, prior_std)
            for _ in range(n_layers)
        ])
        
        # Projection layer (Bayesian)
        self.project = BayesianLinear(width, 1, prior_std)
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input [batch, n_points, 1]
            
        Returns:
            output: Predictions [batch, n_points, 1]
            kl_divergence: Total KL divergence
        """
        # Lift
        x = self.lift(x)  # [batch, n_points, width]
        
        total_kl = 0.0
        
        # Spectral layers (simplified - real FNO uses FFT)
        for layer in self.spectral_layers:
            # Reshape for linear layer
            batch, n_points, channels = x.shape
            x_flat = x.reshape(batch * n_points, channels)
            
            # Apply Bayesian layer
            x_flat, kl = layer(x_flat)
            total_kl += kl
            
            # Reshape back
            x = x_flat.reshape(batch, n_points, channels)
            x = self.activation(x)
        
        # Project
        batch, n_points, channels = x.shape
        x_flat = x.reshape(batch * n_points, channels)
        x_flat, kl = self.project(x_flat)
        total_kl += kl
        
        output = x_flat.reshape(batch, n_points, 1)
        
        return output, total_kl
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with uncertainty."""
        self.train()  # Enable sampling
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred, _ = self.forward(x)
                samples.append(pred)
        
        samples = torch.stack(samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        return mean, std, samples


class VariationalInference:
    """Variational Inference trainer for Bayesian networks.
    
    Implements ELBO (Evidence Lower Bound) optimization:
    ELBO = E[log p(y|x,w)] - KL(q(w|θ) || p(w))
    
    Args:
        model: Bayesian model
        n_data: Number of training samples (for KL weighting)
    """
    
    def __init__(self, model: nn.Module, n_data: int):
        self.model = model
        self.n_data = n_data
        self.kl_weight = 1.0 / n_data  # Weighting for minibatch
    
    def elbo_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        kl_divergence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            kl_divergence: KL divergence from forward pass
            
        Returns:
            total_loss: ELBO loss (to minimize)
            nll: Negative log likelihood
            kl: Weighted KL divergence
        """
        # Negative log likelihood (reconstruction loss)
        nll = F.mse_loss(predictions, targets, reduction='sum')
        
        # Weighted KL divergence
        kl = self.kl_weight * kl_divergence
        
        # ELBO = NLL + KL
        total_loss = nll + kl
        
        return total_loss, nll, kl
