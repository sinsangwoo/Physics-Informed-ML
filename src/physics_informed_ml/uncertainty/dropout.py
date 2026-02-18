"""Monte Carlo Dropout for Uncertainty Quantification.

Implements:
- Dropout as Bayesian approximation
- Uncertainty estimation via multiple forward passes
- Efficient sampling
"""

import torch
import torch.nn as nn
from typing import Tuple


class MCDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.
    
    Treats dropout as Bayesian approximation:
    - Training: Standard dropout
    - Inference: Keep dropout active, sample multiple times
    - Uncertainty: Variance across samples
    
    Benefits:
    - No architecture changes
    - Single model training
    - Fast inference
    
    Args:
        model: Base model
        dropout_rate: Dropout probability
    
    Example:
        >>> from physics_informed_ml.models import FNO1d
        >>> base_model = FNO1d(modes=12, width=32)
        >>> mc_model = MCDropout(base_model, dropout_rate=0.1)
        >>> mean, std = mc_model.predict_with_uncertainty(x_test, n_samples=50)
    """
    
    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        
        # Add dropout layers after each activation
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Recursively add dropout layers to model."""
        for name, module in self.model.named_children():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
                # Add dropout after activation
                setattr(
                    self.model,
                    name,
                    nn.Sequential(
                        module,
                        nn.Dropout(self.dropout_rate)
                    )
                )
            elif len(list(module.children())) > 0:
                # Recursively process nested modules
                self._add_dropout_layers_recursive(module)
    
    def _add_dropout_layers_recursive(self, module: nn.Module):
        """Helper for recursive dropout addition."""
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
                setattr(
                    module,
                    name,
                    nn.Sequential(
                        child,
                        nn.Dropout(self.dropout_rate)
                    )
                )
            elif len(list(child.children())) > 0:
                self._add_dropout_layers_recursive(child)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (delegates to base model)."""
        return self.model(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with MC Dropout uncertainty.
        
        Args:
            x: Input tensor [batch, ...]
            n_samples: Number of forward passes
            
        Returns:
            mean: Mean prediction [batch, ...]
            std: Standard deviation [batch, ...]
            samples: All predictions [n_samples, batch, ...]
        """
        # Enable dropout during inference
        self.model.train()
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                samples.append(pred)
        
        samples = torch.stack(samples)  # [n_samples, batch, ...]
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        return mean, std, samples
    
    def enable_dropout(self):
        """Enable dropout layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_dropout(self):
        """Disable dropout layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
