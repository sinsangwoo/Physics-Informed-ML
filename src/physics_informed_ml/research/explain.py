"""Model explainability and interpretability for PINNs.

Tools:
- SHAP (SHapley Additive exPlanations)
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Sensitivity analysis
- Feature importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np


class SHAPExplainer:
    """SHAP-based explainability for neural operators.
    
    Computes Shapley values to explain model predictions:
    - Feature importance
    - Local explanations
    - Interaction effects
    
    Example:
        >>> explainer = SHAPExplainer(model)
        >>> shap_values = explainer.explain(x_test, background_data)
        >>> explainer.plot_importance(shap_values)
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 100):
        """Initialize SHAP explainer.
        
        Args:
            model: Model to explain
            n_samples: Number of background samples for approximation
        """
        self.model = model
        self.n_samples = n_samples
        self.model.eval()
    
    def explain(
        self,
        x: torch.Tensor,
        background: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SHAP values for input.
        
        Args:
            x: Input to explain (batch_size, ...)
            background: Background dataset for baseline
            
        Returns:
            SHAP values (same shape as x)
        """
        batch_size = x.shape[0]
        shap_values = torch.zeros_like(x)
        
        with torch.no_grad():
            # Base prediction on background
            base_pred = self.model(background).mean(dim=0)
            
            # Compute SHAP values
            for i in range(batch_size):
                x_i = x[i:i+1]
                
                # Sample coalitions
                for _ in range(self.n_samples):
                    # Random feature mask
                    mask = torch.rand_like(x_i) > 0.5
                    
                    # Combine input and background
                    x_masked = torch.where(
                        mask,
                        x_i,
                        background[torch.randint(len(background), (1,))]
                    )
                    
                    # Marginal contribution
                    pred_masked = self.model(x_masked)
                    contribution = pred_masked - base_pred
                    
                    # Accumulate SHAP values
                    shap_values[i] += contribution.squeeze() * mask.float().squeeze()
                
                shap_values[i] /= self.n_samples
        
        return shap_values
    
    def feature_importance(
        self,
        x: torch.Tensor,
        background: torch.Tensor,
    ) -> torch.Tensor:
        """Compute global feature importance.
        
        Args:
            x: Dataset to explain
            background: Background dataset
            
        Returns:
            Feature importance scores
        """
        shap_values = self.explain(x, background)
        importance = shap_values.abs().mean(dim=0)
        return importance


class GradCAM:
    """Gradient-weighted Class Activation Mapping.
    
    Visualizes which input regions are important for predictions.
    Particularly useful for spatial data (2D/3D PDEs).
    
    Example:
        >>> gradcam = GradCAM(model, target_layer='conv_final')
        >>> heatmap = gradcam.generate_heatmap(x, target_output)
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """Initialize Grad-CAM.
        
        Args:
            model: Model to explain
            target_layer: Layer name to compute activations
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def generate_heatmap(
        self,
        x: torch.Tensor,
        target_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate Grad-CAM heatmap.
        
        Args:
            x: Input
            target_output: Target output for gradient computation
                          If None, uses model prediction
            
        Returns:
            Heatmap (batch_size, spatial_dims...)
        """
        self.model.eval()
        x.requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        
        # Compute gradients
        if target_output is None:
            target_output = output
        
        self.model.zero_grad()
        output.backward(gradient=torch.ones_like(target_output))
        
        # Compute Grad-CAM
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3], keepdim=True)
        
        # Weight activations by gradients
        weighted_activations = self.activations * pooled_gradients
        
        # Aggregate across channels
        heatmap = weighted_activations.sum(dim=1)
        
        # ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap


class SensitivityAnalysis:
    """Sensitivity analysis for physics parameters.
    
    Analyzes how model predictions change with:
    - Input perturbations
    - Parameter variations
    - Boundary condition changes
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    def input_sensitivity(
        self,
        x: torch.Tensor,
        epsilon: float = 0.01,
    ) -> torch.Tensor:
        """Compute input sensitivity (Jacobian norm).
        
        Args:
            x: Input
            epsilon: Perturbation magnitude
            
        Returns:
            Sensitivity map
        """
        x.requires_grad_(True)
        
        output = self.model(x)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
        )[0]
        
        # Sensitivity = gradient norm
        sensitivity = gradients.norm(dim=-1)
        
        return sensitivity
    
    def parameter_sweep(
        self,
        x: torch.Tensor,
        param_name: str,
        param_range: np.ndarray,
    ) -> Dict[float, torch.Tensor]:
        """Sweep physics parameter and record predictions.
        
        Args:
            x: Input
            param_name: Parameter to vary
            param_range: Array of parameter values
            
        Returns:
            Dictionary mapping param_value -> prediction
        """
        results = {}
        
        with torch.no_grad():
            for param_value in param_range:
                # Set parameter (implementation depends on model)
                # This is a placeholder - actual implementation needs
                # model-specific parameter setting
                
                pred = self.model(x)
                results[float(param_value)] = pred.cpu()
        
        return results
    
    def uncertainty_propagation(
        self,
        x: torch.Tensor,
        x_std: torch.Tensor,
        n_samples: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate input uncertainty through model.
        
        Args:
            x: Mean input
            x_std: Input standard deviation
            n_samples: Number of Monte Carlo samples
            
        Returns:
            (output_mean, output_std)
        """
        outputs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from input distribution
                x_sample = x + x_std * torch.randn_like(x)
                
                # Forward pass
                output = self.model(x_sample)
                outputs.append(output)
        
        outputs = torch.stack(outputs)
        output_mean = outputs.mean(dim=0)
        output_std = outputs.std(dim=0)
        
        return output_mean, output_std
