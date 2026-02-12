"""Transfer learning for physics-informed neural networks.

Enables:
- Knowledge transfer across different PDEs
- Domain adaptation for new physics
- Few-shot learning for new scenarios
- Meta-learning for rapid adaptation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


class TransferLearning:
    """Transfer learning framework for neural operators.
    
    Strategies:
    1. Feature extraction: Freeze early layers, train only final layers
    2. Fine-tuning: Gradually unfreeze and adapt all layers
    3. Progressive unfreezing: Layer-by-layer adaptation
    
    Example:
        >>> # Train on heat equation
        >>> source_model = FNO1d(...)
        >>> # ... training ...
        >>> 
        >>> # Transfer to wave equation
        >>> transfer = TransferLearning(source_model)
        >>> target_model = transfer.adapt(wave_data, strategy='fine-tune')
    """
    
    def __init__(self, source_model: nn.Module):
        """Initialize with pre-trained source model.
        
        Args:
            source_model: Pre-trained model on source task
        """
        self.source_model = source_model
    
    def freeze_layers(self, model: nn.Module, freeze_until: Optional[str] = None):
        """Freeze model layers up to a specific layer.
        
        Args:
            model: Model to freeze
            freeze_until: Layer name to freeze until (None = freeze all)
        """
        freeze = True
        for name, param in model.named_parameters():
            if freeze_until and name == freeze_until:
                freeze = False
            param.requires_grad = not freeze
    
    def feature_extraction(
        self,
        target_data: Tuple[torch.Tensor, torch.Tensor],
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> nn.Module:
        """Feature extraction: Only train final layers.
        
        Best for:
        - Similar physics, different parameters
        - Small target dataset
        - Fast adaptation
        
        Args:
            target_data: (X_target, y_target) training data
            n_epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Adapted model
        """
        model = deepcopy(self.source_model)
        
        # Freeze all except final layer
        for name, param in model.named_parameters():
            if 'output' not in name and 'final' not in name:
                param.requires_grad = False
        
        # Train only unfrozen layers
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        
        X, y = target_data
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
        
        return model
    
    def fine_tuning(
        self,
        target_data: Tuple[torch.Tensor, torch.Tensor],
        n_epochs: int = 100,
        lr: float = 1e-4,
        warmup_epochs: int = 20,
    ) -> nn.Module:
        """Fine-tuning: Adapt all layers with small learning rate.
        
        Best for:
        - Related physics with significant differences
        - Medium-sized target dataset
        - High accuracy requirements
        
        Args:
            target_data: Training data
            n_epochs: Training epochs
            lr: Learning rate (smaller than initial training)
            warmup_epochs: Epochs to train only final layer first
            
        Returns:
            Fine-tuned model
        """
        model = deepcopy(self.source_model)
        X, y = target_data
        criterion = nn.MSELoss()
        
        # Phase 1: Warmup - train only final layer
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze final layer
        for name, param in model.named_parameters():
            if 'output' in name or 'final' in name:
                param.requires_grad = True
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 10
        )
        
        print("Phase 1: Warmup (final layer only)")
        model.train()
        for epoch in range(warmup_epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        # Phase 2: Fine-tune all layers
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        print("Phase 2: Fine-tuning (all layers)")
        for epoch in range(n_epochs - warmup_epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {warmup_epochs + epoch + 1}/{n_epochs}, Loss: {loss.item():.6f}")
        
        return model
    
    def progressive_unfreezing(
        self,
        target_data: Tuple[torch.Tensor, torch.Tensor],
        layer_groups: List[List[str]],
        epochs_per_group: int = 20,
        lr: float = 1e-4,
    ) -> nn.Module:
        """Progressive unfreezing: Gradually unfreeze layer groups.
        
        Best for:
        - Very different physics
        - Large target dataset
        - Avoiding catastrophic forgetting
        
        Args:
            target_data: Training data
            layer_groups: List of layer name groups to unfreeze progressively
            epochs_per_group: Epochs to train each group
            lr: Learning rate
            
        Returns:
            Adapted model
        """
        model = deepcopy(self.source_model)
        X, y = target_data
        criterion = nn.MSELoss()
        
        # Freeze all initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Progressively unfreeze and train
        for i, group in enumerate(layer_groups):
            print(f"\nUnfreezing group {i+1}/{len(layer_groups)}: {group}")
            
            # Unfreeze current group
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in group):
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr
            )
            
            model.train()
            for epoch in range(epochs_per_group):
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs_per_group}, Loss: {loss.item():.6f}")
        
        return model


class DomainAdaptation:
    """Domain adaptation for physics problems.
    
    Adapts model trained on source domain to target domain
    with different:
    - Physical parameters (Re, Pr, etc.)
    - Boundary conditions
    - Geometries
    
    Uses adversarial domain adaptation.
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        predictor: nn.Module,
        discriminator: Optional[nn.Module] = None,
    ):
        """Initialize domain adaptation.
        
        Args:
            feature_extractor: Shared feature extraction network
            predictor: Task-specific prediction head
            discriminator: Domain classifier (created if None)
        """
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        
        if discriminator is None:
            # Simple discriminator
            self.discriminator = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.discriminator = discriminator
    
    def adapt(
        self,
        source_data: Tuple[torch.Tensor, torch.Tensor],
        target_data: torch.Tensor,  # Unlabeled target data
        n_epochs: int = 100,
        lambda_adv: float = 0.1,
    ) -> Dict[str, nn.Module]:
        """Adversarial domain adaptation.
        
        Args:
            source_data: (X_source, y_source) labeled source data
            target_data: X_target unlabeled target data
            n_epochs: Training epochs
            lambda_adv: Adversarial loss weight
            
        Returns:
            Dictionary with adapted models
        """
        X_source, y_source = source_data
        
        # Optimizers
        opt_main = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.predictor.parameters()),
            lr=1e-3
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        
        criterion_task = nn.MSELoss()
        criterion_domain = nn.BCELoss()
        
        for epoch in range(n_epochs):
            # Train discriminator
            opt_disc.zero_grad()
            
            # Source domain (label = 0)
            feat_source = self.feature_extractor(X_source)
            pred_source_domain = self.discriminator(feat_source.detach())
            loss_disc_source = criterion_domain(
                pred_source_domain,
                torch.zeros_like(pred_source_domain)
            )
            
            # Target domain (label = 1)
            feat_target = self.feature_extractor(target_data)
            pred_target_domain = self.discriminator(feat_target.detach())
            loss_disc_target = criterion_domain(
                pred_target_domain,
                torch.ones_like(pred_target_domain)
            )
            
            loss_disc = loss_disc_source + loss_disc_target
            loss_disc.backward()
            opt_disc.step()
            
            # Train feature extractor and predictor
            opt_main.zero_grad()
            
            # Task loss on source
            feat_source = self.feature_extractor(X_source)
            pred_source = self.predictor(feat_source)
            loss_task = criterion_task(pred_source, y_source)
            
            # Adversarial loss (fool discriminator)
            feat_target = self.feature_extractor(target_data)
            pred_target_domain = self.discriminator(feat_target)
            loss_adv = criterion_domain(
                pred_target_domain,
                torch.zeros_like(pred_target_domain)  # Pretend target is source
            )
            
            loss_main = loss_task + lambda_adv * loss_adv
            loss_main.backward()
            opt_main.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Task={loss_task.item():.4f}, "
                      f"Adv={loss_adv.item():.4f}, Disc={loss_disc.item():.4f}")
        
        return {
            'feature_extractor': self.feature_extractor,
            'predictor': self.predictor,
        }
