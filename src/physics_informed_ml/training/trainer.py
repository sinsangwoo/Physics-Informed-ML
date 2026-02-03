"""PINN training loop with advanced optimization."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Callable, Any
from pathlib import Path
from pydantic import BaseModel, Field
from loguru import logger
from tqdm import tqdm


class TrainingConfig(BaseModel):
    """Configuration for PINN training.

    Attributes:
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        optimizer: Optimizer name ('adam', 'lbfgs', 'sgd')
        scheduler: LR scheduler ('step', 'cosine', 'plateau', None)
        scheduler_patience: Patience for ReduceLROnPlateau
        scheduler_factor: Factor for reducing LR
        weight_decay: L2 regularization weight
        grad_clip: Gradient clipping value (None = no clipping)
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Save checkpoint every N epochs
        log_freq: Log metrics every N epochs
        device: Device to use ('cuda', 'cpu', 'mps')
    """

    epochs: int = Field(default=1000, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    optimizer: str = Field(default="adam")
    scheduler: Optional[str] = Field(default="step")
    scheduler_patience: int = Field(default=10, gt=0)
    scheduler_factor: float = Field(default=0.5, gt=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    grad_clip: Optional[float] = Field(default=None)
    checkpoint_dir: Optional[str] = Field(default="checkpoints")
    checkpoint_freq: int = Field(default=100, gt=0)
    log_freq: int = Field(default=10, gt=0)
    device: str = Field(default="cpu")

    class Config:
        """Pydantic config."""

        frozen = True


class PINNTrainer:
    """Trainer for Physics-Informed Neural Networks.

    Handles the complete training loop including:
    - Multi-component loss computation
    - Adaptive learning rate scheduling
    - Gradient clipping
    - Checkpointing
    - Logging

    Example:
        >>> config = TrainingConfig(epochs=1000, learning_rate=1e-3)
        >>> trainer = PINNTrainer(model, config)
        >>> history = trainer.train(
        ...     x_data=x_train,
        ...     y_data=y_train,
        ...     x_physics=x_collocation
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.config = config

        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup checkpoint directory
        if config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history: Dict[str, list[float]] = {
            "total_loss": [],
            "data_loss": [],
            "physics_loss": [],
            "ic_loss": [],
            "bc_loss": [],
        }

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "lbfgs":
            return optim.LBFGS(
                self.model.parameters(),
                lr=self.config.learning_rate,
                max_iter=20,
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.config.scheduler is None:
            return None

        if self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor,
            )
        elif self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _to_device(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Move tensor to device if not None."""
        return tensor.to(self.device) if tensor is not None else None

    def train_epoch(
        self,
        x_data: Optional[torch.Tensor] = None,
        y_data: Optional[torch.Tensor] = None,
        x_physics: Optional[torch.Tensor] = None,
        x_ic: Optional[torch.Tensor] = None,
        y_ic: Optional[torch.Tensor] = None,
        x_bc: Optional[torch.Tensor] = None,
        y_bc: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            x_data: Data input points
            y_data: Data target values
            x_physics: Physics collocation points
            x_ic: Initial condition points
            y_ic: Initial condition values
            x_bc: Boundary condition points
            y_bc: Boundary condition values

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        # Move data to device
        x_data = self._to_device(x_data)
        y_data = self._to_device(y_data)
        x_physics = self._to_device(x_physics)
        x_ic = self._to_device(x_ic)
        y_ic = self._to_device(y_ic)
        x_bc = self._to_device(x_bc)
        y_bc = self._to_device(y_bc)

        # Compute loss
        def closure() -> torch.Tensor:
            self.optimizer.zero_grad()
            loss, loss_dict = self.model.compute_total_loss(
                x_data=x_data,
                y_data=y_data,
                x_physics=x_physics,
                x_ic=x_ic,
                y_ic=y_ic,
                x_bc=x_bc,
                y_bc=y_bc,
            )
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            return loss

        # Optimizer step
        if isinstance(self.optimizer, optim.LBFGS):
            self.optimizer.step(closure)
            loss = closure()
        else:
            loss = closure()
            self.optimizer.step()

        # Compute losses for logging
        with torch.no_grad():
            _, loss_dict = self.model.compute_total_loss(
                x_data=x_data,
                y_data=y_data,
                x_physics=x_physics,
                x_ic=x_ic,
                y_ic=y_ic,
                x_bc=x_bc,
                y_bc=y_bc,
            )

        return loss_dict

    def train(
        self,
        x_data: Optional[torch.Tensor] = None,
        y_data: Optional[torch.Tensor] = None,
        x_physics: Optional[torch.Tensor] = None,
        x_ic: Optional[torch.Tensor] = None,
        y_ic: Optional[torch.Tensor] = None,
        x_bc: Optional[torch.Tensor] = None,
        y_bc: Optional[torch.Tensor] = None,
        val_data: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, list[float]]:
        """Complete training loop.

        Args:
            x_data: Training data inputs
            y_data: Training data targets
            x_physics: Physics collocation points
            x_ic: Initial condition points
            y_ic: Initial condition values
            x_bc: Boundary condition points
            y_bc: Boundary condition values
            val_data: Optional validation (x, y) tuple

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        pbar = tqdm(range(self.config.epochs), desc="Training")

        for epoch in pbar:
            # Train one epoch
            loss_dict = self.train_epoch(
                x_data=x_data,
                y_data=y_data,
                x_physics=x_physics,
                x_ic=x_ic,
                y_ic=y_ic,
                x_bc=x_bc,
                y_bc=y_bc,
            )

            # Update history
            self.history["total_loss"].append(loss_dict.get("total", 0.0))
            self.history["data_loss"].append(loss_dict.get("data", 0.0))
            self.history["physics_loss"].append(loss_dict.get("physics", 0.0))
            self.history["ic_loss"].append(loss_dict.get("ic", 0.0))
            self.history["bc_loss"].append(loss_dict.get("bc", 0.0))

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss_dict["total"])
                else:
                    self.scheduler.step()

            # Logging
            if (epoch + 1) % self.config.log_freq == 0:
                log_msg = f"Epoch {epoch+1}/{self.config.epochs} - "
                log_msg += f"Total: {loss_dict['total']:.6f}"
                if "data" in loss_dict:
                    log_msg += f" | Data: {loss_dict['data']:.6f}"
                if "physics" in loss_dict:
                    log_msg += f" | Physics: {loss_dict['physics']:.6f}"

                pbar.set_postfix_str(log_msg.split(" - ")[1])

            # Checkpointing
            if (
                self.config.checkpoint_dir
                and (epoch + 1) % self.config.checkpoint_freq == 0
            ):
                self.save_checkpoint(epoch + 1)

        logger.info("Training completed")
        return self.history

    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number
        """
        if not self.config.checkpoint_dir:
            return

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            checkpoint_path,
        )
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)

        epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch