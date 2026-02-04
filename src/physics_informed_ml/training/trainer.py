"""Training framework for physics-informed models."""

from pathlib import Path
from typing import Dict, List, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pydantic import BaseModel, Field
from loguru import logger

from physics_informed_ml.models.base import PhysicsInformedModel
from physics_informed_ml.training.dataset import PhysicsDataset


class TrainingConfig(BaseModel):
    """Configuration for model training.

    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for data loader
        learning_rate: Initial learning rate
        lambda_physics: Weight for physics loss
        lambda_data: Weight for data loss
        physics_points_per_batch: Physics points to sample per batch
        checkpoint_dir: Directory for saving checkpoints
        log_interval: Logging frequency (epochs)
        device: Device for training ('cpu' or 'cuda')
    """

    epochs: int = Field(default=1000, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    lambda_physics: float = Field(default=1.0, ge=0)
    lambda_data: float = Field(default=1.0, ge=0)
    physics_points_per_batch: int = Field(default=100, gt=0)
    checkpoint_dir: str = Field(default="checkpoints")
    log_interval: int = Field(default=100, gt=0)
    device: str = Field(default="cpu")

    class Config:
        """Pydantic config."""

        frozen = True


class PINNTrainer:
    """Trainer for physics-informed neural networks.

    Handles training loop, loss computation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: PhysicsInformedModel,
        config: TrainingConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Initialize PINN trainer.

        Args:
            model: Physics-informed model to train
            config: Training configuration
            optimizer: Optimizer (defaults to Adam)
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )

        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "data_loss": [],
            "physics_loss": [],
        }

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataset: PhysicsDataset,
        val_dataset: PhysicsDataset | None = None,
        callbacks: List[Callable] | None = None,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            dataset: Training dataset
            val_dataset: Validation dataset (optional)
            callbacks: List of callback functions called after each epoch

        Returns:
            Training history dictionary
        """
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Lambda physics: {self.config.lambda_physics}, "
            f"Lambda data: {self.config.lambda_data}"
        )

        for epoch in range(self.config.epochs):
            epoch_losses = self._train_epoch(dataloader, dataset)

            # Update history
            for key, value in epoch_losses.items():
                self.history[key].append(value)

            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                log_msg = f"Epoch {epoch + 1}/{self.config.epochs}"
                for key, value in epoch_losses.items():
                    log_msg += f" | {key}: {value:.6f}"
                logger.info(log_msg)

                # Validation
                if val_dataset is not None:
                    val_losses = self._validate(val_dataset)
                    val_msg = "Validation:"
                    for key, value in val_losses.items():
                        val_msg += f" | {key}: {value:.6f}"
                    logger.info(val_msg)

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, epoch_losses)

        logger.info("Training complete!")
        return self.history

    def _train_epoch(
        self, dataloader: DataLoader, dataset: PhysicsDataset
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Data loader for supervised data
            dataset: Full dataset for physics points

        Returns:
            Average losses for the epoch
        """
        self.model.train()
        epoch_losses: Dict[str, float] = {
            "total_loss": 0.0,
            "data_loss": 0.0,
            "physics_loss": 0.0,
        }

        n_batches = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Sample physics points
            x_physics = (
                dataset.get_physics_points(self.config.physics_points_per_batch)
                .to(self.device)
                .requires_grad_(True)
            )

            # Forward pass and compute losses
            self.optimizer.zero_grad()

            # Data loss
            y_pred = self.model(x_batch)
            data_loss = torch.mean((y_pred - y_batch) ** 2)

            # Physics loss
            physics_loss = self.model.compute_physics_loss(x_physics)

            # Total loss
            total_loss = (
                self.config.lambda_data * data_loss
                + self.config.lambda_physics * physics_loss
            )

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Accumulate losses
            epoch_losses["total_loss"] += total_loss.item()
            epoch_losses["data_loss"] += data_loss.item()
            epoch_losses["physics_loss"] += physics_loss.item()
            n_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        return epoch_losses

    def _validate(self, val_dataset: PhysicsDataset) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_dataset: Validation dataset

        Returns:
            Validation losses
        """
        self.model.eval()

        with torch.no_grad():
            x_val = val_dataset.x_data.to(self.device)
            y_val = val_dataset.y_data.to(self.device)

            y_pred = self.model(x_val)
            data_loss = torch.mean((y_pred - y_val) ** 2)

            x_physics = val_dataset.x_physics.to(self.device).requires_grad_(True)
            physics_loss = self.model.compute_physics_loss(x_physics)

            total_loss = (
                self.config.lambda_data * data_loss
                + self.config.lambda_physics * physics_loss
            )

        return {
            "total_loss": total_loss.item(),
            "data_loss": data_loss.item(),
            "physics_loss": physics_loss.item(),
        }

    def save_checkpoint(self, epoch: int, filename: str | None = None) -> Path:
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            filename: Checkpoint filename (auto-generated if None)

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"

        checkpoint_path = Path(self.config.checkpoint_dir) / filename

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "config": self.config.model_dump(),
            },
            checkpoint_path,
        )

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]

        epoch = checkpoint["epoch"]
        logger.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch