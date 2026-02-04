"""Example: Train a PINN for pendulum dynamics.

This script demonstrates how to:
1. Create a pendulum simulator and generate training data
2. Build and configure a PendulumPINN model
3. Train the model with physics-informed learning
4. Evaluate the trained model
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from physics_informed_ml.models.pinn import PINNConfig
from physics_informed_ml.solvers.pendulum import (
    PendulumSimulator,
    PendulumConfig,
    PendulumPINN,
)
from physics_informed_ml.training.trainer import PINNTrainer, TrainingConfig
from physics_informed_ml.training.dataset import create_pendulum_dataset
from physics_informed_ml.core.config import SimulationConfig


def main() -> None:
    """Main training script."""
    # Configuration
    pendulum_config = PendulumConfig(length=1.0, gravity=9.81, damping=0.0)
    sim_config = SimulationConfig(dt=0.01, t_max=10.0)

    # Create training dataset
    print("Generating training data...")
    dataset = create_pendulum_dataset(
        pendulum_config=pendulum_config,
        sim_config=sim_config,
        n_trajectories=20,
        angle_range=(5.0, 45.0),
        n_physics_points=2000,
    )
    print(f"Dataset size: {len(dataset)} points")
    print(f"Physics collocation points: {len(dataset.x_physics)}")

    # Create PINN model
    print("\nBuilding PINN model...")
    pinn_config = PINNConfig(
        input_dim=1,  # Time
        output_dim=1,  # Angle
        hidden_dims=[64, 64, 64, 64],
        activation="tanh",
        use_batch_norm=False,
        dropout_rate=0.0,
    )
    model = PendulumPINN(pinn_config, pendulum_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Configure training
    train_config = TrainingConfig(
        epochs=2000,
        batch_size=64,
        learning_rate=1e-3,
        lambda_physics=1.0,  # Equal weight for physics and data loss
        lambda_data=1.0,
        physics_points_per_batch=200,
        checkpoint_dir="checkpoints/pendulum",
        log_interval=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create trainer
    trainer = PINNTrainer(model, train_config)

    # Train model
    print(f"\nTraining on device: {train_config.device}")
    print("Starting training...\n")

    history = trainer.train(dataset)

    # Save final checkpoint
    checkpoint_path = trainer.save_checkpoint(
        epoch=train_config.epochs, filename="final_model.pth"
    )
    print(f"\nFinal model saved to: {checkpoint_path}")

    # Plot training history
    print("\nPlotting training history...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["total_loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True)

    axes[1].plot(history["data_loss"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Data Loss")
    axes[1].set_title("Data Loss (MSE)")
    axes[1].set_yscale("log")
    axes[1].grid(True)

    axes[2].plot(history["physics_loss"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Physics Loss")
    axes[2].set_title("Physics Loss (PDE Residual)")
    axes[2].set_yscale("log")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    print("Training history saved to: training_history.png")

    # Test model predictions
    print("\nTesting model predictions...")
    model.eval()

    # Generate test trajectory
    simulator = PendulumSimulator(pendulum_config, sim_config)
    theta0 = 0.5  # 28.6 degrees
    times, angles_true, _, _ = simulator.simulate(theta0)

    # Predict with PINN
    t_tensor = torch.tensor(times, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        angles_pred = model(t_tensor).numpy().flatten()

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Angle comparison
    axes[0].plot(times, angles_true, label="True (Simulator)", linewidth=2)
    axes[0].plot(times, angles_pred, "--", label="PINN Prediction", linewidth=2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].set_title(f"Pendulum Dynamics (Initial Angle = {theta0:.2f} rad)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error
    error = angles_true - angles_pred
    axes[1].plot(times, error, color="red", linewidth=1.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (rad)")
    axes[1].set_title("Prediction Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("prediction_comparison.png", dpi=150, bbox_inches="tight")
    print("Prediction comparison saved to: prediction_comparison.png")

    # Compute error metrics
    mse = torch.mean((torch.tensor(angles_true) - torch.tensor(angles_pred)) ** 2)
    rmse = torch.sqrt(mse)
    print(f"\nPrediction Metrics:")
    print(f"  RMSE: {rmse:.6f} rad")
    print(f"  Max Error: {abs(error).max():.6f} rad")
    print(f"  Mean Absolute Error: {abs(error).mean():.6f} rad")

    plt.show()


if __name__ == "__main__":
    main()