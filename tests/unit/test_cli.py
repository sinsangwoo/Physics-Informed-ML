"""Tests for command-line interface."""

from typer.testing import CliRunner
from physics_informed_ml.cli import app

runner = CliRunner()


class TestCLI:
    """Test suite for CLI."""

    def test_version_command(self) -> None:
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Physics-Informed ML version" in result.stdout

    def test_train_command(self) -> None:
        """Test train command."""
        result = runner.invoke(app, ["train", "--config", "test_config.yaml"])
        assert result.exit_code == 0
        assert "Training model" in result.stdout

    def test_infer_command(self) -> None:
        """Test infer command."""
        result = runner.invoke(
            app, ["infer", "--model", "test_model.pth", "--input-data", "test_input.json"]
        )
        assert result.exit_code == 0
        assert "Running inference" in result.stdout

    def test_visualize_command(self) -> None:
        """Test visualize command."""
        result = runner.invoke(app, ["visualize", "--problem", "pendulum"])
        assert result.exit_code == 0
        assert "Visualizing problem" in result.stdout

    def test_visualize_interactive(self) -> None:
        """Test visualize command with interactive flag."""
        result = runner.invoke(
            app, ["visualize", "--problem", "pendulum", "--interactive"]
        )
        assert result.exit_code == 0
        assert "Interactive mode enabled" in result.stdout