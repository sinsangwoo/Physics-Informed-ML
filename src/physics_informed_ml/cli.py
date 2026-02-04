"""Command-line interface for Physics-Informed ML."""

import typer
from rich.console import Console
from typing_extensions import Annotated

app = typer.Typer(
    name="physics-ml",
    help="Physics-Informed ML: Neural Operators for Real-Time Simulation",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    config: Annotated[str, typer.Option(help="Path to training configuration file")],
) -> None:
    """Train a physics-informed model."""
    console.print(f"[green]Training model with config: {config}[/green]")
    console.print("[yellow]Note: Training implementation coming in Phase 1[/yellow]")


@app.command()
def infer(
    model: Annotated[str, typer.Option(help="Path to trained model")],
    input_data: Annotated[str, typer.Option(help="Path to input data")],
) -> None:
    """Run inference with a trained model."""
    console.print(f"[green]Running inference with model: {model}[/green]")
    console.print(f"[green]Input data: {input_data}[/green]")
    console.print("[yellow]Note: Inference implementation coming in Phase 1[/yellow]")


@app.command()
def visualize(
    problem: Annotated[str, typer.Option(help="Problem type to visualize")],
    interactive: Annotated[bool, typer.Option(help="Enable interactive mode")] = False,
) -> None:
    """Visualize simulation results."""
    console.print(f"[green]Visualizing problem: {problem}[/green]")
    if interactive:
        console.print("[cyan]Interactive mode enabled[/cyan]")
    console.print("[yellow]Note: Visualization implementation coming in Phase 4[/yellow]")


@app.command()
def version() -> None:
    """Show version information."""
    from physics_informed_ml import __version__

    console.print(f"[cyan]Physics-Informed ML version: {__version__}[/cyan]")


if __name__ == "__main__":
    app()