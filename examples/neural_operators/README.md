# Neural Operator Examples

This directory contains examples demonstrating Neural Operator models for solving PDEs.

## Examples

### 1. FNO Benchmark (`run_fno_benchmark.py`)

Benchmarks Fourier Neural Operator on 1D heat equation.

**Key Features:**
- Resolution-invariant learning
- Multi-resolution evaluation
- Performance comparison
- Visualization of results

**Usage:**
```bash
python examples/neural_operators/run_fno_benchmark.py
```

**Expected Output:**
- Training loss curve
- Prediction vs ground truth plots
- Performance metrics (accuracy, speed, memory)
- JSON results file

**What to Look For:**
1. **Resolution Invariance**: Error should stay similar across 64, 128, 256 grid points
2. **Speed**: Should be much faster than traditional PDE solvers (100-1000x)
3. **Accuracy**: L2 relative error < 1% for this simple problem

### 2. Coming Soon:
- 2D Navier-Stokes with FNO
- DeepONet implementation
- Multi-physics problems
- Transfer learning examples

## Understanding the Results

### Metrics Explained

**Accuracy Metrics:**
- **L2 Relative Error**: Normalized error (lower is better, <0.01 is excellent)
- **L∞ Error**: Maximum pointwise error (checks worst-case accuracy)
- **R² Score**: Coefficient of determination (1.0 = perfect, 0.0 = mean baseline)

**Performance Metrics:**
- **Inference Time**: Forward pass time (milliseconds)
- **Training Time**: Total training duration (seconds)
- **Memory**: Peak memory usage (MB)

**Resolution Invariance:**
A key advantage of Neural Operators! If trained on 64 points:
- Traditional CNN: Must retrain for different resolutions
- Neural Operator: Works on 128, 256, 512... without retraining!

## Customization

### Modify Problem Parameters

```python
problem = HeatEquation1D(
    alpha=0.01,  # Thermal diffusivity
    L=1.0,       # Domain length
)
```

### Tune FNO Architecture

```python
fno = FNO1d(
    modes=12,      # More modes = finer details (but more parameters)
    width=32,      # Hidden dimension (model capacity)
    n_layers=4,    # Depth (deeper = more expressive)
)
```

**Guidelines:**
- More complex PDEs → Increase `width` and `n_layers`
- Higher frequency content → Increase `modes`
- Limited GPU memory → Decrease `width`

### Benchmark Configuration

```python
results = runner.run(
    train_resolution=64,    # Training grid size
    test_resolutions=[64, 128, 256],  # Test generalization
    n_train_samples=1000,   # More samples = better learning
    epochs=100,             # Training iterations
    batch_size=32,          # Batch size
    lr=1e-3,               # Learning rate
)
```

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size`
- Decrease `width` in FNO
- Lower `train_resolution`

**Poor Accuracy?**
- Increase `epochs`
- Increase `n_train_samples`
- Tune `modes` and `width`
- Check if problem needs data normalization

**Slow Training?**
- Use GPU if available
- Reduce `n_train_samples`
- Lower `train_resolution`

## Next Steps

1. Try different PDE problems (Wave, Burgers, Navier-Stokes)
2. Experiment with hyperparameters
3. Compare FNO vs PINN vs traditional solvers
4. Test on real-world datasets
