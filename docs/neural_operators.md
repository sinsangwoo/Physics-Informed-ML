# Neural Operators

## Overview

Neural Operators are a class of neural networks that learn mappings between **function spaces** rather than finite-dimensional vectors. This makes them particularly powerful for solving parametric PDEs.

### Key Advantages

1. **Resolution Invariance**: Train on 64 grid points, evaluate on 256 points without retraining
2. **Speed**: 100-1000x faster than traditional PDE solvers for inference
3. **Generalization**: Learn families of PDEs, not single instances
4. **Global Receptive Field**: Fourier transform gives instant global information

## Fourier Neural Operator (FNO)

### Theory

FNO learns operators by:
1. Lifting input to higher dimension
2. Processing in Fourier space (spectral convolution)
3. Projecting back to output dimension

**Mathematical Formulation:**

For function $u(x)$ and operator $\mathcal{G}$:

$$v(x) = \mathcal{G}(u)(x)$$

FNO approximates $\mathcal{G}$ through:

$$v_\theta(x) = \mathcal{Q} \circ \sigma(W + \mathcal{K}) \circ \cdots \circ \sigma(W + \mathcal{K}) \circ \mathcal{P}(u)(x)$$

Where:
- $\mathcal{P}$: Lifting layer (increase channels)
- $\mathcal{K}$: Fourier layer (spectral convolution)
- $W$: Skip connection (local operation)
- $\sigma$: Activation function (GELU)
- $\mathcal{Q}$: Projection layer (reduce channels)

### Spectral Convolution

The key innovation is convolution in Fourier space:

$$\mathcal{K}(v)(x) = \mathcal{F}^{-1}(R \cdot \mathcal{F}(v))(x)$$

Where $R$ are learnable weights in frequency domain.

**Why Fourier Space?**
- **Global information**: All spatial points interact instantly
- **Efficiency**: FFT is $O(N \log N)$ vs $O(N^2)$ for standard convolution
- **Resolution invariance**: Frequency representation is independent of discretization

### Implementation

```python
from physics_informed_ml.models import FNO1d, FNO2d, FNO3d

# 1D Heat Equation
fno1d = FNO1d(
    modes=12,        # Number of Fourier modes
    width=32,        # Hidden channels
    in_channels=1,   # Initial temperature
    out_channels=1,  # Final temperature
    n_layers=4,      # Depth
)

# 2D Navier-Stokes
fno2d = FNO2d(
    modes1=12,
    modes2=12,
    width=64,
    in_channels=3,   # [u, v, p]
    out_channels=3,
    n_layers=4,
)

# 3D Elasticity
fno3d = FNO3d(
    modes1=8, modes2=8, modes3=8,
    width=32,
    in_channels=6,   # Stress tensor components
    out_channels=3,  # Displacement vector
    n_layers=4,
)
```

### Hyperparameter Guide

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `modes` | Frequency resolution | 8-16 for smooth problems, 20-32 for complex |
| `width` | Model capacity | 32-64 typical, 128+ for hard problems |
| `n_layers` | Model depth | 4 standard, 6-8 for very complex PDEs |
| `in_channels` | Input features | Match problem (1 for scalar, 3 for vector) |
| `out_channels` | Output features | Match desired prediction |

**Memory vs Accuracy Trade-off:**
- More modes → Better frequency resolution → More parameters
- More width → Better representation → More parameters
- More layers → Deeper hierarchy → Marginal parameter increase

## Benchmarking

### Quick Start

```python
from physics_informed_ml.benchmarks import (
    BenchmarkRunner,
    HeatEquation1D,
)
from physics_informed_ml.models import FNO1d

# Setup
runner = BenchmarkRunner(device="cuda")
problem = HeatEquation1D(alpha=0.01)
model = FNO1d(modes=12, width=32, n_layers=4)

# Run benchmark
results = runner.run(
    model=model,
    problem=problem,
    train_resolution=64,
    test_resolutions=[64, 128, 256],
    epochs=100,
)

# Results available at different resolutions
for res, metrics in results.items():
    print(f"Resolution {res}: Error={metrics.l2_relative_error:.4f}")
```

### Benchmark Problems

#### 1. Heat Equation (Parabolic)

**PDE:** $\frac{\partial u}{\partial t} = \alpha \nabla^2 u$

**Physics:** Heat diffusion, mass transport

**Difficulty:** Easy (linear, smooth)

**Expected Performance:**
- L2 Error: < 1%
- Speed: 500+ predictions/sec on GPU

```python
from physics_informed_ml.benchmarks import HeatEquation1D

problem = HeatEquation1D(
    alpha=0.01,  # Thermal diffusivity
    L=1.0,       # Domain length
)
```

#### 2. Wave Equation (Hyperbolic)

**PDE:** $\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$

**Physics:** Acoustic waves, vibrations

**Difficulty:** Medium (oscillatory)

**Expected Performance:**
- L2 Error: 1-3%
- Speed: 400+ predictions/sec

```python
from physics_informed_ml.benchmarks import WaveEquation1D

problem = WaveEquation1D(
    c=1.0,  # Wave speed
    L=1.0,
)
```

#### 3. Burgers Equation (Nonlinear)

**PDE:** $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \nabla^2 u$

**Physics:** Shock waves, turbulence model

**Difficulty:** Hard (shocks, nonlinear)

**Expected Performance:**
- L2 Error: 3-5%
- Requires more modes/width

```python
from physics_informed_ml.benchmarks import BurgersEquation1D

problem = BurgersEquation1D(
    nu=0.01,  # Viscosity (smaller = sharper shocks)
    L=1.0,
)
```

#### 4. Navier-Stokes (2D Incompressible)

**PDE:** 
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$$
$$\nabla \cdot \mathbf{u} = 0$$

**Physics:** Fluid dynamics

**Difficulty:** Very Hard (nonlinear, coupled, turbulent)

```python
from physics_informed_ml.benchmarks import NavierStokes2D

problem = NavierStokes2D(
    nu=0.001,  # Kinematic viscosity
    Re=100,    # Reynolds number
)
```

### Metrics Explanation

#### Accuracy Metrics

**L2 Relative Error:**
$$\text{L2 Error} = \frac{\|u_{\text{pred}} - u_{\text{true}}\|_2}{\|u_{\text{true}}\|_2}$$

- Most important metric
- Normalized (scale-invariant)
- Lower is better
- < 0.01 = Excellent
- 0.01-0.05 = Good
- > 0.1 = Needs improvement

**L∞ Error:**
$$\text{L}^\infty \text{ Error} = \max_x |u_{\text{pred}}(x) - u_{\text{true}}(x)|$$

- Maximum pointwise error
- Catches outliers
- Important for safety-critical applications

**R² Score:**
$$R^2 = 1 - \frac{\sum (u_{\text{true}} - u_{\text{pred}})^2}{\sum (u_{\text{true}} - \bar{u}_{\text{true}})^2}$$

- 1.0 = Perfect prediction
- 0.0 = As good as mean baseline
- < 0.0 = Worse than predicting mean

#### Performance Metrics

**Inference Time:**
- Time for single forward pass
- Measured with GPU synchronization
- Compare to traditional solver runtime

**Speedup Calculation:**
```
Speedup = Traditional Solver Time / Neural Operator Time
```

Typical speedups:
- Simple PDEs: 100-500x
- Complex PDEs: 50-200x
- 3D problems: 500-2000x

**Memory Usage:**
- Peak GPU memory
- Scales with: width² × modes × layers
- FNO is very memory-efficient compared to CNNs

## Best Practices

### Training Tips

1. **Learning Rate Schedule:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=50, gamma=0.5
)
```

2. **Data Normalization:**
```python
# Normalize to zero mean, unit variance
X_mean, X_std = X.mean(), X.std()
X_normalized = (X - X_mean) / X_std
```

3. **Batch Size:**
- Larger batches = More stable gradients
- Smaller batches = Better generalization
- Recommendation: 32-64 for most problems

4. **Multi-Resolution Training:**
```python
# Train on mixed resolutions for better generalization
for epoch in range(epochs):
    res = random.choice([32, 64, 128])
    X, y = problem.generate_data(n_samples, res)
    # ... train ...
```

### Common Issues

**Problem: High error on test resolutions**
- Solution: Use more Fourier modes
- Solution: Train on multiple resolutions

**Problem: Training unstable**
- Solution: Lower learning rate
- Solution: Add gradient clipping
- Solution: Check data normalization

**Problem: Out of memory**
- Solution: Reduce width
- Solution: Reduce batch size
- Solution: Use gradient checkpointing

**Problem: Slow convergence**
- Solution: More training data
- Solution: Increase width
- Solution: Better initialization

## Comparison: FNO vs PINN

| Aspect | FNO | PINN |
|--------|-----|------|
| Training data | Requires paired data | Can use physics only |
| Speed | Very fast (1ms) | Moderate (10-100ms) |
| Resolution | Invariant ✓ | Grid-dependent |
| Generalization | Parametric families | Single instance |
| Physics constraints | Implicit (from data) | Explicit (loss term) |
| Best for | Repeated queries | Novel PDEs |

**When to use FNO:**
- Have/can generate training data
- Need real-time inference
- Want parametric families
- Have variable resolutions

**When to use PINN:**
- Limited/no data
- Novel PDE formulation
- Strong physics constraints
- Single high-accuracy solution

## References

1. **FNO Paper:**
   Li, Z., et al. "Fourier Neural Operator for Parametric Partial Differential Equations." 
   ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

2. **Neural Operators Review:**
   Kovachki, N., et al. "Neural Operator: Learning Maps Between Function Spaces."
   [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)

3. **Applications:**
   - Fluid Dynamics: Pathak et al., Nature 2022
   - Weather Forecasting: Kurth et al., 2022
   - Materials Science: Kollmann et al., 2021
