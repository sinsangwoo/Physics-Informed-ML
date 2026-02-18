# Uncertainty Quantification

## Overview

This module provides state-of-the-art uncertainty quantification methods for Physics-Informed Neural Networks.

**Three complementary approaches:**

1. **Bayesian Neural Networks**: Weight uncertainty via variational inference
2. **Deep Ensemble**: Model disagreement from independent training
3. **Monte Carlo Dropout**: Dropout as Bayesian approximation

---

## Why Uncertainty Matters

### For Physics
- **Safety**: Know when predictions are unreliable
- **Active Learning**: Sample where uncertainty is highest
- **Model Validation**: Calibrated uncertainty = trustworthy predictions
- **OOD Detection**: Identify out-of-distribution inputs

### Two Types of Uncertainty

**Epistemic (Model Uncertainty)**
- "What we don't know"
- Reducible with more data
- High in unexplored regions
- Captured by: Bayesian methods, ensemble disagreement

**Aleatoric (Data Uncertainty)**  
- "Inherent noise in data"
- Irreducible
- High in noisy regions
- Captured by: Learned noise parameters

---

## Methods

### 1. Bayesian Neural Networks

**Key Idea**: Treat weights as distributions, not point estimates.

#### Implementation

```python
from physics_informed_ml.uncertainty import (
    BayesianPINN,
    VariationalInference
)

# Create Bayesian model
model = BayesianPINN(
    input_dim=2,
    hidden_dims=[64, 64],
    output_dim=1,
    prior_std=1.0  # Prior N(0, 1)
)

# Variational inference trainer
vi = VariationalInference(model, n_data=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(epochs):
    predictions, kl_divergence = model(X_batch)
    
    # ELBO loss = Negative Log Likelihood + KL Divergence
    loss, nll, kl = vi.elbo_loss(predictions, y_batch, kl_divergence)
    
    loss.backward()
    optimizer.step()

# Prediction with uncertainty
mean, std, samples = model.predict_with_uncertainty(
    X_test, 
    n_samples=100
)
```

#### Mathematical Details

**Variational Inference:**
```
Posterior: q(w|θ) ≈ p(w|D)

ELBO = 𝔼[log p(y|x,w)] - KL(q(w|θ) || p(w))
      ↑                    ↑
   Accuracy           Regularization
```

**Reparameterization Trick:**
```
w = μ + σ ⊙ ε,  ε ~ N(0,1)
```

Allows backpropagation through sampling.

---

### 2. Deep Ensemble

**Key Idea**: Train N independent models, disagreement = uncertainty.

#### Implementation

```python
from physics_informed_ml.uncertainty import DeepEnsemble
from physics_informed_ml.models import FNO1d

# Create ensemble
ensemble = DeepEnsemble(
    base_model=lambda: FNO1d(modes=12, width=32),
    n_models=5,
    device="cuda"
)

# Train ensemble
ensemble.train(
    train_loader=train_loader,
    epochs=100,
    lr=1e-3
)

# Predict with uncertainty
mean, std, individuals = ensemble.predict(
    X_test,
    return_individuals=True
)
```

#### Why It Works

- **Diversity**: Different initializations → different local minima
- **Simple**: No architecture changes needed
- **Effective**: Often outperforms Bayesian methods
- **Parallelizable**: Train models independently

#### Advanced Features

```python
from physics_informed_ml.uncertainty import EnsemblePredictor

predictor = EnsemblePredictor(ensemble)

# Confidence intervals
mean, lower, upper = predictor.predict_with_confidence(
    X_test,
    confidence=0.95
)

# Outlier detection
outliers = predictor.detect_outliers(
    X_test,
    threshold=3.0  # 3σ threshold
)
```

---

### 3. Monte Carlo Dropout

**Key Idea**: Keep dropout active during inference, sample multiple times.

#### Implementation

```python
from physics_informed_ml.uncertainty import MCDropout
from physics_informed_ml.models import FNO1d

# Wrap existing model
base_model = FNO1d(modes=12, width=32)
mc_model = MCDropout(base_model, dropout_rate=0.1)

# Train normally
# ... training code ...

# Predict with uncertainty
mean, std, samples = mc_model.predict_with_uncertainty(
    X_test,
    n_samples=50
)
```

#### Advantages

- **No Architecture Change**: Add dropout to any model
- **Single Model**: No need to train ensemble
- **Fast**: Quick inference
- **Theoretically Grounded**: Approximates Gaussian Process

---

## Calibration

A model is **well-calibrated** if its uncertainty matches observed errors.

### Metrics

#### 1. Expected Calibration Error (ECE)

```python
from physics_informed_ml.uncertainty import CalibrationMetrics

metrics = CalibrationMetrics(n_bins=10)
ece = metrics.expected_calibration_error(
    predictions=pred_mean,
    uncertainties=pred_std,
    targets=y_true
)

print(f"ECE: {ece:.4f}")  # Lower is better, < 0.05 is good
```

**Interpretation:**
- ECE < 0.05: Well calibrated
- ECE > 0.10: Poorly calibrated (overconfident or underconfident)

#### 2. Prediction Interval Coverage

```python
coverage, below, above = metrics.prediction_interval_coverage(
    predictions=pred_mean,
    uncertainties=pred_std,
    targets=y_true,
    confidence=0.95
)

print(f"95% CI Coverage: {coverage:.2%}")  # Should be ≈ 95%
```

**Interpretation:**
- Coverage ≈ confidence level: Well calibrated
- Coverage < confidence: Overconfident (intervals too narrow)
- Coverage > confidence: Underconfident (intervals too wide)

#### 3. Sharpness

```python
sharpness = metrics.sharpness(pred_std)
print(f"Average uncertainty: {sharpness:.4f}")  # Lower is better
```

**Goal**: Minimize uncertainty while maintaining calibration.

#### 4. CRPS (Continuous Ranked Probability Score)

```python
crps = metrics.continuous_ranked_probability_score(
    samples=prediction_samples,
    targets=y_true
)
```

**Proper scoring rule** that rewards both accuracy and calibration.

---

## Visualization

### Reliability Diagram

```python
from physics_informed_ml.uncertainty import plot_calibration_curve

plot_calibration_curve(
    predictions=pred_mean,
    uncertainties=pred_std,
    targets=y_true,
    n_bins=10,
    save_path='calibration.png'
)
```

Shows confidence vs accuracy. Perfect calibration = diagonal line.

### Prediction Intervals

```python
from physics_informed_ml.uncertainty import plot_prediction_intervals

plot_prediction_intervals(
    x=X_test.numpy(),
    predictions=pred_mean,
    uncertainties=pred_std,
    targets=y_true,
    confidence=0.95,
    save_path='intervals.png'
)
```

---

## Best Practices

### Choosing a Method

| Method | Pros | Cons | Use When |
|--------|------|------|----------|
| **Bayesian** | Theoretically grounded, single model | Harder to train, slower | Need epistemic uncertainty |
| **Ensemble** | Simple, effective, parallelizable | Multiple models, more memory | Production applications |
| **MC Dropout** | Fastest, minimal changes | Less accurate | Quick prototyping |

### Recommendations

**For Research:**
- Start with Bayesian for theoretical rigor
- Use ensemble as baseline
- Compare all three methods

**For Production:**
- Use ensemble (5-10 models)
- Monitor calibration metrics
- Retrain periodically

**For Prototyping:**
- Use MC Dropout for quick uncertainty estimates
- Upgrade to ensemble if needed

---

## Examples

See `examples/uncertainty/` for complete examples:

1. **`bayesian_example.py`**: Bayesian PINN training and calibration
2. **`ensemble_example.py`**: Deep ensemble and outlier detection
3. **`mcdropout_example.py`**: MC Dropout comparison (TODO)

---

## References

**Bayesian Deep Learning:**
- Blundell et al. (2015) - Weight Uncertainty in Neural Networks
- Graves (2011) - Practical Variational Inference

**Deep Ensembles:**
- Lakshminarayanan et al. (2017) - Simple and Scalable Predictive Uncertainty Estimation

**MC Dropout:**
- Gal & Ghahramani (2016) - Dropout as a Bayesian Approximation

**Calibration:**
- Guo et al. (2017) - On Calibration of Modern Neural Networks
- Kuleshov et al. (2018) - Accurate Uncertainties for Deep Learning

---

## API Reference

See module docstrings for detailed API documentation:

```python
help(BayesianPINN)
help(DeepEnsemble)
help(CalibrationMetrics)
```
