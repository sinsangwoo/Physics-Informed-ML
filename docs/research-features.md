# Research-Grade Features

## Overview

Phase 5 adds cutting-edge research capabilities for advanced physics-informed machine learning applications.

---

## 1. Uncertainty Quantification

### Monte Carlo Dropout

```python
from physics_informed_ml.research import MCDropout

# Wrap existing model
base_model = FNO1d(modes=12, width=32)
mc_model = MCDropout(base_model, dropout_rate=0.1, n_samples=50)

# Predict with uncertainty
mean, std = mc_model.predict_with_uncertainty(x_test)

# Get credible intervals
q05, q95 = mc_model.predict_quantiles(x_test, quantiles=[0.05, 0.95])
```

**When to use:**
- Safety-critical applications
- Model debugging
- Active learning
- Out-of-distribution detection

### Bayesian PINNs

```python
from physics_informed_ml.research import BayesianPINN

model = BayesianPINN(
    input_dim=2,
    hidden_dims=[64, 64, 64],
    output_dim=1,
    prior_std=1.0
)

# ELBO loss for training
loss = model.elbo_loss(
    x_train,
    y_train,
    physics_loss=pde_residual,
    beta=0.01  # KL weight
)
```

**Advantages:**
- Principled uncertainty quantification
- Automatic regularization
- Calibrated confidence intervals

---

## 2. Transfer Learning

### Strategy Selection

| Strategy | Data Size | Similarity | Speed |
|----------|-----------|------------|-------|
| Feature Extraction | Small | High | Fast |
| Fine-Tuning | Medium | Medium | Medium |
| Progressive Unfreezing | Large | Low | Slow |
| Domain Adaptation | Unlabeled | Low | Medium |

### Feature Extraction

```python
from physics_informed_ml.research import TransferLearning

# Train on heat equation
heat_model = train_heat_equation()

# Transfer to wave equation (small dataset)
transfer = TransferLearning(heat_model)
wave_model = transfer.feature_extraction(
    target_data=(X_wave, y_wave),
    n_epochs=50
)
```

### Fine-Tuning

```python
# Medium dataset, related physics
wave_model = transfer.fine_tuning(
    target_data=(X_wave, y_wave),
    n_epochs=100,
    lr=1e-4,
    warmup_epochs=20
)
```

### Progressive Unfreezing

```python
# Large dataset, different physics
layer_groups = [
    ['fc.output'],  # Final layer first
    ['fc.2', 'fc.3'],  # Then middle layers
    ['fc.0', 'fc.1'],  # Finally early layers
]

wave_model = transfer.progressive_unfreezing(
    target_data=(X_wave, y_wave),
    layer_groups=layer_groups,
    epochs_per_group=20
)
```

### Domain Adaptation

```python
from physics_informed_ml.research import DomainAdaptation

# Adapt to new Reynolds number (unlabeled)
da = DomainAdaptation(
    feature_extractor=feature_net,
    predictor=pred_head
)

adapted_models = da.adapt(
    source_data=(X_source, y_source),
    target_data=X_target_unlabeled,
    lambda_adv=0.1
)
```

---

## 3. Explainability

### SHAP Analysis

```python
from physics_informed_ml.research import SHAPExplainer

explainer = SHAPExplainer(model, n_samples=100)

# Local explanation
shap_values = explainer.explain(x_test, background_data)

# Global feature importance
importance = explainer.feature_importance(x_test, background_data)

print(f"Most important features: {importance.topk(5)}")
```

**Interpretation:**
- Positive SHAP value: Feature increases prediction
- Negative SHAP value: Feature decreases prediction
- Magnitude: Importance of feature

### Grad-CAM Visualization

```python
from physics_informed_ml.research import GradCAM
import matplotlib.pyplot as plt

gradcam = GradCAM(model, target_layer='conv_final')
heatmap = gradcam.generate_heatmap(x_test)

# Visualize
plt.imshow(heatmap[0].cpu(), cmap='hot')
plt.title('Important Spatial Regions')
plt.colorbar()
plt.show()
```

**Use Cases:**
- Spatial PDE problems (2D/3D)
- Debugging model attention
- Understanding physics encoding

### Sensitivity Analysis

```python
from physics_informed_ml.research import SensitivityAnalysis

analyzer = SensitivityAnalysis(model)

# Input sensitivity
sensitivity = analyzer.input_sensitivity(x_test, epsilon=0.01)

# Parameter sweep
Re_values = np.linspace(100, 1000, 20)
results = analyzer.parameter_sweep(x_test, 'reynolds', Re_values)

# Uncertainty propagation
mean, std = analyzer.uncertainty_propagation(
    x_mean, x_std, n_samples=100
)
```

---

## 4. Best Practices

### Uncertainty Quantification

1. **Always visualize uncertainty**
   ```python
   plt.fill_between(x, mean - 2*std, mean + 2*std, alpha=0.3)
   ```

2. **Validate calibration**
   - Expected: 90% of true values in 90% CI
   - Compute empirical coverage

3. **Use for active learning**
   - Sample where uncertainty is highest
   - Improves data efficiency

### Transfer Learning

1. **Start with feature extraction**
   - Fast baseline
   - Low risk of overfitting

2. **Monitor validation loss**
   - Stop if diverging
   - Use early stopping

3. **Layer-wise learning rates**
   ```python
   optimizer = optim.Adam([
       {'params': model.early_layers.parameters(), 'lr': 1e-5},
       {'params': model.late_layers.parameters(), 'lr': 1e-3}
   ])
   ```

### Explainability

1. **Sanity checks**
   - Do important features make physical sense?
   - Are heatmaps consistent across similar inputs?

2. **Multiple methods**
   - Use SHAP + Grad-CAM together
   - Cross-validate insights

3. **Document assumptions**
   - SHAP assumes feature independence
   - Grad-CAM limited to CNN-like architectures

---

## 5. Research Applications

### Aerospace
- **Uncertainty**: Safety margins for flight envelopes
- **Transfer**: Different aircraft geometries
- **Explain**: Understanding turbulence patterns

### Energy
- **Uncertainty**: Wind farm power predictions
- **Transfer**: Different turbine designs
- **Explain**: Flow feature importance

### Manufacturing
- **Uncertainty**: Material property bounds
- **Transfer**: New alloy compositions
- **Explain**: Stress concentration factors

### Climate
- **Uncertainty**: Model ensemble disagreement
- **Transfer**: Different emission scenarios
- **Explain**: Driver attribution

---

## 6. Performance Considerations

| Method | Overhead | Memory | Use Case |
|--------|----------|--------|----------|
| MC Dropout | 50x | 1x | Fast uncertainty |
| Bayesian PINN | 1x | 2x | Principled UQ |
| Feature Extract | 0.5x | 1x | Quick transfer |
| Fine-Tuning | 1x | 1x | Full transfer |
| SHAP | 100x | 1x | Local explain |
| Grad-CAM | 1x | 1x | Visual explain |
| Sensitivity | 10x | 1x | Parameter study |

---

## 7. Future Directions

### Planned
- [ ] Conformal prediction
- [ ] Multi-task learning
- [ ] Neural architecture search
- [ ] Symbolic regression
- [ ] Causal inference

### Research Ideas
- Hybrid physics-ML uncertainty
- Few-shot PDE learning
- Interpretable neural operators
- Adaptive mesh refinement with ML

---

## References

1. **Uncertainty Quantification**
   - Gal & Ghahramani (2016) - Dropout as Bayesian Approximation
   - Blundell et al. (2015) - Weight Uncertainty in Neural Networks

2. **Transfer Learning**
   - Yosinski et al. (2014) - How transferable are features
   - Ganin et al. (2016) - Domain-Adversarial Training

3. **Explainability**
   - Lundberg & Lee (2017) - SHAP
   - Selvaraju et al. (2017) - Grad-CAM

---

**Next Steps:** Try the examples in `examples/research/` directory!
