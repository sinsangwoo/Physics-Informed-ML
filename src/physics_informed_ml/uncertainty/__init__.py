"""Uncertainty Quantification for Physics-Informed Neural Networks.

Provides:
- Bayesian Neural Networks (BNN)
- Monte Carlo Dropout
- Deep Ensemble
- Calibration metrics
- Prediction intervals
"""

from physics_informed_ml.uncertainty.bayesian import (
    BayesianLinear,
    BayesianFNO,
    BayesianPINN,
    VariationalInference,
)
from physics_informed_ml.uncertainty.ensemble import (
    DeepEnsemble,
    EnsemblePredictor,
)
from physics_informed_ml.uncertainty.dropout import MCDropout
from physics_informed_ml.uncertainty.calibration import (
    CalibrationMetrics,
    plot_calibration_curve,
    plot_prediction_intervals,
)

__all__ = [
    "BayesianLinear",
    "BayesianFNO",
    "BayesianPINN",
    "VariationalInference",
    "DeepEnsemble",
    "EnsemblePredictor",
    "MCDropout",
    "CalibrationMetrics",
    "plot_calibration_curve",
    "plot_prediction_intervals",
]
