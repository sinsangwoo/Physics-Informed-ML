"""Research-grade features for advanced physics-informed ML.

Modules:
- uncertainty: Bayesian uncertainty quantification
- transfer: Transfer learning for new physics
- explain: Model explainability and interpretability
- adaptive: Adaptive sampling and mesh refinement
"""

from physics_informed_ml.research.uncertainty import BayesianPINN, MCDropout
from physics_informed_ml.research.transfer import TransferLearning, DomainAdaptation
from physics_informed_ml.research.explain import SHAPExplainer, GradCAM

__all__ = [
    "BayesianPINN",
    "MCDropout",
    "TransferLearning",
    "DomainAdaptation",
    "SHAPExplainer",
    "GradCAM",
]
