"""Tests for uncertainty quantification module."""

import pytest
import torch
import torch.nn as nn
from physics_informed_ml.uncertainty import (
    BayesianLinear,
    BayesianPINN,
    BayesianFNO,
    VariationalInference,
    DeepEnsemble,
    EnsemblePredictor,
    MCDropout,
    CalibrationMetrics,
)


class TestBayesianLayers:
    """Test Bayesian layer implementations."""
    
    def test_bayesian_linear_forward(self):
        """Test BayesianLinear forward pass."""
        layer = BayesianLinear(10, 5)
        x = torch.randn(32, 10)
        
        output, kl = layer(x)
        
        assert output.shape == (32, 5)
        assert isinstance(kl, torch.Tensor)
        assert kl.item() > 0  # KL should be positive
    
    def test_bayesian_pinn(self):
        """Test BayesianPINN model."""
        model = BayesianPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            output_dim=1
        )
        
        x = torch.randn(16, 2)
        output, kl = model(x)
        
        assert output.shape == (16, 1)
        assert kl.item() > 0
    
    def test_bayesian_pinn_uncertainty(self):
        """Test uncertainty prediction."""
        model = BayesianPINN(
            input_dim=1,
            hidden_dims=[16],
            output_dim=1
        )
        
        x = torch.randn(10, 1)
        mean, std, samples = model.predict_with_uncertainty(x, n_samples=20)
        
        assert mean.shape == (10, 1)
        assert std.shape == (10, 1)
        assert samples.shape == (20, 10, 1)
        assert (std > 0).all()  # Uncertainty should be positive


class TestVariationalInference:
    """Test variational inference."""
    
    def test_elbo_loss(self):
        """Test ELBO loss computation."""
        model = BayesianPINN(input_dim=1, hidden_dims=[16], output_dim=1)
        vi = VariationalInference(model, n_data=100)
        
        x = torch.randn(10, 1)
        y = torch.randn(10, 1)
        
        predictions, kl = model(x)
        total_loss, nll, kl_weighted = vi.elbo_loss(predictions, y, kl)
        
        assert total_loss.item() > 0
        assert nll.item() > 0
        assert kl_weighted.item() > 0
        
        # ELBO = NLL + KL
        assert torch.isclose(total_loss, nll + kl_weighted, rtol=1e-5)


class TestDeepEnsemble:
    """Test deep ensemble."""
    
    @pytest.fixture
    def simple_model(self):
        """Simple model factory."""
        def factory():
            return nn.Sequential(
                nn.Linear(1, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
        return factory
    
    def test_ensemble_initialization(self, simple_model):
        """Test ensemble initialization."""
        ensemble = DeepEnsemble(
            base_model=simple_model,
            n_models=3,
            device="cpu"
        )
        
        assert len(ensemble.models) == 3
        assert all(isinstance(m, nn.Module) for m in ensemble.models)
    
    def test_ensemble_prediction(self, simple_model):
        """Test ensemble prediction."""
        ensemble = DeepEnsemble(
            base_model=simple_model,
            n_models=3,
            device="cpu"
        )
        
        x = torch.randn(10, 1)
        mean, std, individuals = ensemble.predict(x, return_individuals=True)
        
        assert mean.shape == (10, 1)
        assert std.shape == (10, 1)
        assert individuals.shape == (3, 10, 1)
        assert (std >= 0).all()


class TestEnsemblePredictor:
    """Test ensemble predictor utilities."""
    
    @pytest.fixture
    def ensemble(self):
        """Create test ensemble."""
        def factory():
            return nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        
        return DeepEnsemble(base_model=factory, n_models=3, device="cpu")
    
    def test_confidence_intervals(self, ensemble):
        """Test confidence interval prediction."""
        predictor = EnsemblePredictor(ensemble)
        x = torch.randn(10, 1)
        
        mean, lower, upper = predictor.predict_with_confidence(x, confidence=0.95)
        
        assert mean.shape == (10, 1)
        assert lower.shape == (10, 1)
        assert upper.shape == (10, 1)
        assert (lower <= mean).all()
        assert (mean <= upper).all()
    
    def test_outlier_detection(self, ensemble):
        """Test outlier detection."""
        predictor = EnsemblePredictor(ensemble)
        x = torch.randn(20, 1)
        
        outliers = predictor.detect_outliers(x, threshold=2.0)
        
        assert outliers.shape == (20,)
        assert outliers.dtype == torch.bool


class TestMCDropout:
    """Test Monte Carlo Dropout."""
    
    def test_mc_dropout_initialization(self):
        """Test MC Dropout initialization."""
        base_model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        mc_model = MCDropout(base_model, dropout_rate=0.1)
        
        assert isinstance(mc_model, nn.Module)
    
    def test_mc_dropout_uncertainty(self):
        """Test uncertainty estimation with MC Dropout."""
        base_model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        mc_model = MCDropout(base_model, dropout_rate=0.2)
        x = torch.randn(10, 1)
        
        mean, std, samples = mc_model.predict_with_uncertainty(x, n_samples=30)
        
        assert mean.shape == (10, 1)
        assert std.shape == (10, 1)
        assert samples.shape == (30, 10, 1)
        assert (std > 0).all()


class TestCalibrationMetrics:
    """Test calibration metrics."""
    
    @pytest.fixture
    def predictions(self):
        """Generate test predictions."""
        torch.manual_seed(42)
        n = 100
        predictions = torch.randn(n)
        uncertainties = torch.rand(n) * 0.5 + 0.1
        targets = predictions + torch.randn(n) * uncertainties
        return predictions, uncertainties, targets
    
    def test_expected_calibration_error(self, predictions):
        """Test ECE computation."""
        pred, unc, targets = predictions
        metrics = CalibrationMetrics(n_bins=10)
        
        ece = metrics.expected_calibration_error(pred, unc, targets)
        
        assert isinstance(ece, float)
        assert 0 <= ece <= 1
    
    def test_prediction_interval_coverage(self, predictions):
        """Test prediction interval coverage."""
        pred, unc, targets = predictions
        metrics = CalibrationMetrics()
        
        coverage, below, above = metrics.prediction_interval_coverage(
            pred, unc, targets, confidence=0.95
        )
        
        assert 0 <= coverage <= 1
        assert 0 <= below <= 1
        assert 0 <= above <= 1
        assert abs((below + coverage + above) - 1.0) < 0.01
    
    def test_sharpness(self, predictions):
        """Test sharpness metric."""
        _, unc, _ = predictions
        metrics = CalibrationMetrics()
        
        sharpness = metrics.sharpness(unc)
        
        assert isinstance(sharpness, float)
        assert sharpness > 0
    
    def test_crps(self, predictions):
        """Test CRPS computation."""
        pred, unc, targets = predictions
        metrics = CalibrationMetrics()
        
        # Generate samples
        samples = pred.unsqueeze(0) + torch.randn(50, len(pred)) * unc.unsqueeze(0)
        
        crps = metrics.continuous_ranked_probability_score(samples, targets)
        
        assert isinstance(crps, float)
        assert crps >= 0


class TestIntegration:
    """Integration tests."""
    
    def test_bayesian_training_loop(self):
        """Test complete Bayesian training loop."""
        # Generate data
        torch.manual_seed(42)
        X = torch.randn(50, 1)
        y = torch.sin(X) + torch.randn_like(X) * 0.1
        
        # Model
        model = BayesianPINN(input_dim=1, hidden_dims=[16], output_dim=1)
        vi = VariationalInference(model, n_data=len(X))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Training
        initial_loss = None
        for epoch in range(10):
            optimizer.zero_grad()
            predictions, kl = model(X)
            loss, _, _ = vi.elbo_loss(predictions, y, kl)
            loss.backward()
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss.item()
        
        # Loss should decrease
        final_loss = loss.item()
        assert final_loss < initial_loss
    
    def test_ensemble_training_and_prediction(self):
        """Test ensemble training and prediction."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Generate data
        torch.manual_seed(42)
        X = torch.randn(50, 1)
        y = torch.sin(X) + torch.randn_like(X) * 0.1
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        
        # Create and train ensemble
        def factory():
            return nn.Sequential(
                nn.Linear(1, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
        
        ensemble = DeepEnsemble(base_model=factory, n_models=2, device="cpu")
        ensemble.train(loader, epochs=5, verbose=False)
        
        # Predict
        X_test = torch.randn(10, 1)
        mean, std, _ = ensemble.predict(X_test)
        
        assert mean.shape == (10, 1)
        assert std.shape == (10, 1)
        assert (std > 0).all()
