"""Tests for REST API."""

import pytest
from fastapi.testclient import TestClient
import torch
import tempfile
from pathlib import Path

from physics_informed_ml.api.main import app
from physics_informed_ml.models import FNO1d


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_model():
    """Create a sample FNO model for testing."""
    model = FNO1d(modes=8, width=16, n_layers=2)
    return model


@pytest.fixture
def model_checkpoint(sample_model, tmp_path):
    """Save model checkpoint for loading tests."""
    checkpoint_path = tmp_path / "test_model.pth"
    
    checkpoint = {
        "model": sample_model,
        "metadata": {
            "input_shape": [32, 1],
            "output_shape": [32, 1],
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "gpu_available" in data
        assert "device" in data
        assert "loaded_models" in data
        
        assert data["status"] == "healthy"
        assert isinstance(data["gpu_available"], bool)
        assert isinstance(data["loaded_models"], list)


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self, client):
        """Test API root."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestModelEndpoints:
    """Test model management endpoints."""
    
    def test_load_model(self, client, model_checkpoint):
        """Test model loading."""
        response = client.post(
            "/models/load",
            params={
                "model_path": model_checkpoint,
                "model_name": "test_fno",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["model_name"] == "test_fno"
    
    def test_load_nonexistent_model(self, client):
        """Test loading nonexistent model."""
        response = client.post(
            "/models/load",
            params={
                "model_path": "/nonexistent/model.pth",
                "model_name": "fake_model",
            },
        )
        
        assert response.status_code == 500
    
    def test_get_model_info(self, client, model_checkpoint):
        """Test getting model info."""
        # First load a model
        client.post(
            "/models/load",
            params={
                "model_path": model_checkpoint,
                "model_name": "test_fno",
            },
        )
        
        # Get info
        response = client.get("/models/test_fno")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "test_fno"
        assert "type" in data
        assert "parameters" in data
        assert "device" in data
    
    def test_get_nonexistent_model_info(self, client):
        """Test getting info for nonexistent model."""
        response = client.get("/models/nonexistent")
        
        assert response.status_code == 404


class TestInferenceEndpoints:
    """Test inference endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_model(self, client, model_checkpoint):
        """Load a model before each test."""
        client.post(
            "/models/load",
            params={
                "model_path": model_checkpoint,
                "model_name": "test_fno",
            },
        )
    
    def test_single_inference(self, client):
        """Test single inference."""
        input_data = [[0.5] * 32]  # 32 features
        
        response = client.post(
            "/predict",
            json={
                "model_name": "test_fno",
                "input_data": input_data,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "model_name" in data
        assert "inference_time_ms" in data
        assert "input_shape" in data
        assert "output_shape" in data
        
        assert data["model_name"] == "test_fno"
        assert data["inference_time_ms"] > 0
        assert isinstance(data["prediction"], list)
    
    def test_inference_with_invalid_model(self, client):
        """Test inference with nonexistent model."""
        response = client.post(
            "/predict",
            json={
                "model_name": "nonexistent",
                "input_data": [[0.5]],
            },
        )
        
        assert response.status_code == 404
    
    def test_inference_with_empty_input(self, client):
        """Test inference with empty input."""
        response = client.post(
            "/predict",
            json={
                "model_name": "test_fno",
                "input_data": [[]],
            },
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_inference(self, client):
        """Test batch inference."""
        input_data_list = [
            [[0.5] * 32],
            [[0.8] * 32],
            [[0.3] * 32],
        ]
        
        response = client.post(
            "/predict/batch",
            json={
                "model_name": "test_fno",
                "input_data_list": input_data_list,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        
        # Check each prediction
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "inference_time_ms" in pred
