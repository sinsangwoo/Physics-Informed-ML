"""Python client example for Physics-Informed ML API.

Demonstrates:
- Model loading
- Single and batch inference
- Error handling
- Async requests
"""

import requests
import numpy as np
import time
from typing import List, Dict, Any


class PhysicsMLClient:
    """Client for Physics-Informed ML API.
    
    Usage:
        client = PhysicsMLClient("http://localhost:8000")
        client.load_model("/path/to/model.pth", "my_model")
        result = client.predict("my_model", [[0.5, 0.3, 0.1]])
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client.
        
        Args:
            base_url: API base URL
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()  # Reuse connection
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health.
        
        Returns:
            Health status
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """Load a pre-trained model.
        
        Args:
            model_path: Path to model checkpoint
            model_name: Name to register model
            
        Returns:
            Loading status
        """
        response = self.session.post(
            f"{self.base_url}/models/load",
            params={
                "model_path": model_path,
                "model_name": model_name,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
        """
        response = self.session.get(f"{self.base_url}/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def predict(self, model_name: str, input_data: List[List[float]]) -> Dict[str, Any]:
        """Run single inference.
        
        Args:
            model_name: Name of model to use
            input_data: Input data as 2D list
            
        Returns:
            Prediction with metadata
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json={
                "model_name": model_name,
                "input_data": input_data,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, model_name: str, input_data_list: List[List[List[float]]]) -> List[Dict[str, Any]]:
        """Run batch inference.
        
        Args:
            model_name: Name of model to use
            input_data_list: List of input samples
            
        Returns:
            List of predictions
        """
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={
                "model_name": model_name,
                "input_data_list": input_data_list,
            },
        )
        response.raise_for_status()
        return response.json()["predictions"]


def example_single_inference():
    """Example: Single inference."""
    print("="*70)
    print("Example 1: Single Inference")
    print("="*70)
    
    client = PhysicsMLClient()
    
    # Check health
    health = client.health_check()
    print(f"\nAPI Status: {health['status']}")
    print(f"Device: {health['device']}")
    
    # Note: Uncomment below lines when you have a trained model
    # client.load_model("/path/to/model.pth", "heat_equation_fno")
    # result = client.predict("heat_equation_fno", [[0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])
    # print(f"\nInference time: {result['inference_time_ms']:.2f}ms")
    # print(f"Prediction: {result['prediction'][:5]}...")  # First 5 values


def example_batch_inference():
    """Example: Batch inference for better throughput."""
    print("\n" + "="*70)
    print("Example 2: Batch Inference")
    print("="*70)
    
    client = PhysicsMLClient()
    
    # Generate multiple samples
    n_samples = 10
    input_data_list = [
        [[np.sin(i * 0.1 + j * 0.01) for j in range(64)]]
        for i in range(n_samples)
    ]
    
    print(f"\nPrepared {n_samples} samples for batch inference")
    print("Note: Uncomment API calls when model is loaded")
    
    # Note: Uncomment below lines when you have a trained model
    # start_time = time.time()
    # results = client.predict_batch("heat_equation_fno", input_data_list)
    # elapsed = time.time() - start_time
    # print(f"\nProcessed {n_samples} samples in {elapsed:.3f}s")
    # print(f"Throughput: {n_samples/elapsed:.1f} samples/sec")
    # avg_time = np.mean([r['inference_time_ms'] for r in results])
    # print(f"Average inference time: {avg_time:.2f}ms")


def example_error_handling():
    """Example: Error handling."""
    print("\n" + "="*70)
    print("Example 3: Error Handling")
    print("="*70)
    
    client = PhysicsMLClient()
    
    # Try to use nonexistent model
    try:
        client.predict("nonexistent_model", [[0.5]])
    except requests.HTTPError as e:
        print(f"\nExpected error caught: {e}")
        print(f"Status code: {e.response.status_code}")
        print(f"Detail: {e.response.json()['detail']}")
    
    # Try invalid input
    try:
        client.predict("some_model", [[]])  # Empty input
    except requests.HTTPError as e:
        print(f"\nValidation error caught")
        print(f"Status code: {e.response.status_code}")


def example_model_comparison():
    """Example: Compare multiple models."""
    print("\n" + "="*70)
    print("Example 4: Model Comparison")
    print("="*70)
    
    client = PhysicsMLClient()
    
    print("\nNote: This example shows structure for model comparison")
    print("Uncomment API calls when models are loaded")
    
    # Note: Uncomment below lines when you have trained models
    # models = {
    #     "fno": "/path/to/fno_model.pth",
    #     "pinn": "/path/to/pinn_model.pth",
    # }
    # 
    # for name, path in models.items():
    #     client.load_model(path, name)
    #     info = client.get_model_info(name)
    #     print(f"\n{name.upper()}:")
    #     print(f"  Type: {info['type']}")
    #     print(f"  Parameters: {info['parameters']:,}")
    # 
    # input_data = [[0.5] * 64]
    # for name in models.keys():
    #     result = client.predict(name, input_data)
    #     print(f"\n{name.upper()} Inference: {result['inference_time_ms']:.2f}ms")


if __name__ == "__main__":
    print("\nPhysics-Informed ML API Client Examples")
    print("Note: Make sure API server is running at http://localhost:8000\n")
    
    # Run examples
    example_single_inference()
    example_batch_inference()
    example_error_handling()
    example_model_comparison()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
