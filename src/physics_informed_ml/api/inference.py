"""Inference engine for model loading and prediction.

Handles:
- Model loading and caching
- Inference with timing
- Batch processing
- Device management
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
import time
import asyncio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages model inference.
    
    Features:
    - Model caching (avoid reloading)
    - Automatic device placement
    - Batched inference
    - Timing and profiling
    """
    
    def __init__(self, device: str | None = None):
        """Initialize inference engine.
        
        Args:
            device: Device to use ("cpu", "cuda", "cuda:0", etc.)
                   If None, automatically selects GPU if available
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.models: Dict[str, nn.Module] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"InferenceEngine initialized on {self.device}")
    
    def load_model(self, model_path: str, model_name: str) -> None:
        """Load a model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            model_name: Name to register model under
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model and metadata
            if isinstance(checkpoint, dict):
                model = checkpoint.get("model")
                metadata = checkpoint.get("metadata", {})
            else:
                # Assume it's just the model state dict
                model = checkpoint
                metadata = {}
            
            # Move to device and set to eval mode
            if isinstance(model, nn.Module):
                model = model.to(self.device)
                model.eval()
            else:
                raise ValueError("Checkpoint must contain a PyTorch model")
            
            # Store model and info
            self.models[model_name] = model
            self.model_info[model_name] = {
                "type": model.__class__.__name__,
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": str(self.device),
                **metadata,
            }
            
            logger.info(f"Loaded model '{model_name}' on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get metadata for a loaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata dictionary
            
        Raises:
            KeyError: If model not found
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not loaded")
        
        return {
            "name": model_name,
            **self.model_info[model_name],
        }
    
    async def predict(
        self, model_name: str, input_data: List[List[float]]
    ) -> Dict[str, Any]:
        """Run single inference.
        
        Args:
            model_name: Name of model to use
            input_data: Input as 2D list
            
        Returns:
            Prediction with metadata
            
        Raises:
            KeyError: If model not found
            ValueError: If input shape invalid
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        
        # Convert input to tensor
        try:
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            input_tensor = input_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Invalid input data: {str(e)}")
        
        # Run inference with timing
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Synchronize if using GPU (important for accurate timing)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Convert output to list
        prediction = output.cpu().numpy().tolist()
        
        return {
            "prediction": prediction,
            "model_name": model_name,
            "inference_time_ms": inference_time,
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output.shape),
        }
    
    async def predict_batch(
        self, model_name: str, input_data_list: List[List[List[float]]]
    ) -> List[Dict[str, Any]]:
        """Run batch inference.
        
        Processes multiple samples concurrently.
        
        Args:
            model_name: Name of model to use
            input_data_list: List of input samples
            
        Returns:
            List of predictions
        """
        # Run predictions concurrently
        tasks = [
            self.predict(model_name, input_data)
            for input_data in input_data_list
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up inference engine...")
        
        # Clear models
        self.models.clear()
        self.model_info.clear()
        
        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Cleanup complete")
