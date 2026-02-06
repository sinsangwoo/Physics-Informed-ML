"""Pydantic models for API request/response validation.

Defines schemas for:
- Inference requests (single and batch)
- Model metadata
- Health checks
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import numpy as np


class InferenceRequest(BaseModel):
    """Single inference request.
    
    Example:
        {
            "model_name": "heat_equation_fno",
            "input_data": [[0.1, 0.2, 0.3, ...]]  # 2D list
        }
    """
    
    model_name: str = Field(
        ...,
        description="Name of the model to use for inference",
        examples=["heat_equation_fno", "burgers_pinn"],
    )
    
    input_data: List[List[float]] = Field(
        ...,
        description="Input data as 2D array (batch_size x features)",
        examples=[[[0.1, 0.2, 0.3]]],
    )
    
    @field_validator("input_data")
    @classmethod
    def validate_input_shape(cls, v):
        """Validate input data is not empty."""
        if not v or not v[0]:
            raise ValueError("Input data cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "heat_equation_fno",
                "input_data": [[0.5, 0.3, 0.1, 0.0, -0.1]],
            }
        }


class InferenceResponse(BaseModel):
    """Single inference response.
    
    Returns prediction with timing information.
    """
    
    prediction: List[List[float]] = Field(
        ...,
        description="Model prediction as 2D array",
    )
    
    model_name: str = Field(
        ...,
        description="Name of model used",
    )
    
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
        ge=0.0,
    )
    
    input_shape: List[int] = Field(
        ...,
        description="Shape of input data",
    )
    
    output_shape: List[int] = Field(
        ...,
        description="Shape of output data",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": [[0.4, 0.25, 0.08, 0.01, -0.02]],
                "model_name": "heat_equation_fno",
                "inference_time_ms": 2.5,
                "input_shape": [1, 5],
                "output_shape": [1, 5],
            }
        }


class BatchInferenceRequest(BaseModel):
    """Batch inference request.
    
    Processes multiple samples efficiently.
    """
    
    model_name: str = Field(
        ...,
        description="Name of the model to use",
    )
    
    input_data_list: List[List[List[float]]] = Field(
        ...,
        description="List of input samples, each as 2D array",
        min_length=1,
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "heat_equation_fno",
                "input_data_list": [
                    [[0.5, 0.3, 0.1]],
                    [[0.8, 0.6, 0.4]],
                ],
            }
        }


class BatchInferenceResponse(BaseModel):
    """Batch inference response."""
    
    predictions: List[Dict[str, Any]] = Field(
        ...,
        description="List of predictions with metadata",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": [[0.4, 0.25, 0.08]],
                        "inference_time_ms": 2.1,
                    },
                    {
                        "prediction": [[0.7, 0.55, 0.35]],
                        "inference_time_ms": 2.0,
                    },
                ],
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        ...,
        description="Service status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    
    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available",
    )
    
    device: str = Field(
        ...,
        description="Device being used (cpu/cuda)",
    )
    
    loaded_models: List[str] = Field(
        default_factory=list,
        description="List of loaded model names",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "gpu_available": True,
                "device": "cuda:0",
                "loaded_models": ["heat_equation_fno", "burgers_pinn"],
            }
        }


class ModelInfo(BaseModel):
    """Model metadata."""
    
    name: str = Field(
        ...,
        description="Model name",
    )
    
    type: str = Field(
        ...,
        description="Model type (FNO, PINN, etc.)",
    )
    
    parameters: int = Field(
        ...,
        description="Number of trainable parameters",
        ge=0,
    )
    
    input_shape: Optional[List[int]] = Field(
        None,
        description="Expected input shape",
    )
    
    output_shape: Optional[List[int]] = Field(
        None,
        description="Expected output shape",
    )
    
    device: str = Field(
        ...,
        description="Device model is loaded on",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "heat_equation_fno",
                "type": "FNO1d",
                "parameters": 125000,
                "input_shape": [64, 1],
                "output_shape": [64, 1],
                "device": "cuda:0",
            }
        }
