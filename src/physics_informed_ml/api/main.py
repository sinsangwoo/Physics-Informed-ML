"""FastAPI application for neural operator inference.

Provides RESTful API for:
- Model loading and management
- Single and batch inference
- Health checks and metrics
- Model metadata
- WebSocket streaming
"""

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
from typing import Dict, Any
import logging
import json

from physics_informed_ml.api.models import (
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    HealthResponse,
    ModelInfo,
)
from physics_informed_ml.api.inference import InferenceEngine
from physics_informed_ml.api.websocket import manager, handle_streaming_inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference engine
inference_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles:
    - Model loading on startup
    - Resource cleanup on shutdown
    """
    global inference_engine
    
    # Startup
    logger.info("Starting Physics-Informed ML API...")
    inference_engine = InferenceEngine()
    logger.info("Inference engine initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Physics-Informed ML API...")
    if inference_engine:
        inference_engine.cleanup()
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="Physics-Informed ML API",
    description="REST API for neural operator and PINN inference",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Physics-Informed ML API",
        "version": "0.1.0",
        "description": "Neural operators for real-time PDE solving",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns system status including:
    - API status
    - GPU availability
    - Loaded models
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        device=str(inference_engine.device),
        loaded_models=list(inference_engine.models.keys()),
    )


@app.post("/models/load")
async def load_model(model_path: str, model_name: str):
    """Load a pre-trained model.
    
    Args:
        model_path: Path to model checkpoint
        model_name: Name to register model under
        
    Returns:
        Model loading status
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    try:
        inference_engine.load_model(model_path, model_name)
        return {"status": "success", "model_name": model_name}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a loaded model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model metadata
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    if model_name not in inference_engine.models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )
    
    return inference_engine.get_model_info(model_name)


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Single inference endpoint.
    
    Accepts input data and returns model prediction.
    
    Args:
        request: Input data and model specification
        
    Returns:
        Model prediction with metadata
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    try:
        result = await inference_engine.predict(
            model_name=request.model_name,
            input_data=request.input_data,
        )
        return InferenceResponse(**result)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchInferenceResponse)
async def predict_batch(request: BatchInferenceRequest):
    """Batch inference endpoint.
    
    Processes multiple inputs simultaneously for better throughput.
    
    Args:
        request: Batch of input data
        
    Returns:
        Batch of predictions
    """
    if not inference_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    try:
        results = await inference_engine.predict_batch(
            model_name=request.model_name,
            input_data_list=request.input_data_list,
        )
        return BatchInferenceResponse(predictions=results)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming.
    
    Accepts:
        {
            "action": "stream",
            "model_name": "heat_equation_fno",
            "input_data": [0.5, 0.3, ...],
            "time_steps": 50
        }
    
    Returns:
        Stream of predictions with progress updates
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "stream":
                if not inference_engine:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "Inference engine not initialized"
                    }, websocket)
                    continue
                
                # Start streaming inference
                await handle_streaming_inference(
                    websocket=websocket,
                    model_name=message["model_name"],
                    input_data=message["input_data"],
                    time_steps=message["time_steps"],
                    inference_engine=inference_engine,
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
