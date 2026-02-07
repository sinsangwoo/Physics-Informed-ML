"""WebSocket endpoints for real-time streaming inference.

Provides:
- Real-time prediction streaming
- Animation frame updates
- Live performance metrics
"""

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
from typing import Dict, Set
import torch

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections.
    
    Features:
    - Multiple concurrent connections
    - Broadcast to all clients
    - Connection lifecycle management
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send message to specific client."""
        await websocket.send_json(message)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


async def handle_streaming_inference(
    websocket: WebSocket,
    model_name: str,
    input_data: list,
    time_steps: int,
    inference_engine
):
    """Stream predictions frame by frame.
    
    Args:
        websocket: WebSocket connection
        model_name: Model to use
        input_data: Initial conditions
        time_steps: Number of time steps to simulate
        inference_engine: Inference engine instance
    """
    try:
        # Send initial status
        await manager.send_personal_message({
            "type": "status",
            "message": "Starting simulation...",
            "progress": 0,
        }, websocket)
        
        # Convert input
        current_state = torch.tensor([input_data], dtype=torch.float32)
        current_state = current_state.to(inference_engine.device)
        
        predictions = []
        
        # Stream predictions
        for step in range(time_steps):
            # Run inference
            with torch.no_grad():
                if model_name in inference_engine.models:
                    model = inference_engine.models[model_name]
                    output = model(current_state)
                else:
                    raise ValueError(f"Model {model_name} not loaded")
            
            # Convert to list
            prediction = output.cpu().numpy().tolist()[0]
            predictions.append(prediction)
            
            # Send frame
            await manager.send_personal_message({
                "type": "frame",
                "step": step,
                "total_steps": time_steps,
                "prediction": prediction,
                "progress": int((step + 1) / time_steps * 100),
            }, websocket)
            
            # Update state for next iteration
            current_state = output
            
            # Small delay for smoother animation
            await asyncio.sleep(0.02)  # 50 FPS
        
        # Send completion
        await manager.send_personal_message({
            "type": "complete",
            "total_frames": len(predictions),
            "message": "Simulation complete",
        }, websocket)
        
    except Exception as e:
        logger.error(f"Streaming inference error: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": str(e),
        }, websocket)
