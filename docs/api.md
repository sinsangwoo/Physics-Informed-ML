# API Documentation

## Overview

The Physics-Informed ML API provides RESTful endpoints for neural operator inference. Built with FastAPI, it offers:

- **High Performance**: Async endpoints with GPU acceleration
- **Type Safety**: Pydantic models for request/response validation
- **Auto Documentation**: Interactive Swagger UI and ReDoc
- **Production Ready**: Docker, health checks, structured logging

## Getting Started

### Starting the Server

**Development:**
```bash
uvicorn physics_informed_ml.api.main:app --reload
```

**Production:**
```bash
uvicorn physics_informed_ml.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Docker:**
```bash
docker-compose up api-cpu
```

### Interactive Documentation

Once the server is running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Reference

### Base URL

```
http://localhost:8000
```

### Authentication

Currently no authentication (development only). 
For production, implement API keys or OAuth2.

## Endpoints

### GET `/`

Root endpoint with API information.

**Response:**
```json
{
  "name": "Physics-Informed ML API",
  "version": "0.1.0",
  "description": "Neural operators for real-time PDE solving",
  "docs": "/docs",
  "health": "/health"
}
```

### GET `/health`

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "device": "cuda:0",
  "loaded_models": ["heat_equation_fno", "burgers_pinn"]
}
```

### POST `/models/load`

Load a pre-trained model.

**Parameters:**
- `model_path` (string): Path to model checkpoint
- `model_name` (string): Name to register model

**Example:**
```bash
curl -X POST "http://localhost:8000/models/load?model_path=/models/fno.pth&model_name=my_fno"
```

**Response:**
```json
{
  "status": "success",
  "model_name": "my_fno"
}
```

### GET `/models/{model_name}`

Get model metadata.

**Example:**
```bash
curl http://localhost:8000/models/my_fno
```

**Response:**
```json
{
  "name": "my_fno",
  "type": "FNO1d",
  "parameters": 125000,
  "input_shape": [64, 1],
  "output_shape": [64, 1],
  "device": "cuda:0"
}
```

### POST `/predict`

Single inference endpoint.

**Request Body:**
```json
{
  "model_name": "my_fno",
  "input_data": [[0.5, 0.3, 0.1, 0.0, -0.1]]
}
```

**Response:**
```json
{
  "prediction": [[0.4, 0.25, 0.08, 0.01, -0.02]],
  "model_name": "my_fno",
  "inference_time_ms": 2.5,
  "input_shape": [1, 5],
  "output_shape": [1, 5]
}
```

### POST `/predict/batch`

Batch inference for multiple samples.

**Request Body:**
```json
{
  "model_name": "my_fno",
  "input_data_list": [
    [[0.5, 0.3, 0.1]],
    [[0.8, 0.6, 0.4]],
    [[0.2, 0.1, 0.0]]
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": [[0.4, 0.25, 0.08]],
      "inference_time_ms": 2.1
    },
    {
      "prediction": [[0.7, 0.55, 0.35]],
      "inference_time_ms": 2.0
    },
    {
      "prediction": [[0.15, 0.08, 0.01]],
      "inference_time_ms": 2.2
    }
  ]
}
```

## Error Responses

### 400 Bad Request

Invalid input data.

```json
{
  "detail": "Invalid input: Input data cannot be empty"
}
```

### 404 Not Found

Model not found.

```json
{
  "detail": "Model 'nonexistent' not found"
}
```

### 500 Internal Server Error

Server error.

```json
{
  "detail": "Prediction failed: CUDA out of memory"
}
```

### 503 Service Unavailable

Inference engine not initialized.

```json
{
  "detail": "Inference engine not initialized"
}
```

## Data Formats

### Input Data

All inputs must be 2D arrays (lists of lists):

```python
# Single sample with 5 features
input_data = [[0.5, 0.3, 0.1, 0.0, -0.1]]

# Multiple samples (batch of 3)
input_data = [
    [[0.5, 0.3, 0.1]],
    [[0.8, 0.6, 0.4]],
    [[0.2, 0.1, 0.0]]
]
```

### Output Data

Outputs maintain the same format:

```python
prediction = [[0.4, 0.25, 0.08, 0.01, -0.02]]
```

## Performance

### Benchmarks

Typical performance on NVIDIA A100:

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Single inference | 2-5ms | 200-500 req/s |
| Batch (10 samples) | 8-15ms | 600-1000 samples/s |
| Model loading | 50-200ms | - |

### Optimization Tips

1. **Use Batch Endpoint**: 5-10x better throughput
2. **Keep-Alive Connections**: Reuse HTTP connections
3. **GPU Acceleration**: Enable CUDA for 100x speedup
4. **Multiple Workers**: Scale with `--workers N`

## Deployment

See [examples/api/README.md](../examples/api/README.md) for detailed deployment guides.
