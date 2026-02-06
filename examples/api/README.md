# API Usage Examples

This directory contains examples for using the Physics-Informed ML REST API.

## Quick Start

### 1. Start the API Server

```bash
# Development mode (with auto-reload)
uvicorn physics_informed_ml.api.main:app --reload

# Production mode
uvicorn physics_informed_ml.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With Docker
docker-compose up api-cpu

# With GPU
docker-compose --profile gpu up api-gpu
```

The API will be available at `http://localhost:8000`.

### 2. Interactive API Documentation

Open your browser and navigate to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "device": "cuda:0",
  "loaded_models": ["heat_equation_fno"]
}
```

### Load a Model

```bash
curl -X POST "http://localhost:8000/models/load?model_path=/path/to/model.pth&model_name=my_model"
```

### Get Model Info

```bash
curl http://localhost:8000/models/my_model
```

Response:
```json
{
  "name": "my_model",
  "type": "FNO1d",
  "parameters": 125000,
  "input_shape": [64, 1],
  "output_shape": [64, 1],
  "device": "cuda:0"
}
```

### Single Inference

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "heat_equation_fno",
    "input_data": [[0.5, 0.3, 0.1, 0.0, -0.1]]
  }'
```

Response:
```json
{
  "prediction": [[0.4, 0.25, 0.08, 0.01, -0.02]],
  "model_name": "heat_equation_fno",
  "inference_time_ms": 2.5,
  "input_shape": [1, 5],
  "output_shape": [1, 5]
}
```

### Batch Inference

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "heat_equation_fno",
    "input_data_list": [
      [[0.5, 0.3, 0.1]],
      [[0.8, 0.6, 0.4]]
    ]
  }'
```

## Python Client Example

See `client_example.py` for a complete Python client implementation.

## Performance Tips

### 1. Batch Processing

For multiple predictions, use the batch endpoint:

```python
import requests

# Instead of multiple single requests
for data in input_list:
    response = requests.post("/predict", json={...})  # Slow!

# Use batch endpoint
response = requests.post("/predict/batch", json={
    "model_name": "my_model",
    "input_data_list": input_list,  # Fast!
})
```

### 2. Keep-Alive Connections

Reuse connections for better throughput:

```python
import requests

session = requests.Session()  # Reuse connection
for _ in range(100):
    response = session.post("/predict", json={...})
```

### 3. Async Clients

For high concurrency, use async HTTP clients:

```python
import asyncio
import httpx

async def predict_async(client, data):
    response = await client.post("/predict", json=data)
    return response.json()

async def main():
    async with httpx.AsyncClient() as client:
        tasks = [predict_async(client, data) for data in dataset]
        results = await asyncio.gather(*tasks)

asyncio.run(main())
```

## Deployment

### Docker Compose (Recommended)

```bash
# CPU-only
docker-compose up -d api-cpu

# With GPU
docker-compose --profile gpu up -d api-gpu

# Scale to multiple workers
docker-compose up -d --scale api-cpu=4
```

### Kubernetes

See `kubernetes/` directory for deployment manifests.

### Cloud Deployment

**AWS:**
```bash
# ECS with Fargate
aws ecs create-service --cluster ml-cluster --task-definition physics-ml-api
```

**GCP:**
```bash
# Cloud Run
gcloud run deploy physics-ml-api --image gcr.io/PROJECT/physics-ml-api
```

**Azure:**
```bash
# Container Instances
az container create --resource-group ml-rg --name physics-ml-api
```

## Monitoring

### Health Checks

The API includes health check endpoints for orchestration:

```yaml
# Docker Compose
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 3s
  retries: 3
```

### Logging

API uses structured logging:

```bash
# Set log level
export LOG_LEVEL=debug
uvicorn physics_informed_ml.api.main:app --log-level debug
```

### Metrics

For production, integrate with:
- **Prometheus**: Add prometheus-fastapi-instrumentator
- **DataDog**: Use ddtrace
- **New Relic**: Use newrelic agent

## Security

### API Keys (Production)

Add authentication:

```python
# In main.py
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/predict")
async def predict(
    request: InferenceRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # Verify credentials
    ...
```

### Rate Limiting

Use slowapi for rate limiting:

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("60/minute")
async def predict(request: Request, ...):
    ...
```

## Troubleshooting

**Issue: Model loading fails**
```
Solution: Check file path and permissions
Verify model was saved correctly
```

**Issue: Slow inference**
```
Solution: Enable GPU if available
Use batch endpoint for multiple predictions
Check model is in eval() mode
```

**Issue: Out of memory**
```
Solution: Reduce batch size
Clear CUDA cache between requests
Use smaller model or reduce precision (float16)
```
