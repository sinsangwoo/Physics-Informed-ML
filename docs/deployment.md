# Deployment Guide

## 🚀 Production Deployment

### Overview

The Physics-Informed ML stack consists of three components:

1. **Backend API**: FastAPI serving neural operator models
2. **Frontend**: React app with 3D visualization
3. **Database** (optional): For storing simulation results

## 📦 Backend Deployment

### Docker (Recommended)

```bash
# CPU-only deployment
docker-compose up -d api-cpu

# GPU-enabled deployment
docker-compose --profile gpu up -d api-gpu

# Scale to multiple workers
docker-compose up -d --scale api-cpu=4
```

### Manual Deployment

```bash
# Install dependencies
pip install -e ".[api]"

# Run with Uvicorn
uvicorn physics_informed_ml.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4

# Production with Gunicorn
gunicorn physics_informed_ml.api.main:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: physics-ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: physics-ml-api
  template:
    metadata:
      labels:
        app: physics-ml-api
    spec:
      containers:
      - name: api
        image: physics-ml-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: physics-ml-api
spec:
  selector:
    app: physics-ml-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Apply:
```bash
kubectl apply -f deployment.yaml
```

## 🎨 Frontend Deployment

### Build for Production

```bash
cd frontend

# Install dependencies
npm install

# Build optimized bundle
npm run build

# Preview build
npm run preview
```

### Serve with Nginx

```nginx
# /etc/nginx/sites-available/physics-ml
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/physics-ml/dist;
    index index.html;

    # Frontend
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript;
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/physics-ml /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd frontend
vercel
```

Configure `vercel.json`:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "https://your-api.com/$1"
    },
    {
      "source": "/(.*)",
      "destination": "/"
    }
  ]
}
```

## ☁️ Cloud Platforms

### AWS

**ECS Fargate:**
```bash
# Build and push image
docker build -t physics-ml-api .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URI
docker tag physics-ml-api:latest YOUR_ECR_URI/physics-ml-api:latest
docker push YOUR_ECR_URI/physics-ml-api:latest

# Create ECS service
aws ecs create-service \
  --cluster physics-ml-cluster \
  --service-name physics-ml-api \
  --task-definition physics-ml-api:1 \
  --desired-count 3
```

**S3 + CloudFront (Frontend):**
```bash
# Deploy frontend
cd frontend
npm run build
aws s3 sync dist/ s3://your-bucket/ --delete
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

### Google Cloud Platform

**Cloud Run:**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT/physics-ml-api
gcloud run deploy physics-ml-api \
  --image gcr.io/YOUR_PROJECT/physics-ml-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**Firebase Hosting (Frontend):**
```bash
cd frontend
npm run build
firebase init hosting
firebase deploy
```

### Azure

**Container Instances:**
```bash
# Deploy API
az container create \
  --resource-group physics-ml-rg \
  --name physics-ml-api \
  --image YOUR_REGISTRY/physics-ml-api:latest \
  --cpu 2 --memory 4 \
  --ports 8000
```

**Static Web Apps (Frontend):**
```bash
az staticwebapp create \
  --name physics-ml-frontend \
  --resource-group physics-ml-rg \
  --source frontend \
  --location "Central US" \
  --branch main
```

## 🔒 Security

### API Authentication

Add to `api/main.py`:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/predict")
async def predict(
    request: InferenceRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify API key
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    ...
```

### HTTPS

Use Let's Encrypt:
```bash
sudo certbot --nginx -d your-domain.com
```

### Rate Limiting

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

## 📊 Monitoring

### Prometheus

Add metrics:
```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(...):
    with prediction_latency.time():
        result = await engine.predict(...)
    prediction_counter.inc()
    return result
```

### Grafana Dashboard

Example queries:
```promql
# Request rate
rate(predictions_total[5m])

# Average latency
rate(prediction_latency_seconds_sum[5m]) / rate(prediction_latency_seconds_count[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m])
```

## 🐛 Troubleshooting

### Check logs

```bash
# Docker
docker-compose logs -f api-cpu

# Kubernetes
kubectl logs -f deployment/physics-ml-api

# Systemd
journalctl -u physics-ml-api -f
```

### Health check

```bash
curl http://localhost:8000/health
```

### Performance testing

```bash
# Load test with Apache Bench
ab -n 1000 -c 10 -p request.json -T application/json http://localhost:8000/predict

# Or with hey
hey -n 1000 -c 10 -m POST -D request.json http://localhost:8000/predict
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up -d --scale api-cpu=5

# Kubernetes
kubectl scale deployment physics-ml-api --replicas=5

# Auto-scaling
kubectl autoscale deployment physics-ml-api --min=3 --max=10 --cpu-percent=70
```

### Caching

Add Redis for prediction caching:

```python
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379)

@app.post("/predict")
async def predict(request: InferenceRequest):
    # Generate cache key
    key = hashlib.md5(json.dumps(request.dict()).encode()).hexdigest()
    
    # Check cache
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    # Predict and cache
    result = await engine.predict(...)
    redis_client.setex(key, 3600, json.dumps(result))  # 1 hour TTL
    return result
```
