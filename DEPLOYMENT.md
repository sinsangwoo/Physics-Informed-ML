# Deployment Guide - Phase 4.2 Complete

## 🚀 Quick Start

### Option 1: Vercel (Frontend Only)

```bash
cd frontend
npm install -g vercel
vercel
```

### Option 2: Full Stack (Docker)

```bash
# Start everything
docker-compose up -d

# Frontend: http://localhost:3000
# API: http://localhost:8000
```

### Option 3: Serverless (AWS Lambda)

```bash
cd infrastructure/terraform
terraform init
terraform apply
```

---

## 🌐 Deployment Options

### Frontend Deployment

#### Vercel (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd frontend
vercel --prod

# Set environment variables
vercel env add VITE_API_URL
```

**Secrets to configure:**
- `VERCEL_TOKEN`: Your Vercel API token
- `VERCEL_ORG_ID`: Organization ID
- `VERCEL_PROJECT_ID`: Project ID
- `API_URL`: Backend API endpoint

#### Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
cd frontend
netlify deploy --prod
```

**Secrets to configure:**
- `NETLIFY_AUTH_TOKEN`: Authentication token
- `NETLIFY_SITE_ID`: Site identifier

#### Cloudflare Pages

```bash
# Build
cd frontend
npm run build

# Deploy with Wrangler
npm install -g wrangler
wrangler pages publish dist
```

---

### Backend Deployment

#### AWS Lambda (Serverless)

```bash
# Package dependencies
pip install -r requirements-lambda.txt -t package/
cd package && zip -r ../lambda-deployment.zip . && cd ..
zip -g lambda-deployment.zip handler.py

# Deploy with Terraform
cd infrastructure/terraform
terraform init
terraform plan
terraform apply

# Or use AWS CLI
aws lambda update-function-code \
  --function-name physics-ml-api \
  --zip-file fileb://lambda-deployment.zip
```

**Cost estimate:** $0-5/month for hobby use

#### GCP Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/physics-ml-api

# Deploy
gcloud run deploy physics-ml-api \
  --image gcr.io/PROJECT_ID/physics-ml-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

**Cost estimate:** Pay per request, ~$0.10/100K requests

#### AWS ECS Fargate

```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin ECR_URI
docker tag physics-ml-api:latest ECR_URI/physics-ml-api:latest
docker push ECR_URI/physics-ml-api:latest

# Create ECS service
aws ecs create-service \
  --cluster physics-ml \
  --service-name api \
  --task-definition physics-ml-api:1 \
  --desired-count 2
```

---

### Edge Deployment (Cloudflare Workers)

```bash
cd infrastructure/cloudflare

# Configure
wrangler login

# Deploy
wrangler deploy

# Test
curl https://physics-ml.your-domain.workers.dev/health
```

**Benefits:**
- 200+ edge locations worldwide
- Sub-10ms latency
- Free tier: 100K requests/day

---

### Kubernetes

#### CPU Deployment

```bash
# Apply manifests
kubectl apply -f infrastructure/kubernetes/deployment.yaml

# Check status
kubectl get pods
kubectl get svc physics-ml-api

# Get external IP
kubectl get svc physics-ml-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

#### GPU Deployment

```bash
# Create GPU node pool first (GKE example)
gcloud container node-pools create gpu-pool \
  --cluster=physics-ml \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type=n1-standard-4 \
  --num-nodes=2

# Deploy GPU workload
kubectl apply -f infrastructure/kubernetes/gpu-deployment.yaml
```

---

## 📊 Monitoring

### Prometheus + Grafana

```bash
# Install with Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack

# Port forward Grafana
kubectl port-forward svc/prometheus-grafana 3000:80

# Import dashboard
# Dashboard ID: see infrastructure/monitoring/grafana-dashboard.json
```

### CloudWatch (AWS)

```bash
# View Lambda logs
aws logs tail /aws/lambda/physics-ml-api --follow

# Create alarm
aws cloudwatch put-metric-alarm \
  --alarm-name high-error-rate \
  --metric-name Errors \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold
```

### Cloud Logging (GCP)

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50

# Create alert
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --condition-threshold-value=10 \
  --condition-filter='metric.type="run.googleapis.com/request_count"'
```

---

## 🔒 Security

### API Authentication

```python
# Add to main.py
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(
    request: InferenceRequest,
    credentials = Depends(security)
):
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(401)
    ...
```

### HTTPS/TLS

```bash
# Let's Encrypt with Certbot
sudo certbot --nginx -d api.physics-ml.com

# Or use cloud provider's managed certificates
# (automatic with Vercel, Netlify, Cloud Run)
```

### Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("60/minute")
async def predict(request: Request, ...):
    ...
```

---

## 💰 Cost Optimization

### Tips

1. **Use serverless for variable loads**
   - Lambda: Pay per request
   - Cloud Run: Scale to zero

2. **Cache aggressively**
   - CDN for frontend
   - Model predictions in Redis
   - Edge caching with Cloudflare

3. **Right-size resources**
   - Start small, monitor, scale up
   - Use spot instances for batch jobs

4. **Optimize cold starts**
   - Minimize dependencies
   - Use provisioned concurrency for critical endpoints

### Cost Comparison (Monthly)

| Platform | Free Tier | Low Traffic | High Traffic |
|----------|-----------|-------------|-------------|
| Vercel | Yes | $0 | $20 |
| Netlify | Yes | $0 | $19 |
| AWS Lambda | 1M requests | $0-5 | $50-200 |
| GCP Cloud Run | 2M requests | $0-5 | $40-150 |
| Cloudflare Workers | 100K/day | $0 | $5 |
| ECS Fargate | No | $30 | $100+ |
| GKE | No | $70 | $200+ |

---

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] Secrets stored securely (not in code)
- [ ] HTTPS enabled
- [ ] Authentication/authorization implemented
- [ ] Rate limiting configured
- [ ] Monitoring and alerting set up
- [ ] Backup strategy defined
- [ ] Scaling limits configured
- [ ] Cost alerts enabled
- [ ] Error tracking (Sentry, etc.)
- [ ] Load testing performed
- [ ] Disaster recovery plan documented

---

## 🆘 Troubleshooting

### Lambda Cold Start

**Problem:** First request takes >5s

**Solution:**
```python
# Use provisioned concurrency
aws lambda put-provisioned-concurrency-config \
  --function-name physics-ml-api \
  --provisioned-concurrent-executions 2

# Or reduce package size
# - Remove unused dependencies
# - Use Lambda layers for large libraries
```

### OOM Errors

**Problem:** Lambda/Container runs out of memory

**Solution:**
- Increase memory allocation
- Lazy load models
- Clear cache between requests
- Use model quantization

### Slow Inference

**Problem:** Predictions take too long

**Solution:**
- Use GPU instances
- Batch requests
- Model optimization (quantization, pruning)
- Cache frequent predictions

---

**Status:** Phase 4.2 COMPLETE ✅
