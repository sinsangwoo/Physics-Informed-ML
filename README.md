# Physics-Informed ML: Neural Operators for Real-Time Simulation

[![CI](https://github.com/sinsangwoo/Physics-Informed-ML/workflows/CI/badge.svg)](https://github.com/sinsangwoo/Physics-Informed-ML/actions)
[![codecov](https://codecov.io/gh/sinsangwoo/Physics-Informed-ML/branch/main/graph/badge.svg)](https://codecov.io/gh/sinsangwoo/Physics-Informed-ML)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready framework for Physics-Informed Neural Networks with uncertainty quantification, real-time 3D visualization, and automated deployment pipelines.**

## 🎯 Vision

Complete ML-powered physics simulation platform from research to production:

- **Speed**: 100-1000x faster than traditional PDE solvers
- **Accuracy**: Physics constraints + uncertainty quantification
- **Scalability**: Auto-scaling cloud deployment
- **Interactivity**: Real-time 3D visualization with WebSocket
- **Reliability**: Calibrated uncertainty estimates

## 🚀 Key Features

### Core ML
- **Neural Operators (FNO)**: Resolution-invariant learning
- **Physics-Informed Neural Networks**: PDE-constrained training
- **Uncertainty Quantification**: Bayesian + Ensemble + MC Dropout ✨ **NEW**

### Production Stack
- **REST API**: FastAPI with async endpoints
- **WebSocket**: Real-time streaming at 50 FPS
- **Frontend**: React + Three.js 3D visualization
- **Deployment**: Vercel + AWS Lambda + Terraform ✨ **NEW**

### Research Features ✨ **NEW**
- **Bayesian Neural Networks**: Variational inference (ELBO)
- **Deep Ensemble**: Model disagreement quantification
- **Calibration**: ECE, prediction intervals, CRPS
- **Visualization**: Reliability diagrams, uncertainty plots

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/sinsangwoo/Physics-Informed-ML.git
cd Physics-Informed-ML

# Backend
pip install -e \".[dev,api]\"

# Frontend
cd frontend && npm install

# Uncertainty (research features)
pip install -e \".[uncertainty]\"
```

## 💻 Quick Start

### Web Interface (Recommended)

```bash
# Terminal 1: API
uvicorn physics_informed_ml.api.main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

Visit `http://localhost:3000` for interactive visualization!

### Uncertainty Quantification

```python
from physics_informed_ml.uncertainty import BayesianPINN, DeepEnsemble

# Bayesian approach
model = BayesianPINN(input_dim=2, hidden_dims=[64, 64], output_dim=1)
mean, std, samples = model.predict_with_uncertainty(x_test, n_samples=100)

# Ensemble approach
ensemble = DeepEnsemble(base_model=create_model, n_models=5)
ensemble.train(train_loader, epochs=100)
mean, std, _ = ensemble.predict(x_test)
```

## 📊 Benchmarks

| Problem | Traditional | PINN | FNO | Speedup |
|---------|------------|------|-----|---------|
| Heat 1D | 1.0s | 0.5s | 0.002s | **500x** |
| Burgers | 10s | 2s | 0.01s | **1000x** |
| Navier-Stokes 2D | 300s | 30s | 0.5s | **600x** |

### Uncertainty Quantification

```
Bayesian FNO (100 samples):
- Epistemic uncertainty: \u03c3 = 0.05 (in-distribution)
- Epistemic uncertainty: \u03c3 = 0.35 (out-of-distribution)
- Expected Calibration Error: 0.03 (well-calibrated)
- 95% CI Coverage: 94.2% (accurate intervals)
```

## 🗺️ Roadmap

### Phase 0-3: Foundation & API ✅
- [x] Modern Python project structure
- [x] PINN + Neural Operators
- [x] FastAPI REST API
- [x] Docker deployment

### Phase 4: Interactive Frontend ✅
- [x] React + TypeScript + Vite
- [x] Three.js 3D visualization
- [x] WebSocket streaming
- [x] Export functionality

### Phase 4.2: Deployment Pipelines ✅ **COMPLETED**
- [x] Vercel/Netlify frontend deployment
- [x] AWS Lambda + API Gateway
- [x] GCP Cloud Run
- [x] Terraform infrastructure as code
- [x] GitHub Actions CI/CD
- [x] Docker Hub auto-publish

### Phase 5: Research Features ✅ **COMPLETED**
- [x] Bayesian Neural Networks
- [x] Deep Ensemble
- [x] Monte Carlo Dropout
- [x] Calibration metrics (ECE, CRPS)
- [x] Reliability diagrams
- [x] Comprehensive tests & docs

### Phase 6: Advanced Research (Next)
- [ ] Transfer learning for new physics
- [ ] Explainability (attention, gradients)
- [ ] Multi-fidelity learning
- [ ] Active learning strategies

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│   Frontend (Vercel/Netlify)             │
│   • React + TypeScript                  │
│   • Three.js 3D Visualization           │
│   • Real-time WebSocket                 │
└──────────────┬──────────────────────────┘
               │ HTTPS/WSS
┌──────────────▼──────────────────────────┐
│   API Gateway (AWS/GCP)                 │
│   • Load Balancing                      │
│   • Auto-scaling                        │
│   • CORS & Auth                         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   Inference Service                     │
│   • FastAPI Backend                     │
│   • Uncertainty Quantification          │
│   • Model Ensemble                      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   Neural Operators                      │
│   • FNO (Fourier)                       │
│   • PINN (Physics-Informed)             │
│   • Bayesian Layers                     │
└─────────────────────────────────────────┘
```

## 🛠️ Technology Stack

**ML/Scientific:**
- PyTorch 2.1+ (deep learning)
- NumPy, SciPy (numerical computing)
- Matplotlib, Plotly (visualization)

**Backend:**
- FastAPI (async API)
- WebSockets (real-time)
- Pydantic (validation)

**Frontend:**
- React 18 + TypeScript
- Three.js + R3F (3D)
- Recharts (2D plots)
- Tailwind CSS (styling)

**DevOps:**
- GitHub Actions (CI/CD)
- Docker (containerization)
- Terraform (IaC)
- Vercel/AWS/GCP (hosting)

## 📚 Documentation

- [Neural Operators Guide](docs/neural_operators.md)
- [Uncertainty Quantification](docs/uncertainty.md) ✨ **NEW**
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Frontend README](frontend/README.md)

## 🔬 Research

### Uncertainty Quantification

**Three complementary methods:**

1. **Bayesian Neural Networks**
   - Weight uncertainty via variational inference
   - ELBO loss: E[log p(y|x,w)] - KL(q(w) || p(w))
   - Epistemic uncertainty

2. **Deep Ensemble**
   - 5-10 independent models
   - Disagreement = uncertainty
   - Often outperforms Bayesian

3. **Monte Carlo Dropout**
   - Dropout as Bayesian approximation
   - Fast single-model approach
   - Good for prototyping

**Calibration:**
- Expected Calibration Error (ECE)
- Prediction interval coverage
- Continuous Ranked Probability Score (CRPS)
- Reliability diagrams

See [`docs/uncertainty.md`](docs/uncertainty.md) for complete guide.

## 🎓 Examples

Check `examples/` for:

1. **Neural Operators** (`neural_operators/`)
   - FNO benchmark on 1D heat equation
   - Multi-resolution evaluation

2. **API Usage** (`api/`)
   - Python client with batch inference
   - WebSocket streaming

3. **Uncertainty** (`uncertainty/`) ✨ **NEW**
   - Bayesian PINN training
   - Deep ensemble comparison
   - Calibration analysis
   - Outlier detection

## 🚀 Deployment

### Frontend (Vercel)

```bash
cd frontend
vercel --prod
```

### Backend (AWS Lambda)

```bash
# Using Terraform
cd infrastructure/terraform
terraform init
terraform apply

# Or GitHub Actions (auto-deploy on push)
```

### Docker

```bash
docker-compose up -d
```

See [`docs/deployment.md`](docs/deployment.md) for complete guide.

## 🤝 Contributing

We welcome contributions! Areas of interest:

- **Research**: New uncertainty methods, physics problems
- **Engineering**: Performance optimization, new deployment targets
- **Documentation**: Tutorials, examples, papers

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

```bibtex
@software{physics_informed_ml,
  author = {Sin, Sangwoo},
  title = {Physics-Informed ML: Neural Operators with Uncertainty},
  year = {2025},
  url = {https://github.com/sinsangwoo/Physics-Informed-ML}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

**Research:**
- Raissi et al. (2019) - Physics-Informed Neural Networks
- Li et al. (2021) - Fourier Neural Operator  
- Blundell et al. (2015) - Bayesian Deep Learning
- Lakshminarayanan et al. (2017) - Deep Ensembles

**Tools:**
- PyTorch, FastAPI, React, Three.js communities

---

**Built for the future of physics simulation.**  
**Research-grade accuracy. Production-ready deployment. 🚀**
