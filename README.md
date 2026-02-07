# Physics-Informed ML: Neural Operators for Real-Time Simulation

[![CI](https://github.com/sinsangwoo/Physics-Informed-ML/workflows/CI/badge.svg)](https://github.com/sinsangwoo/Physics-Informed-ML/actions)
[![codecov](https://codecov.io/gh/sinsangwoo/Physics-Informed-ML/branch/main/graph/badge.svg)](https://codecov.io/gh/sinsangwoo/Physics-Informed-ML)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-ready framework for Physics-Informed Neural Networks (PINNs) and Neural Operators, enabling 100-1000x faster PDE solving with interactive 3D visualization.**

## 🎯 Vision

Bridge the gap between academic research and industrial deployment of physics-informed machine learning:

- **Speed**: 100-1000x speedup over traditional PDE solvers
- **Accuracy**: Physics constraints ensure physically plausible predictions
- **Scalability**: Production-ready API and deployment infrastructure
- **Interactivity**: Real-time 3D visualization with WebSocket streaming ✨ **NEW**

## 🚀 Key Features

### Core ML
- **Physics-Informed Neural Networks (PINNs)**: Embed PDE constraints directly into loss functions
- **Neural Operators (FNO)**: Learn solution operators for parametric PDE families
- **Resolution-Invariant Learning**: Train on 64 grid points, test on 256 without retraining

### Production API
- **REST API**: FastAPI with async endpoints
- **WebSocket Streaming**: Real-time prediction streaming at 50 FPS ✨ **NEW**
- **Docker**: CPU and GPU containerization
- **Batch Processing**: High-throughput inference

### Interactive Frontend ✨ **NEW**
- **3D Visualization**: Real-time Three.js rendering
- **2D Charts**: Interactive line plots with Recharts
- **Export**: JSON, CSV, and PNG screenshot support
- **Live Metrics**: Inference time and throughput monitoring

## 🎨 Screenshots

```
┌─────────────────┬──────────────────────┬─────────────────┐
│   Controls      │   3D Visualization   │  Documentation  │
│   • PDE Type    │                      │  • Equations    │
│   • Resolution  │    [Three.js Canvas] │  • How it works │
│   • Time Steps  │                      │  • Features     │
│                 │                      │                 │
│   Metrics       │   ──────────────     │  Export         │
│   • 2.5ms       │   WebSocket Stream   │  • JSON         │
│   • 600/s       │                      │  • CSV          │
│                 │                      │  • Screenshot   │
└─────────────────┴──────────────────────┴─────────────────┘
```

## 📦 Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/sinsangwoo/Physics-Informed-ML.git
cd Physics-Informed-ML

# Backend setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev,api]"

# Frontend setup
cd frontend
npm install
```

### Run Full Stack

```bash
# Terminal 1: Start API
uvicorn physics_informed_ml.api.main:app --reload

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Visit `http://localhost:3000` for interactive UI!

### Docker Deployment

```bash
# Backend + Frontend
docker-compose up -d

# GPU-enabled
docker-compose --profile gpu up -d
```

## 💻 Usage

### Web Interface (Recommended)

1. Start the stack (API + Frontend)
2. Open `http://localhost:3000`
3. Select PDE type (Heat, Wave, Burgers)
4. Adjust resolution and time steps
5. Click "Play" to watch real-time solution
6. Toggle 2D/3D views
7. Export results

### Python API

```python
from physics_informed_ml.models import FNO1d
from physics_informed_ml.benchmarks import HeatEquation1D, BenchmarkRunner

# Initialize model
fno = FNO1d(modes=12, width=32, n_layers=4)

# Run benchmark
runner = BenchmarkRunner(device="cuda")
problem = HeatEquation1D(alpha=0.01)

results = runner.run(
    model=fno,
    problem=problem,
    train_resolution=64,
    test_resolutions=[64, 128, 256],
)
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "heat_equation_fno",
    "input_data": [[0.5, 0.3, 0.1]]
  }'
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/ws')

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'stream',
    model_name: 'heat_equation_fno',
    input_data: [0.5, 0.4, 0.3, ...],
    time_steps: 50
  }))
}

ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  if (data.type === 'frame') {
    // Update visualization
    updateVisualization(data.prediction)
  }
}
```

## 📊 Benchmarks

| Problem | Traditional | PINN | FNO | Speedup |
|---------|------------|------|-----|---------|
| Heat Equation 1D | 1.0s | 0.5s | 0.002s | **500x** |
| Burgers' Equation | 10s | 2s | 0.01s | **1000x** |
| Navier-Stokes 2D | 300s | 30s | 0.5s | **600x** |

### Resolution Invariance

```
Train on 64 points → Test on:
• 64:  L2=0.0082 ✓
• 128: L2=0.0085 ✓
• 256: L2=0.0089 ✓
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           React Frontend                │
│  • Three.js 3D Visualization            │
│  • Recharts 2D Plots                    │
│  • WebSocket Client                     │
└──────────────┬──────────────────────────┘
               │ HTTP/WS
┌──────────────▼──────────────────────────┐
│          FastAPI Backend                │
│  • REST Endpoints                       │
│  • WebSocket Streaming                  │
│  • Model Management                     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Inference Engine                 │
│  • FNO Models                           │
│  • PINN Models                          │
│  • GPU Acceleration                     │
└─────────────────────────────────────────┘
```

## 🗺️ Roadmap

### Phase 0: Foundation ✅
- [x] Modern Python project structure
- [x] CI/CD pipeline
- [x] Testing framework

### Phase 1: Physics-Informed Architecture ✅
- [x] PINN implementation
- [x] Multi-body dynamics
- [x] Comprehensive benchmarks

### Phase 2: Neural Operators ✅
- [x] Fourier Neural Operator (1D/2D/3D)
- [x] Spectral convolution layers
- [x] Resolution-invariant learning
- [x] Benchmark suite

### Phase 3: Production API ✅
- [x] FastAPI REST API
- [x] Docker containerization
- [x] Python client SDK
- [x] Complete documentation

### Phase 4: Interactive Frontend ✅ **COMPLETED**
- [x] React + TypeScript setup
- [x] Three.js 3D visualization
- [x] Real-time parameter tuning
- [x] WebSocket streaming
- [x] 2D plotting integration
- [x] Export functionality (JSON/CSV/PNG)

### Phase 5: Research Features (Next)
- [ ] Bayesian uncertainty quantification
- [ ] Transfer learning
- [ ] Explainability tools
- [ ] DeepONet implementation

## 🛠️ Technology Stack

**Backend:**
- PyTorch 2.1+
- FastAPI
- WebSockets
- Docker

**Frontend:**
- React 18 + TypeScript
- Three.js (@react-three/fiber)
- Recharts
- Zustand
- Tailwind CSS

**DevOps:**
- GitHub Actions
- pytest
- Ruff + mypy

## 📚 Documentation

- [Neural Operators Guide](docs/neural_operators.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Frontend README](frontend/README.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 Citation

```bibtex
@software{physics_informed_ml,
  author = {Sin, Sangwoo},
  title = {Physics-Informed ML: Neural Operators for Real-Time Simulation},
  year = {2025},
  url = {https://github.com/sinsangwoo/Physics-Informed-ML}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Raissi et al. (2019) - Physics-Informed Neural Networks
- Li et al. (2021) - Fourier Neural Operator
- Lu et al. (2021) - DeepONet

---

**Built for the future of physics simulation. Made with ❤️ and ⚡**
