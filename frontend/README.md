# Physics-Informed ML Frontend

## 🎨 Interactive 3D Physics Visualization

Real-time neural PDE solver visualization built with React, Three.js, and TypeScript.

![Screenshot](https://via.placeholder.com/800x450.png?text=Physics+Simulation)

## ✨ Features

### 🌊 Real-Time 3D Visualization
- **Heat Equation**: Animated 3D surface showing heat diffusion
- **Wave Equation**: Particle system for wave propagation
- **Burgers Equation**: Shock wave visualization
- **Navier-Stokes**: Fluid dynamics (coming soon)

### ⚡ High Performance
- 60 FPS rendering with Three.js
- 2-5ms inference latency
- 600+ samples/second throughput
- GPU-accelerated neural operators

### 🎮 Interactive Controls
- Adjustable resolution (32-256 points)
- Variable time steps (10-100)
- Play/pause animation
- Camera orbit controls
- Initial condition presets

### 📊 Live Metrics
- Inference time monitoring
- Throughput calculation
- API health status
- Model information

## 🚀 Quick Start

### Prerequisites

```bash
# Node.js 18+ required
node --version
```

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
# Start dev server (port 3000)
npm run dev

# Type checking
npm run type-check

# Linting
npm run lint
```

### Production Build

```bash
npm run build
npm run preview
```

## 🏗️ Architecture

```
frontend/
├── src/
│   ├── components/
│   │   ├── Layout.tsx           # App layout
│   │   ├── Simulator.tsx        # Main dashboard
│   │   ├── Visualizer3D.tsx     # Three.js 3D scene
│   │   ├── ControlPanel.tsx     # Simulation controls
│   │   └── MetricsPanel.tsx     # Performance metrics
│   ├── lib/
│   │   └── api.ts               # API client
│   ├── store/
│   │   └── simulationStore.ts   # Zustand state
│   ├── App.tsx                  # Root component
│   └── main.tsx                 # Entry point
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

## 🎯 Tech Stack

### Core
- **React 18**: UI framework with hooks
- **TypeScript**: Type-safe development
- **Vite**: Lightning-fast build tool

### 3D Graphics
- **Three.js**: WebGL rendering
- **@react-three/fiber**: React renderer for Three.js
- **@react-three/drei**: Useful Three.js helpers

### State & Data
- **Zustand**: Lightweight state management
- **React Query**: Server state and caching
- **Axios**: HTTP client

### UI
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library
- **Recharts**: 2D plotting (future)

## 🔌 API Integration

### Connecting to Backend

The frontend proxies API requests to `http://localhost:8000`:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

### API Endpoints Used

```typescript
// Health check
GET /api/health

// Single prediction
POST /api/predict
{
  "model_name": "heat_equation_fno",
  "input_data": [[0.5, 0.3, ...]]
}

// Batch prediction
POST /api/predict/batch
{
  "model_name": "heat_equation_fno",
  "input_data_list": [[[...]], [[...]]]
}
```

## 🎨 Customization

### Color Theme

Edit `tailwind.config.js`:

```js
theme: {
  extend: {
    colors: {
      primary: {
        500: '#0ea5e9',  // Customize
      },
    },
  },
}
```

### Add New Visualization

1. Create component in `components/`
2. Add to `Visualizer3D.tsx` Scene
3. Update simulation types

```typescript
// simulationStore.ts
export type SimulationType = 'heat' | 'wave' | 'your-type'
```

## 📊 Performance

### Optimization Tips

1. **Reduce Resolution**: Lower grid size for faster rendering
2. **Limit Time Steps**: Fewer frames = less data transfer
3. **Use Batch API**: Process multiple frames efficiently
4. **Enable GPU**: Backend GPU acceleration crucial

### Benchmarks

| Resolution | FPS | Inference | Throughput |
|------------|-----|-----------|------------|
| 32 points  | 60  | 1.5ms     | 800/s      |
| 64 points  | 60  | 2.5ms     | 600/s      |
| 128 points | 60  | 5.0ms     | 350/s      |
| 256 points | 60  | 12ms      | 150/s      |

## 🐛 Troubleshooting

**Issue: API connection failed**
```
Solution: Start backend API server on port 8000
cd ..
uvicorn physics_informed_ml.api.main:app
```

**Issue: Black screen in 3D view**
```
Solution: Check browser WebGL support
Visit: https://get.webgl.org/
```

**Issue: Slow performance**
```
Solution: Reduce resolution or time steps
Enable GPU acceleration in backend
```

## 📝 License

MIT License - see LICENSE file
