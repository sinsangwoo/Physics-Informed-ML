# Physics-Informed ML Frontend

Interactive 3D visualization for neural operator simulations.

## Features

- 🎭 **Real-time 3D Rendering**: Three.js + React Three Fiber
- ⚡ **Instant Inference**: Live parameter adjustments
- 🎮 **Interactive Controls**: Intuitive UI for simulation
- 📊 **Performance Metrics**: FPS and inference time tracking
- 🎨 **Beautiful Visualizations**: Heat maps, wave surfaces

## Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn
- Backend API running on port 8000

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
# Start dev server
npm run dev

# Open browser at http://localhost:3000
```

### Build for Production

```bash
npm run build
npm run preview
```

## Configuration

Create `.env` file:

```env
VITE_API_URL=http://localhost:8000
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Visualization3D.tsx      # Main 3D canvas
│   │   ├── ControlPanel.tsx         # Parameter controls
│   │   ├── ModelSelector.tsx        # Model switcher
│   │   ├── PerformanceMetrics.tsx   # Stats overlay
│   │   └── visualizations/
│   │       ├── HeatField.tsx        # Heat simulation viz
│   │       └── WaveField.tsx        # Wave simulation viz
│   ├── store/
│   │   └── simulationStore.ts       # Zustand state
│   ├── api/
│   │   └── client.ts                # API client
│   ├── App.tsx
│   └── main.tsx
├── package.json
└── vite.config.ts
```

## Usage

### 1. Select Model

Click the model dropdown in the header to switch between:
- Heat Equation FNO
- Burgers PINN
- Wave Equation

### 2. Adjust Parameters

Use sliders in the control panel:
- **Thermal Diffusivity (α)**: Heat spread rate
- **Resolution**: Grid density (32-128 points)
- **Time Step**: Simulation speed

### 3. Run Simulation

- Click **Start** to begin real-time simulation
- Use mouse to rotate 3D view
- Scroll to zoom in/out
- Watch performance metrics in top-right

### 4. Observe

- **Heat Field**: Color gradient (blue=cold, red=hot)
- **Wave Field**: Oscillating surface with transparency
- **Performance**: Inference time and FPS

## Technologies

- **React 18**: UI framework
- **Three.js**: 3D rendering engine
- **React Three Fiber**: React renderer for Three.js
- **Zustand**: Lightweight state management
- **Vite**: Build tool and dev server
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **Lucide Icons**: UI icons

## Performance

### Optimization Tips

1. **Lower Resolution**: For slower GPUs, use 32-64 points
2. **Reduce FPS**: Change interval in ControlPanel.tsx
3. **Disable Shadows**: Comment out `castShadow` in visualizations
4. **Simple Materials**: Use `MeshBasicMaterial` instead of `MeshStandardMaterial`

### Expected Performance

| GPU | Resolution | FPS | Inference Time |
|-----|-----------|-----|----------------|
| RTX 3080 | 128x128 | 60 | 2-5ms |
| GTX 1060 | 64x64 | 30-60 | 5-10ms |
| Integrated | 32x32 | 30 | 10-20ms |

## Development

### Adding New Visualizations

1. Create component in `src/components/visualizations/`
2. Implement using React Three Fiber
3. Add to `Visualization3D.tsx` switch statement

```tsx
// Example: NewField.tsx
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'

function NewField({ data }) {
  const meshRef = useRef()
  
  useFrame(() => {
    // Animation logic
  })
  
  return (
    <mesh ref={meshRef}>
      <boxGeometry />
      <meshStandardMaterial />
    </mesh>
  )
}
```

### Adding Parameters

1. Add to `simulationStore.ts` initial state
2. Add slider in `ControlPanel.tsx`
3. Use in inference or visualization

## Troubleshooting

**Black screen?**
- Check browser console for errors
- Ensure backend API is running
- Check API URL in `.env`

**Poor performance?**
- Lower resolution
- Disable shadows
- Check GPU usage

**No data showing?**
- Verify model is loaded in backend
- Check network tab for API errors
- Ensure data format matches expectations

## Deployment

### Vercel (Recommended)

```bash
npm run build
vercel --prod
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

### Static Hosting

Build and deploy `dist/` folder to:
- Netlify
- Vercel
- GitHub Pages
- AWS S3 + CloudFront

## License

MIT
