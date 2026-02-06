import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Stats } from '@react-three/drei'
import { Suspense } from 'react'
import HeatField from './visualizations/HeatField'
import WaveField from './visualizations/WaveField'
import { useSimulationStore } from '../store/simulationStore'

function Visualization3D() {
  const { selectedModel, currentData } = useSimulationStore()

  // Select visualization based on model type
  const getVisualization = () => {
    if (selectedModel.includes('heat')) {
      return <HeatField data={currentData} />
    } else if (selectedModel.includes('wave')) {
      return <WaveField data={currentData} />
    }
    // Default: heat field
    return <HeatField data={currentData} />
  }

  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [5, 5, 5], fov: 50 }}
        shadows
        dpr={[1, 2]} // Responsive pixel ratio
      >
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />
        
        {/* Scene */}
        <Suspense fallback={null}>
          {getVisualization()}
        </Suspense>
        
        {/* Grid helper */}
        <Grid
          args={[10, 10]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#6b7280"
          sectionSize={2}
          sectionThickness={1}
          sectionColor="#3b82f6"
          fadeDistance={30}
          fadeStrength={1}
          followCamera={false}
          infiniteGrid
        />
        
        {/* Controls */}
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={3}
          maxDistance={20}
        />
        
        {/* Performance stats (dev only) */}
        {import.meta.env.DEV && <Stats />}
      </Canvas>
      
      {/* Loading indicator */}
      {currentData.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-gray-300">Initializing simulation...</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default Visualization3D
