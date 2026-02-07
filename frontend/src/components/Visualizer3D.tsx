import { useEffect, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei'
import { useSimulationStore } from '../store/simulationStore'
import * as THREE from 'three'

// 3D surface from 1D data (heat equation visualization)
function HeatSurface() {
  const meshRef = useRef<THREE.Mesh>(null)
  const { predictions, currentStep, resolution } = useSimulationStore()

  useEffect(() => {
    if (!meshRef.current || !predictions || predictions.length === 0) return

    const geometry = meshRef.current.geometry as THREE.PlaneGeometry
    const positions = geometry.attributes.position

    // Update heights based on current timestep
    const data = predictions[currentStep] || predictions[0]
    
    for (let i = 0; i < positions.count; i++) {
      const x = i % resolution
      const dataIndex = Math.floor((x / resolution) * data.length)
      const height = data[dataIndex] || 0
      positions.setZ(i, height * 2) // Scale for visibility
    }

    positions.needsUpdate = true
    geometry.computeVertexNormals()
  }, [predictions, currentStep, resolution])

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[10, 10, resolution - 1, resolution - 1]} />
      <meshStandardMaterial
        color="#0ea5e9"
        wireframe={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

// Animated wave visualization
function WaveVisualization() {
  const pointsRef = useRef<THREE.Points>(null)
  const { predictions, currentStep, resolution } = useSimulationStore()

  useFrame((state) => {
    if (!pointsRef.current) return
    pointsRef.current.rotation.y = state.clock.elapsedTime * 0.1
  })

  useEffect(() => {
    if (!pointsRef.current || !predictions || predictions.length === 0) return

    const geometry = pointsRef.current.geometry as THREE.BufferGeometry
    const positions = geometry.attributes.position

    const data = predictions[currentStep] || predictions[0]
    
    for (let i = 0; i < data.length; i++) {
      const x = (i / data.length) * 10 - 5
      const y = data[i] * 3
      const z = 0
      
      positions.setXYZ(i, x, y, z)
    }

    positions.needsUpdate = true
  }, [predictions, currentStep, resolution])

  const points = new Float32Array(resolution * 3)
  for (let i = 0; i < resolution; i++) {
    points[i * 3] = (i / resolution) * 10 - 5
    points[i * 3 + 1] = 0
    points[i * 3 + 2] = 0
  }

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length / 3}
          array={points}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.1} color="#22d3ee" />
    </points>
  )
}

// Main 3D Scene
function Scene() {
  const { type } = useSimulationStore()

  return (
    <>
      {/* Camera */}
      <PerspectiveCamera makeDefault position={[8, 6, 8]} />
      <OrbitControls enableDamping dampingFactor={0.05} />

      {/* Lights */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <pointLight position={[-10, -10, -5]} intensity={0.5} color="#0ea5e9" />

      {/* Grid */}
      <Grid
        args={[20, 20]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#334155"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#475569"
        fadeDistance={25}
        fadeStrength={1}
        followCamera={false}
      />

      {/* Visualization based on simulation type */}
      {type === 'heat' && <HeatSurface />}
      {type === 'wave' && <WaveVisualization />}
      {type === 'burgers' && <WaveVisualization />}
    </>
  )
}

export function Visualizer3D() {
  return (
    <div className="w-full h-full bg-slate-950">
      <Canvas>
        <Scene />
      </Canvas>
    </div>
  )
}
