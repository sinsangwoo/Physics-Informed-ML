import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface WaveFieldProps {
  data: number[][]
}

function WaveField({ data }: WaveFieldProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  // Create wave surface
  const { geometry, positions, colors } = useMemo(() => {
    if (data.length === 0) return { geometry: null, positions: null, colors: null }
    
    const resolution = data.length
    const geometry = new THREE.PlaneGeometry(10, 10, resolution - 1, resolution - 1)
    
    const flatData = data.flat()
    const positions = new Float32Array(flatData.length * 3)
    const colors = new Float32Array(flatData.length * 3)
    
    flatData.forEach((value, i) => {
      const x = (i % resolution) / resolution * 10 - 5
      const z = Math.floor(i / resolution) / resolution * 10 - 5
      const y = value * 3 // Larger amplitude for waves
      
      positions[i * 3] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = z
      
      // Cyan-purple gradient for waves
      const normalized = (value + 1) / 2
      colors[i * 3] = 0.2 + normalized * 0.5 // R
      colors[i * 3 + 1] = 0.8 - normalized * 0.3 // G  
      colors[i * 3 + 2] = 1.0 // B
    })
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.computeVertexNormals()
    
    return { geometry, positions, colors }
  }, [data])
  
  // Smooth rotation
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.1
    }
  })
  
  if (!geometry) {
    return null
  }
  
  return (
    <group>
      <mesh
        ref={meshRef}
        geometry={geometry}
        rotation={[-Math.PI / 2, 0, 0]}
        castShadow
        receiveShadow
      >
        <meshPhysicalMaterial
          vertexColors
          wireframe={false}
          side={THREE.DoubleSide}
          transmission={0.1}
          thickness={0.5}
          roughness={0.2}
          metalness={0.1}
        />
      </mesh>
      
      {/* Wireframe overlay */}
      <mesh
        geometry={geometry}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.01, 0]}
      >
        <meshBasicMaterial
          color="#3b82f6"
          wireframe
          transparent
          opacity={0.3}
        />
      </mesh>
    </group>
  )
}

export default WaveField
