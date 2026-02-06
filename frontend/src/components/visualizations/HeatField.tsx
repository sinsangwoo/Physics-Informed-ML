import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface HeatFieldProps {
  data: number[][]
}

function HeatField({ data }: HeatFieldProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  // Create geometry and material
  const { geometry, positions, colors } = useMemo(() => {
    if (data.length === 0) return { geometry: null, positions: null, colors: null }
    
    const resolution = data.length
    const geometry = new THREE.PlaneGeometry(10, 10, resolution - 1, resolution - 1)
    
    // Flatten 2D data to 1D
    const flatData = data.flat()
    const positions = new Float32Array(flatData.length * 3)
    const colors = new Float32Array(flatData.length * 3)
    
    // Create height map and color map
    flatData.forEach((value, i) => {
      const x = (i % resolution) / resolution * 10 - 5
      const z = Math.floor(i / resolution) / resolution * 10 - 5
      const y = value * 2 // Scale height
      
      positions[i * 3] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = z
      
      // Color based on value (blue = cold, red = hot)
      const normalized = (value + 1) / 2 // Assume data in [-1, 1]
      colors[i * 3] = normalized // R
      colors[i * 3 + 1] = 0.3 // G
      colors[i * 3 + 2] = 1 - normalized // B
    })
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.computeVertexNormals()
    
    return { geometry, positions, colors }
  }, [data])
  
  // Animate subtle wave effect
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.2) * 0.05
    }
  })
  
  if (!geometry) {
    return null
  }
  
  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      rotation={[-Math.PI / 2, 0, 0]}
      receiveShadow
      castShadow
    >
      <meshStandardMaterial
        vertexColors
        wireframe={false}
        side={THREE.DoubleSide}
        metalness={0.3}
        roughness={0.7}
      />
    </mesh>
  )
}

export default HeatField
