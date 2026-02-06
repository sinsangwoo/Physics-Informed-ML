import { Play, Pause, RotateCcw, Settings } from 'lucide-react'
import { useSimulationStore } from '../store/simulationStore'
import ParameterSlider from './ParameterSlider'
import { useState, useEffect } from 'react'
import { runInference } from '../api/client'

function ControlPanel() {
  const {
    isRunning,
    setRunning,
    parameters,
    updateParameter,
    selectedModel,
    updateData,
    addHistory,
    updateMetrics,
    reset,
  } = useSimulationStore()

  const [autoRun, setAutoRun] = useState(false)

  // Simulation loop
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(async () => {
      try {
        const startTime = performance.now()
        
        // Generate input data (example: sine wave)
        const resolution = parameters.resolution
        const inputData = Array.from({ length: resolution }, (_, i) => [
          Math.sin(i / resolution * Math.PI * 2) * 0.5
        ])

        // Run inference
        const result = await runInference(selectedModel, inputData)
        
        const endTime = performance.now()
        const inferenceTime = endTime - startTime

        // Update visualization
        updateData(result.prediction)
        addHistory(result.prediction)
        updateMetrics(inferenceTime, 60) // Assume 60 FPS

      } catch (error) {
        console.error('Inference failed:', error)
        setRunning(false)
      }
    }, 1000 / 30) // 30 FPS

    return () => clearInterval(interval)
  }, [isRunning, parameters, selectedModel])

  return (
    <div className="p-6 space-y-6">
      {/* Playback Controls */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-100 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Simulation Controls
        </h2>
        
        <div className="flex gap-2">
          <button
            onClick={() => setRunning(!isRunning)}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-colors ${
              isRunning
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isRunning ? (
              <>
                <Pause className="w-5 h-5" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Start
              </>
            )}
          </button>
          
          <button
            onClick={reset}
            className="px-4 py-3 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Reset"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Parameters */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-100">Parameters</h3>
        
        <ParameterSlider
          label="Thermal Diffusivity (α)"
          value={parameters.alpha}
          onChange={(value) => updateParameter('alpha', value)}
          min={0.001}
          max={0.1}
          step={0.001}
          unit=""
        />
        
        <ParameterSlider
          label="Resolution"
          value={parameters.resolution}
          onChange={(value) => updateParameter('resolution', value)}
          min={32}
          max={128}
          step={16}
          unit="pts"
        />
        
        <ParameterSlider
          label="Time Step"
          value={parameters.timestep}
          onChange={(value) => updateParameter('timestep', value)}
          min={0.001}
          max={0.1}
          step={0.001}
          unit="s"
        />
      </div>

      {/* Info Panel */}
      <div className="bg-gray-700 rounded-lg p-4 space-y-2">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Current State
        </h3>
        
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Model:</span>
            <span className="text-gray-200 font-mono">{selectedModel}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Resolution:</span>
            <span className="text-gray-200">{parameters.resolution}×{parameters.resolution}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Status:</span>
            <span className={isRunning ? 'text-green-400' : 'text-gray-400'}>
              {isRunning ? 'Running' : 'Stopped'}
            </span>
          </div>
        </div>
      </div>

      {/* Tips */}
      <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-blue-300 mb-2">💡 Tips</h3>
        <ul className="text-xs text-blue-200 space-y-1">
          <li>• Use mouse to rotate the 3D view</li>
          <li>• Scroll to zoom in/out</li>
          <li>• Adjust parameters in real-time</li>
          <li>• Watch inference time in metrics</li>
        </ul>
      </div>
    </div>
  )
}

export default ControlPanel
