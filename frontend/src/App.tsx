import { useState } from 'react'
import Visualization3D from './components/Visualization3D'
import ControlPanel from './components/ControlPanel'
import ModelSelector from './components/ModelSelector'
import PerformanceMetrics from './components/PerformanceMetrics'
import { useSimulationStore } from './store/simulationStore'

function App() {
  const { isRunning } = useSimulationStore()

  return (
    <div className="h-screen w-screen bg-gray-900 text-white flex flex-col overflow-hidden">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-blue-400">
              Physics-Informed ML
            </h1>
            <p className="text-sm text-gray-400">
              Real-time Neural Operator Visualization
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
              }`} />
              <span className="text-sm text-gray-300">
                {isRunning ? 'Running' : 'Paused'}
              </span>
            </div>
            
            <ModelSelector />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* 3D Visualization */}
        <div className="flex-1 relative">
          <Visualization3D />
          
          {/* Performance Overlay */}
          <div className="absolute top-4 right-4">
            <PerformanceMetrics />
          </div>
        </div>

        {/* Control Panel */}
        <div className="w-96 bg-gray-800 border-l border-gray-700 overflow-y-auto">
          <ControlPanel />
        </div>
      </div>
    </div>
  )
}

export default App
