import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Play, Pause, RotateCcw, Settings, BarChart3 } from 'lucide-react'
import { apiClient } from '../lib/api'
import { useSimulationStore } from '../store/simulationStore'
import { Visualizer3D } from './Visualizer3D'
import { ControlPanel } from './ControlPanel'
import { MetricsPanel } from './MetricsPanel'
import { ChartPanel } from './ChartPanel'
import { ExportPanel } from './ExportPanel'

export function Simulator() {
  const [apiError, setApiError] = useState<string | null>(null)
  const [showChart, setShowChart] = useState(false)
  const { isPlaying, setIsPlaying } = useSimulationStore()

  // Health check
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: apiClient.health,
    refetchInterval: 30000, // Refresh every 30s
    onError: (error: any) => {
      setApiError(error.message || 'API connection failed')
    },
  })

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-180px)]">
      {/* Left Panel - Controls */}
      <div className="lg:col-span-3 space-y-4">
        <div className="card p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Controls</h2>
            <Settings className="w-5 h-5 text-slate-400" />
          </div>
          <ControlPanel />
        </div>

        {/* API Status */}
        <div className="card p-4">
          <h3 className="text-sm font-medium mb-2">API Status</h3>
          {healthLoading ? (
            <div className="text-sm text-slate-400">Connecting...</div>
          ) : apiError ? (
            <div className="text-sm text-red-400">{apiError}</div>
          ) : health ? (
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Status</span>
                <span className="text-green-400 capitalize">{health.status}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Device</span>
                <span className="font-mono text-xs">{health.device}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Models</span>
                <span>{health.loaded_models.length}</span>
              </div>
            </div>
          ) : null}
        </div>

        {/* Metrics */}
        <div className="card p-4">
          <h3 className="text-sm font-medium mb-2">Performance</h3>
          <MetricsPanel />
        </div>

        {/* Export */}
        <div className="card p-4">
          <ExportPanel />
        </div>
      </div>

      {/* Center Panel - Visualization */}
      <div className="lg:col-span-6">
        <div className="card h-full flex flex-col">
          {/* Toolbar */}
          <div className="p-4 border-b border-slate-800 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              {showChart ? '2D Chart' : '3D Visualization'}
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowChart(!showChart)}
                className="btn-secondary flex items-center gap-2"
              >
                <BarChart3 className="w-4 h-4" />
                {showChart ? '3D' : '2D'}
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="btn-primary flex items-center gap-2"
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Play
                  </>
                )}
              </button>
              <button className="btn-secondary">
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Canvas */}
          <div className="flex-1 relative">
            {showChart ? (
              <div className="w-full h-full p-4">
                <ChartPanel />
              </div>
            ) : (
              <Visualizer3D />
            )}
          </div>
        </div>
      </div>

      {/* Right Panel - Documentation */}
      <div className="lg:col-span-3 space-y-4">
        <div className="card p-4">
          <h3 className="text-sm font-medium mb-3">About</h3>
          <div className="text-sm text-slate-400 space-y-2">
            <p>
              Interactive visualization of physics-informed neural networks solving PDEs in real-time.
            </p>
            <p className="font-semibold text-slate-300 mt-4">How it works:</p>
            <ol className="list-decimal list-inside space-y-1">
              <li>Select PDE type</li>
              <li>Adjust initial conditions</li>
              <li>Watch neural network solve</li>
              <li>Export results</li>
            </ol>
          </div>
        </div>

        <div className="card p-4">
          <h3 className="text-sm font-medium mb-3">Equations</h3>
          <div className="text-xs font-mono text-slate-400 space-y-3">
            <div>
              <div className="text-slate-300 mb-1">Heat Equation</div>
              <div>∂u/∂t = α∂²u/∂x²</div>
            </div>
            <div>
              <div className="text-slate-300 mb-1">Wave Equation</div>
              <div>∂²u/∂t² = c²∂²u/∂x²</div>
            </div>
            <div>
              <div className="text-slate-300 mb-1">Burgers Equation</div>
              <div>∂u/∂t + u∂u/∂x = ν∂²u/∂x²</div>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <h3 className="text-sm font-medium mb-3">Features</h3>
          <ul className="text-xs text-slate-400 space-y-1.5">
            <li>✓ Real-time 3D visualization</li>
            <li>✓ 2D line plots</li>
            <li>✓ WebSocket streaming</li>
            <li>✓ Export JSON/CSV</li>
            <li>✓ Screenshot capture</li>
            <li>✓ GPU acceleration</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
