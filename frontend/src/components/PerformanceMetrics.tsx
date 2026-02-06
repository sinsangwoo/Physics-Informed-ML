import { Activity, Zap } from 'lucide-react'
import { useSimulationStore } from '../store/simulationStore'

function PerformanceMetrics() {
  const { inferenceTime, fps } = useSimulationStore()

  return (
    <div className="bg-black bg-opacity-70 backdrop-blur-sm rounded-lg p-4 space-y-3 min-w-[200px]">
      <div className="flex items-center gap-2">
        <Zap className="w-4 h-4 text-yellow-400" />
        <div className="flex-1">
          <div className="text-xs text-gray-400">Inference</div>
          <div className="text-lg font-mono font-bold text-white">
            {inferenceTime.toFixed(1)} <span className="text-sm text-gray-400">ms</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-green-400" />
        <div className="flex-1">
          <div className="text-xs text-gray-400">Frame Rate</div>
          <div className="text-lg font-mono font-bold text-white">
            {fps} <span className="text-sm text-gray-400">FPS</span>
          </div>
        </div>
      </div>

      {/* Performance indicator */}
      <div className="pt-2 border-t border-gray-700">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">Performance</span>
          <span className={`font-semibold ${
            inferenceTime < 10 ? 'text-green-400' :
            inferenceTime < 50 ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {inferenceTime < 10 ? 'Excellent' :
             inferenceTime < 50 ? 'Good' :
             'Slow'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default PerformanceMetrics
