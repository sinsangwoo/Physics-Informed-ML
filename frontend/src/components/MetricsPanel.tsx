import { useSimulationStore } from '../store/simulationStore'
import { Clock, Zap, Database } from 'lucide-react'

export function MetricsPanel() {
  const { inferenceTime, predictions, resolution } = useSimulationStore()

  const formatTime = (ms: number | null) => {
    if (ms === null) return 'N/A'
    return ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : `${ms.toFixed(2)}ms`
  }

  const calculateThroughput = () => {
    if (!inferenceTime || !predictions) return 'N/A'
    const samplesPerSecond = (1000 / inferenceTime).toFixed(0)
    return `${samplesPerSecond} samples/s`
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-slate-400">
          <Clock className="w-4 h-4" />
          <span>Inference Time</span>
        </div>
        <span className="font-mono text-primary-400">
          {formatTime(inferenceTime)}
        </span>
      </div>

      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-slate-400">
          <Zap className="w-4 h-4" />
          <span>Throughput</span>
        </div>
        <span className="font-mono text-green-400">
          {calculateThroughput()}
        </span>
      </div>

      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-slate-400">
          <Database className="w-4 h-4" />
          <span>Data Points</span>
        </div>
        <span className="font-mono">{resolution}</span>
      </div>

      {predictions && (
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-400">Frames</span>
          <span className="font-mono">{predictions.length}</span>
        </div>
      )}
    </div>
  )
}
