import { useSimulationStore, SimulationType } from '../store/simulationStore'
import { RefreshCw } from 'lucide-react'

const SIMULATION_TYPES: { value: SimulationType; label: string }[] = [
  { value: 'heat', label: 'Heat Equation' },
  { value: 'wave', label: 'Wave Equation' },
  { value: 'burgers', label: 'Burgers Equation' },
  { value: 'navier-stokes', label: 'Navier-Stokes' },
]

const RESOLUTIONS = [32, 64, 128, 256]

export function ControlPanel() {
  const {
    type,
    resolution,
    timeSteps,
    setType,
    setResolution,
    setTimeSteps,
    reset,
  } = useSimulationStore()

  const handleGenerateInitialConditions = () => {
    const data = Array.from({ length: resolution }, (_, i) => {
      const x = i / resolution
      // Gaussian pulse
      return Math.exp(-Math.pow(x - 0.5, 2) / 0.01)
    })
    useSimulationStore.getState().setInputData(data)
  }

  return (
    <div className="space-y-4">
      {/* Simulation Type */}
      <div>
        <label className="block text-sm font-medium mb-2">PDE Type</label>
        <select
          value={type}
          onChange={(e) => setType(e.target.value as SimulationType)}
          className="w-full input"
        >
          {SIMULATION_TYPES.map(({ value, label }) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>

      {/* Resolution */}
      <div>
        <label className="block text-sm font-medium mb-2">
          Resolution: {resolution}
        </label>
        <div className="flex gap-2">
          {RESOLUTIONS.map((res) => (
            <button
              key={res}
              onClick={() => setResolution(res)}
              className={
                resolution === res
                  ? 'px-3 py-1 rounded bg-primary-600 text-sm'
                  : 'px-3 py-1 rounded bg-slate-800 text-sm hover:bg-slate-700'
              }
            >
              {res}
            </button>
          ))}
        </div>
      </div>

      {/* Time Steps */}
      <div>
        <label className="block text-sm font-medium mb-2">
          Time Steps: {timeSteps}
        </label>
        <input
          type="range"
          min="10"
          max="100"
          value={timeSteps}
          onChange={(e) => setTimeSteps(Number(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Actions */}
      <div className="space-y-2 pt-4 border-t border-slate-800">
        <button
          onClick={handleGenerateInitialConditions}
          className="w-full btn-primary"
        >
          Generate Initial Conditions
        </button>
        <button onClick={reset} className="w-full btn-secondary flex items-center justify-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Reset
        </button>
      </div>
    </div>
  )
}
