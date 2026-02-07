import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useSimulationStore } from '../store/simulationStore'

export function ChartPanel() {
  const { predictions, currentStep, inputData } = useSimulationStore()

  // Prepare chart data
  const chartData = useMemo(() => {
    if (!predictions || predictions.length === 0) {
      // Show initial data if available
      if (inputData) {
        return inputData.map((value, index) => ({
          x: index,
          initial: value,
        }))
      }
      return []
    }

    const currentPrediction = predictions[currentStep] || predictions[0]
    
    return currentPrediction.map((value, index) => ({
      x: index,
      value: value,
      initial: inputData ? inputData[index] : undefined,
    }))
  }, [predictions, currentStep, inputData])

  if (chartData.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400">
        No data to display
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="x" 
          stroke="#94a3b8"
          label={{ value: 'Position', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
        />
        <YAxis 
          stroke="#94a3b8"
          label={{ value: 'Value', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1e293b', 
            border: '1px solid #334155',
            borderRadius: '0.5rem'
          }}
          labelStyle={{ color: '#94a3b8' }}
        />
        <Legend wrapperStyle={{ color: '#94a3b8' }} />
        
        {inputData && (
          <Line 
            type="monotone" 
            dataKey="initial" 
            stroke="#64748b" 
            strokeWidth={2}
            dot={false}
            name="Initial Condition"
          />
        )}
        
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#0ea5e9" 
          strokeWidth={2}
          dot={false}
          name="Current Solution"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
