import { ChevronDown } from 'lucide-react'
import { useSimulationStore } from '../store/simulationStore'
import { useState, useRef, useEffect } from 'react'

function ModelSelector() {
  const { selectedModel, availableModels, setModel } = useSimulationStore()
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-sm"
      >
        <span className="font-medium">Model:</span>
        <span className="text-blue-400 font-mono">{selectedModel}</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${
          isOpen ? 'rotate-180' : ''
        }`} />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-gray-800 border border-gray-700 rounded-lg shadow-lg overflow-hidden z-50">
          {availableModels.map((model) => (
            <button
              key={model}
              onClick={() => {
                setModel(model)
                setIsOpen(false)
              }}
              className={`w-full text-left px-4 py-3 hover:bg-gray-700 transition-colors ${
                model === selectedModel ? 'bg-gray-700 text-blue-400' : 'text-gray-300'
              }`}
            >
              <div className="font-mono text-sm">{model}</div>
              <div className="text-xs text-gray-500 mt-1">
                {model.includes('heat') && 'Heat diffusion simulation'}
                {model.includes('burgers') && 'Nonlinear advection-diffusion'}
                {model.includes('wave') && 'Wave propagation'}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default ModelSelector
