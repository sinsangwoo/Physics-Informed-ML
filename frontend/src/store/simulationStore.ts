import { create } from 'zustand'

interface SimulationState {
  // Simulation state
  isRunning: boolean
  currentTime: number
  
  // Model selection
  selectedModel: string
  availableModels: string[]
  
  // Parameters
  parameters: Record<string, number>
  
  // Simulation data
  currentData: number[][]
  history: number[][][]
  
  // Performance metrics
  inferenceTime: number
  fps: number
  
  // Actions
  setRunning: (running: boolean) => void
  setModel: (model: string) => void
  updateParameter: (key: string, value: number) => void
  updateData: (data: number[][]) => void
  addHistory: (data: number[][]) => void
  updateMetrics: (inferenceTime: number, fps: number) => void
  reset: () => void
}

const initialState = {
  isRunning: false,
  currentTime: 0,
  selectedModel: 'heat_equation_fno',
  availableModels: ['heat_equation_fno', 'burgers_pinn', 'wave_equation'],
  parameters: {
    alpha: 0.01,
    resolution: 64,
    timestep: 0.01,
  },
  currentData: [],
  history: [],
  inferenceTime: 0,
  fps: 0,
}

export const useSimulationStore = create<SimulationState>((set) => ({
  ...initialState,
  
  setRunning: (running) => set({ isRunning: running }),
  
  setModel: (model) => set({ 
    selectedModel: model,
    currentData: [],
    history: [],
  }),
  
  updateParameter: (key, value) => set((state) => ({
    parameters: { ...state.parameters, [key]: value },
  })),
  
  updateData: (data) => set({ currentData: data }),
  
  addHistory: (data) => set((state) => ({
    history: [...state.history.slice(-100), data], // Keep last 100 frames
  })),
  
  updateMetrics: (inferenceTime, fps) => set({ inferenceTime, fps }),
  
  reset: () => set(initialState),
}))
