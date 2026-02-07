import { create } from 'zustand'

export type SimulationType = 'heat' | 'wave' | 'burgers' | 'navier-stokes'

interface SimulationState {
  // Simulation config
  type: SimulationType
  resolution: number
  timeSteps: number
  
  // Animation
  isPlaying: boolean
  currentStep: number
  
  // Data
  inputData: number[] | null
  predictions: number[][] | null
  
  // Performance
  inferenceTime: number | null
  
  // Actions
  setType: (type: SimulationType) => void
  setResolution: (resolution: number) => void
  setTimeSteps: (steps: number) => void
  setIsPlaying: (playing: boolean) => void
  setCurrentStep: (step: number) => void
  setInputData: (data: number[]) => void
  setPredictions: (predictions: number[][]) => void
  setInferenceTime: (time: number) => void
  reset: () => void
}

const initialState = {
  type: 'heat' as SimulationType,
  resolution: 64,
  timeSteps: 50,
  isPlaying: false,
  currentStep: 0,
  inputData: null,
  predictions: null,
  inferenceTime: null,
}

export const useSimulationStore = create<SimulationState>((set) => ({
  ...initialState,
  
  setType: (type) => set({ type }),
  setResolution: (resolution) => set({ resolution }),
  setTimeSteps: (timeSteps) => set({ timeSteps }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setCurrentStep: (currentStep) => set({ currentStep }),
  setInputData: (inputData) => set({ inputData }),
  setPredictions: (predictions) => set({ predictions }),
  setInferenceTime: (inferenceTime) => set({ inferenceTime }),
  reset: () => set(initialState),
}))
