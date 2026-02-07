import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface PredictionRequest {
  model_name: string
  input_data: number[][]
}

export interface PredictionResponse {
  prediction: number[][]
  model_name: string
  inference_time_ms: number
  input_shape: number[]
  output_shape: number[]
}

export interface HealthResponse {
  status: string
  gpu_available: boolean
  device: string
  loaded_models: string[]
}

export interface ModelInfo {
  name: string
  type: string
  parameters: number
  input_shape?: number[]
  output_shape?: number[]
  device: string
}

// API endpoints
export const apiClient = {
  // Health check
  health: async (): Promise<HealthResponse> => {
    const { data } = await api.get('/health')
    return data
  },

  // Load model
  loadModel: async (modelPath: string, modelName: string) => {
    const { data } = await api.post('/models/load', null, {
      params: { model_path: modelPath, model_name: modelName },
    })
    return data
  },

  // Get model info
  getModelInfo: async (modelName: string): Promise<ModelInfo> => {
    const { data } = await api.get(`/models/${modelName}`)
    return data
  },

  // Single prediction
  predict: async (request: PredictionRequest): Promise<PredictionResponse> => {
    const { data } = await api.post('/predict', request)
    return data
  },

  // Batch prediction
  predictBatch: async (
    modelName: string,
    inputDataList: number[][][]
  ): Promise<PredictionResponse[]> => {
    const { data } = await api.post('/predict/batch', {
      model_name: modelName,
      input_data_list: inputDataList,
    })
    return data.predictions
  },
}

export default api
