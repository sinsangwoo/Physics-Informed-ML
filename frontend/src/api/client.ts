import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface InferenceResult {
  prediction: number[][]
  model_name: string
  inference_time_ms: number
  input_shape: number[]
  output_shape: number[]
}

export interface ModelInfo {
  name: string
  type: string
  parameters: number
  input_shape?: number[]
  output_shape?: number[]
  device: string
}

export async function healthCheck() {
  const response = await client.get('/health')
  return response.data
}

export async function listModels() {
  const response = await client.get('/models')
  return response.data
}

export async function getModelInfo(modelName: string): Promise<ModelInfo> {
  const response = await client.get(`/models/${modelName}`)
  return response.data
}

export async function runInference(
  modelName: string,
  inputData: number[][]
): Promise<InferenceResult> {
  const response = await client.post('/predict', {
    model_name: modelName,
    input_data: inputData,
  })
  return response.data
}

export async function runBatchInference(
  modelName: string,
  inputDataList: number[][][]
): Promise<InferenceResult[]> {
  const response = await client.post('/predict/batch', {
    model_name: modelName,
    input_data_list: inputDataList,
  })
  return response.data.predictions
}

export default client
