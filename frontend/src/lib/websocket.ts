/**
 * WebSocket client for real-time streaming
 */

export type MessageType = 'status' | 'frame' | 'complete' | 'error'

export interface WebSocketMessage {
  type: MessageType
  step?: number
  total_steps?: number
  prediction?: number[]
  progress?: number
  message?: string
  total_frames?: number
}

export type MessageHandler = (message: WebSocketMessage) => void

export class StreamingClient {
  private ws: WebSocket | null = null
  private handlers: Map<MessageType, MessageHandler[]> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000

  constructor(private url: string = 'ws://localhost:8000/ws') {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)

        this.ws.onopen = () => {
          console.log('WebSocket connected')
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            this.handleMessage(message)
          } catch (error) {
            console.error('Failed to parse message:', error)
          }
        }

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          reject(error)
        }

        this.ws.onclose = () => {
          console.log('WebSocket disconnected')
          this.attemptReconnect()
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(
        `Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`
      )
      setTimeout(() => this.connect(), this.reconnectDelay)
    }
  }

  private handleMessage(message: WebSocketMessage) {
    const handlers = this.handlers.get(message.type) || []
    handlers.forEach((handler) => handler(message))
  }

  on(type: MessageType, handler: MessageHandler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, [])
    }
    this.handlers.get(type)!.push(handler)
  }

  off(type: MessageType, handler: MessageHandler) {
    const handlers = this.handlers.get(type)
    if (handlers) {
      const index = handlers.indexOf(handler)
      if (index > -1) {
        handlers.splice(index, 1)
      }
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.error('WebSocket not connected')
    }
  }

  startStreaming(modelName: string, inputData: number[], timeSteps: number) {
    this.send({
      action: 'stream',
      model_name: modelName,
      input_data: inputData,
      time_steps: timeSteps,
    })
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}
