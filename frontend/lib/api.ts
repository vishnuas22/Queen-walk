/**
 * API Client for MasterX Quantum Intelligence Platform
 * 
 * Centralized API client with proper error handling, type safety,
 * and integration with the backend quantum intelligence services.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ChatMessage {
  message_id?: string
  session_id?: string
  user_id?: string
  message_type?: 'text' | 'image' | 'audio' | 'video' | 'file' | 'code'
  content: string
  metadata?: Record<string, any>
  timestamp?: string
}

export interface ChatResponse {
  success: boolean
  message: string
  response: string
  session_id: string
  message_id: string
  suggestions?: string[]
  learning_insights?: Record<string, any>
  personalization_data?: Record<string, any>
  timestamp: string
}

export interface ChatSession {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  user_id?: string
}

export interface StreamingResponse {
  type: 'content' | 'metadata' | 'complete'
  data: any
}

class APIClient {
  private baseURL: string

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`
    
    const defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error)
      throw error
    }
  }

  // ===== CHAT API METHODS =====

  async sendMessage(message: string, sessionId?: string): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/v1/chat/message', {
      method: 'POST',
      body: JSON.stringify({
        message,
        session_id: sessionId,
        message_type: 'text'
      })
    })
  }

  async streamMessage(
    message: string, 
    sessionId?: string,
    onChunk?: (chunk: StreamingResponse) => void
  ): Promise<void> {
    const url = `${this.baseURL}/api/v1/chat/stream`
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        stream: true
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body reader available')
    }

    const decoder = new TextDecoder()

    try {
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              onChunk?.(data)
            } catch (error) {
              console.error('Failed to parse streaming data:', error)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  async getChatSessions(): Promise<{ sessions: ChatSession[] }> {
    return this.request<{ sessions: ChatSession[] }>('/api/v1/chat/sessions')
  }

  async getSessionMessages(sessionId: string): Promise<{ messages: ChatMessage[] }> {
    return this.request<{ messages: ChatMessage[] }>(`/api/v1/chat/sessions/${sessionId}/messages`)
  }

  // ===== LEARNING API METHODS =====

  async getLearningProgress(userId?: string, timeframe: string = 'week') {
    return this.request('/api/v1/learning/progress', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId || 'dev_user_001',
        timeframe
      })
    })
  }

  // ===== PERSONALIZATION API METHODS =====

  async getPersonalizationProfile(userId?: string) {
    return this.request(`/api/v1/personalization/profile?user_id=${userId || 'dev_user_001'}`)
  }

  // ===== ANALYTICS API METHODS =====

  async getAnalyticsDashboard() {
    return this.request('/api/v1/analytics/dashboard')
  }

  async getLearningAnalytics() {
    return this.request('/api/v1/analytics/learning-progress')
  }

  // ===== CONTENT GENERATION API METHODS =====

  async generateContent(params: {
    topic: string
    content_type: string
    difficulty_level?: number
    duration_minutes?: number
    learning_objectives?: string[]
  }) {
    return this.request('/api/v1/content/generate', {
      method: 'POST',
      body: JSON.stringify(params)
    })
  }

  // ===== FILE UPLOAD API METHODS =====

  async uploadFile(file: File): Promise<{ file_id: string; file_url: string }> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${this.baseURL}/api/v1/files/upload`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`)
    }

    return response.json()
  }

  // ===== HEALTH CHECK =====

  async healthCheck() {
    return this.request('/health')
  }
}

// Export singleton instance
export const apiClient = new APIClient()

// Export utility functions
export const formatTimestamp = (timestamp: string | Date): string => {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export const formatDate = (timestamp: string | Date): string => {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
  return date.toLocaleDateString([], { 
    year: 'numeric', 
    month: 'short', 
    day: 'numeric' 
  })
}

export const generateSessionTitle = (firstMessage: string): string => {
  // Generate a title from the first message (max 50 chars)
  const title = firstMessage.length > 50 
    ? firstMessage.substring(0, 47) + '...'
    : firstMessage
  
  return title || 'New Conversation'
}
