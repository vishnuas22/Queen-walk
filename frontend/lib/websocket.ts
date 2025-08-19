// Real-time WebSocket service for MasterX collaboration

// ===== TYPES =====

export interface WebSocketMessage {
  id: string
  type: string
  payload: any
  timestamp: Date
  userId?: string
  sessionId?: string
}

export interface CollaborationEvent {
  type: 'user_joined' | 'user_left' | 'typing_start' | 'typing_stop' | 'message_sent' | 'session_updated'
  userId: string
  sessionId: string
  data?: any
  timestamp: Date
}

export interface UserPresence {
  userId: string
  name: string
  avatar?: string
  status: 'online' | 'away' | 'busy' | 'offline'
  lastSeen: Date
  currentSession?: string
  isTyping?: boolean
}

export interface WebSocketConfig {
  url: string
  reconnectInterval: number
  maxReconnectAttempts: number
  heartbeatInterval: number
  enableLogging: boolean
}

// ===== WEBSOCKET SERVICE =====

export class WebSocketService {
  private static instance: WebSocketService
  private ws: WebSocket | null = null
  private config: WebSocketConfig
  private isConnected = false
  private reconnectAttempts = 0
  private heartbeatTimer: NodeJS.Timeout | null = null
  private reconnectTimer: NodeJS.Timeout | null = null
  
  // Event handlers
  private messageHandlers = new Map<string, ((message: WebSocketMessage) => void)[]>()
  private connectionHandlers: ((connected: boolean) => void)[] = []
  private errorHandlers: ((error: Event) => void)[] = []
  
  // Collaboration state
  private currentUserId: string | null = null
  private currentSessionId: string | null = null
  private connectedUsers = new Map<string, UserPresence>()
  private typingUsers = new Set<string>()

  private constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = {
      url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/ws',
      reconnectInterval: 3000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      enableLogging: process.env.NODE_ENV === 'development',
      ...config,
    }
  }

  public static getInstance(config?: Partial<WebSocketConfig>): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService(config)
    }
    return WebSocketService.instance
  }

  // ===== CONNECTION MANAGEMENT =====

  public connect(userId: string, sessionId?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.isConnected) {
        resolve()
        return
      }

      this.currentUserId = userId
      this.currentSessionId = sessionId || null

      try {
        this.ws = new WebSocket(this.config.url)
        
        this.ws.onopen = () => {
          this.isConnected = true
          this.reconnectAttempts = 0
          this.startHeartbeat()
          
          // Send authentication message
          this.send({
            type: 'auth',
            payload: {
              userId,
              sessionId,
            },
          })

          this.log('WebSocket connected')
          this.notifyConnectionHandlers(true)
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            this.handleMessage(message)
          } catch (error) {
            this.log('Failed to parse WebSocket message:', error)
          }
        }

        this.ws.onclose = (event) => {
          this.isConnected = false
          this.stopHeartbeat()
          this.log('WebSocket disconnected:', event.code, event.reason)
          this.notifyConnectionHandlers(false)
          
          if (!event.wasClean && this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.scheduleReconnect()
          }
        }

        this.ws.onerror = (error) => {
          this.log('WebSocket error:', error)
          this.notifyErrorHandlers(error)
          reject(error)
        }

      } catch (error) {
        this.log('Failed to create WebSocket connection:', error)
        reject(error)
      }
    })
  }

  public disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect')
      this.ws = null
    }
    
    this.isConnected = false
    this.stopHeartbeat()
    this.clearReconnectTimer()
    this.connectedUsers.clear()
    this.typingUsers.clear()
  }

  private scheduleReconnect(): void {
    this.clearReconnectTimer()
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++
      this.log(`Reconnection attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts}`)
      
      if (this.currentUserId) {
        this.connect(this.currentUserId, this.currentSessionId).catch(() => {
          // Reconnection failed, will try again if under limit
        })
      }
    }, this.config.reconnectInterval)
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  // ===== HEARTBEAT =====

  private startHeartbeat(): void {
    this.stopHeartbeat()
    
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected) {
        this.send({
          type: 'ping',
          payload: { timestamp: Date.now() },
        })
      }
    }, this.config.heartbeatInterval)
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  // ===== MESSAGE HANDLING =====

  private handleMessage(message: WebSocketMessage): void {
    this.log('Received message:', message)

    // Handle system messages
    switch (message.type) {
      case 'pong':
        // Heartbeat response
        break
        
      case 'user_presence':
        this.handleUserPresence(message.payload)
        break
        
      case 'typing_indicator':
        this.handleTypingIndicator(message.payload)
        break
        
      case 'collaboration_event':
        this.handleCollaborationEvent(message.payload)
        break
        
      default:
        // Forward to registered handlers
        const handlers = this.messageHandlers.get(message.type) || []
        handlers.forEach(handler => handler(message))
    }
  }

  private handleUserPresence(payload: any): void {
    const { userId, presence } = payload
    
    if (presence) {
      this.connectedUsers.set(userId, presence)
    } else {
      this.connectedUsers.delete(userId)
    }
    
    // Notify presence handlers
    const handlers = this.messageHandlers.get('user_presence') || []
    handlers.forEach(handler => handler({
      id: `presence-${Date.now()}`,
      type: 'user_presence',
      payload: { userId, presence, connectedUsers: Array.from(this.connectedUsers.values()) },
      timestamp: new Date(),
    }))
  }

  private handleTypingIndicator(payload: any): void {
    const { userId, isTyping } = payload
    
    if (isTyping) {
      this.typingUsers.add(userId)
    } else {
      this.typingUsers.delete(userId)
    }
    
    // Notify typing handlers
    const handlers = this.messageHandlers.get('typing_indicator') || []
    handlers.forEach(handler => handler({
      id: `typing-${Date.now()}`,
      type: 'typing_indicator',
      payload: { userId, isTyping, typingUsers: Array.from(this.typingUsers) },
      timestamp: new Date(),
    }))
  }

  private handleCollaborationEvent(payload: CollaborationEvent): void {
    // Notify collaboration handlers
    const handlers = this.messageHandlers.get('collaboration_event') || []
    handlers.forEach(handler => handler({
      id: `collab-${Date.now()}`,
      type: 'collaboration_event',
      payload,
      timestamp: new Date(),
    }))
  }

  // ===== SENDING MESSAGES =====

  public send(message: Omit<WebSocketMessage, 'id' | 'timestamp'>): boolean {
    if (!this.isConnected || !this.ws) {
      this.log('Cannot send message: WebSocket not connected')
      return false
    }

    const fullMessage: WebSocketMessage = {
      ...message,
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      userId: this.currentUserId || undefined,
      sessionId: this.currentSessionId || undefined,
    }

    try {
      this.ws.send(JSON.stringify(fullMessage))
      this.log('Sent message:', fullMessage)
      return true
    } catch (error) {
      this.log('Failed to send message:', error)
      return false
    }
  }

  // ===== COLLABORATION METHODS =====

  public joinSession(sessionId: string): void {
    this.currentSessionId = sessionId
    this.send({
      type: 'join_session',
      payload: { sessionId },
    })
  }

  public leaveSession(): void {
    if (this.currentSessionId) {
      this.send({
        type: 'leave_session',
        payload: { sessionId: this.currentSessionId },
      })
      this.currentSessionId = null
    }
  }

  public sendTypingIndicator(isTyping: boolean): void {
    this.send({
      type: 'typing_indicator',
      payload: { isTyping },
    })
  }

  public sendMessage(content: string, messageType = 'chat_message'): void {
    this.send({
      type: messageType,
      payload: { content },
    })
  }

  public shareSession(sessionId: string, userIds: string[]): void {
    this.send({
      type: 'share_session',
      payload: { sessionId, userIds },
    })
  }

  // ===== EVENT HANDLERS =====

  public onMessage(type: string, handler: (message: WebSocketMessage) => void): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, [])
    }
    
    this.messageHandlers.get(type)!.push(handler)
    
    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(type)
      if (handlers) {
        const index = handlers.indexOf(handler)
        if (index > -1) {
          handlers.splice(index, 1)
        }
      }
    }
  }

  public onConnection(handler: (connected: boolean) => void): () => void {
    this.connectionHandlers.push(handler)
    
    // Return unsubscribe function
    return () => {
      const index = this.connectionHandlers.indexOf(handler)
      if (index > -1) {
        this.connectionHandlers.splice(index, 1)
      }
    }
  }

  public onError(handler: (error: Event) => void): () => void {
    this.errorHandlers.push(handler)
    
    // Return unsubscribe function
    return () => {
      const index = this.errorHandlers.indexOf(handler)
      if (index > -1) {
        this.errorHandlers.splice(index, 1)
      }
    }
  }

  // ===== UTILITY METHODS =====

  private notifyConnectionHandlers(connected: boolean): void {
    this.connectionHandlers.forEach(handler => handler(connected))
  }

  private notifyErrorHandlers(error: Event): void {
    this.errorHandlers.forEach(handler => handler(error))
  }

  private log(...args: any[]): void {
    if (this.config.enableLogging) {
      console.log('[WebSocket]', ...args)
    }
  }

  // ===== GETTERS =====

  public getConnectionStatus(): boolean {
    return this.isConnected
  }

  public getConnectedUsers(): UserPresence[] {
    return Array.from(this.connectedUsers.values())
  }

  public getTypingUsers(): string[] {
    return Array.from(this.typingUsers)
  }

  public getCurrentUserId(): string | null {
    return this.currentUserId
  }

  public getCurrentSessionId(): string | null {
    return this.currentSessionId
  }
}

export default WebSocketService
