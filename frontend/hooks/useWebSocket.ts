// React hook for WebSocket collaboration features

import { useState, useEffect, useCallback, useRef } from 'react'
import { WebSocketService, WebSocketMessage, UserPresence, CollaborationEvent } from '../lib/websocket'
import { useUserState, useUIActions } from '../store'

// ===== TYPES =====

interface UseWebSocketOptions {
  autoConnect?: boolean
  sessionId?: string
  onMessage?: (message: WebSocketMessage) => void
  onUserPresence?: (users: UserPresence[]) => void
  onTypingIndicator?: (typingUsers: string[]) => void
  onCollaborationEvent?: (event: CollaborationEvent) => void
}

interface UseWebSocketReturn {
  // Connection state
  isConnected: boolean
  isConnecting: boolean
  connectionError: string | null
  
  // User presence
  connectedUsers: UserPresence[]
  typingUsers: string[]
  
  // Actions
  connect: () => Promise<void>
  disconnect: () => void
  sendMessage: (content: string, type?: string) => void
  sendTypingIndicator: (isTyping: boolean) => void
  joinSession: (sessionId: string) => void
  leaveSession: () => void
  shareSession: (sessionId: string, userIds: string[]) => void
  
  // Event subscriptions
  subscribe: (type: string, handler: (message: WebSocketMessage) => void) => () => void
}

// ===== HOOK IMPLEMENTATION =====

export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketReturn => {
  const {
    autoConnect = false,
    sessionId,
    onMessage,
    onUserPresence,
    onTypingIndicator,
    onCollaborationEvent,
  } = options

  // State
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [connectedUsers, setConnectedUsers] = useState<UserPresence[]>([])
  const [typingUsers, setTypingUsers] = useState<string[]>([])

  // Store and services
  const userState = useUserState()
  const { addNotification } = useUIActions()
  const wsRef = useRef<WebSocketService | null>(null)
  const unsubscribersRef = useRef<(() => void)[]>([])

  // Initialize WebSocket service
  useEffect(() => {
    wsRef.current = WebSocketService.getInstance()
    
    return () => {
      // Cleanup on unmount
      unsubscribersRef.current.forEach(unsubscribe => unsubscribe())
      wsRef.current?.disconnect()
    }
  }, [])

  // Setup event listeners
  useEffect(() => {
    const ws = wsRef.current
    if (!ws) return

    // Clear previous subscriptions
    unsubscribersRef.current.forEach(unsubscribe => unsubscribe())
    unsubscribersRef.current = []

    // Connection status
    const unsubscribeConnection = ws.onConnection((connected) => {
      setIsConnected(connected)
      setIsConnecting(false)
      
      if (connected) {
        setConnectionError(null)
        addNotification({
          type: 'success',
          title: 'Connected',
          message: 'Real-time collaboration is now active',
          duration: 3000,
        })
      } else {
        addNotification({
          type: 'warning',
          title: 'Disconnected',
          message: 'Real-time collaboration is temporarily unavailable',
          duration: 5000,
        })
      }
    })

    // Error handling
    const unsubscribeError = ws.onError((error) => {
      setConnectionError('Connection failed')
      setIsConnecting(false)
      
      addNotification({
        type: 'error',
        title: 'Connection Error',
        message: 'Failed to connect to collaboration service',
        duration: 5000,
      })
    })

    // User presence updates
    const unsubscribePresence = ws.onMessage('user_presence', (message) => {
      const { connectedUsers: users } = message.payload
      setConnectedUsers(users || [])
      onUserPresence?.(users || [])
    })

    // Typing indicators
    const unsubscribeTyping = ws.onMessage('typing_indicator', (message) => {
      const { typingUsers: users } = message.payload
      setTypingUsers(users || [])
      onTypingIndicator?.(users || [])
    })

    // Collaboration events
    const unsubscribeCollab = ws.onMessage('collaboration_event', (message) => {
      const event = message.payload as CollaborationEvent
      onCollaborationEvent?.(event)
      
      // Handle specific collaboration events
      switch (event.type) {
        case 'user_joined':
          addNotification({
            type: 'info',
            title: 'User Joined',
            message: `A user joined the session`,
            duration: 3000,
          })
          break
          
        case 'user_left':
          addNotification({
            type: 'info',
            title: 'User Left',
            message: `A user left the session`,
            duration: 3000,
          })
          break
      }
    })

    // General message handler
    const unsubscribeMessage = ws.onMessage('chat_message', (message) => {
      onMessage?.(message)
    })

    // Store unsubscribers
    unsubscribersRef.current = [
      unsubscribeConnection,
      unsubscribeError,
      unsubscribePresence,
      unsubscribeTyping,
      unsubscribeCollab,
      unsubscribeMessage,
    ]

  }, [onMessage, onUserPresence, onTypingIndicator, onCollaborationEvent, addNotification])

  // Auto-connect when user is authenticated
  useEffect(() => {
    if (autoConnect && userState?.isAuthenticated && userState?.user?.id && !isConnected && !isConnecting) {
      connect()
    }
  }, [autoConnect, userState?.isAuthenticated, userState?.user?.id, isConnected, isConnecting])

  // ===== ACTIONS =====

  const connect = useCallback(async () => {
    const ws = wsRef.current
    if (!ws || !userState?.user?.id || isConnected || isConnecting) return

    setIsConnecting(true)
    setConnectionError(null)

    try {
      await ws.connect(userState.user.id, sessionId)
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : 'Connection failed')
      setIsConnecting(false)
    }
  }, [userState?.user?.id, sessionId, isConnected, isConnecting])

  const disconnect = useCallback(() => {
    const ws = wsRef.current
    if (!ws) return

    ws.disconnect()
    setIsConnected(false)
    setIsConnecting(false)
    setConnectedUsers([])
    setTypingUsers([])
  }, [])

  const sendMessage = useCallback((content: string, type = 'chat_message') => {
    const ws = wsRef.current
    if (!ws || !isConnected) return

    ws.sendMessage(content, type)
  }, [isConnected])

  const sendTypingIndicator = useCallback((isTyping: boolean) => {
    const ws = wsRef.current
    if (!ws || !isConnected) return

    ws.sendTypingIndicator(isTyping)
  }, [isConnected])

  const joinSession = useCallback((sessionId: string) => {
    const ws = wsRef.current
    if (!ws || !isConnected) return

    ws.joinSession(sessionId)
  }, [isConnected])

  const leaveSession = useCallback(() => {
    const ws = wsRef.current
    if (!ws || !isConnected) return

    ws.leaveSession()
  }, [isConnected])

  const shareSession = useCallback((sessionId: string, userIds: string[]) => {
    const ws = wsRef.current
    if (!ws || !isConnected) return

    ws.shareSession(sessionId, userIds)
    
    addNotification({
      type: 'success',
      title: 'Session Shared',
      message: `Session shared with ${userIds.length} user(s)`,
      duration: 3000,
    })
  }, [isConnected, addNotification])

  const subscribe = useCallback((type: string, handler: (message: WebSocketMessage) => void) => {
    const ws = wsRef.current
    if (!ws) return () => {}

    return ws.onMessage(type, handler)
  }, [])

  return {
    // Connection state
    isConnected,
    isConnecting,
    connectionError,
    
    // User presence
    connectedUsers,
    typingUsers,
    
    // Actions
    connect,
    disconnect,
    sendMessage,
    sendTypingIndicator,
    joinSession,
    leaveSession,
    shareSession,
    
    // Event subscriptions
    subscribe,
  }
}

// ===== TYPING INDICATOR HOOK =====

export const useTypingIndicator = (sessionId?: string) => {
  const { sendTypingIndicator, typingUsers, isConnected } = useWebSocket({ sessionId })
  const [isTyping, setIsTyping] = useState(false)
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const startTyping = useCallback(() => {
    if (!isConnected) return

    if (!isTyping) {
      setIsTyping(true)
      sendTypingIndicator(true)
    }

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current)
    }

    // Set new timeout to stop typing after 3 seconds of inactivity
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false)
      sendTypingIndicator(false)
    }, 3000)
  }, [isConnected, isTyping, sendTypingIndicator])

  const stopTyping = useCallback(() => {
    if (isTyping) {
      setIsTyping(false)
      sendTypingIndicator(false)
    }

    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current)
      typingTimeoutRef.current = null
    }
  }, [isTyping, sendTypingIndicator])

  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current)
      }
    }
  }, [])

  return {
    isTyping,
    typingUsers,
    startTyping,
    stopTyping,
  }
}

export default useWebSocket
