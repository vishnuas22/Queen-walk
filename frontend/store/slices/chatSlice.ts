// Chat slice for Zustand store

import type { StateCreator } from 'zustand'
import type { Store, ChatState, ChatActions, Message, ChatSession } from '../types'
import { apiClient } from '../../lib/api'
import { UndoRedoManager } from '../undoRedo'

// ===== INITIAL STATE =====

const initialChatState: ChatState = {
  // Current session
  currentSessionId: null,
  currentMessages: [],
  
  // All sessions
  sessions: [],
  sessionsLoading: false,
  sessionsError: null,
  
  // Message state
  isLoading: false,
  isStreaming: false,
  streamingMessageId: null,
  
  // Input state
  inputMessage: '',
  inputHistory: [],
  inputHistoryIndex: -1,
  
  // UI state
  sidebarOpen: false,
  selectedMessageId: null,
  
  // Performance state
  virtualScrollEnabled: true,
  messagesInView: 50,
  totalMessagesLoaded: 0,
}

// ===== CHAT SLICE CREATOR =====

export const createChatSlice: StateCreator<
  Store,
  [['zustand/immer', never], ['zustand/devtools', never], ['zustand/persist', unknown], ['zustand/subscribeWithSelector', never]],
  [],
  ChatState & ChatActions
> = (set, get, api) => ({
  // Initial state - properly namespaced under 'chat'
  chat: initialChatState,

  // ===== SESSION MANAGEMENT =====

  setCurrentSession: (sessionId: string | null) => {
    set((state) => {
      state.chat.currentSessionId = sessionId
      state.chat.currentMessages = []
      state.chat.selectedMessageId = null
      state.chat.inputMessage = ''
      state.chat.inputHistoryIndex = -1
    })

    // Load messages for the session
    if (sessionId) {
      get().loadSessionMessages(sessionId)
    }
  },

  createSession: async () => {
    set((state) => {
      state.chat.sessionsLoading = true
      state.chat.sessionsError = null
    })

    try {
      const response = await fetch('/api/chat/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })

      if (!response.ok) throw new Error('Failed to create session')
      
      const newSession: ChatSession = await response.json()

      set((state) => {
        state.chat.sessions.unshift(newSession)
        state.chat.currentSessionId = newSession.session_id
        state.chat.currentMessages = []
        state.chat.sessionsLoading = false
      })

      return newSession.session_id
    } catch (error) {
      set((state) => {
        state.chat.sessionsLoading = false
        state.chat.sessionsError = error instanceof Error ? error.message : 'Unknown error'
      })
      throw error
    }
  },

  deleteSession: async (sessionId: string) => {
    try {
      const response = await fetch(`/api/chat/sessions/${sessionId}`, {
        method: 'DELETE',
      })

      if (!response.ok) throw new Error('Failed to delete session')

      set((state) => {
        // Remove session from list
        state.chat.sessions = state.chat.sessions.filter(s => s.session_id !== sessionId)
        
        // If this was the current session, clear it
        if (state.chat.currentSessionId === sessionId) {
          state.chat.currentSessionId = null
          state.chat.currentMessages = []
          state.chat.selectedMessageId = null
        }
      })
    } catch (error) {
      console.error('Failed to delete session:', error)
      throw error
    }
  },

  updateSessionTitle: async (sessionId: string, title: string) => {
    try {
      const response = await fetch(`/api/chat/sessions/${sessionId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      })

      if (!response.ok) throw new Error('Failed to update session title')

      set((state) => {
        const session = state.chat.sessions.find(s => s.session_id === sessionId)
        if (session) {
          session.title = title
        }
      })
    } catch (error) {
      console.error('Failed to update session title:', error)
      throw error
    }
  },

  // ===== MESSAGE MANAGEMENT =====

  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => {
    const newMessage: Message = {
      ...message,
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
    }

    set((state) => {
      state.chat.currentMessages.push(newMessage)
      state.chat.totalMessagesLoaded = state.chat.currentMessages.length
    })

    // Track change for undo/redo
    const undoRedo = UndoRedoManager.getInstance()
    undoRedo.trackChange('Add Message', `Added ${message.sender} message`, 'chat')
  },

  updateMessage: (messageId: string, updates: Partial<Message>) => {
    set((state) => {
      const messageIndex = state.chat.currentMessages.findIndex(m => m.id === messageId)
      if (messageIndex !== -1) {
        Object.assign(state.chat.currentMessages[messageIndex], updates)
      }
    })
  },

  deleteMessage: (messageId: string) => {
    set((state) => {
      state.chat.currentMessages = state.chat.currentMessages.filter(m => m.id !== messageId)
      state.chat.totalMessagesLoaded = state.chat.currentMessages.length

      if (state.chat.selectedMessageId === messageId) {
        state.chat.selectedMessageId = null
      }
    })

    // Track change for undo/redo
    const undoRedo = UndoRedoManager.getInstance()
    undoRedo.trackChange('Delete Message', 'Deleted message', 'chat')
  },

  clearMessages: () => {
    set((state) => {
      state.chat.currentMessages = []
      state.chat.selectedMessageId = null
      state.chat.totalMessagesLoaded = 0
    })
  },

  // ===== INPUT MANAGEMENT =====

  setInputMessage: (message: string) => {
    set((state) => {
      state.chat.inputMessage = message
    })
  },

  addToInputHistory: (message: string) => {
    set((state) => {
      // Add to history if it's not empty and not the same as the last entry
      if (message.trim() && state.chat.inputHistory[0] !== message) {
        state.chat.inputHistory.unshift(message)
        
        // Keep only last 50 entries
        if (state.chat.inputHistory.length > 50) {
          state.chat.inputHistory = state.chat.inputHistory.slice(0, 50)
        }
      }
      
      state.chat.inputHistoryIndex = -1
    })
  },

  navigateInputHistory: (direction: 'up' | 'down') => {
    set((state) => {
      const history = state.chat.inputHistory
      if (history.length === 0) return

      if (direction === 'up') {
        if (state.chat.inputHistoryIndex < history.length - 1) {
          state.chat.inputHistoryIndex++
          state.chat.inputMessage = history[state.chat.inputHistoryIndex]
        }
      } else {
        if (state.chat.inputHistoryIndex > 0) {
          state.chat.inputHistoryIndex--
          state.chat.inputMessage = history[state.chat.inputHistoryIndex]
        } else if (state.chat.inputHistoryIndex === 0) {
          state.chat.inputHistoryIndex = -1
          state.chat.inputMessage = ''
        }
      }
    })
  },

  // ===== SENDING MESSAGES =====

  sendMessage: async (content: string) => {
    const {
      currentSessionId,
      addMessage,
      updateMessage,
      addToInputHistory,
      setLoading,
      setStreaming,
      setInputMessage
    } = get()

    // Clear input immediately for better UX
    setInputMessage('')

    // Create optimistic user message
    const userMessageId = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    addMessage({
      id: userMessageId,
      content,
      sender: 'user',
      timestamp: new Date(),
    })

    // Add to input history
    addToInputHistory(content)

    // Create optimistic AI message placeholder
    const aiMessageId = `ai-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    addMessage({
      id: aiMessageId,
      content: '',
      sender: 'ai',
      timestamp: new Date(),
      isStreaming: true,
    })

    // Set loading and streaming states
    setLoading(true)
    setStreaming(true, aiMessageId)

    try {
      const response = await apiClient.sendMessage(content, currentSessionId || undefined)

      // Update session ID if this is a new conversation
      if (response.session_id && !currentSessionId) {
        set((state) => {
          state.chat.currentSessionId = response.session_id
        })
      }

      // Update the optimistic AI message with real response
      updateMessage(aiMessageId, {
        content: response.response || 'Sorry, I encountered an error processing your request.',
        isStreaming: false,
        metadata: {
          model: response.model,
          tokens: response.tokens,
          processingTime: response.processingTime,
        },
      })

      // Update session in list
      set((state) => {
        const session = state.chat.sessions.find(s => s.session_id === response.session_id)
        if (session) {
          session.message_count++
          session.last_message_at = new Date().toISOString()
          session.updated_at = new Date().toISOString()
        }
      })

    } catch (error) {
      console.error('Error sending message:', error)

      // Update the optimistic AI message with error
      updateMessage(aiMessageId, {
        content: 'Sorry, I encountered an error. Please try again.',
        isStreaming: false,
        metadata: {
          error: true,
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
        },
      })

      // Add retry action to the message
      updateMessage(aiMessageId, {
        metadata: {
          ...get().chat.currentMessages.find(m => m.id === aiMessageId)?.metadata,
          retryAction: () => get().resendMessage(userMessageId),
        },
      })
    } finally {
      setLoading(false)
      setStreaming(false)
    }
  },

  resendMessage: async (messageId: string) => {
    const { currentMessages, sendMessage } = get()
    const message = currentMessages.find(m => m.id === messageId)
    
    if (message && message.sender === 'user') {
      await sendMessage(message.content)
    }
  },

  // ===== UI ACTIONS =====

  toggleSidebar: () => {
    set((state) => {
      state.chat.sidebarOpen = !state.chat.sidebarOpen
    })
  },

  setSidebarOpen: (open: boolean) => {
    set((state) => {
      state.chat.sidebarOpen = open
    })
  },

  selectMessage: (messageId: string | null) => {
    set((state) => {
      state.chat.selectedMessageId = messageId
    })
  },

  // ===== LOADING STATES =====

  setLoading: (loading: boolean) => {
    set((state) => {
      state.chat.isLoading = loading
    })
  },

  setStreaming: (streaming: boolean, messageId?: string) => {
    set((state) => {
      state.chat.isStreaming = streaming
      state.chat.streamingMessageId = streaming ? messageId || null : null
    })
  },

  // ===== OPTIMISTIC UPDATE HELPERS =====

  rollbackMessage: (messageId: string) => {
    set((state) => {
      state.chat.currentMessages = state.chat.currentMessages.filter(m => m.id !== messageId)
      state.chat.totalMessagesLoaded = state.chat.currentMessages.length
    })
  },

  retryMessage: async (messageId: string) => {
    const { currentMessages, sendMessage } = get()
    const message = currentMessages.find(m => m.id === messageId)

    if (message && message.sender === 'user') {
      // Remove the failed message and retry
      get().rollbackMessage(messageId)
      await sendMessage(message.content)
    }
  },

  // ===== CONFLICT RESOLUTION =====

  resolveConflict: (localMessage: Message, serverMessage: Message) => {
    // Simple conflict resolution: server wins
    // In a more sophisticated system, you might merge changes or ask the user
    set((state) => {
      const messageIndex = state.chat.currentMessages.findIndex(m => m.id === localMessage.id)
      if (messageIndex !== -1) {
        state.chat.currentMessages[messageIndex] = {
          ...serverMessage,
          id: localMessage.id, // Keep local ID for consistency
        }
      }
    })
  },

  // ===== HELPER METHODS =====

  loadSessionMessages: async (sessionId: string) => {
    set((state) => {
      state.chat.isLoading = true
    })

    try {
      // This would typically load from React Query cache or API
      // For now, we'll simulate loading
      const messages: Message[] = []

      set((state) => {
        state.chat.currentMessages = messages
        state.chat.totalMessagesLoaded = messages.length
        state.chat.isLoading = false
      })
    } catch (error) {
      console.error('Failed to load session messages:', error)
      set((state) => {
        state.chat.isLoading = false
      })
    }
  },

  loadSessions: async () => {
    set((state) => {
      state.chat.sessionsLoading = true
      state.chat.sessionsError = null
    })

    try {
      const data = await apiClient.getChatSessions()

      set((state) => {
        state.chat.sessions = data.sessions || []
        state.chat.sessionsLoading = false
      })
    } catch (error) {
      set((state) => {
        state.chat.sessionsLoading = false
        state.chat.sessionsError = error instanceof Error ? error.message : 'Unknown error'
      })
    }
  },

  // ===== OFFLINE SUPPORT =====

  syncOfflineMessages: async () => {
    const { currentMessages } = get()
    const offlineMessages = currentMessages.filter(m => m.metadata?.offline)

    for (const message of offlineMessages) {
      try {
        if (message.sender === 'user') {
          // Retry sending offline user messages
          await get().sendMessage(message.content)
          // Remove the offline message after successful send
          get().deleteMessage(message.id)
        }
      } catch (error) {
        console.error('Failed to sync offline message:', error)
        // Mark message as failed
        get().updateMessage(message.id, {
          metadata: {
            ...message.metadata,
            syncFailed: true,
            error: error instanceof Error ? error.message : 'Sync failed',
          },
        })
      }
    }
  },

  // ===== PERFORMANCE OPTIMIZATIONS =====

  optimizeMessageList: () => {
    const { currentMessages, messagesInView } = get()

    // Keep only recent messages in memory for performance
    if (currentMessages.length > messagesInView * 2) {
      set((state) => {
        // Keep the most recent messages
        state.chat.currentMessages = state.chat.currentMessages.slice(-messagesInView)
        state.chat.totalMessagesLoaded = state.chat.currentMessages.length
      })
    }
  },
})
