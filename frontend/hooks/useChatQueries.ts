// Intelligent caching hooks for chat functionality

import React from 'react'
import { useQuery, useMutation, useQueryClient, useInfiniteQuery } from '@tanstack/react-query'
import { queryKeys, optimisticUpdates, backgroundSync } from '../lib/queryClient'
import { apiClient } from '../lib/api'

// ===== TYPES =====

interface Message {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  isStreaming?: boolean
}

interface ChatSession {
  session_id: string
  title: string
  created_at: string
  message_count: number
  last_message_at?: string
}

interface SendMessageResponse {
  message_id: string
  response: string
  session_id: string
}

// ===== CHAT SESSIONS =====

// Hook to fetch and cache chat sessions
export const useChatSessions = () => {
  return useQuery({
    queryKey: queryKeys.chat.sessions(),
    queryFn: async (): Promise<{ sessions: ChatSession[] }> => {
      const data = await apiClient.getChatSessions()
      return data
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    refetchOnWindowFocus: true,
    refetchOnMount: false, // Don't refetch if data is fresh
  })
}

// ===== CHAT MESSAGES =====

// Hook to fetch messages for a specific session with infinite scrolling
export const useChatMessages = (sessionId: string | null, enabled: boolean = true) => {
  return useInfiniteQuery({
    queryKey: queryKeys.chat.messages(sessionId || ''),
    queryFn: async ({ pageParam = 0 }): Promise<{ messages: Message[], hasMore: boolean, nextCursor?: number }> => {
      if (!sessionId) return { messages: [], hasMore: false }

      // Simulate API call for message pagination
      const response = await fetch(`/api/chat/sessions/${sessionId}/messages?cursor=${pageParam}&limit=50`)
      if (!response.ok) throw new Error('Failed to fetch messages')

      const data = await response.json()
      return {
        messages: data.messages || [],
        hasMore: data.hasMore || false,
        nextCursor: data.nextCursor,
      }
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage) => lastPage.hasMore ? lastPage.nextCursor : undefined,
    enabled: enabled && !!sessionId,
    staleTime: 30 * 1000, // 30 seconds
    gcTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false, // Don't refetch messages on focus
    refetchOnMount: false,
  })
}

// Hook to get flattened messages from infinite query
export const useFlattenedMessages = (sessionId: string | null) => {
  const { data, ...rest } = useChatMessages(sessionId)
  
  const messages = data?.pages.flatMap(page => page.messages) || []
  
  return {
    messages,
    ...rest,
  }
}

// ===== SEND MESSAGE MUTATION =====

// Hook to send messages with optimistic updates
export const useSendMessage = (sessionId: string | null) => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (content: string): Promise<SendMessageResponse> => {
      const response = await apiClient.sendMessage(content, sessionId || undefined)
      return response
    },
    
    onMutate: async (content: string) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: queryKeys.chat.messages(sessionId || '') })
      
      // Create optimistic user message
      const optimisticUserMessage: Message = {
        id: `temp-user-${Date.now()}`,
        content,
        sender: 'user',
        timestamp: new Date(),
      }
      
      // Add optimistic user message
      if (sessionId) {
        optimisticUpdates.addMessage(sessionId, optimisticUserMessage)
      }
      
      // Return context for rollback
      return { optimisticUserMessage }
    },
    
    onSuccess: (data, content, context) => {
      if (!sessionId) return
      
      // Remove optimistic user message and add real messages
      if (context?.optimisticUserMessage) {
        optimisticUpdates.removeMessage(sessionId, context.optimisticUserMessage.id)
      }
      
      // Add real user message
      const realUserMessage: Message = {
        id: `user-${Date.now()}`,
        content,
        sender: 'user',
        timestamp: new Date(),
      }
      
      // Add AI response
      const aiMessage: Message = {
        id: data.message_id,
        content: data.response,
        sender: 'ai',
        timestamp: new Date(),
      }
      
      optimisticUpdates.addMessage(sessionId, realUserMessage)
      optimisticUpdates.addMessage(sessionId, aiMessage)
      
      // Invalidate sessions to update last message time
      queryClient.invalidateQueries({ queryKey: queryKeys.chat.sessions() })
    },
    
    onError: (error, content, context) => {
      // Rollback optimistic update
      if (sessionId && context?.optimisticUserMessage) {
        optimisticUpdates.removeMessage(sessionId, context.optimisticUserMessage.id)
      }
      
      console.error('Failed to send message:', error)
    },
  })
}

// ===== SESSION MANAGEMENT =====

// Hook to create new chat session
export const useCreateSession = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (): Promise<ChatSession> => {
      const response = await fetch('/api/chat/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      
      if (!response.ok) throw new Error('Failed to create session')
      return response.json()
    },
    
    onSuccess: (newSession) => {
      // Add new session to cache
      queryClient.setQueryData(queryKeys.chat.sessions(), (old: any) => {
        if (!old) return { sessions: [newSession] }
        return { sessions: [newSession, ...old.sessions] }
      })
    },
  })
}

// Hook to delete chat session
export const useDeleteSession = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (sessionId: string): Promise<void> => {
      const response = await fetch(`/api/chat/sessions/${sessionId}`, {
        method: 'DELETE',
      })
      
      if (!response.ok) throw new Error('Failed to delete session')
    },
    
    onMutate: async (sessionId: string) => {
      // Cancel queries
      await queryClient.cancelQueries({ queryKey: queryKeys.chat.sessions() })
      await queryClient.cancelQueries({ queryKey: queryKeys.chat.session(sessionId) })
      
      // Optimistically remove session
      queryClient.setQueryData(queryKeys.chat.sessions(), (old: any) => {
        if (!old) return old
        return {
          sessions: old.sessions.filter((session: ChatSession) => session.session_id !== sessionId)
        }
      })
      
      // Remove session messages from cache
      queryClient.removeQueries({ queryKey: queryKeys.chat.messages(sessionId) })
    },
    
    onError: (error, sessionId) => {
      // Refetch sessions on error to restore state
      queryClient.invalidateQueries({ queryKey: queryKeys.chat.sessions() })
      console.error('Failed to delete session:', error)
    },
  })
}

// ===== BACKGROUND SYNC HOOKS =====

// Hook to manage background sync for active session
export const useBackgroundSync = (sessionId: string | null, enabled: boolean = true) => {
  const queryClient = useQueryClient()
  
  // Start/stop background sync based on session and enabled state
  React.useEffect(() => {
    if (sessionId && enabled) {
      backgroundSync.start(sessionId)
      console.log(`🔄 Background sync started for session: ${sessionId}`)
    } else if (sessionId) {
      backgroundSync.stop(sessionId)
      console.log(`⏹️ Background sync stopped for session: ${sessionId}`)
    }
    
    return () => {
      if (sessionId) {
        backgroundSync.stop(sessionId)
      }
    }
  }, [sessionId, enabled])
}

// ===== CACHE WARMING =====

// Hook to warm cache with commonly needed data
export const useCacheWarming = () => {
  const queryClient = useQueryClient()
  
  const warmCache = React.useCallback(async () => {
    try {
      // Prefetch sessions
      await queryClient.prefetchQuery({
        queryKey: queryKeys.chat.sessions(),
        queryFn: async () => {
          const data = await apiClient.getChatSessions()
          return data
        },
        staleTime: 2 * 60 * 1000,
      })
      
      console.log('🔥 Cache warmed successfully')
    } catch (error) {
      console.warn('Failed to warm cache:', error)
    }
  }, [queryClient])
  
  return { warmCache }
}

// ===== OFFLINE SUPPORT =====

// Hook to handle offline message queuing
export const useOfflineMessageQueue = () => {
  const [offlineQueue, setOfflineQueue] = React.useState<Array<{ content: string, sessionId: string }>>([])
  const sendMessage = useSendMessage(null)
  
  // Add message to offline queue
  const queueMessage = React.useCallback((content: string, sessionId: string) => {
    setOfflineQueue(prev => [...prev, { content, sessionId }])
    
    // Store in localStorage for persistence
    const stored = localStorage.getItem('masterx-offline-queue') || '[]'
    const queue = JSON.parse(stored)
    queue.push({ content, sessionId, timestamp: Date.now() })
    localStorage.setItem('masterx-offline-queue', JSON.stringify(queue))
  }, [])
  
  // Process offline queue when back online
  const processOfflineQueue = React.useCallback(async () => {
    const stored = localStorage.getItem('masterx-offline-queue')
    if (!stored) return
    
    const queue = JSON.parse(stored)
    if (queue.length === 0) return
    
    console.log(`📤 Processing ${queue.length} offline messages...`)
    
    for (const item of queue) {
      try {
        await sendMessage.mutateAsync(item.content)
      } catch (error) {
        console.error('Failed to send offline message:', error)
      }
    }
    
    // Clear queue after processing
    localStorage.removeItem('masterx-offline-queue')
    setOfflineQueue([])
  }, [sendMessage])
  
  // Auto-process queue when coming online
  React.useEffect(() => {
    const handleOnline = () => {
      processOfflineQueue()
    }
    
    window.addEventListener('online', handleOnline)
    return () => window.removeEventListener('online', handleOnline)
  }, [processOfflineQueue])
  
  return {
    offlineQueue,
    queueMessage,
    processOfflineQueue,
  }
}
