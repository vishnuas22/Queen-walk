// Advanced caching configuration for React Query

import { QueryClient, DefaultOptions } from '@tanstack/react-query'

// ===== CACHE CONFIGURATION =====

// Default query options for optimal caching
const defaultQueryOptions: DefaultOptions = {
  queries: {
    // Cache data for 5 minutes by default
    staleTime: 5 * 60 * 1000, // 5 minutes
    
    // Keep data in cache for 10 minutes after component unmount
    gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
    
    // Retry failed requests 3 times with exponential backoff
    retry: (failureCount, error: any) => {
      // Don't retry on 4xx errors (client errors)
      if (error?.status >= 400 && error?.status < 500) {
        return false
      }
      // Retry up to 3 times for other errors
      return failureCount < 3
    },
    
    // Retry delay with exponential backoff
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    
    // Refetch on window focus for real-time data
    refetchOnWindowFocus: true,
    
    // Refetch when coming back online
    refetchOnReconnect: true,
    
    // Don't refetch on mount if data is fresh
    refetchOnMount: true,
    
    // Enable background refetching
    refetchInterval: false, // Disabled by default, can be enabled per query
    
    // Network mode for offline support
    networkMode: 'online',
  },
  mutations: {
    // Retry mutations once on failure
    retry: 1,
    
    // Network mode for mutations
    networkMode: 'online',
  },
}

// ===== QUERY CLIENT INSTANCE =====

export const queryClient = new QueryClient({
  defaultOptions: defaultQueryOptions,
})

// ===== QUERY KEYS =====

// Centralized query key factory for consistency
export const queryKeys = {
  // Chat-related queries
  chat: {
    all: ['chat'] as const,
    sessions: () => [...queryKeys.chat.all, 'sessions'] as const,
    session: (sessionId: string) => [...queryKeys.chat.all, 'session', sessionId] as const,
    messages: (sessionId: string) => [...queryKeys.chat.session(sessionId), 'messages'] as const,
    message: (sessionId: string, messageId: string) => [...queryKeys.chat.messages(sessionId), messageId] as const,
  },
  
  // User-related queries
  user: {
    all: ['user'] as const,
    profile: () => [...queryKeys.user.all, 'profile'] as const,
    preferences: () => [...queryKeys.user.all, 'preferences'] as const,
  },
  
  // System-related queries
  system: {
    all: ['system'] as const,
    health: () => [...queryKeys.system.all, 'health'] as const,
    config: () => [...queryKeys.system.all, 'config'] as const,
  },
} as const

// ===== CACHE UTILITIES =====

// Utility to invalidate related queries
export const invalidateQueries = {
  // Invalidate all chat-related data
  allChat: () => queryClient.invalidateQueries({ queryKey: queryKeys.chat.all }),
  
  // Invalidate specific session data
  chatSession: (sessionId: string) => 
    queryClient.invalidateQueries({ queryKey: queryKeys.chat.session(sessionId) }),
  
  // Invalidate messages for a session
  chatMessages: (sessionId: string) => 
    queryClient.invalidateQueries({ queryKey: queryKeys.chat.messages(sessionId) }),
  
  // Invalidate all user data
  allUser: () => queryClient.invalidateQueries({ queryKey: queryKeys.user.all }),
  
  // Invalidate system data
  allSystem: () => queryClient.invalidateQueries({ queryKey: queryKeys.system.all }),
}

// ===== PREFETCHING UTILITIES =====

// Prefetch commonly needed data
export const prefetchQueries = {
  // Prefetch chat sessions
  chatSessions: async () => {
    await queryClient.prefetchQuery({
      queryKey: queryKeys.chat.sessions(),
      queryFn: async () => {
        const response = await fetch('/api/chat/sessions')
        if (!response.ok) throw new Error('Failed to fetch sessions')
        return response.json()
      },
      staleTime: 2 * 60 * 1000, // 2 minutes
    })
  },
  
  // Prefetch user profile
  userProfile: async () => {
    await queryClient.prefetchQuery({
      queryKey: queryKeys.user.profile(),
      queryFn: async () => {
        const response = await fetch('/api/user/profile')
        if (!response.ok) throw new Error('Failed to fetch profile')
        return response.json()
      },
      staleTime: 10 * 60 * 1000, // 10 minutes
    })
  },
}

// ===== CACHE PERSISTENCE =====

// Persist important queries to localStorage
export const persistQueries = {
  // Save critical data to localStorage
  save: () => {
    try {
      const chatSessions = queryClient.getQueryData(queryKeys.chat.sessions())
      const userProfile = queryClient.getQueryData(queryKeys.user.profile())
      
      const persistedData = {
        chatSessions,
        userProfile,
        timestamp: Date.now(),
      }
      
      localStorage.setItem('masterx-cache', JSON.stringify(persistedData))
    } catch (error) {
      console.warn('Failed to persist cache:', error)
    }
  },
  
  // Load critical data from localStorage
  load: () => {
    try {
      const cached = localStorage.getItem('masterx-cache')
      if (!cached) return
      
      const data = JSON.parse(cached)
      const age = Date.now() - data.timestamp
      
      // Only use cached data if it's less than 1 hour old
      if (age > 60 * 60 * 1000) {
        localStorage.removeItem('masterx-cache')
        return
      }
      
      // Restore cached data
      if (data.chatSessions) {
        queryClient.setQueryData(queryKeys.chat.sessions(), data.chatSessions)
      }
      
      if (data.userProfile) {
        queryClient.setQueryData(queryKeys.user.profile(), data.userProfile)
      }
      
      console.log('Cache restored from localStorage')
    } catch (error) {
      console.warn('Failed to load persisted cache:', error)
      localStorage.removeItem('masterx-cache')
    }
  },
  
  // Clear persisted cache
  clear: () => {
    localStorage.removeItem('masterx-cache')
  },
}

// ===== OPTIMISTIC UPDATES =====

// Utilities for optimistic updates
export const optimisticUpdates = {
  // Add message optimistically
  addMessage: (sessionId: string, message: any) => {
    const queryKey = queryKeys.chat.messages(sessionId)
    
    queryClient.setQueryData(queryKey, (old: any) => {
      if (!old) return [message]
      return [...old, message]
    })
  },
  
  // Update message optimistically
  updateMessage: (sessionId: string, messageId: string, updates: any) => {
    const queryKey = queryKeys.chat.messages(sessionId)
    
    queryClient.setQueryData(queryKey, (old: any) => {
      if (!old) return []
      return old.map((msg: any) => 
        msg.id === messageId ? { ...msg, ...updates } : msg
      )
    })
  },
  
  // Remove message optimistically
  removeMessage: (sessionId: string, messageId: string) => {
    const queryKey = queryKeys.chat.messages(sessionId)
    
    queryClient.setQueryData(queryKey, (old: any) => {
      if (!old) return []
      return old.filter((msg: any) => msg.id !== messageId)
    })
  },
}

// ===== BACKGROUND SYNC =====

// Background sync utilities
export const backgroundSync = {
  // Start background sync for active session
  start: (sessionId: string) => {
    const queryKey = queryKeys.chat.messages(sessionId)
    
    // Enable background refetching every 30 seconds
    queryClient.setQueryDefaults(queryKey, {
      refetchInterval: 30 * 1000, // 30 seconds
      refetchIntervalInBackground: true,
    })
  },
  
  // Stop background sync
  stop: (sessionId: string) => {
    const queryKey = queryKeys.chat.messages(sessionId)
    
    queryClient.setQueryDefaults(queryKey, {
      refetchInterval: false,
      refetchIntervalInBackground: false,
    })
  },
}

// ===== DEVELOPMENT UTILITIES =====

// Development-only utilities
export const devUtils = {
  // Log cache contents
  logCache: () => {
    if (process.env.NODE_ENV === 'development') {
      console.group('🗄️ React Query Cache Contents')
      
      const cache = queryClient.getQueryCache()
      const queries = cache.getAll()
      
      queries.forEach((query) => {
        console.log(`${query.queryKey.join(' → ')}:`, {
          data: query.state.data,
          status: query.state.status,
          dataUpdatedAt: new Date(query.state.dataUpdatedAt),
          error: query.state.error,
        })
      })
      
      console.groupEnd()
    }
  },
  
  // Clear all cache
  clearCache: () => {
    queryClient.clear()
    persistQueries.clear()
    console.log('🗑️ Cache cleared')
  },
  
  // Get cache stats
  getCacheStats: () => {
    const cache = queryClient.getQueryCache()
    const queries = cache.getAll()
    
    return {
      totalQueries: queries.length,
      activeQueries: queries.filter(q => q.getObserversCount() > 0).length,
      staleQueries: queries.filter(q => q.isStale()).length,
      errorQueries: queries.filter(q => q.state.status === 'error').length,
    }
  },
}

// Initialize cache persistence on load
if (typeof window !== 'undefined') {
  persistQueries.load()
  
  // Save cache periodically
  setInterval(persistQueries.save, 5 * 60 * 1000) // Every 5 minutes
  
  // Save cache before page unload
  window.addEventListener('beforeunload', persistQueries.save)
}
