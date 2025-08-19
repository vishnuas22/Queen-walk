// Optimized selectors for MasterX store

import { useMemo } from 'react'
import type { RootState, Message, ChatSession, Notification } from './types'
import { useStore } from './index'

// ===== MEMOIZED SELECTORS =====

// Chat selectors
export const useChatSelectors = () => {
  const store = useStore()
  
  return useMemo(() => ({
    // Current session
    currentSession: store.chat.sessions.find(s => s.session_id === store.chat.currentSessionId),
    
    // Message counts
    totalMessages: store.chat.currentMessages.length,
    userMessages: store.chat.currentMessages.filter(m => m.sender === 'user').length,
    aiMessages: store.chat.currentMessages.filter(m => m.sender === 'ai').length,
    
    // Recent messages (last 10)
    recentMessages: store.chat.currentMessages.slice(-10),
    
    // Streaming state
    isStreaming: store.chat.isStreaming,
    streamingMessage: store.chat.currentMessages.find(m => m.id === store.chat.streamingMessageId),
    
    // Input state
    hasInput: store.chat.inputMessage.trim().length > 0,
    canSend: store.chat.inputMessage.trim().length > 0 && !store.chat.isLoading,
    
    // Session stats
    sessionStats: {
      totalSessions: store.chat.sessions.length,
      activeSessions: store.chat.sessions.filter(s => s.message_count > 0).length,
      recentSessions: store.chat.sessions.slice(0, 5),
    },
  }), [
    store.chat.currentSessionId,
    store.chat.sessions,
    store.chat.currentMessages,
    store.chat.isStreaming,
    store.chat.streamingMessageId,
    store.chat.inputMessage,
    store.chat.isLoading,
  ])
}

// UI selectors
export const useUISelectors = () => {
  const store = useStore()
  
  return useMemo(() => ({
    // Layout
    layout: {
      sidebarOpen: store.ui.sidebarOpen,
      sidebarWidth: store.ui.sidebarWidth,
      isMobile: store.ui.isMobile,
      screenSize: store.ui.screenSize,
    },
    
    // Loading states
    loading: {
      global: store.ui.globalLoading,
      states: store.ui.loadingStates,
      isLoading: store.ui.globalLoading || Object.keys(store.ui.loadingStates).length > 0,
      activeStates: Object.keys(store.ui.loadingStates).filter(key => store.ui.loadingStates[key]),
    },
    
    // Notifications
    notifications: {
      all: store.ui.notifications,
      unread: store.ui.notifications.filter(n => !n.persistent),
      persistent: store.ui.notifications.filter(n => n.persistent),
      byType: {
        info: store.ui.notifications.filter(n => n.type === 'info'),
        success: store.ui.notifications.filter(n => n.type === 'success'),
        warning: store.ui.notifications.filter(n => n.type === 'warning'),
        error: store.ui.notifications.filter(n => n.type === 'error'),
      },
    },
    
    // Modal state
    modal: {
      active: store.ui.activeModal,
      data: store.ui.modalData,
      isOpen: !!store.ui.activeModal,
    },
    
    // Accessibility
    accessibility: {
      keyboardNavigation: store.ui.keyboardNavigation,
      focusedElement: store.ui.focusedElement,
      voiceListening: store.ui.voiceListening,
      voiceEnabled: store.ui.voiceEnabled,
    },
    
    // Performance
    performance: {
      mode: store.ui.performanceMode,
      debug: store.ui.debugMode,
    },
    
    // Theme
    theme: {
      current: store.ui.theme,
      isDark: store.ui.theme === 'dark',
      isLight: store.ui.theme === 'light',
      isAuto: store.ui.theme === 'auto',
    },
  }), [
    store.ui.sidebarOpen,
    store.ui.sidebarWidth,
    store.ui.isMobile,
    store.ui.screenSize,
    store.ui.globalLoading,
    store.ui.loadingStates,
    store.ui.notifications,
    store.ui.activeModal,
    store.ui.modalData,
    store.ui.keyboardNavigation,
    store.ui.focusedElement,
    store.ui.voiceListening,
    store.ui.voiceEnabled,
    store.ui.performanceMode,
    store.ui.debugMode,
    store.ui.theme,
  ])
}

// User selectors
export const useUserSelectors = () => {
  const store = useStore()
  
  return useMemo(() => ({
    // Authentication
    auth: {
      isAuthenticated: store.user.isAuthenticated,
      user: store.user.user,
      loading: store.user.userLoading,
      error: store.user.userError,
    },
    
    // Preferences
    preferences: {
      all: store.user.preferences,
      theme: store.user.preferences.theme,
      language: store.user.preferences.language,
      accessibility: store.user.preferences.accessibility,
      chat: store.user.preferences.chat,
      performance: store.user.preferences.performance,
    },
    
    // Session
    session: {
      expiry: store.user.sessionExpiry,
      lastActivity: store.user.lastActivity,
      isExpired: store.user.sessionExpiry ? new Date() > store.user.sessionExpiry : false,
      timeUntilExpiry: store.user.sessionExpiry ? 
        Math.max(0, store.user.sessionExpiry.getTime() - Date.now()) : 0,
    },
    
    // Profile
    profile: {
      name: store.user.user?.name,
      email: store.user.user?.email,
      avatar: store.user.user?.avatar,
      subscription: store.user.user?.subscription,
    },
  }), [
    store.user.isAuthenticated,
    store.user.user,
    store.user.userLoading,
    store.user.userError,
    store.user.preferences,
    store.user.sessionExpiry,
    store.user.lastActivity,
  ])
}

// App selectors
export const useAppSelectors = () => {
  const store = useStore()
  
  return useMemo(() => ({
    // App info
    info: {
      version: store.app.version,
      buildTime: store.app.buildTime,
      environment: store.app.environment,
      uptime: Date.now() - store.app.analytics.sessionStart.getTime(),
    },
    
    // Connection
    connection: {
      isOnline: store.app.isOnline,
      quality: store.app.connectionQuality,
      status: store.app.isOnline ? 'connected' : 'disconnected',
    },
    
    // Performance
    performance: {
      metrics: store.app.performanceMetrics,
      score: Math.round(
        (100 - Math.min(store.app.performanceMetrics.loadTime / 50, 100)) * 0.4 +
        (100 - Math.min(store.app.performanceMetrics.renderTime / 20, 100)) * 0.3 +
        (100 - Math.min(store.app.performanceMetrics.memoryUsage / 2, 100)) * 0.3
      ),
    },
    
    // Feature flags
    features: {
      flags: store.app.featureFlags,
      experimental: store.app.featureFlags.experimentalFeatures,
      beta: store.app.featureFlags.betaFeatures,
      debug: store.app.featureFlags.debugMode,
    },
    
    // Errors
    errors: {
      all: store.app.errors,
      unresolved: store.app.errors.filter(e => !e.resolved),
      critical: store.app.errors.filter(e => e.severity === 'critical' && !e.resolved),
      recent: store.app.errors.slice(-5),
    },
    
    // Analytics
    analytics: {
      session: store.app.analytics,
      errorRate: store.app.analytics.errors / Math.max(store.app.analytics.interactions, 1),
      interactionRate: store.app.analytics.interactions / Math.max(store.app.analytics.pageViews, 1),
    },
  }), [
    store.app.version,
    store.app.buildTime,
    store.app.environment,
    store.app.analytics.sessionStart,
    store.app.isOnline,
    store.app.connectionQuality,
    store.app.performanceMetrics,
    store.app.featureFlags,
    store.app.errors,
    store.app.analytics,
  ])
}

// ===== COMPUTED SELECTORS =====

export const useComputedSelectors = () => {
  const chat = useChatSelectors()
  const ui = useUISelectors()
  const user = useUserSelectors()
  const app = useAppSelectors()
  
  return useMemo(() => ({
    // Overall app state
    appState: {
      isReady: !ui.loading.global && app.connection.isOnline,
      hasErrors: app.errors.critical.length > 0,
      needsAttention: app.errors.unresolved.length > 0 || user.session.isExpired,
    },
    
    // Chat state
    chatState: {
      isEmpty: chat.totalMessages === 0,
      isActive: chat.totalMessages > 0 && !chat.isStreaming,
      canInteract: chat.canSend && app.connection.isOnline,
      hasHistory: chat.sessionStats.totalSessions > 0,
    },
    
    // User experience
    userExperience: {
      isOptimal: app.performance.score > 80 && app.connection.quality === 'excellent',
      needsOptimization: app.performance.score < 60 || app.connection.quality === 'poor',
      accessibilityEnabled: Object.values(user.preferences.accessibility).some(Boolean),
    },
    
    // System health
    systemHealth: {
      score: Math.round(
        (app.connection.isOnline ? 25 : 0) +
        (app.performance.score * 0.5) +
        (app.errors.critical.length === 0 ? 25 : 0)
      ),
      status: app.errors.critical.length > 0 ? 'critical' :
              app.performance.score < 60 ? 'degraded' : 'healthy',
    },
  }), [chat, ui, user, app])
}

// ===== PERFORMANCE SELECTORS =====

export const usePerformanceSelectors = () => {
  const store = useStore()
  
  return useMemo(() => {
    const metrics = store.app.performanceMetrics
    const messageCount = store.chat.currentMessages.length
    const notificationCount = store.ui.notifications.length
    
    return {
      // Memory usage
      memory: {
        current: metrics.memoryUsage,
        estimated: messageCount * 0.5 + notificationCount * 0.1, // KB estimate
        threshold: 100, // 100MB threshold
        isHigh: metrics.memoryUsage > 100,
      },
      
      // Render performance
      render: {
        time: metrics.renderTime,
        fps: metrics.renderTime > 0 ? Math.round(1000 / metrics.renderTime) : 60,
        isSmooth: metrics.renderTime < 16.67, // 60fps threshold
      },
      
      // Load performance
      load: {
        time: metrics.loadTime,
        isFast: metrics.loadTime < 1000,
        isSlow: metrics.loadTime > 3000,
      },
      
      // Cache performance
      cache: {
        hitRate: metrics.cacheHitRate,
        isEfficient: metrics.cacheHitRate > 0.8,
      },
      
      // Overall score
      score: Math.round(
        (metrics.loadTime < 1000 ? 25 : Math.max(0, 25 - (metrics.loadTime - 1000) / 100)) +
        (metrics.renderTime < 16.67 ? 25 : Math.max(0, 25 - (metrics.renderTime - 16.67) * 2)) +
        (metrics.memoryUsage < 50 ? 25 : Math.max(0, 25 - (metrics.memoryUsage - 50) / 2)) +
        (metrics.cacheHitRate * 25)
      ),
    }
  }, [
    store.app.performanceMetrics,
    store.chat.currentMessages.length,
    store.ui.notifications.length,
  ])
}

// ===== EXPORT HOOKS =====

export const useOptimizedSelectors = () => ({
  chat: useChatSelectors(),
  ui: useUISelectors(),
  user: useUserSelectors(),
  app: useAppSelectors(),
  computed: useComputedSelectors(),
  performance: usePerformanceSelectors(),
})

export default useOptimizedSelectors
