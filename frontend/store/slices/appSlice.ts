// App slice for Zustand store

import type { StateCreator } from 'zustand'
import type { Store, AppState, AppActions, AppError } from '../types'

// ===== INITIAL STATE =====

const initialAppState: AppState = {
  // App metadata
  version: '2.0.0',
  buildTime: new Date().toISOString(),
  environment: process.env.NODE_ENV as 'development' | 'staging' | 'production',
  
  // Connection state
  isOnline: true,
  connectionQuality: 'excellent',
  
  // Performance metrics
  performanceMetrics: {
    loadTime: 0,
    renderTime: 0,
    memoryUsage: 0,
    cacheHitRate: 0,
  },
  
  // Feature flags
  featureFlags: {
    voiceNavigation: true,
    advancedAnalytics: true,
    experimentalFeatures: false,
    betaFeatures: false,
    debugMode: process.env.NODE_ENV === 'development',
  },
  
  // Error tracking
  errors: [],
  
  // Analytics
  analytics: {
    sessionStart: new Date(),
    pageViews: 0,
    interactions: 0,
    errors: 0,
  },
}

// ===== APP SLICE CREATOR =====

export const createAppSlice: StateCreator<
  Store,
  [['zustand/immer', never], ['zustand/devtools', never], ['zustand/persist', unknown], ['zustand/subscribeWithSelector', never]],
  [],
  AppState & AppActions
> = (set, get, api) => ({
  // Initial state - properly namespaced under 'app'
  app: initialAppState,

  // ===== CONNECTION ACTIONS =====

  setOnlineStatus: (online: boolean) => {
    set((state) => {
      state.app.isOnline = online
    })

    // Update connection quality based on online status
    if (online) {
      get().updateConnectionQuality('excellent')
      
      // Notify when back online
      const { addNotification } = get()
      addNotification({
        type: 'success',
        title: 'Back Online',
        message: 'Connection restored. Syncing data...',
        duration: 3000,
      })
    } else {
      get().updateConnectionQuality('poor')
      
      // Notify when offline
      const { addNotification } = get()
      addNotification({
        type: 'warning',
        title: 'Connection Lost',
        message: 'You are now offline. Some features may be limited.',
        duration: 5000,
        persistent: true,
      })
    }
  },

  updateConnectionQuality: (quality: 'poor' | 'good' | 'excellent') => {
    set((state) => {
      state.app.connectionQuality = quality
    })

    // Adjust app behavior based on connection quality
    if (quality === 'poor') {
      // Enable performance mode for poor connections
      const { setPerformanceMode } = get()
      setPerformanceMode(true)
    }
  },

  // ===== PERFORMANCE ACTIONS =====

  updatePerformanceMetrics: (metrics: Partial<AppState['performanceMetrics']>) => {
    set((state) => {
      state.app.performanceMetrics = { ...state.app.performanceMetrics, ...metrics }
    })

    // Log performance metrics in development
    if (process.env.NODE_ENV === 'development') {
      console.log('📊 Performance Metrics Updated:', metrics)
    }

    // Check for performance issues
    const currentMetrics = get().app.performanceMetrics
    
    if (currentMetrics.loadTime > 3000) {
      const { addNotification } = get()
      addNotification({
        type: 'warning',
        title: 'Slow Loading',
        message: 'The app is loading slowly. Consider enabling performance mode.',
        duration: 5000,
        actions: [
          {
            label: 'Enable Performance Mode',
            action: () => get().setPerformanceMode(true),
            style: 'primary',
          },
        ],
      })
    }

    if (currentMetrics.memoryUsage > 100) { // 100MB threshold
      console.warn('High memory usage detected:', currentMetrics.memoryUsage, 'MB')
    }
  },

  // ===== FEATURE FLAG ACTIONS =====

  setFeatureFlag: (flag: string, enabled: boolean) => {
    set((state) => {
      state.app.featureFlags[flag] = enabled
    })

    // Log feature flag changes
    console.log(`🚩 Feature flag '${flag}' ${enabled ? 'enabled' : 'disabled'}`)

    // Notify about feature flag changes
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title: 'Feature Updated',
      message: `${flag} has been ${enabled ? 'enabled' : 'disabled'}`,
      duration: 3000,
    })
  },

  // ===== ERROR HANDLING ACTIONS =====

  addError: (error: Omit<AppError, 'id' | 'timestamp'>) => {
    const newError: AppError = {
      ...error,
      id: `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
    }

    set((state) => {
      state.app.errors.push(newError)
      state.app.analytics.errors++
      
      // Limit error history to 100 entries
      if (state.app.errors.length > 100) {
        state.app.errors = state.app.errors.slice(-100)
      }
    })

    // Log error
    console.error('🚨 Application Error:', newError)

    // Show notification for high severity errors
    if (error.severity === 'high' || error.severity === 'critical') {
      const { addNotification } = get()
      addNotification({
        type: 'error',
        title: 'Application Error',
        message: error.message,
        duration: error.severity === 'critical' ? 0 : 10000, // Critical errors persist
        persistent: error.severity === 'critical',
        actions: error.severity === 'critical' ? [
          {
            label: 'Reload App',
            action: () => window.location.reload(),
            style: 'primary',
          },
        ] : undefined,
      })
    }

    // Auto-report critical errors
    if (error.severity === 'critical') {
      get().reportError(newError)
    }
  },

  resolveError: (errorId: string) => {
    set((state) => {
      const error = state.app.errors.find(e => e.id === errorId)
      if (error) {
        error.resolved = true
      }
    })
  },

  clearErrors: () => {
    set((state) => {
      state.app.errors = []
    })

    // Notify about error clearing
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title: 'Errors Cleared',
      message: 'All application errors have been cleared',
      duration: 2000,
    })
  },

  // ===== ANALYTICS ACTIONS =====

  trackPageView: () => {
    try {
      set((state) => {
        // Ensure analytics structure exists before accessing
        if (!state || !state.app) {
          console.warn('trackPageView: app state not initialized')
          return
        }
        if (!state.app.analytics) {
          console.warn('trackPageView: analytics state not initialized')
          return
        }
        state.app.analytics.pageViews++
      })

      // Log page view in development
      if (process.env.NODE_ENV === 'development') {
        console.log('📄 Page view tracked')
      }
    } catch (error) {
      console.warn('trackPageView failed:', error)
      // Silently fail to prevent app crashes
    }
  },

  trackInteraction: (type: string, data?: any) => {
    try {
      set((state) => {
        // Ensure analytics structure exists before accessing
        if (!state || !state.app || !state.app.analytics) {
          console.warn('trackInteraction: analytics state not initialized')
          return
        }
        state.app.analytics.interactions++
      })
    } catch (error) {
      console.warn('trackInteraction failed:', error)
      // Silently fail to prevent app crashes
      return
    }

    // Log interaction in development
    if (process.env.NODE_ENV === 'development') {
      console.log('👆 Interaction tracked:', type, data)
    }

    // Track specific interaction types
    if (type === 'message_sent') {
      // Track message sending
      console.log('💬 Message sent interaction tracked')
    } else if (type === 'voice_command') {
      // Track voice command usage
      console.log('🎤 Voice command interaction tracked')
    }
  },

  trackError: (error: Error) => {
    try {
      const { addError } = get()
      if (typeof addError === 'function') {
        addError({
          message: error.message,
          stack: error.stack,
          severity: 'medium',
          context: {
            userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
            url: typeof window !== 'undefined' ? window.location.href : 'unknown',
            timestamp: new Date().toISOString(),
          },
        })
      }
    } catch (trackError) {
      console.warn('trackError failed:', trackError)
      // Silently fail to prevent app crashes
    }
  },

  // ===== UTILITY METHODS =====

  getAppInfo: () => {
    const fullState = get()
    if (!fullState.app) {
      console.warn('getAppInfo: app state not initialized')
      return {
        version: '0.0.0',
        buildTime: new Date(),
        environment: 'development',
        uptime: 0,
        isOnline: true,
        connectionQuality: 'good',
      }
    }
    const state = fullState.app
    return {
      version: state.version,
      buildTime: state.buildTime,
      environment: state.environment,
      uptime: state.analytics?.sessionStart ? Date.now() - state.analytics.sessionStart.getTime() : 0,
      isOnline: state.isOnline,
      connectionQuality: state.connectionQuality,
    }
  },

  getPerformanceReport: () => {
    const state = get().app
    const uptime = Date.now() - state.analytics.sessionStart.getTime()
    
    return {
      ...state.performanceMetrics,
      uptime,
      pageViews: state.analytics.pageViews,
      interactions: state.analytics.interactions,
      errors: state.analytics.errors,
      errorRate: state.analytics.errors / Math.max(state.analytics.interactions, 1),
    }
  },

  reportError: async (error: AppError) => {
    try {
      // Send error to monitoring service
      await fetch('/api/errors/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(error),
      })
      
      console.log('📤 Error reported to monitoring service')
    } catch (reportError) {
      console.error('Failed to report error:', reportError)
    }
  },

  checkHealth: async () => {
    try {
      const response = await fetch('/api/health')
      const health = await response.json()
      
      // Update connection quality based on health check
      if (health.status === 'healthy') {
        get().updateConnectionQuality('excellent')
      } else if (health.status === 'degraded') {
        get().updateConnectionQuality('good')
      } else {
        get().updateConnectionQuality('poor')
      }
      
      return health
    } catch (error) {
      get().updateConnectionQuality('poor')
      throw error
    }
  },

  measurePerformance: (name: string, fn: () => void | Promise<void>) => {
    const start = performance.now()
    
    const finish = () => {
      const end = performance.now()
      const duration = end - start
      
      console.log(`⏱️ ${name} took ${duration.toFixed(2)}ms`)
      
      // Update performance metrics
      if (name === 'render') {
        get().updatePerformanceMetrics({ renderTime: duration })
      } else if (name === 'load') {
        get().updatePerformanceMetrics({ loadTime: duration })
      }
    }
    
    const result = fn()
    
    if (result instanceof Promise) {
      return result.finally(finish)
    } else {
      finish()
      return result
    }
  },

  enableExperimentalFeatures: () => {
    set((state) => {
      state.app.featureFlags.experimentalFeatures = true
      state.app.featureFlags.betaFeatures = true
    })

    // Notify about experimental features
    const { addNotification } = get()
    addNotification({
      type: 'warning',
      title: 'Experimental Features Enabled',
      message: 'You have enabled experimental features. Some functionality may be unstable.',
      duration: 10000,
      persistent: true,
    })
  },

  disableExperimentalFeatures: () => {
    set((state) => {
      state.app.featureFlags.experimentalFeatures = false
      state.app.featureFlags.betaFeatures = false
    })

    // Notify about disabling experimental features
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title: 'Experimental Features Disabled',
      message: 'Experimental features have been disabled for stability.',
      duration: 3000,
    })
  },
})
