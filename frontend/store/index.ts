// Main Zustand store for MasterX global state management

import { create } from 'zustand'
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import type { Store, RootState, PersistConfig } from './types'
import { createChatSlice } from './slices/chatSlice'
import { createUISlice } from './slices/uiSlice'
import { createUserSlice } from './slices/userSlice'
import { createAppSlice } from './slices/appSlice'

// ===== STORE CONFIGURATION =====

const persistConfig: PersistConfig = {
  name: 'masterx-store-v2',
  version: 2,
  storage: 'localStorage',
  whitelist: ['user', 'ui'],
  blacklist: ['chat', 'app'],
  migrate: (persistedState: any, version: number) => {
    console.log(`🔄 Migrating store from version ${version} to 2`)

    // Handle store migrations between versions
    if (version === 0) {
      // Migration from version 0 to 1
      return {
        ...persistedState,
        ui: {
          ...persistedState.ui,
          theme: persistedState.ui.theme || 'auto',
        },
      }
    }

    if (version === 1) {
      // Migration from version 1 to 2
      return {
        ...persistedState,
        ui: {
          ...persistedState.ui,
          notifications: [], // Reset notifications on migration
          performanceMode: false,
          debugMode: process.env.NODE_ENV === 'development',
        },
        user: {
          ...persistedState.user,
          preferences: {
            ...persistedState.user?.preferences,
            accessibility: {
              ...persistedState.user?.preferences?.accessibility,
              voiceNavigation: true, // Enable voice navigation by default
            },
          },
        },
      }
    }

    return persistedState
  },
}

// ===== STORE CREATION =====

export const useStore = create<Store>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get, api) => ({
          // Combine all slices
          ...createChatSlice(set, get, api),
          ...createUISlice(set, get, api),
          ...createUserSlice(set, get, api),
          ...createAppSlice(set, get, api),
        }))
      ),
      {
        name: persistConfig.name,
        version: persistConfig.version,
        partialize: (state) => {
          // Only persist whitelisted slices
          const persistedState: Partial<RootState> = {}

          if (persistConfig.whitelist?.includes('user')) {
            persistedState.user = state.user
          }
          if (persistConfig.whitelist?.includes('ui')) {
            persistedState.ui = {
              ...state.ui,
              // Don't persist temporary UI state
              globalLoading: false,
              loadingStates: {},
              notifications: [],
              activeModal: null,
              modalData: null,
            }
          }

          return persistedState
        },
        migrate: persistConfig.migrate,
      }
    ),
    {
      name: 'MasterX Store',
      enabled: process.env.NODE_ENV === 'development',
    }
  )
)

// ===== STORE UTILITIES =====

// Safe selector hooks with initialization check
export const useChatState = () => {
  try {
    const state = useStore.getState()
    if (!state || !state.chat) {
      console.warn('Chat state not ready, returning default state')
      return {
        currentSessionId: null,
        sessions: [],
        isLoading: false,
        inputMessage: '',
        sidebarOpen: false,
        isStreaming: false,
        streamingMessageId: null,
        inputHistory: [],
        inputHistoryIndex: -1,
      }
    }
    return useStore((state) => state.chat)
  } catch (error) {
    console.warn('Chat state not ready, returning default state')
    return {
      currentSessionId: null,
      currentMessages: [],
      sessions: [],
      isLoading: false,
      inputMessage: '',
      sidebarOpen: false,
      isStreaming: false,
      streamingMessageId: null,
      inputHistory: [],
      inputHistoryIndex: -1,
    }
  }
}

export const useUIState = () => {
  try {
    const state = useStore.getState()
    if (!state || !state.ui) {
      console.warn('UI state not ready, returning default state')
      return {
        sidebarOpen: false,
        activeModal: null,
        modalData: null,
        globalLoading: false,
        loadingStates: {},
        notifications: [],
        keyboardNavigation: false,
        focusedElement: null,
        performanceMode: false,
        debugMode: false,
        voiceListening: false,
        voiceEnabled: true,
        theme: 'auto' as const,
        isMobile: false,
        screenSize: 'lg' as const,
      }
    }
    return useStore((state) => state.ui)
  } catch (error) {
    console.warn('UI state not ready, returning default state')
    return {
      sidebarOpen: false,
      sidebarWidth: 320,
      activeModal: null,
      modalData: null,
      globalLoading: false,
      loadingStates: {},
      notifications: [],
      keyboardNavigation: false,
      focusedElement: null,
      performanceMode: false,
      debugMode: false,
      voiceListening: false,
      voiceEnabled: true,
      theme: 'auto' as const,
      isMobile: false,
      screenSize: 'lg' as const,
    }
  }
}

export const useUserState = () => {
  try {
    return useStore((state) => state.user)
  } catch (error) {
    console.warn('User state not ready, returning default state')
    return {
      isAuthenticated: false,
      user: null,
      userLoading: false,
      userError: null,
      sessionExpiry: null,
      lastActivity: new Date(),
      preferences: {
        theme: 'auto' as const,
        language: 'en',
        accessibility: {
          reducedMotion: false,
          highContrast: false,
          largeText: false,
          screenReaderMode: false,
          voiceNavigation: true,
        },
        chat: {
          voiceInput: true,
          autoScroll: true,
          showTimestamps: true,
          compactMode: false,
        },
        performance: {
          enableAnimations: true,
          enableVirtualization: true,
          enableCaching: true,
        },
      },
    }
  }
}

export const useAppState = () => {
  try {
    return useStore((state) => state.app)
  } catch (error) {
    console.warn('App state not ready, returning default state')
    return {
      version: '2.0.0',
      buildTime: new Date().toISOString(),
      environment: 'development' as const,
      isOnline: true,
      connectionQuality: 'excellent' as const,
      performanceMetrics: {
        loadTime: 0,
        renderTime: 0,
        memoryUsage: 0,
        cacheHitRate: 0,
      },
      featureFlags: {
        voiceNavigation: true,
        advancedAnalytics: true,
        experimentalFeatures: false,
        betaFeatures: false,
        debugMode: false,
      },
      errors: [],
      analytics: {
        sessionStart: new Date(),
        pageViews: 0,
        interactions: 0,
        errors: 0,
      },
    }
  }
}

// Safe action hooks with error handling
export const useChatActions = () => {
  try {
    return useStore((state) => ({
      setCurrentSession: state.setCurrentSession,
      createSession: state.createSession,
      deleteSession: state.deleteSession,
      updateSessionTitle: state.updateSessionTitle,
      addMessage: state.addMessage,
      updateMessage: state.updateMessage,
      deleteMessage: state.deleteMessage,
      clearMessages: state.clearMessages,
      setInputMessage: state.setInputMessage,
      addToInputHistory: state.addToInputHistory,
      navigateInputHistory: state.navigateInputHistory,
      sendMessage: state.sendMessage,
      resendMessage: state.resendMessage,
      toggleSidebar: state.toggleSidebar,
      setSidebarOpen: state.setSidebarOpen,
      selectMessage: state.selectMessage,
      setLoading: state.setLoading,
      setStreaming: state.setStreaming,
    }))
  } catch (error) {
    console.warn('Chat actions not ready, returning no-op functions')
    return {
      setCurrentSession: () => {},
      createSession: () => {},
      deleteSession: () => {},
      updateSessionTitle: () => {},
      addMessage: () => {},
      updateMessage: () => {},
      deleteMessage: () => {},
      clearMessages: () => {},
      setInputMessage: () => {},
      addToInputHistory: () => {},
      navigateInputHistory: () => {},
      sendMessage: () => {},
      resendMessage: () => {},
      toggleSidebar: () => {},
      setSidebarOpen: () => {},
      selectMessage: () => {},
      setLoading: () => {},
      setStreaming: () => {},
    }
  }
}

export const useUIActions = () => {
  try {
    return useStore((state) => ({
      toggleSidebar: state.toggleSidebar,
      setSidebarWidth: state.setSidebarWidth,
      openModal: state.openModal,
      closeModal: state.closeModal,
      setGlobalLoading: state.setGlobalLoading,
      setLoadingState: state.setLoadingState,
      addNotification: state.addNotification,
      removeNotification: state.removeNotification,
      clearNotifications: state.clearNotifications,
      setKeyboardNavigation: state.setKeyboardNavigation,
      setFocusedElement: state.setFocusedElement,
      setPerformanceMode: state.setPerformanceMode,
      setDebugMode: state.setDebugMode,
      setVoiceListening: state.setVoiceListening,
      setVoiceEnabled: state.setVoiceEnabled,
      setTheme: state.setTheme,
      setScreenSize: state.setScreenSize,
      setIsMobile: state.setIsMobile,
    }))
  } catch (error) {
    console.warn('UI actions not ready, returning no-op functions')
    return {
      toggleSidebar: () => {},
      setSidebarWidth: () => {},
      openModal: () => {},
      closeModal: () => {},
      setGlobalLoading: () => {},
      setLoadingState: () => {},
      addNotification: () => {},
      removeNotification: () => {},
      clearNotifications: () => {},
      setKeyboardNavigation: () => {},
      setFocusedElement: () => {},
      setPerformanceMode: () => {},
      setDebugMode: () => {},
      setVoiceListening: () => {},
      setVoiceEnabled: () => {},
      setTheme: () => {},
      setScreenSize: () => {},
      setIsMobile: () => {},
    }
  }
}

export const useUserActions = () => {
  try {
    return useStore((state) => ({
      login: state.login,
      logout: state.logout,
      refreshToken: state.refreshToken,
      updateUser: state.updateUser,
      updatePreferences: state.updatePreferences,
      updateLastActivity: state.updateLastActivity,
      checkSessionExpiry: state.checkSessionExpiry,
    }))
  } catch (error) {
    console.warn('User actions not ready, returning no-op functions')
    return {
      login: () => {},
      logout: () => {},
      refreshToken: () => {},
      updateUser: () => {},
      updatePreferences: () => {},
      updateLastActivity: () => {},
      checkSessionExpiry: () => {},
    }
  }
}

export const useAppActions = () => {
  try {
    return useStore((state) => ({
      setOnlineStatus: state.setOnlineStatus,
      updateConnectionQuality: state.updateConnectionQuality,
      updatePerformanceMetrics: state.updatePerformanceMetrics,
      setFeatureFlag: state.setFeatureFlag,
      addError: state.addError,
      resolveError: state.resolveError,
      clearErrors: state.clearErrors,
      trackPageView: state.trackPageView,
      trackInteraction: state.trackInteraction,
      trackError: state.trackError,
    }))
  } catch (error) {
    console.warn('App actions not ready, returning no-op functions')
    return {
      setOnlineStatus: () => {},
      updateConnectionQuality: () => {},
      updatePerformanceMetrics: () => {},
      setFeatureFlag: () => {},
      addError: () => {},
      resolveError: () => {},
      clearErrors: () => {},
      trackPageView: () => {},
      trackInteraction: () => {},
      trackError: () => {},
    }
  }
}

// ===== COMPUTED SELECTORS =====

// Chat selectors
export const useCurrentSession = () => useStore((state) => 
  state.chat.sessions.find(s => s.session_id === state.chat.currentSessionId)
)

export const useCurrentMessages = () => useStore((state) => state.chat.currentMessages)

export const useIsLoading = () => useStore((state) => 
  state.chat.isLoading || state.ui.globalLoading
)

export const useHasMessages = () => useStore((state) => 
  state.chat.currentMessages.length > 0
)

// UI selectors
export const useIsSidebarOpen = () => useStore((state) => state.ui.sidebarOpen)

export const useActiveModal = () => useStore((state) => ({
  modal: state.ui.activeModal,
  data: state.ui.modalData,
}))

export const useNotifications = () => {
  // Use a safer approach with conditional rendering
  return useStore((state) => {
    // Return empty array if store is not ready
    if (!state || !state.ui || !Array.isArray(state.ui.notifications)) {
      return []
    }
    return state.ui.notifications
  }, (a, b) => {
    // Custom equality function to prevent unnecessary re-renders
    if (!Array.isArray(a) || !Array.isArray(b)) return a === b
    return a.length === b.length && a.every((item, index) => item.id === b[index]?.id)
  })
}

export const useTheme = () => {
  return useStore((state) => {
    if (!state || !state.ui) {
      return 'auto'
    }
    return state.ui.theme || 'auto'
  })
}

// User selectors
export const useIsAuthenticated = () => {
  return useStore((state) => {
    if (!state || !state.user) {
      return false
    }
    return state.user.isAuthenticated || false
  })
}

export const useUserPreferences = () => useStore((state) => state.user.preferences)

export const useAccessibilitySettings = () => useStore((state) => 
  state.user.preferences.accessibility
)

// App selectors
export const useIsOnline = () => useStore((state) => state.app.isOnline)

export const usePerformanceMetrics = () => useStore((state) => state.app.performanceMetrics)

export const useFeatureFlags = () => useStore((state) => state.app.featureFlags)

// ===== STORE SUBSCRIPTIONS =====

// Subscribe to store changes for side effects
export const subscribeToStore = () => {
  // Subscribe to theme changes
  useStore.subscribe(
    (state) => state.ui.theme,
    (theme) => {
      if (typeof window !== 'undefined') {
        document.documentElement.setAttribute('data-theme', theme)
      }
    },
    { fireImmediately: true }
  )

  // Subscribe to accessibility changes
  useStore.subscribe(
    (state) => state.user.preferences.accessibility,
    (accessibility) => {
      if (typeof window !== 'undefined') {
        const body = document.body
        
        // Apply accessibility classes
        body.classList.toggle('reduce-motion', accessibility.reducedMotion)
        body.classList.toggle('high-contrast', accessibility.highContrast)
        body.classList.toggle('large-text', accessibility.largeText)
        body.classList.toggle('screen-reader-mode', accessibility.screenReaderMode)
      }
    },
    { fireImmediately: true }
  )

  // Subscribe to online status
  useStore.subscribe(
    (state) => state.app.isOnline,
    (isOnline) => {
      console.log(`🌐 Connection status: ${isOnline ? 'Online' : 'Offline'}`)
    }
  )

  // Subscribe to errors for logging
  useStore.subscribe(
    (state) => state.app.errors,
    (errors) => {
      const newErrors = errors.filter(error => !error.resolved)
      if (newErrors.length > 0) {
        console.error('🚨 New application errors:', newErrors)
      }
    }
  )
}

// ===== STORE INITIALIZATION =====

let isStoreInitialized = false

export const initializeStore = () => {
  if (isStoreInitialized) {
    console.log('🏪 Store already initialized')
    return
  }

  try {
    const store = useStore.getState()

    // Verify store structure with more detailed logging
    if (!store) {
      console.warn('⚠️ Store is null or undefined, will retry initialization')
      return
    }

    // Check individual store slices (warn instead of error)
    const missingSlices = []
    if (!store.ui) missingSlices.push('ui')
    if (!store.app) missingSlices.push('app')
    if (!store.chat) missingSlices.push('chat')
    if (!store.user) missingSlices.push('user')

    if (missingSlices.length > 0) {
      console.error('❌ Store structure is invalid')
      console.warn(`⚠️ Store slices not ready: ${missingSlices.join(', ')}`)
      console.log('Store keys:', Object.keys(store))
      // Return early to prevent further errors
      return
    }

    // Verify app slice structure
    if (store.app && !store.app.analytics) {
      console.warn('⚠️ App analytics not initialized, skipping analytics setup')
    }

    // Verify required methods exist (with safe fallbacks)
    const missingMethods = []
    if (typeof store.setOnlineStatus !== 'function') missingMethods.push('setOnlineStatus')
    if (typeof store.updateLastActivity !== 'function') missingMethods.push('updateLastActivity')

    if (missingMethods.length > 0) {
      console.warn(`⚠️ Store methods not ready: ${missingMethods.join(', ')}`)
    }

    // Initialize app state (with safe checks)
    try {
      if (typeof store.setOnlineStatus === 'function') {
        store.setOnlineStatus(navigator.onLine)
      }
      if (typeof store.updateLastActivity === 'function') {
        store.updateLastActivity()
      }
    } catch (error) {
      console.warn('⚠️ Failed to initialize app state:', error)
    }

    // Set up online/offline listeners
    if (typeof window !== 'undefined') {
      window.addEventListener('online', () => store.setOnlineStatus(true))
      window.addEventListener('offline', () => store.setOnlineStatus(false))

      // Set up responsive listeners
      const updateScreenSize = () => {
        const width = window.innerWidth
        if (width < 640) store.setScreenSize('sm')
        else if (width < 768) store.setScreenSize('md')
        else if (width < 1024) store.setScreenSize('lg')
        else store.setScreenSize('xl')

        store.setIsMobile(width < 768)
      }

      updateScreenSize()
      window.addEventListener('resize', updateScreenSize)

      // Set up activity tracking
      const updateActivity = () => store.updateLastActivity()
      document.addEventListener('mousedown', updateActivity)
      document.addEventListener('keydown', updateActivity)
      document.addEventListener('scroll', updateActivity)
    }

    // Subscribe to store changes
    subscribeToStore()

    isStoreInitialized = true
    console.log('🏪 MasterX Store initialized')
  } catch (error) {
    console.error('❌ Store initialization failed:', error)
  }
}

export const isStoreReady = () => isStoreInitialized

// ===== DEVELOPMENT UTILITIES =====

if (process.env.NODE_ENV === 'development') {
  // Expose store to window for debugging
  if (typeof window !== 'undefined') {
    ;(window as any).store = useStore
    ;(window as any).getStoreState = () => useStore.getState()
    ;(window as any).logStoreState = () => console.log('Store State:', useStore.getState())
  }
}

// Re-export selectors for convenience
export { usePerformanceSelectors } from './selectors'

export default useStore
