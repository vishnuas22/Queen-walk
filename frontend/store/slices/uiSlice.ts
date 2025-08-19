// UI slice for Zustand store

import type { StateCreator } from 'zustand'
import type { Store, UIState, UIActions, Notification } from '../types'

// ===== INITIAL STATE =====

const initialUIState: UIState = {
  // Layout
  sidebarOpen: false,
  sidebarWidth: 320,
  
  // Modals and overlays
  activeModal: null,
  modalData: null,
  
  // Loading states
  globalLoading: false,
  loadingStates: {},
  
  // Notifications
  notifications: [],
  
  // Accessibility
  keyboardNavigation: false,
  focusedElement: null,
  
  // Performance
  performanceMode: false,
  debugMode: false,
  
  // Voice
  voiceListening: false,
  voiceEnabled: true,
  
  // Theme
  theme: 'auto',
  
  // Responsive
  isMobile: false,
  screenSize: 'lg',
}

// ===== UI SLICE CREATOR =====

export const createUISlice: StateCreator<
  Store,
  [['zustand/immer', never], ['zustand/devtools', never], ['zustand/persist', unknown], ['zustand/subscribeWithSelector', never]],
  [],
  UIState & UIActions
> = (set, get, api) => ({
  // Initial state - properly namespaced under 'ui'
  ui: initialUIState,

  // ===== LAYOUT ACTIONS =====

  toggleSidebar: () => {
    set((state) => {
      state.ui.sidebarOpen = !state.ui.sidebarOpen
    })
  },

  setSidebarWidth: (width: number) => {
    set((state) => {
      state.ui.sidebarWidth = Math.max(200, Math.min(600, width))
    })
  },

  // ===== MODAL ACTIONS =====

  openModal: (modalId: string, data?: any) => {
    set((state) => {
      state.ui.activeModal = modalId
      state.ui.modalData = data || null
    })

    // Announce modal opening for screen readers
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title: 'Modal opened',
      message: `${modalId} modal is now open`,
      duration: 1000,
    })
  },

  closeModal: () => {
    set((state) => {
      state.ui.activeModal = null
      state.ui.modalData = null
    })
  },

  // ===== LOADING ACTIONS =====

  setGlobalLoading: (loading: boolean) => {
    set((state) => {
      state.ui.globalLoading = loading
    })
  },

  setLoadingState: (key: string, loading: boolean) => {
    set((state) => {
      if (loading) {
        state.ui.loadingStates[key] = loading
      } else {
        delete state.ui.loadingStates[key]
      }
    })
  },

  // ===== NOTIFICATION ACTIONS =====

  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const newNotification: Notification = {
      ...notification,
      id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
    }

    set((state) => {
      state.ui.notifications.push(newNotification)
      
      // Limit to 10 notifications
      if (state.ui.notifications.length > 10) {
        state.ui.notifications = state.ui.notifications.slice(-10)
      }
    })

    // Auto-remove notification after duration
    if (notification.duration && !notification.persistent) {
      setTimeout(() => {
        get().removeNotification(newNotification.id)
      }, notification.duration)
    }

    // Announce notification for screen readers
    if (typeof window !== 'undefined') {
      const announcement = `${notification.type}: ${notification.title}. ${notification.message}`
      
      // Use ARIA live region
      const liveRegion = document.getElementById('aria-live-polite')
      if (liveRegion) {
        liveRegion.textContent = announcement
      }
    }
  },

  removeNotification: (id: string) => {
    set((state) => {
      state.ui.notifications = state.ui.notifications.filter(n => n.id !== id)
    })
  },

  clearNotifications: () => {
    set((state) => {
      state.ui.notifications = []
    })
  },

  // ===== ACCESSIBILITY ACTIONS =====

  setKeyboardNavigation: (enabled: boolean) => {
    set((state) => {
      state.ui.keyboardNavigation = enabled
    })

    // Apply keyboard navigation class to body
    if (typeof window !== 'undefined') {
      document.body.classList.toggle('keyboard-navigation', enabled)
    }
  },

  setFocusedElement: (elementId: string | null) => {
    set((state) => {
      state.ui.focusedElement = elementId
    })
  },

  // ===== PERFORMANCE ACTIONS =====

  setPerformanceMode: (enabled: boolean) => {
    set((state) => {
      state.ui.performanceMode = enabled
    })

    // Apply performance optimizations
    if (typeof window !== 'undefined') {
      document.body.classList.toggle('performance-mode', enabled)
      
      if (enabled) {
        // Disable animations in performance mode
        document.body.classList.add('reduce-motion')
      } else {
        // Re-enable animations based on user preference
        const { user } = get()
        if (!user.preferences.accessibility.reducedMotion) {
          document.body.classList.remove('reduce-motion')
        }
      }
    }

    // Notify about performance mode change
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title: 'Performance Mode',
      message: `Performance mode ${enabled ? 'enabled' : 'disabled'}`,
      duration: 2000,
    })
  },

  setDebugMode: (enabled: boolean) => {
    set((state) => {
      state.ui.debugMode = enabled
    })

    // Enable/disable debug logging
    if (enabled) {
      console.log('🐛 Debug mode enabled')
      
      // Expose debug utilities
      if (typeof window !== 'undefined') {
        ;(window as any).debugMode = true
        ;(window as any).logState = () => console.log('State:', get())
      }
    } else {
      console.log('🐛 Debug mode disabled')
      
      if (typeof window !== 'undefined') {
        ;(window as any).debugMode = false
      }
    }
  },

  // ===== VOICE ACTIONS =====

  setVoiceListening: (listening: boolean) => {
    set((state) => {
      state.ui.voiceListening = listening
    })

    // Announce voice state change
    const { addNotification } = get()
    if (listening) {
      addNotification({
        type: 'info',
        title: 'Voice Recognition',
        message: 'Listening for voice commands...',
        duration: 2000,
      })
    }
  },

  setVoiceEnabled: (enabled: boolean) => {
    set((state) => {
      state.ui.voiceEnabled = enabled
    })

    // Update user preferences
    const { updatePreferences } = get()
    updatePreferences({
      chat: {
        ...get().user.preferences.chat,
        voiceInput: enabled,
      },
    })
  },

  // ===== THEME ACTIONS =====

  setTheme: (theme: 'light' | 'dark' | 'auto') => {
    set((state) => {
      state.ui.theme = theme
    })

    // Apply theme to document
    if (typeof window !== 'undefined') {
      let actualTheme = theme
      
      if (theme === 'auto') {
        actualTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
      }
      
      document.documentElement.setAttribute('data-theme', actualTheme)
      document.documentElement.classList.toggle('dark', actualTheme === 'dark')
    }

    // Update user preferences
    const { updatePreferences } = get()
    updatePreferences({
      theme,
    })

    // Announce theme change
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title: 'Theme Changed',
      message: `Theme set to ${theme}`,
      duration: 2000,
    })
  },

  // ===== RESPONSIVE ACTIONS =====

  setScreenSize: (size: 'sm' | 'md' | 'lg' | 'xl') => {
    set((state) => {
      state.ui.screenSize = size
    })

    // Auto-close sidebar on small screens
    if (size === 'sm' || size === 'md') {
      const { setSidebarOpen } = get()
      setSidebarOpen(false)
    }
  },

  setIsMobile: (isMobile: boolean) => {
    set((state) => {
      state.ui.isMobile = isMobile
    })

    // Apply mobile class to body
    if (typeof window !== 'undefined') {
      document.body.classList.toggle('mobile', isMobile)
    }
  },

  // ===== UTILITY METHODS =====

  showSuccessNotification: (message: string, title = 'Success') => {
    const { addNotification } = get()
    addNotification({
      type: 'success',
      title,
      message,
      duration: 3000,
    })
  },

  showErrorNotification: (message: string, title = 'Error') => {
    const { addNotification } = get()
    addNotification({
      type: 'error',
      title,
      message,
      duration: 5000,
    })
  },

  showWarningNotification: (message: string, title = 'Warning') => {
    const { addNotification } = get()
    addNotification({
      type: 'warning',
      title,
      message,
      duration: 4000,
    })
  },

  showInfoNotification: (message: string, title = 'Info') => {
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title,
      message,
      duration: 3000,
    })
  },

  isLoading: (key?: string) => {
    const state = get()
    if (key) {
      return state.ui.loadingStates[key] || false
    }
    return state.ui.globalLoading || Object.keys(state.ui.loadingStates).length > 0
  },

  getActiveLoadingStates: () => {
    const state = get()
    return Object.keys(state.ui.loadingStates).filter(key => state.ui.loadingStates[key])
  },
})
