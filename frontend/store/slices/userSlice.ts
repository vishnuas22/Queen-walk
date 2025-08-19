// User slice for Zustand store

import type { StateCreator } from 'zustand'
import type { Store, UserState, UserActions, User, UserPreferences } from '../types'

// ===== INITIAL STATE =====

const initialUserPreferences: UserPreferences = {
  theme: 'auto',
  language: 'en',
  accessibility: {
    reducedMotion: false,
    highContrast: false,
    largeText: false,
    screenReaderMode: false,
    voiceNavigation: true,
  },
  chat: {
    defaultModel: 'gpt-4',
    autoSave: true,
    showTimestamps: true,
    enableNotifications: true,
    voiceInput: true,
    voiceOutput: true,
  },
  performance: {
    enableVirtualScrolling: true,
    maxMessagesInMemory: 1000,
    enableAnimations: true,
    enableCaching: true,
  },
}

const initialUserState: UserState = {
  // User data
  user: null,
  isAuthenticated: false,
  
  // Preferences
  preferences: initialUserPreferences,
  
  // Loading states
  userLoading: false,
  preferencesLoading: false,
  
  // Errors
  userError: null,
  preferencesError: null,
  
  // Session
  sessionExpiry: null,
  lastActivity: new Date(),
}

// ===== USER SLICE CREATOR =====

export const createUserSlice: StateCreator<
  Store,
  [['zustand/immer', never], ['zustand/devtools', never], ['zustand/persist', unknown], ['zustand/subscribeWithSelector', never]],
  [],
  UserState & UserActions
> = (set, get, api) => ({
  // Initial state - properly namespaced under 'user'
  user: initialUserState,

  // ===== AUTHENTICATION ACTIONS =====

  login: async (credentials: any) => {
    set((state) => {
      state.user.userLoading = true
      state.user.userError = null
    })

    try {
      // Simulate login API call
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials),
      })

      if (!response.ok) {
        throw new Error('Login failed')
      }

      const userData = await response.json()

      set((state) => {
        state.user.user = userData.user
        state.user.isAuthenticated = true
        state.user.userLoading = false
        state.user.sessionExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
        state.user.lastActivity = new Date()
      })

      // Load user preferences
      await get().loadUserPreferences()

      // Notify successful login
      const { addNotification } = get()
      addNotification({
        type: 'success',
        title: 'Welcome back!',
        message: `Logged in as ${userData.user.name}`,
        duration: 3000,
      })

    } catch (error) {
      set((state) => {
        state.user.userLoading = false
        state.user.userError = error instanceof Error ? error.message : 'Login failed'
      })

      // Notify login error
      const { addNotification } = get()
      addNotification({
        type: 'error',
        title: 'Login Failed',
        message: 'Please check your credentials and try again',
        duration: 5000,
      })

      throw error
    }
  },

  logout: async () => {
    try {
      await fetch('/api/auth/logout', { method: 'POST' })
    } catch (error) {
      console.warn('Logout API call failed:', error)
    }

    set((state) => {
      state.user.user = null
      state.user.isAuthenticated = false
      state.user.sessionExpiry = null
      state.user.userError = null
      state.user.preferencesError = null
    })

    // Clear chat data on logout
    const { clearMessages } = get()
    clearMessages()

    // Notify logout
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title: 'Logged out',
      message: 'You have been successfully logged out',
      duration: 3000,
    })
  },

  refreshToken: async () => {
    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        credentials: 'include',
      })

      if (!response.ok) {
        throw new Error('Token refresh failed')
      }

      const data = await response.json()

      set((state) => {
        state.user.sessionExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000)
        state.user.lastActivity = new Date()
      })

      return data.token
    } catch (error) {
      console.error('Token refresh failed:', error)
      
      // Force logout on refresh failure
      await get().logout()
      throw error
    }
  },

  // ===== USER DATA ACTIONS =====

  updateUser: async (updates: Partial<User>) => {
    set((state) => {
      state.user.userLoading = true
      state.user.userError = null
    })

    try {
      const response = await fetch('/api/user/profile', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      })

      if (!response.ok) {
        throw new Error('Failed to update user profile')
      }

      const updatedUser = await response.json()

      set((state) => {
        state.user.user = updatedUser
        state.user.userLoading = false
      })

      // Notify successful update
      const { addNotification } = get()
      addNotification({
        type: 'success',
        title: 'Profile Updated',
        message: 'Your profile has been successfully updated',
        duration: 3000,
      })

    } catch (error) {
      set((state) => {
        state.user.userLoading = false
        state.user.userError = error instanceof Error ? error.message : 'Update failed'
      })

      // Notify update error
      const { addNotification } = get()
      addNotification({
        type: 'error',
        title: 'Update Failed',
        message: 'Failed to update your profile. Please try again.',
        duration: 5000,
      })

      throw error
    }
  },

  updatePreferences: async (preferences: Partial<UserPreferences>) => {
    set((state) => {
      state.user.preferencesLoading = true
      state.user.preferencesError = null
    })

    try {
      // Update local state immediately for better UX
      set((state) => {
        state.user.preferences = { ...state.user.preferences, ...preferences }
      })

      // Apply preferences immediately
      get().applyPreferences()

      // Save to server if authenticated
      if (get().user.isAuthenticated) {
        const response = await fetch('/api/user/preferences', {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(preferences),
        })

        if (!response.ok) {
          throw new Error('Failed to save preferences')
        }
      }

      set((state) => {
        state.user.preferencesLoading = false
      })

    } catch (error) {
      set((state) => {
        state.user.preferencesLoading = false
        state.user.preferencesError = error instanceof Error ? error.message : 'Failed to save preferences'
      })

      console.error('Failed to save preferences:', error)
    }
  },

  // ===== SESSION MANAGEMENT =====

  updateLastActivity: () => {
    set((state) => {
      state.user.lastActivity = new Date()
    })
  },

  checkSessionExpiry: () => {
    const { sessionExpiry, isAuthenticated } = get().user
    
    if (!isAuthenticated || !sessionExpiry) {
      return false
    }

    const now = new Date()
    const isExpired = now > sessionExpiry

    if (isExpired) {
      console.log('Session expired, logging out...')
      get().logout()
      return true
    }

    // Warn if session expires in 5 minutes
    const fiveMinutesFromNow = new Date(now.getTime() + 5 * 60 * 1000)
    if (sessionExpiry < fiveMinutesFromNow) {
      const { addNotification } = get()
      addNotification({
        type: 'warning',
        title: 'Session Expiring',
        message: 'Your session will expire soon. Please save your work.',
        duration: 10000,
        persistent: true,
        actions: [
          {
            label: 'Extend Session',
            action: () => get().refreshToken(),
            style: 'primary',
          },
        ],
      })
    }

    return false
  },

  // ===== HELPER METHODS =====

  loadUserPreferences: async () => {
    if (!get().user.isAuthenticated) return

    set((state) => {
      state.user.preferencesLoading = true
    })

    try {
      const response = await fetch('/api/user/preferences')
      
      if (response.ok) {
        const preferences = await response.json()
        
        set((state) => {
          state.user.preferences = { ...initialUserPreferences, ...preferences }
        })

        // Apply loaded preferences
        get().applyPreferences()
      }
    } catch (error) {
      console.error('Failed to load user preferences:', error)
    } finally {
      set((state) => {
        state.user.preferencesLoading = false
      })
    }
  },

  applyPreferences: () => {
    const { preferences } = get().user
    
    if (typeof window === 'undefined') return

    // Apply theme
    const { setTheme } = get()
    setTheme(preferences.theme)

    // Apply accessibility preferences
    const body = document.body
    body.classList.toggle('reduce-motion', preferences.accessibility.reducedMotion)
    body.classList.toggle('high-contrast', preferences.accessibility.highContrast)
    body.classList.toggle('large-text', preferences.accessibility.largeText)
    body.classList.toggle('screen-reader-mode', preferences.accessibility.screenReaderMode)

    // Apply performance preferences
    body.classList.toggle('disable-animations', !preferences.performance.enableAnimations)

    // Update UI state
    const { setVoiceEnabled } = get()
    setVoiceEnabled(preferences.chat.voiceInput)
  },

  resetPreferences: () => {
    set((state) => {
      state.user.preferences = { ...initialUserPreferences }
    })

    get().applyPreferences()

    // Notify reset
    const { addNotification } = get()
    addNotification({
      type: 'info',
      title: 'Preferences Reset',
      message: 'All preferences have been reset to defaults',
      duration: 3000,
    })
  },

  exportPreferences: () => {
    const { preferences } = get().user
    const dataStr = JSON.stringify(preferences, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'masterx-preferences.json'
    link.click()
    
    URL.revokeObjectURL(url)

    // Notify export
    const { addNotification } = get()
    addNotification({
      type: 'success',
      title: 'Preferences Exported',
      message: 'Your preferences have been downloaded',
      duration: 3000,
    })
  },

  importPreferences: async (file: File) => {
    try {
      const text = await file.text()
      const preferences = JSON.parse(text)
      
      // Validate preferences structure
      if (typeof preferences === 'object' && preferences !== null) {
        await get().updatePreferences(preferences)
        
        // Notify import success
        const { addNotification } = get()
        addNotification({
          type: 'success',
          title: 'Preferences Imported',
          message: 'Your preferences have been successfully imported',
          duration: 3000,
        })
      } else {
        throw new Error('Invalid preferences file format')
      }
    } catch (error) {
      // Notify import error
      const { addNotification } = get()
      addNotification({
        type: 'error',
        title: 'Import Failed',
        message: 'Failed to import preferences. Please check the file format.',
        duration: 5000,
      })
      
      throw error
    }
  },
})
