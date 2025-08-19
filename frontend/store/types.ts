// Global state management types for MasterX

// ===== CORE TYPES =====

export interface Message {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  isStreaming?: boolean
  metadata?: {
    model?: string
    tokens?: number
    confidence?: number
    processingTime?: number
  }
}

export interface ChatSession {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  last_message_at?: string
  metadata?: {
    model?: string
    totalTokens?: number
    averageResponseTime?: number
  }
}

export interface User {
  id: string
  name: string
  email: string
  avatar?: string
  preferences: UserPreferences
  subscription?: {
    plan: 'free' | 'pro' | 'enterprise'
    expiresAt?: string
    features: string[]
  }
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto'
  language: string
  accessibility: {
    reducedMotion: boolean
    highContrast: boolean
    largeText: boolean
    screenReaderMode: boolean
    voiceNavigation: boolean
  }
  chat: {
    defaultModel: string
    autoSave: boolean
    showTimestamps: boolean
    enableNotifications: boolean
    voiceInput: boolean
    voiceOutput: boolean
  }
  performance: {
    enableVirtualScrolling: boolean
    maxMessagesInMemory: number
    enableAnimations: boolean
    enableCaching: boolean
  }
}

// ===== STATE INTERFACES =====

export interface ChatState {
  // Current session
  currentSessionId: string | null
  currentMessages: Message[]
  
  // All sessions
  sessions: ChatSession[]
  sessionsLoading: boolean
  sessionsError: string | null
  
  // Message state
  isLoading: boolean
  isStreaming: boolean
  streamingMessageId: string | null
  
  // Input state
  inputMessage: string
  inputHistory: string[]
  inputHistoryIndex: number
  
  // UI state
  sidebarOpen: boolean
  selectedMessageId: string | null
  
  // Performance state
  virtualScrollEnabled: boolean
  messagesInView: number
  totalMessagesLoaded: number
}

export interface UIState {
  // Layout
  sidebarOpen: boolean
  sidebarWidth: number
  
  // Modals and overlays
  activeModal: string | null
  modalData: any
  
  // Loading states
  globalLoading: boolean
  loadingStates: Record<string, boolean>
  
  // Notifications
  notifications: Notification[]
  
  // Accessibility
  keyboardNavigation: boolean
  focusedElement: string | null
  
  // Performance
  performanceMode: boolean
  debugMode: boolean
  
  // Voice
  voiceListening: boolean
  voiceEnabled: boolean
  
  // Theme
  theme: 'light' | 'dark' | 'auto'
  
  // Responsive
  isMobile: boolean
  screenSize: 'sm' | 'md' | 'lg' | 'xl'
}

export interface UserState {
  // User data
  user: User | null
  isAuthenticated: boolean
  
  // Preferences
  preferences: UserPreferences
  
  // Loading states
  userLoading: boolean
  preferencesLoading: boolean
  
  // Errors
  userError: string | null
  preferencesError: string | null
  
  // Session
  sessionExpiry: Date | null
  lastActivity: Date | null
}

export interface AppState {
  // App metadata
  version: string
  buildTime: string
  environment: 'development' | 'staging' | 'production'
  
  // Connection state
  isOnline: boolean
  connectionQuality: 'poor' | 'good' | 'excellent'
  
  // Performance metrics
  performanceMetrics: {
    loadTime: number
    renderTime: number
    memoryUsage: number
    cacheHitRate: number
  }
  
  // Feature flags
  featureFlags: Record<string, boolean>
  
  // Error tracking
  errors: AppError[]
  
  // Analytics
  analytics: {
    sessionStart: Date
    pageViews: number
    interactions: number
    errors: number
  }
}

// ===== UTILITY TYPES =====

export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
  timestamp: Date
  duration?: number
  actions?: NotificationAction[]
  persistent?: boolean
}

export interface NotificationAction {
  label: string
  action: () => void
  style?: 'primary' | 'secondary' | 'danger'
}

export interface AppError {
  id: string
  message: string
  stack?: string
  timestamp: Date
  context?: Record<string, any>
  severity: 'low' | 'medium' | 'high' | 'critical'
  resolved?: boolean
}

export interface LoadingState {
  isLoading: boolean
  progress?: number
  message?: string
  startTime?: Date
}

export interface CacheEntry<T = any> {
  data: T
  timestamp: Date
  expiresAt: Date
  key: string
  size: number
}

// ===== ACTION TYPES =====

export interface ChatActions {
  // Session management
  setCurrentSession: (sessionId: string | null) => void
  createSession: () => Promise<string>
  deleteSession: (sessionId: string) => Promise<void>
  updateSessionTitle: (sessionId: string, title: string) => Promise<void>
  
  // Message management
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  updateMessage: (messageId: string, updates: Partial<Message>) => void
  deleteMessage: (messageId: string) => void
  clearMessages: () => void
  
  // Input management
  setInputMessage: (message: string) => void
  addToInputHistory: (message: string) => void
  navigateInputHistory: (direction: 'up' | 'down') => void
  
  // Sending messages
  sendMessage: (content: string) => Promise<void>
  resendMessage: (messageId: string) => Promise<void>
  
  // UI actions
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  selectMessage: (messageId: string | null) => void
  
  // Loading states
  setLoading: (loading: boolean) => void
  setStreaming: (streaming: boolean, messageId?: string) => void
}

export interface UIActions {
  // Layout
  toggleSidebar: () => void
  setSidebarWidth: (width: number) => void
  
  // Modals
  openModal: (modalId: string, data?: any) => void
  closeModal: () => void
  
  // Loading
  setGlobalLoading: (loading: boolean) => void
  setLoadingState: (key: string, loading: boolean) => void
  
  // Notifications
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void
  removeNotification: (id: string) => void
  clearNotifications: () => void
  
  // Accessibility
  setKeyboardNavigation: (enabled: boolean) => void
  setFocusedElement: (elementId: string | null) => void
  
  // Performance
  setPerformanceMode: (enabled: boolean) => void
  setDebugMode: (enabled: boolean) => void
  
  // Voice
  setVoiceListening: (listening: boolean) => void
  setVoiceEnabled: (enabled: boolean) => void
  
  // Theme
  setTheme: (theme: 'light' | 'dark' | 'auto') => void
  
  // Responsive
  setScreenSize: (size: 'sm' | 'md' | 'lg' | 'xl') => void
  setIsMobile: (isMobile: boolean) => void
}

export interface UserActions {
  // Authentication
  login: (credentials: any) => Promise<void>
  logout: () => Promise<void>
  refreshToken: () => Promise<void>
  
  // User data
  updateUser: (updates: Partial<User>) => Promise<void>
  updatePreferences: (preferences: Partial<UserPreferences>) => Promise<void>
  
  // Session management
  updateLastActivity: () => void
  checkSessionExpiry: () => boolean
}

export interface AppActions {
  // Connection
  setOnlineStatus: (online: boolean) => void
  updateConnectionQuality: (quality: 'poor' | 'good' | 'excellent') => void
  
  // Performance
  updatePerformanceMetrics: (metrics: Partial<AppState['performanceMetrics']>) => void
  
  // Feature flags
  setFeatureFlag: (flag: string, enabled: boolean) => void
  
  // Error handling
  addError: (error: Omit<AppError, 'id' | 'timestamp'>) => void
  resolveError: (errorId: string) => void
  clearErrors: () => void
  
  // Analytics
  trackPageView: () => void
  trackInteraction: (type: string, data?: any) => void
  trackError: (error: Error) => void
}

// ===== STORE TYPES =====

export interface RootState {
  chat: ChatState
  ui: UIState
  user: UserState
  app: AppState
}

export interface RootActions {
  chat: ChatActions
  ui: UIActions
  user: UserActions
  app: AppActions
}

export type Store = RootState & RootActions

// ===== PERSISTENCE TYPES =====

export interface PersistConfig {
  name: string
  version: number
  storage: 'localStorage' | 'sessionStorage' | 'indexedDB'
  whitelist?: (keyof RootState)[]
  blacklist?: (keyof RootState)[]
  migrate?: (persistedState: any, version: number) => any
}

export interface StoreSubscription {
  id: string
  selector: (state: RootState) => any
  callback: (value: any, previousValue: any) => void
  immediate?: boolean
}

export default RootState
