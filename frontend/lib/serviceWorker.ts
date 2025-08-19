// Service Worker registration and management

// ===== TYPES =====

interface ServiceWorkerMessage {
  type: string
  data?: any
}

interface CacheStats {
  [cacheName: string]: number
}

// ===== SERVICE WORKER REGISTRATION =====

export const registerServiceWorker = async (): Promise<ServiceWorkerRegistration | null> => {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    console.log('Service Worker not supported')
    return null
  }

  try {
    console.log('🔧 Registering Service Worker...')
    
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    })

    console.log('✅ Service Worker registered successfully:', registration.scope)

    // Handle updates
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing
      if (newWorker) {
        console.log('🔄 New Service Worker installing...')
        
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            console.log('🆕 New Service Worker available')
            
            // Notify user about update
            if (window.confirm('A new version is available. Reload to update?')) {
              window.location.reload()
            }
          }
        })
      }
    })

    // Listen for messages from Service Worker
    navigator.serviceWorker.addEventListener('message', handleServiceWorkerMessage)

    return registration
  } catch (error) {
    console.error('❌ Service Worker registration failed:', error)
    return null
  }
}

// ===== MESSAGE HANDLING =====

const handleServiceWorkerMessage = (event: MessageEvent<ServiceWorkerMessage>) => {
  const { type, data } = event.data

  switch (type) {
    case 'OFFLINE_SYNC_COMPLETE':
      console.log(`✅ Offline sync completed: ${data.syncedCount} messages`)
      
      // Show notification to user
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('MasterX', {
          body: `Synced ${data.syncedCount} offline messages`,
          icon: '/favicon.ico',
          tag: 'offline-sync',
        })
      }
      
      // Dispatch custom event for React components
      window.dispatchEvent(new CustomEvent('offline-sync-complete', { detail: data }))
      break

    case 'CACHE_UPDATE':
      console.log('📦 Cache updated:', data)
      break

    default:
      console.log('Unknown Service Worker message:', type, data)
  }
}

// ===== CACHE MANAGEMENT =====

export const getCacheStats = async (): Promise<CacheStats | null> => {
  if (!navigator.serviceWorker.controller) {
    console.warn('No active Service Worker')
    return null
  }

  return new Promise((resolve) => {
    const messageChannel = new MessageChannel()
    
    messageChannel.port1.onmessage = (event) => {
      resolve(event.data)
    }

    navigator.serviceWorker.controller!.postMessage(
      { type: 'CACHE_STATS' },
      [messageChannel.port2]
    )
  })
}

export const clearAllCaches = async (): Promise<boolean> => {
  if (!navigator.serviceWorker.controller) {
    console.warn('No active Service Worker')
    return false
  }

  return new Promise((resolve) => {
    const messageChannel = new MessageChannel()
    
    messageChannel.port1.onmessage = (event) => {
      resolve(event.data.success || false)
    }

    navigator.serviceWorker.controller!.postMessage(
      { type: 'CLEAR_CACHE' },
      [messageChannel.port2]
    )
  })
}

// ===== BACKGROUND SYNC =====

export const requestBackgroundSync = async (tag: string): Promise<void> => {
  if (!('serviceWorker' in navigator) || !('sync' in window.ServiceWorkerRegistration.prototype)) {
    console.warn('Background Sync not supported')
    return
  }

  try {
    const registration = await navigator.serviceWorker.ready
    // Type assertion for background sync API
    await (registration as any).sync.register(tag)
    console.log(`🔄 Background sync registered: ${tag}`)
  } catch (error) {
    console.error('❌ Background sync registration failed:', error)
  }
}

// ===== OFFLINE DETECTION =====

export class OfflineManager {
  private listeners: Set<(isOnline: boolean) => void> = new Set()
  private _isOnline: boolean = typeof navigator !== 'undefined' ? navigator.onLine : true

  constructor() {
    if (typeof window !== 'undefined') {
      window.addEventListener('online', this.handleOnline)
      window.addEventListener('offline', this.handleOffline)
    }
  }

  private handleOnline = () => {
    console.log('🌐 Connection restored')
    this._isOnline = true
    this.notifyListeners(true)
    
    // Trigger background sync when coming online
    requestBackgroundSync('background-sync-messages')
  }

  private handleOffline = () => {
    console.log('📴 Connection lost')
    this._isOnline = false
    this.notifyListeners(false)
  }

  private notifyListeners(isOnline: boolean) {
    this.listeners.forEach(listener => listener(isOnline))
  }

  public get isOnline(): boolean {
    return this._isOnline
  }

  public addListener(listener: (isOnline: boolean) => void): () => void {
    this.listeners.add(listener)
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener)
    }
  }

  public destroy() {
    if (typeof window !== 'undefined') {
      window.removeEventListener('online', this.handleOnline)
      window.removeEventListener('offline', this.handleOffline)
    }
    this.listeners.clear()
  }
}

// ===== NOTIFICATION MANAGEMENT =====

export const requestNotificationPermission = async (): Promise<NotificationPermission> => {
  if (!('Notification' in window)) {
    console.warn('Notifications not supported')
    return 'denied'
  }

  if (Notification.permission === 'granted') {
    return 'granted'
  }

  if (Notification.permission === 'denied') {
    return 'denied'
  }

  // Request permission
  const permission = await Notification.requestPermission()
  console.log('🔔 Notification permission:', permission)
  
  return permission
}

export const showNotification = (title: string, options?: NotificationOptions) => {
  if (Notification.permission !== 'granted') {
    console.warn('Notification permission not granted')
    return
  }

  const notification = new Notification(title, {
    icon: '/favicon.ico',
    badge: '/favicon.ico',
    ...options,
  })

  // Auto-close after 5 seconds
  setTimeout(() => {
    notification.close()
  }, 5000)

  return notification
}

// ===== INSTALLATION PROMPT =====

export class InstallPromptManager {
  private deferredPrompt: any = null
  private listeners: Set<(canInstall: boolean) => void> = new Set()

  constructor() {
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeinstallprompt', this.handleBeforeInstallPrompt)
      window.addEventListener('appinstalled', this.handleAppInstalled)
    }
  }

  private handleBeforeInstallPrompt = (event: Event) => {
    console.log('📱 Install prompt available')
    
    // Prevent the mini-infobar from appearing
    event.preventDefault()
    
    // Save the event for later use
    this.deferredPrompt = event
    
    // Notify listeners
    this.notifyListeners(true)
  }

  private handleAppInstalled = () => {
    console.log('✅ App installed successfully')
    this.deferredPrompt = null
    this.notifyListeners(false)
  }

  private notifyListeners(canInstall: boolean) {
    this.listeners.forEach(listener => listener(canInstall))
  }

  public get canInstall(): boolean {
    return !!this.deferredPrompt
  }

  public async showInstallPrompt(): Promise<boolean> {
    if (!this.deferredPrompt) {
      console.warn('No install prompt available')
      return false
    }

    try {
      // Show the install prompt
      this.deferredPrompt.prompt()
      
      // Wait for the user's response
      const { outcome } = await this.deferredPrompt.userChoice
      
      console.log('Install prompt outcome:', outcome)
      
      // Clear the prompt
      this.deferredPrompt = null
      this.notifyListeners(false)
      
      return outcome === 'accepted'
    } catch (error) {
      console.error('Install prompt failed:', error)
      return false
    }
  }

  public addListener(listener: (canInstall: boolean) => void): () => void {
    this.listeners.add(listener)
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener)
    }
  }

  public destroy() {
    if (typeof window !== 'undefined') {
      window.removeEventListener('beforeinstallprompt', this.handleBeforeInstallPrompt)
      window.removeEventListener('appinstalled', this.handleAppInstalled)
    }
    this.listeners.clear()
  }
}

// ===== GLOBAL INSTANCES =====

let _offlineManager: OfflineManager | null = null
let _installPromptManager: InstallPromptManager | null = null

export const getOfflineManager = (): OfflineManager | null => {
  if (typeof window === 'undefined') return null
  if (!_offlineManager) {
    _offlineManager = new OfflineManager()
  }
  return _offlineManager
}

export const getInstallPromptManager = (): InstallPromptManager | null => {
  if (typeof window === 'undefined') return null
  if (!_installPromptManager) {
    _installPromptManager = new InstallPromptManager()
  }
  return _installPromptManager
}

// Legacy exports for backward compatibility - only available on client
export const offlineManager = typeof window !== 'undefined' ? getOfflineManager() : null
export const installPromptManager = typeof window !== 'undefined' ? getInstallPromptManager() : null

// ===== INITIALIZATION =====

export const initializeServiceWorker = async () => {
  console.log('🚀 Initializing Service Worker...')
  
  // Register service worker
  const registration = await registerServiceWorker()
  
  if (registration) {
    // Request notification permission
    await requestNotificationPermission()
    
    console.log('✅ Service Worker initialization complete')
  }
  
  return registration
}
