// Advanced state persistence for MasterX store

import type { RootState, PersistConfig } from './types'

// ===== STORAGE ADAPTERS =====

interface StorageAdapter {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

class LocalStorageAdapter implements StorageAdapter {
  getItem(key: string): string | null {
    try {
      return localStorage.getItem(key)
    } catch (error) {
      console.warn('LocalStorage getItem failed:', error)
      return null
    }
  }

  setItem(key: string, value: string): void {
    try {
      localStorage.setItem(key, value)
    } catch (error) {
      console.warn('LocalStorage setItem failed:', error)
    }
  }

  removeItem(key: string): void {
    try {
      localStorage.removeItem(key)
    } catch (error) {
      console.warn('LocalStorage removeItem failed:', error)
    }
  }
}

class SessionStorageAdapter implements StorageAdapter {
  getItem(key: string): string | null {
    try {
      return sessionStorage.getItem(key)
    } catch (error) {
      console.warn('SessionStorage getItem failed:', error)
      return null
    }
  }

  setItem(key: string, value: string): void {
    try {
      sessionStorage.setItem(key, value)
    } catch (error) {
      console.warn('SessionStorage setItem failed:', error)
    }
  }

  removeItem(key: string): void {
    try {
      sessionStorage.removeItem(key)
    } catch (error) {
      console.warn('SessionStorage removeItem failed:', error)
    }
  }
}

class IndexedDBAdapter implements StorageAdapter {
  private dbName = 'masterx-store'
  private version = 1
  private storeName = 'state'

  async getItem(key: string): Promise<string | null> {
    try {
      const db = await this.openDB()
      const transaction = db.transaction([this.storeName], 'readonly')
      const store = transaction.objectStore(this.storeName)
      const result = await this.promisifyRequest(store.get(key))
      return result?.value || null
    } catch (error) {
      console.warn('IndexedDB getItem failed:', error)
      return null
    }
  }

  async setItem(key: string, value: string): Promise<void> {
    try {
      const db = await this.openDB()
      const transaction = db.transaction([this.storeName], 'readwrite')
      const store = transaction.objectStore(this.storeName)
      await this.promisifyRequest(store.put({ key, value }))
    } catch (error) {
      console.warn('IndexedDB setItem failed:', error)
    }
  }

  async removeItem(key: string): Promise<void> {
    try {
      const db = await this.openDB()
      const transaction = db.transaction([this.storeName], 'readwrite')
      const store = transaction.objectStore(this.storeName)
      await this.promisifyRequest(store.delete(key))
    } catch (error) {
      console.warn('IndexedDB removeItem failed:', error)
    }
  }

  private openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version)
      
      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve(request.result)
      
      request.onupgradeneeded = () => {
        const db = request.result
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: 'key' })
        }
      }
    })
  }

  private promisifyRequest<T>(request: IDBRequest<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve(request.result)
    })
  }

  // Sync methods for compatibility
  getItem(key: string): string | null {
    // For sync compatibility, return null and use async version
    this.getItem(key).then(value => value).catch(() => null)
    return null
  }

  setItem(key: string, value: string): void {
    this.setItem(key, value).catch(console.warn)
  }

  removeItem(key: string): void {
    this.removeItem(key).catch(console.warn)
  }
}

// ===== STORAGE FACTORY =====

export const createStorageAdapter = (type: 'localStorage' | 'sessionStorage' | 'indexedDB'): StorageAdapter => {
  if (typeof window === 'undefined') {
    // Server-side fallback
    return {
      getItem: () => null,
      setItem: () => {},
      removeItem: () => {},
    }
  }

  switch (type) {
    case 'localStorage':
      return new LocalStorageAdapter()
    case 'sessionStorage':
      return new SessionStorageAdapter()
    case 'indexedDB':
      return new IndexedDBAdapter()
    default:
      return new LocalStorageAdapter()
  }
}

// ===== PERSISTENCE MANAGER =====

export class PersistenceManager {
  private storage: StorageAdapter
  private config: PersistConfig

  constructor(config: PersistConfig) {
    this.config = config
    this.storage = createStorageAdapter(config.storage)
  }

  // Serialize state for storage
  serialize(state: Partial<RootState>): string {
    try {
      return JSON.stringify({
        state,
        version: this.config.version,
        timestamp: Date.now(),
      })
    } catch (error) {
      console.error('Failed to serialize state:', error)
      return '{}'
    }
  }

  // Deserialize state from storage
  deserialize(data: string): Partial<RootState> | null {
    try {
      const parsed = JSON.parse(data)
      
      // Check version compatibility
      if (parsed.version !== this.config.version) {
        console.log(`State version mismatch. Migrating from ${parsed.version} to ${this.config.version}`)
        return this.migrate(parsed.state, parsed.version)
      }
      
      return parsed.state
    } catch (error) {
      console.error('Failed to deserialize state:', error)
      return null
    }
  }

  // Migrate state between versions
  migrate(state: any, fromVersion: number): Partial<RootState> | null {
    if (this.config.migrate) {
      try {
        return this.config.migrate(state, fromVersion)
      } catch (error) {
        console.error('State migration failed:', error)
        return null
      }
    }
    return state
  }

  // Filter state based on whitelist/blacklist
  filterState(state: RootState): Partial<RootState> {
    const filtered: Partial<RootState> = {}

    // Apply whitelist
    if (this.config.whitelist) {
      for (const key of this.config.whitelist) {
        if (key in state) {
          filtered[key] = state[key]
        }
      }
    } else {
      // Copy all if no whitelist
      Object.assign(filtered, state)
    }

    // Apply blacklist
    if (this.config.blacklist) {
      for (const key of this.config.blacklist) {
        delete filtered[key]
      }
    }

    return filtered
  }

  // Save state to storage
  saveState(state: RootState): void {
    try {
      const filteredState = this.filterState(state)
      const serialized = this.serialize(filteredState)
      this.storage.setItem(this.config.name, serialized)
      
      console.log('💾 State persisted to storage')
    } catch (error) {
      console.error('Failed to save state:', error)
    }
  }

  // Load state from storage
  loadState(): Partial<RootState> | null {
    try {
      const data = this.storage.getItem(this.config.name)
      if (!data) return null
      
      const state = this.deserialize(data)
      console.log('📂 State loaded from storage')
      return state
    } catch (error) {
      console.error('Failed to load state:', error)
      return null
    }
  }

  // Clear persisted state
  clearState(): void {
    try {
      this.storage.removeItem(this.config.name)
      console.log('🗑️ Persisted state cleared')
    } catch (error) {
      console.error('Failed to clear state:', error)
    }
  }

  // Check if storage is available
  isStorageAvailable(): boolean {
    try {
      const testKey = '__storage_test__'
      this.storage.setItem(testKey, 'test')
      this.storage.removeItem(testKey)
      return true
    } catch (error) {
      return false
    }
  }
}

// ===== PERSISTENCE UTILITIES =====

export const createPersistenceMiddleware = (config: PersistConfig) => {
  const manager = new PersistenceManager(config)
  
  return (storeApi: any) => (set: any, get: any, api: any) => {
    // Load initial state
    const persistedState = manager.loadState()
    if (persistedState) {
      // Merge persisted state with initial state
      set((state: RootState) => ({
        ...state,
        ...persistedState,
      }))
    }

    // Set up auto-save
    let saveTimeout: NodeJS.Timeout | null = null
    const debouncedSave = () => {
      if (saveTimeout) clearTimeout(saveTimeout)
      saveTimeout = setTimeout(() => {
        manager.saveState(get())
      }, 1000) // Debounce saves by 1 second
    }

    // Subscribe to state changes
    api.subscribe((state: RootState) => {
      debouncedSave()
    })

    // Expose persistence methods
    return {
      ...storeApi(set, get, api),
      _persistence: {
        save: () => manager.saveState(get()),
        load: () => {
          const state = manager.loadState()
          if (state) set(state)
        },
        clear: () => manager.clearState(),
        isAvailable: () => manager.isStorageAvailable(),
      },
    }
  }
}

// ===== EXPORT DEFAULT PERSISTENCE CONFIG =====

export const defaultPersistConfig: PersistConfig = {
  name: 'masterx-store-v2',
  version: 2,
  storage: 'localStorage',
  whitelist: ['user', 'ui'],
  blacklist: ['chat', 'app'],
  migrate: (persistedState: any, version: number) => {
    // Handle migrations between versions
    if (version === 1) {
      // Migration from v1 to v2
      return {
        ...persistedState,
        ui: {
          ...persistedState.ui,
          theme: persistedState.ui.theme || 'auto',
          notifications: [], // Reset notifications on migration
        },
      }
    }
    return persistedState
  },
}

export default PersistenceManager
