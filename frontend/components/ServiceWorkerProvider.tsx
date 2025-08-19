'use client'

import React, { useEffect, useState } from 'react'
import { initializeServiceWorker, getOfflineManager, getInstallPromptManager, getCacheStats, clearAllCaches } from '../lib/serviceWorker'

interface ServiceWorkerProviderProps {
  children: React.ReactNode
}

// ===== INSTALL PROMPT COMPONENT =====

const InstallPrompt: React.FC = () => {
  const [canInstall, setCanInstall] = useState(false)
  const [isInstalling, setIsInstalling] = useState(false)

  useEffect(() => {
    const manager = getInstallPromptManager()
    if (!manager) return

    const unsubscribe = manager.addListener(setCanInstall)
    setCanInstall(manager.canInstall)

    return unsubscribe
  }, [])

  const handleInstall = async () => {
    const manager = getInstallPromptManager()
    if (!manager) return

    setIsInstalling(true)
    try {
      const installed = await manager.showInstallPrompt()
      if (installed) {
        console.log('✅ App installed successfully')
      }
    } catch (error) {
      console.error('Install failed:', error)
    } finally {
      setIsInstalling(false)
    }
  }

  if (!canInstall) return null

  return (
    <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:max-w-sm z-50">
      <div className="bg-white border border-slate-200 rounded-xl shadow-lg p-4">
        <div className="flex items-start space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-xl flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-slate-900 text-sm">Install MasterX</h3>
            <p className="text-xs text-slate-600 mt-1">
              Add to your home screen for quick access and offline support.
            </p>
            <div className="flex space-x-2 mt-3">
              <button
                onClick={handleInstall}
                disabled={isInstalling}
                className="px-3 py-1.5 bg-indigo-600 text-white text-xs rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
              >
                {isInstalling ? 'Installing...' : 'Install'}
              </button>
              <button
                onClick={() => setCanInstall(false)}
                className="px-3 py-1.5 text-slate-600 text-xs hover:text-slate-800 transition-colors"
              >
                Later
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ===== OFFLINE INDICATOR =====

const OfflineIndicator: React.FC = () => {
  const [isOnline, setIsOnline] = useState(true)

  useEffect(() => {
    const manager = getOfflineManager()
    if (!manager) return

    const unsubscribe = manager.addListener(setIsOnline)
    setIsOnline(manager.isOnline)

    return unsubscribe
  }, [])

  if (isOnline) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-red-600 text-white text-center py-2 text-sm">
      <div className="flex items-center justify-center space-x-2">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636l-12.728 12.728m0-12.728l12.728 12.728" />
        </svg>
        <span>You're offline. Messages will be sent when connection is restored.</span>
      </div>
    </div>
  )
}

// ===== SYNC STATUS INDICATOR =====

const SyncStatusIndicator: React.FC = () => {
  const [syncStatus, setSyncStatus] = useState<'idle' | 'syncing' | 'synced'>('idle')

  useEffect(() => {
    const handleSyncComplete = (event: CustomEvent) => {
      setSyncStatus('synced')
      setTimeout(() => setSyncStatus('idle'), 3000)
    }

    window.addEventListener('offline-sync-complete', handleSyncComplete as EventListener)
    
    return () => {
      window.removeEventListener('offline-sync-complete', handleSyncComplete as EventListener)
    }
  }, [])

  if (syncStatus === 'idle') return null

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className="bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg text-sm flex items-center space-x-2">
        {syncStatus === 'syncing' ? (
          <>
            <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span>Syncing messages...</span>
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span>Messages synced</span>
          </>
        )}
      </div>
    </div>
  )
}

// ===== CACHE MANAGEMENT COMPONENT =====

const CacheManager: React.FC = () => {
  const [cacheStats, setCacheStats] = useState<any>(null)
  const [isClearing, setIsClearing] = useState(false)

  const loadCacheStats = async () => {
    try {
      const stats = await getCacheStats()
      setCacheStats(stats)
    } catch (error) {
      console.error('Failed to load cache stats:', error)
    }
  }

  const handleClearCache = async () => {
    if (!confirm('Are you sure you want to clear all caches? This will remove offline data.')) {
      return
    }

    setIsClearing(true)
    try {
      await clearAllCaches()
      await loadCacheStats()
      console.log('✅ Cache cleared successfully')
    } catch (error) {
      console.error('Failed to clear cache:', error)
    } finally {
      setIsClearing(false)
    }
  }

  // Only show in development
  if (process.env.NODE_ENV !== 'development') {
    return null
  }

  return (
    <div className="fixed bottom-20 right-4 z-40">
      <div className="bg-slate-800 text-white p-3 rounded-lg shadow-lg text-xs max-w-xs">
        <div className="flex items-center justify-between mb-2">
          <span className="font-semibold">Cache Manager</span>
          <button
            onClick={loadCacheStats}
            className="text-blue-400 hover:text-blue-300"
          >
            Refresh
          </button>
        </div>
        
        {cacheStats && (
          <div className="space-y-1 mb-2">
            {Object.entries(cacheStats).map(([name, count]) => (
              <div key={name} className="flex justify-between">
                <span className="truncate">{name}:</span>
                <span>{count as number}</span>
              </div>
            ))}
          </div>
        )}
        
        <button
          onClick={handleClearCache}
          disabled={isClearing}
          className="w-full px-2 py-1 bg-red-600 hover:bg-red-700 rounded text-xs disabled:opacity-50"
        >
          {isClearing ? 'Clearing...' : 'Clear Cache'}
        </button>
      </div>
    </div>
  )
}

// ===== MAIN SERVICE WORKER PROVIDER =====

export const ServiceWorkerProvider: React.FC<ServiceWorkerProviderProps> = ({ children }) => {
  const [isInitialized, setIsInitialized] = useState(false)
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    // Mark as client-side
    setIsClient(true)

    const initialize = async () => {
      try {
        await initializeServiceWorker()
        setIsInitialized(true)
        console.log('✅ Service Worker Provider initialized')
      } catch (error) {
        console.error('❌ Service Worker initialization failed:', error)
        setIsInitialized(true) // Continue even if SW fails
      }
    }

    initialize()
  }, [])

  return (
    <>
      {children}

      {/* Service Worker UI Components - Only render on client */}
      {isClient && isInitialized && (
        <>
          <OfflineIndicator />
          <SyncStatusIndicator />
          <InstallPrompt />
          <CacheManager />
        </>
      )}
    </>
  )
}

export default ServiceWorkerProvider
