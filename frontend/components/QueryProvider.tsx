'use client'

import React, { useState, useEffect } from 'react'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient, devUtils } from '../lib/queryClient'
import { LazyReactQueryDevtools } from './LazyComponents'

interface QueryProviderProps {
  children: React.ReactNode
}

// ===== NETWORK STATUS MONITORING =====

const useNetworkStatus = () => {
  const [isOnline, setIsOnline] = useState(true)
  
  useEffect(() => {
    const updateOnlineStatus = () => {
      setIsOnline(navigator.onLine)
      
      if (navigator.onLine) {
        console.log('🌐 Network connection restored')
        // Refetch all queries when coming back online
        queryClient.refetchQueries()
      } else {
        console.log('📴 Network connection lost')
      }
    }
    
    window.addEventListener('online', updateOnlineStatus)
    window.addEventListener('offline', updateOnlineStatus)
    
    return () => {
      window.removeEventListener('online', updateOnlineStatus)
      window.removeEventListener('offline', updateOnlineStatus)
    }
  }, [])
  
  return isOnline
}

// ===== CACHE MONITORING =====

const useCacheMonitoring = () => {
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      // Log cache stats periodically in development
      const interval = setInterval(() => {
        const stats = devUtils.getCacheStats()
        if (stats.totalQueries > 0) {
          console.log('📊 Cache Stats:', stats)
        }
      }, 30000) // Every 30 seconds
      
      return () => clearInterval(interval)
    }
  }, [])
}

// ===== ERROR BOUNDARY =====

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
}

class QueryErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }
  
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }
  
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Query Error Boundary caught an error:', error, errorInfo)
    
    // Log error to monitoring service in production
    if (process.env.NODE_ENV === 'production') {
      // TODO: Send to error monitoring service
    }
  }
  
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
          <div className="max-w-md mx-auto text-center p-6">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-slate-900 mb-2">
              Something went wrong
            </h2>
            <p className="text-slate-600 mb-4">
              We encountered an error while loading the application. Please try refreshing the page.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Refresh Page
            </button>
            
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mt-4 text-left">
                <summary className="cursor-pointer text-sm text-slate-500">
                  Error Details (Development)
                </summary>
                <pre className="mt-2 p-3 bg-slate-100 rounded text-xs overflow-auto">
                  {this.state.error.stack}
                </pre>
              </details>
            )}
          </div>
        </div>
      )
    }
    
    return this.props.children
  }
}

// ===== NETWORK STATUS INDICATOR =====

const NetworkStatusIndicator: React.FC<{ isOnline: boolean }> = ({ isOnline }) => {
  if (isOnline) return null
  
  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-red-600 text-white text-center py-2 text-sm">
      <div className="flex items-center justify-center space-x-2">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636l-12.728 12.728m0-12.728l12.728 12.728" />
        </svg>
        <span>You're offline. Some features may not be available.</span>
      </div>
    </div>
  )
}

// ===== MAIN QUERY PROVIDER =====

export const QueryProvider: React.FC<QueryProviderProps> = ({ children }) => {
  const isOnline = useNetworkStatus()
  useCacheMonitoring()
  
  // Development-only cache management
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      // Expose cache utilities to window for debugging
      ;(window as any).queryCache = {
        client: queryClient,
        log: devUtils.logCache,
        clear: devUtils.clearCache,
        stats: devUtils.getCacheStats,
      }
      
      console.log('🔧 Query cache utilities available at window.queryCache')
    }
  }, [])
  
  return (
    <QueryErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <NetworkStatusIndicator isOnline={isOnline} />
        {children}
        
        {/* React Query DevTools - Development Only */}
        {process.env.NODE_ENV === 'development' && (
          <LazyReactQueryDevtools initialIsOpen={false} />
        )}
      </QueryClientProvider>
    </QueryErrorBoundary>
  )
}

export default QueryProvider
