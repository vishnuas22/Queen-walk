'use client'

import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Cpu, HardDrive, Wifi, AlertTriangle, CheckCircle } from 'lucide-react'
import { usePerformanceSelectors } from '../store/selectors'
import { useAppActions, useUIState, isStoreReady } from '../store'

// ===== PERFORMANCE MONITOR COMPONENT =====

export const PerformanceMonitor: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false)

  // Check if store is ready
  if (!isStoreReady()) {
    return null
  }

  // Safe store access with error handling
  let performanceMetrics, updatePerformanceMetrics, debugMode

  try {
    performanceMetrics = usePerformanceSelectors()
    const appActions = useAppActions()
    updatePerformanceMetrics = appActions.updatePerformanceMetrics
    const uiState = useUIState()
    debugMode = uiState.debugMode
  } catch (error) {
    console.warn('PerformanceMonitor: Store not ready, skipping initialization')
    return null
  }

  // Toggle visibility with keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'P') {
        e.preventDefault()
        setIsVisible(!isVisible)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isVisible])

  // Monitor performanceMetrics metrics
  useEffect(() => {
    if (!debugMode && !isVisible) return

    const measurePerformance = () => {
      // Measure memory usage
      if ('memory' in performanceMetrics) {
        const memory = (performanceMetrics as any).memory
        updatePerformanceMetrics({
          memoryUsage: memory.usedJSHeapSize / 1024 / 1024, // Convert to MB
        })
      }

      // Measure render time
      const renderStart = performance.now()
      requestAnimationFrame(() => {
        const renderTime = performance.now() - renderStart
        updatePerformanceMetrics({ renderTime })
      })
    }

    const interval = setInterval(measurePerformance, 2000)
    return () => clearInterval(interval)
  }, [debugMode, isVisible, updatePerformanceMetrics])

  if (!debugMode && !isVisible) return null

  return (
    <AnimatePresence>
      {(debugMode || isVisible) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          className="fixed bottom-4 left-4 z-50 bg-black/90 text-white p-4 rounded-lg shadow-xl backdrop-blur-sm max-w-sm"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold flex items-center">
              <Activity className="h-4 w-4 mr-2" />
              Performance Monitor
            </h3>
            <button
              onClick={() => setIsVisible(false)}
              className="text-gray-400 hover:text-white text-xs"
            >
              ✕
            </button>
          </div>

          <div className="space-y-3">
            {/* Overall Score */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-300">Overall Score</span>
              <div className="flex items-center">
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  performanceMetrics.score > 80 ? 'bg-green-500' :
                  performanceMetrics.score > 60 ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span className="text-sm font-mono">{performanceMetrics.score}/100</span>
              </div>
            </div>

            {/* Memory Usage */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-300 flex items-center">
                <HardDrive className="h-3 w-3 mr-1" />
                Memory
              </span>
              <div className="flex items-center">
                {performanceMetrics.memory.isHigh && (
                  <AlertTriangle className="h-3 w-3 text-yellow-500 mr-1" />
                )}
                <span className="text-sm font-mono">
                  {performanceMetrics.memory.current.toFixed(1)}MB
                </span>
              </div>
            </div>

            {/* Render Performance */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-300 flex items-center">
                <Cpu className="h-3 w-3 mr-1" />
                Render
              </span>
              <div className="flex items-center">
                {performanceMetrics.render.isSmooth ? (
                  <CheckCircle className="h-3 w-3 text-green-500 mr-1" />
                ) : (
                  <AlertTriangle className="h-3 w-3 text-yellow-500 mr-1" />
                )}
                <span className="text-sm font-mono">
                  {performanceMetrics.render.fps}fps
                </span>
              </div>
            </div>

            {/* Load Time */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-300">Load Time</span>
              <div className="flex items-center">
                {performanceMetrics.load.isFast ? (
                  <CheckCircle className="h-3 w-3 text-green-500 mr-1" />
                ) : performanceMetrics.load.isSlow ? (
                  <AlertTriangle className="h-3 w-3 text-red-500 mr-1" />
                ) : (
                  <AlertTriangle className="h-3 w-3 text-yellow-500 mr-1" />
                )}
                <span className="text-sm font-mono">
                  {performanceMetrics.load.time.toFixed(0)}ms
                </span>
              </div>
            </div>

            {/* Cache Hit Rate */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-300">Cache Hit</span>
              <div className="flex items-center">
                {performanceMetrics.cache.isEfficient ? (
                  <CheckCircle className="h-3 w-3 text-green-500 mr-1" />
                ) : (
                  <AlertTriangle className="h-3 w-3 text-yellow-500 mr-1" />
                )}
                <span className="text-sm font-mono">
                  {(performanceMetrics.cache.hitRate * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-gray-700">
            <div className="text-xs text-gray-400">
              Press Ctrl+Shift+P to toggle
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ===== PERFORMANCE ALERTS =====

export const PerformanceAlerts: React.FC = () => {
  const performanceMetrics = usePerformanceSelectors()
  const [alerts, setAlerts] = useState<string[]>([])

  useEffect(() => {
    const newAlerts: string[] = []

    if (performanceMetrics.memory.isHigh) {
      newAlerts.push('High memory usage detected')
    }

    if (!performanceMetrics.render.isSmooth) {
      newAlerts.push('Low frame rate detected')
    }

    if (performanceMetrics.load.isSlow) {
      newAlerts.push('Slow loading times detected')
    }

    if (!performanceMetrics.cache.isEfficient) {
      newAlerts.push('Low cache efficiency')
    }

    setAlerts(newAlerts)
  }, [performanceMetrics])

  if (alerts.length === 0) return null

  return (
    <div className="fixed top-20 right-4 z-40 space-y-2">
      <AnimatePresence>
        {alerts.map((alert, index) => (
          <motion.div
            key={alert}
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 100 }}
            transition={{ delay: index * 0.1 }}
            className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-2 rounded-lg shadow-sm text-sm"
          >
            <div className="flex items-center">
              <AlertTriangle className="h-4 w-4 mr-2" />
              {alert}
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

// ===== PERFORMANCE OPTIMIZATION SUGGESTIONS =====

export const PerformanceOptimizer: React.FC = () => {
  const performanceMetrics = usePerformanceSelectors()
  // const { setPerformanceMode } = useAppActions() // Temporarily disabled

  const suggestions = React.useMemo(() => {
    const suggestions: Array<{
      title: string
      description: string
      action: () => void
      severity: 'low' | 'medium' | 'high'
    }> = []

    if (performanceMetrics.memory.isHigh) {
      suggestions.push({
        title: 'Enable Performance Mode',
        description: 'Reduce memory usage by disabling animations and limiting features',
        action: () => {}, // setPerformanceMode(true) - temporarily disabled
        severity: 'high',
      })
    }

    if (!performanceMetrics.render.isSmooth) {
      suggestions.push({
        title: 'Reduce Visual Effects',
        description: 'Disable animations and transitions to improve frame rate',
        action: () => {}, // setPerformanceMode(true) - temporarily disabled
        severity: 'medium',
      })
    }

    if (performanceMetrics.load.isSlow) {
      suggestions.push({
        title: 'Clear Cache',
        description: 'Clear browser cache to improve loading times',
        action: () => {
          if ('caches' in window) {
            caches.keys().then(names => {
              names.forEach(name => caches.delete(name))
            })
          }
        },
        severity: 'medium',
      })
    }

    return suggestions
  }, [performanceMetrics])

  if (suggestions.length === 0) return null

  return (
    <div className="fixed bottom-20 right-4 z-40 bg-white border border-gray-200 rounded-lg shadow-lg p-4 max-w-sm">
      <h4 className="text-sm font-semibold text-gray-900 mb-3">
        Performance Suggestions
      </h4>
      
      <div className="space-y-3">
        {suggestions.map((suggestion, index) => (
          <div key={index} className="border-l-4 border-blue-500 pl-3">
            <h5 className="text-sm font-medium text-gray-900">
              {suggestion.title}
            </h5>
            <p className="text-xs text-gray-600 mt-1">
              {suggestion.description}
            </p>
            <button
              onClick={suggestion.action}
              className="mt-2 text-xs bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition-colors"
            >
              Apply
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

export default PerformanceMonitor
