// Bundle analysis and performance monitoring utilities

// ===== TYPES =====

interface BundleMetrics {
  totalSize: number
  gzippedSize: number
  chunks: ChunkInfo[]
  loadTime: number
  cacheHitRate: number
}

interface ChunkInfo {
  name: string
  size: number
  loadTime: number
  cached: boolean
}

interface PerformanceMetrics {
  fcp: number // First Contentful Paint
  lcp: number // Largest Contentful Paint
  fid: number // First Input Delay
  cls: number // Cumulative Layout Shift
  ttfb: number // Time to First Byte
}

// ===== BUNDLE SIZE ANALYZER =====

export class BundleAnalyzer {
  private metrics: BundleMetrics | null = null
  private observers: PerformanceObserver[] = []

  constructor() {
    if (typeof window !== 'undefined') {
      this.initializeObservers()
    }
  }

  private initializeObservers() {
    // Observe navigation timing
    if ('PerformanceObserver' in window) {
      const navObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        entries.forEach((entry) => {
          if (entry.entryType === 'navigation') {
            this.processNavigationEntry(entry as PerformanceNavigationTiming)
          }
        })
      })
      
      navObserver.observe({ entryTypes: ['navigation'] })
      this.observers.push(navObserver)

      // Observe resource timing
      const resourceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        entries.forEach((entry) => {
          if (entry.entryType === 'resource') {
            this.processResourceEntry(entry as PerformanceResourceTiming)
          }
        })
      })
      
      resourceObserver.observe({ entryTypes: ['resource'] })
      this.observers.push(resourceObserver)
    }
  }

  private processNavigationEntry(entry: PerformanceNavigationTiming) {
    console.log('📊 Navigation Timing:', {
      domContentLoaded: entry.domContentLoadedEventEnd - entry.domContentLoadedEventStart,
      loadComplete: entry.loadEventEnd - entry.loadEventStart,
      ttfb: entry.responseStart - entry.requestStart,
      domInteractive: entry.domInteractive - entry.navigationStart,
    })
  }

  private processResourceEntry(entry: PerformanceResourceTiming) {
    if (entry.name.includes('/_next/static/')) {
      const isFromCache = entry.transferSize === 0
      console.log('📦 Chunk Loaded:', {
        name: entry.name.split('/').pop(),
        size: entry.transferSize,
        loadTime: entry.responseEnd - entry.requestStart,
        cached: isFromCache,
      })
    }
  }

  // Analyze current bundle performance
  public async analyzeBundlePerformance(): Promise<BundleMetrics> {
    const chunks = await this.getChunkInfo()
    const totalSize = chunks.reduce((sum, chunk) => sum + chunk.size, 0)
    const avgLoadTime = chunks.reduce((sum, chunk) => sum + chunk.loadTime, 0) / chunks.length
    const cacheHitRate = chunks.filter(chunk => chunk.cached).length / chunks.length

    this.metrics = {
      totalSize,
      gzippedSize: totalSize * 0.7, // Estimate
      chunks,
      loadTime: avgLoadTime,
      cacheHitRate,
    }

    return this.metrics
  }

  private async getChunkInfo(): Promise<ChunkInfo[]> {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
    
    return resources
      .filter(resource => resource.name.includes('/_next/static/'))
      .map(resource => ({
        name: resource.name.split('/').pop() || 'unknown',
        size: resource.transferSize || resource.encodedBodySize || 0,
        loadTime: resource.responseEnd - resource.requestStart,
        cached: resource.transferSize === 0,
      }))
  }

  // Get Core Web Vitals
  public async getCoreWebVitals(): Promise<PerformanceMetrics> {
    return new Promise((resolve) => {
      const metrics: Partial<PerformanceMetrics> = {}

      // First Contentful Paint
      const fcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        const fcp = entries.find(entry => entry.name === 'first-contentful-paint')
        if (fcp) {
          metrics.fcp = fcp.startTime
          fcpObserver.disconnect()
        }
      })
      fcpObserver.observe({ entryTypes: ['paint'] })

      // Largest Contentful Paint
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        const lastEntry = entries[entries.length - 1]
        metrics.lcp = lastEntry.startTime
      })
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] })

      // First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        entries.forEach((entry: any) => {
          metrics.fid = entry.processingStart - entry.startTime
        })
        fidObserver.disconnect()
      })
      fidObserver.observe({ entryTypes: ['first-input'] })

      // Cumulative Layout Shift
      let clsValue = 0
      const clsObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        entries.forEach((entry: any) => {
          if (!entry.hadRecentInput) {
            clsValue += entry.value
          }
        })
        metrics.cls = clsValue
      })
      clsObserver.observe({ entryTypes: ['layout-shift'] })

      // Time to First Byte
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
      if (navigation) {
        metrics.ttfb = navigation.responseStart - navigation.requestStart
      }

      // Resolve after a delay to collect metrics
      setTimeout(() => {
        resolve(metrics as PerformanceMetrics)
      }, 3000)
    })
  }

  // Generate performance report
  public generateReport(): string {
    if (!this.metrics) {
      return 'No metrics available. Run analyzeBundlePerformance() first.'
    }

    const { totalSize, chunks, loadTime, cacheHitRate } = this.metrics

    return `
📊 Bundle Performance Report
============================

📦 Bundle Size:
   Total: ${(totalSize / 1024).toFixed(2)} KB
   Chunks: ${chunks.length}
   
⚡ Performance:
   Avg Load Time: ${loadTime.toFixed(2)}ms
   Cache Hit Rate: ${(cacheHitRate * 100).toFixed(1)}%
   
📋 Chunk Details:
${chunks.map(chunk => 
  `   ${chunk.name}: ${(chunk.size / 1024).toFixed(2)} KB ${chunk.cached ? '(cached)' : ''}`
).join('\n')}

🎯 Recommendations:
${this.generateRecommendations()}
    `.trim()
  }

  private generateRecommendations(): string {
    if (!this.metrics) return 'No recommendations available.'

    const recommendations: string[] = []
    const { totalSize, cacheHitRate, loadTime } = this.metrics

    if (totalSize > 1024 * 1024) { // > 1MB
      recommendations.push('   • Consider further code splitting for large bundles')
    }

    if (cacheHitRate < 0.8) {
      recommendations.push('   • Improve caching strategy for better performance')
    }

    if (loadTime > 1000) {
      recommendations.push('   • Optimize chunk loading for faster initial load')
    }

    if (recommendations.length === 0) {
      recommendations.push('   • Bundle performance looks good! 🎉')
    }

    return recommendations.join('\n')
  }

  // Cleanup observers
  public destroy() {
    this.observers.forEach(observer => observer.disconnect())
    this.observers = []
  }
}

// ===== BUNDLE SIZE COMPARISON =====

export const compareBundleSizes = (before: BundleMetrics, after: BundleMetrics) => {
  const sizeDiff = after.totalSize - before.totalSize
  const loadTimeDiff = after.loadTime - before.loadTime
  const cacheRateDiff = after.cacheHitRate - before.cacheHitRate

  return {
    sizeChange: {
      absolute: sizeDiff,
      percentage: (sizeDiff / before.totalSize) * 100,
    },
    loadTimeChange: {
      absolute: loadTimeDiff,
      percentage: (loadTimeDiff / before.loadTime) * 100,
    },
    cacheRateChange: {
      absolute: cacheRateDiff,
      percentage: (cacheRateDiff / before.cacheHitRate) * 100,
    },
  }
}

// ===== PERFORMANCE MONITORING =====

export class PerformanceMonitor {
  private static instance: PerformanceMonitor
  private bundleAnalyzer: BundleAnalyzer
  private isMonitoring = false

  private constructor() {
    this.bundleAnalyzer = new BundleAnalyzer()
  }

  public static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor()
    }
    return PerformanceMonitor.instance
  }

  public startMonitoring() {
    if (this.isMonitoring) return

    this.isMonitoring = true
    console.log('🔍 Performance monitoring started')

    // Monitor every 30 seconds
    setInterval(async () => {
      const metrics = await this.bundleAnalyzer.analyzeBundlePerformance()
      const webVitals = await this.bundleAnalyzer.getCoreWebVitals()
      
      this.logMetrics(metrics, webVitals)
    }, 30000)
  }

  private logMetrics(bundle: BundleMetrics, vitals: PerformanceMetrics) {
    console.group('📊 Performance Metrics')
    console.log('Bundle Size:', (bundle.totalSize / 1024).toFixed(2), 'KB')
    console.log('Cache Hit Rate:', (bundle.cacheHitRate * 100).toFixed(1), '%')
    console.log('FCP:', vitals.fcp?.toFixed(2), 'ms')
    console.log('LCP:', vitals.lcp?.toFixed(2), 'ms')
    console.log('FID:', vitals.fid?.toFixed(2), 'ms')
    console.log('CLS:', vitals.cls?.toFixed(4))
    console.groupEnd()
  }

  public stopMonitoring() {
    this.isMonitoring = false
    this.bundleAnalyzer.destroy()
    console.log('⏹️ Performance monitoring stopped')
  }
}

// ===== GLOBAL UTILITIES =====

// Initialize performance monitoring in development
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  const monitor = PerformanceMonitor.getInstance()
  monitor.startMonitoring()

  // Expose to window for debugging
  ;(window as any).bundleAnalyzer = new BundleAnalyzer()
  ;(window as any).performanceMonitor = monitor
}

export default BundleAnalyzer
