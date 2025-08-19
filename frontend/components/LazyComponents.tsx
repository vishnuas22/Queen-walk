// Lazy-loaded components for bundle optimization

import React from 'react'
import dynamic from 'next/dynamic'
import { ComponentType } from 'react'

// ===== LOADING COMPONENTS =====

const LoadingSpinner = () => (
  <div className="flex items-center justify-center p-8">
    <div className="w-8 h-8 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin"></div>
  </div>
)

const LoadingCard = () => (
  <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 animate-pulse">
    <div className="h-4 bg-slate-200 rounded w-3/4 mb-3"></div>
    <div className="h-4 bg-slate-200 rounded w-1/2 mb-3"></div>
    <div className="h-4 bg-slate-200 rounded w-5/6"></div>
  </div>
)

const LoadingMessageList = () => (
  <div className="space-y-4 p-4">
    {[...Array(3)].map((_, i) => (
      <div key={i} className="flex space-x-3">
        <div className="w-8 h-8 bg-slate-200 rounded-full animate-pulse"></div>
        <div className="flex-1">
          <div className="h-4 bg-slate-200 rounded w-1/4 mb-2 animate-pulse"></div>
          <div className="h-16 bg-slate-200 rounded animate-pulse"></div>
        </div>
      </div>
    ))}
  </div>
)

// ===== HEAVY COMPONENTS WITH DYNAMIC IMPORTS =====

// Virtualized Message List - Heavy component with react-window
export const LazyVirtualizedMessageList = dynamic(
  () => import('./VirtualizedMessageList'),
  {
    loading: LoadingMessageList,
    ssr: false, // Disable SSR for performance components
  }
)

// Markdown Renderer - Heavy component with syntax highlighting
export const LazyMarkdownRenderer = dynamic(
  () => import('./MarkdownRenderer'),
  {
    loading: () => (
      <div className="bg-slate-100 rounded p-4 animate-pulse">
        <div className="h-4 bg-slate-200 rounded w-full mb-2"></div>
        <div className="h-4 bg-slate-200 rounded w-3/4"></div>
      </div>
    ),
    ssr: false,
  }
)

// Monaco Editor - Very heavy component
export const LazyMonacoEditor = dynamic(
  () => import('@monaco-editor/react').then(mod => ({ default: mod.Editor })),
  {
    loading: () => (
      <div className="w-full h-64 bg-slate-100 rounded border-2 border-dashed border-slate-300 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-slate-600 text-sm">Loading Code Editor...</p>
        </div>
      </div>
    ),
    ssr: false,
  }
)

// React Query DevTools - Development only, heavy component
export const LazyReactQueryDevtools = dynamic(
  () => import('@tanstack/react-query-devtools').then(mod => ({ default: mod.ReactQueryDevtools })),
  {
    loading: LoadingSpinner,
    ssr: false,
  }
)

// ===== FUTURE COMPONENTS (Placeholder for when they're created) =====

// These components will be created in future phases
// For now, they're commented out to avoid build errors

/*
// Chart components - Heavy visualization libraries
export const LazyPerformanceChart = dynamic(
  () => import('./PerformanceChart'),
  {
    loading: LoadingCard,
    ssr: false,
  }
)

// File upload component with drag & drop
export const LazyFileUpload = dynamic(
  () => import('./FileUpload'),
  {
    loading: () => (
      <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center animate-pulse">
        <div className="w-12 h-12 bg-slate-200 rounded mx-auto mb-4"></div>
        <div className="h-4 bg-slate-200 rounded w-1/2 mx-auto"></div>
      </div>
    ),
    ssr: false,
  }
)

// Settings page components
export const LazySettingsPanel = dynamic(
  () => import('./SettingsPanel'),
  {
    loading: LoadingCard,
    ssr: false,
  }
)

// Profile management
export const LazyProfileManager = dynamic(
  () => import('./ProfileManager'),
  {
    loading: LoadingCard,
    ssr: false,
  }
)

// Analytics dashboard
export const LazyAnalyticsDashboard = dynamic(
  () => import('./AnalyticsDashboard'),
  {
    loading: () => (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[...Array(6)].map((_, i) => (
          <LoadingCard key={i} />
        ))}
      </div>
    ),
    ssr: false,
  }
)
*/

// ===== UTILITY FUNCTIONS =====

// Preload component for better UX
export const preloadComponent = (componentImport: () => Promise<any>) => {
  if (typeof window !== 'undefined') {
    // Preload on user interaction or idle time
    const preload = () => componentImport()
    
    // Preload on mouse enter or focus events
    return {
      onMouseEnter: preload,
      onFocus: preload,
    }
  }
  return {}
}

// Intersection observer for lazy loading
export const useLazyLoad = (ref: React.RefObject<HTMLElement>, callback: () => void) => {
  React.useEffect(() => {
    const element = ref.current
    if (!element) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          callback()
          observer.unobserve(element)
        }
      },
      { threshold: 0.1 }
    )

    observer.observe(element)
    return () => observer.disconnect()
  }, [callback])
}

// Bundle size analyzer helper
export const getBundleInfo = () => {
  if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
    return {
      // Estimate current bundle size based on loaded modules
      loadedModules: Object.keys(window as any).filter(key => 
        key.startsWith('__webpack') || key.startsWith('__next')
      ).length,
      
      // Performance metrics
      navigationTiming: performance.getEntriesByType('navigation')[0],
      
      // Memory usage (if available)
      memory: (performance as any).memory,
    }
  }
  return null
}

// ===== EXPORT TYPES =====

export type LazyComponentProps = {
  loading?: ComponentType
  error?: ComponentType<{ error: Error; retry: () => void }>
}

export type PreloadableComponent<T = {}> = ComponentType<T> & {
  preload?: () => Promise<void>
}

// ===== DEVELOPMENT HELPERS =====

if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  // Log bundle loading in development
  const originalImport = (window as any).__webpack_require__
  if (originalImport) {
    (window as any).__webpack_require__ = function(...args: any[]) {
      console.log('📦 Loading chunk:', args[0])
      return originalImport.apply(this, args)
    }
  }
}
