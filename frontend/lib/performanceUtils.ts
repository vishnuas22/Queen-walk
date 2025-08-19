// Performance testing utilities for virtual scrolling

interface Message {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  isStreaming?: boolean
}

// Generate test messages for performance testing
export const generateTestMessages = (count: number): Message[] => {
  const messages: Message[] = []
  const sampleContents = [
    // Short messages
    "Hello!",
    "How are you?",
    "Thanks for your help.",
    "That's great!",
    
    // Medium messages
    "I'm working on a React project and need help with state management. Can you explain the differences between useState and useReducer?",
    "Could you help me understand how virtual scrolling works? I'm trying to optimize a chat interface with thousands of messages.",
    "What are the best practices for implementing animations in React? I want to make my UI feel more responsive and delightful.",
    
    // Long messages with code
    `Here's a complex React component that demonstrates advanced patterns:

\`\`\`typescript
import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface Props {
  data: any[]
  onSelect: (item: any) => void
  loading?: boolean
}

export const AdvancedComponent: React.FC<Props> = ({ data, onSelect, loading }) => {
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set())
  const [searchTerm, setSearchTerm] = useState('')
  
  const filteredData = useMemo(() => {
    return data.filter(item => 
      item.name.toLowerCase().includes(searchTerm.toLowerCase())
    )
  }, [data, searchTerm])
  
  const handleSelect = useCallback((item: any) => {
    setSelectedItems(prev => {
      const newSet = new Set(prev)
      if (newSet.has(item.id)) {
        newSet.delete(item.id)
      } else {
        newSet.add(item.id)
      }
      return newSet
    })
    onSelect(item)
  }, [onSelect])
  
  return (
    <div className="advanced-component">
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search..."
      />
      
      <AnimatePresence>
        {filteredData.map(item => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            onClick={() => handleSelect(item)}
            className={selectedItems.has(item.id) ? 'selected' : ''}
          >
            {item.name}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
\`\`\`

This component demonstrates several important concepts:
1. **State Management**: Using useState for local state and useMemo for derived state
2. **Performance Optimization**: Memoizing expensive calculations and using useCallback
3. **Animation**: Smooth enter/exit animations with Framer Motion
4. **User Interaction**: Handling selection state and search functionality

The key to good performance is minimizing re-renders and using the right hooks for the right purposes.`,

    // AI-style responses with markdown
    `# Understanding Virtual Scrolling

Virtual scrolling is a technique used to efficiently render large lists by only rendering the items that are currently visible in the viewport. Here's how it works:

## Key Concepts

1. **Viewport**: The visible area where items are displayed
2. **Buffer**: Extra items rendered outside the viewport for smooth scrolling
3. **Item Height**: Either fixed or dynamic height calculation
4. **Scroll Position**: Tracking where the user is in the list

## Implementation Steps

### 1. Calculate Visible Range
\`\`\`javascript
const startIndex = Math.floor(scrollTop / itemHeight)
const endIndex = Math.min(startIndex + visibleCount + buffer, totalItems)
\`\`\`

### 2. Render Only Visible Items
\`\`\`javascript
const visibleItems = items.slice(startIndex, endIndex)
\`\`\`

### 3. Handle Scroll Events
\`\`\`javascript
const handleScroll = (event) => {
  const scrollTop = event.target.scrollTop
  updateVisibleRange(scrollTop)
}
\`\`\`

## Benefits

- **Performance**: Only renders what's needed
- **Memory**: Reduces DOM nodes
- **Smooth Scrolling**: Maintains 60fps even with thousands of items

## Challenges

- **Dynamic Heights**: Harder to calculate positions
- **Scroll Position**: Maintaining position during updates
- **Accessibility**: Ensuring screen readers work correctly

Virtual scrolling is essential for applications dealing with large datasets while maintaining excellent user experience.`
  ]
  
  for (let i = 0; i < count; i++) {
    const isUser = i % 3 === 0 // Every 3rd message is from user
    const contentIndex = Math.floor(Math.random() * sampleContents.length)
    const baseContent = sampleContents[contentIndex]
    
    // Add some variation to content
    const variation = Math.random()
    let content = baseContent
    
    if (variation > 0.8) {
      // Add extra content for some messages
      content += `\n\nAdditional context for message ${i + 1}. This helps test variable height calculations and ensures our virtual scrolling handles different message sizes correctly.`
    }
    
    messages.push({
      id: `test-message-${i}`,
      content,
      sender: isUser ? 'user' : 'ai',
      timestamp: new Date(Date.now() - (count - i) * 60000), // Messages 1 minute apart
      isStreaming: false
    })
  }
  
  return messages
}

// Performance measurement utilities
export class PerformanceMonitor {
  private measurements: Map<string, number[]> = new Map()
  
  startMeasurement(key: string): void {
    const start = performance.now()
    if (!this.measurements.has(key)) {
      this.measurements.set(key, [])
    }
    this.measurements.get(key)!.push(start)
  }
  
  endMeasurement(key: string): number {
    const measurements = this.measurements.get(key)
    if (!measurements || measurements.length === 0) {
      console.warn(`No start measurement found for key: ${key}`)
      return 0
    }
    
    const start = measurements.pop()!
    const duration = performance.now() - start
    
    return duration
  }
  
  getAverageTime(key: string, sampleSize: number = 10): number {
    const measurements = this.measurements.get(key)
    if (!measurements || measurements.length === 0) {
      return 0
    }
    
    const recentMeasurements = measurements.slice(-sampleSize)
    const total = recentMeasurements.reduce((sum, time) => sum + time, 0)
    return total / recentMeasurements.length
  }
  
  logPerformanceReport(): void {
    console.group('🚀 Performance Report')
    
    for (const [key, measurements] of this.measurements.entries()) {
      if (measurements.length > 0) {
        const avg = this.getAverageTime(key)
        const min = Math.min(...measurements)
        const max = Math.max(...measurements)
        
        console.log(`${key}:`, {
          average: `${avg.toFixed(2)}ms`,
          min: `${min.toFixed(2)}ms`,
          max: `${max.toFixed(2)}ms`,
          samples: measurements.length
        })
      }
    }
    
    console.groupEnd()
  }
  
  clear(): void {
    this.measurements.clear()
  }
}

// Global performance monitor instance
export const performanceMonitor = new PerformanceMonitor()

// FPS monitoring
export class FPSMonitor {
  private frameCount = 0
  private lastTime = performance.now()
  private fps = 0
  private isRunning = false
  
  start(): void {
    if (this.isRunning) return
    
    this.isRunning = true
    this.frameCount = 0
    this.lastTime = performance.now()
    this.measureFPS()
  }
  
  stop(): void {
    this.isRunning = false
  }
  
  private measureFPS(): void {
    if (!this.isRunning) return
    
    this.frameCount++
    const currentTime = performance.now()
    
    if (currentTime - this.lastTime >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime))
      this.frameCount = 0
      this.lastTime = currentTime
      
      // Log FPS if it drops below 50
      if (this.fps < 50) {
        console.warn(`⚠️ Low FPS detected: ${this.fps}fps`)
      }
    }
    
    requestAnimationFrame(() => this.measureFPS())
  }
  
  getCurrentFPS(): number {
    return this.fps
  }
}

// Memory usage monitoring
export const getMemoryUsage = (): any => {
  if ('memory' in performance) {
    return {
      used: Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024),
      total: Math.round((performance as any).memory.totalJSHeapSize / 1024 / 1024),
      limit: Math.round((performance as any).memory.jsHeapSizeLimit / 1024 / 1024)
    }
  }
  return null
}

// Scroll performance testing
export const testScrollPerformance = (element: HTMLElement, duration: number = 5000): Promise<any> => {
  return new Promise((resolve) => {
    const fpsMonitor = new FPSMonitor()
    const startTime = performance.now()
    const initialMemory = getMemoryUsage()
    
    fpsMonitor.start()
    
    // Simulate rapid scrolling
    let scrollPosition = 0
    const maxScroll = element.scrollHeight - element.clientHeight
    const scrollSpeed = maxScroll / (duration / 16) // 60fps
    
    const scrollTest = () => {
      const elapsed = performance.now() - startTime
      
      if (elapsed < duration) {
        scrollPosition += scrollSpeed
        if (scrollPosition > maxScroll) {
          scrollPosition = 0 // Reset to top
        }
        
        element.scrollTop = scrollPosition
        requestAnimationFrame(scrollTest)
      } else {
        fpsMonitor.stop()
        const finalMemory = getMemoryUsage()
        
        resolve({
          averageFPS: fpsMonitor.getCurrentFPS(),
          memoryUsage: {
            initial: initialMemory,
            final: finalMemory,
            difference: finalMemory ? finalMemory.used - initialMemory.used : null
          },
          duration,
          scrollDistance: maxScroll
        })
      }
    }
    
    requestAnimationFrame(scrollTest)
  })
}
