// Advanced undo/redo system for MasterX store

import type { RootState } from './types'
import { useStore } from './index'

// ===== UNDO/REDO TYPES =====

interface HistoryEntry {
  id: string
  timestamp: Date
  action: string
  previousState: Partial<RootState>
  currentState: Partial<RootState>
  metadata?: {
    description?: string
    category?: string
    undoable?: boolean
  }
}

interface UndoRedoState {
  history: HistoryEntry[]
  currentIndex: number
  maxHistorySize: number
  isUndoing: boolean
  isRedoing: boolean
}

interface UndoRedoOptions {
  maxHistorySize?: number
  excludeKeys?: (keyof RootState)[]
  includeKeys?: (keyof RootState)[]
  debounceMs?: number
}

// ===== UNDO/REDO MANAGER =====

export class UndoRedoManager {
  private static instance: UndoRedoManager
  private state: UndoRedoState
  private options: UndoRedoOptions
  private debounceTimer: NodeJS.Timeout | null = null
  private lastSnapshot: Partial<RootState> | null = null

  private constructor(options: UndoRedoOptions = {}) {
    this.options = {
      maxHistorySize: 50,
      excludeKeys: ['app'], // Don't track app state changes
      debounceMs: 500,
      ...options,
    }

    this.state = {
      history: [],
      currentIndex: -1,
      maxHistorySize: this.options.maxHistorySize!,
      isUndoing: false,
      isRedoing: false,
    }
  }

  public static getInstance(options?: UndoRedoOptions): UndoRedoManager {
    if (!UndoRedoManager.instance) {
      UndoRedoManager.instance = new UndoRedoManager(options)
    }
    return UndoRedoManager.instance
  }

  // ===== HISTORY TRACKING =====

  public trackChange(action: string, description?: string, category?: string): void {
    if (this.state.isUndoing || this.state.isRedoing) {
      return // Don't track changes during undo/redo operations
    }

    // Debounce rapid changes
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer)
    }

    this.debounceTimer = setTimeout(() => {
      this.captureSnapshot(action, description, category)
    }, this.options.debounceMs)
  }

  private captureSnapshot(action: string, description?: string, category?: string): void {
    const currentState = useStore.getState()
    const filteredState = this.filterState(currentState)

    // Don't create entry if state hasn't changed
    if (this.lastSnapshot && this.deepEqual(filteredState, this.lastSnapshot)) {
      return
    }

    const entry: HistoryEntry = {
      id: `history-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      action,
      previousState: this.lastSnapshot || {},
      currentState: filteredState,
      metadata: {
        description,
        category,
        undoable: true,
      },
    }

    // Remove any entries after current index (when creating new history after undo)
    if (this.state.currentIndex < this.state.history.length - 1) {
      this.state.history = this.state.history.slice(0, this.state.currentIndex + 1)
    }

    // Add new entry
    this.state.history.push(entry)
    this.state.currentIndex = this.state.history.length - 1

    // Maintain max history size
    if (this.state.history.length > this.state.maxHistorySize) {
      this.state.history = this.state.history.slice(-this.state.maxHistorySize)
      this.state.currentIndex = this.state.history.length - 1
    }

    this.lastSnapshot = filteredState
    console.log(`📝 History entry created: ${action}`)
  }

  // ===== UNDO/REDO OPERATIONS =====

  public undo(): boolean {
    if (!this.canUndo()) {
      return false
    }

    this.state.isUndoing = true
    
    try {
      const currentEntry = this.state.history[this.state.currentIndex]
      
      // Apply previous state
      this.applyState(currentEntry.previousState)
      
      // Move index back
      this.state.currentIndex--
      
      console.log(`↶ Undid: ${currentEntry.action}`)
      return true
    } catch (error) {
      console.error('Undo failed:', error)
      return false
    } finally {
      this.state.isUndoing = false
    }
  }

  public redo(): boolean {
    if (!this.canRedo()) {
      return false
    }

    this.state.isRedoing = true
    
    try {
      // Move index forward first
      this.state.currentIndex++
      
      const nextEntry = this.state.history[this.state.currentIndex]
      
      // Apply next state
      this.applyState(nextEntry.currentState)
      
      console.log(`↷ Redid: ${nextEntry.action}`)
      return true
    } catch (error) {
      console.error('Redo failed:', error)
      this.state.currentIndex-- // Revert index on error
      return false
    } finally {
      this.state.isRedoing = false
    }
  }

  public canUndo(): boolean {
    return this.state.currentIndex >= 0
  }

  public canRedo(): boolean {
    return this.state.currentIndex < this.state.history.length - 1
  }

  // ===== HISTORY MANAGEMENT =====

  public getHistory(): HistoryEntry[] {
    return [...this.state.history]
  }

  public getCurrentIndex(): number {
    return this.state.currentIndex
  }

  public clearHistory(): void {
    this.state.history = []
    this.state.currentIndex = -1
    this.lastSnapshot = null
    console.log('🗑️ History cleared')
  }

  public getHistoryStats(): {
    totalEntries: number
    currentIndex: number
    canUndo: boolean
    canRedo: boolean
    categories: Record<string, number>
  } {
    const categories: Record<string, number> = {}
    
    this.state.history.forEach(entry => {
      const category = entry.metadata?.category || 'uncategorized'
      categories[category] = (categories[category] || 0) + 1
    })

    return {
      totalEntries: this.state.history.length,
      currentIndex: this.state.currentIndex,
      canUndo: this.canUndo(),
      canRedo: this.canRedo(),
      categories,
    }
  }

  // ===== BRANCHING SUPPORT =====

  public createBranch(name: string): string {
    const branchId = `branch-${Date.now()}-${name}`
    
    // Save current history as a branch
    const branchData = {
      id: branchId,
      name,
      timestamp: new Date(),
      history: [...this.state.history],
      currentIndex: this.state.currentIndex,
    }
    
    localStorage.setItem(`masterx-branch-${branchId}`, JSON.stringify(branchData))
    console.log(`🌿 Branch created: ${name}`)
    
    return branchId
  }

  public switchToBranch(branchId: string): boolean {
    try {
      const branchData = localStorage.getItem(`masterx-branch-${branchId}`)
      if (!branchData) {
        throw new Error('Branch not found')
      }

      const branch = JSON.parse(branchData)
      
      // Save current state as a branch before switching
      this.createBranch('auto-save-before-switch')
      
      // Load branch history
      this.state.history = branch.history
      this.state.currentIndex = branch.currentIndex
      
      // Apply the branch's current state
      if (this.state.currentIndex >= 0) {
        const currentEntry = this.state.history[this.state.currentIndex]
        this.applyState(currentEntry.currentState)
      }
      
      console.log(`🌿 Switched to branch: ${branch.name}`)
      return true
    } catch (error) {
      console.error('Failed to switch branch:', error)
      return false
    }
  }

  // ===== UTILITY METHODS =====

  private filterState(state: RootState): Partial<RootState> {
    const filtered: Partial<RootState> = {}
    
    for (const key in state) {
      const stateKey = key as keyof RootState
      
      // Apply include/exclude filters
      if (this.options.includeKeys) {
        if (this.options.includeKeys.includes(stateKey)) {
          filtered[stateKey] = state[stateKey]
        }
      } else if (this.options.excludeKeys) {
        if (!this.options.excludeKeys.includes(stateKey)) {
          filtered[stateKey] = state[stateKey]
        }
      } else {
        filtered[stateKey] = state[stateKey]
      }
    }
    
    return filtered
  }

  private applyState(state: Partial<RootState>): void {
    const currentState = useStore.getState()
    const newState = { ...currentState, ...state }
    useStore.setState(newState)
    this.lastSnapshot = this.filterState(newState)
  }

  private deepEqual(obj1: any, obj2: any): boolean {
    if (obj1 === obj2) return true
    
    if (obj1 == null || obj2 == null) return false
    
    if (typeof obj1 !== typeof obj2) return false
    
    if (typeof obj1 !== 'object') return obj1 === obj2
    
    const keys1 = Object.keys(obj1)
    const keys2 = Object.keys(obj2)
    
    if (keys1.length !== keys2.length) return false
    
    for (const key of keys1) {
      if (!keys2.includes(key)) return false
      if (!this.deepEqual(obj1[key], obj2[key])) return false
    }
    
    return true
  }

  // ===== KEYBOARD SHORTCUTS =====

  public setupKeyboardShortcuts(): void {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z' && !e.shiftKey) {
          e.preventDefault()
          this.undo()
        } else if ((e.key === 'z' && e.shiftKey) || e.key === 'y') {
          e.preventDefault()
          this.redo()
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    
    // Return cleanup function
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }
}

// ===== REACT HOOKS =====

export const useUndoRedo = (options?: UndoRedoOptions) => {
  const manager = UndoRedoManager.getInstance(options)
  
  return {
    undo: manager.undo.bind(manager),
    redo: manager.redo.bind(manager),
    canUndo: manager.canUndo.bind(manager),
    canRedo: manager.canRedo.bind(manager),
    trackChange: manager.trackChange.bind(manager),
    clearHistory: manager.clearHistory.bind(manager),
    getHistory: manager.getHistory.bind(manager),
    getHistoryStats: manager.getHistoryStats.bind(manager),
    createBranch: manager.createBranch.bind(manager),
    switchToBranch: manager.switchToBranch.bind(manager),
    setupKeyboardShortcuts: manager.setupKeyboardShortcuts.bind(manager),
  }
}

export default UndoRedoManager
