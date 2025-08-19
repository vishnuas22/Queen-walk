// Advanced backup and restore system for MasterX store

import type { RootState } from './types'
import { useStore } from './index'

// ===== BACKUP TYPES =====

interface BackupMetadata {
  version: string
  timestamp: Date
  userAgent: string
  storeVersion: number
  checksum: string
}

interface StoreBackup {
  metadata: BackupMetadata
  state: Partial<RootState>
  compressed?: boolean
}

interface BackupOptions {
  includeChat?: boolean
  includeUser?: boolean
  includeUI?: boolean
  includeApp?: boolean
  compress?: boolean
  encrypt?: boolean
}

// ===== BACKUP MANAGER =====

export class BackupManager {
  private static instance: BackupManager
  private maxBackups = 10
  private backupPrefix = 'masterx-backup-'

  private constructor() {}

  public static getInstance(): BackupManager {
    if (!BackupManager.instance) {
      BackupManager.instance = new BackupManager()
    }
    return BackupManager.instance
  }

  // ===== BACKUP OPERATIONS =====

  public async createBackup(options: BackupOptions = {}): Promise<string> {
    const store = useStore.getState()
    
    // Default options
    const opts = {
      includeChat: false, // Don't backup chat by default (too large)
      includeUser: true,
      includeUI: true,
      includeApp: false,
      compress: true,
      encrypt: false,
      ...options,
    }

    // Build state to backup
    const stateToBackup: Partial<RootState> = {}
    
    if (opts.includeChat) stateToBackup.chat = store.chat
    if (opts.includeUser) stateToBackup.user = store.user
    if (opts.includeUI) stateToBackup.ui = store.ui
    if (opts.includeApp) stateToBackup.app = store.app

    // Create backup metadata
    const metadata: BackupMetadata = {
      version: '2.0.0',
      timestamp: new Date(),
      userAgent: navigator.userAgent,
      storeVersion: 2,
      checksum: await this.generateChecksum(stateToBackup),
    }

    // Create backup object
    const backup: StoreBackup = {
      metadata,
      state: stateToBackup,
      compressed: opts.compress,
    }

    // Compress if requested
    let backupData = JSON.stringify(backup)
    if (opts.compress) {
      backupData = await this.compressData(backupData)
    }

    // Generate backup ID
    const backupId = `${this.backupPrefix}${Date.now()}`
    
    // Store backup
    try {
      localStorage.setItem(backupId, backupData)
      
      // Manage backup count
      await this.cleanupOldBackups()
      
      console.log(`💾 Backup created: ${backupId}`)
      return backupId
    } catch (error) {
      console.error('Failed to create backup:', error)
      throw new Error('Failed to create backup')
    }
  }

  public async restoreBackup(backupId: string): Promise<boolean> {
    try {
      const backupData = localStorage.getItem(backupId)
      if (!backupData) {
        throw new Error('Backup not found')
      }

      // Decompress if needed
      let decompressedData = backupData
      try {
        const testParse = JSON.parse(backupData)
        if (testParse.compressed) {
          decompressedData = await this.decompressData(backupData)
        }
      } catch {
        // Try decompressing anyway
        decompressedData = await this.decompressData(backupData)
      }

      const backup: StoreBackup = JSON.parse(decompressedData)
      
      // Validate backup
      if (!this.validateBackup(backup)) {
        throw new Error('Invalid backup format')
      }

      // Verify checksum
      const currentChecksum = await this.generateChecksum(backup.state)
      if (currentChecksum !== backup.metadata.checksum) {
        console.warn('Backup checksum mismatch - data may be corrupted')
      }

      // Apply backup to store
      const store = useStore.getState()
      
      // Merge backup state with current state
      const newState = {
        ...store,
        ...backup.state,
      }

      // Use store's setState to apply the backup
      useStore.setState(newState)
      
      console.log(`📂 Backup restored: ${backupId}`)
      return true
    } catch (error) {
      console.error('Failed to restore backup:', error)
      return false
    }
  }

  public async exportBackup(backupId: string): Promise<void> {
    try {
      const backupData = localStorage.getItem(backupId)
      if (!backupData) {
        throw new Error('Backup not found')
      }

      const backup: StoreBackup = JSON.parse(backupData)
      const filename = `masterx-backup-${backup.metadata.timestamp.toISOString().split('T')[0]}.json`
      
      const blob = new Blob([backupData], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.click()
      
      URL.revokeObjectURL(url)
      console.log(`📤 Backup exported: ${filename}`)
    } catch (error) {
      console.error('Failed to export backup:', error)
      throw error
    }
  }

  public async importBackup(file: File): Promise<string> {
    try {
      const text = await file.text()
      const backup: StoreBackup = JSON.parse(text)
      
      if (!this.validateBackup(backup)) {
        throw new Error('Invalid backup file format')
      }

      // Generate new backup ID
      const backupId = `${this.backupPrefix}imported-${Date.now()}`
      
      // Store imported backup
      localStorage.setItem(backupId, text)
      
      console.log(`📥 Backup imported: ${backupId}`)
      return backupId
    } catch (error) {
      console.error('Failed to import backup:', error)
      throw error
    }
  }

  // ===== BACKUP MANAGEMENT =====

  public listBackups(): Array<{ id: string; metadata: BackupMetadata }> {
    const backups: Array<{ id: string; metadata: BackupMetadata }> = []
    
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith(this.backupPrefix)) {
        try {
          const data = localStorage.getItem(key)
          if (data) {
            const backup: StoreBackup = JSON.parse(data)
            backups.push({
              id: key,
              metadata: backup.metadata,
            })
          }
        } catch (error) {
          console.warn(`Invalid backup found: ${key}`)
        }
      }
    }
    
    // Sort by timestamp (newest first)
    return backups.sort((a, b) => 
      new Date(b.metadata.timestamp).getTime() - new Date(a.metadata.timestamp).getTime()
    )
  }

  public async deleteBackup(backupId: string): Promise<boolean> {
    try {
      localStorage.removeItem(backupId)
      console.log(`🗑️ Backup deleted: ${backupId}`)
      return true
    } catch (error) {
      console.error('Failed to delete backup:', error)
      return false
    }
  }

  private async cleanupOldBackups(): Promise<void> {
    const backups = this.listBackups()
    
    if (backups.length > this.maxBackups) {
      const toDelete = backups.slice(this.maxBackups)
      
      for (const backup of toDelete) {
        await this.deleteBackup(backup.id)
      }
    }
  }

  // ===== UTILITY METHODS =====

  private validateBackup(backup: any): backup is StoreBackup {
    return (
      backup &&
      typeof backup === 'object' &&
      backup.metadata &&
      backup.state &&
      typeof backup.metadata.version === 'string' &&
      typeof backup.metadata.timestamp === 'string' &&
      typeof backup.metadata.storeVersion === 'number'
    )
  }

  private async generateChecksum(data: any): Promise<string> {
    const text = JSON.stringify(data)
    const encoder = new TextEncoder()
    const dataBuffer = encoder.encode(text)
    
    if (crypto.subtle) {
      const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer)
      const hashArray = Array.from(new Uint8Array(hashBuffer))
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
    } else {
      // Fallback for environments without crypto.subtle
      let hash = 0
      for (let i = 0; i < text.length; i++) {
        const char = text.charCodeAt(i)
        hash = ((hash << 5) - hash) + char
        hash = hash & hash // Convert to 32-bit integer
      }
      return hash.toString(16)
    }
  }

  private async compressData(data: string): Promise<string> {
    // Simple compression using built-in compression
    if ('CompressionStream' in window) {
      const stream = new CompressionStream('gzip')
      const writer = stream.writable.getWriter()
      const reader = stream.readable.getReader()
      
      writer.write(new TextEncoder().encode(data))
      writer.close()
      
      const chunks: Uint8Array[] = []
      let done = false
      
      while (!done) {
        const { value, done: readerDone } = await reader.read()
        done = readerDone
        if (value) chunks.push(value)
      }
      
      const compressed = new Uint8Array(chunks.reduce((acc, chunk) => acc + chunk.length, 0))
      let offset = 0
      for (const chunk of chunks) {
        compressed.set(chunk, offset)
        offset += chunk.length
      }
      
      return btoa(String.fromCharCode(...compressed))
    } else {
      // Fallback: just return the original data
      return data
    }
  }

  private async decompressData(data: string): Promise<string> {
    // Simple decompression
    if ('DecompressionStream' in window) {
      try {
        const compressed = Uint8Array.from(atob(data), c => c.charCodeAt(0))
        const stream = new DecompressionStream('gzip')
        const writer = stream.writable.getWriter()
        const reader = stream.readable.getReader()
        
        writer.write(compressed)
        writer.close()
        
        const chunks: Uint8Array[] = []
        let done = false
        
        while (!done) {
          const { value, done: readerDone } = await reader.read()
          done = readerDone
          if (value) chunks.push(value)
        }
        
        const decompressed = new Uint8Array(chunks.reduce((acc, chunk) => acc + chunk.length, 0))
        let offset = 0
        for (const chunk of chunks) {
          decompressed.set(chunk, offset)
          offset += chunk.length
        }
        
        return new TextDecoder().decode(decompressed)
      } catch (error) {
        console.warn('Decompression failed, returning original data')
        return data
      }
    } else {
      // Fallback: assume data is not compressed
      return data
    }
  }
}

// ===== REACT HOOKS =====

export const useBackup = () => {
  const backupManager = BackupManager.getInstance()
  
  return {
    createBackup: backupManager.createBackup.bind(backupManager),
    restoreBackup: backupManager.restoreBackup.bind(backupManager),
    exportBackup: backupManager.exportBackup.bind(backupManager),
    importBackup: backupManager.importBackup.bind(backupManager),
    listBackups: backupManager.listBackups.bind(backupManager),
    deleteBackup: backupManager.deleteBackup.bind(backupManager),
  }
}

export default BackupManager
