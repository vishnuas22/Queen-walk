'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Undo2, 
  Redo2, 
  Save, 
  Download, 
  Upload, 
  History, 
  Settings,
  Trash2,
  GitBranch,
  Clock
} from 'lucide-react'
import { useBackup } from '../store/backup'
import { useUndoRedo } from '../store/undoRedo'
import { useUIActions } from '../store'

// ===== UNDO/REDO TOOLBAR =====

export const UndoRedoToolbar: React.FC = () => {
  const { undo, redo, canUndo, canRedo, getHistoryStats } = useUndoRedo()
  const [stats, setStats] = useState(getHistoryStats())

  useEffect(() => {
    const interval = setInterval(() => {
      setStats(getHistoryStats())
    }, 1000)

    return () => clearInterval(interval)
  }, [getHistoryStats])

  return (
    <div className="fixed bottom-4 right-20 z-40 bg-white border border-gray-200 rounded-lg shadow-lg p-2 flex items-center space-x-2">
      <button
        onClick={undo}
        disabled={!canUndo()}
        className={`p-2 rounded transition-colors ${
          canUndo() 
            ? 'text-gray-700 hover:bg-gray-100' 
            : 'text-gray-300 cursor-not-allowed'
        }`}
        title={`Undo (${stats.totalEntries} actions)`}
      >
        <Undo2 className="h-4 w-4" />
      </button>

      <button
        onClick={redo}
        disabled={!canRedo()}
        className={`p-2 rounded transition-colors ${
          canRedo() 
            ? 'text-gray-700 hover:bg-gray-100' 
            : 'text-gray-300 cursor-not-allowed'
        }`}
        title="Redo"
      >
        <Redo2 className="h-4 w-4" />
      </button>

      <div className="w-px h-6 bg-gray-300" />

      <div className="text-xs text-gray-500 px-2">
        {stats.currentIndex + 1}/{stats.totalEntries}
      </div>
    </div>
  )
}

// ===== BACKUP MANAGEMENT PANEL =====

export const BackupPanel: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false)
  const [backups, setBackups] = useState<any[]>([])
  const [isCreating, setIsCreating] = useState(false)
  const [isRestoring, setIsRestoring] = useState(false)
  
  const backup = useBackup()
  const { addNotification } = useUIActions()

  useEffect(() => {
    if (isOpen) {
      setBackups(backup.listBackups())
    }
  }, [isOpen, backup])

  const handleCreateBackup = async () => {
    setIsCreating(true)
    try {
      const backupId = await backup.createBackup({
        includeUser: true,
        includeUI: true,
        includeChat: false, // Don't backup chat for performance
      })
      
      addNotification({
        type: 'success',
        title: 'Backup Created',
        message: 'Your settings have been backed up successfully',
        duration: 3000,
      })
      
      setBackups(backup.listBackups())
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Backup Failed',
        message: 'Failed to create backup. Please try again.',
        duration: 5000,
      })
    } finally {
      setIsCreating(false)
    }
  }

  const handleRestoreBackup = async (backupId: string) => {
    setIsRestoring(true)
    try {
      const success = await backup.restoreBackup(backupId)
      
      if (success) {
        addNotification({
          type: 'success',
          title: 'Backup Restored',
          message: 'Your settings have been restored successfully',
          duration: 3000,
        })
      } else {
        throw new Error('Restore failed')
      }
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Restore Failed',
        message: 'Failed to restore backup. Please try again.',
        duration: 5000,
      })
    } finally {
      setIsRestoring(false)
    }
  }

  const handleExportBackup = async (backupId: string) => {
    try {
      await backup.exportBackup(backupId)
      
      addNotification({
        type: 'success',
        title: 'Backup Exported',
        message: 'Backup file has been downloaded',
        duration: 3000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Export Failed',
        message: 'Failed to export backup',
        duration: 5000,
      })
    }
  }

  const handleImportBackup = async (file: File) => {
    try {
      const backupId = await backup.importBackup(file)
      
      addNotification({
        type: 'success',
        title: 'Backup Imported',
        message: 'Backup has been imported successfully',
        duration: 3000,
      })
      
      setBackups(backup.listBackups())
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Import Failed',
        message: 'Failed to import backup file',
        duration: 5000,
      })
    }
  }

  const handleDeleteBackup = async (backupId: string) => {
    try {
      await backup.deleteBackup(backupId)
      setBackups(backup.listBackups())
      
      addNotification({
        type: 'info',
        title: 'Backup Deleted',
        message: 'Backup has been removed',
        duration: 2000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Delete Failed',
        message: 'Failed to delete backup',
        duration: 5000,
      })
    }
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 z-40 bg-indigo-600 text-white p-3 rounded-full shadow-lg hover:bg-indigo-700 transition-colors"
        title="Backup & Restore"
      >
        <Save className="h-5 w-5" />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4 max-h-96 overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Backup & Restore
              </h2>

              {/* Create Backup */}
              <div className="mb-6">
                <button
                  onClick={handleCreateBackup}
                  disabled={isCreating}
                  className="w-full flex items-center justify-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
                >
                  <Save className="h-4 w-4 mr-2" />
                  {isCreating ? 'Creating...' : 'Create Backup'}
                </button>
              </div>

              {/* Import Backup */}
              <div className="mb-6">
                <label className="block">
                  <input
                    type="file"
                    accept=".json"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) handleImportBackup(file)
                    }}
                    className="hidden"
                  />
                  <div className="w-full flex items-center justify-center px-4 py-2 border-2 border-dashed border-gray-300 rounded-lg hover:border-gray-400 transition-colors cursor-pointer">
                    <Upload className="h-4 w-4 mr-2" />
                    Import Backup
                  </div>
                </label>
              </div>

              {/* Backup List */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-gray-700 mb-2">
                  Available Backups ({backups.length})
                </h3>
                
                {backups.length === 0 ? (
                  <p className="text-sm text-gray-500 text-center py-4">
                    No backups found
                  </p>
                ) : (
                  backups.map((backup) => (
                    <div
                      key={backup.id}
                      className="flex items-center justify-between p-3 border border-gray-200 rounded-lg"
                    >
                      <div className="flex-1">
                        <div className="text-sm font-medium text-gray-900">
                          {new Date(backup.metadata.timestamp).toLocaleDateString()}
                        </div>
                        <div className="text-xs text-gray-500">
                          {new Date(backup.metadata.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        <button
                          onClick={() => handleRestoreBackup(backup.id)}
                          disabled={isRestoring}
                          className="p-1 text-blue-600 hover:bg-blue-50 rounded"
                          title="Restore"
                        >
                          <History className="h-4 w-4" />
                        </button>
                        
                        <button
                          onClick={() => handleExportBackup(backup.id)}
                          className="p-1 text-green-600 hover:bg-green-50 rounded"
                          title="Export"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                        
                        <button
                          onClick={() => handleDeleteBackup(backup.id)}
                          className="p-1 text-red-600 hover:bg-red-50 rounded"
                          title="Delete"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>

              <div className="mt-6 flex justify-end">
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

// ===== HISTORY VIEWER =====

export const HistoryViewer: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false)
  const { getHistory, getHistoryStats, undo, redo } = useUndoRedo()
  const [history, setHistory] = useState(getHistory())
  const [stats, setStats] = useState(getHistoryStats())

  useEffect(() => {
    if (isOpen) {
      setHistory(getHistory())
      setStats(getHistoryStats())
    }
  }, [isOpen, getHistory, getHistoryStats])

  const handleJumpToState = (index: number) => {
    const currentIndex = stats.currentIndex
    
    if (index < currentIndex) {
      // Undo to reach the target
      for (let i = currentIndex; i > index; i--) {
        undo()
      }
    } else if (index > currentIndex) {
      // Redo to reach the target
      for (let i = currentIndex; i < index; i++) {
        redo()
      }
    }
    
    setStats(getHistoryStats())
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-16 right-4 z-40 bg-gray-600 text-white p-2 rounded-full shadow-lg hover:bg-gray-700 transition-colors"
        title="View History"
      >
        <Clock className="h-4 w-4" />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white rounded-lg shadow-xl p-6 max-w-lg w-full mx-4 max-h-96 overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Action History
              </h2>

              <div className="space-y-2 mb-4">
                {history.length === 0 ? (
                  <p className="text-sm text-gray-500 text-center py-4">
                    No history available
                  </p>
                ) : (
                  history.map((entry, index) => (
                    <div
                      key={entry.id}
                      className={`flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors ${
                        index === stats.currentIndex
                          ? 'bg-indigo-50 border border-indigo-200'
                          : index < stats.currentIndex
                          ? 'bg-gray-50'
                          : 'bg-white border border-gray-200'
                      }`}
                      onClick={() => handleJumpToState(index)}
                    >
                      <div className="flex-1">
                        <div className="text-sm font-medium text-gray-900">
                          {entry.action}
                        </div>
                        {entry.metadata?.description && (
                          <div className="text-xs text-gray-500">
                            {entry.metadata.description}
                          </div>
                        )}
                      </div>
                      
                      <div className="text-xs text-gray-400">
                        {entry.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  ))
                )}
              </div>

              <div className="flex justify-between items-center text-sm text-gray-600 mb-4">
                <span>Total: {stats.totalEntries} actions</span>
                <span>Current: {stats.currentIndex + 1}</span>
              </div>

              <div className="flex justify-end">
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default BackupPanel
