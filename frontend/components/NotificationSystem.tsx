'use client'

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react'
import { useNotifications, useUIActions, isStoreReady } from '../store'

// ===== NOTIFICATION COMPONENT =====

interface NotificationProps {
  notification: {
    id: string
    type: 'info' | 'success' | 'warning' | 'error'
    title: string
    message: string
    timestamp: Date
    duration?: number
    actions?: Array<{
      label: string
      action: () => void
      style?: 'primary' | 'secondary' | 'danger'
    }>
    persistent?: boolean
  }
  onRemove: (id: string) => void
}

const NotificationItem: React.FC<NotificationProps> = ({ notification, onRemove }) => {
  const getIcon = () => {
    switch (notification.type) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />
      default:
        return <Info className="h-5 w-5 text-blue-500" />
    }
  }

  const getBackgroundColor = () => {
    switch (notification.type) {
      case 'success':
        return 'bg-green-50 border-green-200'
      case 'warning':
        return 'bg-yellow-50 border-yellow-200'
      case 'error':
        return 'bg-red-50 border-red-200'
      default:
        return 'bg-blue-50 border-blue-200'
    }
  }

  const getTextColor = () => {
    switch (notification.type) {
      case 'success':
        return 'text-green-800'
      case 'warning':
        return 'text-yellow-800'
      case 'error':
        return 'text-red-800'
      default:
        return 'text-blue-800'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -50, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className={`relative p-4 rounded-lg border shadow-lg backdrop-blur-sm ${getBackgroundColor()}`}
      role="alert"
      aria-live={notification.type === 'error' ? 'assertive' : 'polite'}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          {getIcon()}
        </div>
        
        <div className="flex-1 min-w-0">
          <h4 className={`text-sm font-semibold ${getTextColor()}`}>
            {notification.title}
          </h4>
          <p className={`text-sm mt-1 ${getTextColor()}`}>
            {notification.message}
          </p>
          
          {notification.actions && notification.actions.length > 0 && (
            <div className="mt-3 flex space-x-2">
              {notification.actions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => {
                    action.action()
                    if (!notification.persistent) {
                      onRemove(notification.id)
                    }
                  }}
                  className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                    action.style === 'primary'
                      ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                      : action.style === 'danger'
                      ? 'bg-red-600 text-white hover:bg-red-700'
                      : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {action.label}
                </button>
              ))}
            </div>
          )}
        </div>
        
        <button
          onClick={() => onRemove(notification.id)}
          className={`flex-shrink-0 p-1 rounded-full hover:bg-white/50 transition-colors ${getTextColor()}`}
          aria-label="Dismiss notification"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      
      {/* Progress bar for timed notifications */}
      {notification.duration && !notification.persistent && (
        <motion.div
          className="absolute bottom-0 left-0 h-1 bg-current opacity-30 rounded-b-lg"
          initial={{ width: '100%' }}
          animate={{ width: '0%' }}
          transition={{ duration: notification.duration / 1000, ease: 'linear' }}
        />
      )}
    </motion.div>
  )
}

// ===== NOTIFICATION SYSTEM COMPONENT =====

export const NotificationSystem: React.FC = () => {
  // Check if store is ready before attempting to access it
  if (!isStoreReady()) {
    return null
  }

  // Safe store access with error boundary
  let notifications, removeNotification

  try {
    notifications = useNotifications()
    const uiActions = useUIActions()
    removeNotification = uiActions.removeNotification
  } catch (error) {
    console.warn('NotificationSystem: Store not ready, skipping render')
    return null
  }

  // Additional safety check
  if (!Array.isArray(notifications)) {
    return null
  }

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm w-full">
      <AnimatePresence mode="popLayout">
        {notifications.map((notification) => (
          <NotificationItem
            key={notification.id}
            notification={notification}
            onRemove={removeNotification}
          />
        ))}
      </AnimatePresence>
    </div>
  )
}

// ===== NOTIFICATION HOOKS =====

export const useNotificationActions = () => {
  const { addNotification } = useUIActions()

  const showSuccess = (title: string, message: string, duration = 3000) => {
    addNotification({
      type: 'success',
      title,
      message,
      duration,
    })
  }

  const showError = (title: string, message: string, duration = 5000) => {
    addNotification({
      type: 'error',
      title,
      message,
      duration,
    })
  }

  const showWarning = (title: string, message: string, duration = 4000) => {
    addNotification({
      type: 'warning',
      title,
      message,
      duration,
    })
  }

  const showInfo = (title: string, message: string, duration = 3000) => {
    addNotification({
      type: 'info',
      title,
      message,
      duration,
    })
  }

  const showPersistent = (
    type: 'info' | 'success' | 'warning' | 'error',
    title: string,
    message: string,
    actions?: Array<{
      label: string
      action: () => void
      style?: 'primary' | 'secondary' | 'danger'
    }>
  ) => {
    addNotification({
      type,
      title,
      message,
      persistent: true,
      actions,
    })
  }

  return {
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showPersistent,
  }
}

// ===== TOAST NOTIFICATIONS =====

export const useToast = () => {
  const actions = useNotificationActions()

  const toast = {
    success: (message: string, title = 'Success') => actions.showSuccess(title, message),
    error: (message: string, title = 'Error') => actions.showError(title, message),
    warning: (message: string, title = 'Warning') => actions.showWarning(title, message),
    info: (message: string, title = 'Info') => actions.showInfo(title, message),
    
    // Convenience methods
    loading: (message: string) => actions.showInfo('Loading', message),
    saved: () => actions.showSuccess('Saved', 'Your changes have been saved'),
    copied: () => actions.showSuccess('Copied', 'Copied to clipboard'),
    deleted: () => actions.showSuccess('Deleted', 'Item has been deleted'),
    
    // Confirmation toasts
    confirm: (
      message: string,
      onConfirm: () => void,
      onCancel?: () => void,
      title = 'Confirm Action'
    ) => {
      actions.showPersistent('warning', title, message, [
        {
          label: 'Confirm',
          action: onConfirm,
          style: 'primary',
        },
        {
          label: 'Cancel',
          action: onCancel || (() => {}),
          style: 'secondary',
        },
      ])
    },
  }

  return toast
}

export default NotificationSystem
