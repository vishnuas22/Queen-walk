'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Users, 
  Wifi, 
  WifiOff, 
  Share2, 
  UserPlus, 
  MessageCircle,
  Eye,
  Edit3,
  Clock
} from 'lucide-react'
import { useWebSocket, useTypingIndicator } from '../hooks/useWebSocket'
import { useUIActions } from '../store'
import SessionSharing from './SessionSharing'

// ===== TYPES =====

interface CollaborationPanelProps {
  sessionId?: string
  className?: string
}

interface UserAvatarProps {
  user: {
    userId: string
    name: string
    avatar?: string
    status: 'online' | 'away' | 'busy' | 'offline'
    isTyping?: boolean
  }
  size?: 'sm' | 'md' | 'lg'
  showStatus?: boolean
}

// ===== USER AVATAR COMPONENT =====

const UserAvatar: React.FC<UserAvatarProps> = ({ 
  user, 
  size = 'md', 
  showStatus = true 
}) => {
  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-8 h-8 text-sm',
    lg: 'w-10 h-10 text-base',
  }

  const statusColors = {
    online: 'bg-green-500',
    away: 'bg-yellow-500',
    busy: 'bg-red-500',
    offline: 'bg-gray-400',
  }

  return (
    <div className="relative">
      <div className={`${sizeClasses[size]} rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-medium overflow-hidden`}>
        {user.avatar ? (
          <img 
            src={user.avatar} 
            alt={user.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <span>{user.name.charAt(0).toUpperCase()}</span>
        )}
      </div>
      
      {showStatus && (
        <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-white ${statusColors[user.status]}`} />
      )}
      
      {user.isTyping && (
        <motion.div
          className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 1, repeat: Infinity }}
        />
      )}
    </div>
  )
}

// ===== TYPING INDICATOR COMPONENT =====

const TypingIndicator: React.FC<{ typingUsers: string[] }> = ({ typingUsers }) => {
  if (typingUsers.length === 0) return null

  const getTypingText = () => {
    if (typingUsers.length === 1) {
      return `${typingUsers[0]} is typing...`
    } else if (typingUsers.length === 2) {
      return `${typingUsers[0]} and ${typingUsers[1]} are typing...`
    } else {
      return `${typingUsers.length} people are typing...`
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 10 }}
      className="flex items-center space-x-2 text-sm text-gray-500 px-4 py-2"
    >
      <div className="flex space-x-1">
        <motion.div
          className="w-1 h-1 bg-gray-400 rounded-full"
          animate={{ scale: [1, 1.5, 1] }}
          transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
        />
        <motion.div
          className="w-1 h-1 bg-gray-400 rounded-full"
          animate={{ scale: [1, 1.5, 1] }}
          transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
        />
        <motion.div
          className="w-1 h-1 bg-gray-400 rounded-full"
          animate={{ scale: [1, 1.5, 1] }}
          transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
        />
      </div>
      <span>{getTypingText()}</span>
    </motion.div>
  )
}

// ===== CONNECTION STATUS COMPONENT =====

const ConnectionStatus: React.FC<{ 
  isConnected: boolean
  isConnecting: boolean
  connectionError: string | null
}> = ({ isConnected, isConnecting, connectionError }) => {
  const getStatusIcon = () => {
    if (isConnecting) {
      return (
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        >
          <Wifi className="h-4 w-4 text-yellow-500" />
        </motion.div>
      )
    }
    
    return isConnected ? (
      <Wifi className="h-4 w-4 text-green-500" />
    ) : (
      <WifiOff className="h-4 w-4 text-red-500" />
    )
  }

  const getStatusText = () => {
    if (isConnecting) return 'Connecting...'
    if (connectionError) return 'Connection failed'
    return isConnected ? 'Connected' : 'Disconnected'
  }

  const getStatusColor = () => {
    if (isConnecting) return 'text-yellow-600'
    if (connectionError) return 'text-red-600'
    return isConnected ? 'text-green-600' : 'text-gray-600'
  }

  return (
    <div className={`flex items-center space-x-2 text-xs ${getStatusColor()}`}>
      {getStatusIcon()}
      <span>{getStatusText()}</span>
    </div>
  )
}

// ===== COLLABORATION PANEL COMPONENT =====

export const CollaborationPanel: React.FC<CollaborationPanelProps> = ({
  sessionId,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showShareDialog, setShowShareDialog] = useState(false)
  const [showSessionSharing, setShowSessionSharing] = useState(false)
  
  const {
    isConnected,
    isConnecting,
    connectionError,
    connectedUsers,
    typingUsers,
    connect,
    disconnect,
    shareSession,
  } = useWebSocket({ 
    autoConnect: true, 
    sessionId 
  })

  const { addNotification } = useUIActions()

  const handleShare = () => {
    if (!sessionId) {
      addNotification({
        type: 'warning',
        title: 'No Session',
        message: 'Start a conversation to share it with others',
        duration: 3000,
      })
      return
    }

    setShowSessionSharing(true)
  }

  const handleToggleConnection = () => {
    if (isConnected) {
      disconnect()
    } else {
      connect()
    }
  }

  return (
    <div className={`bg-white border border-gray-200 rounded-lg shadow-sm ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-gray-100">
        <div className="flex items-center space-x-2">
          <Users className="h-4 w-4 text-gray-600" />
          <span className="text-sm font-medium text-gray-900">
            Collaboration ({connectedUsers.length})
          </span>
        </div>
        
        <div className="flex items-center space-x-2">
          <ConnectionStatus 
            isConnected={isConnected}
            isConnecting={isConnecting}
            connectionError={connectionError}
          />
          
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </motion.div>
          </button>
        </div>
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-3 space-y-3">
              {/* Connected Users */}
              {connectedUsers.length > 0 ? (
                <div>
                  <h4 className="text-xs font-medium text-gray-700 mb-2">
                    Connected Users
                  </h4>
                  <div className="space-y-2">
                    {connectedUsers.map((user) => (
                      <div key={user.userId} className="flex items-center space-x-2">
                        <UserAvatar user={user} size="sm" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {user.name}
                          </p>
                          <p className="text-xs text-gray-500 capitalize">
                            {user.status}
                          </p>
                        </div>
                        {user.isTyping && (
                          <Edit3 className="h-3 w-3 text-blue-500" />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-4">
                  <Users className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                  <p className="text-sm text-gray-500">No other users connected</p>
                </div>
              )}

              {/* Actions */}
              <div className="flex space-x-2 pt-2 border-t border-gray-100">
                <button
                  onClick={handleShare}
                  disabled={!sessionId}
                  className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 text-xs bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Share2 className="h-3 w-3" />
                  <span>Share</span>
                </button>
                
                <button
                  onClick={handleToggleConnection}
                  className={`flex-1 flex items-center justify-center space-x-1 px-3 py-2 text-xs rounded transition-colors ${
                    isConnected
                      ? 'bg-red-100 text-red-700 hover:bg-red-200'
                      : 'bg-green-100 text-green-700 hover:bg-green-200'
                  }`}
                >
                  {isConnected ? <WifiOff className="h-3 w-3" /> : <Wifi className="h-3 w-3" />}
                  <span>{isConnected ? 'Disconnect' : 'Connect'}</span>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Session Sharing */}
      <AnimatePresence>
        {showSessionSharing && sessionId && (
          <SessionSharing
            sessionId={sessionId}
            onClose={() => setShowSessionSharing(false)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

// ===== COLLABORATION STATUS BAR =====

export const CollaborationStatusBar: React.FC<{ sessionId?: string }> = ({ sessionId }) => {
  const { isConnected, connectedUsers, typingUsers } = useWebSocket({ sessionId })

  if (!isConnected && connectedUsers.length === 0) return null

  return (
    <div className="bg-gray-50 border-t border-gray-200 px-4 py-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {/* Connected Users Avatars */}
          {connectedUsers.length > 0 && (
            <div className="flex items-center space-x-1">
              <div className="flex -space-x-1">
                {connectedUsers.slice(0, 3).map((user) => (
                  <UserAvatar key={user.userId} user={user} size="sm" showStatus={false} />
                ))}
              </div>
              {connectedUsers.length > 3 && (
                <span className="text-xs text-gray-500 ml-2">
                  +{connectedUsers.length - 3} more
                </span>
              )}
            </div>
          )}
          
          {/* Connection Status */}
          <div className="flex items-center space-x-1">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-gray-400'}`} />
            <span className="text-xs text-gray-600">
              {isConnected ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Typing Indicator */}
        <AnimatePresence>
          {typingUsers.length > 0 && (
            <TypingIndicator typingUsers={typingUsers} />
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

export default CollaborationPanel
