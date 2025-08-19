'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Share2, 
  Copy, 
  Users, 
  UserPlus, 
  UserMinus, 
  Crown, 
  Shield,
  Eye,
  Edit,
  X,
  Check,
  Mail,
  Link
} from 'lucide-react'
import { useWebSocket } from '../hooks/useWebSocket'
import { useUIActions, useUserState } from '../store'

// ===== TYPES =====

interface SessionSharingProps {
  sessionId: string
  onClose: () => void
}

interface SharedUser {
  id: string
  name: string
  email: string
  avatar?: string
  role: 'owner' | 'editor' | 'viewer'
  status: 'pending' | 'accepted' | 'active'
  joinedAt?: Date
}

interface SharePermission {
  canEdit: boolean
  canInvite: boolean
  canRemove: boolean
  expiresAt?: Date
}

// ===== SESSION SHARING COMPONENT =====

export const SessionSharing: React.FC<SessionSharingProps> = ({ sessionId, onClose }) => {
  const [activeTab, setActiveTab] = useState<'share' | 'manage'>('share')
  const [shareMethod, setShareMethod] = useState<'link' | 'email'>('link')
  const [emailInput, setEmailInput] = useState('')
  const [selectedRole, setSelectedRole] = useState<'editor' | 'viewer'>('viewer')
  const [shareLink, setShareLink] = useState('')
  const [sharedUsers, setSharedUsers] = useState<SharedUser[]>([])
  const [permissions, setPermissions] = useState<SharePermission>({
    canEdit: true,
    canInvite: true,
    canRemove: true,
  })

  const { shareSession, isConnected } = useWebSocket()
  const { addNotification } = useUIActions()
  const userState = useUserState()

  // Generate share link
  useEffect(() => {
    const baseUrl = window.location.origin
    const link = `${baseUrl}/chat?session=${sessionId}&join=true`
    setShareLink(link)
  }, [sessionId])

  // Mock shared users data
  useEffect(() => {
    setSharedUsers([
      {
        id: 'user-1',
        name: 'Alice Johnson',
        email: 'alice@example.com',
        role: 'owner',
        status: 'active',
        joinedAt: new Date(Date.now() - 3600000),
      },
      {
        id: 'user-2',
        name: 'Bob Smith',
        email: 'bob@example.com',
        role: 'editor',
        status: 'active',
        joinedAt: new Date(Date.now() - 1800000),
      },
      {
        id: 'user-3',
        name: 'Carol Davis',
        email: 'carol@example.com',
        role: 'viewer',
        status: 'pending',
      },
    ])
  }, [])

  // ===== HANDLERS =====

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareLink)
      addNotification({
        type: 'success',
        title: 'Link Copied',
        message: 'Share link copied to clipboard',
        duration: 2000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Copy Failed',
        message: 'Failed to copy link to clipboard',
        duration: 3000,
      })
    }
  }

  const handleEmailInvite = () => {
    if (!emailInput.trim()) {
      addNotification({
        type: 'warning',
        title: 'Email Required',
        message: 'Please enter an email address',
        duration: 3000,
      })
      return
    }

    // Simulate sending email invitation
    const newUser: SharedUser = {
      id: `user-${Date.now()}`,
      name: emailInput.split('@')[0],
      email: emailInput,
      role: selectedRole,
      status: 'pending',
    }

    setSharedUsers(prev => [...prev, newUser])
    setEmailInput('')

    addNotification({
      type: 'success',
      title: 'Invitation Sent',
      message: `Invitation sent to ${emailInput}`,
      duration: 3000,
    })
  }

  const handleRoleChange = (userId: string, newRole: 'editor' | 'viewer') => {
    setSharedUsers(prev => 
      prev.map(user => 
        user.id === userId ? { ...user, role: newRole } : user
      )
    )

    addNotification({
      type: 'info',
      title: 'Role Updated',
      message: 'User role has been updated',
      duration: 2000,
    })
  }

  const handleRemoveUser = (userId: string) => {
    setSharedUsers(prev => prev.filter(user => user.id !== userId))

    addNotification({
      type: 'info',
      title: 'User Removed',
      message: 'User has been removed from the session',
      duration: 2000,
    })
  }

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'owner':
        return <Crown className="h-4 w-4 text-yellow-500" />
      case 'editor':
        return <Edit className="h-4 w-4 text-blue-500" />
      case 'viewer':
        return <Eye className="h-4 w-4 text-gray-500" />
      default:
        return <Users className="h-4 w-4 text-gray-400" />
    }
  }

  const getStatusBadge = (status: string) => {
    const colors = {
      active: 'bg-green-100 text-green-800',
      pending: 'bg-yellow-100 text-yellow-800',
      accepted: 'bg-blue-100 text-blue-800',
    }

    return (
      <span className={`px-2 py-1 text-xs rounded-full ${colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800'}`}>
        {status}
      </span>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-2">
            <Share2 className="h-5 w-5 text-indigo-600" />
            <h2 className="text-xl font-semibold text-gray-900">Share Session</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab('share')}
            className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === 'share'
                ? 'text-indigo-600 border-b-2 border-indigo-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Share Access
          </button>
          <button
            onClick={() => setActiveTab('manage')}
            className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === 'manage'
                ? 'text-indigo-600 border-b-2 border-indigo-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Manage Users ({sharedUsers.length})
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-96">
          {activeTab === 'share' ? (
            <div className="space-y-6">
              {/* Share Method Selection */}
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-3">Share Method</h3>
                <div className="flex space-x-4">
                  <button
                    onClick={() => setShareMethod('link')}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-colors ${
                      shareMethod === 'link'
                        ? 'border-indigo-500 bg-indigo-50 text-indigo-700'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                  >
                    <Link className="h-4 w-4" />
                    <span>Share Link</span>
                  </button>
                  <button
                    onClick={() => setShareMethod('email')}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-colors ${
                      shareMethod === 'email'
                        ? 'border-indigo-500 bg-indigo-50 text-indigo-700'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                  >
                    <Mail className="h-4 w-4" />
                    <span>Email Invite</span>
                  </button>
                </div>
              </div>

              {/* Share Link */}
              {shareMethod === 'link' && (
                <div>
                  <h3 className="text-sm font-medium text-gray-900 mb-3">Share Link</h3>
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={shareLink}
                      readOnly
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-sm"
                    />
                    <button
                      onClick={handleCopyLink}
                      className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center space-x-1"
                    >
                      <Copy className="h-4 w-4" />
                      <span>Copy</span>
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Anyone with this link can join the session with viewer permissions.
                  </p>
                </div>
              )}

              {/* Email Invite */}
              {shareMethod === 'email' && (
                <div>
                  <h3 className="text-sm font-medium text-gray-900 mb-3">Email Invitation</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Email Address
                      </label>
                      <input
                        type="email"
                        value={emailInput}
                        onChange={(e) => setEmailInput(e.target.value)}
                        placeholder="Enter email address"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Role
                      </label>
                      <select
                        value={selectedRole}
                        onChange={(e) => setSelectedRole(e.target.value as 'editor' | 'viewer')}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                      >
                        <option value="viewer">Viewer - Can view messages</option>
                        <option value="editor">Editor - Can send messages</option>
                      </select>
                    </div>
                    
                    <button
                      onClick={handleEmailInvite}
                      disabled={!emailInput.trim()}
                      className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                    >
                      <UserPlus className="h-4 w-4" />
                      <span>Send Invitation</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Manage Users Tab */
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900">Session Members</h3>
              
              <div className="space-y-3">
                {sharedUsers.map((user) => (
                  <div key={user.id} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white text-sm font-medium">
                        {user.name.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">{user.name}</p>
                        <p className="text-xs text-gray-500">{user.email}</p>
                        {user.joinedAt && (
                          <p className="text-xs text-gray-400">
                            Joined {user.joinedAt.toLocaleDateString()}
                          </p>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {getRoleIcon(user.role)}
                      {getStatusBadge(user.status)}
                      
                      {user.role !== 'owner' && (
                        <div className="flex items-center space-x-1">
                          <select
                            value={user.role}
                            onChange={(e) => handleRoleChange(user.id, e.target.value as 'editor' | 'viewer')}
                            className="text-xs border border-gray-300 rounded px-2 py-1"
                          >
                            <option value="viewer">Viewer</option>
                            <option value="editor">Editor</option>
                          </select>
                          
                          <button
                            onClick={() => handleRemoveUser(user.id)}
                            className="p-1 text-red-500 hover:text-red-700 transition-colors"
                            title="Remove user"
                          >
                            <UserMinus className="h-4 w-4" />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end space-x-2 p-6 border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </motion.div>
    </motion.div>
  )
}

export default SessionSharing
