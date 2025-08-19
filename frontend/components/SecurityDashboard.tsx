'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Shield, 
  Lock, 
  Key, 
  Eye, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Globe,
  Smartphone,
  Download,
  RefreshCw,
  Settings,
  Activity
} from 'lucide-react'
import { AuthService, AuditLogger, SecurityUtils, type AuditEvent } from '../lib/auth'
import { useUIActions, useUserState } from '../store'

// ===== TYPES =====

interface SecurityDashboardProps {
  onClose: () => void
  className?: string
}

interface SecurityMetric {
  id: string
  name: string
  status: 'secure' | 'warning' | 'critical'
  value: string
  description: string
  action?: string
}

interface LoginSession {
  id: string
  ipAddress: string
  location: string
  device: string
  browser: string
  loginTime: Date
  lastActivity: Date
  isCurrent: boolean
}

// ===== SECURITY DASHBOARD =====

export const SecurityDashboard: React.FC<SecurityDashboardProps> = ({ 
  onClose, 
  className = '' 
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'sessions' | 'audit' | 'settings'>('overview')
  const [isLoading, setIsLoading] = useState(false)
  const [securityMetrics, setSecurityMetrics] = useState<SecurityMetric[]>([])
  const [loginSessions, setLoginSessions] = useState<LoginSession[]>([])
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([])
  const [mfaEnabled, setMfaEnabled] = useState(false)
  const [showMfaSetup, setShowMfaSetup] = useState(false)

  const { addNotification } = useUIActions()
  const userState = useUserState()
  const authService = AuthService.getInstance()
  const auditLogger = AuditLogger.getInstance()

  // Load security data
  useEffect(() => {
    loadSecurityData()
  }, [])

  const loadSecurityData = async () => {
    setIsLoading(true)

    try {
      // Mock security metrics
      setSecurityMetrics([
        {
          id: 'mfa-status',
          name: 'Multi-Factor Authentication',
          status: mfaEnabled ? 'secure' : 'warning',
          value: mfaEnabled ? 'Enabled' : 'Disabled',
          description: 'Adds an extra layer of security to your account',
          action: mfaEnabled ? 'Manage' : 'Enable'
        },
        {
          id: 'password-strength',
          name: 'Password Strength',
          status: 'secure',
          value: 'Strong',
          description: 'Your password meets security requirements',
        },
        {
          id: 'login-alerts',
          name: 'Login Alerts',
          status: 'secure',
          value: 'Enabled',
          description: 'Get notified of suspicious login attempts',
        },
        {
          id: 'session-timeout',
          name: 'Session Timeout',
          status: 'secure',
          value: '24 hours',
          description: 'Automatic logout after inactivity',
        },
      ])

      // Mock login sessions
      setLoginSessions([
        {
          id: 'session-1',
          ipAddress: '192.168.1.100',
          location: 'San Francisco, CA',
          device: 'MacBook Pro',
          browser: 'Chrome 120.0',
          loginTime: new Date(Date.now() - 3600000),
          lastActivity: new Date(Date.now() - 300000),
          isCurrent: true,
        },
        {
          id: 'session-2',
          ipAddress: '10.0.0.50',
          location: 'New York, NY',
          device: 'iPhone 15',
          browser: 'Safari 17.0',
          loginTime: new Date(Date.now() - 86400000),
          lastActivity: new Date(Date.now() - 7200000),
          isCurrent: false,
        },
      ])

      // Load audit events
      setAuditEvents(auditLogger.getEvents().slice(-20))

    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Load Error',
        message: 'Failed to load security data',
        duration: 5000,
      })
    } finally {
      setIsLoading(false)
    }
  }

  // ===== SECURITY ACTIONS =====

  const handleEnableMFA = async () => {
    setIsLoading(true)

    try {
      const result = await authService.enableMFA()

      if (result.error) {
        addNotification({
          type: 'error',
          title: 'MFA Setup Failed',
          message: result.error,
          duration: 5000,
        })
      } else {
        setMfaEnabled(true)
        setShowMfaSetup(true)
        
        // Log MFA enablement
        await auditLogger.log({
          userId: userState.user?.id,
          action: 'mfa_enabled',
          resource: 'security',
        })

        addNotification({
          type: 'success',
          title: 'MFA Enabled',
          message: 'Multi-factor authentication has been enabled',
          duration: 3000,
        })
      }
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'MFA Setup Error',
        message: 'An unexpected error occurred',
        duration: 5000,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleDisableMFA = async () => {
    const password = prompt('Please enter your password to disable MFA:')
    if (!password) return

    setIsLoading(true)

    try {
      const result = await authService.disableMFA(password)

      if (result.error) {
        addNotification({
          type: 'error',
          title: 'MFA Disable Failed',
          message: result.error,
          duration: 5000,
        })
      } else {
        setMfaEnabled(false)
        
        // Log MFA disablement
        await auditLogger.log({
          userId: userState.user?.id,
          action: 'mfa_disabled',
          resource: 'security',
        })

        addNotification({
          type: 'warning',
          title: 'MFA Disabled',
          message: 'Multi-factor authentication has been disabled',
          duration: 3000,
        })
      }
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'MFA Disable Error',
        message: 'An unexpected error occurred',
        duration: 5000,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleTerminateSession = async (sessionId: string) => {
    try {
      // Mock session termination
      setLoginSessions(prev => prev.filter(session => session.id !== sessionId))
      
      // Log session termination
      await auditLogger.log({
        userId: userState.user?.id,
        action: 'session_terminated',
        resource: 'security',
        metadata: { sessionId }
      })

      addNotification({
        type: 'success',
        title: 'Session Terminated',
        message: 'The session has been terminated successfully',
        duration: 3000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Termination Failed',
        message: 'Failed to terminate the session',
        duration: 5000,
      })
    }
  }

  const exportAuditLog = () => {
    try {
      const data = JSON.stringify(auditEvents, null, 2)
      const blob = new Blob([data], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      
      const a = document.createElement('a')
      a.href = url
      a.download = `audit-log-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      
      URL.revokeObjectURL(url)

      addNotification({
        type: 'success',
        title: 'Export Complete',
        message: 'Audit log has been downloaded',
        duration: 3000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Export Failed',
        message: 'Failed to export audit log',
        duration: 5000,
      })
    }
  }

  // ===== RENDER HELPERS =====

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'secure':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'critical':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <Shield className="h-5 w-5 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'secure':
        return 'text-green-700 bg-green-50 border-green-200'
      case 'warning':
        return 'text-yellow-700 bg-yellow-50 border-yellow-200'
      case 'critical':
        return 'text-red-700 bg-red-50 border-red-200'
      default:
        return 'text-gray-700 bg-gray-50 border-gray-200'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className={`fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4 ${className}`}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-red-600 to-pink-600 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Shield className="h-6 w-6 text-white" />
              <h2 className="text-xl font-semibold text-white">Security Dashboard</h2>
            </div>
            <button
              onClick={onClose}
              className="text-white hover:text-gray-200 transition-colors"
            >
              <XCircle className="h-6 w-6" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          {[
            { id: 'overview', label: 'Overview', icon: <Eye className="h-4 w-4" /> },
            { id: 'sessions', label: 'Active Sessions', icon: <Globe className="h-4 w-4" /> },
            { id: 'audit', label: 'Audit Log', icon: <Activity className="h-4 w-4" /> },
            { id: 'settings', label: 'Settings', icon: <Settings className="h-4 w-4" /> },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-red-600 border-b-2 border-red-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-96">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
              <span className="ml-2 text-gray-600">Loading security data...</span>
            </div>
          ) : (
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                {/* Overview Tab */}
                {activeTab === 'overview' && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-900">Security Overview</h3>
                    
                    <div className="grid gap-4">
                      {securityMetrics.map((metric) => (
                        <div
                          key={metric.id}
                          className={`border rounded-lg p-4 ${getStatusColor(metric.status)}`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              {getStatusIcon(metric.status)}
                              <div>
                                <h4 className="font-medium">{metric.name}</h4>
                                <p className="text-sm opacity-75">{metric.description}</p>
                              </div>
                            </div>
                            
                            <div className="text-right">
                              <p className="font-medium">{metric.value}</p>
                              {metric.action && (
                                <button
                                  onClick={() => {
                                    if (metric.id === 'mfa-status') {
                                      mfaEnabled ? handleDisableMFA() : handleEnableMFA()
                                    }
                                  }}
                                  className="text-sm underline hover:no-underline"
                                >
                                  {metric.action}
                                </button>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Sessions Tab */}
                {activeTab === 'sessions' && (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-gray-900">Active Sessions</h3>
                      <button
                        onClick={loadSecurityData}
                        className="flex items-center space-x-1 text-sm text-gray-600 hover:text-gray-800"
                      >
                        <RefreshCw className="h-4 w-4" />
                        <span>Refresh</span>
                      </button>
                    </div>
                    
                    <div className="space-y-3">
                      {loginSessions.map((session) => (
                        <div key={session.id} className="border rounded-lg p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              <div className={`w-3 h-3 rounded-full ${session.isCurrent ? 'bg-green-500' : 'bg-gray-400'}`} />
                              <div>
                                <div className="flex items-center space-x-2">
                                  <p className="font-medium">{session.device}</p>
                                  {session.isCurrent && (
                                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                                      Current
                                    </span>
                                  )}
                                </div>
                                <p className="text-sm text-gray-600">
                                  {session.browser} • {session.location}
                                </p>
                                <p className="text-xs text-gray-500">
                                  IP: {session.ipAddress} • Last active: {session.lastActivity.toLocaleString()}
                                </p>
                              </div>
                            </div>
                            
                            {!session.isCurrent && (
                              <button
                                onClick={() => handleTerminateSession(session.id)}
                                className="px-3 py-1 text-sm text-red-600 hover:text-red-800 border border-red-300 rounded hover:bg-red-50 transition-colors"
                              >
                                Terminate
                              </button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Audit Log Tab */}
                {activeTab === 'audit' && (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-gray-900">Security Audit Log</h3>
                      <button
                        onClick={exportAuditLog}
                        className="flex items-center space-x-1 px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                      >
                        <Download className="h-4 w-4" />
                        <span>Export</span>
                      </button>
                    </div>
                    
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {auditEvents.length > 0 ? (
                        auditEvents.map((event) => (
                          <div key={event.id} className="border rounded-lg p-3">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="font-medium text-sm">{event.action.replace('_', ' ').toUpperCase()}</p>
                                <p className="text-xs text-gray-600">
                                  Resource: {event.resource}
                                </p>
                              </div>
                              <div className="text-right">
                                <p className="text-xs text-gray-500">
                                  {event.timestamp.toLocaleString()}
                                </p>
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-8 text-gray-500">
                          <Activity className="h-8 w-8 mx-auto mb-2 text-gray-300" />
                          <p>No audit events recorded</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Settings Tab */}
                {activeTab === 'settings' && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-900">Security Settings</h3>
                    
                    <div className="space-y-4">
                      <div className="border rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="font-medium">Multi-Factor Authentication</h4>
                            <p className="text-sm text-gray-600">
                              Add an extra layer of security to your account
                            </p>
                          </div>
                          <button
                            onClick={mfaEnabled ? handleDisableMFA : handleEnableMFA}
                            disabled={isLoading}
                            className={`px-4 py-2 rounded-lg transition-colors disabled:opacity-50 ${
                              mfaEnabled
                                ? 'bg-red-100 text-red-700 hover:bg-red-200'
                                : 'bg-green-100 text-green-700 hover:bg-green-200'
                            }`}
                          >
                            {mfaEnabled ? 'Disable' : 'Enable'}
                          </button>
                        </div>
                      </div>
                      
                      <div className="border rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="font-medium">Login Notifications</h4>
                            <p className="text-sm text-gray-600">
                              Get notified of new login attempts
                            </p>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input type="checkbox" className="sr-only peer" defaultChecked />
                            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
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

export default SecurityDashboard
