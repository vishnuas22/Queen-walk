'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Link, 
  Zap, 
  Settings, 
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Database,
  Plug,
  Download,
  Upload,
  Trash2,
  Play,
  Pause,
  RefreshCw,
  Plus,
  Search,
  Filter,
  BarChart3,
  Globe,
  Shield
} from 'lucide-react'
import { 
  APIIntegrationService, 
  PluginManager,
  type APIConnection, 
  type APIEndpoint,
  type Plugin,
  type DataSyncConfig,
  type APIRequest
} from '../lib/api-integration'
import { useUIActions } from '../store'

// ===== TYPES =====

interface APIIntegrationProps {
  onConnectionChange?: (connection: APIConnection) => void
  onPluginActivated?: (plugin: Plugin) => void
  className?: string
}

// ===== API INTEGRATION COMPONENT =====

export const APIIntegration: React.FC<APIIntegrationProps> = ({
  onConnectionChange,
  onPluginActivated,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'connections' | 'plugins' | 'sync' | 'requests'>('connections')
  const [connections, setConnections] = useState<APIConnection[]>([])
  const [plugins, setPlugins] = useState<Plugin[]>([])
  const [syncConfigs, setSyncConfigs] = useState<DataSyncConfig[]>([])
  const [requestHistory, setRequestHistory] = useState<APIRequest[]>([])
  const [selectedConnection, setSelectedConnection] = useState<string | null>(null)
  const [selectedPlugin, setSelectedPlugin] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterCategory, setFilterCategory] = useState<string>('all')

  const { addNotification } = useUIActions()
  const apiService = APIIntegrationService.getInstance()
  const pluginManager = PluginManager.getInstance()

  // Initialize data
  useEffect(() => {
    loadConnections()
    loadPlugins()
    loadSyncConfigs()
    loadRequestHistory()

    // Set up event listeners
    const unsubscribeConnection = apiService.onConnection((connection) => {
      setConnections(prev => prev.map(c => c.id === connection.id ? connection : c))
      onConnectionChange?.(connection)
    })

    const unsubscribePluginActivate = pluginManager.onActivate((plugin) => {
      setPlugins(prev => prev.map(p => p.id === plugin.id ? plugin : p))
      onPluginActivated?.(plugin)
    })

    return () => {
      unsubscribeConnection()
      unsubscribePluginActivate()
    }
  }, [onConnectionChange, onPluginActivated])

  const loadConnections = () => {
    setConnections(apiService.getConnections())
  }

  const loadPlugins = () => {
    setPlugins(pluginManager.getPlugins())
  }

  const loadSyncConfigs = () => {
    setSyncConfigs(apiService.getSyncConfigs())
  }

  const loadRequestHistory = () => {
    setRequestHistory(apiService.getRequestHistory())
  }

  // ===== CONNECTION MANAGEMENT =====

  const testConnection = async (connectionId: string) => {
    try {
      const success = await apiService.testConnection(connectionId)
      
      addNotification({
        type: success ? 'success' : 'error',
        title: 'Connection Test',
        message: success ? 'Connection successful' : 'Connection failed',
        duration: 3000,
      })
      
      loadConnections()
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Connection Error',
        message: (error as Error).message,
        duration: 5000,
      })
    }
  }

  // ===== PLUGIN MANAGEMENT =====

  const activatePlugin = async (pluginId: string) => {
    try {
      await pluginManager.activatePlugin(pluginId)
      
      addNotification({
        type: 'success',
        title: 'Plugin Activated',
        message: 'Plugin has been successfully activated',
        duration: 3000,
      })
      
      loadPlugins()
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Plugin Activation Failed',
        message: (error as Error).message,
        duration: 5000,
      })
    }
  }

  const deactivatePlugin = async (pluginId: string) => {
    try {
      await pluginManager.deactivatePlugin(pluginId)
      
      addNotification({
        type: 'info',
        title: 'Plugin Deactivated',
        message: 'Plugin has been deactivated',
        duration: 3000,
      })
      
      loadPlugins()
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Plugin Deactivation Failed',
        message: (error as Error).message,
        duration: 5000,
      })
    }
  }

  // ===== DATA SYNC =====

  const executeSync = async (syncId: string) => {
    try {
      await apiService.executeSyncConfig(syncId)
      
      addNotification({
        type: 'success',
        title: 'Sync Complete',
        message: 'Data synchronization completed successfully',
        duration: 3000,
      })
      
      loadSyncConfigs()
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Sync Failed',
        message: (error as Error).message,
        duration: 5000,
      })
    }
  }

  // ===== RENDER HELPERS =====

  const getConnectionStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'disconnected': return <XCircle className="h-5 w-5 text-gray-400" />
      case 'error': return <AlertTriangle className="h-5 w-5 text-red-500" />
      case 'pending': return <Clock className="h-5 w-5 text-yellow-500" />
      default: return <XCircle className="h-5 w-5 text-gray-400" />
    }
  }

  const getPluginStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100'
      case 'inactive': return 'text-gray-600 bg-gray-100'
      case 'error': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'ai': return <Zap className="h-4 w-4" />
      case 'productivity': return <BarChart3 className="h-4 w-4" />
      case 'integration': return <Link className="h-4 w-4" />
      case 'visualization': return <Database className="h-4 w-4" />
      case 'utility': return <Settings className="h-4 w-4" />
      default: return <Plug className="h-4 w-4" />
    }
  }

  const filteredPlugins = plugins.filter(plugin => {
    const matchesSearch = plugin.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         plugin.description.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesCategory = filterCategory === 'all' || plugin.category === filterCategory
    return matchesSearch && matchesCategory
  })

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Link className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">API Integration & Plugins</h2>
          </div>
          
          <div className="flex items-center space-x-2 text-white text-sm">
            <div className="flex items-center space-x-1">
              <CheckCircle className="h-4 w-4" />
              <span>{connections.filter(c => c.status === 'connected').length} Connected</span>
            </div>
            <div className="flex items-center space-x-1">
              <Zap className="h-4 w-4" />
              <span>{plugins.filter(p => p.status === 'active').length} Active Plugins</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'connections', label: 'API Connections', icon: <Link className="h-4 w-4" /> },
          { id: 'plugins', label: 'Plugins', icon: <Plug className="h-4 w-4" /> },
          { id: 'sync', label: 'Data Sync', icon: <RefreshCw className="h-4 w-4" /> },
          { id: 'requests', label: 'Request History', icon: <BarChart3 className="h-4 w-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-indigo-600 border-b-2 border-indigo-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {/* API Connections Tab */}
            {activeTab === 'connections' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">API Connections</h3>
                  <button className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center space-x-2">
                    <Plus className="h-4 w-4" />
                    <span>Add Connection</span>
                  </button>
                </div>

                <div className="grid gap-4">
                  {connections.map((connection) => (
                    <motion.div
                      key={connection.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                        selectedConnection === connection.id ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200'
                      }`}
                      onClick={() => setSelectedConnection(selectedConnection === connection.id ? null : connection.id)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          {getConnectionStatusIcon(connection.status)}
                          <div>
                            <h4 className="font-semibold text-gray-900">{connection.name}</h4>
                            <p className="text-sm text-gray-600">{connection.type.toUpperCase()} • {connection.metadata.category}</p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              testConnection(connection.id)
                            }}
                            className="p-2 text-gray-400 hover:text-indigo-600 transition-colors"
                            title="Test Connection"
                          >
                            <Zap className="h-4 w-4" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              // Open settings
                            }}
                            className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                            title="Settings"
                          >
                            <Settings className="h-4 w-4" />
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Base URL:</span>
                          <p className="font-mono text-xs text-gray-900 truncate">{connection.baseUrl}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Auth Type:</span>
                          <p className="text-gray-900">{connection.authentication.type}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Rate Limit:</span>
                          <p className="text-gray-900">{connection.rateLimit.requests}/min</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Last Sync:</span>
                          <p className="text-gray-900">
                            {connection.lastSync ? new Date(connection.lastSync).toLocaleString() : 'Never'}
                          </p>
                        </div>
                      </div>

                      {selectedConnection === connection.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-4 pt-4 border-t border-gray-200"
                        >
                          <h5 className="font-medium text-gray-900 mb-2">Available Endpoints</h5>
                          <div className="space-y-2">
                            {apiService.getEndpoints(connection.id).map((endpoint) => (
                              <div key={endpoint.id} className="flex items-center justify-between bg-gray-50 rounded p-2">
                                <div>
                                  <span className="font-mono text-sm text-gray-900">{endpoint.method}</span>
                                  <span className="ml-2 text-sm text-gray-600">{endpoint.path}</span>
                                </div>
                                <span className="text-xs text-gray-500">{endpoint.description}</span>
                              </div>
                            ))}
                          </div>
                        </motion.div>
                      )}
                    </motion.div>
                  ))}
                </div>

                {connections.length === 0 && (
                  <div className="text-center py-12">
                    <Link className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No API connections configured</p>
                  </div>
                )}
              </div>
            )}

            {/* Plugins Tab */}
            {activeTab === 'plugins' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Plugin Management</h3>
                  <div className="flex items-center space-x-2">
                    <div className="relative">
                      <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                      <input
                        type="text"
                        placeholder="Search plugins..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                      />
                    </div>
                    <select
                      value={filterCategory}
                      onChange={(e) => setFilterCategory(e.target.value)}
                      className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    >
                      <option value="all">All Categories</option>
                      <option value="ai">AI</option>
                      <option value="productivity">Productivity</option>
                      <option value="integration">Integration</option>
                      <option value="visualization">Visualization</option>
                      <option value="utility">Utility</option>
                    </select>
                    <button className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center space-x-2">
                      <Download className="h-4 w-4" />
                      <span>Install Plugin</span>
                    </button>
                  </div>
                </div>

                <div className="grid gap-4">
                  {filteredPlugins.map((plugin) => (
                    <motion.div
                      key={plugin.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                        selectedPlugin === plugin.id ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200'
                      }`}
                      onClick={() => setSelectedPlugin(selectedPlugin === plugin.id ? null : plugin.id)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-gray-100 rounded-lg">
                            {getCategoryIcon(plugin.category)}
                          </div>
                          <div>
                            <h4 className="font-semibold text-gray-900">{plugin.name}</h4>
                            <p className="text-sm text-gray-600">v{plugin.version} by {plugin.author}</p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs rounded-full ${getPluginStatusColor(plugin.status)}`}>
                            {plugin.status}
                          </span>
                          
                          {plugin.status === 'active' ? (
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                deactivatePlugin(plugin.id)
                              }}
                              className="p-2 text-red-400 hover:text-red-600 transition-colors"
                              title="Deactivate"
                            >
                              <Pause className="h-4 w-4" />
                            </button>
                          ) : (
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                activatePlugin(plugin.id)
                              }}
                              className="p-2 text-green-400 hover:text-green-600 transition-colors"
                              title="Activate"
                            >
                              <Play className="h-4 w-4" />
                            </button>
                          )}
                          
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              // Open settings
                            }}
                            className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                            title="Settings"
                          >
                            <Settings className="h-4 w-4" />
                          </button>
                        </div>
                      </div>

                      <p className="text-gray-600 mb-3">{plugin.description}</p>

                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center space-x-4">
                          <span className="text-gray-600">Category: {plugin.category}</span>
                          <span className="text-gray-600">Permissions: {plugin.permissions.length}</span>
                        </div>
                        <span className="text-gray-500">
                          Installed: {new Date(plugin.installDate).toLocaleDateString()}
                        </span>
                      </div>

                      {selectedPlugin === plugin.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-4 pt-4 border-t border-gray-200"
                        >
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <h5 className="font-medium text-gray-900 mb-2">Permissions</h5>
                              <div className="space-y-1">
                                {plugin.permissions.map((permission, index) => (
                                  <div key={index} className="flex items-center space-x-2 text-sm">
                                    <Shield className="h-3 w-3 text-gray-400" />
                                    <span className="text-gray-600">{permission.description}</span>
                                    {permission.required && (
                                      <span className="text-red-500 text-xs">(Required)</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                            
                            <div>
                              <h5 className="font-medium text-gray-900 mb-2">Dependencies</h5>
                              {plugin.dependencies.length > 0 ? (
                                <div className="space-y-1">
                                  {plugin.dependencies.map((dep, index) => (
                                    <div key={index} className="text-sm text-gray-600">
                                      • {dep}
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-sm text-gray-500">No dependencies</p>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </motion.div>
                  ))}
                </div>

                {filteredPlugins.length === 0 && (
                  <div className="text-center py-12">
                    <Plug className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">
                      {searchQuery || filterCategory !== 'all' 
                        ? 'No plugins match your search criteria' 
                        : 'No plugins installed'
                      }
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Data Sync Tab */}
            {activeTab === 'sync' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Data Synchronization</h3>
                  <button className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center space-x-2">
                    <Plus className="h-4 w-4" />
                    <span>Create Sync</span>
                  </button>
                </div>

                <div className="grid gap-4">
                  {syncConfigs.map((config) => (
                    <motion.div
                      key={config.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="border border-gray-200 rounded-lg p-4"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h4 className="font-semibold text-gray-900">{config.name}</h4>
                          <p className="text-sm text-gray-600">
                            {config.schedule.type} • {config.enabled ? 'Enabled' : 'Disabled'}
                          </p>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => executeSync(config.id)}
                            className="p-2 text-indigo-400 hover:text-indigo-600 transition-colors"
                            title="Run Sync"
                          >
                            <Play className="h-4 w-4" />
                          </button>
                          <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                            <Settings className="h-4 w-4" />
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Source:</span>
                          <p className="text-gray-900">{config.sourceEndpoint}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Transformations:</span>
                          <p className="text-gray-900">{config.transformations.length}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Last Run:</span>
                          <p className="text-gray-900">
                            {config.lastRun ? new Date(config.lastRun).toLocaleString() : 'Never'}
                          </p>
                        </div>
                        <div>
                          <span className="text-gray-600">Next Run:</span>
                          <p className="text-gray-900">
                            {config.nextRun ? new Date(config.nextRun).toLocaleString() : 'Manual'}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>

                {syncConfigs.length === 0 && (
                  <div className="text-center py-12">
                    <RefreshCw className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No data sync configurations</p>
                  </div>
                )}
              </div>
            )}

            {/* Request History Tab */}
            {activeTab === 'requests' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">API Request History</h3>
                  <button
                    onClick={loadRequestHistory}
                    className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                    title="Refresh"
                  >
                    <RefreshCw className="h-4 w-4" />
                  </button>
                </div>

                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {requestHistory.map((request) => (
                    <motion.div
                      key={request.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`border-l-4 pl-4 py-2 ${
                        request.status === 'success' ? 'border-green-500 bg-green-50' :
                        request.status === 'error' ? 'border-red-500 bg-red-50' :
                        request.status === 'pending' ? 'border-yellow-500 bg-yellow-50' :
                        'border-gray-500 bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-medium text-gray-900">{request.endpointId}</span>
                          <span className="ml-2 text-sm text-gray-600">
                            {new Date(request.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2 text-sm">
                          {request.duration && (
                            <span className="text-gray-500">{request.duration}ms</span>
                          )}
                          <span className={`px-2 py-1 rounded text-xs ${
                            request.status === 'success' ? 'bg-green-100 text-green-800' :
                            request.status === 'error' ? 'bg-red-100 text-red-800' :
                            request.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {request.status}
                          </span>
                        </div>
                      </div>
                      
                      {request.error && (
                        <p className="text-sm text-red-600 mt-1">{request.error}</p>
                      )}
                    </motion.div>
                  ))}
                </div>

                {requestHistory.length === 0 && (
                  <div className="text-center py-12">
                    <BarChart3 className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No API requests recorded</p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

export default APIIntegration
