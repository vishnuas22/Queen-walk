'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Cloud, 
  Building, 
  Zap, 
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity,
  Shield,
  Database,
  Settings,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Play,
  Pause,
  RefreshCw,
  BarChart3,
  PieChart,
  Users,
  Lock,
  Globe,
  Server
} from 'lucide-react'
import { 
  EnterpriseConnectorService, 
  CloudServiceManager,
  type EnterpriseConnector, 
  type CloudService,
  type SyncResult
} from '../lib/enterprise-connectors'
import { useUIActions } from '../store'

// ===== TYPES =====

interface EnterpriseCloudProps {
  onConnectorChange?: (connector: EnterpriseConnector) => void
  onServiceChange?: (service: CloudService) => void
  className?: string
}

// ===== ENTERPRISE CLOUD COMPONENT =====

export const EnterpriseCloud: React.FC<EnterpriseCloudProps> = ({
  onConnectorChange,
  onServiceChange,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'connectors' | 'cloud' | 'usage' | 'billing'>('connectors')
  const [connectors, setConnectors] = useState<EnterpriseConnector[]>([])
  const [cloudServices, setCloudServices] = useState<CloudService[]>([])
  const [syncResults, setSyncResults] = useState<SyncResult[]>([])
  const [selectedConnector, setSelectedConnector] = useState<string | null>(null)
  const [selectedService, setSelectedService] = useState<string | null>(null)
  const [filterType, setFilterType] = useState<string>('all')

  const { addNotification } = useUIActions()
  const enterpriseService = EnterpriseConnectorService.getInstance()
  const cloudManager = CloudServiceManager.getInstance()

  // Initialize data
  useEffect(() => {
    loadConnectors()
    loadCloudServices()

    // Set up event listeners
    const unsubscribeConnector = enterpriseService.onConnection((connector) => {
      setConnectors(prev => prev.map(c => c.id === connector.id ? connector : c))
      onConnectorChange?.(connector)
    })

    const unsubscribeSync = enterpriseService.onSync((result) => {
      setSyncResults(prev => [result, ...prev.slice(0, 19)])
    })

    const unsubscribeService = cloudManager.onService((service) => {
      setCloudServices(prev => prev.map(s => s.id === service.id ? service : s))
      onServiceChange?.(service)
    })

    return () => {
      unsubscribeConnector()
      unsubscribeSync()
      unsubscribeService()
    }
  }, [onConnectorChange, onServiceChange])

  const loadConnectors = () => {
    setConnectors(enterpriseService.getConnectors())
  }

  const loadCloudServices = () => {
    setCloudServices(cloudManager.getServices())
  }

  // ===== CONNECTOR MANAGEMENT =====

  const connectEnterprise = async (connectorId: string) => {
    try {
      const success = await enterpriseService.connectEnterprise(connectorId)
      
      addNotification({
        type: success ? 'success' : 'error',
        title: 'Enterprise Connection',
        message: success ? 'Connected successfully' : 'Connection failed',
        duration: 3000,
      })
      
      loadConnectors()
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Connection Error',
        message: (error as Error).message,
        duration: 5000,
      })
    }
  }

  const syncEnterpriseData = async (connectorId: string) => {
    try {
      const result = await enterpriseService.syncEnterpriseData(connectorId)
      
      addNotification({
        type: result.status === 'success' ? 'success' : 'warning',
        title: 'Data Sync Complete',
        message: `Processed ${result.recordsProcessed} records`,
        duration: 3000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Sync Failed',
        message: (error as Error).message,
        duration: 5000,
      })
    }
  }

  // ===== CLOUD SERVICE MANAGEMENT =====

  const activateCloudService = async (serviceId: string) => {
    try {
      const success = await cloudManager.activateService(serviceId)
      
      addNotification({
        type: success ? 'success' : 'error',
        title: 'Cloud Service',
        message: success ? 'Service activated' : 'Activation failed',
        duration: 3000,
      })
      
      loadCloudServices()
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Service Error',
        message: (error as Error).message,
        duration: 5000,
      })
    }
  }

  // ===== RENDER HELPERS =====

  const getConnectorStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'disconnected': return <XCircle className="h-5 w-5 text-gray-400" />
      case 'error': return <AlertTriangle className="h-5 w-5 text-red-500" />
      case 'configuring': return <Clock className="h-5 w-5 text-yellow-500" />
      default: return <XCircle className="h-5 w-5 text-gray-400" />
    }
  }

  const getServiceStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100'
      case 'inactive': return 'text-gray-600 bg-gray-100'
      case 'error': return 'text-red-600 bg-red-100'
      case 'configuring': return 'text-yellow-600 bg-yellow-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'erp': return <Building className="h-4 w-4" />
      case 'crm': return <Users className="h-4 w-4" />
      case 'hrms': return <Users className="h-4 w-4" />
      case 'bi': return <BarChart3 className="h-4 w-4" />
      case 'ai': return <Zap className="h-4 w-4" />
      case 'storage': return <Database className="h-4 w-4" />
      case 'security': return <Shield className="h-4 w-4" />
      default: return <Server className="h-4 w-4" />
    }
  }

  const getProviderIcon = (provider: string) => {
    switch (provider) {
      case 'aws': return <Cloud className="h-4 w-4 text-orange-500" />
      case 'azure': return <Cloud className="h-4 w-4 text-blue-500" />
      case 'gcp': return <Cloud className="h-4 w-4 text-green-500" />
      case 'openai': return <Zap className="h-4 w-4 text-purple-500" />
      default: return <Cloud className="h-4 w-4 text-gray-500" />
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount)
  }

  const filteredConnectors = connectors.filter(connector => 
    filterType === 'all' || connector.type === filterType
  )

  const filteredServices = cloudServices.filter(service => 
    filterType === 'all' || service.category === filterType
  )

  const totalCloudCost = cloudManager.getTotalCost()
  const totalBudget = cloudManager.getTotalBudget()
  const budgetUtilization = totalBudget > 0 ? (totalCloudCost / totalBudget) * 100 : 0

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Building className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">Enterprise & Cloud Services</h2>
          </div>
          
          <div className="flex items-center space-x-4 text-white text-sm">
            <div className="flex items-center space-x-1">
              <CheckCircle className="h-4 w-4" />
              <span>{connectors.filter(c => c.status === 'connected').length} Connected</span>
            </div>
            <div className="flex items-center space-x-1">
              <Cloud className="h-4 w-4" />
              <span>{cloudServices.filter(s => s.status === 'active').length} Active</span>
            </div>
            <div className="flex items-center space-x-1">
              <DollarSign className="h-4 w-4" />
              <span>{formatCurrency(totalCloudCost)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'connectors', label: 'Enterprise Connectors', icon: <Building className="h-4 w-4" /> },
          { id: 'cloud', label: 'Cloud Services', icon: <Cloud className="h-4 w-4" /> },
          { id: 'usage', label: 'Usage Analytics', icon: <BarChart3 className="h-4 w-4" /> },
          { id: 'billing', label: 'Cost Management', icon: <DollarSign className="h-4 w-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-blue-600 border-b-2 border-blue-600'
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
            {/* Enterprise Connectors Tab */}
            {activeTab === 'connectors' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Enterprise System Connectors</h3>
                  <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="all">All Systems</option>
                    <option value="erp">ERP</option>
                    <option value="crm">CRM</option>
                    <option value="hrms">HRMS</option>
                    <option value="bi">Business Intelligence</option>
                  </select>
                </div>

                <div className="grid gap-4">
                  {filteredConnectors.map((connector) => (
                    <motion.div
                      key={connector.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                        selectedConnector === connector.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                      }`}
                      onClick={() => setSelectedConnector(selectedConnector === connector.id ? null : connector.id)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          {getConnectorStatusIcon(connector.status)}
                          <div className="flex items-center space-x-2">
                            {getTypeIcon(connector.type)}
                            <div>
                              <h4 className="font-semibold text-gray-900">{connector.name}</h4>
                              <p className="text-sm text-gray-600">{connector.vendor} • {connector.version}</p>
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {connector.status === 'connected' ? (
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                syncEnterpriseData(connector.id)
                              }}
                              className="p-2 text-blue-400 hover:text-blue-600 transition-colors"
                              title="Sync Data"
                            >
                              <RefreshCw className="h-4 w-4" />
                            </button>
                          ) : (
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                connectEnterprise(connector.id)
                              }}
                              className="p-2 text-green-400 hover:text-green-600 transition-colors"
                              title="Connect"
                            >
                              <Play className="h-4 w-4" />
                            </button>
                          )}
                          <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                            <Settings className="h-4 w-4" />
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Type:</span>
                          <p className="text-gray-900 capitalize">{connector.type}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Auth:</span>
                          <p className="text-gray-900">{connector.config.authentication.type}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Capabilities:</span>
                          <p className="text-gray-900">{connector.capabilities.length}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Last Sync:</span>
                          <p className="text-gray-900">
                            {connector.lastSync ? new Date(connector.lastSync).toLocaleDateString() : 'Never'}
                          </p>
                        </div>
                      </div>

                      {selectedConnector === connector.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-4 pt-4 border-t border-gray-200"
                        >
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <h5 className="font-medium text-gray-900 mb-2">Capabilities</h5>
                              <div className="space-y-1">
                                {connector.capabilities.map((capability, index) => (
                                  <div key={index} className="text-sm">
                                    <span className="font-medium text-gray-700">{capability.name}</span>
                                    <span className="ml-2 text-gray-500">({capability.type})</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                            
                            <div>
                              <h5 className="font-medium text-gray-900 mb-2">Security & Compliance</h5>
                              <div className="flex flex-wrap gap-1">
                                {Object.entries(connector.config.security.compliance).map(([key, enabled]) => (
                                  enabled && (
                                    <span
                                      key={key}
                                      className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded uppercase"
                                    >
                                      {key}
                                    </span>
                                  )
                                ))}
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </motion.div>
                  ))}
                </div>

                {filteredConnectors.length === 0 && (
                  <div className="text-center py-12">
                    <Building className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No enterprise connectors match your filter</p>
                  </div>
                )}
              </div>
            )}

            {/* Cloud Services Tab */}
            {activeTab === 'cloud' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Cloud AI Services</h3>
                  <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="all">All Services</option>
                    <option value="ai">AI Services</option>
                    <option value="storage">Storage</option>
                    <option value="compute">Compute</option>
                    <option value="analytics">Analytics</option>
                  </select>
                </div>

                <div className="grid gap-4">
                  {filteredServices.map((service) => (
                    <motion.div
                      key={service.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                        selectedService === service.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                      }`}
                      onClick={() => setSelectedService(selectedService === service.id ? null : service.id)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          {getProviderIcon(service.provider)}
                          <div>
                            <h4 className="font-semibold text-gray-900">{service.name}</h4>
                            <p className="text-sm text-gray-600">{service.provider} • {service.config.region}</p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs rounded-full ${getServiceStatusColor(service.status)}`}>
                            {service.status}
                          </span>
                          
                          {service.status === 'active' ? (
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                cloudManager.deactivateService(service.id)
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
                                activateCloudService(service.id)
                              }}
                              className="p-2 text-green-400 hover:text-green-600 transition-colors"
                              title="Activate"
                            >
                              <Play className="h-4 w-4" />
                            </button>
                          )}
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Category:</span>
                          <p className="text-gray-900 capitalize">{service.category}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Current Cost:</span>
                          <p className="text-gray-900">{formatCurrency(service.billing.currentCost)}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Requests:</span>
                          <p className="text-gray-900">{service.usage.current.requests.toLocaleString()}</p>
                        </div>
                        <div>
                          <span className="text-gray-600">Last Activity:</span>
                          <p className="text-gray-900">
                            {service.lastActivity ? new Date(service.lastActivity).toLocaleDateString() : 'Never'}
                          </p>
                        </div>
                      </div>

                      {selectedService === service.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-4 pt-4 border-t border-gray-200"
                        >
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <h5 className="font-medium text-gray-900 mb-2">Capabilities</h5>
                              <div className="space-y-2">
                                {service.capabilities.map((capability, index) => (
                                  <div key={index} className="bg-gray-50 rounded p-2">
                                    <div className="font-medium text-sm text-gray-900">{capability.name}</div>
                                    <div className="text-xs text-gray-600">{capability.description}</div>
                                    <div className="text-xs text-gray-500 mt-1">
                                      {formatCurrency(capability.pricing.price)} per {capability.pricing.unit}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                            
                            <div>
                              <h5 className="font-medium text-gray-900 mb-2">Usage & Billing</h5>
                              <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                  <span className="text-gray-600">Current Cost:</span>
                                  <span className="font-medium">{formatCurrency(service.billing.currentCost)}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-600">Projected Cost:</span>
                                  <span className="font-medium">{formatCurrency(service.billing.projectedCost)}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-600">Budget:</span>
                                  <span className="font-medium">{formatCurrency(service.billing.budget)}</span>
                                </div>
                                <div className="mt-2">
                                  <div className="flex justify-between text-xs mb-1">
                                    <span>Budget Utilization</span>
                                    <span>{((service.billing.currentCost / service.billing.budget) * 100).toFixed(1)}%</span>
                                  </div>
                                  <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div
                                      className={`h-2 rounded-full ${
                                        (service.billing.currentCost / service.billing.budget) > 0.8 ? 'bg-red-500' :
                                        (service.billing.currentCost / service.billing.budget) > 0.6 ? 'bg-yellow-500' : 'bg-green-500'
                                      }`}
                                      style={{ width: `${Math.min((service.billing.currentCost / service.billing.budget) * 100, 100)}%` }}
                                    />
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </motion.div>
                  ))}
                </div>

                {filteredServices.length === 0 && (
                  <div className="text-center py-12">
                    <Cloud className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No cloud services match your filter</p>
                  </div>
                )}
              </div>
            )}

            {/* Usage Analytics Tab */}
            {activeTab === 'usage' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Usage Analytics</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-6 text-white">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-blue-100">Total Requests</p>
                        <p className="text-2xl font-bold">
                          {cloudServices.reduce((sum, s) => sum + s.usage.current.requests, 0).toLocaleString()}
                        </p>
                      </div>
                      <Activity className="h-8 w-8 text-blue-200" />
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-lg p-6 text-white">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-green-100">Active Services</p>
                        <p className="text-2xl font-bold">
                          {cloudServices.filter(s => s.status === 'active').length}
                        </p>
                      </div>
                      <CheckCircle className="h-8 w-8 text-green-200" />
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-6 text-white">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-purple-100">Connected Systems</p>
                        <p className="text-2xl font-bold">
                          {connectors.filter(c => c.status === 'connected').length}
                        </p>
                      </div>
                      <Building className="h-8 w-8 text-purple-200" />
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-6">
                  <h4 className="font-medium text-gray-900 mb-4">Recent Sync Results</h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {syncResults.map((result) => (
                      <div
                        key={result.id}
                        className={`flex items-center justify-between p-3 rounded border-l-4 ${
                          result.status === 'success' ? 'border-green-500 bg-green-50' :
                          result.status === 'partial' ? 'border-yellow-500 bg-yellow-50' :
                          'border-red-500 bg-red-50'
                        }`}
                      >
                        <div>
                          <p className="font-medium text-gray-900">{result.connectorId}</p>
                          <p className="text-sm text-gray-600">
                            {result.recordsProcessed} records • {result.performance.latency}ms
                          </p>
                        </div>
                        <div className="text-right">
                          <span className={`px-2 py-1 text-xs rounded ${
                            result.status === 'success' ? 'bg-green-100 text-green-800' :
                            result.status === 'partial' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {result.status}
                          </span>
                          <p className="text-xs text-gray-500 mt-1">
                            {new Date(result.endTime).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {syncResults.length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      <BarChart3 className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                      <p>No sync results available</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Billing Tab */}
            {activeTab === 'billing' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Cost Management</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-gray-600">Current Cost</p>
                        <p className="text-2xl font-bold text-gray-900">{formatCurrency(totalCloudCost)}</p>
                      </div>
                      <DollarSign className="h-8 w-8 text-blue-500" />
                    </div>
                  </div>
                  
                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-gray-600">Total Budget</p>
                        <p className="text-2xl font-bold text-gray-900">{formatCurrency(totalBudget)}</p>
                      </div>
                      <PieChart className="h-8 w-8 text-green-500" />
                    </div>
                  </div>
                  
                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-gray-600">Utilization</p>
                        <p className="text-2xl font-bold text-gray-900">{budgetUtilization.toFixed(1)}%</p>
                      </div>
                      {budgetUtilization > 80 ? (
                        <TrendingUp className="h-8 w-8 text-red-500" />
                      ) : (
                        <TrendingDown className="h-8 w-8 text-green-500" />
                      )}
                    </div>
                  </div>
                  
                  <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-gray-600">Projected</p>
                        <p className="text-2xl font-bold text-gray-900">
                          {formatCurrency(cloudServices.reduce((sum, s) => sum + s.billing.projectedCost, 0))}
                        </p>
                      </div>
                      <TrendingUp className="h-8 w-8 text-purple-500" />
                    </div>
                  </div>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h4 className="font-medium text-gray-900 mb-4">Cost Breakdown by Service</h4>
                  <div className="space-y-3">
                    {cloudServices
                      .filter(service => service.billing.currentCost > 0)
                      .sort((a, b) => b.billing.currentCost - a.billing.currentCost)
                      .map((service) => (
                        <div key={service.id} className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            {getProviderIcon(service.provider)}
                            <span className="font-medium text-gray-900">{service.name}</span>
                          </div>
                          <div className="flex items-center space-x-4">
                            <div className="text-right">
                              <p className="font-medium text-gray-900">{formatCurrency(service.billing.currentCost)}</p>
                              <p className="text-sm text-gray-500">
                                {((service.billing.currentCost / totalCloudCost) * 100).toFixed(1)}% of total
                              </p>
                            </div>
                            <div className="w-24 bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-blue-500 h-2 rounded-full"
                                style={{ width: `${(service.billing.currentCost / totalCloudCost) * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h4 className="font-medium text-gray-900 mb-4">Budget Alerts</h4>
                  <div className="space-y-2">
                    {cloudServices.flatMap(service => 
                      service.billing.alerts.map(alert => ({
                        ...alert,
                        serviceName: service.name,
                        serviceId: service.id
                      }))
                    ).map((alert, index) => (
                      <div
                        key={index}
                        className={`flex items-center justify-between p-3 rounded ${
                          alert.triggered ? 'bg-red-50 border border-red-200' : 'bg-gray-50 border border-gray-200'
                        }`}
                      >
                        <div className="flex items-center space-x-3">
                          {alert.triggered ? (
                            <AlertTriangle className="h-5 w-5 text-red-500" />
                          ) : (
                            <CheckCircle className="h-5 w-5 text-green-500" />
                          )}
                          <div>
                            <p className="font-medium text-gray-900">{alert.serviceName}</p>
                            <p className="text-sm text-gray-600">
                              {alert.type} alert at {formatCurrency(alert.threshold)}
                            </p>
                          </div>
                        </div>
                        <span className={`px-2 py-1 text-xs rounded ${
                          alert.triggered ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                        }`}>
                          {alert.triggered ? 'Triggered' : 'OK'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

export default EnterpriseCloud
