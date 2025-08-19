'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  MessageCircle, 
  Clock, 
  Activity,
  Zap,
  Target,
  Eye,
  Download,
  Filter,
  Calendar,
  RefreshCw
} from 'lucide-react'
import { useUIActions } from '../store'

// ===== TYPES =====

interface AnalyticsProps {
  sessionId?: string
  className?: string
}

interface AnalyticsMetric {
  id: string
  name: string
  value: number
  change: number
  trend: 'up' | 'down' | 'stable'
  icon: React.ReactNode
  color: string
}

interface UsageData {
  date: string
  messages: number
  users: number
  sessions: number
  avgResponseTime: number
}

interface UserBehavior {
  userId: string
  name: string
  totalMessages: number
  avgSessionTime: number
  lastActive: Date
  preferredFeatures: string[]
}

// ===== ANALYTICS DASHBOARD =====

export const AnalyticsDashboard: React.FC<AnalyticsProps> = ({ 
  sessionId, 
  className = '' 
}) => {
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | '90d'>('7d')
  const [isLoading, setIsLoading] = useState(false)
  const [metrics, setMetrics] = useState<AnalyticsMetric[]>([])
  const [usageData, setUsageData] = useState<UsageData[]>([])
  const [userBehavior, setUserBehavior] = useState<UserBehavior[]>([])
  const [activeTab, setActiveTab] = useState<'overview' | 'usage' | 'users' | 'performance'>('overview')

  const { addNotification } = useUIActions()

  // Mock analytics data
  useEffect(() => {
    setIsLoading(true)
    
    // Simulate API call
    setTimeout(() => {
      setMetrics([
        {
          id: 'total-messages',
          name: 'Total Messages',
          value: 1247,
          change: 12.5,
          trend: 'up',
          icon: <MessageCircle className="h-5 w-5" />,
          color: 'text-blue-600',
        },
        {
          id: 'active-users',
          name: 'Active Users',
          value: 89,
          change: 8.2,
          trend: 'up',
          icon: <Users className="h-5 w-5" />,
          color: 'text-green-600',
        },
        {
          id: 'avg-response-time',
          name: 'Avg Response Time',
          value: 1.2,
          change: -5.3,
          trend: 'down',
          icon: <Zap className="h-5 w-5" />,
          color: 'text-yellow-600',
        },
        {
          id: 'session-duration',
          name: 'Avg Session Duration',
          value: 15.8,
          change: 3.7,
          trend: 'up',
          icon: <Clock className="h-5 w-5" />,
          color: 'text-purple-600',
        },
      ])

      setUsageData([
        { date: '2024-01-01', messages: 145, users: 23, sessions: 34, avgResponseTime: 1.1 },
        { date: '2024-01-02', messages: 167, users: 28, sessions: 41, avgResponseTime: 1.3 },
        { date: '2024-01-03', messages: 189, users: 31, sessions: 45, avgResponseTime: 1.2 },
        { date: '2024-01-04', messages: 203, users: 35, sessions: 52, avgResponseTime: 1.0 },
        { date: '2024-01-05', messages: 178, users: 29, sessions: 38, avgResponseTime: 1.4 },
        { date: '2024-01-06', messages: 234, users: 42, sessions: 58, avgResponseTime: 1.1 },
        { date: '2024-01-07', messages: 256, users: 48, sessions: 63, avgResponseTime: 0.9 },
      ])

      setUserBehavior([
        {
          userId: 'user-1',
          name: 'Alice Johnson',
          totalMessages: 156,
          avgSessionTime: 18.5,
          lastActive: new Date(Date.now() - 3600000),
          preferredFeatures: ['Voice Chat', 'File Upload', 'Code Analysis'],
        },
        {
          userId: 'user-2',
          name: 'Bob Smith',
          totalMessages: 89,
          avgSessionTime: 12.3,
          lastActive: new Date(Date.now() - 7200000),
          preferredFeatures: ['Text Chat', 'Document Analysis'],
        },
        {
          userId: 'user-3',
          name: 'Carol Davis',
          totalMessages: 234,
          avgSessionTime: 25.7,
          lastActive: new Date(Date.now() - 1800000),
          preferredFeatures: ['Voice Chat', 'Real-time Collaboration', 'Analytics'],
        },
      ])

      setIsLoading(false)
    }, 1000)
  }, [timeRange])

  const refreshData = () => {
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
      addNotification({
        type: 'success',
        title: 'Data Refreshed',
        message: 'Analytics data has been updated',
        duration: 2000,
      })
    }, 1000)
  }

  const exportData = () => {
    // Simulate data export
    addNotification({
      type: 'info',
      title: 'Export Started',
      message: 'Analytics data is being prepared for download',
      duration: 3000,
    })
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="h-4 w-4 text-green-500" />
      case 'down':
        return <TrendingUp className="h-4 w-4 text-red-500 rotate-180" />
      default:
        return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const formatValue = (value: number, type: string) => {
    switch (type) {
      case 'avg-response-time':
        return `${value}s`
      case 'session-duration':
        return `${value}m`
      default:
        return value.toLocaleString()
    }
  }

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <BarChart3 className="h-5 w-5 text-indigo-600" />
          <h2 className="text-xl font-semibold text-gray-900">Analytics Dashboard</h2>
        </div>
        
        <div className="flex items-center space-x-2">
          {/* Time Range Selector */}
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-3 py-1 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
          
          <button
            onClick={refreshData}
            disabled={isLoading}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
            title="Refresh data"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          
          <button
            onClick={exportData}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
            title="Export data"
          >
            <Download className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'overview', label: 'Overview', icon: <Eye className="h-4 w-4" /> },
          { id: 'usage', label: 'Usage', icon: <Activity className="h-4 w-4" /> },
          { id: 'users', label: 'Users', icon: <Users className="h-4 w-4" /> },
          { id: 'performance', label: 'Performance', icon: <Zap className="h-4 w-4" /> },
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
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
            <span className="ml-2 text-gray-600">Loading analytics...</span>
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
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  {/* Key Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {metrics.map((metric) => (
                      <div key={metric.id} className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div className={metric.color}>
                            {metric.icon}
                          </div>
                          {getTrendIcon(metric.trend)}
                        </div>
                        <div className="mt-2">
                          <p className="text-2xl font-bold text-gray-900">
                            {formatValue(metric.value, metric.id)}
                          </p>
                          <p className="text-sm text-gray-600">{metric.name}</p>
                          <p className={`text-xs mt-1 ${
                            metric.trend === 'up' ? 'text-green-600' : 
                            metric.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                          }`}>
                            {metric.change > 0 ? '+' : ''}{metric.change}% from last period
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Usage Chart Placeholder */}
                  <div className="bg-gray-50 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Usage Trends</h3>
                    <div className="h-64 flex items-center justify-center text-gray-500">
                      <div className="text-center">
                        <BarChart3 className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                        <p>Interactive charts would be displayed here</p>
                        <p className="text-sm">Integration with charting library needed</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'usage' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-gray-900">Usage Statistics</h3>
                  
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Date
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Messages
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Users
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Sessions
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Avg Response Time
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {usageData.map((data, index) => (
                          <tr key={index}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {new Date(data.date).toLocaleDateString()}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {data.messages}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {data.users}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {data.sessions}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {data.avgResponseTime}s
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {activeTab === 'users' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-gray-900">User Behavior</h3>
                  
                  <div className="grid gap-4">
                    {userBehavior.map((user) => (
                      <div key={user.userId} className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white font-medium">
                              {user.name.charAt(0)}
                            </div>
                            <div>
                              <p className="font-medium text-gray-900">{user.name}</p>
                              <p className="text-sm text-gray-500">
                                Last active: {user.lastActive.toLocaleString()}
                              </p>
                            </div>
                          </div>
                          
                          <div className="text-right">
                            <p className="text-sm text-gray-600">
                              {user.totalMessages} messages
                            </p>
                            <p className="text-sm text-gray-600">
                              {user.avgSessionTime}m avg session
                            </p>
                          </div>
                        </div>
                        
                        <div className="mt-3">
                          <p className="text-sm text-gray-600 mb-2">Preferred Features:</p>
                          <div className="flex flex-wrap gap-1">
                            {user.preferredFeatures.map((feature, index) => (
                              <span
                                key={index}
                                className="px-2 py-1 bg-indigo-100 text-indigo-800 text-xs rounded-full"
                              >
                                {feature}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === 'performance' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-gray-900">Performance Metrics</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">Response Times</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Average</span>
                          <span className="text-sm font-medium">1.2s</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">95th Percentile</span>
                          <span className="text-sm font-medium">2.8s</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">99th Percentile</span>
                          <span className="text-sm font-medium">4.1s</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">System Health</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Uptime</span>
                          <span className="text-sm font-medium text-green-600">99.9%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Error Rate</span>
                          <span className="text-sm font-medium text-green-600">0.1%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">CPU Usage</span>
                          <span className="text-sm font-medium">45%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  )
}

export default AnalyticsDashboard
