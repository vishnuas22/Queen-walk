'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Send, Mic, Upload, Settings, MessageSquare, Plus, Sidebar, X,
  Brain, Wand2, GitBranch, Glasses, Eye, Hand, Heart, LinkIcon,
  Building, BarChart3, Zap, FileText, Users, Activity
} from 'lucide-react'

// Import all the comprehensive AI components
import { QuantumIntelligence } from '../../components/QuantumIntelligence'
import { CreativeAI } from '../../components/CreativeAI'
import { DecisionSupport } from '../../components/DecisionSupport'
import { ARInterface } from '../../components/ARInterface'
import { AdvancedInput } from '../../components/AdvancedInput'
import { GestureRecognition } from '../../components/GestureRecognition'
import { EmotionRecognition } from '../../components/EmotionRecognition'
import { APIIntegration } from '../../components/APIIntegration'
import { EnterpriseCloud } from '../../components/EnterpriseCloud'
import AnalyticsDashboard from '../../components/Analytics'
import VoiceNavigation, { VoiceStatusIndicator } from '../../components/VoiceNavigation'
// import FileUpload from '../../components/FileUpload'
import { CollaborationPanel, CollaborationStatusBar } from '../../components/Collaboration'

// Import only essential React Query hooks for data management
import { useChatSessions, useFlattenedMessages, useSendMessage } from '../../hooks/useChatQueries'
// Temporarily removed WebSocket and complex features that depend on store
// import { useTypingIndicator } from '../../hooks/useWebSocket'

// Types
interface ChatSession {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
  duration?: number
}

// Comprehensive MasterX Chat Page with all 10 AI tools
export default function ChatPage() {
  // Simple React state instead of complex Zustand store
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [inputMessage, setInputMessage] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  // AI Tools visibility state
  const [showQuantumIntelligence, setShowQuantumIntelligence] = useState(false)
  const [showCreativeAI, setShowCreativeAI] = useState(false)
  const [showDecisionSupport, setShowDecisionSupport] = useState(false)
  const [showARInterface, setShowARInterface] = useState(false)
  const [showAdvancedInput, setShowAdvancedInput] = useState(false)
  const [showGestureRecognition, setShowGestureRecognition] = useState(false)
  const [showEmotionRecognition, setShowEmotionRecognition] = useState(false)
  const [showAPIIntegration, setShowAPIIntegration] = useState(false)
  const [showEnterpriseCloud, setShowEnterpriseCloud] = useState(false)
  const [showAnalyticsDashboard, setShowAnalyticsDashboard] = useState(false)

  // Simple notification state
  const [notifications, setNotifications] = useState<Notification[]>([])

  // Simple function to add notifications
  const addNotification = (notification: Omit<Notification, 'id'>) => {
    const id = Date.now().toString()
    const newNotification = { ...notification, id }
    setNotifications(prev => [...prev, newNotification])

    // Auto-remove after duration
    if (notification.duration) {
      setTimeout(() => {
        setNotifications(prev => prev.filter(n => n.id !== id))
      }, notification.duration)
    }
  }

  const inputRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Essential React Query hooks only - simplified approach
  let sessionsData = null
  let sessionsLoading = false
  let messages: any[] = []
  let messagesLoading = false
  let sendMessageMutation: any = null

  try {
    // Only use essential hooks that actually work
    const sessionsResult = useChatSessions()
    if (sessionsResult) {
      sessionsData = sessionsResult.data || null
      sessionsLoading = sessionsResult.isLoading || false
    }

    const messagesResult = useFlattenedMessages(currentSessionId)
    if (messagesResult) {
      messages = messagesResult.messages || []
      messagesLoading = messagesResult.isLoading || false
    }

    const sendResult = useSendMessage(currentSessionId)
    if (sendResult) {
      sendMessageMutation = sendResult
    }

  } catch (error) {
    console.error('React Query hooks failed:', error)
    // Use fallback values already set above
  }

  // Simplified typing indicator (no WebSocket dependency)
  const typingUsers: any[] = []

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Removed background sync and cache warming to eliminate store dependencies

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const content = inputMessage.trim()
    setIsLoading(true)

    try {
      if (sendMessageMutation && sendMessageMutation.mutate) {
        // Use React Query mutation
        sendMessageMutation.mutate(content, {
          onSuccess: () => {
            setInputMessage('') // Clear input after successful send
            addNotification({
              type: 'success',
              title: 'Message Sent',
              message: 'Your message has been sent successfully.',
              duration: 2000,
            })
          },
          onError: (error: any) => {
            console.error('Error sending message:', error)
            addNotification({
              type: 'error',
              title: 'Message Failed',
              message: 'Failed to send message. Please try again.',
              duration: 5000,
            })
          }
        })
      } else {
        // Fallback: just add message to local state for demo
        setInputMessage('')
        addNotification({
          type: 'info',
          title: 'Demo Mode',
          message: 'Message sent in demo mode (no backend connection).',
          duration: 3000,
        })
      }
    } catch (error) {
      console.error('Error sending message:', error)
      addNotification({
        type: 'error',
        title: 'Message Failed',
        message: 'Failed to send message. Please try again.',
        duration: 5000,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const startNewChat = () => {
    setCurrentSessionId(null)
    setSidebarOpen(false)
    setInputMessage('')
  }

  const loadSession = (session: ChatSession) => {
    setCurrentSessionId(session.session_id)
    setSidebarOpen(false)
  }

  // AI Tools configuration
  const aiTools = [
    {
      icon: Brain,
      name: 'Quantum Intelligence',
      color: 'indigo',
      show: showQuantumIntelligence,
      setShow: setShowQuantumIntelligence,
      description: 'Advanced quantum reasoning and problem-solving'
    },
    {
      icon: Wand2,
      name: 'Creative AI',
      color: 'purple',
      show: showCreativeAI,
      setShow: setShowCreativeAI,
      description: 'Generate creative content and innovative ideas'
    },
    {
      icon: GitBranch,
      name: 'Decision Support',
      color: 'blue',
      show: showDecisionSupport,
      setShow: setShowDecisionSupport,
      description: 'Analyze options and make informed decisions'
    },
    {
      icon: Glasses,
      name: 'AR Interface',
      color: 'green',
      show: showARInterface,
      setShow: setShowARInterface,
      description: 'Augmented reality visualization and interaction'
    },
    {
      icon: Eye,
      name: 'Advanced Input',
      color: 'yellow',
      show: showAdvancedInput,
      setShow: setShowAdvancedInput,
      description: 'Multi-modal input with voice, gesture, and emotion'
    },
    {
      icon: Hand,
      name: 'Gesture Recognition',
      color: 'pink',
      show: showGestureRecognition,
      setShow: setShowGestureRecognition,
      description: 'Hand gesture control and navigation'
    },
    {
      icon: Heart,
      name: 'Emotion Recognition',
      color: 'red',
      show: showEmotionRecognition,
      setShow: setShowEmotionRecognition,
      description: 'Emotional intelligence and mood analysis'
    },
    {
      icon: LinkIcon,
      name: 'API Integration',
      color: 'cyan',
      show: showAPIIntegration,
      setShow: setShowAPIIntegration,
      description: 'Connect with external services and APIs'
    },
    {
      icon: Building,
      name: 'Enterprise Cloud',
      color: 'gray',
      show: showEnterpriseCloud,
      setShow: setShowEnterpriseCloud,
      description: 'Enterprise-grade cloud services and connectors'
    },
    {
      icon: BarChart3,
      name: 'Analytics',
      color: 'orange',
      show: showAnalyticsDashboard,
      setShow: setShowAnalyticsDashboard,
      description: 'Advanced analytics and performance insights'
    }
  ]

  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex overflow-hidden">
      {/* Voice Status Indicator */}
      <VoiceStatusIndicator />

      {/* Collaboration Status Bar */}
      <CollaborationStatusBar />

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -320 }}
            animate={{ x: 0 }}
            exit={{ x: -320 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="w-80 bg-white/80 backdrop-blur-lg border-r border-gray-200/50 flex flex-col shadow-xl"
          >
            <div className="p-6 border-b border-gray-200/50">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    MasterX
                  </h2>
                  <p className="text-sm text-gray-600">Quantum Intelligence Platform</p>
                </div>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto">
              {/* New Chat Button */}
              <div className="p-4 border-b border-gray-200/50">
                <button
                  onClick={startNewChat}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left text-gray-700 hover:bg-blue-50 rounded-xl transition-colors group"
                >
                  <div className="p-2 bg-blue-100 group-hover:bg-blue-200 rounded-lg transition-colors">
                    <Plus className="w-4 h-4 text-blue-600" />
                  </div>
                  <span className="font-medium">New Chat</span>
                </button>
              </div>

              {/* Chat Sessions */}
              <div className="p-4">
                <h3 className="text-sm font-semibold text-gray-900 mb-3">Recent Chats</h3>
                {sessionsLoading ? (
                  <div className="space-y-2">
                    {[...Array(3)].map((_, i) => (
                      <div key={i} className="h-12 bg-gray-200 rounded-lg animate-pulse" />
                    ))}
                  </div>
                ) : sessionsData && Array.isArray(sessionsData) && sessionsData.length > 0 ? (
                  <div className="space-y-2">
                    {sessionsData.map((session: ChatSession) => (
                      <button
                        key={session.session_id}
                        onClick={() => loadSession(session)}
                        className={`w-full text-left p-3 rounded-lg transition-colors ${
                          currentSessionId === session.session_id
                            ? 'bg-blue-100 text-blue-900'
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        <div className="font-medium text-sm truncate">{session.title}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          {session.message_count} messages
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No previous chats</p>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white/80 backdrop-blur-lg border-b border-gray-200/50 px-6 py-4 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Sidebar className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                MasterX AI
              </h1>
              <p className="text-sm text-gray-600">Quantum Intelligence Platform</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <VoiceNavigation className="mr-2" />
            <button
              onClick={() => setShowAnalyticsDashboard(true)}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
              title="Analytics Dashboard"
            >
              <Activity className="w-5 h-5" />
            </button>
            <button className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* AI Tools Grid */}
        <div className="bg-white/50 backdrop-blur-sm border-b border-gray-200/50 px-6 py-4">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-semibold text-gray-900">AI Tools</span>
          </div>
          <div className="grid grid-cols-5 gap-3">
            {aiTools.map((tool, index) => (
              <motion.button
                key={tool.name}
                onClick={() => tool.setShow(true)}
                className={`group relative p-3 rounded-xl transition-all duration-200 hover:scale-105 ${
                  tool.color === 'indigo' ? 'bg-indigo-100 hover:bg-indigo-200 text-indigo-700' :
                  tool.color === 'purple' ? 'bg-purple-100 hover:bg-purple-200 text-purple-700' :
                  tool.color === 'blue' ? 'bg-blue-100 hover:bg-blue-200 text-blue-700' :
                  tool.color === 'green' ? 'bg-green-100 hover:bg-green-200 text-green-700' :
                  tool.color === 'yellow' ? 'bg-yellow-100 hover:bg-yellow-200 text-yellow-700' :
                  tool.color === 'pink' ? 'bg-pink-100 hover:bg-pink-200 text-pink-700' :
                  tool.color === 'red' ? 'bg-red-100 hover:bg-red-200 text-red-700' :
                  tool.color === 'cyan' ? 'bg-cyan-100 hover:bg-cyan-200 text-cyan-700' :
                  tool.color === 'gray' ? 'bg-gray-100 hover:bg-gray-200 text-gray-700' :
                  'bg-orange-100 hover:bg-orange-200 text-orange-700'
                }`}
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.95 }}
                title={tool.description}
              >
                <tool.icon className="w-5 h-5 mx-auto mb-1" />
                <div className="text-xs font-medium text-center">{tool.name}</div>

                {/* Tooltip */}
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
                  {tool.description}
                </div>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {!messages || messages.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg">
                <MessageSquare className="w-10 h-10 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-3">Welcome to MasterX</h3>
              <p className="text-gray-600 mb-6">Experience the future of AI with quantum intelligence</p>
              <div className="flex flex-wrap justify-center gap-2 text-sm text-gray-500">
                <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full">Quantum Reasoning</span>
                <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full">Creative AI</span>
                <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full">AR Interface</span>
                <span className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full">Analytics</span>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message: any) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-2xl px-6 py-4 rounded-2xl shadow-sm ${
                      message.role === 'user'
                        ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white'
                        : 'bg-white/80 backdrop-blur-sm text-gray-900 border border-gray-200/50'
                    }`}
                  >
                    <div className="prose prose-sm max-w-none">
                      <p className="mb-0">{message.content}</p>
                    </div>
                    <div className={`text-xs mt-2 ${
                      message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                    }`}>
                      {message.timestamp ? new Date(message.timestamp).toLocaleTimeString() : 'Now'}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {/* Typing Indicator */}
          {typingUsers && typingUsers.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="bg-white/80 backdrop-blur-sm text-gray-900 max-w-xs px-4 py-3 rounded-2xl border border-gray-200/50">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm text-gray-600">AI is thinking...</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="bg-white/80 backdrop-blur-sm text-gray-900 max-w-xs px-4 py-3 rounded-2xl border border-gray-200/50">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm text-gray-600">Processing...</span>
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white/80 backdrop-blur-lg border-t border-gray-200/50 p-6">
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <div className="relative">
                <textarea
                  ref={inputRef}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask MasterX anything... Use quantum intelligence, creative AI, or any of the 10 AI tools"
                  className="w-full px-6 py-4 pr-12 border border-gray-300/50 rounded-2xl resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/50 backdrop-blur-sm shadow-sm"
                  rows={1}
                  style={{ minHeight: '56px', maxHeight: '120px' }}
                />
                <div className="absolute right-3 top-3 text-gray-400">
                  <FileText className="w-5 h-5" />
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* FileUpload component temporarily disabled for build
              <FileUpload
                maxFiles={5}
                maxFileSize={25}
                onFileSelect={(files) => {
                  console.log('Files selected:', files)
                  addNotification({
                    type: 'info',
                    title: 'Files Selected',
                    message: `${files.length} file(s) ready to upload`,
                    duration: 2000,
                  })
                }}
                onFileUpload={(files) => {
                  console.log('Files uploaded:', files)
                  addNotification({
                    type: 'success',
                    title: 'Upload Complete',
                    message: 'Files are ready to be processed by AI',
                    duration: 3000,
                  })
                }}
              />
              */}

              <button
                onClick={() => setShowAdvancedInput(true)}
                className="p-3 text-gray-600 hover:text-gray-900 hover:bg-white/50 rounded-xl transition-colors"
                title="Advanced Input (Voice, Gesture, Emotion)"
              >
                <Eye className="w-5 h-5" />
              </button>

              <button
                onClick={sendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="p-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
                title="Send Message"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-2 mt-4 text-sm text-gray-500">
            <span>Quick actions:</span>
            <button
              onClick={() => setShowQuantumIntelligence(true)}
              className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full hover:bg-indigo-200 transition-colors"
            >
              Quantum Analysis
            </button>
            <button
              onClick={() => setShowCreativeAI(true)}
              className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full hover:bg-purple-200 transition-colors"
            >
              Creative Ideas
            </button>
            <button
              onClick={() => setShowDecisionSupport(true)}
              className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors"
            >
              Decision Help
            </button>
          </div>
        </div>
      </div>

      {/* AI Tool Modals */}

      {/* Quantum Intelligence Modal */}
      <AnimatePresence>
        {showQuantumIntelligence && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setShowQuantumIntelligence(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <QuantumIntelligence
                query="How can we enhance the MasterX AI platform with breakthrough features?"
                onSolutionSelect={(solution) => {
                  console.log('Selected solution:', solution)
                  addNotification({
                    type: 'info',
                    title: 'Solution Selected',
                    message: `Selected: ${solution.title}`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowQuantumIntelligence(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Creative AI Modal */}
      <AnimatePresence>
        {showCreativeAI && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setShowCreativeAI(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <CreativeAI
                onIdeaGenerated={(idea) => {
                  console.log('Idea generated:', idea)
                  addNotification({
                    type: 'success',
                    title: 'Creative Idea Generated',
                    message: `New idea: ${idea.title}`,
                    duration: 3000,
                  })
                }}
                onWorkflowCreated={(workflow) => {
                  console.log('Workflow created:', workflow)
                  addNotification({
                    type: 'success',
                    title: 'Workflow Created',
                    message: `New workflow: ${workflow.name}`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowCreativeAI(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Decision Support Modal */}
      <AnimatePresence>
        {showDecisionSupport && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setShowDecisionSupport(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <DecisionSupport
                decision="Should we prioritize advanced AI features or focus on performance optimization?"
                onDecisionMade={(decision) => {
                  console.log('Decision made:', decision)
                  addNotification({
                    type: 'success',
                    title: 'Decision Made',
                    message: `Selected: ${decision.selectedOption.title}`,
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowDecisionSupport(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* AR Interface Modal */}
      <AnimatePresence>
        {showARInterface && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setShowARInterface(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <ARInterface
                onARStart={() => {
                  console.log('AR session started')
                  addNotification({
                    type: 'success',
                    title: 'AR Session Started',
                    message: 'Augmented reality is now active',
                    duration: 3000,
                  })
                }}
                onAREnd={() => {
                  console.log('AR session ended')
                  addNotification({
                    type: 'info',
                    title: 'AR Session Ended',
                    message: 'Augmented reality session has ended',
                    duration: 3000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowARInterface(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Advanced Input Modal */}
      <AnimatePresence>
        {showAdvancedInput && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setShowAdvancedInput(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <AdvancedInput
                onEyeTracking={(data) => {
                  console.log('Eye tracking data:', data)
                }}
                onGestureDetected={(gesture) => {
                  console.log('Gesture detected:', gesture)
                  addNotification({
                    type: 'info',
                    title: 'Gesture Detected',
                    message: `${gesture.type} gesture detected`,
                    duration: 2000,
                  })
                }}
                onEmotionDetected={(emotion) => {
                  console.log('Emotion detected:', emotion)
                  addNotification({
                    type: 'info',
                    title: 'Emotion Detected',
                    message: `${emotion.primary} emotion detected`,
                    duration: 2000,
                  })
                }}
              />

              <div className="bg-white border-t border-gray-200 px-6 py-4 flex justify-end">
                <button
                  onClick={() => setShowAdvancedInput(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Notifications */}
      <AnimatePresence>
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            initial={{ opacity: 0, y: -50, x: 300 }}
            animate={{ opacity: 1, y: 0, x: 0 }}
            exit={{ opacity: 0, y: -50, x: 300 }}
            className="fixed top-4 right-4 z-50"
          >
            <div className={`p-4 rounded-lg shadow-lg max-w-sm ${
              notification.type === 'success' ? 'bg-green-500 text-white' :
              notification.type === 'error' ? 'bg-red-500 text-white' :
              notification.type === 'warning' ? 'bg-yellow-500 text-white' :
              'bg-blue-500 text-white'
            }`}>
              <div className="font-semibold">{notification.title}</div>
              <div className="text-sm opacity-90">{notification.message}</div>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
