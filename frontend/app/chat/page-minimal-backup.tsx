'use client'

import { useState } from 'react'
import Link from 'next/link'
import {
  Brain,
  Send,
  ArrowLeft,
  BarChart3,
  Wand2,
  GitBranch,
  Glasses,
  Eye,
  Hand,
  Heart,
  Link as LinkIcon,
  Building
} from 'lucide-react'

export default function MinimalChatPage() {
  const [message, setMessage] = useState('')

  const aiTools = [
    { icon: Brain, name: 'Quantum Intelligence', color: 'indigo' },
    { icon: Wand2, name: 'Creative AI', color: 'purple' },
    { icon: GitBranch, name: 'Decision Support', color: 'blue' },
    { icon: Glasses, name: 'AR Interface', color: 'green' },
    { icon: Eye, name: 'Advanced Input', color: 'yellow' },
    { icon: Hand, name: 'Gesture Recognition', color: 'pink' },
    { icon: Heart, name: 'Emotion Recognition', color: 'red' },
    { icon: LinkIcon, name: 'API Integration', color: 'cyan' },
    { icon: Building, name: 'Enterprise Cloud', color: 'gray' },
    { icon: BarChart3, name: 'Analytics', color: 'orange' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50">
      {/* Navigation */}
      <nav className="relative z-10 px-6 py-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl flex items-center justify-center shadow-lg">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">MasterX</h1>
              <p className="text-xs text-gray-500 font-medium">Quantum Intelligence</p>
            </div>
          </Link>
          
          <Link 
            href="/" 
            className="flex items-center text-gray-600 hover:text-gray-900 font-medium transition-colors"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Home
          </Link>
        </div>
      </nav>

      {/* Chat Interface */}
      <main className="relative">
        <div className="max-w-7xl mx-auto px-6">
          {/* Chat Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Quantum Intelligence Chat
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Powered by advanced AI algorithms
            </p>

            {/* AI Tool Buttons */}
            <div className="flex flex-wrap justify-center gap-3 mb-8">
              {aiTools.map((tool, index) => (
                <button
                  key={index}
                  className="enterprise-button enterprise-button-secondary flex items-center space-x-2 text-sm"
                  onClick={() => console.log(`${tool.name} clicked`)}
                >
                  <tool.icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{tool.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Chat Area */}
          <div className="max-w-4xl mx-auto">
            <div className="enterprise-card p-8 mb-6">
              <div className="text-center py-12">
                <Brain className="h-16 w-16 text-indigo-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  Welcome to MasterX
                </h3>
                <p className="text-gray-600">
                  Start a conversation with our quantum intelligence AI. Ask questions, get help with coding, or explore creative ideas.
                </p>
              </div>
            </div>

            {/* Input Area */}
            <div className="enterprise-card p-6">
              <div className="flex space-x-4">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Type your message here..."
                  className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      console.log('Message sent:', message)
                      setMessage('')
                    }
                  }}
                />
                <button
                  onClick={() => {
                    console.log('Message sent:', message)
                    setMessage('')
                  }}
                  className="enterprise-button enterprise-button-primary px-6 py-3"
                >
                  <Send className="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
