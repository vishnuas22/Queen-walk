'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  Brain, 
  ArrowLeft,
  BookOpen,
  Code,
  MessageSquare,
  Settings,
  Shield,
  Zap,
  ExternalLink
} from 'lucide-react'

export default function DocsPage() {
  const docSections = [
    {
      icon: BookOpen,
      title: 'Getting Started',
      description: 'Learn the basics of MasterX and start your AI journey',
      links: [
        'Quick Start Guide',
        'First Conversation',
        'Understanding AI Responses',
        'Basic Commands'
      ]
    },
    {
      icon: MessageSquare,
      title: 'AI Tools & Features',
      description: 'Explore all 10 breakthrough AI capabilities',
      links: [
        'Quantum Intelligence',
        'Creative AI Studio',
        'Decision Support System',
        'Augmented Reality Interface',
        'Advanced Input Methods',
        'Gesture Recognition',
        'Emotion Recognition',
        'API Integration',
        'Enterprise Cloud',
        'Analytics Dashboard'
      ]
    },
    {
      icon: Code,
      title: 'API Reference',
      description: 'Integrate MasterX into your applications',
      links: [
        'Authentication',
        'REST API Endpoints',
        'WebSocket Connections',
        'SDKs & Libraries',
        'Rate Limits',
        'Error Handling'
      ]
    },
    {
      icon: Settings,
      title: 'Configuration',
      description: 'Customize MasterX to fit your workflow',
      links: [
        'User Preferences',
        'AI Model Settings',
        'Notification Settings',
        'Privacy Controls',
        'Export & Import'
      ]
    },
    {
      icon: Shield,
      title: 'Security & Privacy',
      description: 'Learn about our security measures and privacy policies',
      links: [
        'Data Protection',
        'Encryption Standards',
        'Privacy Policy',
        'GDPR Compliance',
        'Security Best Practices'
      ]
    },
    {
      icon: Zap,
      title: 'Advanced Usage',
      description: 'Master advanced features and optimization techniques',
      links: [
        'Performance Optimization',
        'Custom Workflows',
        'Batch Processing',
        'Advanced Analytics',
        'Enterprise Features'
      ]
    }
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

      {/* Hero Section */}
      <main className="relative">
        <div className="max-w-7xl mx-auto px-6 py-20">
          <motion.div
            className="text-center mb-20"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center px-4 py-2 bg-indigo-50 rounded-full text-indigo-700 text-sm font-medium mb-8">
              <BookOpen className="h-4 w-4 mr-2" />
              Comprehensive Documentation
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6 tracking-tight">
              Learn &
              <span className="text-gradient block">Master MasterX</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-600 mb-12 max-w-4xl mx-auto leading-relaxed">
              Everything you need to know about using MasterX effectively. From basic concepts to advanced techniques.
            </p>
          </motion.div>

          {/* Documentation Grid */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            {docSections.map((section, index) => (
              <motion.div
                key={index}
                className="enterprise-card p-8 group hover:scale-105 transition-all duration-300"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
              >
                <div className="w-16 h-16 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:from-indigo-200 group-hover:to-purple-200 transition-all duration-300">
                  <section.icon className="h-8 w-8 text-indigo-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-4 text-center">
                  {section.title}
                </h3>
                <p className="text-gray-600 leading-relaxed mb-6 text-center">
                  {section.description}
                </p>
                <ul className="space-y-3">
                  {section.links.map((link, linkIndex) => (
                    <li key={linkIndex}>
                      <button className="flex items-center justify-between w-full text-left text-sm text-gray-600 hover:text-indigo-600 transition-colors group/link">
                        <span>{link}</span>
                        <ExternalLink className="h-3 w-3 opacity-0 group-hover/link:opacity-100 transition-opacity" />
                      </button>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </motion.div>

          {/* Quick Start Section */}
          <motion.div
            className="mt-20 text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <div className="enterprise-card p-12 max-w-4xl mx-auto">
              <h2 className="text-3xl font-bold text-gray-900 mb-6">
                Ready to Get Started?
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Jump right in and start exploring MasterX's powerful AI capabilities. Our intelligent assistant is ready to help you learn and create.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  href="/chat"
                  className="enterprise-button enterprise-button-primary text-lg px-8 py-4"
                >
                  Start Your First Conversation
                  <MessageSquare className="ml-2 h-5 w-5" />
                </Link>
                <button className="enterprise-button enterprise-button-secondary text-lg px-8 py-4">
                  View Quick Start Guide
                  <BookOpen className="ml-2 h-5 w-5" />
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
}
