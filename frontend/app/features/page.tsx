'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  Brain, 
  MessageSquare, 
  Code, 
  BarChart3,
  Zap,
  Shield,
  Users,
  ArrowLeft,
  Sparkles
} from 'lucide-react'

export default function FeaturesPage() {
  const features = [
    {
      icon: Brain,
      title: 'Quantum Intelligence',
      description: 'Advanced AI powered by quantum computing principles for superior reasoning and problem-solving capabilities.',
      details: [
        'Quantum-enhanced neural networks',
        'Advanced pattern recognition',
        'Complex problem solving',
        'Multi-dimensional analysis'
      ]
    },
    {
      icon: MessageSquare,
      title: 'Natural Conversations',
      description: 'Engage in fluid, context-aware conversations that adapt to your learning style and preferences.',
      details: [
        'Context-aware responses',
        'Personalized communication',
        'Multi-turn conversations',
        'Emotional intelligence'
      ]
    },
    {
      icon: Code,
      title: 'Code Generation',
      description: 'Generate, debug, and optimize code across multiple programming languages with intelligent suggestions.',
      details: [
        'Multi-language support',
        'Intelligent debugging',
        'Code optimization',
        'Best practices guidance'
      ]
    },
    {
      icon: BarChart3,
      title: 'Learning Analytics',
      description: 'Track your progress with detailed analytics and personalized learning recommendations.',
      details: [
        'Progress tracking',
        'Performance insights',
        'Personalized recommendations',
        'Learning path optimization'
      ]
    },
    {
      icon: Zap,
      title: 'Real-time Processing',
      description: 'Lightning-fast response times with advanced caching and optimization technologies.',
      details: [
        'Sub-second responses',
        'Intelligent caching',
        'Edge computing',
        'Optimized algorithms'
      ]
    },
    {
      icon: Shield,
      title: 'Enterprise Security',
      description: 'Bank-grade security with end-to-end encryption and compliance with industry standards.',
      details: [
        'End-to-end encryption',
        'SOC 2 compliance',
        'GDPR compliant',
        'Zero-trust architecture'
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
              <Sparkles className="h-4 w-4 mr-2" />
              Advanced AI Features
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6 tracking-tight">
              Breakthrough
              <span className="text-gradient block">AI Capabilities</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-600 mb-12 max-w-4xl mx-auto leading-relaxed">
              Discover the cutting-edge features that make MasterX the most advanced AI platform for learning, creating, and problem-solving.
            </p>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="enterprise-card p-8 group hover:scale-105 transition-all duration-300"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
              >
                <div className="w-16 h-16 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:from-indigo-200 group-hover:to-purple-200 transition-all duration-300">
                  <feature.icon className="h-8 w-8 text-indigo-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-4 text-center">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed mb-6 text-center">
                  {feature.description}
                </p>
                <ul className="space-y-2">
                  {feature.details.map((detail, detailIndex) => (
                    <li key={detailIndex} className="flex items-center text-sm text-gray-500">
                      <div className="w-1.5 h-1.5 bg-indigo-500 rounded-full mr-3"></div>
                      {detail}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </motion.div>

          {/* CTA Section */}
          <motion.div
            className="text-center mt-20"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-6">
              Ready to Experience Quantum Intelligence?
            </h2>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Start your journey with MasterX and unlock the full potential of AI-powered learning and creativity.
            </p>
            <Link
              href="/chat"
              className="enterprise-button enterprise-button-primary text-lg px-8 py-4"
            >
              Start Conversation
              <MessageSquare className="ml-2 h-5 w-5" />
            </Link>
          </motion.div>
        </div>
      </main>
    </div>
  )
}
