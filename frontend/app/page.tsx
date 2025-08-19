'use client'

import { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { buttonAnimations, fadeInAnimation, messageListAnimation } from '../lib/animations'
import { 
  Brain, 
  MessageSquare, 
  Zap, 
  Shield, 
  Users, 
  ArrowRight,
  Sparkles,
  Code,
  BarChart3,
  Lightbulb
} from 'lucide-react'

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false)

  const features = [
    {
      icon: Brain,
      title: 'Quantum Intelligence',
      description: 'Advanced AI powered by quantum computing principles for superior reasoning and problem-solving capabilities.'
    },
    {
      icon: MessageSquare,
      title: 'Natural Conversations',
      description: 'Engage in fluid, context-aware conversations that adapt to your learning style and preferences.'
    },
    {
      icon: Code,
      title: 'Code Generation',
      description: 'Generate, debug, and optimize code across multiple programming languages with intelligent suggestions.'
    },
    {
      icon: BarChart3,
      title: 'Learning Analytics',
      description: 'Track your progress with detailed analytics and personalized learning recommendations.'
    }
  ]

  const handleGetStarted = () => {
    setIsLoading(true)
    // Navigate to chat after brief loading
    setTimeout(() => {
      window.location.href = '/chat'
    }, 500)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50">
      {/* Navigation */}
      <nav className="relative z-10 px-6 py-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl flex items-center justify-center shadow-lg">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">MasterX</h1>
              <p className="text-xs text-gray-500 font-medium">Quantum Intelligence</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <Link href="/features" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">
              Features
            </Link>
            <Link href="/pricing" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">
              Pricing
            </Link>
            <Link href="/docs" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">
              Documentation
            </Link>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link 
              href="/chat" 
              className="enterprise-button enterprise-button-primary"
            >
              Get Started
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative">
        <div className="max-w-7xl mx-auto px-6 py-20">
          <motion.div
            className="text-center"
            variants={fadeInAnimation}
            initial="hidden"
            animate="visible"
          >
            <div className="inline-flex items-center px-4 py-2 bg-indigo-50 rounded-full text-indigo-700 text-sm font-medium mb-8">
              <Sparkles className="h-4 w-4 mr-2" />
              Powered by Quantum Intelligence
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6 tracking-tight">
              The Future of
              <span className="text-gradient block">AI Conversation</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-600 mb-12 max-w-4xl mx-auto leading-relaxed">
              Experience advanced AI conversations powered by quantum intelligence. 
              Learn, create, and solve complex problems with our next-generation platform.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
              <motion.button
                onClick={handleGetStarted}
                disabled={isLoading}
                className="enterprise-button enterprise-button-primary text-lg px-8 py-4 min-w-[200px]"
                variants={buttonAnimations}
                initial="idle"
                whileHover="hover"
                whileTap="tap"
              >
                {isLoading ? (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Loading...
                  </div>
                ) : (
                  <>
                    Start Conversation
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </>
                )}
              </motion.button>

              <motion.div
                variants={buttonAnimations}
                initial="idle"
                whileHover="hover"
                whileTap="tap"
              >
                <Link
                  href="/demo"
                  className="enterprise-button enterprise-button-secondary text-lg px-8 py-4"
                >
                  Watch Demo
                </Link>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* Features Grid */}
        <div className="max-w-7xl mx-auto px-6 py-20">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Quantum-Powered Capabilities
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Harness the power of advanced AI with features designed for the future of learning and productivity.
            </p>
          </div>
          
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8"
            variants={messageListAnimation}
            initial="hidden"
            animate="visible"
          >
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="enterprise-card p-8 text-center group hover:scale-105 transition-all duration-300"
                variants={fadeInAnimation}
                whileHover={{ scale: 1.05, y: -5 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="w-16 h-16 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:from-indigo-200 group-hover:to-purple-200 transition-all duration-300">
                  <feature.icon className="h-8 w-8 text-indigo-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-4">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </main>
    </div>
  )
}
