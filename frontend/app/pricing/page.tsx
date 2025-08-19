'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  Brain, 
  Check, 
  ArrowLeft,
  Sparkles,
  Zap,
  Crown,
  Building
} from 'lucide-react'

export default function PricingPage() {
  const plans = [
    {
      name: 'Starter',
      icon: Sparkles,
      price: 'Free',
      period: 'Forever',
      description: 'Perfect for individuals getting started with AI',
      features: [
        '10 conversations per day',
        'Basic AI assistance',
        'Standard response time',
        'Community support',
        'Basic analytics'
      ],
      buttonText: 'Get Started',
      buttonStyle: 'enterprise-button enterprise-button-secondary',
      popular: false
    },
    {
      name: 'Professional',
      icon: Zap,
      price: '$29',
      period: 'per month',
      description: 'Ideal for professionals and power users',
      features: [
        'Unlimited conversations',
        'Advanced AI capabilities',
        'Priority response time',
        'Email support',
        'Advanced analytics',
        'Custom AI tools',
        'Export conversations',
        'API access'
      ],
      buttonText: 'Start Free Trial',
      buttonStyle: 'enterprise-button enterprise-button-primary',
      popular: true
    },
    {
      name: 'Enterprise',
      icon: Building,
      price: 'Custom',
      period: 'Contact us',
      description: 'For teams and organizations at scale',
      features: [
        'Everything in Professional',
        'Dedicated AI models',
        'Custom integrations',
        'SSO & advanced security',
        'Dedicated support',
        'Custom training',
        'SLA guarantees',
        'On-premise deployment'
      ],
      buttonText: 'Contact Sales',
      buttonStyle: 'enterprise-button enterprise-button-secondary',
      popular: false
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
              <Crown className="h-4 w-4 mr-2" />
              Simple, Transparent Pricing
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6 tracking-tight">
              Choose Your
              <span className="text-gradient block">AI Journey</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-600 mb-12 max-w-4xl mx-auto leading-relaxed">
              Start free and scale as you grow. No hidden fees, no surprises. Just powerful AI at your fingertips.
            </p>
          </motion.div>

          {/* Pricing Cards */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            {plans.map((plan, index) => (
              <motion.div
                key={index}
                className={`enterprise-card p-8 relative ${plan.popular ? 'ring-2 ring-indigo-500 scale-105' : ''}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ scale: plan.popular ? 1.05 : 1.02, y: -5 }}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                    <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-4 py-2 rounded-full text-sm font-medium">
                      Most Popular
                    </div>
                  </div>
                )}
                
                <div className="text-center mb-8">
                  <div className="w-16 h-16 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <plan.icon className="h-8 w-8 text-indigo-600" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">{plan.name}</h3>
                  <p className="text-gray-600 mb-4">{plan.description}</p>
                  <div className="mb-6">
                    <span className="text-4xl font-bold text-gray-900">{plan.price}</span>
                    {plan.period !== 'Contact us' && (
                      <span className="text-gray-500 ml-2">/{plan.period}</span>
                    )}
                  </div>
                </div>

                <ul className="space-y-4 mb-8">
                  {plan.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center">
                      <Check className="h-5 w-5 text-green-500 mr-3 flex-shrink-0" />
                      <span className="text-gray-600">{feature}</span>
                    </li>
                  ))}
                </ul>

                <button className={`w-full ${plan.buttonStyle} justify-center`}>
                  {plan.buttonText}
                </button>
              </motion.div>
            ))}
          </motion.div>

          {/* FAQ Section */}
          <motion.div
            className="mt-20 text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-6">
              Questions? We're here to help.
            </h2>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Get in touch with our team to learn more about MasterX and find the perfect plan for your needs.
            </p>
            <Link
              href="/chat"
              className="enterprise-button enterprise-button-secondary text-lg px-8 py-4"
            >
              Ask Our AI Assistant
            </Link>
          </motion.div>
        </div>
      </main>
    </div>
  )
}
