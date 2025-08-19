'use client'

import { motion } from 'framer-motion'
import { buttonAnimations, messageAnimations, loadingAnimations } from '../lib/animations'

export default function AnimationTest() {
  return (
    <div className="p-8 space-y-8">
      <h2 className="text-2xl font-bold text-gray-900">Animation System Test</h2>
      
      {/* Button Animation Test */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700">Button Animations</h3>
        <motion.button
          className="px-6 py-3 bg-indigo-600 text-white rounded-xl font-medium"
          variants={buttonAnimations}
          initial="idle"
          whileHover="hover"
          whileTap="tap"
        >
          Animated Button
        </motion.button>
      </div>

      {/* Message Animation Test */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700">Message Animations</h3>
        <motion.div
          className="p-4 bg-white border border-gray-200 rounded-xl shadow-sm"
          variants={messageAnimations}
          initial="hidden"
          animate="visible"
        >
          This is a test message with entrance animation
        </motion.div>
      </div>

      {/* Loading Animation Test */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700">Loading Animations</h3>
        <div className="flex space-x-4">
          <motion.div
            className="w-4 h-4 bg-indigo-600 rounded-full"
            variants={loadingAnimations}
            animate="pulse"
          />
          <motion.div
            className="w-4 h-4 bg-indigo-600 rounded-full"
            variants={loadingAnimations}
            animate="pulse"
            style={{ animationDelay: '0.2s' }}
          />
          <motion.div
            className="w-4 h-4 bg-indigo-600 rounded-full"
            variants={loadingAnimations}
            animate="pulse"
            style={{ animationDelay: '0.4s' }}
          />
        </div>
      </div>
    </div>
  )
}
