'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useVoiceRecognition } from '../lib/speechAPI'

// ===== VOICE NAVIGATION COMPONENT =====

interface VoiceNavigationProps {
  className?: string
  showCommands?: boolean
}

export const VoiceNavigation: React.FC<VoiceNavigationProps> = ({
  className = '',
  showCommands = false,
}) => {
  const {
    isListening,
    lastTranscript,
    lastCommand,
    error,
    startListening,
    stopListening,
    toggleListening,
    speak,
    isSupported,
    commands,
  } = useVoiceRecognition()

  const [showTranscript, setShowTranscript] = useState(false)
  const [showCommandList, setShowCommandList] = useState(showCommands)
  const [audioLevel, setAudioLevel] = useState(0)
  const [showSettings, setShowSettings] = useState(false)

  // Auto-hide transcript after 3 seconds
  useEffect(() => {
    if (lastTranscript) {
      setShowTranscript(true)
      const timer = setTimeout(() => setShowTranscript(false), 3000)
      return () => clearTimeout(timer)
    }
  }, [lastTranscript])

  // Announce command execution
  useEffect(() => {
    if (lastCommand) {
      speak(`Executing ${lastCommand.command}`)
    }
  }, [lastCommand, speak])

  if (!isSupported) {
    return (
      <div className={`bg-yellow-50 border border-yellow-200 rounded-lg p-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <svg className="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <span className="text-yellow-800 text-sm">
            Voice navigation is not supported in this browser
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Voice Control Button */}
      <div className="flex items-center space-x-4">
        <motion.button
          onClick={toggleListening}
          className={`
            relative flex items-center justify-center w-12 h-12 rounded-full
            transition-all duration-300 focus:outline-none focus:ring-4
            ${isListening 
              ? 'bg-red-500 hover:bg-red-600 focus:ring-red-200 shadow-lg' 
              : 'bg-indigo-500 hover:bg-indigo-600 focus:ring-indigo-200'
            }
          `}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          aria-label={isListening ? 'Stop voice recognition' : 'Start voice recognition'}
        >
          {/* Microphone Icon */}
          <svg 
            className="w-6 h-6 text-white" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" 
            />
          </svg>

          {/* Listening Animation with Audio Level */}
          {isListening && (
            <>
              <motion.div
                className="absolute inset-0 rounded-full border-4 border-red-300"
                animate={{
                  scale: [1, 1.1 + audioLevel * 0.3, 1],
                  opacity: [0.7, 0.3, 0.7],
                }}
                transition={{
                  duration: 0.5,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
              />

              {/* Audio Level Indicator */}
              <motion.div
                className="absolute inset-1 rounded-full bg-red-400"
                animate={{
                  scale: 0.8 + audioLevel * 0.4,
                  opacity: 0.3 + audioLevel * 0.4,
                }}
                transition={{
                  duration: 0.1,
                  ease: "easeOut",
                }}
              />
            </>
          )}
        </motion.button>

        <div className="flex-1">
          <div className="text-sm font-medium text-slate-700">
            Voice Navigation
          </div>
          <div className="text-xs text-slate-500">
            {isListening ? 'Listening for commands...' : 'Click to start voice control'}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowCommandList(!showCommandList)}
            className="p-2 text-slate-400 hover:text-slate-600 transition-colors"
            aria-label="Toggle command list"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>

          <VoiceSettings />
        </div>
      </div>

      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-red-50 border border-red-200 rounded-lg p-3"
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="text-red-700 text-sm">Voice recognition error: {error}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Live Transcript */}
      <AnimatePresence>
        {showTranscript && lastTranscript && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="bg-blue-50 border border-blue-200 rounded-lg p-3"
          >
            <div className="flex items-start space-x-2">
              <svg className="w-4 h-4 text-blue-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
              </svg>
              <div>
                <div className="text-blue-700 text-sm font-medium">You said:</div>
                <div className="text-blue-600 text-sm">"{lastTranscript}"</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Last Command Executed */}
      <AnimatePresence>
        {lastCommand && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="bg-green-50 border border-green-200 rounded-lg p-3"
          >
            <div className="flex items-start space-x-2">
              <svg className="w-4 h-4 text-green-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <div>
                <div className="text-green-700 text-sm font-medium">Command executed:</div>
                <div className="text-green-600 text-sm">"{lastCommand.command}"</div>
                <div className="text-green-500 text-xs">{lastCommand.description}</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Available Commands */}
      <AnimatePresence>
        {showCommandList && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-50 border border-slate-200 rounded-lg overflow-hidden"
          >
            <div className="p-4">
              <h3 className="text-sm font-semibold text-slate-700 mb-3">Available Voice Commands</h3>
              
              {/* Group commands by category */}
              {['navigation', 'chat', 'accessibility', 'system'].map(category => {
                const categoryCommands = commands.filter(cmd => cmd.category === category)
                if (categoryCommands.length === 0) return null

                return (
                  <div key={category} className="mb-4 last:mb-0">
                    <h4 className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">
                      {category}
                    </h4>
                    <div className="space-y-2">
                      {categoryCommands.map((command, index) => (
                        <div key={index} className="flex items-start space-x-3">
                          <div className="bg-indigo-100 text-indigo-700 text-xs px-2 py-1 rounded font-mono">
                            "{command.command}"
                          </div>
                          <div className="text-xs text-slate-600 flex-1">
                            {command.description}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ===== VOICE STATUS INDICATOR =====

export const VoiceStatusIndicator: React.FC = () => {
  const { isListening, isSupported } = useVoiceRecognition()

  if (!isSupported) return null

  return (
    <AnimatePresence>
      {isListening && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className="fixed top-4 right-4 z-50 bg-red-500 text-white px-3 py-2 rounded-full shadow-lg"
        >
          <div className="flex items-center space-x-2">
            <motion.div
              className="w-2 h-2 bg-white rounded-full"
              animate={{
                scale: [1, 1.5, 1],
                opacity: [1, 0.5, 1],
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
            <span className="text-sm font-medium">Listening</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ===== VOICE SETTINGS PANEL =====

export const VoiceSettings: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false)
  const [settings, setSettings] = useState({
    language: 'en-US',
    voiceSpeed: 1,
    voicePitch: 1,
    voiceVolume: 1,
    selectedVoice: null as SpeechSynthesisVoice | null,
    noiseReduction: true,
    echoCancellation: true,
    autoGainControl: true,
    visualFeedback: true,
  })
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([])

  useEffect(() => {
    const loadVoices = () => {
      const voices = speechSynthesis.getVoices()
      setAvailableVoices(voices)
    }

    loadVoices()
    speechSynthesis.addEventListener('voiceschanged', loadVoices)

    return () => {
      speechSynthesis.removeEventListener('voiceschanged', loadVoices)
    }
  }, [])

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))
    // Apply settings to speech recognition system
    console.log(`Voice setting changed: ${key} = ${value}`)
  }

  const testVoice = () => {
    const utterance = new SpeechSynthesisUtterance('This is a test of the selected voice settings.')
    if (settings.selectedVoice) {
      utterance.voice = settings.selectedVoice
    }
    utterance.rate = settings.voiceSpeed
    utterance.pitch = settings.voicePitch
    utterance.volume = settings.voiceVolume
    speechSynthesis.speak(utterance)
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
        title="Voice Settings"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4 max-h-96 overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Voice Settings
              </h2>

              <div className="space-y-4">
                {/* Language Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Language
                  </label>
                  <select
                    value={settings.language}
                    onChange={(e) => handleSettingChange('language', e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="en-US">English (US)</option>
                    <option value="en-GB">English (UK)</option>
                    <option value="es-ES">Spanish</option>
                    <option value="fr-FR">French</option>
                    <option value="de-DE">German</option>
                    <option value="it-IT">Italian</option>
                    <option value="pt-BR">Portuguese</option>
                    <option value="ja-JP">Japanese</option>
                    <option value="ko-KR">Korean</option>
                    <option value="zh-CN">Chinese (Simplified)</option>
                  </select>
                </div>

                {/* Voice Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Voice
                  </label>
                  <select
                    value={settings.selectedVoice?.name || ''}
                    onChange={(e) => {
                      const voice = availableVoices.find(v => v.name === e.target.value) || null
                      handleSettingChange('selectedVoice', voice)
                    }}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="">Default Voice</option>
                    {availableVoices.map((voice) => (
                      <option key={voice.name} value={voice.name}>
                        {voice.name} ({voice.lang})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Voice Speed */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Speed: {settings.voiceSpeed.toFixed(1)}x
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.1"
                    value={settings.voiceSpeed}
                    onChange={(e) => handleSettingChange('voiceSpeed', parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Voice Pitch */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Pitch: {settings.voicePitch.toFixed(1)}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={settings.voicePitch}
                    onChange={(e) => handleSettingChange('voicePitch', parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Voice Volume */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Volume: {Math.round(settings.voiceVolume * 100)}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={settings.voiceVolume}
                    onChange={(e) => handleSettingChange('voiceVolume', parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Audio Processing Options */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-gray-700">Audio Processing</h3>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.noiseReduction}
                      onChange={(e) => handleSettingChange('noiseReduction', e.target.checked)}
                      className="mr-2"
                    />
                    <span className="text-sm text-gray-600">Noise Reduction</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.echoCancellation}
                      onChange={(e) => handleSettingChange('echoCancellation', e.target.checked)}
                      className="mr-2"
                    />
                    <span className="text-sm text-gray-600">Echo Cancellation</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.autoGainControl}
                      onChange={(e) => handleSettingChange('autoGainControl', e.target.checked)}
                      className="mr-2"
                    />
                    <span className="text-sm text-gray-600">Auto Gain Control</span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.visualFeedback}
                      onChange={(e) => handleSettingChange('visualFeedback', e.target.checked)}
                      className="mr-2"
                    />
                    <span className="text-sm text-gray-600">Visual Feedback</span>
                  </label>
                </div>

                {/* Test Voice */}
                <button
                  onClick={testVoice}
                  className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  Test Voice
                </button>
              </div>

              <div className="mt-6 flex justify-end space-x-2">
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default VoiceNavigation
