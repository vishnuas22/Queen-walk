'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Eye, 
  Hand, 
  Mic, 
  Brain, 
  Target,
  Zap,
  Activity,
  Heart,
  Smile,
  Frown,
  Meh,
  AlertCircle,
  CheckCircle,
  Settings,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react'
import { useUIActions } from '../store'

// ===== TYPES =====

interface AdvancedInputProps {
  onEyeTracking?: (data: EyeTrackingData) => void
  onGestureDetected?: (gesture: GestureData) => void
  onEmotionDetected?: (emotion: EmotionData) => void
  className?: string
}

interface EyeTrackingData {
  gazePoint: { x: number; y: number }
  pupilDilation: number
  blinkRate: number
  fixationDuration: number
  saccadeVelocity: number
  confidence: number
  timestamp: number
}

interface GestureData {
  type: 'swipe' | 'tap' | 'pinch' | 'rotate' | 'point' | 'grab' | 'wave'
  direction?: 'up' | 'down' | 'left' | 'right'
  intensity: number
  duration: number
  confidence: number
  handedness: 'left' | 'right' | 'both'
  timestamp: number
}

interface EmotionData {
  primary: 'happy' | 'sad' | 'angry' | 'surprised' | 'fearful' | 'disgusted' | 'neutral'
  confidence: number
  valence: number // -1 (negative) to 1 (positive)
  arousal: number // 0 (calm) to 1 (excited)
  intensity: number
  secondary?: string[]
  timestamp: number
}

interface InputCapabilities {
  eyeTracking: boolean
  gestureRecognition: boolean
  emotionRecognition: boolean
  voiceEmotionAnalysis: boolean
  faceDetection: boolean
  handTracking: boolean
}

// ===== ADVANCED INPUT COMPONENT =====

export const AdvancedInput: React.FC<AdvancedInputProps> = ({
  onEyeTracking,
  onGestureDetected,
  onEmotionDetected,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'eye' | 'gesture' | 'emotion' | 'settings'>('eye')
  const [capabilities, setCapabilities] = useState<InputCapabilities>({
    eyeTracking: false,
    gestureRecognition: false,
    emotionRecognition: false,
    voiceEmotionAnalysis: false,
    faceDetection: false,
    handTracking: false,
  })
  
  // Tracking states
  const [isEyeTrackingActive, setIsEyeTrackingActive] = useState(false)
  const [isGestureRecognitionActive, setIsGestureRecognitionActive] = useState(false)
  const [isEmotionRecognitionActive, setIsEmotionRecognitionActive] = useState(false)
  
  // Data states
  const [currentEyeData, setCurrentEyeData] = useState<EyeTrackingData | null>(null)
  const [recentGestures, setRecentGestures] = useState<GestureData[]>([])
  const [currentEmotion, setCurrentEmotion] = useState<EmotionData | null>(null)
  const [emotionHistory, setEmotionHistory] = useState<EmotionData[]>([])

  const { addNotification } = useUIActions()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Initialize capabilities and permissions
  useEffect(() => {
    checkCapabilities()
  }, [])

  const checkCapabilities = async () => {
    const caps: InputCapabilities = {
      eyeTracking: 'EyeDropper' in window, // Proxy for advanced browser features
      gestureRecognition: 'DeviceMotionEvent' in window,
      emotionRecognition: 'mediaDevices' in navigator,
      voiceEmotionAnalysis: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
      faceDetection: 'FaceDetector' in window,
      handTracking: 'MediaPipe' in window || true, // Assume available for demo
    }

    setCapabilities(caps)
  }

  // ===== EYE TRACKING =====

  const startEyeTracking = async () => {
    if (!capabilities.eyeTracking) {
      addNotification({
        type: 'warning',
        title: 'Eye Tracking Unavailable',
        message: 'Eye tracking is not supported on this device',
        duration: 3000,
      })
      return
    }

    try {
      // Request camera access for eye tracking simulation
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }

      setIsEyeTrackingActive(true)
      simulateEyeTracking()

      addNotification({
        type: 'success',
        title: 'Eye Tracking Started',
        message: 'Eye tracking is now active',
        duration: 3000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Eye Tracking Failed',
        message: 'Failed to start eye tracking',
        duration: 3000,
      })
    }
  }

  const stopEyeTracking = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    
    setIsEyeTrackingActive(false)
    setCurrentEyeData(null)
  }

  const simulateEyeTracking = () => {
    if (!isEyeTrackingActive) return

    const eyeData: EyeTrackingData = {
      gazePoint: {
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
      },
      pupilDilation: 0.3 + Math.random() * 0.4,
      blinkRate: 15 + Math.random() * 10,
      fixationDuration: 200 + Math.random() * 300,
      saccadeVelocity: 300 + Math.random() * 200,
      confidence: 0.8 + Math.random() * 0.2,
      timestamp: Date.now(),
    }

    setCurrentEyeData(eyeData)
    onEyeTracking?.(eyeData)

    setTimeout(() => {
      if (isEyeTrackingActive) {
        simulateEyeTracking()
      }
    }, 100)
  }

  // ===== GESTURE RECOGNITION =====

  const startGestureRecognition = () => {
    if (!capabilities.gestureRecognition) {
      addNotification({
        type: 'warning',
        title: 'Gesture Recognition Unavailable',
        message: 'Gesture recognition is not supported on this device',
        duration: 3000,
      })
      return
    }

    setIsGestureRecognitionActive(true)
    simulateGestureRecognition()

    addNotification({
      type: 'success',
      title: 'Gesture Recognition Started',
      message: 'Gesture recognition is now active',
      duration: 3000,
    })
  }

  const stopGestureRecognition = () => {
    setIsGestureRecognitionActive(false)
  }

  const simulateGestureRecognition = () => {
    if (!isGestureRecognitionActive) return

    // Simulate random gestures
    const gestures: GestureData['type'][] = ['swipe', 'tap', 'pinch', 'rotate', 'point', 'grab', 'wave']
    const directions: GestureData['direction'][] = ['up', 'down', 'left', 'right']
    
    setTimeout(() => {
      if (isGestureRecognitionActive && Math.random() > 0.7) {
        const gesture: GestureData = {
          type: gestures[Math.floor(Math.random() * gestures.length)],
          direction: directions[Math.floor(Math.random() * directions.length)],
          intensity: Math.random(),
          duration: 100 + Math.random() * 500,
          confidence: 0.7 + Math.random() * 0.3,
          handedness: Math.random() > 0.5 ? 'right' : 'left',
          timestamp: Date.now(),
        }

        setRecentGestures(prev => [gesture, ...prev.slice(0, 9)])
        onGestureDetected?.(gesture)
      }
      
      if (isGestureRecognitionActive) {
        simulateGestureRecognition()
      }
    }, 1000 + Math.random() * 2000)
  }

  // ===== EMOTION RECOGNITION =====

  const startEmotionRecognition = async () => {
    if (!capabilities.emotionRecognition) {
      addNotification({
        type: 'warning',
        title: 'Emotion Recognition Unavailable',
        message: 'Emotion recognition is not supported on this device',
        duration: 3000,
      })
      return
    }

    try {
      // Request camera access for emotion recognition
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }

      setIsEmotionRecognitionActive(true)
      simulateEmotionRecognition()

      addNotification({
        type: 'success',
        title: 'Emotion Recognition Started',
        message: 'Emotion recognition is now active',
        duration: 3000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Emotion Recognition Failed',
        message: 'Failed to start emotion recognition',
        duration: 3000,
      })
    }
  }

  const stopEmotionRecognition = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    
    setIsEmotionRecognitionActive(false)
    setCurrentEmotion(null)
  }

  const simulateEmotionRecognition = () => {
    if (!isEmotionRecognitionActive) return

    const emotions: EmotionData['primary'][] = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
    
    const emotion: EmotionData = {
      primary: emotions[Math.floor(Math.random() * emotions.length)],
      confidence: 0.6 + Math.random() * 0.4,
      valence: (Math.random() - 0.5) * 2,
      arousal: Math.random(),
      intensity: Math.random(),
      secondary: ['calm', 'focused'],
      timestamp: Date.now(),
    }

    setCurrentEmotion(emotion)
    setEmotionHistory(prev => [emotion, ...prev.slice(0, 19)])
    onEmotionDetected?.(emotion)

    setTimeout(() => {
      if (isEmotionRecognitionActive) {
        simulateEmotionRecognition()
      }
    }, 2000 + Math.random() * 3000)
  }

  // ===== RENDER HELPERS =====

  const getEmotionIcon = (emotion: string) => {
    switch (emotion) {
      case 'happy': return <Smile className="h-5 w-5 text-green-500" />
      case 'sad': return <Frown className="h-5 w-5 text-blue-500" />
      case 'angry': return <AlertCircle className="h-5 w-5 text-red-500" />
      case 'neutral': return <Meh className="h-5 w-5 text-gray-500" />
      default: return <Brain className="h-5 w-5 text-purple-500" />
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Eye className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">Advanced Input Methods</h2>
          </div>
          
          <div className="flex items-center space-x-2 text-white text-sm">
            {isEyeTrackingActive && (
              <div className="flex items-center space-x-1">
                <Eye className="h-4 w-4" />
                <span>Eye</span>
              </div>
            )}
            {isGestureRecognitionActive && (
              <div className="flex items-center space-x-1">
                <Hand className="h-4 w-4" />
                <span>Gesture</span>
              </div>
            )}
            {isEmotionRecognitionActive && (
              <div className="flex items-center space-x-1">
                <Heart className="h-4 w-4" />
                <span>Emotion</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'eye', label: 'Eye Tracking', icon: <Eye className="h-4 w-4" /> },
          { id: 'gesture', label: 'Gestures', icon: <Hand className="h-4 w-4" /> },
          { id: 'emotion', label: 'Emotions', icon: <Heart className="h-4 w-4" /> },
          { id: 'settings', label: 'Settings', icon: <Settings className="h-4 w-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-emerald-600 border-b-2 border-emerald-600'
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
            {/* Eye Tracking Tab */}
            {activeTab === 'eye' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Eye Tracking</h3>
                  <button
                    onClick={isEyeTrackingActive ? stopEyeTracking : startEyeTracking}
                    disabled={!capabilities.eyeTracking}
                    className={`px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 ${
                      isEyeTrackingActive
                        ? 'bg-red-600 text-white hover:bg-red-700'
                        : 'bg-emerald-600 text-white hover:bg-emerald-700'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {isEyeTrackingActive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    <span>{isEyeTrackingActive ? 'Stop' : 'Start'} Tracking</span>
                  </button>
                </div>

                {currentEyeData ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Gaze Data</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Gaze Point:</span>
                            <span className="font-mono">
                              ({Math.round(currentEyeData.gazePoint.x)}, {Math.round(currentEyeData.gazePoint.y)})
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Pupil Dilation:</span>
                            <span className="font-mono">{(currentEyeData.pupilDilation * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Blink Rate:</span>
                            <span className="font-mono">{currentEyeData.blinkRate.toFixed(1)}/min</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Confidence:</span>
                            <span className={`font-mono ${getConfidenceColor(currentEyeData.confidence)}`}>
                              {(currentEyeData.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Attention Metrics</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Fixation Duration:</span>
                            <span className="font-mono">{currentEyeData.fixationDuration}ms</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Saccade Velocity:</span>
                            <span className="font-mono">{currentEyeData.saccadeVelocity}°/s</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">Gaze Visualization</h4>
                      <div className="relative bg-white border rounded h-48 overflow-hidden">
                        <motion.div
                          className="absolute w-3 h-3 bg-red-500 rounded-full"
                          animate={{
                            left: `${(currentEyeData.gazePoint.x / window.innerWidth) * 100}%`,
                            top: `${(currentEyeData.gazePoint.y / window.innerHeight) * 100}%`,
                          }}
                          transition={{ duration: 0.1 }}
                          style={{ transform: 'translate(-50%, -50%)' }}
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Eye className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">
                      {capabilities.eyeTracking 
                        ? 'Start eye tracking to see gaze data and attention metrics'
                        : 'Eye tracking is not available on this device'
                      }
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Gesture Recognition Tab */}
            {activeTab === 'gesture' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Gesture Recognition</h3>
                  <button
                    onClick={isGestureRecognitionActive ? stopGestureRecognition : startGestureRecognition}
                    disabled={!capabilities.gestureRecognition}
                    className={`px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 ${
                      isGestureRecognitionActive
                        ? 'bg-red-600 text-white hover:bg-red-700'
                        : 'bg-emerald-600 text-white hover:bg-emerald-700'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {isGestureRecognitionActive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    <span>{isGestureRecognitionActive ? 'Stop' : 'Start'} Recognition</span>
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Recent Gestures</h4>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {recentGestures.map((gesture, index) => (
                        <motion.div
                          key={`${gesture.timestamp}-${index}`}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          className="bg-gray-50 rounded-lg p-3"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium text-gray-900 capitalize">
                              {gesture.type} {gesture.direction && `(${gesture.direction})`}
                            </span>
                            <span className={`text-sm ${getConfidenceColor(gesture.confidence)}`}>
                              {(gesture.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-xs text-gray-600">
                            {gesture.handedness} hand • {gesture.duration}ms • {gesture.intensity.toFixed(2)} intensity
                          </div>
                        </motion.div>
                      ))}
                      
                      {recentGestures.length === 0 && (
                        <div className="text-center py-8 text-gray-500">
                          <Hand className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                          <p>No gestures detected yet</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Supported Gestures</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {['swipe', 'tap', 'pinch', 'rotate', 'point', 'grab', 'wave'].map((gesture) => (
                        <div key={gesture} className="bg-gray-50 rounded-lg p-3 text-center">
                          <Hand className="h-6 w-6 mx-auto mb-1 text-gray-600" />
                          <p className="text-sm font-medium text-gray-900 capitalize">{gesture}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Emotion Recognition Tab */}
            {activeTab === 'emotion' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Emotion Recognition</h3>
                  <button
                    onClick={isEmotionRecognitionActive ? stopEmotionRecognition : startEmotionRecognition}
                    disabled={!capabilities.emotionRecognition}
                    className={`px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 ${
                      isEmotionRecognitionActive
                        ? 'bg-red-600 text-white hover:bg-red-700'
                        : 'bg-emerald-600 text-white hover:bg-emerald-700'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {isEmotionRecognitionActive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    <span>{isEmotionRecognitionActive ? 'Stop' : 'Start'} Recognition</span>
                  </button>
                </div>

                {currentEmotion ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Current Emotion</h4>
                        <div className="flex items-center space-x-3 mb-3">
                          {getEmotionIcon(currentEmotion.primary)}
                          <div>
                            <p className="font-medium text-gray-900 capitalize">{currentEmotion.primary}</p>
                            <p className={`text-sm ${getConfidenceColor(currentEmotion.confidence)}`}>
                              {(currentEmotion.confidence * 100).toFixed(1)}% confidence
                            </p>
                          </div>
                        </div>
                        
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Valence:</span>
                            <span className={`font-mono ${currentEmotion.valence > 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {currentEmotion.valence.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Arousal:</span>
                            <span className="font-mono">{currentEmotion.arousal.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Intensity:</span>
                            <span className="font-mono">{(currentEmotion.intensity * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Emotion Dimensions</h4>
                        <div className="space-y-3">
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Valence (Positive/Negative)</span>
                              <span>{currentEmotion.valence.toFixed(2)}</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${currentEmotion.valence > 0 ? 'bg-green-500' : 'bg-red-500'}`}
                                style={{ width: `${Math.abs(currentEmotion.valence) * 50 + 50}%` }}
                              />
                            </div>
                          </div>
                          
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Arousal (Calm/Excited)</span>
                              <span>{currentEmotion.arousal.toFixed(2)}</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-blue-500 h-2 rounded-full"
                                style={{ width: `${currentEmotion.arousal * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Emotion History</h4>
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {emotionHistory.map((emotion, index) => (
                          <motion.div
                            key={`${emotion.timestamp}-${index}`}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="flex items-center justify-between bg-gray-50 rounded-lg p-3"
                          >
                            <div className="flex items-center space-x-2">
                              {getEmotionIcon(emotion.primary)}
                              <span className="font-medium text-gray-900 capitalize">{emotion.primary}</span>
                            </div>
                            <div className="text-right">
                              <p className={`text-sm ${getConfidenceColor(emotion.confidence)}`}>
                                {(emotion.confidence * 100).toFixed(0)}%
                              </p>
                              <p className="text-xs text-gray-500">
                                {new Date(emotion.timestamp).toLocaleTimeString()}
                              </p>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Heart className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">
                      {capabilities.emotionRecognition 
                        ? 'Start emotion recognition to analyze facial expressions and voice tone'
                        : 'Emotion recognition is not available on this device'
                      }
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Input Method Settings</h3>
                
                <div className="grid gap-4">
                  {Object.entries(capabilities).map(([feature, supported]) => (
                    <div key={feature} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                      <div>
                        <h4 className="font-medium text-gray-900 capitalize">
                          {feature.replace(/([A-Z])/g, ' $1').trim()}
                        </h4>
                        <p className="text-sm text-gray-600">
                          {supported ? 'Available on this device' : 'Not supported on this device'}
                        </p>
                      </div>
                      <div className={`flex items-center space-x-2 ${supported ? 'text-green-600' : 'text-gray-400'}`}>
                        {supported ? <CheckCircle className="h-5 w-5" /> : <AlertCircle className="h-5 w-5" />}
                        <span className="text-sm font-medium">
                          {supported ? 'Supported' : 'Unavailable'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Camera Feed (Hidden) */}
        <video
          ref={videoRef}
          className="hidden"
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="hidden"
        />
      </div>
    </div>
  )
}

export default AdvancedInput
