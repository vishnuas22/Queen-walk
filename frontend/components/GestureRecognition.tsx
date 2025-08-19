'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Hand, 
  Zap, 
  Target, 
  Activity,
  Settings,
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Layers,
  MousePointer,
  Move,
  RotateCw,
  ZoomIn,
  ZoomOut
} from 'lucide-react'
import { useUIActions } from '../store'

// ===== TYPES =====

interface GestureRecognitionProps {
  onGestureCommand?: (command: GestureCommand) => void
  onCalibrationComplete?: () => void
  className?: string
}

interface GestureCommand {
  action: 'navigate' | 'select' | 'scroll' | 'zoom' | 'rotate' | 'menu' | 'back' | 'confirm'
  parameters?: {
    direction?: 'up' | 'down' | 'left' | 'right'
    intensity?: number
    target?: string
    value?: number
  }
  confidence: number
  timestamp: number
}

interface HandLandmark {
  x: number
  y: number
  z: number
  visibility: number
}

interface HandPose {
  landmarks: HandLandmark[]
  handedness: 'left' | 'right'
  confidence: number
}

interface GesturePattern {
  id: string
  name: string
  description: string
  landmarks: number[]
  threshold: number
  action: GestureCommand['action']
  enabled: boolean
}

interface CalibrationData {
  handSize: number
  reachDistance: number
  gestureSpeed: number
  sensitivity: number
}

// ===== GESTURE RECOGNITION COMPONENT =====

export const GestureRecognition: React.FC<GestureRecognitionProps> = ({
  onGestureCommand,
  onCalibrationComplete,
  className = ''
}) => {
  const [isActive, setIsActive] = useState(false)
  const [isCalibrating, setIsCalibrating] = useState(false)
  const [currentHands, setCurrentHands] = useState<HandPose[]>([])
  const [recognizedGestures, setRecognizedGestures] = useState<GestureCommand[]>([])
  const [gesturePatterns, setGesturePatterns] = useState<GesturePattern[]>([])
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null)
  const [sensitivity, setSensitivity] = useState(0.7)
  const [activeTab, setActiveTab] = useState<'recognition' | 'patterns' | 'calibration'>('recognition')

  const { addNotification } = useUIActions()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationFrameRef = useRef<number>(0)

  // Initialize gesture patterns
  useEffect(() => {
    const defaultPatterns: GesturePattern[] = [
      {
        id: 'point-select',
        name: 'Point & Select',
        description: 'Point with index finger to select items',
        landmarks: [8], // Index finger tip
        threshold: 0.8,
        action: 'select',
        enabled: true,
      },
      {
        id: 'swipe-navigate',
        name: 'Swipe Navigation',
        description: 'Swipe left/right to navigate',
        landmarks: [4, 8, 12, 16, 20], // All fingertips
        threshold: 0.7,
        action: 'navigate',
        enabled: true,
      },
      {
        id: 'pinch-zoom',
        name: 'Pinch Zoom',
        description: 'Pinch to zoom in/out',
        landmarks: [4, 8], // Thumb and index finger
        threshold: 0.9,
        action: 'zoom',
        enabled: true,
      },
      {
        id: 'palm-menu',
        name: 'Palm Menu',
        description: 'Show palm to open menu',
        landmarks: [0, 1, 5, 9, 13, 17], // Palm landmarks
        threshold: 0.8,
        action: 'menu',
        enabled: true,
      },
      {
        id: 'fist-confirm',
        name: 'Fist Confirm',
        description: 'Make a fist to confirm action',
        landmarks: [4, 8, 12, 16, 20], // All fingertips
        threshold: 0.85,
        action: 'confirm',
        enabled: true,
      },
      {
        id: 'peace-back',
        name: 'Peace Sign Back',
        description: 'Peace sign to go back',
        landmarks: [8, 12], // Index and middle finger
        threshold: 0.8,
        action: 'back',
        enabled: true,
      },
    ]

    setGesturePatterns(defaultPatterns)
  }, [])

  // ===== GESTURE RECOGNITION SYSTEM =====

  const startGestureRecognition = async () => {
    try {
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }

      setIsActive(true)
      startHandTracking()

      addNotification({
        type: 'success',
        title: 'Gesture Recognition Started',
        message: 'Hand tracking is now active',
        duration: 3000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Camera Access Failed',
        message: 'Unable to access camera for gesture recognition',
        duration: 5000,
      })
    }
  }

  const stopGestureRecognition = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }

    setIsActive(false)
    setCurrentHands([])
  }

  const startHandTracking = () => {
    // Simulate hand tracking with MediaPipe-like functionality
    const trackHands = () => {
      if (!isActive) return

      // Simulate hand detection
      if (Math.random() > 0.3) {
        const simulatedHands = generateSimulatedHandPoses()
        setCurrentHands(simulatedHands)
        
        // Process gestures
        simulatedHands.forEach(hand => {
          processHandGestures(hand)
        })
      }

      animationFrameRef.current = requestAnimationFrame(trackHands)
    }

    trackHands()
  }

  const generateSimulatedHandPoses = (): HandPose[] => {
    const hands: HandPose[] = []
    
    // Simulate 1-2 hands
    const handCount = Math.random() > 0.7 ? 2 : 1
    
    for (let i = 0; i < handCount; i++) {
      const landmarks: HandLandmark[] = []
      
      // Generate 21 hand landmarks (MediaPipe standard)
      for (let j = 0; j < 21; j++) {
        landmarks.push({
          x: Math.random() * 640,
          y: Math.random() * 480,
          z: Math.random() * 0.1,
          visibility: 0.8 + Math.random() * 0.2,
        })
      }

      hands.push({
        landmarks,
        handedness: i === 0 ? 'right' : 'left',
        confidence: 0.8 + Math.random() * 0.2,
      })
    }

    return hands
  }

  const processHandGestures = (hand: HandPose) => {
    gesturePatterns.forEach(pattern => {
      if (!pattern.enabled) return

      const gestureConfidence = calculateGestureConfidence(hand, pattern)
      
      if (gestureConfidence > pattern.threshold * sensitivity) {
        const command = createGestureCommand(pattern, hand, gestureConfidence)
        executeGestureCommand(command)
      }
    })
  }

  const calculateGestureConfidence = (hand: HandPose, pattern: GesturePattern): number => {
    // Simplified gesture recognition logic
    let confidence = 0

    switch (pattern.id) {
      case 'point-select':
        // Check if index finger is extended
        const indexTip = hand.landmarks[8]
        const indexPip = hand.landmarks[6]
        confidence = indexTip.y < indexPip.y ? 0.9 : 0.1
        break

      case 'pinch-zoom':
        // Check distance between thumb and index finger
        const thumbTip = hand.landmarks[4]
        const indexFingerTip = hand.landmarks[8]
        const distance = Math.sqrt(
          Math.pow(thumbTip.x - indexFingerTip.x, 2) +
          Math.pow(thumbTip.y - indexFingerTip.y, 2)
        )
        confidence = distance < 50 ? 0.9 : 0.1
        break

      case 'fist-confirm':
        // Check if all fingers are closed
        const fingertips = [8, 12, 16, 20]
        const pips = [6, 10, 14, 18]
        let closedFingers = 0
        
        fingertips.forEach((tip, index) => {
          if (hand.landmarks[tip].y > hand.landmarks[pips[index]].y) {
            closedFingers++
          }
        })
        
        confidence = closedFingers >= 3 ? 0.9 : 0.1
        break

      default:
        confidence = Math.random() * 0.5 + 0.3
    }

    return Math.min(confidence * hand.confidence, 1.0)
  }

  const createGestureCommand = (
    pattern: GesturePattern, 
    hand: HandPose, 
    confidence: number
  ): GestureCommand => {
    const command: GestureCommand = {
      action: pattern.action,
      confidence,
      timestamp: Date.now(),
    }

    // Add specific parameters based on gesture type
    switch (pattern.action) {
      case 'navigate':
        // Determine swipe direction
        const wrist = hand.landmarks[0]
        const middleFinger = hand.landmarks[12]
        const deltaX = middleFinger.x - wrist.x
        const deltaY = middleFinger.y - wrist.y
        
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
          command.parameters = { direction: deltaX > 0 ? 'right' : 'left' }
        } else {
          command.parameters = { direction: deltaY > 0 ? 'down' : 'up' }
        }
        break

      case 'zoom':
        // Calculate zoom intensity
        const thumbTip = hand.landmarks[4]
        const indexTip = hand.landmarks[8]
        const distance = Math.sqrt(
          Math.pow(thumbTip.x - indexTip.x, 2) +
          Math.pow(thumbTip.y - indexTip.y, 2)
        )
        command.parameters = { 
          value: distance < 30 ? -1 : 1, // Zoom in or out
          intensity: Math.min(distance / 100, 1)
        }
        break

      case 'select':
        // Add target coordinates
        const pointer = hand.landmarks[8]
        command.parameters = {
          target: `${Math.round(pointer.x)},${Math.round(pointer.y)}`
        }
        break
    }

    return command
  }

  const executeGestureCommand = (command: GestureCommand) => {
    // Debounce rapid gestures
    const lastGesture = recognizedGestures[0]
    if (lastGesture && 
        lastGesture.action === command.action && 
        command.timestamp - lastGesture.timestamp < 1000) {
      return
    }

    setRecognizedGestures(prev => [command, ...prev.slice(0, 9)])
    onGestureCommand?.(command)

    // Show visual feedback
    addNotification({
      type: 'info',
      title: 'Gesture Recognized',
      message: `${command.action} gesture detected`,
      duration: 1500,
    })
  }

  // ===== CALIBRATION =====

  const startCalibration = () => {
    setIsCalibrating(true)
    
    // Simulate calibration process
    setTimeout(() => {
      const calibration: CalibrationData = {
        handSize: 0.8 + Math.random() * 0.4,
        reachDistance: 0.6 + Math.random() * 0.3,
        gestureSpeed: 0.7 + Math.random() * 0.3,
        sensitivity: sensitivity,
      }
      
      setCalibrationData(calibration)
      setIsCalibrating(false)
      onCalibrationComplete?.()
      
      addNotification({
        type: 'success',
        title: 'Calibration Complete',
        message: 'Gesture recognition has been calibrated to your hand',
        duration: 3000,
      })
    }, 3000)
  }

  // ===== RENDER HELPERS =====

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'select': return <MousePointer className="h-4 w-4" />
      case 'navigate': return <Move className="h-4 w-4" />
      case 'zoom': return <ZoomIn className="h-4 w-4" />
      case 'rotate': return <RotateCw className="h-4 w-4" />
      case 'menu': return <Layers className="h-4 w-4" />
      case 'back': return <RotateCcw className="h-4 w-4" />
      case 'confirm': return <CheckCircle className="h-4 w-4" />
      default: return <Hand className="h-4 w-4" />
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
      <div className="bg-gradient-to-r from-orange-600 to-red-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Hand className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">Advanced Gesture Recognition</h2>
            {isActive && (
              <motion.div
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="flex items-center space-x-1 text-white"
              >
                <Activity className="h-4 w-4" />
                <span className="text-sm">Tracking</span>
              </motion.div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={isActive ? stopGestureRecognition : startGestureRecognition}
              className={`px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 ${
                isActive
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-white bg-opacity-20 hover:bg-opacity-30 text-white'
              }`}
            >
              {isActive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              <span>{isActive ? 'Stop' : 'Start'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'recognition', label: 'Recognition', icon: <Hand className="h-4 w-4" /> },
          { id: 'patterns', label: 'Patterns', icon: <Target className="h-4 w-4" /> },
          { id: 'calibration', label: 'Calibration', icon: <Settings className="h-4 w-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-orange-600 border-b-2 border-orange-600'
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
            {/* Recognition Tab */}
            {activeTab === 'recognition' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Hand Tracking Status */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-900">Hand Tracking Status</h3>
                    
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <span className="font-medium text-gray-900">Detected Hands</span>
                        <span className="text-2xl font-bold text-orange-600">{currentHands.length}</span>
                      </div>
                      
                      {currentHands.map((hand, index) => (
                        <div key={index} className="flex items-center justify-between py-2 border-t border-gray-200">
                          <span className="text-sm text-gray-600 capitalize">{hand.handedness} Hand</span>
                          <span className={`text-sm font-medium ${getConfidenceColor(hand.confidence)}`}>
                            {(hand.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                      
                      {currentHands.length === 0 && (
                        <p className="text-sm text-gray-500 text-center py-4">
                          {isActive ? 'No hands detected' : 'Start tracking to detect hands'}
                        </p>
                      )}
                    </div>

                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">Sensitivity</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Low</span>
                          <span>High</span>
                        </div>
                        <input
                          type="range"
                          min="0.3"
                          max="1.0"
                          step="0.1"
                          value={sensitivity}
                          onChange={(e) => setSensitivity(Number(e.target.value))}
                          className="w-full"
                        />
                        <p className="text-center text-sm text-gray-600">
                          {(sensitivity * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Recent Gestures */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-900">Recent Gestures</h3>
                    
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {recognizedGestures.map((gesture, index) => (
                        <motion.div
                          key={`${gesture.timestamp}-${index}`}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          className="bg-gray-50 rounded-lg p-3"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center space-x-2">
                              {getActionIcon(gesture.action)}
                              <span className="font-medium text-gray-900 capitalize">{gesture.action}</span>
                            </div>
                            <span className={`text-sm ${getConfidenceColor(gesture.confidence)}`}>
                              {(gesture.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          
                          {gesture.parameters && (
                            <div className="text-xs text-gray-600">
                              {Object.entries(gesture.parameters).map(([key, value]) => (
                                <span key={key} className="mr-2">
                                  {key}: {String(value)}
                                </span>
                              ))}
                            </div>
                          )}
                          
                          <div className="text-xs text-gray-500 mt-1">
                            {new Date(gesture.timestamp).toLocaleTimeString()}
                          </div>
                        </motion.div>
                      ))}
                      
                      {recognizedGestures.length === 0 && (
                        <div className="text-center py-8 text-gray-500">
                          <Hand className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                          <p>No gestures recognized yet</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Patterns Tab */}
            {activeTab === 'patterns' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Gesture Patterns</h3>
                
                <div className="grid gap-4">
                  {gesturePatterns.map((pattern) => (
                    <div key={pattern.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-3">
                          {getActionIcon(pattern.action)}
                          <div>
                            <h4 className="font-medium text-gray-900">{pattern.name}</h4>
                            <p className="text-sm text-gray-600">{pattern.description}</p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-gray-500">
                            {(pattern.threshold * 100).toFixed(0)}% threshold
                          </span>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={pattern.enabled}
                              onChange={(e) => {
                                setGesturePatterns(prev =>
                                  prev.map(p =>
                                    p.id === pattern.id
                                      ? { ...p, enabled: e.target.checked }
                                      : p
                                  )
                                )
                              }}
                              className="sr-only peer"
                            />
                            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-orange-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-orange-600"></div>
                          </label>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Calibration Tab */}
            {activeTab === 'calibration' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Gesture Calibration</h3>
                  <button
                    onClick={startCalibration}
                    disabled={isCalibrating}
                    className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 flex items-center space-x-2"
                  >
                    {isCalibrating ? (
                      <>
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                        >
                          <Settings className="h-4 w-4" />
                        </motion.div>
                        <span>Calibrating...</span>
                      </>
                    ) : (
                      <>
                        <Target className="h-4 w-4" />
                        <span>Start Calibration</span>
                      </>
                    )}
                  </button>
                </div>

                {calibrationData ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-3">
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        <h4 className="font-medium text-green-900">Calibration Complete</h4>
                      </div>
                      
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-green-700">Hand Size:</span>
                          <span className="font-mono text-green-900">
                            {(calibrationData.handSize * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-green-700">Reach Distance:</span>
                          <span className="font-mono text-green-900">
                            {(calibrationData.reachDistance * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-green-700">Gesture Speed:</span>
                          <span className="font-mono text-green-900">
                            {(calibrationData.gestureSpeed * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">Calibration Tips</h4>
                      <ul className="text-sm text-gray-600 space-y-1">
                        <li>• Keep hands within camera view</li>
                        <li>• Ensure good lighting conditions</li>
                        <li>• Perform gestures at natural speed</li>
                        <li>• Recalibrate if accuracy decreases</li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Target className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500 mb-4">
                      Calibrate gesture recognition for optimal accuracy
                    </p>
                    <p className="text-sm text-gray-400">
                      Calibration adapts to your hand size and gesture style
                    </p>
                  </div>
                )}
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

export default GestureRecognition
