'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Heart, 
  Smile, 
  Frown, 
  Meh,
  Zap,
  AlertTriangle,
  TrendingUp,
  Activity,
  Brain,
  Mic,
  Camera,
  Settings,
  Play,
  Pause,
  BarChart3
} from 'lucide-react'
import { useUIActions } from '../store'

// ===== TYPES =====

interface EmotionRecognitionProps {
  onEmotionChange?: (emotion: EmotionState) => void
  onMoodAnalysis?: (analysis: MoodAnalysis) => void
  className?: string
}

interface EmotionState {
  primary: 'happy' | 'sad' | 'angry' | 'surprised' | 'fearful' | 'disgusted' | 'neutral'
  confidence: number
  valence: number // -1 (negative) to 1 (positive)
  arousal: number // 0 (calm) to 1 (excited)
  intensity: number
  secondary: string[]
  source: 'facial' | 'vocal' | 'combined'
  timestamp: number
}

interface MoodAnalysis {
  averageValence: number
  averageArousal: number
  dominantEmotion: string
  emotionVariability: number
  stressLevel: number
  engagementLevel: number
  timespan: number
  recommendations: string[]
}

interface EmotionHistory {
  emotions: EmotionState[]
  sessionStart: number
  totalDuration: number
}

interface VoiceEmotionData {
  pitch: number
  energy: number
  tempo: number
  spectralCentroid: number
  mfcc: number[]
  emotion: string
  confidence: number
}

// ===== EMOTION RECOGNITION COMPONENT =====

export const EmotionRecognition: React.FC<EmotionRecognitionProps> = ({
  onEmotionChange,
  onMoodAnalysis,
  className = ''
}) => {
  const [isActive, setIsActive] = useState(false)
  const [currentEmotion, setCurrentEmotion] = useState<EmotionState | null>(null)
  const [emotionHistory, setEmotionHistory] = useState<EmotionHistory>({
    emotions: [],
    sessionStart: Date.now(),
    totalDuration: 0,
  })
  const [moodAnalysis, setMoodAnalysis] = useState<MoodAnalysis | null>(null)
  const [activeTab, setActiveTab] = useState<'current' | 'history' | 'analysis' | 'settings'>('current')
  const [recognitionMode, setRecognitionMode] = useState<'facial' | 'vocal' | 'combined'>('combined')
  const [sensitivity, setSensitivity] = useState(0.7)

  const { addNotification } = useUIActions()
  const videoRef = useRef<HTMLVideoElement>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationFrameRef = useRef<number>(0)

  // ===== EMOTION RECOGNITION SYSTEM =====

  const startEmotionRecognition = async () => {
    try {
      // Request camera and microphone access
      const constraints: MediaStreamConstraints = {
        video: recognitionMode !== 'vocal',
        audio: recognitionMode !== 'facial',
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)

      if (videoRef.current && constraints.video) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }

      if (constraints.audio) {
        setupAudioAnalysis(stream)
      }

      setIsActive(true)
      startEmotionDetection()

      addNotification({
        type: 'success',
        title: 'Emotion Recognition Started',
        message: `${recognitionMode} emotion detection is now active`,
        duration: 3000,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Access Failed',
        message: 'Unable to access camera or microphone',
        duration: 5000,
      })
    }
  }

  const stopEmotionRecognition = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }

    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }

    setIsActive(false)
    setCurrentEmotion(null)
  }

  const setupAudioAnalysis = (stream: MediaStream) => {
    audioContextRef.current = new AudioContext()
    const source = audioContextRef.current.createMediaStreamSource(stream)
    analyserRef.current = audioContextRef.current.createAnalyser()
    
    analyserRef.current.fftSize = 2048
    source.connect(analyserRef.current)
  }

  const startEmotionDetection = () => {
    const detectEmotions = () => {
      if (!isActive) return

      // Simulate emotion detection
      const emotion = generateSimulatedEmotion()
      
      setCurrentEmotion(emotion)
      setEmotionHistory(prev => ({
        ...prev,
        emotions: [emotion, ...prev.emotions.slice(0, 99)],
        totalDuration: Date.now() - prev.sessionStart,
      }))

      onEmotionChange?.(emotion)

      // Update mood analysis every 10 emotions
      if (emotionHistory.emotions.length % 10 === 0) {
        updateMoodAnalysis()
      }

      animationFrameRef.current = requestAnimationFrame(detectEmotions)
    }

    detectEmotions()
  }

  const generateSimulatedEmotion = (): EmotionState => {
    const emotions: EmotionState['primary'][] = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
    const primary = emotions[Math.floor(Math.random() * emotions.length)]
    
    // Simulate facial emotion detection
    let facialEmotion: Partial<EmotionState> = {}
    if (recognitionMode !== 'vocal') {
      facialEmotion = {
        primary,
        confidence: 0.7 + Math.random() * 0.3,
        valence: getEmotionValence(primary) + (Math.random() - 0.5) * 0.4,
        arousal: getEmotionArousal(primary) + (Math.random() - 0.5) * 0.4,
        intensity: Math.random(),
      }
    }

    // Simulate vocal emotion detection
    let vocalEmotion: Partial<EmotionState> = {}
    if (recognitionMode !== 'facial') {
      const voiceData = analyzeVoiceEmotion()
      vocalEmotion = {
        primary: voiceData.emotion as EmotionState['primary'],
        confidence: voiceData.confidence,
        valence: (Math.random() - 0.5) * 2,
        arousal: Math.random(),
        intensity: voiceData.energy,
      }
    }

    // Combine facial and vocal if both are active
    let combinedEmotion: EmotionState
    if (recognitionMode === 'combined') {
      combinedEmotion = {
        primary: Math.random() > 0.5 ? facialEmotion.primary! : vocalEmotion.primary!,
        confidence: Math.max(facialEmotion.confidence || 0, vocalEmotion.confidence || 0),
        valence: ((facialEmotion.valence || 0) + (vocalEmotion.valence || 0)) / 2,
        arousal: ((facialEmotion.arousal || 0) + (vocalEmotion.arousal || 0)) / 2,
        intensity: Math.max(facialEmotion.intensity || 0, vocalEmotion.intensity || 0),
        secondary: ['focused', 'engaged'],
        source: 'combined',
        timestamp: Date.now(),
      }
    } else if (recognitionMode === 'facial') {
      combinedEmotion = {
        ...facialEmotion,
        secondary: ['attentive'],
        source: 'facial',
        timestamp: Date.now(),
      } as EmotionState
    } else {
      combinedEmotion = {
        ...vocalEmotion,
        secondary: ['vocal'],
        source: 'vocal',
        timestamp: Date.now(),
      } as EmotionState
    }

    // Apply sensitivity adjustment
    combinedEmotion.confidence *= sensitivity

    return combinedEmotion
  }

  const analyzeVoiceEmotion = (): VoiceEmotionData => {
    if (!analyserRef.current) {
      return {
        pitch: 0,
        energy: 0,
        tempo: 0,
        spectralCentroid: 0,
        mfcc: [],
        emotion: 'neutral',
        confidence: 0,
      }
    }

    const bufferLength = analyserRef.current.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    analyserRef.current.getByteFrequencyData(dataArray)

    // Calculate audio features
    const energy = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength / 255
    const pitch = findPitch(dataArray)
    const spectralCentroid = calculateSpectralCentroid(dataArray)

    // Map audio features to emotions
    let emotion = 'neutral'
    let confidence = 0.5

    if (energy > 0.6 && pitch > 200) {
      emotion = 'happy'
      confidence = 0.8
    } else if (energy < 0.3 && pitch < 150) {
      emotion = 'sad'
      confidence = 0.7
    } else if (energy > 0.7 && spectralCentroid > 2000) {
      emotion = 'angry'
      confidence = 0.75
    } else if (energy > 0.5 && pitch > 250) {
      emotion = 'surprised'
      confidence = 0.7
    }

    return {
      pitch,
      energy,
      tempo: 120 + Math.random() * 60,
      spectralCentroid,
      mfcc: Array.from({ length: 13 }, () => Math.random()),
      emotion,
      confidence,
    }
  }

  const findPitch = (dataArray: Uint8Array): number => {
    // Simplified pitch detection
    let maxIndex = 0
    let maxValue = 0
    
    for (let i = 0; i < dataArray.length; i++) {
      if (dataArray[i] > maxValue) {
        maxValue = dataArray[i]
        maxIndex = i
      }
    }
    
    return maxIndex * (audioContextRef.current?.sampleRate || 44100) / (2 * dataArray.length)
  }

  const calculateSpectralCentroid = (dataArray: Uint8Array): number => {
    let weightedSum = 0
    let magnitudeSum = 0
    
    for (let i = 0; i < dataArray.length; i++) {
      weightedSum += i * dataArray[i]
      magnitudeSum += dataArray[i]
    }
    
    return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0
  }

  const getEmotionValence = (emotion: string): number => {
    const valenceMap: { [key: string]: number } = {
      happy: 0.8,
      surprised: 0.3,
      neutral: 0,
      disgusted: -0.4,
      angry: -0.6,
      fearful: -0.5,
      sad: -0.8,
    }
    return valenceMap[emotion] || 0
  }

  const getEmotionArousal = (emotion: string): number => {
    const arousalMap: { [key: string]: number } = {
      angry: 0.9,
      surprised: 0.8,
      fearful: 0.7,
      happy: 0.6,
      disgusted: 0.4,
      sad: 0.3,
      neutral: 0.2,
    }
    return arousalMap[emotion] || 0.5
  }

  const updateMoodAnalysis = () => {
    if (emotionHistory.emotions.length < 5) return

    const recentEmotions = emotionHistory.emotions.slice(0, 20)
    
    const averageValence = recentEmotions.reduce((sum, e) => sum + e.valence, 0) / recentEmotions.length
    const averageArousal = recentEmotions.reduce((sum, e) => sum + e.arousal, 0) / recentEmotions.length
    
    // Find dominant emotion
    const emotionCounts: { [key: string]: number } = {}
    recentEmotions.forEach(e => {
      emotionCounts[e.primary] = (emotionCounts[e.primary] || 0) + 1
    })
    const dominantEmotion = Object.keys(emotionCounts).reduce((a, b) => 
      emotionCounts[a] > emotionCounts[b] ? a : b
    )

    // Calculate variability
    const valenceVariance = recentEmotions.reduce((sum, e) => 
      sum + Math.pow(e.valence - averageValence, 2), 0
    ) / recentEmotions.length
    const emotionVariability = Math.sqrt(valenceVariance)

    // Calculate stress and engagement
    const stressLevel = Math.max(0, Math.min(1, (1 - averageValence + averageArousal) / 2))
    const engagementLevel = Math.max(0, Math.min(1, averageArousal))

    const analysis: MoodAnalysis = {
      averageValence,
      averageArousal,
      dominantEmotion,
      emotionVariability,
      stressLevel,
      engagementLevel,
      timespan: emotionHistory.totalDuration,
      recommendations: generateRecommendations(averageValence, averageArousal, stressLevel),
    }

    setMoodAnalysis(analysis)
    onMoodAnalysis?.(analysis)
  }

  const generateRecommendations = (valence: number, arousal: number, stress: number): string[] => {
    const recommendations: string[] = []

    if (valence < -0.3) {
      recommendations.push('Consider taking a short break to improve mood')
    }
    if (arousal > 0.7) {
      recommendations.push('Try some calming exercises to reduce arousal')
    }
    if (stress > 0.6) {
      recommendations.push('Stress levels are elevated - consider relaxation techniques')
    }
    if (valence > 0.5 && arousal > 0.5) {
      recommendations.push('Great emotional state for productive work!')
    }

    return recommendations
  }

  // ===== RENDER HELPERS =====

  const getEmotionIcon = (emotion: string) => {
    switch (emotion) {
      case 'happy': return <Smile className="h-6 w-6 text-green-500" />
      case 'sad': return <Frown className="h-6 w-6 text-blue-500" />
      case 'angry': return <AlertTriangle className="h-6 w-6 text-red-500" />
      case 'surprised': return <Zap className="h-6 w-6 text-yellow-500" />
      case 'neutral': return <Meh className="h-6 w-6 text-gray-500" />
      default: return <Heart className="h-6 w-6 text-purple-500" />
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getValenceColor = (valence: number) => {
    if (valence > 0.3) return 'text-green-600'
    if (valence > -0.3) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-pink-600 to-purple-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Heart className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">Emotion Recognition</h2>
            {isActive && (
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="flex items-center space-x-1 text-white"
              >
                <Activity className="h-4 w-4" />
                <span className="text-sm">Detecting</span>
              </motion.div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <select
              value={recognitionMode}
              onChange={(e) => setRecognitionMode(e.target.value as any)}
              className="px-3 py-1 bg-white bg-opacity-20 text-white rounded text-sm border border-white border-opacity-30"
            >
              <option value="facial">Facial Only</option>
              <option value="vocal">Vocal Only</option>
              <option value="combined">Combined</option>
            </select>
            
            <button
              onClick={isActive ? stopEmotionRecognition : startEmotionRecognition}
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
          { id: 'current', label: 'Current', icon: <Heart className="h-4 w-4" /> },
          { id: 'history', label: 'History', icon: <Activity className="h-4 w-4" /> },
          { id: 'analysis', label: 'Analysis', icon: <BarChart3 className="h-4 w-4" /> },
          { id: 'settings', label: 'Settings', icon: <Settings className="h-4 w-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-pink-600 border-b-2 border-pink-600'
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
            {/* Current Emotion Tab */}
            {activeTab === 'current' && (
              <div className="space-y-6">
                {currentEmotion ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-6 text-center">
                        <div className="flex justify-center mb-4">
                          {getEmotionIcon(currentEmotion.primary)}
                        </div>
                        <h3 className="text-2xl font-bold text-gray-900 capitalize mb-2">
                          {currentEmotion.primary}
                        </h3>
                        <p className={`text-lg font-medium ${getConfidenceColor(currentEmotion.confidence)}`}>
                          {(currentEmotion.confidence * 100).toFixed(1)}% confidence
                        </p>
                        <p className="text-sm text-gray-600 mt-2">
                          Source: {currentEmotion.source}
                        </p>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Emotion Dimensions</h4>
                        <div className="space-y-3">
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Valence (Positive/Negative)</span>
                              <span className={getValenceColor(currentEmotion.valence)}>
                                {currentEmotion.valence.toFixed(2)}
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  currentEmotion.valence > 0 ? 'bg-green-500' : 'bg-red-500'
                                }`}
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
                          
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Intensity</span>
                              <span>{(currentEmotion.intensity * 100).toFixed(0)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-purple-500 h-2 rounded-full"
                                style={{ width: `${currentEmotion.intensity * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Secondary Emotions</h4>
                        <div className="flex flex-wrap gap-2">
                          {currentEmotion.secondary.map((emotion, index) => (
                            <span
                              key={index}
                              className="px-3 py-1 bg-purple-100 text-purple-700 text-sm rounded-full"
                            >
                              {emotion}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Real-time Visualization</h4>
                        <div className="relative h-32 bg-white border rounded overflow-hidden">
                          <motion.div
                            className="absolute bottom-0 left-0 bg-gradient-to-t from-pink-500 to-purple-500"
                            animate={{
                              height: `${currentEmotion.intensity * 100}%`,
                              width: '20%',
                            }}
                            transition={{ duration: 0.5 }}
                          />
                          <motion.div
                            className="absolute bottom-0 left-1/4 bg-gradient-to-t from-blue-500 to-cyan-500"
                            animate={{
                              height: `${currentEmotion.arousal * 100}%`,
                              width: '20%',
                            }}
                            transition={{ duration: 0.5 }}
                          />
                          <motion.div
                            className="absolute bottom-0 left-2/4 bg-gradient-to-t from-green-500 to-yellow-500"
                            animate={{
                              height: `${(currentEmotion.valence + 1) * 50}%`,
                              width: '20%',
                            }}
                            transition={{ duration: 0.5 }}
                          />
                          <motion.div
                            className="absolute bottom-0 left-3/4 bg-gradient-to-t from-red-500 to-orange-500"
                            animate={{
                              height: `${currentEmotion.confidence * 100}%`,
                              width: '20%',
                            }}
                            transition={{ duration: 0.5 }}
                          />
                        </div>
                        <div className="flex justify-between text-xs text-gray-600 mt-2">
                          <span>Intensity</span>
                          <span>Arousal</span>
                          <span>Valence</span>
                          <span>Confidence</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Heart className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">
                      {isActive 
                        ? 'Analyzing emotions...' 
                        : 'Start emotion recognition to see current emotional state'
                      }
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* History Tab */}
            {activeTab === 'history' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Emotion History</h3>
                  <div className="text-sm text-gray-600">
                    {emotionHistory.emotions.length} emotions recorded
                  </div>
                </div>

                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {emotionHistory.emotions.map((emotion, index) => (
                    <motion.div
                      key={`${emotion.timestamp}-${index}`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-center justify-between bg-gray-50 rounded-lg p-3"
                    >
                      <div className="flex items-center space-x-3">
                        {getEmotionIcon(emotion.primary)}
                        <div>
                          <p className="font-medium text-gray-900 capitalize">{emotion.primary}</p>
                          <p className="text-xs text-gray-600">
                            {emotion.source} • {(emotion.intensity * 100).toFixed(0)}% intensity
                          </p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <p className={`text-sm font-medium ${getConfidenceColor(emotion.confidence)}`}>
                          {(emotion.confidence * 100).toFixed(0)}%
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(emotion.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                  
                  {emotionHistory.emotions.length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      <Activity className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                      <p>No emotion history yet</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Analysis Tab */}
            {activeTab === 'analysis' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Mood Analysis</h3>
                
                {moodAnalysis ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Overall Mood</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Dominant Emotion:</span>
                            <span className="font-medium capitalize">{moodAnalysis.dominantEmotion}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Average Valence:</span>
                            <span className={`font-medium ${getValenceColor(moodAnalysis.averageValence)}`}>
                              {moodAnalysis.averageValence.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Average Arousal:</span>
                            <span className="font-medium">{moodAnalysis.averageArousal.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Emotion Variability:</span>
                            <span className="font-medium">{moodAnalysis.emotionVariability.toFixed(2)}</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Wellness Indicators</h4>
                        <div className="space-y-3">
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Stress Level</span>
                              <span>{(moodAnalysis.stressLevel * 100).toFixed(0)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  moodAnalysis.stressLevel > 0.6 ? 'bg-red-500' :
                                  moodAnalysis.stressLevel > 0.3 ? 'bg-yellow-500' : 'bg-green-500'
                                }`}
                                style={{ width: `${moodAnalysis.stressLevel * 100}%` }}
                              />
                            </div>
                          </div>
                          
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Engagement Level</span>
                              <span>{(moodAnalysis.engagementLevel * 100).toFixed(0)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-blue-500 h-2 rounded-full"
                                style={{ width: `${moodAnalysis.engagementLevel * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Recommendations</h4>
                        <div className="space-y-2">
                          {moodAnalysis.recommendations.map((rec, index) => (
                            <div key={index} className="flex items-start space-x-2">
                              <TrendingUp className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                              <p className="text-sm text-gray-700">{rec}</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-3">Session Info</h4>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Duration:</span>
                            <span className="font-medium">
                              {Math.round(moodAnalysis.timespan / 60000)} minutes
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Emotions Analyzed:</span>
                            <span className="font-medium">{emotionHistory.emotions.length}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <BarChart3 className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">
                      Mood analysis will be available after detecting several emotions
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Recognition Settings</h3>
                
                <div className="grid gap-6">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Detection Mode</h4>
                    <div className="space-y-2">
                      {[
                        { value: 'facial', label: 'Facial Recognition', icon: <Camera className="h-4 w-4" /> },
                        { value: 'vocal', label: 'Voice Analysis', icon: <Mic className="h-4 w-4" /> },
                        { value: 'combined', label: 'Combined Analysis', icon: <Brain className="h-4 w-4" /> },
                      ].map((mode) => (
                        <label key={mode.value} className="flex items-center space-x-3 cursor-pointer">
                          <input
                            type="radio"
                            name="recognitionMode"
                            value={mode.value}
                            checked={recognitionMode === mode.value}
                            onChange={(e) => setRecognitionMode(e.target.value as any)}
                            className="text-pink-600 focus:ring-pink-500"
                          />
                          <div className="flex items-center space-x-2">
                            {mode.icon}
                            <span className="text-sm font-medium text-gray-900">{mode.label}</span>
                          </div>
                        </label>
                      ))}
                    </div>
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
      </div>
    </div>
  )
}

export default EmotionRecognition
