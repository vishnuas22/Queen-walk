'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Glasses, 
  Eye, 
  Hand, 
  Layers, 
  Zap,
  Target,
  Box,
  Scan,
  MapPin,
  Lightbulb,
  Settings,
  Play,
  Pause,
  Square,
  Camera,
  Volume2
} from 'lucide-react'
import { WebXRService, type WebXRCapabilities, type XRHitTestResult, type XRHandPose } from '../lib/webxr'
import { useUIActions } from '../store'

// ===== TYPES =====

interface ARInterfaceProps {
  onARStart?: () => void
  onAREnd?: () => void
  className?: string
}

interface ARObject {
  id: string
  type: 'text' | 'model' | 'ui' | 'data'
  position: { x: number; y: number; z: number }
  rotation: { x: number; y: number; z: number }
  scale: { x: number; y: number; z: number }
  content: any
  visible: boolean
  interactive: boolean
}

interface SpatialUI {
  id: string
  type: 'panel' | 'button' | 'menu' | 'widget'
  position: { x: number; y: number; z: number }
  size: { width: number; height: number }
  content: React.ReactNode
  anchored: boolean
  anchorId?: string
}

// ===== AR INTERFACE COMPONENT =====

export const ARInterface: React.FC<ARInterfaceProps> = ({
  onARStart,
  onAREnd,
  className = ''
}) => {
  const [isARSupported, setIsARSupported] = useState(false)
  const [isARActive, setIsARActive] = useState(false)
  const [capabilities, setCapabilities] = useState<WebXRCapabilities | null>(null)
  const [arObjects, setARObjects] = useState<ARObject[]>([])
  const [spatialUI, setSpatialUI] = useState<SpatialUI[]>([])
  const [hitTestResults, setHitTestResults] = useState<XRHitTestResult[]>([])
  const [handPoses, setHandPoses] = useState<{ left?: XRHandPose; right?: XRHandPose }>({})
  const [selectedObject, setSelectedObject] = useState<string | null>(null)
  const [placementMode, setPlacementMode] = useState(false)

  const { addNotification } = useUIActions()
  const webXRService = useRef<WebXRService | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Initialize WebXR service
  useEffect(() => {
    webXRService.current = WebXRService.getInstance()
    
    // Check capabilities
    const caps = webXRService.current.getCapabilities()
    setCapabilities(caps)
    setIsARSupported(caps?.isSupported && caps.supportedModes.includes('immersive-ar') || false)

    // Set up event listeners
    const unsubscribeSessionStart = webXRService.current.onSessionStart((session) => {
      setIsARActive(true)
      onARStart?.()
      
      addNotification({
        type: 'success',
        title: 'AR Session Started',
        message: 'Augmented reality is now active',
        duration: 3000,
      })
    })

    const unsubscribeSessionEnd = webXRService.current.onSessionEnd(() => {
      setIsARActive(false)
      onAREnd?.()
      
      addNotification({
        type: 'info',
        title: 'AR Session Ended',
        message: 'Augmented reality session has ended',
        duration: 3000,
      })
    })

    const unsubscribeHitTest = webXRService.current.onHitTest((results) => {
      setHitTestResults(results)
    })

    const unsubscribeHandTracking = webXRService.current.onHandTracking((hands) => {
      setHandPoses(hands)
    })

    return () => {
      unsubscribeSessionStart()
      unsubscribeSessionEnd()
      unsubscribeHitTest()
      unsubscribeHandTracking()
    }
  }, [onARStart, onAREnd, addNotification])

  // ===== AR SESSION MANAGEMENT =====

  const startARSession = async () => {
    if (!webXRService.current || !isARSupported) {
      addNotification({
        type: 'error',
        title: 'AR Not Supported',
        message: 'Augmented reality is not supported on this device',
        duration: 5000,
      })
      return
    }

    try {
      const success = await webXRService.current.startSession({
        mode: 'immersive-ar',
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hit-test', 'hand-tracking', 'light-estimation', 'plane-detection'],
      })

      if (success) {
        // Enable hit testing
        await webXRService.current.enableHitTesting()
        
        // Initialize default AR objects
        initializeARObjects()
      }
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'AR Session Failed',
        message: 'Failed to start augmented reality session',
        duration: 5000,
      })
    }
  }

  const endARSession = async () => {
    if (webXRService.current) {
      await webXRService.current.endSession()
    }
  }

  // ===== AR OBJECT MANAGEMENT =====

  const initializeARObjects = () => {
    const defaultObjects: ARObject[] = [
      {
        id: 'welcome-text',
        type: 'text',
        position: { x: 0, y: 1.5, z: -2 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1, z: 1 },
        content: 'Welcome to MasterX AR',
        visible: true,
        interactive: false,
      },
      {
        id: 'ai-assistant',
        type: 'ui',
        position: { x: 1, y: 1, z: -1.5 },
        rotation: { x: 0, y: -30, z: 0 },
        scale: { x: 1, y: 1, z: 1 },
        content: {
          type: 'assistant-panel',
          title: 'AI Assistant',
          status: 'ready',
        },
        visible: true,
        interactive: true,
      },
      {
        id: 'data-visualization',
        type: 'data',
        position: { x: -1, y: 1.2, z: -1.5 },
        rotation: { x: 0, y: 30, z: 0 },
        scale: { x: 0.8, y: 0.8, z: 0.8 },
        content: {
          type: 'chart',
          data: [65, 78, 90, 81, 56, 55, 40],
          labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        },
        visible: true,
        interactive: true,
      },
    ]

    setARObjects(defaultObjects)

    // Initialize spatial UI
    const defaultSpatialUI: SpatialUI[] = [
      {
        id: 'main-menu',
        type: 'panel',
        position: { x: 0, y: 0.8, z: -1 },
        size: { width: 0.6, height: 0.4 },
        content: <ARMainMenu />,
        anchored: false,
      },
      {
        id: 'controls',
        type: 'widget',
        position: { x: 0.8, y: 0.5, z: -0.8 },
        size: { width: 0.3, height: 0.5 },
        content: <ARControls />,
        anchored: false,
      },
    ]

    setSpatialUI(defaultSpatialUI)
  }

  const placeObjectAtHitTest = (objectType: string) => {
    if (hitTestResults.length === 0) return

    const hitResult = hitTestResults[0]
    const newObject: ARObject = {
      id: `placed-${Date.now()}`,
      type: objectType as any,
      position: hitResult.pose.position,
      rotation: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1, z: 1 },
      content: `Placed ${objectType}`,
      visible: true,
      interactive: true,
    }

    setARObjects(prev => [...prev, newObject])
    setPlacementMode(false)

    addNotification({
      type: 'success',
      title: 'Object Placed',
      message: `${objectType} placed in AR space`,
      duration: 2000,
    })
  }

  // ===== RENDER HELPERS =====

  const getCapabilityStatus = (feature: keyof WebXRCapabilities) => {
    if (!capabilities) return false
    return capabilities[feature] as boolean
  }

  const getHandTrackingStatus = () => {
    const leftTracked = handPoses.left?.isTracked || false
    const rightTracked = handPoses.right?.isTracked || false
    
    if (leftTracked && rightTracked) return 'Both hands tracked'
    if (leftTracked) return 'Left hand tracked'
    if (rightTracked) return 'Right hand tracked'
    return 'No hands tracked'
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-600 to-blue-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Glasses className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">Augmented Reality Interface</h2>
            {isARActive && (
              <motion.div
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="flex items-center space-x-1 text-white"
              >
                <Eye className="h-4 w-4" />
                <span className="text-sm">AR Active</span>
              </motion.div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {isARSupported ? (
              <div className="flex items-center space-x-1 text-white text-sm">
                <Zap className="h-4 w-4" />
                <span>AR Ready</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1 text-white text-sm opacity-60">
                <Square className="h-4 w-4" />
                <span>AR Not Available</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        {!isARSupported ? (
          /* AR Not Supported */
          <div className="text-center py-12">
            <Glasses className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">AR Not Available</h3>
            <p className="text-gray-600 mb-4">
              Augmented reality is not supported on this device or browser.
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-md mx-auto">
              <h4 className="font-medium text-blue-900 mb-2">Requirements:</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• WebXR compatible browser (Chrome, Edge)</li>
                <li>• AR-capable device (Android with ARCore)</li>
                <li>• HTTPS connection required</li>
                <li>• Camera permissions granted</li>
              </ul>
            </div>
          </div>
        ) : !isARActive ? (
          /* AR Ready to Start */
          <div className="space-y-6">
            <div className="text-center">
              <Glasses className="h-16 w-16 text-cyan-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Ready for Augmented Reality</h3>
              <p className="text-gray-600 mb-6">
                Experience MasterX in augmented reality with spatial computing features.
              </p>
              
              <button
                onClick={startARSession}
                className="px-6 py-3 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors flex items-center space-x-2 mx-auto"
              >
                <Play className="h-5 w-5" />
                <span>Start AR Experience</span>
              </button>
            </div>

            {/* Capabilities Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              <div className={`p-4 rounded-lg border ${getCapabilityStatus('hasHitTest') ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
                <Target className={`h-6 w-6 mx-auto mb-2 ${getCapabilityStatus('hasHitTest') ? 'text-green-600' : 'text-gray-400'}`} />
                <p className="text-sm text-center font-medium">Hit Testing</p>
                <p className="text-xs text-center text-gray-500">Object placement</p>
              </div>
              
              <div className={`p-4 rounded-lg border ${getCapabilityStatus('hasHandTracking') ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
                <Hand className={`h-6 w-6 mx-auto mb-2 ${getCapabilityStatus('hasHandTracking') ? 'text-green-600' : 'text-gray-400'}`} />
                <p className="text-sm text-center font-medium">Hand Tracking</p>
                <p className="text-xs text-center text-gray-500">Gesture control</p>
              </div>
              
              <div className={`p-4 rounded-lg border ${getCapabilityStatus('hasPlaneDetection') ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
                <Layers className={`h-6 w-6 mx-auto mb-2 ${getCapabilityStatus('hasPlaneDetection') ? 'text-green-600' : 'text-gray-400'}`} />
                <p className="text-sm text-center font-medium">Plane Detection</p>
                <p className="text-xs text-center text-gray-500">Surface mapping</p>
              </div>
              
              <div className={`p-4 rounded-lg border ${getCapabilityStatus('hasLightEstimation') ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
                <Lightbulb className={`h-6 w-6 mx-auto mb-2 ${getCapabilityStatus('hasLightEstimation') ? 'text-green-600' : 'text-gray-400'}`} />
                <p className="text-sm text-center font-medium">Light Estimation</p>
                <p className="text-xs text-center text-gray-500">Realistic lighting</p>
              </div>
            </div>
          </div>
        ) : (
          /* AR Active Interface */
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">AR Session Active</h3>
              <button
                onClick={endARSession}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center space-x-2"
              >
                <Square className="h-4 w-4" />
                <span>End AR</span>
              </button>
            </div>

            {/* AR Status */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Eye className="h-5 w-5 text-green-600" />
                  <span className="font-medium text-green-900">Tracking Status</span>
                </div>
                <p className="text-sm text-green-700">6DOF tracking active</p>
              </div>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Hand className="h-5 w-5 text-blue-600" />
                  <span className="font-medium text-blue-900">Hand Tracking</span>
                </div>
                <p className="text-sm text-blue-700">{getHandTrackingStatus()}</p>
              </div>
              
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Box className="h-5 w-5 text-purple-600" />
                  <span className="font-medium text-purple-900">AR Objects</span>
                </div>
                <p className="text-sm text-purple-700">{arObjects.length} objects placed</p>
              </div>
            </div>

            {/* Object Placement Controls */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-3">Object Placement</h4>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setPlacementMode(!placementMode)}
                  className={`px-3 py-2 rounded-lg transition-colors ${
                    placementMode 
                      ? 'bg-cyan-600 text-white' 
                      : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Target className="h-4 w-4 inline mr-1" />
                  {placementMode ? 'Cancel Placement' : 'Place Object'}
                </button>
                
                {placementMode && hitTestResults.length > 0 && (
                  <div className="flex space-x-2">
                    <button
                      onClick={() => placeObjectAtHitTest('text')}
                      className="px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                    >
                      Place Text
                    </button>
                    <button
                      onClick={() => placeObjectAtHitTest('ui')}
                      className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      Place UI
                    </button>
                    <button
                      onClick={() => placeObjectAtHitTest('data')}
                      className="px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                    >
                      Place Data
                    </button>
                  </div>
                )}
              </div>
              
              {placementMode && hitTestResults.length === 0 && (
                <p className="text-sm text-gray-600 mt-2">
                  Point your device at a surface to place objects
                </p>
              )}
            </div>

            {/* AR Objects List */}
            <div className="space-y-2">
              <h4 className="font-medium text-gray-900">Placed Objects</h4>
              {arObjects.map((object) => (
                <div
                  key={object.id}
                  className={`flex items-center justify-between p-3 border rounded-lg cursor-pointer transition-colors ${
                    selectedObject === object.id 
                      ? 'border-cyan-500 bg-cyan-50' 
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedObject(selectedObject === object.id ? null : object.id)}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${object.visible ? 'bg-green-500' : 'bg-gray-400'}`} />
                    <div>
                      <p className="font-medium text-gray-900">{object.type}</p>
                      <p className="text-sm text-gray-500">
                        Position: ({object.position.x.toFixed(1)}, {object.position.y.toFixed(1)}, {object.position.z.toFixed(1)})
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setARObjects(prev => 
                          prev.map(obj => 
                            obj.id === object.id 
                              ? { ...obj, visible: !obj.visible }
                              : obj
                          )
                        )
                      }}
                      className="p-1 text-gray-400 hover:text-gray-600"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setARObjects(prev => prev.filter(obj => obj.id !== object.id))
                      }}
                      className="p-1 text-red-400 hover:text-red-600"
                    >
                      <Square className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
              
              {arObjects.length === 0 && (
                <p className="text-gray-500 text-center py-4">No objects placed yet</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ===== AR UI COMPONENTS =====

const ARMainMenu: React.FC = () => (
  <div className="bg-white bg-opacity-90 rounded-lg p-4 backdrop-blur-sm">
    <h3 className="font-semibold text-gray-900 mb-3">MasterX AR</h3>
    <div className="space-y-2">
      <button className="w-full px-3 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition-colors">
        AI Assistant
      </button>
      <button className="w-full px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors">
        Data Visualization
      </button>
      <button className="w-full px-3 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors">
        Collaboration
      </button>
    </div>
  </div>
)

const ARControls: React.FC = () => (
  <div className="bg-white bg-opacity-90 rounded-lg p-3 backdrop-blur-sm">
    <h4 className="font-medium text-gray-900 mb-2">Controls</h4>
    <div className="space-y-2">
      <button className="w-full p-2 bg-gray-100 rounded hover:bg-gray-200 transition-colors">
        <Camera className="h-4 w-4 mx-auto" />
      </button>
      <button className="w-full p-2 bg-gray-100 rounded hover:bg-gray-200 transition-colors">
        <Volume2 className="h-4 w-4 mx-auto" />
      </button>
      <button className="w-full p-2 bg-gray-100 rounded hover:bg-gray-200 transition-colors">
        <Settings className="h-4 w-4 mx-auto" />
      </button>
    </div>
  </div>
)

export default ARInterface
