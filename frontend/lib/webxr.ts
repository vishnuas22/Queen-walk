// WebXR Service for Augmented Reality and Virtual Reality Features

// ===== TYPES =====

export interface WebXRCapabilities {
  isSupported: boolean
  supportedModes: ('immersive-ar' | 'immersive-vr' | 'inline')[]
  hasHandTracking: boolean
  hasEyeTracking: boolean
  hasDepthSensing: boolean
  hasLightEstimation: boolean
  hasHitTest: boolean
  hasAnchors: boolean
  hasPlaneDetection: boolean
}

export interface XRSessionConfig {
  mode: 'immersive-ar' | 'immersive-vr' | 'inline'
  requiredFeatures: string[]
  optionalFeatures: string[]
}

export interface XRPose {
  position: { x: number; y: number; z: number }
  orientation: { x: number; y: number; z: number; w: number }
  linearVelocity?: { x: number; y: number; z: number }
  angularVelocity?: { x: number; y: number; z: number }
}

export interface XRHitTestResult {
  pose: XRPose
  distance: number
  confidence: number
  type: 'plane' | 'point' | 'mesh'
}

export interface XRHandPose {
  joints: { [jointName: string]: XRPose }
  confidence: number
  isTracked: boolean
}

export interface XREnvironmentData {
  lightEstimate?: {
    intensity: number
    direction: { x: number; y: number; z: number }
    color: { r: number; g: number; b: number }
  }
  planes: Array<{
    id: string
    pose: XRPose
    polygon: { x: number; z: number }[]
    orientation: 'horizontal' | 'vertical'
  }>
  anchors: Array<{
    id: string
    pose: XRPose
    persistent: boolean
  }>
}

// ===== WEBXR SERVICE =====

export class WebXRService {
  private static instance: WebXRService
  private session: XRSession | null = null
  private referenceSpace: XRReferenceSpace | null = null
  private renderer: any = null // Three.js renderer would go here
  private scene: any = null // Three.js scene would go here
  private camera: any = null // Three.js camera would go here
  
  // Event handlers
  private sessionStartHandlers: ((session: XRSession) => void)[] = []
  private sessionEndHandlers: (() => void)[] = []
  private frameHandlers: ((frame: XRFrame, session: XRSession) => void)[] = []
  private hitTestHandlers: ((results: XRHitTestResult[]) => void)[] = []
  private handTrackingHandlers: ((hands: { left?: XRHandPose; right?: XRHandPose }) => void)[] = []
  
  // State
  private isSessionActive = false
  private capabilities: WebXRCapabilities | null = null
  private environmentData: XREnvironmentData = { planes: [], anchors: [] }

  private constructor() {
    this.checkCapabilities()
  }

  public static getInstance(): WebXRService {
    if (!WebXRService.instance) {
      WebXRService.instance = new WebXRService()
    }
    return WebXRService.instance
  }

  // ===== CAPABILITY DETECTION =====

  private async checkCapabilities(): Promise<void> {
    if (!navigator.xr) {
      this.capabilities = {
        isSupported: false,
        supportedModes: [],
        hasHandTracking: false,
        hasEyeTracking: false,
        hasDepthSensing: false,
        hasLightEstimation: false,
        hasHitTest: false,
        hasAnchors: false,
        hasPlaneDetection: false,
      }
      return
    }

    try {
      const supportedModes: ('immersive-ar' | 'immersive-vr' | 'inline')[] = []
      
      // Check AR support
      if (await navigator.xr.isSessionSupported('immersive-ar')) {
        supportedModes.push('immersive-ar')
      }
      
      // Check VR support
      if (await navigator.xr.isSessionSupported('immersive-vr')) {
        supportedModes.push('immersive-vr')
      }
      
      // Check inline support
      if (await navigator.xr.isSessionSupported('inline')) {
        supportedModes.push('inline')
      }

      this.capabilities = {
        isSupported: supportedModes.length > 0,
        supportedModes,
        hasHandTracking: 'hand-tracking' in XRSession.prototype,
        hasEyeTracking: false, // Not widely supported yet
        hasDepthSensing: 'depth-sensing' in XRSession.prototype,
        hasLightEstimation: 'light-estimation' in XRSession.prototype,
        hasHitTest: 'hit-test' in XRSession.prototype,
        hasAnchors: 'anchors' in XRSession.prototype,
        hasPlaneDetection: 'plane-detection' in XRSession.prototype,
      }
    } catch (error) {
      console.warn('WebXR capability check failed:', error)
      this.capabilities = {
        isSupported: false,
        supportedModes: [],
        hasHandTracking: false,
        hasEyeTracking: false,
        hasDepthSensing: false,
        hasLightEstimation: false,
        hasHitTest: false,
        hasAnchors: false,
        hasPlaneDetection: false,
      }
    }
  }

  // ===== SESSION MANAGEMENT =====

  public async startSession(config: XRSessionConfig): Promise<boolean> {
    if (!this.capabilities?.isSupported) {
      throw new Error('WebXR is not supported on this device')
    }

    if (!this.capabilities.supportedModes.includes(config.mode)) {
      throw new Error(`XR mode ${config.mode} is not supported`)
    }

    try {
      this.session = await navigator.xr!.requestSession(config.mode, {
        requiredFeatures: config.requiredFeatures,
        optionalFeatures: config.optionalFeatures,
      })

      // Set up reference space
      this.referenceSpace = await this.session.requestReferenceSpace('local-floor')
      
      // Set up event handlers
      this.session.addEventListener('end', () => {
        this.handleSessionEnd()
      })

      // Start render loop
      this.session.requestAnimationFrame((time, frame) => {
        this.handleFrame(time, frame)
      })

      this.isSessionActive = true
      this.notifySessionStart(this.session)
      
      return true
    } catch (error) {
      console.error('Failed to start XR session:', error)
      return false
    }
  }

  public async endSession(): Promise<void> {
    if (this.session) {
      await this.session.end()
    }
  }

  private handleSessionEnd(): void {
    this.session = null
    this.referenceSpace = null
    this.isSessionActive = false
    this.notifySessionEnd()
  }

  private handleFrame(time: number, frame: XRFrame): void {
    if (!this.session || !this.referenceSpace) return

    // Get viewer pose
    const pose = frame.getViewerPose(this.referenceSpace)
    if (!pose) return

    // Handle hit testing
    this.handleHitTesting(frame)
    
    // Handle hand tracking
    this.handleHandTracking(frame)
    
    // Update environment data
    this.updateEnvironmentData(frame)

    // Notify frame handlers
    this.notifyFrameHandlers(frame, this.session)

    // Continue render loop
    this.session.requestAnimationFrame((time, frame) => {
      this.handleFrame(time, frame)
    })
  }

  // ===== HIT TESTING =====

  private hitTestSource: XRHitTestSource | null = null

  public async enableHitTesting(): Promise<boolean> {
    if (!this.session || !this.capabilities?.hasHitTest) {
      return false
    }

    try {
      this.hitTestSource = await this.session.requestHitTestSource({ space: this.referenceSpace! })
      return true
    } catch (error) {
      console.error('Failed to enable hit testing:', error)
      return false
    }
  }

  private handleHitTesting(frame: XRFrame): void {
    if (!this.hitTestSource) return

    const hitTestResults = frame.getHitTestResults(this.hitTestSource)
    const results: XRHitTestResult[] = hitTestResults.map(result => ({
      pose: {
        position: {
          x: result.getPose(this.referenceSpace!)!.transform.position.x,
          y: result.getPose(this.referenceSpace!)!.transform.position.y,
          z: result.getPose(this.referenceSpace!)!.transform.position.z,
        },
        orientation: {
          x: result.getPose(this.referenceSpace!)!.transform.orientation.x,
          y: result.getPose(this.referenceSpace!)!.transform.orientation.y,
          z: result.getPose(this.referenceSpace!)!.transform.orientation.z,
          w: result.getPose(this.referenceSpace!)!.transform.orientation.w,
        },
      },
      distance: Math.sqrt(
        Math.pow(result.getPose(this.referenceSpace!)!.transform.position.x, 2) +
        Math.pow(result.getPose(this.referenceSpace!)!.transform.position.y, 2) +
        Math.pow(result.getPose(this.referenceSpace!)!.transform.position.z, 2)
      ),
      confidence: 0.8, // Mock confidence
      type: 'plane',
    }))

    if (results.length > 0) {
      this.notifyHitTestHandlers(results)
    }
  }

  // ===== HAND TRACKING =====

  private handleHandTracking(frame: XRFrame): void {
    if (!this.capabilities?.hasHandTracking) return

    try {
      // This would be implemented with actual hand tracking API
      // For now, we'll simulate hand tracking data
      const hands = {
        left: this.getSimulatedHandPose('left'),
        right: this.getSimulatedHandPose('right'),
      }

      this.notifyHandTrackingHandlers(hands)
    } catch (error) {
      console.warn('Hand tracking error:', error)
    }
  }

  private getSimulatedHandPose(hand: 'left' | 'right'): XRHandPose {
    // Simulate hand pose data
    const joints: { [jointName: string]: XRPose } = {}
    const jointNames = ['wrist', 'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
    
    jointNames.forEach((jointName, index) => {
      joints[jointName] = {
        position: {
          x: (hand === 'left' ? -0.2 : 0.2) + (index * 0.02),
          y: 1.0 + (index * 0.01),
          z: -0.5,
        },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      }
    })

    return {
      joints,
      confidence: 0.9,
      isTracked: true,
    }
  }

  // ===== ENVIRONMENT DATA =====

  private updateEnvironmentData(frame: XRFrame): void {
    // Update light estimation
    if (this.capabilities?.hasLightEstimation) {
      this.environmentData.lightEstimate = {
        intensity: 0.8,
        direction: { x: 0.5, y: -0.8, z: 0.3 },
        color: { r: 1.0, g: 0.95, b: 0.9 },
      }
    }

    // Update plane detection
    if (this.capabilities?.hasPlaneDetection) {
      // Simulate detected planes
      this.environmentData.planes = [
        {
          id: 'floor-plane',
          pose: {
            position: { x: 0, y: 0, z: 0 },
            orientation: { x: 0, y: 0, z: 0, w: 1 },
          },
          polygon: [
            { x: -2, z: -2 },
            { x: 2, z: -2 },
            { x: 2, z: 2 },
            { x: -2, z: 2 },
          ],
          orientation: 'horizontal',
        },
      ]
    }
  }

  // ===== SPATIAL ANCHORS =====

  public async createAnchor(pose: XRPose): Promise<string | null> {
    if (!this.session || !this.capabilities?.hasAnchors) {
      return null
    }

    try {
      // This would create an actual anchor
      const anchorId = `anchor-${Date.now()}`
      
      this.environmentData.anchors.push({
        id: anchorId,
        pose,
        persistent: true,
      })

      return anchorId
    } catch (error) {
      console.error('Failed to create anchor:', error)
      return null
    }
  }

  public removeAnchor(anchorId: string): boolean {
    const index = this.environmentData.anchors.findIndex(anchor => anchor.id === anchorId)
    if (index > -1) {
      this.environmentData.anchors.splice(index, 1)
      return true
    }
    return false
  }

  // ===== EVENT HANDLERS =====

  public onSessionStart(handler: (session: XRSession) => void): () => void {
    this.sessionStartHandlers.push(handler)
    return () => {
      const index = this.sessionStartHandlers.indexOf(handler)
      if (index > -1) {
        this.sessionStartHandlers.splice(index, 1)
      }
    }
  }

  public onSessionEnd(handler: () => void): () => void {
    this.sessionEndHandlers.push(handler)
    return () => {
      const index = this.sessionEndHandlers.indexOf(handler)
      if (index > -1) {
        this.sessionEndHandlers.splice(index, 1)
      }
    }
  }

  public onFrame(handler: (frame: XRFrame, session: XRSession) => void): () => void {
    this.frameHandlers.push(handler)
    return () => {
      const index = this.frameHandlers.indexOf(handler)
      if (index > -1) {
        this.frameHandlers.splice(index, 1)
      }
    }
  }

  public onHitTest(handler: (results: XRHitTestResult[]) => void): () => void {
    this.hitTestHandlers.push(handler)
    return () => {
      const index = this.hitTestHandlers.indexOf(handler)
      if (index > -1) {
        this.hitTestHandlers.splice(index, 1)
      }
    }
  }

  public onHandTracking(handler: (hands: { left?: XRHandPose; right?: XRHandPose }) => void): () => void {
    this.handTrackingHandlers.push(handler)
    return () => {
      const index = this.handTrackingHandlers.indexOf(handler)
      if (index > -1) {
        this.handTrackingHandlers.splice(index, 1)
      }
    }
  }

  // ===== NOTIFICATION METHODS =====

  private notifySessionStart(session: XRSession): void {
    this.sessionStartHandlers.forEach(handler => handler(session))
  }

  private notifySessionEnd(): void {
    this.sessionEndHandlers.forEach(handler => handler())
  }

  private notifyFrameHandlers(frame: XRFrame, session: XRSession): void {
    this.frameHandlers.forEach(handler => handler(frame, session))
  }

  private notifyHitTestHandlers(results: XRHitTestResult[]): void {
    this.hitTestHandlers.forEach(handler => handler(results))
  }

  private notifyHandTrackingHandlers(hands: { left?: XRHandPose; right?: XRHandPose }): void {
    this.handTrackingHandlers.forEach(handler => handler(hands))
  }

  // ===== GETTERS =====

  public getCapabilities(): WebXRCapabilities | null {
    return this.capabilities
  }

  public isSessionActive(): boolean {
    return this.isSessionActive
  }

  public getEnvironmentData(): XREnvironmentData {
    return this.environmentData
  }

  public getCurrentSession(): XRSession | null {
    return this.session
  }
}

export default WebXRService
