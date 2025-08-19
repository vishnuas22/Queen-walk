// Advanced Speech API integration for voice navigation and accessibility

import React from 'react'

// ===== TYPES =====

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
  resultIndex: number
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string
  message: string
}

interface VoiceCommand {
  command: string
  action: () => void
  description: string
  category: 'navigation' | 'chat' | 'accessibility' | 'system'
}

interface SpeechSettings {
  language: string
  continuous: boolean
  interimResults: boolean
  maxAlternatives: number
  voiceSpeed: number
  voicePitch: number
  voiceVolume: number
  selectedVoice?: SpeechSynthesisVoice | null
  noiseReduction: boolean
  echoCancellation: boolean
  autoGainControl: boolean
  visualFeedback: boolean
}

// ===== SPEECH RECOGNITION =====

export class VoiceSpeechRecognition {
  private recognition: SpeechRecognition | null = null
  private isListening = false
  private commands: Map<string, VoiceCommand> = new Map()
  private settings: SpeechSettings
  private onResultCallback?: (transcript: string, confidence: number) => void
  private onCommandCallback?: (command: VoiceCommand) => void
  private onErrorCallback?: (error: string) => void
  private onAudioLevelCallback?: (level: number) => void

  // Audio visualization
  private audioContext: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private microphone: MediaStreamAudioSourceNode | null = null
  private dataArray: Uint8Array | null = null
  private animationFrame: number | null = null

  constructor(settings: Partial<SpeechSettings> = {}) {
    this.settings = {
      language: 'en-US',
      continuous: true,
      interimResults: true,
      maxAlternatives: 3,
      voiceSpeed: 1,
      voicePitch: 1,
      voiceVolume: 1,
      selectedVoice: null,
      noiseReduction: true,
      echoCancellation: true,
      autoGainControl: true,
      visualFeedback: true,
      ...settings,
    }

    this.initializeRecognition()
    this.registerDefaultCommands()
    this.initializeAudioVisualization()
  }

  private initializeRecognition() {
    // Check if we're in browser environment
    if (typeof window === 'undefined') {
      return
    }

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.warn('Speech Recognition not supported in this browser')
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    this.recognition = new SpeechRecognition()

    this.recognition.continuous = this.settings.continuous
    this.recognition.interimResults = this.settings.interimResults
    this.recognition.lang = this.settings.language
    this.recognition.maxAlternatives = this.settings.maxAlternatives

    this.recognition.onstart = () => {
      console.log('🎤 Voice recognition started')
      this.isListening = true
    }

    this.recognition.onend = () => {
      console.log('🎤 Voice recognition ended')
      this.isListening = false
    }

    this.recognition.onresult = (event: SpeechRecognitionEvent) => {
      const result = event.results[event.resultIndex]
      const transcript = result[0].transcript.toLowerCase().trim()
      const confidence = result[0].confidence

      console.log('🗣️ Speech recognized:', transcript, 'Confidence:', confidence)

      // Check for voice commands
      this.processVoiceCommand(transcript)

      // Call result callback
      this.onResultCallback?.(transcript, confidence)
    }

    this.recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.error('🎤 Speech recognition error:', event.error)
      this.onErrorCallback?.(event.error)
    }
  }

  private registerDefaultCommands() {
    // Navigation commands
    this.registerCommand('go to chat', () => {
      window.location.href = '/chat'
    }, 'Navigate to chat page', 'navigation')

    this.registerCommand('go home', () => {
      window.location.href = '/'
    }, 'Navigate to home page', 'navigation')

    this.registerCommand('scroll up', () => {
      window.scrollBy({ top: -300, behavior: 'smooth' })
    }, 'Scroll up on the page', 'navigation')

    this.registerCommand('scroll down', () => {
      window.scrollBy({ top: 300, behavior: 'smooth' })
    }, 'Scroll down on the page', 'navigation')

    this.registerCommand('scroll to top', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }, 'Scroll to top of page', 'navigation')

    this.registerCommand('scroll to bottom', () => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
    }, 'Scroll to bottom of page', 'navigation')

    // Chat commands
    this.registerCommand('new chat', () => {
      const newChatButton = document.querySelector('[data-testid="new-chat-button"]') as HTMLElement
      newChatButton?.click()
    }, 'Start a new chat conversation', 'chat')

    this.registerCommand('send message', () => {
      const sendButton = document.querySelector('[data-testid="send-button"]') as HTMLElement
      sendButton?.click()
    }, 'Send the current message', 'chat')

    this.registerCommand('focus input', () => {
      const input = document.querySelector('[data-testid="message-input"]') as HTMLElement
      input?.focus()
    }, 'Focus on the message input field', 'chat')

    this.registerCommand('clear input', () => {
      const input = document.querySelector('[data-testid="message-input"]') as HTMLInputElement
      if (input) {
        input.value = ''
        input.dispatchEvent(new Event('input', { bubbles: true }))
      }
    }, 'Clear the message input field', 'chat')

    // Accessibility commands
    this.registerCommand('increase text size', () => {
      document.documentElement.style.fontSize = 
        (parseFloat(getComputedStyle(document.documentElement).fontSize) + 2) + 'px'
    }, 'Increase text size for better readability', 'accessibility')

    this.registerCommand('decrease text size', () => {
      document.documentElement.style.fontSize = 
        (parseFloat(getComputedStyle(document.documentElement).fontSize) - 2) + 'px'
    }, 'Decrease text size', 'accessibility')

    this.registerCommand('reset text size', () => {
      document.documentElement.style.fontSize = '16px'
    }, 'Reset text size to default', 'accessibility')

    this.registerCommand('high contrast mode', () => {
      document.body.classList.toggle('high-contrast')
    }, 'Toggle high contrast mode', 'accessibility')

    // System commands
    this.registerCommand('stop listening', () => {
      this.stopListening()
    }, 'Stop voice recognition', 'system')

    this.registerCommand('help', () => {
      this.announceAvailableCommands()
    }, 'List available voice commands', 'system')
  }

  public registerCommand(phrase: string, action: () => void, description: string, category: VoiceCommand['category']) {
    const command: VoiceCommand = { command: phrase, action, description, category }
    this.commands.set(phrase.toLowerCase(), command)
  }

  private processVoiceCommand(transcript: string) {
    // Direct command match
    const directCommand = this.commands.get(transcript)
    if (directCommand) {
      console.log('🎯 Executing voice command:', directCommand.command)
      directCommand.action()
      this.onCommandCallback?.(directCommand)
      return
    }

    // Fuzzy matching for partial commands
    for (const [phrase, command] of this.commands) {
      if (transcript.includes(phrase) || phrase.includes(transcript)) {
        console.log('🎯 Executing fuzzy matched command:', command.command)
        command.action()
        this.onCommandCallback?.(command)
        return
      }
    }

    console.log('❓ No matching voice command found for:', transcript)
  }

  public async startListening() {
    if (!this.recognition) {
      console.error('Speech recognition not available')
      return false
    }

    if (this.isListening) {
      console.log('Already listening')
      return true
    }

    try {
      // Start audio visualization if enabled
      if (this.settings.visualFeedback) {
        await this.startAudioVisualization()
      }

      this.recognition.start()
      return true
    } catch (error) {
      console.error('Failed to start speech recognition:', error)
      return false
    }
  }

  public stopListening() {
    if (this.recognition && this.isListening) {
      this.recognition.stop()
    }

    // Stop audio visualization
    this.stopAudioVisualization()
  }

  public toggleListening() {
    if (this.isListening) {
      this.stopListening()
    } else {
      this.startListening()
    }
  }

  public announceAvailableCommands() {
    const commandsByCategory = new Map<string, VoiceCommand[]>()
    
    for (const command of this.commands.values()) {
      if (!commandsByCategory.has(command.category)) {
        commandsByCategory.set(command.category, [])
      }
      commandsByCategory.get(command.category)!.push(command)
    }

    let announcement = 'Available voice commands: '
    for (const [category, commands] of commandsByCategory) {
      announcement += `${category}: `
      announcement += commands.map(cmd => cmd.command).join(', ')
      announcement += '. '
    }

    this.speak(announcement)
  }

  public speak(text: string, options: Partial<SpeechSynthesisUtterance> = {}) {
    if (!('speechSynthesis' in window)) {
      console.warn('Speech synthesis not supported')
      return
    }

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = this.settings.voiceSpeed
    utterance.pitch = this.settings.voicePitch
    utterance.volume = this.settings.voiceVolume
    
    Object.assign(utterance, options)

    speechSynthesis.speak(utterance)
  }

  public onResult(callback: (transcript: string, confidence: number) => void) {
    this.onResultCallback = callback
  }

  public onCommand(callback: (command: VoiceCommand) => void) {
    this.onCommandCallback = callback
  }

  public onError(callback: (error: string) => void) {
    this.onErrorCallback = callback
  }

  public updateSettings(newSettings: Partial<SpeechSettings>) {
    this.settings = { ...this.settings, ...newSettings }
    
    if (this.recognition) {
      this.recognition.lang = this.settings.language
      this.recognition.continuous = this.settings.continuous
      this.recognition.interimResults = this.settings.interimResults
      this.recognition.maxAlternatives = this.settings.maxAlternatives
    }
  }

  public getAvailableVoices() {
    if (!('speechSynthesis' in window)) return []
    return speechSynthesis.getVoices()
  }

  public isSupported() {
    if (typeof window === 'undefined') {
      return false
    }
    return ('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window)
  }

  public getListeningState() {
    return this.isListening
  }

  public getCommands() {
    return Array.from(this.commands.values())
  }

  public setOnAudioLevel(callback: (level: number) => void) {
    this.onAudioLevelCallback = callback
  }

  // ===== AUDIO VISUALIZATION =====

  private async initializeAudioVisualization() {
    if (!this.settings.visualFeedback) return

    // Check if we're in browser environment
    if (typeof window === 'undefined') {
      return
    }

    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      this.analyser = this.audioContext.createAnalyser()
      this.analyser.fftSize = 256
      this.dataArray = new Uint8Array(this.analyser.frequencyBinCount)
    } catch (error) {
      console.warn('Audio visualization not supported:', error)
    }
  }

  private async startAudioVisualization() {
    if (!this.audioContext || !this.analyser) return

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: this.settings.echoCancellation,
          noiseSuppression: this.settings.noiseReduction,
          autoGainControl: this.settings.autoGainControl,
        }
      })

      this.microphone = this.audioContext.createMediaStreamSource(stream)
      this.microphone.connect(this.analyser)

      this.startAudioLevelMonitoring()
    } catch (error) {
      console.warn('Failed to start audio visualization:', error)
    }
  }

  private startAudioLevelMonitoring() {
    if (!this.analyser || !this.dataArray) return

    const updateAudioLevel = () => {
      this.analyser!.getByteFrequencyData(this.dataArray!)

      // Calculate average audio level
      const average = this.dataArray!.reduce((sum, value) => sum + value, 0) / this.dataArray!.length
      const normalizedLevel = average / 255

      this.onAudioLevelCallback?.(normalizedLevel)

      if (this.isListening) {
        this.animationFrame = requestAnimationFrame(updateAudioLevel)
      }
    }

    updateAudioLevel()
  }

  private stopAudioVisualization() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame)
      this.animationFrame = null
    }

    if (this.microphone) {
      this.microphone.disconnect()
      this.microphone = null
    }
  }

  public destroy() {
    this.stopListening()
    this.stopAudioVisualization()
    this.commands.clear()
    this.onResultCallback = undefined
    this.onCommandCallback = undefined
    this.onErrorCallback = undefined
    this.onAudioLevelCallback = undefined
  }
}

// ===== GLOBAL INSTANCE =====

let globalVoiceRecognition: VoiceSpeechRecognition | null = null

export const getVoiceRecognition = (): VoiceSpeechRecognition => {
  if (!globalVoiceRecognition) {
    globalVoiceRecognition = new VoiceSpeechRecognition()
  }
  return globalVoiceRecognition
}

// ===== REACT HOOK =====

export const useVoiceRecognition = () => {
  const [isListening, setIsListening] = React.useState(false)
  const [lastTranscript, setLastTranscript] = React.useState('')
  const [lastCommand, setLastCommand] = React.useState<VoiceCommand | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const voiceRecognition = React.useMemo(() => getVoiceRecognition(), [])

  React.useEffect(() => {
    voiceRecognition.onResult((transcript, confidence) => {
      setLastTranscript(transcript)
      setError(null)
    })

    voiceRecognition.onCommand((command) => {
      setLastCommand(command)
    })

    voiceRecognition.onError((error) => {
      setError(error)
      setIsListening(false)
    })

    const checkListeningState = () => {
      setIsListening(voiceRecognition.getListeningState())
    }

    const interval = setInterval(checkListeningState, 100)

    return () => {
      clearInterval(interval)
    }
  }, [voiceRecognition])

  return {
    isListening,
    lastTranscript,
    lastCommand,
    error,
    startListening: () => voiceRecognition.startListening(),
    stopListening: () => voiceRecognition.stopListening(),
    toggleListening: () => voiceRecognition.toggleListening(),
    speak: (text: string) => voiceRecognition.speak(text),
    isSupported: voiceRecognition.isSupported(),
    commands: voiceRecognition.getCommands(),
  }
}

export default VoiceSpeechRecognition
