// Advanced accessibility utilities and screen reader support

import React from 'react'

// ===== TYPES =====

interface AriaLiveRegionProps {
  message: string
  priority: 'polite' | 'assertive'
  id?: string
}

interface AccessibilitySettings {
  reducedMotion: boolean
  highContrast: boolean
  largeText: boolean
  screenReaderMode: boolean
  keyboardNavigation: boolean
}

interface FocusManagementOptions {
  preventScroll?: boolean
  focusVisible?: boolean
  restoreFocus?: boolean
}

// ===== ARIA LIVE REGIONS =====

export class AriaLiveRegionManager {
  private static instance: AriaLiveRegionManager
  private politeRegion: HTMLElement | null = null
  private assertiveRegion: HTMLElement | null = null

  private constructor() {
    this.initializeRegions()
  }

  public static getInstance(): AriaLiveRegionManager {
    if (!AriaLiveRegionManager.instance) {
      AriaLiveRegionManager.instance = new AriaLiveRegionManager()
    }
    return AriaLiveRegionManager.instance
  }

  private initializeRegions() {
    if (typeof window === 'undefined') return

    // Create polite live region
    this.politeRegion = document.createElement('div')
    this.politeRegion.setAttribute('aria-live', 'polite')
    this.politeRegion.setAttribute('aria-atomic', 'true')
    this.politeRegion.setAttribute('class', 'sr-only')
    this.politeRegion.id = 'aria-live-polite'
    document.body.appendChild(this.politeRegion)

    // Create assertive live region
    this.assertiveRegion = document.createElement('div')
    this.assertiveRegion.setAttribute('aria-live', 'assertive')
    this.assertiveRegion.setAttribute('aria-atomic', 'true')
    this.assertiveRegion.setAttribute('class', 'sr-only')
    this.assertiveRegion.id = 'aria-live-assertive'
    document.body.appendChild(this.assertiveRegion)
  }

  public announce(message: string, priority: 'polite' | 'assertive' = 'polite') {
    const region = priority === 'assertive' ? this.assertiveRegion : this.politeRegion
    if (region) {
      // Clear and set new message
      region.textContent = ''
      setTimeout(() => {
        region.textContent = message
      }, 100)
    }
  }

  public announceWithDelay(message: string, delay: number = 1000, priority: 'polite' | 'assertive' = 'polite') {
    setTimeout(() => {
      this.announce(message, priority)
    }, delay)
  }
}

// ===== FOCUS MANAGEMENT =====

export class FocusManager {
  private focusHistory: HTMLElement[] = []
  private trapStack: HTMLElement[] = []

  public saveFocus() {
    const activeElement = document.activeElement as HTMLElement
    if (activeElement && activeElement !== document.body) {
      this.focusHistory.push(activeElement)
    }
  }

  public restoreFocus() {
    const lastFocused = this.focusHistory.pop()
    if (lastFocused && document.contains(lastFocused)) {
      lastFocused.focus()
      return true
    }
    return false
  }

  public focusElement(element: HTMLElement | null, options: FocusManagementOptions = {}) {
    if (!element) return false

    if (options.restoreFocus) {
      this.saveFocus()
    }

    try {
      element.focus({ preventScroll: options.preventScroll })
      
      if (options.focusVisible) {
        element.setAttribute('data-focus-visible', 'true')
      }
      
      return true
    } catch (error) {
      console.warn('Failed to focus element:', error)
      return false
    }
  }

  public trapFocus(container: HTMLElement) {
    this.trapStack.push(container)
    this.setupFocusTrap(container)
  }

  public releaseFocusTrap() {
    const container = this.trapStack.pop()
    if (container) {
      this.removeFocusTrap(container)
    }
  }

  private setupFocusTrap(container: HTMLElement) {
    const focusableElements = this.getFocusableElements(container)
    if (focusableElements.length === 0) return

    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault()
          lastElement.focus()
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault()
          firstElement.focus()
        }
      }
    }

    container.addEventListener('keydown', handleTabKey)
    container.setAttribute('data-focus-trap', 'true')

    // Focus first element
    firstElement.focus()
  }

  private removeFocusTrap(container: HTMLElement) {
    container.removeAttribute('data-focus-trap')
    // Remove event listeners (would need to store reference in real implementation)
  }

  private getFocusableElements(container: HTMLElement): HTMLElement[] {
    const focusableSelectors = [
      'button:not([disabled])',
      'input:not([disabled])',
      'textarea:not([disabled])',
      'select:not([disabled])',
      'a[href]',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]'
    ].join(', ')

    return Array.from(container.querySelectorAll(focusableSelectors)) as HTMLElement[]
  }
}

// ===== ACCESSIBILITY SETTINGS =====

export class AccessibilityManager {
  private settings: AccessibilitySettings = {
    reducedMotion: false,
    highContrast: false,
    largeText: false,
    screenReaderMode: false,
    keyboardNavigation: false,
  }

  private mediaQueries: Map<string, MediaQueryList> = new Map()

  constructor() {
    this.initializeMediaQueries()
    this.detectSystemPreferences()
    this.setupKeyboardNavigation()
  }

  private initializeMediaQueries() {
    if (typeof window === 'undefined') return

    // Reduced motion
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    this.mediaQueries.set('reducedMotion', reducedMotionQuery)
    reducedMotionQuery.addEventListener('change', (e) => {
      this.updateSetting('reducedMotion', e.matches)
    })

    // High contrast
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)')
    this.mediaQueries.set('highContrast', highContrastQuery)
    highContrastQuery.addEventListener('change', (e) => {
      this.updateSetting('highContrast', e.matches)
    })
  }

  private detectSystemPreferences() {
    // Detect reduced motion preference
    const reducedMotionQuery = this.mediaQueries.get('reducedMotion')
    if (reducedMotionQuery) {
      this.settings.reducedMotion = reducedMotionQuery.matches
    }

    // Detect high contrast preference
    const highContrastQuery = this.mediaQueries.get('highContrast')
    if (highContrastQuery) {
      this.settings.highContrast = highContrastQuery.matches
    }

    // Detect screen reader
    this.detectScreenReader()
  }

  private detectScreenReader() {
    // Return false during SSR
    if (typeof window === 'undefined') {
      return false
    }

    // Various methods to detect screen reader usage
    const indicators = [
      // Check for screen reader specific CSS
      () => window.getComputedStyle(document.body).getPropertyValue('speak') !== '',

      // Check for NVDA
      () => 'speechSynthesis' in window && navigator.userAgent.includes('NVDA'),
      
      // Check for JAWS
      () => 'speechSynthesis' in window && navigator.userAgent.includes('JAWS'),
      
      // Check for VoiceOver (Mac)
      () => navigator.userAgent.includes('Mac') && 'speechSynthesis' in window,
      
      // Check for high contrast mode (often used with screen readers)
      () => this.settings.highContrast,
    ]

    this.settings.screenReaderMode = indicators.some(check => check())
  }

  private setupKeyboardNavigation() {
    if (typeof window === 'undefined') return

    // Detect keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        this.settings.keyboardNavigation = true
        document.body.classList.add('keyboard-navigation')
      }
    })

    // Detect mouse usage
    document.addEventListener('mousedown', () => {
      this.settings.keyboardNavigation = false
      document.body.classList.remove('keyboard-navigation')
    })
  }

  public updateSetting(key: keyof AccessibilitySettings, value: boolean) {
    this.settings[key] = value
    this.applySettings()
  }

  private applySettings() {
    const { reducedMotion, highContrast, largeText, screenReaderMode } = this.settings

    // Apply reduced motion
    if (reducedMotion) {
      document.body.classList.add('reduce-motion')
    } else {
      document.body.classList.remove('reduce-motion')
    }

    // Apply high contrast
    if (highContrast) {
      document.body.classList.add('high-contrast')
    } else {
      document.body.classList.remove('high-contrast')
    }

    // Apply large text
    if (largeText) {
      document.body.classList.add('large-text')
    } else {
      document.body.classList.remove('large-text')
    }

    // Apply screen reader mode
    if (screenReaderMode) {
      document.body.classList.add('screen-reader-mode')
    } else {
      document.body.classList.remove('screen-reader-mode')
    }
  }

  public getSettings(): AccessibilitySettings {
    return { ...this.settings }
  }

  public isScreenReaderActive(): boolean {
    return this.settings.screenReaderMode
  }

  public isKeyboardNavigation(): boolean {
    return this.settings.keyboardNavigation
  }
}

// ===== REACT HOOKS =====

export const useAccessibility = () => {
  const [settings, setSettings] = React.useState<AccessibilitySettings>({
    reducedMotion: false,
    highContrast: false,
    largeText: false,
    screenReaderMode: false,
    keyboardNavigation: false,
  })

  const accessibilityManager = React.useMemo(() => new AccessibilityManager(), [])
  const ariaLiveManager = React.useMemo(() => AriaLiveRegionManager.getInstance(), [])
  const focusManager = React.useMemo(() => new FocusManager(), [])

  React.useEffect(() => {
    const updateSettings = () => {
      setSettings(accessibilityManager.getSettings())
    }

    // Initial settings
    updateSettings()

    // Listen for changes (would need event system in real implementation)
    const interval = setInterval(updateSettings, 1000)

    return () => clearInterval(interval)
  }, [accessibilityManager])

  return {
    settings,
    announce: ariaLiveManager.announce.bind(ariaLiveManager),
    announceWithDelay: ariaLiveManager.announceWithDelay.bind(ariaLiveManager),
    focusElement: focusManager.focusElement.bind(focusManager),
    saveFocus: focusManager.saveFocus.bind(focusManager),
    restoreFocus: focusManager.restoreFocus.bind(focusManager),
    trapFocus: focusManager.trapFocus.bind(focusManager),
    releaseFocusTrap: focusManager.releaseFocusTrap.bind(focusManager),
    updateSetting: accessibilityManager.updateSetting.bind(accessibilityManager),
    isScreenReaderActive: accessibilityManager.isScreenReaderActive.bind(accessibilityManager),
    isKeyboardNavigation: accessibilityManager.isKeyboardNavigation.bind(accessibilityManager),
  }
}

// ===== ARIA UTILITIES =====

export const generateAriaLabel = (base: string, context?: string, state?: string): string => {
  let label = base
  if (context) label += `, ${context}`
  if (state) label += `, ${state}`
  return label
}

export const generateAriaDescription = (elements: string[]): string => {
  return elements.filter(Boolean).join('. ')
}

// ===== GLOBAL INSTANCES =====

export const ariaLiveManager = AriaLiveRegionManager.getInstance()
export const focusManager = new FocusManager()
export const accessibilityManager = new AccessibilityManager()

export default AccessibilityManager
