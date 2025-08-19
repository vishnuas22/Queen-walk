'use client'

import React, { useEffect, useState } from 'react'
import { useAccessibility } from '../lib/accessibility'

interface AccessibilityProviderProps {
  children: React.ReactNode
}

// ===== SKIP LINKS COMPONENT =====

const SkipLinks: React.FC = () => {
  return (
    <div className="skip-links">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <a href="#navigation" className="skip-link">
        Skip to navigation
      </a>
      <a href="#chat-input" className="skip-link">
        Skip to chat input
      </a>
    </div>
  )
}

// ===== ACCESSIBILITY SETTINGS PANEL =====

const AccessibilitySettings: React.FC = () => {
  const { settings, updateSetting, announce } = useAccessibility()
  const [isOpen, setIsOpen] = useState(false)

  const handleSettingChange = (setting: keyof typeof settings, value: boolean) => {
    updateSetting(setting, value)
    announce(`${setting} ${value ? 'enabled' : 'disabled'}`, 'polite')
  }

  return (
    <div className="accessibility-settings">
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-controls="accessibility-panel"
        className="fixed bottom-4 left-4 z-50 bg-slate-800 text-white p-3 rounded-full shadow-lg hover:bg-slate-700 transition-colors"
        aria-label="Accessibility settings"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </button>

      {isOpen && (
        <div
          id="accessibility-panel"
          className="fixed bottom-20 left-4 z-50 bg-white border border-slate-200 rounded-lg shadow-xl p-4 w-80"
          role="dialog"
          aria-labelledby="accessibility-title"
        >
          <h3 id="accessibility-title" className="text-lg font-semibold text-slate-900 mb-4">
            Accessibility Settings
          </h3>

          <div className="space-y-4">
            {/* Reduced Motion */}
            <div className="flex items-center justify-between">
              <label htmlFor="reduced-motion" className="text-sm font-medium text-slate-700">
                Reduce Motion
              </label>
              <input
                id="reduced-motion"
                type="checkbox"
                checked={settings.reducedMotion}
                onChange={(e) => handleSettingChange('reducedMotion', e.target.checked)}
                className="w-4 h-4 text-indigo-600 border-slate-300 rounded focus:ring-indigo-500"
                aria-describedby="reduced-motion-desc"
              />
            </div>
            <p id="reduced-motion-desc" className="text-xs text-slate-500">
              Reduces animations and transitions for better accessibility
            </p>

            {/* High Contrast */}
            <div className="flex items-center justify-between">
              <label htmlFor="high-contrast" className="text-sm font-medium text-slate-700">
                High Contrast
              </label>
              <input
                id="high-contrast"
                type="checkbox"
                checked={settings.highContrast}
                onChange={(e) => handleSettingChange('highContrast', e.target.checked)}
                className="w-4 h-4 text-indigo-600 border-slate-300 rounded focus:ring-indigo-500"
                aria-describedby="high-contrast-desc"
              />
            </div>
            <p id="high-contrast-desc" className="text-xs text-slate-500">
              Increases contrast for better visibility
            </p>

            {/* Large Text */}
            <div className="flex items-center justify-between">
              <label htmlFor="large-text" className="text-sm font-medium text-slate-700">
                Large Text
              </label>
              <input
                id="large-text"
                type="checkbox"
                checked={settings.largeText}
                onChange={(e) => handleSettingChange('largeText', e.target.checked)}
                className="w-4 h-4 text-indigo-600 border-slate-300 rounded focus:ring-indigo-500"
                aria-describedby="large-text-desc"
              />
            </div>
            <p id="large-text-desc" className="text-xs text-slate-500">
              Increases text size for better readability
            </p>

            {/* Screen Reader Mode */}
            <div className="flex items-center justify-between">
              <label htmlFor="screen-reader" className="text-sm font-medium text-slate-700">
                Screen Reader Mode
              </label>
              <input
                id="screen-reader"
                type="checkbox"
                checked={settings.screenReaderMode}
                onChange={(e) => handleSettingChange('screenReaderMode', e.target.checked)}
                className="w-4 h-4 text-indigo-600 border-slate-300 rounded focus:ring-indigo-500"
                aria-describedby="screen-reader-desc"
              />
            </div>
            <p id="screen-reader-desc" className="text-xs text-slate-500">
              Optimizes interface for screen reader users
            </p>
          </div>

          <div className="mt-6 flex justify-end space-x-2">
            <button
              onClick={() => setIsOpen(false)}
              className="px-4 py-2 text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ===== KEYBOARD SHORTCUTS HELP =====

const KeyboardShortcuts: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false)

  const shortcuts = [
    { key: 'Tab', description: 'Navigate between interactive elements' },
    { key: 'Enter', description: 'Activate buttons and links' },
    { key: 'Space', description: 'Activate buttons and checkboxes' },
    { key: 'Escape', description: 'Close dialogs and menus' },
    { key: 'Arrow Keys', description: 'Navigate within menus and lists' },
    { key: 'Ctrl + Enter', description: 'Send message in chat' },
    { key: 'Alt + N', description: 'Start new conversation' },
    { key: 'Alt + S', description: 'Focus search/input field' },
    { key: '/', description: 'Open command palette' },
    { key: '?', description: 'Show keyboard shortcuts' },
  ]

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '?' && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault()
        setIsOpen(true)
      }
      if (e.key === 'Escape') {
        setIsOpen(false)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
      role="dialog"
      aria-labelledby="shortcuts-title"
      aria-modal="true"
    >
      <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
        <h2 id="shortcuts-title" className="text-xl font-semibold text-slate-900 mb-4">
          Keyboard Shortcuts
        </h2>
        
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {shortcuts.map((shortcut, index) => (
            <div key={index} className="flex items-center justify-between">
              <span className="text-sm text-slate-600">{shortcut.description}</span>
              <kbd className="px-2 py-1 bg-slate-100 border border-slate-300 rounded text-xs font-mono">
                {shortcut.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={() => setIsOpen(false)}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

// ===== MAIN ACCESSIBILITY PROVIDER =====

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({ children }) => {
  const { announce, isScreenReaderActive } = useAccessibility()

  useEffect(() => {
    // Announce page load for screen readers
    if (isScreenReaderActive()) {
      announce('MasterX application loaded. Use Tab to navigate or press question mark for keyboard shortcuts.', 'polite')
    }
  }, [announce, isScreenReaderActive])

  return (
    <>
      <SkipLinks />
      {children}
      <AccessibilitySettings />
      <KeyboardShortcuts />
    </>
  )
}

export default AccessibilityProvider
