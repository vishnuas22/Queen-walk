'use client'

import React, { createContext, useContext, useState, useCallback } from 'react'
import { AnimatePresence, MotionConfig, motion } from 'framer-motion'
import { animationPresets, triggerHapticFeedback } from './animations'

// ===== ANIMATION CONTEXT =====

interface AnimationContextType {
  // Animation preferences
  reducedMotion: boolean
  animationSpeed: 'slow' | 'normal' | 'fast'
  hapticFeedback: boolean
  
  // Animation controls
  setReducedMotion: (enabled: boolean) => void
  setAnimationSpeed: (speed: 'slow' | 'normal' | 'fast') => void
  setHapticFeedback: (enabled: boolean) => void
  
  // Utility functions
  triggerHaptic: (type?: 'light' | 'medium' | 'heavy') => void
  getAnimationConfig: () => any
}

const AnimationContext = createContext<AnimationContextType | undefined>(undefined)

// ===== ANIMATION PROVIDER =====

interface AnimationProviderProps {
  children: React.ReactNode
}

export function AnimationProvider({ children }: AnimationProviderProps) {
  // Animation preferences state
  const [reducedMotion, setReducedMotion] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.matchMedia('(prefers-reduced-motion: reduce)').matches
    }
    return false
  })
  
  const [animationSpeed, setAnimationSpeed] = useState<'slow' | 'normal' | 'fast'>('normal')
  const [hapticFeedback, setHapticFeedback] = useState(true)

  // Haptic feedback handler
  const triggerHaptic = useCallback((type: 'light' | 'medium' | 'heavy' = 'light') => {
    if (hapticFeedback && !reducedMotion) {
      triggerHapticFeedback(type)
    }
  }, [hapticFeedback, reducedMotion])

  // Get animation configuration based on preferences
  const getAnimationConfig = useCallback(() => {
    if (reducedMotion) {
      return {
        transition: { duration: 0 }
      }
    }

    const speedMultipliers = {
      slow: 1.5,
      normal: 1,
      fast: 0.7
    }

    const multiplier = speedMultipliers[animationSpeed]

    return {
      transition: {
        ...animationPresets.smooth,
        // Spring animations don't use duration, but we can adjust stiffness/damping
        stiffness: (animationPresets.smooth.stiffness || 300) * multiplier,
        damping: animationPresets.smooth.damping || 30
      }
    }
  }, [reducedMotion, animationSpeed])

  // Context value
  const contextValue: AnimationContextType = {
    reducedMotion,
    animationSpeed,
    hapticFeedback,
    setReducedMotion,
    setAnimationSpeed,
    setHapticFeedback,
    triggerHaptic,
    getAnimationConfig
  }

  return (
    <AnimationContext.Provider value={contextValue}>
      <MotionConfig
        reducedMotion={reducedMotion ? "always" : "never"}
        transition={getAnimationConfig().transition}
      >
        <AnimatePresence mode="wait">
          {children}
        </AnimatePresence>
      </MotionConfig>
    </AnimationContext.Provider>
  )
}

// ===== ANIMATION HOOK =====

export function useAnimation() {
  const context = useContext(AnimationContext)
  
  if (context === undefined) {
    throw new Error('useAnimation must be used within an AnimationProvider')
  }
  
  return context
}

// ===== ANIMATION COMPONENTS =====

interface AnimatedContainerProps {
  children: React.ReactNode
  className?: string
  animation?: 'fade' | 'slide' | 'scale' | 'bounce'
  delay?: number
  duration?: number
}

export function AnimatedContainer({ 
  children, 
  className = '',
  animation = 'fade',
  delay = 0,
  duration
}: AnimatedContainerProps) {
  const { getAnimationConfig, reducedMotion } = useAnimation()
  
  if (reducedMotion) {
    return <div className={className}>{children}</div>
  }

  const animations = {
    fade: {
      initial: { opacity: 0 },
      animate: { opacity: 1 },
      exit: { opacity: 0 }
    },
    slide: {
      initial: { opacity: 0, x: -20 },
      animate: { opacity: 1, x: 0 },
      exit: { opacity: 0, x: 20 }
    },
    scale: {
      initial: { opacity: 0, scale: 0.95 },
      animate: { opacity: 1, scale: 1 },
      exit: { opacity: 0, scale: 0.95 }
    },
    bounce: {
      initial: { opacity: 0, y: 20 },
      animate: { opacity: 1, y: 0 },
      exit: { opacity: 0, y: -20 }
    }
  }

  const config = getAnimationConfig()
  const animationConfig = {
    ...animations[animation],
    transition: {
      ...config.transition,
      delay,
      duration: duration || config.transition.duration
    }
  }

  return (
    <motion.div
      className={className}
      {...animationConfig}
    >
      {children}
    </motion.div>
  )
}

// ===== ANIMATED BUTTON COMPONENT =====

interface AnimatedButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  hapticType?: 'light' | 'medium' | 'heavy'
}

export function AnimatedButton({ 
  children, 
  className = '',
  variant = 'primary',
  size = 'md',
  hapticType = 'light',
  onClick,
  ...props 
}: AnimatedButtonProps) {
  const { triggerHaptic, reducedMotion } = useAnimation()

  const handleClick = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
    triggerHaptic(hapticType)
    onClick?.(e)
  }, [onClick, triggerHaptic, hapticType])

  const baseClasses = "inline-flex items-center justify-center font-medium rounded-xl transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2"
  
  const variantClasses = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-700 focus:ring-indigo-500",
    secondary: "bg-white text-gray-900 border border-gray-300 hover:bg-gray-50 focus:ring-indigo-500",
    ghost: "text-gray-600 hover:text-gray-900 hover:bg-gray-100 focus:ring-gray-500"
  }
  
  const sizeClasses = {
    sm: "px-3 py-2 text-sm",
    md: "px-4 py-2 text-base",
    lg: "px-6 py-3 text-lg"
  }

  const buttonClasses = `${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`

  if (reducedMotion) {
    return (
      <button
        className={buttonClasses}
        onClick={handleClick}
        {...props}
      >
        {children}
      </button>
    )
  }

  return (
    <button
      className={buttonClasses}
      onClick={handleClick}
      {...props}
    >
      {children}
    </button>
  )
}

// ===== ANIMATED LIST COMPONENT =====

interface AnimatedListProps {
  children: React.ReactNode
  className?: string
  staggerDelay?: number
}

export function AnimatedList({ 
  children, 
  className = '',
  staggerDelay = 0.1 
}: AnimatedListProps) {
  const { reducedMotion } = useAnimation()

  if (reducedMotion) {
    return <div className={className}>{children}</div>
  }

  return (
    <motion.div
      className={className}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: { opacity: 0 },
        visible: {
          opacity: 1,
          transition: {
            staggerChildren: staggerDelay,
            delayChildren: 0.1
          }
        }
      }}
    >
      {children}
    </motion.div>
  )
}

// ===== ANIMATED LIST ITEM COMPONENT =====

interface AnimatedListItemProps {
  children: React.ReactNode
  className?: string
}

export function AnimatedListItem({ children, className = '' }: AnimatedListItemProps) {
  const { reducedMotion } = useAnimation()

  if (reducedMotion) {
    return <div className={className}>{children}</div>
  }

  return (
    <motion.div
      className={className}
      variants={{
        hidden: { opacity: 0, y: 20 },
        visible: {
          opacity: 1,
          y: 0,
          transition: {
            type: "spring",
            stiffness: 300,
            damping: 30
          }
        }
      }}
    >
      {children}
    </motion.div>
  )
}
