/**
 * MasterX Animation System
 * 
 * Enterprise-grade animation configuration using Framer Motion
 * Designed for 60fps performance and delightful user experiences
 */

import { Variants, Transition } from 'framer-motion'

// ===== ANIMATION CONSTANTS =====

export const ANIMATION_DURATION = {
  fast: 0.15,
  normal: 0.3,
  slow: 0.5,
  slower: 0.8
} as const

export const EASING = {
  easeOut: [0.0, 0.0, 0.2, 1],
  easeIn: [0.4, 0.0, 1, 1],
  easeInOut: [0.4, 0.0, 0.2, 1],
  spring: { type: "spring", stiffness: 300, damping: 30 },
  bouncy: { type: "spring", stiffness: 400, damping: 25 }
} as const

// ===== MESSAGE ANIMATIONS =====

export const messageAnimations: Variants = {
  hidden: {
    opacity: 0,
    y: 20,
    scale: 0.95
  },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30,
      duration: ANIMATION_DURATION.normal
    }
  },
  exit: {
    opacity: 0,
    y: -10,
    scale: 0.95,
    transition: {
      duration: ANIMATION_DURATION.fast,
      ease: EASING.easeIn
    }
  }
}

export const messageStreamingAnimation: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      duration: ANIMATION_DURATION.fast,
      ease: EASING.easeOut
    }
  }
}

export const messageListAnimation: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1
    }
  }
}

// ===== BUTTON ANIMATIONS =====

export const buttonAnimations: Variants = {
  idle: {
    scale: 1,
    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
  },
  hover: {
    scale: 1.02,
    boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 25
    }
  },
  tap: {
    scale: 0.98,
    transition: {
      duration: ANIMATION_DURATION.fast,
      ease: EASING.easeInOut
    }
  }
}

export const iconButtonAnimations: Variants = {
  idle: { scale: 1, rotate: 0 },
  hover: {
    scale: 1.1,
    rotate: 5,
    transition: EASING.spring
  },
  tap: {
    scale: 0.95,
    rotate: -5,
    transition: {
      duration: ANIMATION_DURATION.fast
    }
  }
}

// ===== LOADING ANIMATIONS =====

export const loadingAnimations: Variants = {
  pulse: {
    scale: [1, 1.05, 1],
    opacity: [0.5, 1, 0.5],
    transition: {
      duration: 1.5,
      repeat: Infinity,
      ease: EASING.easeInOut
    }
  },
  spin: {
    rotate: 360,
    transition: {
      duration: 1,
      repeat: Infinity,
      ease: "linear"
    }
  },
  dots: {
    scale: [1, 1.2, 1],
    transition: {
      duration: 0.6,
      repeat: Infinity,
      ease: EASING.easeInOut
    }
  }
}

export const typingIndicatorAnimation: Variants = {
  typing: {
    scale: [1, 1.1, 1],
    opacity: [0.5, 1, 0.5],
    transition: {
      duration: 1.2,
      repeat: Infinity,
      ease: EASING.easeInOut,
      staggerChildren: 0.2
    }
  }
}

// ===== INPUT ANIMATIONS =====

export const inputAnimations: Variants = {
  idle: {
    borderColor: "rgba(203, 213, 225, 1)", // slate-300
    boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
  },
  focus: {
    borderColor: "rgba(99, 102, 241, 1)", // indigo-600
    boxShadow: "0 0 0 3px rgba(99, 102, 241, 0.1)",
    transition: {
      duration: ANIMATION_DURATION.fast,
      ease: EASING.easeOut
    }
  },
  error: {
    borderColor: "rgba(239, 68, 68, 1)", // red-500
    boxShadow: "0 0 0 3px rgba(239, 68, 68, 0.1)",
    x: [-2, 2, -2, 2, 0],
    transition: {
      duration: ANIMATION_DURATION.normal,
      ease: EASING.easeOut
    }
  }
}

// ===== LAYOUT ANIMATIONS =====

export const layoutAnimations: Variants = {
  hidden: {
    opacity: 0,
    x: -20
  },
  visible: {
    opacity: 1,
    x: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30
    }
  },
  exit: {
    opacity: 0,
    x: 20,
    transition: {
      duration: ANIMATION_DURATION.fast,
      ease: EASING.easeIn
    }
  }
}

export const slideInAnimation: Variants = {
  hidden: { x: "100%" },
  visible: {
    x: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 30
    }
  },
  exit: {
    x: "100%",
    transition: {
      duration: ANIMATION_DURATION.normal,
      ease: EASING.easeInOut
    }
  }
}

export const fadeInAnimation: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      duration: ANIMATION_DURATION.normal,
      ease: EASING.easeOut
    }
  },
  exit: {
    opacity: 0,
    transition: {
      duration: ANIMATION_DURATION.fast,
      ease: EASING.easeIn
    }
  }
}

// ===== GESTURE ANIMATIONS =====

export const swipeAnimations: Variants = {
  swipeLeft: {
    x: -100,
    opacity: 0,
    transition: {
      duration: ANIMATION_DURATION.normal,
      ease: EASING.easeInOut
    }
  },
  swipeRight: {
    x: 100,
    opacity: 0,
    transition: {
      duration: ANIMATION_DURATION.normal,
      ease: EASING.easeInOut
    }
  }
}

// ===== SCROLL ANIMATIONS =====

export const scrollAnimations: Variants = {
  hidden: {
    opacity: 0,
    y: 50
  },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: ANIMATION_DURATION.normal,
      ease: EASING.easeOut
    }
  }
}

// ===== UTILITY FUNCTIONS =====

export const createStaggeredAnimation = (
  staggerDelay: number = 0.1,
  childDelay: number = 0
): Variants => ({
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: staggerDelay,
      delayChildren: childDelay
    }
  }
})

export const createBounceAnimation = (
  scale: number = 1.1,
  duration: number = ANIMATION_DURATION.normal
): Variants => ({
  idle: { scale: 1 },
  bounce: {
    scale: [1, scale, 1],
    transition: {
      duration,
      ease: EASING.easeInOut
    }
  }
})

export const createShakeAnimation = (
  intensity: number = 5,
  duration: number = ANIMATION_DURATION.normal
): Variants => ({
  idle: { x: 0 },
  shake: {
    x: [-intensity, intensity, -intensity, intensity, 0],
    transition: {
      duration,
      ease: EASING.easeInOut
    }
  }
})

// ===== HAPTIC FEEDBACK UTILITIES =====

export const triggerHapticFeedback = (type: 'light' | 'medium' | 'heavy' = 'light') => {
  if ('vibrate' in navigator) {
    const patterns = {
      light: [10],
      medium: [20],
      heavy: [30]
    }
    navigator.vibrate(patterns[type])
  }
}

// ===== ANIMATION PRESETS =====

export const animationPresets = {
  // Quick animations for immediate feedback
  quick: {
    duration: ANIMATION_DURATION.fast,
    ease: EASING.easeOut
  },
  
  // Smooth animations for general use
  smooth: {
    type: "spring" as const,
    stiffness: 300,
    damping: 30
  },
  
  // Bouncy animations for playful interactions
  bouncy: {
    type: "spring" as const,
    stiffness: 400,
    damping: 25
  },
  
  // Gentle animations for subtle effects
  gentle: {
    duration: ANIMATION_DURATION.slow,
    ease: EASING.easeInOut
  }
} as const

export type AnimationPreset = keyof typeof animationPresets
