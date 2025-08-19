'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Lock, 
  Mail, 
  Eye, 
  EyeOff, 
  Shield, 
  Smartphone,
  Key,
  CheckCircle,
  AlertCircle,
  Chrome,
  Github,
  Loader2
} from 'lucide-react'
import { AuthService, SecurityUtils, AuditLogger, type LoginCredentials, type MFACredentials } from '../lib/auth'
import { useUIActions } from '../store'

// ===== TYPES =====

interface AuthenticationProps {
  onSuccess: (user: any) => void
  onClose?: () => void
  mode?: 'login' | 'register' | 'mfa'
  className?: string
}

interface FormErrors {
  email?: string
  password?: string
  confirmPassword?: string
  mfaCode?: string
  general?: string
}

// ===== AUTHENTICATION COMPONENT =====

export const Authentication: React.FC<AuthenticationProps> = ({
  onSuccess,
  onClose,
  mode: initialMode = 'login',
  className = ''
}) => {
  const [mode, setMode] = useState(initialMode)
  const [isLoading, setIsLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    rememberMe: false,
    mfaCode: '',
    mfaMethod: 'totp' as 'totp' | 'sms' | 'email'
  })
  const [errors, setErrors] = useState<FormErrors>({})
  const [passwordStrength, setPasswordStrength] = useState(0)

  const { addNotification } = useUIActions()
  const authService = AuthService.getInstance()
  const auditLogger = AuditLogger.getInstance()

  // Password strength calculation
  useEffect(() => {
    if (mode === 'register' && formData.password) {
      const validation = SecurityUtils.validatePassword(formData.password)
      setPasswordStrength((5 - validation.errors.length) * 20)
    }
  }, [formData.password, mode])

  // ===== FORM HANDLERS =====

  const handleInputChange = (field: string, value: string | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    
    // Clear field-specific errors
    if (errors[field as keyof FormErrors]) {
      setErrors(prev => ({ ...prev, [field]: undefined }))
    }
  }

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {}

    // Email validation
    if (!formData.email) {
      newErrors.email = 'Email is required'
    } else if (!SecurityUtils.validateEmail(formData.email)) {
      newErrors.email = 'Please enter a valid email address'
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required'
    } else if (mode === 'register') {
      const validation = SecurityUtils.validatePassword(formData.password)
      if (!validation.isValid) {
        newErrors.password = validation.errors[0]
      }
    }

    // Confirm password validation (register mode)
    if (mode === 'register') {
      if (!formData.confirmPassword) {
        newErrors.confirmPassword = 'Please confirm your password'
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match'
      }
    }

    // MFA code validation
    if (mode === 'mfa') {
      if (!formData.mfaCode) {
        newErrors.mfaCode = 'MFA code is required'
      } else if (formData.mfaCode.length !== 6) {
        newErrors.mfaCode = 'MFA code must be 6 digits'
      }
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  // ===== AUTHENTICATION HANDLERS =====

  const handleLogin = async () => {
    if (!validateForm()) return

    setIsLoading(true)
    setErrors({})

    try {
      const credentials: LoginCredentials = {
        email: SecurityUtils.sanitizeInput(formData.email),
        password: formData.password,
        rememberMe: formData.rememberMe
      }

      const result = await authService.login(credentials)

      if (result.error) {
        setErrors({ general: result.error })
        
        // Log failed login attempt
        await auditLogger.log({
          action: 'login_failed',
          resource: 'authentication',
          metadata: { email: formData.email, reason: result.error }
        })
      } else if (result.mfaRequired) {
        setMode('mfa')
        addNotification({
          type: 'info',
          title: 'MFA Required',
          message: 'Please enter your multi-factor authentication code',
          duration: 5000,
        })
      } else if (result.user) {
        // Log successful login
        await auditLogger.log({
          userId: result.user.id,
          action: 'login_success',
          resource: 'authentication',
          metadata: { email: formData.email }
        })

        addNotification({
          type: 'success',
          title: 'Welcome Back!',
          message: `Successfully logged in as ${result.user.name}`,
          duration: 3000,
        })

        onSuccess(result.user)
      }
    } catch (error) {
      setErrors({ general: 'An unexpected error occurred. Please try again.' })
    } finally {
      setIsLoading(false)
    }
  }

  const handleMFAVerification = async () => {
    if (!validateForm()) return

    setIsLoading(true)
    setErrors({})

    try {
      const credentials: MFACredentials = {
        code: formData.mfaCode,
        method: formData.mfaMethod
      }

      const result = await authService.verifyMFA(credentials)

      if (result.error) {
        setErrors({ mfaCode: result.error })
      } else if (result.user) {
        // Log successful MFA verification
        await auditLogger.log({
          userId: result.user.id,
          action: 'mfa_verified',
          resource: 'authentication',
          metadata: { method: formData.mfaMethod }
        })

        addNotification({
          type: 'success',
          title: 'Authentication Complete',
          message: 'Multi-factor authentication verified successfully',
          duration: 3000,
        })

        onSuccess(result.user)
      }
    } catch (error) {
      setErrors({ general: 'An unexpected error occurred. Please try again.' })
    } finally {
      setIsLoading(false)
    }
  }

  const handleSSO = async (provider: string) => {
    setIsLoading(true)

    try {
      const result = await authService.loginWithSSO(provider)

      if (result.error) {
        addNotification({
          type: 'error',
          title: 'SSO Error',
          message: result.error,
          duration: 5000,
        })
      } else if (result.url) {
        // Log SSO attempt
        await auditLogger.log({
          action: 'sso_initiated',
          resource: 'authentication',
          metadata: { provider }
        })

        window.location.href = result.url
      }
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'SSO Error',
        message: 'Failed to initiate SSO login',
        duration: 5000,
      })
    } finally {
      setIsLoading(false)
    }
  }

  // ===== RENDER HELPERS =====

  const getPasswordStrengthColor = () => {
    if (passwordStrength < 40) return 'bg-red-500'
    if (passwordStrength < 80) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  const getPasswordStrengthText = () => {
    if (passwordStrength < 40) return 'Weak'
    if (passwordStrength < 80) return 'Medium'
    return 'Strong'
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className={`fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4 ${className}`}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="bg-white rounded-lg shadow-xl max-w-md w-full overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
          <div className="flex items-center space-x-2">
            <Shield className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">
              {mode === 'login' && 'Sign In to MasterX'}
              {mode === 'register' && 'Create Your Account'}
              {mode === 'mfa' && 'Multi-Factor Authentication'}
            </h2>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={mode}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              {/* Login/Register Form */}
              {(mode === 'login' || mode === 'register') && (
                <div className="space-y-4">
                  {/* Email Field */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Email Address
                    </label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <input
                        type="email"
                        value={formData.email}
                        onChange={(e) => handleInputChange('email', e.target.value)}
                        className={`w-full pl-10 pr-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 ${
                          errors.email ? 'border-red-500' : 'border-gray-300'
                        }`}
                        placeholder="Enter your email"
                        disabled={isLoading}
                      />
                    </div>
                    {errors.email && (
                      <p className="text-sm text-red-600 mt-1">{errors.email}</p>
                    )}
                  </div>

                  {/* Password Field */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Password
                    </label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <input
                        type={showPassword ? 'text' : 'password'}
                        value={formData.password}
                        onChange={(e) => handleInputChange('password', e.target.value)}
                        className={`w-full pl-10 pr-10 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 ${
                          errors.password ? 'border-red-500' : 'border-gray-300'
                        }`}
                        placeholder="Enter your password"
                        disabled={isLoading}
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                      >
                        {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </button>
                    </div>
                    {errors.password && (
                      <p className="text-sm text-red-600 mt-1">{errors.password}</p>
                    )}

                    {/* Password Strength Indicator (Register Mode) */}
                    {mode === 'register' && formData.password && (
                      <div className="mt-2">
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full transition-all duration-300 ${getPasswordStrengthColor()}`}
                              style={{ width: `${passwordStrength}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-600">{getPasswordStrengthText()}</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Confirm Password Field (Register Mode) */}
                  {mode === 'register' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Confirm Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                        <input
                          type={showPassword ? 'text' : 'password'}
                          value={formData.confirmPassword}
                          onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                          className={`w-full pl-10 pr-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 ${
                            errors.confirmPassword ? 'border-red-500' : 'border-gray-300'
                          }`}
                          placeholder="Confirm your password"
                          disabled={isLoading}
                        />
                      </div>
                      {errors.confirmPassword && (
                        <p className="text-sm text-red-600 mt-1">{errors.confirmPassword}</p>
                      )}
                    </div>
                  )}

                  {/* Remember Me (Login Mode) */}
                  {mode === 'login' && (
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="rememberMe"
                        checked={formData.rememberMe}
                        onChange={(e) => handleInputChange('rememberMe', e.target.checked)}
                        className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                        disabled={isLoading}
                      />
                      <label htmlFor="rememberMe" className="ml-2 text-sm text-gray-700">
                        Remember me for 30 days
                      </label>
                    </div>
                  )}

                  {/* General Error */}
                  {errors.general && (
                    <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <AlertCircle className="h-4 w-4 text-red-500" />
                      <p className="text-sm text-red-700">{errors.general}</p>
                    </div>
                  )}

                  {/* Submit Button */}
                  <button
                    onClick={mode === 'login' ? handleLogin : () => {}}
                    disabled={isLoading}
                    className="w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <span>{mode === 'login' ? 'Sign In' : 'Create Account'}</span>
                      </>
                    )}
                  </button>

                  {/* SSO Options */}
                  <div className="mt-6">
                    <div className="relative">
                      <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-300" />
                      </div>
                      <div className="relative flex justify-center text-sm">
                        <span className="px-2 bg-white text-gray-500">Or continue with</span>
                      </div>
                    </div>

                    <div className="mt-4 grid grid-cols-2 gap-3">
                      <button
                        onClick={() => handleSSO('google')}
                        disabled={isLoading}
                        className="flex items-center justify-center px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
                      >
                        <Chrome className="h-4 w-4 mr-2" />
                        <span className="text-sm">Google</span>
                      </button>
                      
                      <button
                        onClick={() => handleSSO('github')}
                        disabled={isLoading}
                        className="flex items-center justify-center px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
                      >
                        <Github className="h-4 w-4 mr-2" />
                        <span className="text-sm">GitHub</span>
                      </button>
                    </div>
                  </div>

                  {/* Mode Toggle */}
                  <div className="text-center mt-6">
                    <button
                      onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
                      className="text-sm text-indigo-600 hover:text-indigo-700"
                      disabled={isLoading}
                    >
                      {mode === 'login' 
                        ? "Don't have an account? Sign up" 
                        : "Already have an account? Sign in"
                      }
                    </button>
                  </div>
                </div>
              )}

              {/* MFA Form */}
              {mode === 'mfa' && (
                <div className="space-y-4">
                  <div className="text-center">
                    <Smartphone className="h-12 w-12 text-indigo-600 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      Enter Authentication Code
                    </h3>
                    <p className="text-sm text-gray-600">
                      Please enter the 6-digit code from your authenticator app
                    </p>
                  </div>

                  {/* MFA Code Field */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Authentication Code
                    </label>
                    <div className="relative">
                      <Key className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <input
                        type="text"
                        value={formData.mfaCode}
                        onChange={(e) => handleInputChange('mfaCode', e.target.value.replace(/\D/g, '').slice(0, 6))}
                        className={`w-full pl-10 pr-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-center text-lg tracking-widest ${
                          errors.mfaCode ? 'border-red-500' : 'border-gray-300'
                        }`}
                        placeholder="000000"
                        maxLength={6}
                        disabled={isLoading}
                      />
                    </div>
                    {errors.mfaCode && (
                      <p className="text-sm text-red-600 mt-1">{errors.mfaCode}</p>
                    )}
                  </div>

                  {/* General Error */}
                  {errors.general && (
                    <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <AlertCircle className="h-4 w-4 text-red-500" />
                      <p className="text-sm text-red-700">{errors.general}</p>
                    </div>
                  )}

                  {/* Verify Button */}
                  <button
                    onClick={handleMFAVerification}
                    disabled={isLoading || formData.mfaCode.length !== 6}
                    className="w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <CheckCircle className="h-4 w-4" />
                        <span>Verify Code</span>
                      </>
                    )}
                  </button>

                  {/* Back to Login */}
                  <div className="text-center">
                    <button
                      onClick={() => setMode('login')}
                      className="text-sm text-indigo-600 hover:text-indigo-700"
                      disabled={isLoading}
                    >
                      Back to login
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </motion.div>
    </motion.div>
  )
}

export default Authentication
