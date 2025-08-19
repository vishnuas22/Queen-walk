// Advanced Authentication & Security System for MasterX

// ===== TYPES =====

export interface User {
  id: string
  email: string
  name: string
  avatar?: string
  role: 'admin' | 'user' | 'viewer'
  permissions: string[]
  mfaEnabled: boolean
  lastLogin: Date
  createdAt: Date
  preferences: UserPreferences
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto'
  language: string
  notifications: {
    email: boolean
    push: boolean
    inApp: boolean
  }
  privacy: {
    shareAnalytics: boolean
    shareUsageData: boolean
  }
}

export interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  mfaRequired: boolean
  sessionExpiry: Date | null
}

export interface LoginCredentials {
  email: string
  password: string
  rememberMe?: boolean
}

export interface MFACredentials {
  code: string
  method: 'totp' | 'sms' | 'email'
}

export interface AuthConfig {
  apiUrl: string
  tokenKey: string
  refreshTokenKey: string
  sessionTimeout: number
  mfaRequired: boolean
  ssoEnabled: boolean
  providers: string[]
}

// ===== AUTH SERVICE =====

export class AuthService {
  private static instance: AuthService
  private config: AuthConfig
  private refreshTimer: NodeJS.Timeout | null = null

  private constructor(config: Partial<AuthConfig> = {}) {
    this.config = {
      apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080',
      tokenKey: 'masterx_token',
      refreshTokenKey: 'masterx_refresh_token',
      sessionTimeout: 24 * 60 * 60 * 1000, // 24 hours
      mfaRequired: true,
      ssoEnabled: true,
      providers: ['google', 'microsoft', 'github'],
      ...config,
    }
  }

  public static getInstance(config?: Partial<AuthConfig>): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService(config)
    }
    return AuthService.instance
  }

  // ===== AUTHENTICATION METHODS =====

  public async login(credentials: LoginCredentials): Promise<{ user?: User; mfaRequired?: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.config.apiUrl}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      })

      const data = await response.json()

      if (!response.ok) {
        return { error: data.message || 'Login failed' }
      }

      if (data.mfaRequired) {
        return { mfaRequired: true }
      }

      // Store tokens
      this.setTokens(data.token, data.refreshToken)
      
      // Start session management
      this.startSessionManagement()

      return { user: data.user }
    } catch (error) {
      return { error: 'Network error. Please try again.' }
    }
  }

  public async verifyMFA(credentials: MFACredentials): Promise<{ user?: User; error?: string }> {
    try {
      const response = await fetch(`${this.config.apiUrl}/auth/mfa/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      })

      const data = await response.json()

      if (!response.ok) {
        return { error: data.message || 'MFA verification failed' }
      }

      // Store tokens
      this.setTokens(data.token, data.refreshToken)
      
      // Start session management
      this.startSessionManagement()

      return { user: data.user }
    } catch (error) {
      return { error: 'Network error. Please try again.' }
    }
  }

  public async logout(): Promise<void> {
    try {
      const token = this.getToken()
      if (token) {
        await fetch(`${this.config.apiUrl}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        })
      }
    } catch (error) {
      console.warn('Logout request failed:', error)
    } finally {
      this.clearTokens()
      this.stopSessionManagement()
    }
  }

  public async refreshToken(): Promise<{ token?: string; error?: string }> {
    try {
      const refreshToken = this.getRefreshToken()
      if (!refreshToken) {
        return { error: 'No refresh token available' }
      }

      const response = await fetch(`${this.config.apiUrl}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refreshToken }),
      })

      const data = await response.json()

      if (!response.ok) {
        this.clearTokens()
        return { error: data.message || 'Token refresh failed' }
      }

      this.setTokens(data.token, data.refreshToken)
      return { token: data.token }
    } catch (error) {
      this.clearTokens()
      return { error: 'Network error during token refresh' }
    }
  }

  public async getCurrentUser(): Promise<{ user?: User; error?: string }> {
    try {
      const token = this.getToken()
      if (!token) {
        return { error: 'No authentication token' }
      }

      const response = await fetch(`${this.config.apiUrl}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })

      const data = await response.json()

      if (!response.ok) {
        if (response.status === 401) {
          // Try to refresh token
          const refreshResult = await this.refreshToken()
          if (refreshResult.error) {
            return { error: 'Session expired' }
          }
          // Retry with new token
          return this.getCurrentUser()
        }
        return { error: data.message || 'Failed to get user info' }
      }

      return { user: data.user }
    } catch (error) {
      return { error: 'Network error. Please try again.' }
    }
  }

  // ===== SSO METHODS =====

  public async loginWithSSO(provider: string): Promise<{ url?: string; error?: string }> {
    try {
      if (!this.config.ssoEnabled) {
        return { error: 'SSO is not enabled' }
      }

      if (!this.config.providers.includes(provider)) {
        return { error: `Provider ${provider} is not supported` }
      }

      const response = await fetch(`${this.config.apiUrl}/auth/sso/${provider}`, {
        method: 'GET',
      })

      const data = await response.json()

      if (!response.ok) {
        return { error: data.message || 'SSO initialization failed' }
      }

      return { url: data.authUrl }
    } catch (error) {
      return { error: 'Network error. Please try again.' }
    }
  }

  public async handleSSOCallback(code: string, state: string): Promise<{ user?: User; error?: string }> {
    try {
      const response = await fetch(`${this.config.apiUrl}/auth/sso/callback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code, state }),
      })

      const data = await response.json()

      if (!response.ok) {
        return { error: data.message || 'SSO authentication failed' }
      }

      // Store tokens
      this.setTokens(data.token, data.refreshToken)
      
      // Start session management
      this.startSessionManagement()

      return { user: data.user }
    } catch (error) {
      return { error: 'Network error. Please try again.' }
    }
  }

  // ===== MFA MANAGEMENT =====

  public async enableMFA(): Promise<{ qrCode?: string; backupCodes?: string[]; error?: string }> {
    try {
      const token = this.getToken()
      if (!token) {
        return { error: 'Authentication required' }
      }

      const response = await fetch(`${this.config.apiUrl}/auth/mfa/enable`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })

      const data = await response.json()

      if (!response.ok) {
        return { error: data.message || 'Failed to enable MFA' }
      }

      return { qrCode: data.qrCode, backupCodes: data.backupCodes }
    } catch (error) {
      return { error: 'Network error. Please try again.' }
    }
  }

  public async disableMFA(password: string): Promise<{ success?: boolean; error?: string }> {
    try {
      const token = this.getToken()
      if (!token) {
        return { error: 'Authentication required' }
      }

      const response = await fetch(`${this.config.apiUrl}/auth/mfa/disable`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ password }),
      })

      const data = await response.json()

      if (!response.ok) {
        return { error: data.message || 'Failed to disable MFA' }
      }

      return { success: true }
    } catch (error) {
      return { error: 'Network error. Please try again.' }
    }
  }

  // ===== TOKEN MANAGEMENT =====

  private setTokens(token: string, refreshToken: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem(this.config.tokenKey, token)
      localStorage.setItem(this.config.refreshTokenKey, refreshToken)
    }
  }

  private getToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(this.config.tokenKey)
    }
    return null
  }

  private getRefreshToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(this.config.refreshTokenKey)
    }
    return null
  }

  private clearTokens(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(this.config.tokenKey)
      localStorage.removeItem(this.config.refreshTokenKey)
    }
  }

  // ===== SESSION MANAGEMENT =====

  private startSessionManagement(): void {
    this.stopSessionManagement()
    
    // Refresh token every 15 minutes
    this.refreshTimer = setInterval(() => {
      this.refreshToken().catch(console.error)
    }, 15 * 60 * 1000)
  }

  private stopSessionManagement(): void {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer)
      this.refreshTimer = null
    }
  }

  // ===== UTILITY METHODS =====

  public isAuthenticated(): boolean {
    return !!this.getToken()
  }

  public hasPermission(permission: string, user?: User): boolean {
    if (!user) return false
    return user.permissions.includes(permission) || user.role === 'admin'
  }

  public isAdmin(user?: User): boolean {
    return user?.role === 'admin'
  }

  public getAuthHeader(): { Authorization: string } | {} {
    const token = this.getToken()
    return token ? { Authorization: `Bearer ${token}` } : {}
  }
}

// ===== SECURITY UTILITIES =====

export class SecurityUtils {
  public static validatePassword(password: string): { isValid: boolean; errors: string[] } {
    const errors: string[] = []

    if (password.length < 8) {
      errors.push('Password must be at least 8 characters long')
    }

    if (!/[A-Z]/.test(password)) {
      errors.push('Password must contain at least one uppercase letter')
    }

    if (!/[a-z]/.test(password)) {
      errors.push('Password must contain at least one lowercase letter')
    }

    if (!/\d/.test(password)) {
      errors.push('Password must contain at least one number')
    }

    if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
      errors.push('Password must contain at least one special character')
    }

    return {
      isValid: errors.length === 0,
      errors,
    }
  }

  public static generateSecurePassword(length = 16): string {
    const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*(),.?":{}|<>'
    let password = ''

    for (let i = 0; i < length; i++) {
      password += charset.charAt(Math.floor(Math.random() * charset.length))
    }

    return password
  }

  public static sanitizeInput(input: string): string {
    return input
      .replace(/[<>]/g, '') // Remove potential HTML tags
      .trim()
  }

  public static validateEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(email)
  }

  public static hashData(data: string): Promise<string> {
    if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
      return window.crypto.subtle.digest('SHA-256', new TextEncoder().encode(data))
        .then(hashBuffer => {
          const hashArray = Array.from(new Uint8Array(hashBuffer))
          return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
        })
    }

    // Fallback for environments without crypto.subtle
    return Promise.resolve(btoa(data))
  }
}

// ===== AUDIT LOGGING =====

export interface AuditEvent {
  id: string
  userId?: string
  action: string
  resource: string
  timestamp: Date
  ipAddress?: string
  userAgent?: string
  metadata?: Record<string, any>
}

export class AuditLogger {
  private static instance: AuditLogger
  private events: AuditEvent[] = []
  private apiUrl: string

  private constructor(apiUrl: string) {
    this.apiUrl = apiUrl
  }

  public static getInstance(apiUrl?: string): AuditLogger {
    if (!AuditLogger.instance) {
      AuditLogger.instance = new AuditLogger(apiUrl || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080')
    }
    return AuditLogger.instance
  }

  public async log(event: Omit<AuditEvent, 'id' | 'timestamp'>): Promise<void> {
    const auditEvent: AuditEvent = {
      ...event,
      id: `audit-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
    }

    // Store locally
    this.events.push(auditEvent)

    // Send to server
    try {
      await fetch(`${this.apiUrl}/audit/log`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...AuthService.getInstance().getAuthHeader(),
        },
        body: JSON.stringify(auditEvent),
      })
    } catch (error) {
      console.warn('Failed to send audit log to server:', error)
    }
  }

  public getEvents(): AuditEvent[] {
    return [...this.events]
  }

  public clearEvents(): void {
    this.events = []
  }
}

export default AuthService
