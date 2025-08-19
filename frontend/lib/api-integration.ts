// API Integration Service for Third-party Connections

// ===== TYPES =====

export interface APIConnection {
  id: string
  name: string
  type: 'rest' | 'graphql' | 'websocket' | 'grpc'
  baseUrl: string
  authentication: APIAuthentication
  headers: Record<string, string>
  rateLimit: {
    requests: number
    window: number // in milliseconds
  }
  status: 'connected' | 'disconnected' | 'error' | 'pending'
  lastSync: Date | null
  metadata: Record<string, any>
}

export interface APIAuthentication {
  type: 'none' | 'api-key' | 'bearer' | 'oauth2' | 'basic'
  credentials: {
    apiKey?: string
    token?: string
    username?: string
    password?: string
    clientId?: string
    clientSecret?: string
    refreshToken?: string
  }
  expiresAt?: Date
}

export interface APIEndpoint {
  id: string
  connectionId: string
  path: string
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  description: string
  parameters: APIParameter[]
  responseSchema: any
  cached: boolean
  cacheTTL: number
}

export interface APIParameter {
  name: string
  type: 'string' | 'number' | 'boolean' | 'object' | 'array'
  required: boolean
  description: string
  defaultValue?: any
  validation?: {
    min?: number
    max?: number
    pattern?: string
    enum?: any[]
  }
}

export interface APIRequest {
  id: string
  endpointId: string
  parameters: Record<string, any>
  timestamp: Date
  status: 'pending' | 'success' | 'error' | 'timeout'
  response?: any
  error?: string
  duration?: number
}

export interface DataSyncConfig {
  id: string
  name: string
  sourceEndpoint: string
  targetEndpoint?: string
  schedule: {
    type: 'manual' | 'interval' | 'cron'
    value: string | number
  }
  transformations: DataTransformation[]
  filters: DataFilter[]
  enabled: boolean
  lastRun: Date | null
  nextRun: Date | null
}

export interface DataTransformation {
  id: string
  type: 'map' | 'filter' | 'aggregate' | 'join' | 'custom'
  config: Record<string, any>
  order: number
}

export interface DataFilter {
  field: string
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'nin' | 'contains'
  value: any
}

// ===== API INTEGRATION SERVICE =====

export class APIIntegrationService {
  private static instance: APIIntegrationService
  private connections: Map<string, APIConnection> = new Map()
  private endpoints: Map<string, APIEndpoint> = new Map()
  private syncConfigs: Map<string, DataSyncConfig> = new Map()
  private requestHistory: APIRequest[] = []
  private rateLimiters: Map<string, { count: number; resetTime: number }> = new Map()

  // Event handlers
  private connectionHandlers: ((connection: APIConnection) => void)[] = []
  private syncHandlers: ((syncId: string, data: any) => void)[] = []
  private errorHandlers: ((error: Error, context: string) => void)[] = []

  private constructor() {
    this.initializeDefaultConnections()
  }

  public static getInstance(): APIIntegrationService {
    if (!APIIntegrationService.instance) {
      APIIntegrationService.instance = new APIIntegrationService()
    }
    return APIIntegrationService.instance
  }

  // ===== CONNECTION MANAGEMENT =====

  private initializeDefaultConnections(): void {
    // Initialize popular third-party service connections
    const defaultConnections: APIConnection[] = [
      {
        id: 'openai-api',
        name: 'OpenAI API',
        type: 'rest',
        baseUrl: 'https://api.openai.com/v1',
        authentication: {
          type: 'bearer',
          credentials: { token: process.env.OPENAI_API_KEY || '' }
        },
        headers: { 'Content-Type': 'application/json' },
        rateLimit: { requests: 60, window: 60000 },
        status: 'disconnected',
        lastSync: null,
        metadata: { provider: 'OpenAI', category: 'AI' }
      },
      {
        id: 'google-workspace',
        name: 'Google Workspace',
        type: 'rest',
        baseUrl: 'https://www.googleapis.com',
        authentication: {
          type: 'oauth2',
          credentials: {
            clientId: process.env.GOOGLE_CLIENT_ID || '',
            clientSecret: process.env.GOOGLE_CLIENT_SECRET || ''
          }
        },
        headers: { 'Content-Type': 'application/json' },
        rateLimit: { requests: 100, window: 60000 },
        status: 'disconnected',
        lastSync: null,
        metadata: { provider: 'Google', category: 'Productivity' }
      },
      {
        id: 'slack-api',
        name: 'Slack API',
        type: 'rest',
        baseUrl: 'https://slack.com/api',
        authentication: {
          type: 'bearer',
          credentials: { token: process.env.SLACK_BOT_TOKEN || '' }
        },
        headers: { 'Content-Type': 'application/json' },
        rateLimit: { requests: 50, window: 60000 },
        status: 'disconnected',
        lastSync: null,
        metadata: { provider: 'Slack', category: 'Communication' }
      },
      {
        id: 'github-api',
        name: 'GitHub API',
        type: 'rest',
        baseUrl: 'https://api.github.com',
        authentication: {
          type: 'bearer',
          credentials: { token: process.env.GITHUB_TOKEN || '' }
        },
        headers: { 'Accept': 'application/vnd.github.v3+json' },
        rateLimit: { requests: 5000, window: 3600000 },
        status: 'disconnected',
        lastSync: null,
        metadata: { provider: 'GitHub', category: 'Development' }
      },
      {
        id: 'salesforce-api',
        name: 'Salesforce API',
        type: 'rest',
        baseUrl: 'https://your-instance.salesforce.com/services/data/v58.0',
        authentication: {
          type: 'oauth2',
          credentials: {
            clientId: process.env.SALESFORCE_CLIENT_ID || '',
            clientSecret: process.env.SALESFORCE_CLIENT_SECRET || ''
          }
        },
        headers: { 'Content-Type': 'application/json' },
        rateLimit: { requests: 1000, window: 86400000 },
        status: 'disconnected',
        lastSync: null,
        metadata: { provider: 'Salesforce', category: 'CRM' }
      }
    ]

    defaultConnections.forEach(connection => {
      this.connections.set(connection.id, connection)
    })

    this.initializeDefaultEndpoints()
  }

  private initializeDefaultEndpoints(): void {
    const defaultEndpoints: APIEndpoint[] = [
      // OpenAI Endpoints
      {
        id: 'openai-chat',
        connectionId: 'openai-api',
        path: '/chat/completions',
        method: 'POST',
        description: 'Create a chat completion',
        parameters: [
          { name: 'model', type: 'string', required: true, description: 'Model to use' },
          { name: 'messages', type: 'array', required: true, description: 'Chat messages' },
          { name: 'temperature', type: 'number', required: false, description: 'Sampling temperature' }
        ],
        responseSchema: { type: 'object' },
        cached: false,
        cacheTTL: 0
      },
      {
        id: 'openai-embeddings',
        connectionId: 'openai-api',
        path: '/embeddings',
        method: 'POST',
        description: 'Create embeddings',
        parameters: [
          { name: 'model', type: 'string', required: true, description: 'Embedding model' },
          { name: 'input', type: 'string', required: true, description: 'Text to embed' }
        ],
        responseSchema: { type: 'object' },
        cached: true,
        cacheTTL: 3600000
      },

      // Google Workspace Endpoints
      {
        id: 'gmail-messages',
        connectionId: 'google-workspace',
        path: '/gmail/v1/users/me/messages',
        method: 'GET',
        description: 'List Gmail messages',
        parameters: [
          { name: 'q', type: 'string', required: false, description: 'Search query' },
          { name: 'maxResults', type: 'number', required: false, description: 'Max results' }
        ],
        responseSchema: { type: 'object' },
        cached: true,
        cacheTTL: 300000
      },
      {
        id: 'calendar-events',
        connectionId: 'google-workspace',
        path: '/calendar/v3/calendars/primary/events',
        method: 'GET',
        description: 'List calendar events',
        parameters: [
          { name: 'timeMin', type: 'string', required: false, description: 'Start time' },
          { name: 'timeMax', type: 'string', required: false, description: 'End time' }
        ],
        responseSchema: { type: 'object' },
        cached: true,
        cacheTTL: 600000
      },

      // Slack Endpoints
      {
        id: 'slack-channels',
        connectionId: 'slack-api',
        path: '/conversations.list',
        method: 'GET',
        description: 'List Slack channels',
        parameters: [
          { name: 'types', type: 'string', required: false, description: 'Channel types' }
        ],
        responseSchema: { type: 'object' },
        cached: true,
        cacheTTL: 1800000
      },
      {
        id: 'slack-messages',
        connectionId: 'slack-api',
        path: '/conversations.history',
        method: 'GET',
        description: 'Get channel messages',
        parameters: [
          { name: 'channel', type: 'string', required: true, description: 'Channel ID' },
          { name: 'limit', type: 'number', required: false, description: 'Message limit' }
        ],
        responseSchema: { type: 'object' },
        cached: true,
        cacheTTL: 60000
      },

      // GitHub Endpoints
      {
        id: 'github-repos',
        connectionId: 'github-api',
        path: '/user/repos',
        method: 'GET',
        description: 'List user repositories',
        parameters: [
          { name: 'sort', type: 'string', required: false, description: 'Sort order' },
          { name: 'per_page', type: 'number', required: false, description: 'Results per page' }
        ],
        responseSchema: { type: 'array' },
        cached: true,
        cacheTTL: 1800000
      },
      {
        id: 'github-issues',
        connectionId: 'github-api',
        path: '/repos/{owner}/{repo}/issues',
        method: 'GET',
        description: 'List repository issues',
        parameters: [
          { name: 'owner', type: 'string', required: true, description: 'Repository owner' },
          { name: 'repo', type: 'string', required: true, description: 'Repository name' },
          { name: 'state', type: 'string', required: false, description: 'Issue state' }
        ],
        responseSchema: { type: 'array' },
        cached: true,
        cacheTTL: 300000
      },

      // Salesforce Endpoints
      {
        id: 'salesforce-accounts',
        connectionId: 'salesforce-api',
        path: '/sobjects/Account',
        method: 'GET',
        description: 'List Salesforce accounts',
        parameters: [
          { name: 'limit', type: 'number', required: false, description: 'Record limit' }
        ],
        responseSchema: { type: 'object' },
        cached: true,
        cacheTTL: 1800000
      },
      {
        id: 'salesforce-opportunities',
        connectionId: 'salesforce-api',
        path: '/sobjects/Opportunity',
        method: 'GET',
        description: 'List Salesforce opportunities',
        parameters: [
          { name: 'limit', type: 'number', required: false, description: 'Record limit' }
        ],
        responseSchema: { type: 'object' },
        cached: true,
        cacheTTL: 900000
      }
    ]

    defaultEndpoints.forEach(endpoint => {
      this.endpoints.set(endpoint.id, endpoint)
    })
  }

  public async testConnection(connectionId: string): Promise<boolean> {
    const connection = this.connections.get(connectionId)
    if (!connection) {
      throw new Error(`Connection ${connectionId} not found`)
    }

    try {
      // Simple health check request
      const response = await fetch(connection.baseUrl, {
        method: 'HEAD',
        headers: this.buildHeaders(connection)
      })

      const isConnected = response.ok || response.status === 405 // Some APIs don't support HEAD
      
      connection.status = isConnected ? 'connected' : 'error'
      connection.lastSync = new Date()
      
      this.notifyConnectionHandlers(connection)
      return isConnected
    } catch (error) {
      connection.status = 'error'
      this.notifyErrorHandlers(error as Error, `Connection test for ${connectionId}`)
      return false
    }
  }

  public async makeRequest(endpointId: string, parameters: Record<string, any> = {}): Promise<any> {
    const endpoint = this.endpoints.get(endpointId)
    if (!endpoint) {
      throw new Error(`Endpoint ${endpointId} not found`)
    }

    const connection = this.connections.get(endpoint.connectionId)
    if (!connection) {
      throw new Error(`Connection ${endpoint.connectionId} not found`)
    }

    // Check rate limiting
    if (!this.checkRateLimit(connection.id, connection.rateLimit)) {
      throw new Error(`Rate limit exceeded for ${connection.name}`)
    }

    const request: APIRequest = {
      id: `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      endpointId,
      parameters,
      timestamp: new Date(),
      status: 'pending'
    }

    try {
      const startTime = Date.now()
      
      // Build URL with path parameters
      let url = connection.baseUrl + endpoint.path
      Object.keys(parameters).forEach(key => {
        url = url.replace(`{${key}}`, encodeURIComponent(parameters[key]))
      })

      // Build request options
      const options: RequestInit = {
        method: endpoint.method,
        headers: this.buildHeaders(connection)
      }

      // Add body for POST/PUT/PATCH requests
      if (['POST', 'PUT', 'PATCH'].includes(endpoint.method)) {
        options.body = JSON.stringify(parameters)
      } else if (endpoint.method === 'GET') {
        // Add query parameters for GET requests
        const queryParams = new URLSearchParams()
        Object.keys(parameters).forEach(key => {
          if (!endpoint.path.includes(`{${key}}`)) {
            queryParams.append(key, parameters[key])
          }
        })
        if (queryParams.toString()) {
          url += '?' + queryParams.toString()
        }
      }

      const response = await fetch(url, options)
      const data = await response.json()

      request.status = response.ok ? 'success' : 'error'
      request.response = data
      request.duration = Date.now() - startTime

      if (!response.ok) {
        request.error = `HTTP ${response.status}: ${response.statusText}`
      }

      this.requestHistory.unshift(request)
      this.requestHistory = this.requestHistory.slice(0, 100) // Keep last 100 requests

      return data
    } catch (error) {
      request.status = 'error'
      request.error = (error as Error).message
      request.duration = Date.now() - request.timestamp.getTime()
      
      this.requestHistory.unshift(request)
      this.notifyErrorHandlers(error as Error, `API request to ${endpointId}`)
      
      throw error
    }
  }

  // ===== DATA SYNCHRONIZATION =====

  public createSyncConfig(config: Omit<DataSyncConfig, 'id' | 'lastRun' | 'nextRun'>): string {
    const syncConfig: DataSyncConfig = {
      ...config,
      id: `sync-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      lastRun: null,
      nextRun: this.calculateNextRun(config.schedule)
    }

    this.syncConfigs.set(syncConfig.id, syncConfig)
    return syncConfig.id
  }

  public async executeSyncConfig(syncId: string): Promise<any> {
    const config = this.syncConfigs.get(syncId)
    if (!config || !config.enabled) {
      throw new Error(`Sync config ${syncId} not found or disabled`)
    }

    try {
      // Fetch data from source endpoint
      const sourceData = await this.makeRequest(config.sourceEndpoint)
      
      // Apply filters
      let filteredData = this.applyFilters(sourceData, config.filters)
      
      // Apply transformations
      const transformedData = this.applyTransformations(filteredData, config.transformations)
      
      // Update sync config
      config.lastRun = new Date()
      config.nextRun = this.calculateNextRun(config.schedule)
      
      this.notifySyncHandlers(syncId, transformedData)
      return transformedData
    } catch (error) {
      this.notifyErrorHandlers(error as Error, `Data sync ${syncId}`)
      throw error
    }
  }

  // ===== UTILITY METHODS =====

  private buildHeaders(connection: APIConnection): Record<string, string> {
    const headers = { ...connection.headers }

    switch (connection.authentication.type) {
      case 'api-key':
        if (connection.authentication.credentials.apiKey) {
          headers['X-API-Key'] = connection.authentication.credentials.apiKey
        }
        break
      case 'bearer':
        if (connection.authentication.credentials.token) {
          headers['Authorization'] = `Bearer ${connection.authentication.credentials.token}`
        }
        break
      case 'basic':
        if (connection.authentication.credentials.username && connection.authentication.credentials.password) {
          const credentials = btoa(`${connection.authentication.credentials.username}:${connection.authentication.credentials.password}`)
          headers['Authorization'] = `Basic ${credentials}`
        }
        break
    }

    return headers
  }

  private checkRateLimit(connectionId: string, rateLimit: APIConnection['rateLimit']): boolean {
    const now = Date.now()
    const limiter = this.rateLimiters.get(connectionId)

    if (!limiter || now > limiter.resetTime) {
      this.rateLimiters.set(connectionId, {
        count: 1,
        resetTime: now + rateLimit.window
      })
      return true
    }

    if (limiter.count >= rateLimit.requests) {
      return false
    }

    limiter.count++
    return true
  }

  private calculateNextRun(schedule: DataSyncConfig['schedule']): Date {
    const now = new Date()
    
    switch (schedule.type) {
      case 'interval':
        return new Date(now.getTime() + (schedule.value as number))
      case 'cron':
        // Simplified cron parsing - in real implementation, use a cron library
        return new Date(now.getTime() + 3600000) // Default to 1 hour
      default:
        return now
    }
  }

  private applyFilters(data: any, filters: DataFilter[]): any {
    if (!Array.isArray(data) || filters.length === 0) {
      return data
    }

    return data.filter(item => {
      return filters.every(filter => {
        const value = item[filter.field]
        
        switch (filter.operator) {
          case 'eq': return value === filter.value
          case 'ne': return value !== filter.value
          case 'gt': return value > filter.value
          case 'gte': return value >= filter.value
          case 'lt': return value < filter.value
          case 'lte': return value <= filter.value
          case 'in': return Array.isArray(filter.value) && filter.value.includes(value)
          case 'nin': return Array.isArray(filter.value) && !filter.value.includes(value)
          case 'contains': return String(value).includes(String(filter.value))
          default: return true
        }
      })
    })
  }

  private applyTransformations(data: any, transformations: DataTransformation[]): any {
    let result = data

    transformations
      .sort((a, b) => a.order - b.order)
      .forEach(transformation => {
        switch (transformation.type) {
          case 'map':
            if (Array.isArray(result)) {
              result = result.map(item => this.mapObject(item, transformation.config))
            }
            break
          case 'filter':
            if (Array.isArray(result)) {
              result = result.filter(item => this.evaluateCondition(item, transformation.config))
            }
            break
          case 'aggregate':
            if (Array.isArray(result)) {
              result = this.aggregateData(result, transformation.config)
            }
            break
        }
      })

    return result
  }

  private mapObject(obj: any, mapping: Record<string, any>): any {
    const result: any = {}
    
    Object.keys(mapping).forEach(key => {
      const sourcePath = mapping[key]
      result[key] = this.getNestedValue(obj, sourcePath)
    })
    
    return result
  }

  private getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj)
  }

  private evaluateCondition(obj: any, condition: any): boolean {
    // Simplified condition evaluation
    return true
  }

  private aggregateData(data: any[], config: any): any {
    // Simplified aggregation
    return data
  }

  // ===== EVENT HANDLERS =====

  public onConnection(handler: (connection: APIConnection) => void): () => void {
    this.connectionHandlers.push(handler)
    return () => {
      const index = this.connectionHandlers.indexOf(handler)
      if (index > -1) {
        this.connectionHandlers.splice(index, 1)
      }
    }
  }

  public onSync(handler: (syncId: string, data: any) => void): () => void {
    this.syncHandlers.push(handler)
    return () => {
      const index = this.syncHandlers.indexOf(handler)
      if (index > -1) {
        this.syncHandlers.splice(index, 1)
      }
    }
  }

  public onError(handler: (error: Error, context: string) => void): () => void {
    this.errorHandlers.push(handler)
    return () => {
      const index = this.errorHandlers.indexOf(handler)
      if (index > -1) {
        this.errorHandlers.splice(index, 1)
      }
    }
  }

  private notifyConnectionHandlers(connection: APIConnection): void {
    this.connectionHandlers.forEach(handler => handler(connection))
  }

  private notifySyncHandlers(syncId: string, data: any): void {
    this.syncHandlers.forEach(handler => handler(syncId, data))
  }

  private notifyErrorHandlers(error: Error, context: string): void {
    this.errorHandlers.forEach(handler => handler(error, context))
  }

  // ===== GETTERS =====

  public getConnections(): APIConnection[] {
    return Array.from(this.connections.values())
  }

  public getConnection(id: string): APIConnection | undefined {
    return this.connections.get(id)
  }

  public getEndpoints(connectionId?: string): APIEndpoint[] {
    const endpoints = Array.from(this.endpoints.values())
    return connectionId 
      ? endpoints.filter(endpoint => endpoint.connectionId === connectionId)
      : endpoints
  }

  public getSyncConfigs(): DataSyncConfig[] {
    return Array.from(this.syncConfigs.values())
  }

  public getRequestHistory(): APIRequest[] {
    return this.requestHistory
  }
}

export default APIIntegrationService

// ===== PLUGIN ARCHITECTURE =====

export interface Plugin {
  id: string
  name: string
  version: string
  description: string
  author: string
  category: 'ai' | 'productivity' | 'integration' | 'visualization' | 'utility'
  permissions: PluginPermission[]
  dependencies: string[]
  config: PluginConfig
  status: 'installed' | 'active' | 'inactive' | 'error'
  installDate: Date
  lastUpdate: Date
  metadata: Record<string, any>
}

export interface PluginPermission {
  type: 'api' | 'storage' | 'ui' | 'notifications' | 'camera' | 'microphone' | 'location'
  description: string
  required: boolean
}

export interface PluginConfig {
  settings: Record<string, PluginSetting>
  hooks: PluginHook[]
  components: PluginComponent[]
  apis: PluginAPI[]
}

export interface PluginSetting {
  type: 'string' | 'number' | 'boolean' | 'select' | 'multiselect'
  label: string
  description: string
  defaultValue: any
  options?: { label: string; value: any }[]
  validation?: {
    required?: boolean
    min?: number
    max?: number
    pattern?: string
  }
}

export interface PluginHook {
  event: string
  handler: string
  priority: number
}

export interface PluginComponent {
  id: string
  type: 'widget' | 'panel' | 'modal' | 'toolbar' | 'sidebar'
  position: string
  component: string
  props?: Record<string, any>
}

export interface PluginAPI {
  endpoint: string
  method: string
  handler: string
  permissions: string[]
}

export interface PluginManifest {
  name: string
  version: string
  description: string
  author: string
  category: Plugin['category']
  permissions: PluginPermission[]
  dependencies: string[]
  main: string
  config?: Partial<PluginConfig>
}

export interface PluginRuntime {
  plugin: Plugin
  context: PluginContext
  sandbox: PluginSandbox
  loaded: boolean
  error?: Error
}

export interface PluginContext {
  api: {
    ui: PluginUIAPI
    storage: PluginStorageAPI
    notifications: PluginNotificationAPI
    http: PluginHttpAPI
  }
  events: PluginEventEmitter
  settings: Record<string, any>
}

export interface PluginUIAPI {
  showModal: (component: React.ComponentType, props?: any) => void
  showNotification: (message: string, type?: 'info' | 'success' | 'warning' | 'error') => void
  addToolbarButton: (button: ToolbarButton) => void
  addSidebarPanel: (panel: SidebarPanel) => void
  registerComponent: (id: string, component: React.ComponentType) => void
}

export interface PluginStorageAPI {
  get: (key: string) => Promise<any>
  set: (key: string, value: any) => Promise<void>
  delete: (key: string) => Promise<void>
  clear: () => Promise<void>
}

export interface PluginNotificationAPI {
  send: (notification: PluginNotification) => void
  subscribe: (callback: (notification: PluginNotification) => void) => () => void
}

export interface PluginHttpAPI {
  request: (url: string, options?: RequestInit) => Promise<Response>
  get: (url: string, headers?: Record<string, string>) => Promise<any>
  post: (url: string, data?: any, headers?: Record<string, string>) => Promise<any>
}

export interface PluginEventEmitter {
  on: (event: string, handler: Function) => void
  off: (event: string, handler: Function) => void
  emit: (event: string, ...args: any[]) => void
}

export interface PluginSandbox {
  globals: Record<string, any>
  restrictions: string[]
  timeoutMs: number
}

export interface ToolbarButton {
  id: string
  label: string
  icon: string
  onClick: () => void
  position?: 'left' | 'right'
}

export interface SidebarPanel {
  id: string
  title: string
  component: React.ComponentType
  icon?: string
  order?: number
}

export interface PluginNotification {
  id: string
  title: string
  message: string
  type: 'info' | 'success' | 'warning' | 'error'
  actions?: PluginNotificationAction[]
  persistent?: boolean
}

export interface PluginNotificationAction {
  label: string
  action: () => void
  style?: 'primary' | 'secondary' | 'danger'
}

// ===== PLUGIN MANAGER =====

export class PluginManager {
  private static instance: PluginManager
  private plugins: Map<string, Plugin> = new Map()
  private runtimes: Map<string, PluginRuntime> = new Map()
  private registry: Map<string, PluginManifest> = new Map()

  // Event handlers
  private installHandlers: ((plugin: Plugin) => void)[] = []
  private activateHandlers: ((plugin: Plugin) => void)[] = []
  private errorHandlers: ((pluginId: string, error: Error) => void)[] = []

  private constructor() {
    this.initializeBuiltinPlugins()
  }

  public static getInstance(): PluginManager {
    if (!PluginManager.instance) {
      PluginManager.instance = new PluginManager()
    }
    return PluginManager.instance
  }

  // ===== PLUGIN LIFECYCLE =====

  private initializeBuiltinPlugins(): void {
    const builtinPlugins: Plugin[] = [
      {
        id: 'ai-assistant-enhancer',
        name: 'AI Assistant Enhancer',
        version: '1.0.0',
        description: 'Enhances AI responses with additional context and formatting',
        author: 'MasterX Team',
        category: 'ai',
        permissions: [
          { type: 'api', description: 'Access to AI API endpoints', required: true },
          { type: 'ui', description: 'Modify chat interface', required: true }
        ],
        dependencies: [],
        config: {
          settings: {
            enhanceResponses: {
              type: 'boolean',
              label: 'Enhance AI Responses',
              description: 'Add formatting and context to AI responses',
              defaultValue: true
            },
            contextWindow: {
              type: 'number',
              label: 'Context Window',
              description: 'Number of previous messages to include',
              defaultValue: 5,
              validation: { min: 1, max: 20 }
            }
          },
          hooks: [
            { event: 'ai.response.received', handler: 'enhanceResponse', priority: 10 }
          ],
          components: [],
          apis: []
        },
        status: 'installed',
        installDate: new Date(),
        lastUpdate: new Date(),
        metadata: { builtin: true }
      },
      {
        id: 'productivity-dashboard',
        name: 'Productivity Dashboard',
        version: '1.0.0',
        description: 'Track and visualize productivity metrics',
        author: 'MasterX Team',
        category: 'productivity',
        permissions: [
          { type: 'storage', description: 'Store productivity data', required: true },
          { type: 'ui', description: 'Add dashboard components', required: true }
        ],
        dependencies: [],
        config: {
          settings: {
            trackingEnabled: {
              type: 'boolean',
              label: 'Enable Tracking',
              description: 'Track productivity metrics',
              defaultValue: true
            },
            updateInterval: {
              type: 'select',
              label: 'Update Interval',
              description: 'How often to update metrics',
              defaultValue: '5m',
              options: [
                { label: '1 minute', value: '1m' },
                { label: '5 minutes', value: '5m' },
                { label: '15 minutes', value: '15m' }
              ]
            }
          },
          hooks: [
            { event: 'user.action', handler: 'trackAction', priority: 5 }
          ],
          components: [
            {
              id: 'productivity-widget',
              type: 'widget',
              position: 'sidebar',
              component: 'ProductivityWidget'
            }
          ],
          apis: [
            {
              endpoint: '/api/plugins/productivity/metrics',
              method: 'GET',
              handler: 'getMetrics',
              permissions: ['storage']
            }
          ]
        },
        status: 'installed',
        installDate: new Date(),
        lastUpdate: new Date(),
        metadata: { builtin: true }
      },
      {
        id: 'slack-integration',
        name: 'Slack Integration',
        version: '1.0.0',
        description: 'Integrate with Slack for seamless communication',
        author: 'MasterX Team',
        category: 'integration',
        permissions: [
          { type: 'api', description: 'Access Slack API', required: true },
          { type: 'notifications', description: 'Send notifications', required: true }
        ],
        dependencies: ['api-integration'],
        config: {
          settings: {
            workspaceUrl: {
              type: 'string',
              label: 'Workspace URL',
              description: 'Your Slack workspace URL',
              defaultValue: '',
              validation: { required: true, pattern: 'https://.*\\.slack\\.com' }
            },
            autoSync: {
              type: 'boolean',
              label: 'Auto Sync Messages',
              description: 'Automatically sync Slack messages',
              defaultValue: false
            }
          },
          hooks: [
            { event: 'message.received', handler: 'handleSlackMessage', priority: 8 }
          ],
          components: [
            {
              id: 'slack-panel',
              type: 'panel',
              position: 'sidebar',
              component: 'SlackPanel'
            }
          ],
          apis: [
            {
              endpoint: '/api/plugins/slack/send',
              method: 'POST',
              handler: 'sendMessage',
              permissions: ['api']
            }
          ]
        },
        status: 'installed',
        installDate: new Date(),
        lastUpdate: new Date(),
        metadata: { builtin: true }
      },
      {
        id: 'data-visualizer',
        name: 'Advanced Data Visualizer',
        version: '1.0.0',
        description: 'Create interactive charts and visualizations',
        author: 'MasterX Team',
        category: 'visualization',
        permissions: [
          { type: 'ui', description: 'Render visualizations', required: true },
          { type: 'storage', description: 'Cache visualization data', required: false }
        ],
        dependencies: [],
        config: {
          settings: {
            defaultChartType: {
              type: 'select',
              label: 'Default Chart Type',
              description: 'Default visualization type',
              defaultValue: 'line',
              options: [
                { label: 'Line Chart', value: 'line' },
                { label: 'Bar Chart', value: 'bar' },
                { label: 'Pie Chart', value: 'pie' },
                { label: 'Scatter Plot', value: 'scatter' }
              ]
            },
            animationsEnabled: {
              type: 'boolean',
              label: 'Enable Animations',
              description: 'Animate chart transitions',
              defaultValue: true
            }
          },
          hooks: [
            { event: 'data.visualize', handler: 'createVisualization', priority: 10 }
          ],
          components: [
            {
              id: 'chart-toolbar',
              type: 'toolbar',
              position: 'top',
              component: 'ChartToolbar'
            }
          ],
          apis: [
            {
              endpoint: '/api/plugins/visualizer/render',
              method: 'POST',
              handler: 'renderChart',
              permissions: ['ui']
            }
          ]
        },
        status: 'installed',
        installDate: new Date(),
        lastUpdate: new Date(),
        metadata: { builtin: true }
      },
      {
        id: 'voice-commands',
        name: 'Voice Commands',
        version: '1.0.0',
        description: 'Control the interface with voice commands',
        author: 'MasterX Team',
        category: 'utility',
        permissions: [
          { type: 'microphone', description: 'Access microphone for voice input', required: true },
          { type: 'ui', description: 'Control interface elements', required: true }
        ],
        dependencies: [],
        config: {
          settings: {
            wakeWord: {
              type: 'string',
              label: 'Wake Word',
              description: 'Word to activate voice commands',
              defaultValue: 'MasterX'
            },
            language: {
              type: 'select',
              label: 'Language',
              description: 'Voice recognition language',
              defaultValue: 'en-US',
              options: [
                { label: 'English (US)', value: 'en-US' },
                { label: 'English (UK)', value: 'en-GB' },
                { label: 'Spanish', value: 'es-ES' },
                { label: 'French', value: 'fr-FR' }
              ]
            }
          },
          hooks: [
            { event: 'voice.command', handler: 'executeCommand', priority: 10 }
          ],
          components: [
            {
              id: 'voice-indicator',
              type: 'widget',
              position: 'header',
              component: 'VoiceIndicator'
            }
          ],
          apis: []
        },
        status: 'installed',
        installDate: new Date(),
        lastUpdate: new Date(),
        metadata: { builtin: true }
      }
    ]

    builtinPlugins.forEach(plugin => {
      this.plugins.set(plugin.id, plugin)
    })
  }

  public async installPlugin(manifest: PluginManifest, code: string): Promise<string> {
    try {
      // Validate manifest
      this.validateManifest(manifest)

      // Check dependencies
      await this.checkDependencies(manifest.dependencies)

      // Create plugin
      const plugin: Plugin = {
        id: `plugin-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: manifest.name,
        version: manifest.version,
        description: manifest.description,
        author: manifest.author,
        category: manifest.category,
        permissions: manifest.permissions,
        dependencies: manifest.dependencies,
        config: {
          settings: manifest.config?.settings || {},
          hooks: manifest.config?.hooks || [],
          components: manifest.config?.components || [],
          apis: manifest.config?.apis || []
        },
        status: 'installed',
        installDate: new Date(),
        lastUpdate: new Date(),
        metadata: { code }
      }

      this.plugins.set(plugin.id, plugin)
      this.notifyInstallHandlers(plugin)

      return plugin.id
    } catch (error) {
      throw new Error(`Plugin installation failed: ${(error as Error).message}`)
    }
  }

  public async activatePlugin(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId)
    if (!plugin) {
      throw new Error(`Plugin ${pluginId} not found`)
    }

    if (plugin.status === 'active') {
      return
    }

    try {
      // Create runtime context
      const context = this.createPluginContext(plugin)
      const sandbox = this.createPluginSandbox(plugin)

      // Load plugin code
      const runtime: PluginRuntime = {
        plugin,
        context,
        sandbox,
        loaded: false
      }

      // Execute plugin initialization
      await this.loadPluginCode(runtime)

      plugin.status = 'active'
      runtime.loaded = true

      this.runtimes.set(pluginId, runtime)
      this.notifyActivateHandlers(plugin)
    } catch (error) {
      plugin.status = 'error'
      this.notifyErrorHandlers(pluginId, error as Error)
      throw error
    }
  }

  public async deactivatePlugin(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId)
    const runtime = this.runtimes.get(pluginId)

    if (!plugin || !runtime) {
      return
    }

    try {
      // Cleanup plugin resources
      await this.cleanupPluginRuntime(runtime)

      plugin.status = 'inactive'
      this.runtimes.delete(pluginId)
    } catch (error) {
      this.notifyErrorHandlers(pluginId, error as Error)
      throw error
    }
  }

  public uninstallPlugin(pluginId: string): void {
    const plugin = this.plugins.get(pluginId)
    if (!plugin) {
      return
    }

    // Deactivate if active
    if (plugin.status === 'active') {
      this.deactivatePlugin(pluginId)
    }

    // Remove plugin
    this.plugins.delete(pluginId)
  }

  // ===== PLUGIN RUNTIME =====

  private createPluginContext(plugin: Plugin): PluginContext {
    return {
      api: {
        ui: this.createUIAPI(plugin),
        storage: this.createStorageAPI(plugin),
        notifications: this.createNotificationAPI(plugin),
        http: this.createHttpAPI(plugin)
      },
      events: this.createEventEmitter(plugin),
      settings: this.getPluginSettings(plugin)
    }
  }

  private createPluginSandbox(plugin: Plugin): PluginSandbox {
    return {
      globals: {
        console: {
          log: (...args: any[]) => console.log(`[${plugin.name}]`, ...args),
          warn: (...args: any[]) => console.warn(`[${plugin.name}]`, ...args),
          error: (...args: any[]) => console.error(`[${plugin.name}]`, ...args)
        },
        setTimeout,
        clearTimeout,
        setInterval,
        clearInterval
      },
      restrictions: [
        'eval',
        'Function',
        'document',
        'window',
        'global',
        'process'
      ],
      timeoutMs: 30000
    }
  }

  private async loadPluginCode(runtime: PluginRuntime): Promise<void> {
    const { plugin, context, sandbox } = runtime
    const code = plugin.metadata.code

    if (!code) {
      throw new Error('Plugin code not found')
    }

    // Create sandboxed execution environment
    const sandboxedCode = this.sandboxCode(code, sandbox)

    // Execute plugin code
    try {
      const pluginModule = new Function('context', 'sandbox', sandboxedCode)
      await pluginModule(context, sandbox)
    } catch (error) {
      throw new Error(`Plugin execution failed: ${(error as Error).message}`)
    }
  }

  private sandboxCode(code: string, sandbox: PluginSandbox): string {
    // Remove restricted globals
    let sandboxedCode = code

    sandbox.restrictions.forEach(restriction => {
      const regex = new RegExp(`\\b${restriction}\\b`, 'g')
      sandboxedCode = sandboxedCode.replace(regex, 'undefined')
    })

    // Wrap in timeout
    return `
      const timeoutId = setTimeout(() => {
        throw new Error('Plugin execution timeout');
      }, ${sandbox.timeoutMs});

      try {
        ${sandboxedCode}
      } finally {
        clearTimeout(timeoutId);
      }
    `
  }

  private async cleanupPluginRuntime(runtime: PluginRuntime): Promise<void> {
    // Cleanup event listeners, timers, etc.
    // Implementation would depend on specific runtime tracking
  }

  // ===== PLUGIN APIs =====

  private createUIAPI(plugin: Plugin): PluginUIAPI {
    return {
      showModal: (component, props) => {
        // Implementation would integrate with UI system
        console.log(`[${plugin.name}] Show modal:`, component, props)
      },
      showNotification: (message, type = 'info') => {
        // Implementation would integrate with notification system
        console.log(`[${plugin.name}] Notification:`, message, type)
      },
      addToolbarButton: (button) => {
        // Implementation would add button to toolbar
        console.log(`[${plugin.name}] Add toolbar button:`, button)
      },
      addSidebarPanel: (panel) => {
        // Implementation would add panel to sidebar
        console.log(`[${plugin.name}] Add sidebar panel:`, panel)
      },
      registerComponent: (id, component) => {
        // Implementation would register React component
        console.log(`[${plugin.name}] Register component:`, id, component)
      }
    }
  }

  private createStorageAPI(plugin: Plugin): PluginStorageAPI {
    const storageKey = `plugin-${plugin.id}`

    return {
      get: async (key) => {
        const data = localStorage.getItem(`${storageKey}-${key}`)
        return data ? JSON.parse(data) : null
      },
      set: async (key, value) => {
        localStorage.setItem(`${storageKey}-${key}`, JSON.stringify(value))
      },
      delete: async (key) => {
        localStorage.removeItem(`${storageKey}-${key}`)
      },
      clear: async () => {
        Object.keys(localStorage).forEach(key => {
          if (key.startsWith(storageKey)) {
            localStorage.removeItem(key)
          }
        })
      }
    }
  }

  private createNotificationAPI(plugin: Plugin): PluginNotificationAPI {
    return {
      send: (notification) => {
        // Implementation would integrate with notification system
        console.log(`[${plugin.name}] Send notification:`, notification)
      },
      subscribe: (callback) => {
        // Implementation would subscribe to notifications
        console.log(`[${plugin.name}] Subscribe to notifications`)
        return () => {} // Unsubscribe function
      }
    }
  }

  private createHttpAPI(plugin: Plugin): PluginHttpAPI {
    return {
      request: async (url, options) => {
        // Implementation would make HTTP request with plugin permissions
        return fetch(url, options)
      },
      get: async (url, headers) => {
        const response = await fetch(url, { method: 'GET', headers })
        return response.json()
      },
      post: async (url, data, headers) => {
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...headers },
          body: JSON.stringify(data)
        })
        return response.json()
      }
    }
  }

  private createEventEmitter(plugin: Plugin): PluginEventEmitter {
    const listeners = new Map<string, Function[]>()

    return {
      on: (event, handler) => {
        if (!listeners.has(event)) {
          listeners.set(event, [])
        }
        listeners.get(event)!.push(handler)
      },
      off: (event, handler) => {
        const eventListeners = listeners.get(event)
        if (eventListeners) {
          const index = eventListeners.indexOf(handler)
          if (index > -1) {
            eventListeners.splice(index, 1)
          }
        }
      },
      emit: (event, ...args) => {
        const eventListeners = listeners.get(event)
        if (eventListeners) {
          eventListeners.forEach(handler => {
            try {
              handler(...args)
            } catch (error) {
              console.error(`[${plugin.name}] Event handler error:`, error)
            }
          })
        }
      }
    }
  }

  // ===== UTILITY METHODS =====

  private validateManifest(manifest: PluginManifest): void {
    if (!manifest.name || !manifest.version || !manifest.author) {
      throw new Error('Invalid plugin manifest: missing required fields')
    }

    if (!['ai', 'productivity', 'integration', 'visualization', 'utility'].includes(manifest.category)) {
      throw new Error('Invalid plugin category')
    }
  }

  private async checkDependencies(dependencies: string[]): Promise<void> {
    for (const dep of dependencies) {
      const plugin = Array.from(this.plugins.values()).find(p => p.name === dep)
      if (!plugin || plugin.status !== 'active') {
        throw new Error(`Dependency ${dep} not found or not active`)
      }
    }
  }

  private getPluginSettings(plugin: Plugin): Record<string, any> {
    const settings: Record<string, any> = {}

    Object.entries(plugin.config.settings).forEach(([key, setting]) => {
      // Load from storage or use default
      const storageKey = `plugin-${plugin.id}-setting-${key}`
      const stored = localStorage.getItem(storageKey)
      settings[key] = stored ? JSON.parse(stored) : setting.defaultValue
    })

    return settings
  }

  // ===== EVENT HANDLERS =====

  public onInstall(handler: (plugin: Plugin) => void): () => void {
    this.installHandlers.push(handler)
    return () => {
      const index = this.installHandlers.indexOf(handler)
      if (index > -1) {
        this.installHandlers.splice(index, 1)
      }
    }
  }

  public onActivate(handler: (plugin: Plugin) => void): () => void {
    this.activateHandlers.push(handler)
    return () => {
      const index = this.activateHandlers.indexOf(handler)
      if (index > -1) {
        this.activateHandlers.splice(index, 1)
      }
    }
  }

  public onError(handler: (pluginId: string, error: Error) => void): () => void {
    this.errorHandlers.push(handler)
    return () => {
      const index = this.errorHandlers.indexOf(handler)
      if (index > -1) {
        this.errorHandlers.splice(index, 1)
      }
    }
  }

  private notifyInstallHandlers(plugin: Plugin): void {
    this.installHandlers.forEach(handler => handler(plugin))
  }

  private notifyActivateHandlers(plugin: Plugin): void {
    this.activateHandlers.forEach(handler => handler(plugin))
  }

  private notifyErrorHandlers(pluginId: string, error: Error): void {
    this.errorHandlers.forEach(handler => handler(pluginId, error))
  }

  // ===== GETTERS =====

  public getPlugins(): Plugin[] {
    return Array.from(this.plugins.values())
  }

  public getPlugin(id: string): Plugin | undefined {
    return this.plugins.get(id)
  }

  public getActivePlugins(): Plugin[] {
    return Array.from(this.plugins.values()).filter(plugin => plugin.status === 'active')
  }

  public getPluginsByCategory(category: Plugin['category']): Plugin[] {
    return Array.from(this.plugins.values()).filter(plugin => plugin.category === category)
  }

  public getPluginRuntime(id: string): PluginRuntime | undefined {
    return this.runtimes.get(id)
  }
}
