// MasterX Service Worker for Advanced Caching and Offline Support

const CACHE_NAME = 'masterx-v1'
const STATIC_CACHE_NAME = 'masterx-static-v1'
const DYNAMIC_CACHE_NAME = 'masterx-dynamic-v1'

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/chat',
  '/manifest.json',
  // Add other critical assets
]

// API endpoints to cache
const CACHEABLE_APIS = [
  '/api/chat/sessions',
  '/api/user/profile',
  '/api/system/health',
]

// ===== INSTALLATION =====

self.addEventListener('install', (event) => {
  console.log('🔧 Service Worker installing...')
  
  event.waitUntil(
    caches.open(STATIC_CACHE_NAME)
      .then((cache) => {
        console.log('📦 Caching static assets')
        return cache.addAll(STATIC_ASSETS)
      })
      .then(() => {
        console.log('✅ Service Worker installed successfully')
        return self.skipWaiting()
      })
      .catch((error) => {
        console.error('❌ Service Worker installation failed:', error)
      })
  )
})

// ===== ACTIVATION =====

self.addEventListener('activate', (event) => {
  console.log('🚀 Service Worker activating...')
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            // Delete old caches
            if (cacheName !== STATIC_CACHE_NAME && 
                cacheName !== DYNAMIC_CACHE_NAME &&
                cacheName !== CACHE_NAME) {
              console.log('🗑️ Deleting old cache:', cacheName)
              return caches.delete(cacheName)
            }
          })
        )
      })
      .then(() => {
        console.log('✅ Service Worker activated')
        return self.clients.claim()
      })
  )
})

// ===== FETCH HANDLING =====

self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return
  }
  
  // Handle different types of requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleApiRequest(request))
  } else if (url.pathname.startsWith('/_next/static/')) {
    event.respondWith(handleStaticAssets(request))
  } else {
    event.respondWith(handlePageRequest(request))
  }
})

// ===== REQUEST HANDLERS =====

// Handle API requests with intelligent caching
async function handleApiRequest(request) {
  const url = new URL(request.url)
  const isCacheable = CACHEABLE_APIs.some(api => url.pathname.startsWith(api))
  
  if (!isCacheable) {
    // For non-cacheable APIs, try network first
    try {
      const response = await fetch(request)
      return response
    } catch (error) {
      console.warn('API request failed:', error)
      return new Response(
        JSON.stringify({ error: 'Network unavailable' }),
        { 
          status: 503,
          headers: { 'Content-Type': 'application/json' }
        }
      )
    }
  }
  
  // For cacheable APIs, use cache-first strategy with background update
  try {
    const cache = await caches.open(DYNAMIC_CACHE_NAME)
    const cachedResponse = await cache.match(request)
    
    // Background fetch to update cache
    const fetchPromise = fetch(request)
      .then((response) => {
        if (response.ok) {
          cache.put(request, response.clone())
        }
        return response
      })
      .catch(() => null)
    
    // Return cached response immediately if available
    if (cachedResponse) {
      // Update cache in background
      fetchPromise
      return cachedResponse
    }
    
    // If no cache, wait for network
    const networkResponse = await fetchPromise
    if (networkResponse) {
      return networkResponse
    }
    
    // Fallback response
    return new Response(
      JSON.stringify({ error: 'Data unavailable offline' }),
      { 
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    )
  } catch (error) {
    console.error('Cache handling failed:', error)
    return new Response(
      JSON.stringify({ error: 'Cache error' }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }
}

// Handle static assets with cache-first strategy
async function handleStaticAssets(request) {
  try {
    const cache = await caches.open(STATIC_CACHE_NAME)
    const cachedResponse = await cache.match(request)
    
    if (cachedResponse) {
      return cachedResponse
    }
    
    // If not in cache, fetch and cache
    const response = await fetch(request)
    if (response.ok) {
      cache.put(request, response.clone())
    }
    
    return response
  } catch (error) {
    console.error('Static asset handling failed:', error)
    return fetch(request)
  }
}

// Handle page requests with network-first strategy
async function handlePageRequest(request) {
  try {
    // Try network first
    const response = await fetch(request)
    
    // Cache successful responses
    if (response.ok) {
      const cache = await caches.open(DYNAMIC_CACHE_NAME)
      cache.put(request, response.clone())
    }
    
    return response
  } catch (error) {
    // Fallback to cache
    const cache = await caches.open(DYNAMIC_CACHE_NAME)
    const cachedResponse = await cache.match(request)
    
    if (cachedResponse) {
      return cachedResponse
    }
    
    // Ultimate fallback - offline page
    return new Response(
      `
      <!DOCTYPE html>
      <html>
        <head>
          <title>MasterX - Offline</title>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body { 
              font-family: system-ui, sans-serif; 
              text-align: center; 
              padding: 2rem;
              background: #f8fafc;
            }
            .container {
              max-width: 400px;
              margin: 0 auto;
              padding: 2rem;
              background: white;
              border-radius: 1rem;
              box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .icon {
              width: 64px;
              height: 64px;
              margin: 0 auto 1rem;
              background: #e2e8f0;
              border-radius: 50%;
              display: flex;
              align-items: center;
              justify-content: center;
              font-size: 24px;
            }
            h1 { color: #1e293b; margin-bottom: 0.5rem; }
            p { color: #64748b; margin-bottom: 1.5rem; }
            button {
              background: #6366f1;
              color: white;
              border: none;
              padding: 0.75rem 1.5rem;
              border-radius: 0.5rem;
              cursor: pointer;
              font-size: 1rem;
            }
            button:hover { background: #5855eb; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="icon">📴</div>
            <h1>You're Offline</h1>
            <p>Please check your internet connection and try again.</p>
            <button onclick="window.location.reload()">Retry</button>
          </div>
        </body>
      </html>
      `,
      {
        headers: { 'Content-Type': 'text/html' }
      }
    )
  }
}

// ===== BACKGROUND SYNC =====

self.addEventListener('sync', (event) => {
  console.log('🔄 Background sync triggered:', event.tag)
  
  if (event.tag === 'background-sync-messages') {
    event.waitUntil(syncOfflineMessages())
  }
})

// Sync offline messages when connection is restored
async function syncOfflineMessages() {
  try {
    console.log('📤 Syncing offline messages...')
    
    // Get offline queue from IndexedDB or localStorage
    const offlineQueue = await getOfflineQueue()
    
    if (offlineQueue.length === 0) {
      console.log('✅ No offline messages to sync')
      return
    }
    
    // Process each queued message
    for (const item of offlineQueue) {
      try {
        const response = await fetch('/api/chat/send', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            content: item.content,
            session_id: item.sessionId,
          }),
        })
        
        if (response.ok) {
          console.log('✅ Synced offline message:', item.content.substring(0, 50))
        } else {
          console.warn('⚠️ Failed to sync message:', response.status)
        }
      } catch (error) {
        console.error('❌ Error syncing message:', error)
      }
    }
    
    // Clear offline queue after successful sync
    await clearOfflineQueue()
    
    // Notify clients about successful sync
    const clients = await self.clients.matchAll()
    clients.forEach(client => {
      client.postMessage({
        type: 'OFFLINE_SYNC_COMPLETE',
        data: { syncedCount: offlineQueue.length }
      })
    })
    
    console.log(`✅ Successfully synced ${offlineQueue.length} offline messages`)
  } catch (error) {
    console.error('❌ Background sync failed:', error)
  }
}

// ===== UTILITY FUNCTIONS =====

async function getOfflineQueue() {
  // In a real implementation, you'd use IndexedDB
  // For now, return empty array as localStorage is not available in SW
  return []
}

async function clearOfflineQueue() {
  // Clear the offline queue after successful sync
  console.log('🗑️ Clearing offline queue')
}

// ===== MESSAGE HANDLING =====

self.addEventListener('message', (event) => {
  const { type, data } = event.data
  
  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting()
      break
      
    case 'CACHE_STATS':
      getCacheStats().then(stats => {
        event.ports[0].postMessage(stats)
      })
      break
      
    case 'CLEAR_CACHE':
      clearAllCaches().then(() => {
        event.ports[0].postMessage({ success: true })
      })
      break
      
    default:
      console.log('Unknown message type:', type)
  }
})

// Get cache statistics
async function getCacheStats() {
  const cacheNames = await caches.keys()
  const stats = {}
  
  for (const cacheName of cacheNames) {
    const cache = await caches.open(cacheName)
    const keys = await cache.keys()
    stats[cacheName] = keys.length
  }
  
  return stats
}

// Clear all caches
async function clearAllCaches() {
  const cacheNames = await caches.keys()
  await Promise.all(cacheNames.map(name => caches.delete(name)))
  console.log('🗑️ All caches cleared')
}

console.log('🚀 MasterX Service Worker loaded')
