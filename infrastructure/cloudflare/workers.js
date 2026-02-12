/**
 * Cloudflare Workers - Edge deployment for low-latency inference
 * 
 * Provides:
 * - Global edge network (200+ locations)
 * - Sub-10ms latency worldwide
 * - Automatic HTTPS and DDoS protection
 * - WebSocket support
 */

// Environment bindings
// BACKEND_API: KV namespace for model storage
// API_ENDPOINT: Origin API endpoint

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url)
    
    // CORS handling
    if (request.method === 'OPTIONS') {
      return handleCORS()
    }
    
    // Health check at edge
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'healthy',
        edge: true,
        location: request.cf?.colo || 'unknown',
        timestamp: new Date().toISOString()
      }), {
        headers: {
          'Content-Type': 'application/json',
          ...getCORSHeaders()
        }
      })
    }
    
    // WebSocket upgrade
    if (request.headers.get('Upgrade') === 'websocket') {
      return handleWebSocket(request, env)
    }
    
    // Cache GET requests
    if (request.method === 'GET') {
      const cache = caches.default
      let response = await cache.match(request)
      
      if (!response) {
        response = await fetch(request)
        // Cache for 1 hour
        response = new Response(response.body, response)
        response.headers.set('Cache-Control', 'public, max-age=3600')
        ctx.waitUntil(cache.put(request, response.clone()))
      }
      
      return addCORSHeaders(response)
    }
    
    // Proxy POST requests to origin
    if (url.pathname.startsWith('/api/')) {
      return proxyToOrigin(request, env)
    }
    
    // Default: serve static files from R2
    return serveStatic(request, env)
  }
}

function handleCORS() {
  return new Response(null, {
    headers: getCORSHeaders()
  })
}

function getCORSHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400'
  }
}

function addCORSHeaders(response) {
  const newResponse = new Response(response.body, response)
  Object.entries(getCORSHeaders()).forEach(([key, value]) => {
    newResponse.headers.set(key, value)
  })
  return newResponse
}

async function proxyToOrigin(request, env) {
  const url = new URL(request.url)
  url.hostname = env.API_ENDPOINT
  url.protocol = 'https:'
  
  const newRequest = new Request(url, request)
  
  try {
    const response = await fetch(newRequest)
    return addCORSHeaders(response)
  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Backend unavailable',
      message: error.message
    }), {
      status: 503,
      headers: {
        'Content-Type': 'application/json',
        ...getCORSHeaders()
      }
    })
  }
}

async function handleWebSocket(request, env) {
  const upgradeHeader = request.headers.get('Upgrade')
  if (!upgradeHeader || upgradeHeader !== 'websocket') {
    return new Response('Expected Upgrade: websocket', { status: 426 })
  }
  
  // Create WebSocket pair
  const webSocketPair = new WebSocketPair()
  const [client, server] = Object.values(webSocketPair)
  
  // Handle WebSocket messages
  server.accept()
  
  server.addEventListener('message', async (event) => {
    try {
      const data = JSON.parse(event.data)
      
      // Simple echo for now - replace with actual inference
      server.send(JSON.stringify({
        type: 'frame',
        data: data,
        timestamp: Date.now()
      }))
    } catch (error) {
      server.send(JSON.stringify({
        type: 'error',
        message: error.message
      }))
    }
  })
  
  return new Response(null, {
    status: 101,
    webSocket: client
  })
}

async function serveStatic(request, env) {
  // Serve from R2 or KV
  const url = new URL(request.url)
  const objectKey = url.pathname.slice(1) || 'index.html'
  
  try {
    const object = await env.STATIC_ASSETS.get(objectKey)
    
    if (!object) {
      return new Response('Not found', { status: 404 })
    }
    
    return new Response(object.body, {
      headers: {
        'Content-Type': getContentType(objectKey),
        'Cache-Control': 'public, max-age=3600'
      }
    })
  } catch (error) {
    return new Response('Error loading asset', { status: 500 })
  }
}

function getContentType(path) {
  const ext = path.split('.').pop()
  const types = {
    'html': 'text/html',
    'css': 'text/css',
    'js': 'application/javascript',
    'json': 'application/json',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'svg': 'image/svg+xml',
    'woff2': 'font/woff2'
  }
  return types[ext] || 'application/octet-stream'
}
