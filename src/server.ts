import http from 'node:http'
import { createLlmQueue, type ChatTransport, type QueueOptions } from './core.js'

export interface LlmServerOptions extends QueueOptions {
  transport: ChatTransport
  port?: number
  host?: string
}

interface ChatBody {
  system?: string
  user?: string
  priority?: boolean
}

/**
 * An HTTP service wrapping one llm-queue around one transport. All requests from
 * all clients funnel through a single serialized queue, so the local model is
 * never hit concurrently no matter how many processes connect.
 *
 *   POST /chat   {system, user, priority?}  ->  {content} | {error}
 *   GET  /health ->  "ok"
 *
 * Zero deps (node:http). Pair with createServiceTransport (./client) on clients.
 */
export function createLlmServer(opts: LlmServerOptions) {
  const { transport, port = 11500, host = '127.0.0.1', ...queueOpts } = opts
  const queue = createLlmQueue(transport, queueOpts)

  // Allow any origin so browser-context clients (an MV3 service worker, a
  // vitest/jsdom test) can read responses — same role as Ollama's OLLAMA_ORIGINS.
  const CORS = {
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'POST, GET, OPTIONS',
    'access-control-allow-headers': 'content-type',
  }

  const server = http.createServer((req, res) => {
    if (req.method === 'OPTIONS') {
      res.writeHead(204, CORS)
      res.end()
      return
    }
    if (req.method === 'GET' && req.url === '/health') {
      res.writeHead(200, { ...CORS, 'content-type': 'text/plain' })
      res.end('ok')
      return
    }
    if (req.method === 'POST' && req.url === '/chat') {
      let body = ''
      req.on('data', (c) => (body += c))
      req.on('end', () => {
        let parsed: ChatBody
        try {
          parsed = JSON.parse(body) as ChatBody
        } catch {
          res.writeHead(400, { ...CORS, 'content-type': 'application/json' })
          res.end(JSON.stringify({ error: 'invalid JSON body' }))
          return
        }
        queue
          .chat(parsed.system ?? '', parsed.user ?? '', parsed.priority ?? false)
          .then((content) => {
            res.writeHead(200, { ...CORS, 'content-type': 'application/json' })
            res.end(JSON.stringify({ content }))
          })
          .catch((err: unknown) => {
            res.writeHead(500, { ...CORS, 'content-type': 'application/json' })
            res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }))
          })
      })
      return
    }
    res.writeHead(404, { ...CORS, 'content-type': 'application/json' })
    res.end(JSON.stringify({ error: 'not found' }))
  })

  return {
    server,
    queue,
    listen: (): Promise<void> =>
      new Promise((resolve) => server.listen(port, host, () => resolve())),
    close: (): Promise<void> =>
      new Promise((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      ),
  }
}
