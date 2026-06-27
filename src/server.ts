import http from 'node:http'
import {
  createLlmQueue,
  type ChatMessage,
  type ChatTransport,
  type QueueOptions,
  type SamplingParams,
} from './core.js'

export interface LlmServerOptions extends QueueOptions {
  transport: ChatTransport
  port?: number
  host?: string
  /** Model name reported by GET /v1/models and echoed back in completions. */
  model?: string
}

interface ChatBody {
  system?: string
  user?: string
  priority?: boolean
  /** Context window this client wants; the service runs at the max seen. */
  numCtx?: number
  /** Ask the backend to constrain output to JSON. Off by default. */
  json?: boolean
  /** Override the model for this call (default: the service's configured model). */
  model?: string
  /** Sampling params (temperature, max_tokens, …) forwarded to the backend. */
  params?: SamplingParams
}

interface CompletionsBody extends SamplingParams {
  model?: string
  messages?: Array<{ role?: string; content?: unknown }>
  stream?: boolean
  response_format?: { type?: string }
}

/**
 * An HTTP service wrapping one llm-queue around one transport. All requests from
 * all clients funnel through a single serialized queue, so the local model is
 * never hit concurrently no matter how many processes connect.
 *
 *   POST /chat                {system, user, priority?, numCtx?, json?} -> {content} | {error}
 *   POST /v1/chat/completions {model, messages, stream?, response_format?}  (OpenAI-compatible)
 *   GET  /v1/models           -> OpenAI model list
 *   GET  /health              -> "ok"
 *
 * Zero deps (node:http). Pair with createServiceTransport (./client) on clients,
 * or point any OpenAI client at `<url>/v1`.
 */
export function createLlmServer(opts: LlmServerOptions) {
  const { transport, port = 11500, host = '127.0.0.1', model: modelName = 'local', ...queueOpts } = opts
  const queue = createLlmQueue(transport, queueOpts)
  const log = queueOpts.logger ?? (() => {})

  // The model reloads in Ollama whenever num_ctx changes, so different clients
  // disagreeing on context size would thrash the runner. We run at the HIGH-WATER
  // max instead: once a client asks for a larger window we keep it there and
  // never drop back, so the model loads up at most once and then stays put.
  let effectiveNumCtx = queueOpts.numCtx ?? 0
  function negotiateNumCtx(requested?: number): number {
    if (typeof requested === 'number' && requested > effectiveNumCtx) {
      log(`raising shared num_ctx ${effectiveNumCtx} → ${requested}`, 'info')
      effectiveNumCtx = requested
    }
    return effectiveNumCtx
  }

  // Allow any origin so browser-context clients (an MV3 service worker, a
  // vitest/jsdom test) can read responses — same role as Ollama's OLLAMA_ORIGINS.
  // `authorization` is allowed so OpenAI SDKs (which always send a bearer token)
  // pass preflight; the service doesn't require it.
  const CORS = {
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'POST, GET, OPTIONS',
    'access-control-allow-headers': 'content-type, authorization, x-llmq-priority, x-llmq-num-ctx',
  }

  const sendJson = (res: http.ServerResponse, status: number, body: unknown): void => {
    res.writeHead(status, { ...CORS, 'content-type': 'application/json' })
    res.end(JSON.stringify(body))
  }

  const readBody = (req: http.IncomingMessage): Promise<string> =>
    new Promise((resolve) => {
      let body = ''
      req.on('data', (c) => (body += c))
      req.on('end', () => resolve(body))
    })

  const server = http.createServer(async (req, res) => {
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

    // OpenAI model list — many clients probe this on startup. Single model.
    if (req.method === 'GET' && (req.url === '/v1/models' || req.url === '/models')) {
      sendJson(res, 200, {
        object: 'list',
        data: [{ id: modelName, object: 'model', created: 0, owned_by: 'llm-queue' }],
      })
      return
    }

    // Native {system, user} API. JSON mode is now opt-in (`json: true`).
    if (req.method === 'POST' && req.url === '/chat') {
      let parsed: ChatBody
      try {
        parsed = JSON.parse(await readBody(req)) as ChatBody
      } catch {
        sendJson(res, 400, { error: 'invalid JSON body' })
        return
      }
      try {
        const content = await queue.chatMessages(
          [
            { role: 'system', content: parsed.system ?? '' },
            { role: 'user', content: parsed.user ?? '' },
          ],
          parsed.priority ?? false,
          {
            numCtx: negotiateNumCtx(parsed.numCtx) || undefined,
            json: parsed.json ?? false,
            model: parsed.model,
            params: parsed.params,
          },
        )
        sendJson(res, 200, { content })
      } catch (err) {
        sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) })
      }
      return
    }

    // OpenAI-compatible chat completions. Funnels through the same single queue.
    if (req.method === 'POST' && (req.url === '/v1/chat/completions' || req.url === '/chat/completions')) {
      let body: CompletionsBody
      try {
        body = JSON.parse(await readBody(req)) as CompletionsBody
      } catch {
        sendJson(res, 400, { error: { message: 'invalid JSON body', type: 'invalid_request_error' } })
        return
      }

      const messages: ChatMessage[] = Array.isArray(body.messages)
        ? body.messages
            .filter((m) => m && typeof m.content === 'string')
            .map((m) => ({ role: (m.role as ChatMessage['role']) || 'user', content: m.content as string }))
        : []
      if (messages.length === 0) {
        sendJson(res, 400, {
          error: { message: 'messages must be a non-empty array of {role, content}', type: 'invalid_request_error' },
        })
        return
      }

      // Standard OpenAI clients can't express the queue's priority/num_ctx, so
      // accept them as optional non-standard headers (ignored by normal clients).
      const wantJson = body.response_format?.type === 'json_object'
      const priority = req.headers['x-llmq-priority'] === '1' || req.headers['x-llmq-priority'] === 'true'
      const hdrNumCtx = Number(req.headers['x-llmq-num-ctx'])
      const numCtx = negotiateNumCtx(Number.isFinite(hdrNumCtx) ? hdrNumCtx : undefined) || undefined
      // Respect the requested model; fall back to the service's configured one.
      const requestedModel = typeof body.model === 'string' && body.model.trim() ? body.model : undefined
      const model = requestedModel ?? modelName // echoed back to the client
      const params: SamplingParams = {
        temperature: body.temperature,
        top_p: body.top_p,
        max_tokens: body.max_tokens,
        frequency_penalty: body.frequency_penalty,
        presence_penalty: body.presence_penalty,
        stop: body.stop,
        seed: body.seed,
      }

      let content: string
      try {
        content = await queue.chatMessages(messages, priority, {
          numCtx,
          json: wantJson,
          model: requestedModel,
          params,
        })
      } catch (err) {
        sendJson(res, 500, {
          error: { message: err instanceof Error ? err.message : String(err), type: 'internal_error' },
        })
        return
      }

      const id = `chatcmpl-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
      const created = Math.floor(Date.now() / 1000)
      const promptChars = messages.reduce((n, m) => n + m.content.length, 0)
      const usage = {
        prompt_tokens: Math.ceil(promptChars / 4),
        completion_tokens: Math.ceil(content.length / 4),
        total_tokens: Math.ceil((promptChars + content.length) / 4),
      }

      // The queue resolves the whole completion at once, so streaming is emulated:
      // emit the full content as a single delta. Protocol-compatible with every
      // OpenAI client (which is what matters); not true token-by-token streaming.
      if (body.stream) {
        res.writeHead(200, {
          ...CORS,
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
        })
        const base = { id, object: 'chat.completion.chunk', created, model }
        const send = (choices: unknown): void => {
          res.write(`data: ${JSON.stringify({ ...base, choices })}\n\n`)
        }
        send([{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }])
        send([{ index: 0, delta: { content }, finish_reason: null }])
        send([{ index: 0, delta: {}, finish_reason: 'stop' }])
        res.write('data: [DONE]\n\n')
        res.end()
        return
      }

      sendJson(res, 200, {
        id,
        object: 'chat.completion',
        created,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
        usage,
      })
      return
    }

    sendJson(res, 404, { error: 'not found' })
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
