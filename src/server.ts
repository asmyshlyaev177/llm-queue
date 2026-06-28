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
  /**
   * Starting context window (tokens). The service runs at the HIGH-WATER max any
   * client requests and never drops back, so the model loads at most once; this
   * is the floor it negotiates up from. Not a queue concern — the queue moves
   * raw strings — so it lives here, not in QueueOptions.
   */
  numCtx?: number
}

interface CompletionsBody extends SamplingParams {
  model?: string
  messages?: Array<{ role?: string; content?: unknown }>
  stream?: boolean
  response_format?: { type?: string }
  /**
   * llm-queue extension fields. Stock OpenAI clients omit them (and any OpenAI
   * server ignores them); ours read them straight off the body — the queue's
   * priority and the shared context window have no OpenAI-standard equivalent.
   */
  priority?: boolean
  /** Context window this client wants; the service runs at the max seen. */
  numCtx?: number
}

/**
 * An HTTP service wrapping one llm-queue around one transport. All requests from
 * all clients funnel through a single serialized queue, so the local model is
 * never hit concurrently no matter how many processes connect.
 *
 *   POST /v1/chat/completions {model, messages, stream?, response_format?,
 *                              priority?, numCtx?}  (OpenAI-compatible + extensions)
 *   GET  /v1/models           -> OpenAI model list
 *   GET  /health              -> "ok"
 *
 * `priority` and `numCtx` are llm-queue body extensions (see CompletionsBody);
 * stock OpenAI clients omit them and any OpenAI server ignores them. Zero deps
 * (node:http). Clients are plain HTTP — point any OpenAI client at `<url>/v1`, or
 * POST the shape above directly (see the README).
 */
export function createLlmServer(opts: LlmServerOptions) {
  const {
    transport,
    port = 11500,
    host = '127.0.0.1',
    model: modelName = 'local',
    numCtx = 0,
    ...queueOpts
  } = opts
  const queue = createLlmQueue(transport, queueOpts)
  const log = queueOpts.logger ?? (() => {})

  // The model reloads in Ollama whenever num_ctx changes, so different clients
  // disagreeing on context size would thrash the runner. We run at the HIGH-WATER
  // max instead: once a client asks for a larger window we keep it there and
  // never drop back, so the model loads up at most once and then stays put.
  let effectiveNumCtx = numCtx
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
    'access-control-allow-headers': 'content-type, authorization',
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

      // Standard OpenAI clients can't express the queue's priority/numCtx, so we
      // read them as extension body fields (ignored by stock OpenAI servers).
      const wantJson = body.response_format?.type === 'json_object'
      const priority = body.priority === true
      const numCtx = negotiateNumCtx(typeof body.numCtx === 'number' ? body.numCtx : undefined) || undefined
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

      const id = `chatcmpl-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
      const created = Math.floor(Date.now() / 1000)
      const chatOpts = { numCtx, json: wantJson, model: requestedModel, params }

      // Streaming: relay the queue's token deltas as OpenAI SSE chunks. The queue
      // still serializes — the stream holds the single worker until it completes.
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
        try {
          for await (const piece of queue.chatMessagesStream(messages, priority, chatOpts)) {
            send([{ index: 0, delta: { content: piece }, finish_reason: null }])
          }
        } catch (err) {
          // Headers are already sent, so we can't switch to a 500. Log and close
          // the SSE cleanly — the client treats [DONE] as end-of-stream and keeps
          // whatever partial content already arrived.
          log(`stream error: ${err instanceof Error ? err.message : String(err)}`, 'error')
        }
        send([{ index: 0, delta: {}, finish_reason: 'stop' }])
        res.write('data: [DONE]\n\n')
        res.end()
        return
      }

      // Non-streaming: one whole completion.
      let content: string
      try {
        content = await queue.chatMessages(messages, priority, chatOpts)
      } catch (err) {
        sendJson(res, 500, {
          error: { message: err instanceof Error ? err.message : String(err), type: 'internal_error' },
        })
        return
      }

      const promptChars = messages.reduce((n, m) => n + m.content.length, 0)
      const usage = {
        prompt_tokens: Math.ceil(promptChars / 4),
        completion_tokens: Math.ceil(content.length / 4),
        total_tokens: Math.ceil((promptChars + content.length) / 4),
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
