import { toOllamaOptions, type ChatMessage, type ChatOptions, type ChatTransport } from './core.js'

export interface FetchTransportConfig {
  /** Base URL of the Ollama server, e.g. http://localhost:11434 */
  url: string
  /** Default model; per-call `ChatOptions.model` overrides it. */
  model: string
  /** Ollama context window (num_ctx). */
  numCtx?: number
}

/**
 * Browser/service-worker-safe transport: talks to a local Ollama server over
 * plain `fetch` (point `url` at any Ollama-compatible host). No Node
 * dependencies. In an MV3 extension this must run in the service worker
 * (host_permissions bypass page CORS).
 */
export function createFetchTransport(config: FetchTransportConfig): ChatTransport {
  async function chatRaw(messages: ChatMessage[], opts: ChatOptions = {}): Promise<string> {
    const effectiveNumCtx = opts.numCtx ?? config.numCtx
    const model = opts.model || config.model

    const res = await fetch(`${config.url}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        format: opts.json ? 'json' : undefined,
        stream: false,
        options: toOllamaOptions(effectiveNumCtx, opts.params),
        messages,
      }),
    })
    if (!res.ok) throw new Error(`ollama ${res.status}: ${await res.text()}`)
    const data = (await res.json()) as { message?: { content?: string } }
    return data.message?.content?.trim() ?? ''
  }

  // Token-by-token streaming over Ollama's NDJSON (`stream: true` on /api/chat):
  // one JSON object per line, each carrying a `message.content` delta. Yields the
  // non-empty deltas as they arrive. Runs to completion so the queue frees its slot.
  async function* chatStream(messages: ChatMessage[], opts: ChatOptions = {}): AsyncGenerator<string> {
    const effectiveNumCtx = opts.numCtx ?? config.numCtx
    const model = opts.model || config.model

    const res = await fetch(`${config.url}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        format: opts.json ? 'json' : undefined,
        stream: true,
        options: toOllamaOptions(effectiveNumCtx, opts.params),
        messages,
      }),
    })
    if (!res.ok || !res.body) throw new Error(`ollama ${res.status}: ${await res.text()}`)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    const emit = function* (chunk: string): Generator<string> {
      buf += chunk
      let nl: number
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim()
        buf = buf.slice(nl + 1)
        if (!line) continue
        const piece = (JSON.parse(line) as { message?: { content?: string } }).message?.content
        if (piece) yield piece
      }
    }
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      yield* emit(decoder.decode(value, { stream: true }))
    }
    // Flush a trailing line with no newline terminator.
    const tail = buf.trim()
    if (tail) {
      const piece = (JSON.parse(tail) as { message?: { content?: string } }).message?.content
      if (piece) yield piece
    }
  }

  return {
    chatRaw,
    chatStream,
    // Plain (system, user) chat. JSON mode is opt-in via chatRaw/ChatOptions —
    // callers that want to parse JSON pass `json: true` themselves.
    chat(systemPrompt: string, userContent: string, numCtx?: number): Promise<string> {
      return chatRaw(
        [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent },
        ],
        { numCtx },
      )
    },
  }
}
