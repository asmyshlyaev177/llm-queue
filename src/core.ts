import { jsonrepair } from 'jsonrepair'

export type ChatRole = 'system' | 'user' | 'assistant'

export interface ChatMessage {
  role: ChatRole
  content: string
}

/** OpenAI-style sampling parameters, forwarded to the backend. */
export interface SamplingParams {
  temperature?: number
  top_p?: number
  /** OpenAI name; mapped to Ollama's `num_predict`. */
  max_tokens?: number
  frequency_penalty?: number
  presence_penalty?: number
  stop?: string | string[]
  seed?: number
}

/** Per-call knobs shared by the transports. */
export interface ChatOptions {
  /** Override the context window for this one call (the service's high-water max). */
  numCtx?: number
  /**
   * Constrain output to JSON (Ollama `format`, llama.cpp `response_format`). The
   * classify path sets this; general OpenAI-style chat does not, unless the
   * client asks for `response_format: { type: 'json_object' }`.
   */
  json?: boolean
  /** Override the backend model for this call (default: the transport's model). */
  model?: string
  /** Sampling parameters (temperature, max_tokens, …) forwarded to the backend. */
  params?: SamplingParams
}

/**
 * A chat transport hits the model once and returns the raw text response.
 * Implementations live in ./browser (fetch) and ./node (ollama pkg).
 *
 * `numCtx` optionally overrides the transport's configured context window for
 * this one call. The service uses it to keep every client on a single shared
 * context size (the high-water max) so the local model never reloads.
 */
export interface ChatTransport {
  chat(systemPrompt: string, userContent: string, numCtx?: number): Promise<string>
  /**
   * Native multi-message path for OpenAI-style requests. Optional: when a
   * transport doesn't implement it, the queue collapses the messages back into
   * the `chat(system, user)` signature (see {@link collapseMessages}).
   */
  chatRaw?(messages: ChatMessage[], opts?: ChatOptions): Promise<string>
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'
export type Logger = (msg: string, level?: LogLevel) => void

export interface QueueOptions {
  /** Model context window in tokens, used to size input truncation. */
  numCtx?: number
  /** Hard cap on user-content chars regardless of context budget. */
  maxChars?: number
  /** Per-attempt timeout. */
  timeoutMs?: number
  /** Attempts per classify call (incl. the first). */
  maxAttempts?: number
  logger?: Logger
}

const DEFAULTS = {
  numCtx: 8192,
  maxChars: 8000,
  timeoutMs: 60000,
  maxAttempts: 2,
} satisfies Required<Omit<QueueOptions, 'logger'>>

// Rough token estimate; 4 chars/token is conservative for English/code mixes.
const CHARS_PER_TOKEN = 4
const RESPONSE_TOKEN_BUDGET = 256

const FATAL_MODEL_RE = /unable to load model|model not found|no such file|pull model/i

interface QueueItem {
  run: () => Promise<string>
  resolve: (v: string) => void
  reject: (e: unknown) => void
  priority: boolean
}

/**
 * Collapse a messages array into the `(system, user)` pair the base transport
 * takes — the fallback when a transport has no native `chatRaw`. System messages
 * join into the system prompt; the remaining turns render as a simple transcript
 * (lossless for the common single-user-turn case).
 */
export function collapseMessages(messages: ChatMessage[]): { system: string; user: string } {
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => m.content)
    .join('\n\n')
  const rest = messages.filter((m) => m.role !== 'system')
  const user =
    rest.length <= 1
      ? (rest[0]?.content ?? '')
      : rest.map((m) => `${m.role}: ${m.content}`).join('\n\n')
  return { system, user }
}

/**
 * Map generic sampling params (+ num_ctx) to an Ollama `options` object, used by
 * both the fetch and SDK transports. Returns undefined when nothing is set, so
 * callers can omit `options` entirely. `max_tokens` becomes Ollama's `num_predict`.
 */
export function toOllamaOptions(
  numCtx?: number,
  params?: SamplingParams,
): Record<string, unknown> | undefined {
  const o: Record<string, unknown> = {}
  if (numCtx) o.num_ctx = numCtx
  if (params?.temperature != null) o.temperature = params.temperature
  if (params?.top_p != null) o.top_p = params.top_p
  if (params?.max_tokens != null) o.num_predict = params.max_tokens
  if (params?.frequency_penalty != null) o.frequency_penalty = params.frequency_penalty
  if (params?.presence_penalty != null) o.presence_penalty = params.presence_penalty
  if (params?.stop != null) o.stop = params.stop
  if (params?.seed != null) o.seed = params.seed
  return Object.keys(o).length ? o : undefined
}

/**
 * Build a serialized (concurrency-1) priority queue around one transport, plus a
 * `classify()` helper that truncates input to the context window, retries with a
 * timeout, repairs malformed JSON, and maps it to a typed result.
 *
 * One instance == one logical model. Hits are never concurrent; `priority` items
 * jump ahead of queued normal items but never preempt the running request.
 */
export function createLlmQueue(transport: ChatTransport, options: QueueOptions = {}) {
  const opts = { ...DEFAULTS, ...options }
  const log: Logger = options.logger ?? (() => {})

  const queue: QueueItem[] = []
  let draining = false

  function enqueue(run: () => Promise<string>, priority: boolean): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const item: QueueItem = { run, resolve, reject, priority }
      if (priority) {
        const idx = queue.findIndex((q) => !q.priority)
        if (idx === -1) queue.push(item)
        else queue.splice(idx, 0, item)
      } else {
        queue.push(item)
      }
      void drain()
    })
  }

  async function drain(): Promise<void> {
    if (draining) return
    draining = true
    try {
      let item: QueueItem | undefined
      while ((item = queue.shift())) {
        try {
          item.resolve(await item.run())
        } catch (err) {
          item.reject(err)
        }
      }
    } finally {
      draining = false
    }
  }

  // Run one model call with a per-attempt timeout + retry. `call` performs the
  // single underlying request, so the retry/timeout policy is independent of
  // whether the input was (system, user) or a full messages array.
  async function withRetry(call: () => Promise<string>): Promise<string> {
    for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
      let timer: ReturnType<typeof setTimeout>
      const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`LLM timeout after ${opts.timeoutMs}ms`)),
          opts.timeoutMs,
        )
      })
      try {
        return await Promise.race([call(), timeout])
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        if (FATAL_MODEL_RE.test(msg)) throw err
        if (attempt < opts.maxAttempts) {
          log(`[llm] attempt ${attempt}/${opts.maxAttempts} failed (${msg}), retrying…`, 'warn')
        } else {
          throw err
        }
      } finally {
        clearTimeout(timer!)
      }
    }
    throw new Error('unreachable')
  }

  function truncate(systemPrompt: string, userContent: string, numCtx = opts.numCtx): string {
    const sysTokens = Math.ceil(systemPrompt.length / CHARS_PER_TOKEN)
    const budget = numCtx - sysTokens - RESPONSE_TOKEN_BUDGET
    const ctxCap = Math.max(0, budget * CHARS_PER_TOKEN)
    const cap = Math.min(opts.maxChars, ctxCap)
    return userContent.length > cap ? userContent.slice(0, cap) : userContent
  }

  /**
   * Serialized raw chat: one model request at a time, with timeout + retry.
   * `priority` jumps the queue ahead of normal items. Returns the model's raw
   * string output (no parsing). The service (./server) exposes this over HTTP.
   */
  function chat(
    systemPrompt: string,
    userContent: string,
    priority = false,
    numCtx?: number,
  ): Promise<string> {
    return enqueue(() => withRetry(() => transport.chat(systemPrompt, userContent, numCtx)), priority)
  }

  /**
   * Serialized chat over a full OpenAI-style messages array. Uses the transport's
   * native `chatRaw` when available, else collapses the messages through `chat`.
   * Same single-worker queue as `chat` — the OpenAI endpoint funnels through here.
   */
  function chatMessages(
    messages: ChatMessage[],
    priority = false,
    chatOpts: ChatOptions = {},
  ): Promise<string> {
    const run = transport.chatRaw
      ? () => transport.chatRaw!(messages, chatOpts)
      : () => {
          const { system, user } = collapseMessages(messages)
          return transport.chat(system, user, chatOpts.numCtx)
        }
    return enqueue(() => withRetry(run), priority)
  }

  /**
   * Run one classification through the queue. `parse` maps the repaired JSON
   * object to your typed result. Returns `null` on a non-fatal failure (network,
   * timeout, unparseable output); throws only on fatal model errors.
   */
  async function classify<T>(
    name: string,
    systemPrompt: string,
    userContent: string,
    parse: (parsed: Record<string, unknown>) => T,
    priority = false,
    numCtx?: number,
  ): Promise<T | null> {
    const truncated = truncate(systemPrompt, userContent, numCtx)
    try {
      // classify parses the output as JSON, so it asks the backend for JSON mode
      // explicitly (the library no longer forces JSON on every call).
      const raw = await chatMessages(
        [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: truncated },
        ],
        priority,
        { numCtx, json: true },
      )
      log(`[llm] ${name} → ${raw}`, 'debug')
      const parsed = JSON.parse(jsonrepair(raw)) as Record<string, unknown>
      return parse(parsed)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      if (FATAL_MODEL_RE.test(msg)) throw new Error(`[llm] Fatal: ${msg}`, { cause: err })
      log(`[llm] ${name} failed (${msg})`, 'error')
      return null
    }
  }

  return { classify, chat, chatMessages, enqueue }
}

export type LlmQueue = ReturnType<typeof createLlmQueue>

/** Coerce a possibly-stringified boolean field from model JSON. */
export function boolField(parsed: Record<string, unknown>, key: string): boolean {
  return parsed[key] === true || String(parsed[key]).toLowerCase() === 'true'
}
