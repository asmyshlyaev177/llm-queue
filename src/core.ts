import { jsonrepair } from 'jsonrepair'

/**
 * A chat transport hits the model once and returns the raw text response.
 * Implementations live in ./browser (fetch) and ./node (ollama pkg).
 */
export interface ChatTransport {
  chat(systemPrompt: string, userContent: string): Promise<string>
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

  async function chatWithRetry(systemPrompt: string, userContent: string): Promise<string> {
    for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
      let timer: ReturnType<typeof setTimeout>
      const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`LLM timeout after ${opts.timeoutMs}ms`)),
          opts.timeoutMs,
        )
      })
      try {
        return await Promise.race([transport.chat(systemPrompt, userContent), timeout])
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

  function truncate(systemPrompt: string, userContent: string): string {
    const sysTokens = Math.ceil(systemPrompt.length / CHARS_PER_TOKEN)
    const budget = opts.numCtx - sysTokens - RESPONSE_TOKEN_BUDGET
    const ctxCap = Math.max(0, budget * CHARS_PER_TOKEN)
    const cap = Math.min(opts.maxChars, ctxCap)
    return userContent.length > cap ? userContent.slice(0, cap) : userContent
  }

  /**
   * Run one classification through the queue. `parse` maps the repaired JSON
   * object to your typed result. Returns `null` on a non-fatal failure (network,
   * timeout, unparseable output); throws only on fatal model errors.
   */
  /**
   * Serialized raw chat: one model request at a time, with timeout + retry.
   * `priority` jumps the queue ahead of normal items. Returns the model's raw
   * string output (no parsing). The service (./server) exposes this over HTTP.
   */
  function chat(systemPrompt: string, userContent: string, priority = false): Promise<string> {
    return enqueue(() => chatWithRetry(systemPrompt, userContent), priority)
  }

  async function classify<T>(
    name: string,
    systemPrompt: string,
    userContent: string,
    parse: (parsed: Record<string, unknown>) => T,
    priority = false,
  ): Promise<T | null> {
    const truncated = truncate(systemPrompt, userContent)
    try {
      const raw = await chat(systemPrompt, truncated, priority)
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

  return { classify, chat, enqueue }
}

export type LlmQueue = ReturnType<typeof createLlmQueue>

/** Coerce a possibly-stringified boolean field from model JSON. */
export function boolField(parsed: Record<string, unknown>, key: string): boolean {
  return parsed[key] === true || String(parsed[key]).toLowerCase() === 'true'
}
