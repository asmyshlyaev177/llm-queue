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
   * Constrain output to JSON (Ollama `format`). Off unless the client asks for it
   * (the OpenAI endpoint maps `response_format: { type: 'json_object' }` to this).
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
  /**
   * Token-by-token streaming path: yields content deltas as the model produces
   * them. Optional — when absent, the queue's stream method falls back to a
   * single chunk (the whole `chatRaw`/`chat` response). The async iterator must
   * run to completion (or throw) so the queue can release its single worker.
   */
  chatStream?(messages: ChatMessage[], opts?: ChatOptions): AsyncIterable<string>
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'
export type Logger = (msg: string, level?: LogLevel) => void

export interface QueueOptions {
  /** Per-attempt timeout (ms). */
  timeoutMs?: number
  /** Attempts per request, including the first (so `1` = no retry). */
  maxAttempts?: number
  logger?: Logger
}

const DEFAULTS = {
  timeoutMs: 60000,
  maxAttempts: 2,
} satisfies Required<Omit<QueueOptions, 'logger'>>

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
 * A minimal async channel: the producer calls `push`/`close`, the consumer
 * `for await`s `iterable`. No backpressure — pushes buffer until consumed, which
 * is fine for relaying a model's token stream (bounded by the response size).
 * `close(err)` makes the iterable throw `err` after draining buffered values.
 */
function createAsyncChannel<T>() {
  const buffer: T[] = []
  const waiters: Array<(r: IteratorResult<T>) => void> = []
  let closed = false
  let failure: unknown

  return {
    push(value: T): void {
      if (closed) return
      const waiter = waiters.shift()
      if (waiter) waiter({ value, done: false })
      else buffer.push(value)
    },
    close(err?: unknown): void {
      if (closed) return
      closed = true
      failure = err
      while (waiters.length) waiters.shift()!({ value: undefined as never, done: true })
    },
    iterable: (async function* (): AsyncGenerator<T> {
      while (true) {
        if (buffer.length) {
          yield buffer.shift()!
          continue
        }
        if (closed) {
          if (failure) throw failure
          return
        }
        const next = await new Promise<IteratorResult<T>>((resolve) => waiters.push(resolve))
        if (next.done) {
          if (failure) throw failure
          return
        }
        yield next.value
      }
    })(),
  }
}

/**
 * Build a serialized (concurrency-1) priority queue around one transport. Every
 * call runs one at a time with a per-attempt timeout + retry; `priority` items
 * jump ahead of queued normal items but never preempt the running request.
 *
 * One instance == one logical model. The queue moves raw strings — JSON parsing,
 * truncation, and result mapping are the caller's concern (see the consumers).
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
   * Streaming variant of {@link chatMessages}: yields content deltas as the model
   * produces them, still serialized through the single worker — the queue slot is
   * held for the whole stream, so other items wait. Falls back to a single chunk
   * (the entire response) when the transport has no `chatStream`. No retry: a
   * stream can't be resumed mid-flight, so a failure surfaces to the consumer.
   */
  function chatMessagesStream(
    messages: ChatMessage[],
    priority = false,
    chatOpts: ChatOptions = {},
  ): AsyncIterable<string> {
    if (!transport.chatStream) {
      // No streaming transport — emit the whole (serialized) response as one chunk.
      return (async function* () {
        const whole = await chatMessages(messages, priority, chatOpts)
        if (whole) yield whole
      })()
    }
    const channel = createAsyncChannel<string>()
    // Hold the worker for the full stream; push deltas to the channel as they land.
    const run = async (): Promise<string> => {
      try {
        for await (const piece of transport.chatStream!(messages, chatOpts)) channel.push(piece)
        channel.close()
      } catch (err) {
        channel.close(err)
      }
      return ''
    }
    void enqueue(run, priority)
    return channel.iterable
  }

  return { chat, chatMessages, chatMessagesStream, enqueue }
}

export type LlmQueue = ReturnType<typeof createLlmQueue>
