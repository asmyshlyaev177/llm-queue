import { collapseMessages, type ChatMessage, type ChatOptions, type ChatTransport } from './core.js'

export interface ServiceTransportConfig {
  /** Base URL of the llm-queue service, e.g. http://localhost:11500 */
  url: string
  /** Send requests as priority (jump the shared queue). */
  priority?: boolean
  /**
   * Context window this client needs (tokens). Sent with every request; the
   * service runs the model at the MAX numCtx any client asks for and never drops
   * below it, so clients that disagree on context size don't make the model
   * reload. Omit to accept whatever the service is already running.
   */
  numCtx?: number
  /**
   * Ask the service to constrain output to JSON (Ollama `format` / llama.cpp
   * `response_format`). Off by default. `classify()` sets this per-call on its
   * own, so you only need it for raw `chat()` callers that want JSON.
   */
  json?: boolean
}

/**
 * A transport that forwards chat requests to a running `llm-queue serve` service
 * over HTTP. Plain fetch — safe in Node, a browser, or an MV3 service worker.
 * Use this so multiple processes (a service worker, a cron job, a benchmark)
 * share ONE queue against a single local model instead of each hammering it
 * concurrently.
 */
export function createServiceTransport(config: ServiceTransportConfig): ChatTransport {
  const base = config.url.replace(/\/$/, '')

  async function post(body: Record<string, unknown>): Promise<string> {
    const res = await fetch(`${base}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      throw new Error(`llm-queue service ${res.status}: ${await res.text()}`)
    }
    const data = (await res.json()) as { content?: string; error?: string }
    if (data.error) throw new Error(`llm-queue service: ${data.error}`)
    return data.content ?? ''
  }

  return {
    chat(systemPrompt: string, userContent: string, numCtx?: number): Promise<string> {
      return post({
        system: systemPrompt,
        user: userContent,
        priority: config.priority ?? false,
        numCtx: numCtx ?? config.numCtx,
        json: config.json ?? false,
      })
    },
    // The service speaks the {system, user} /chat shape, so collapse the messages
    // to that pair while forwarding the JSON request (classify relies on this),
    // plus any model override and sampling params.
    chatRaw(messages: ChatMessage[], opts?: ChatOptions): Promise<string> {
      const { system, user } = collapseMessages(messages)
      return post({
        system,
        user,
        priority: config.priority ?? false,
        numCtx: opts?.numCtx ?? config.numCtx,
        json: opts?.json ?? config.json ?? false,
        model: opts?.model,
        params: opts?.params,
      })
    },
  }
}
