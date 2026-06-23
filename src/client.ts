import type { ChatTransport } from './core.js'

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
}

/**
 * A transport that forwards chat requests to a running `llm-queue serve` service
 * over HTTP. Plain fetch — safe in Node, a browser, or an MV3 service worker.
 * Use this so multiple processes (extension, jobbot, benches) share ONE queue
 * against a single local model instead of each hammering it concurrently.
 */
export function createServiceTransport(config: ServiceTransportConfig): ChatTransport {
  return {
    async chat(systemPrompt: string, userContent: string, numCtx?: number): Promise<string> {
      const res = await fetch(`${config.url.replace(/\/$/, '')}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          system: systemPrompt,
          user: userContent,
          priority: config.priority ?? false,
          numCtx: numCtx ?? config.numCtx,
        }),
      })
      if (!res.ok) {
        throw new Error(`llm-queue service ${res.status}: ${await res.text()}`)
      }
      const data = (await res.json()) as { content?: string; error?: string }
      if (data.error) throw new Error(`llm-queue service: ${data.error}`)
      return data.content ?? ''
    },
  }
}
