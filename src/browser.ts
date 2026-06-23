import type { ChatTransport } from './core.js'

export type Backend = 'ollama' | 'llamacpp'

export interface FetchTransportConfig {
  /** Base URL, e.g. http://localhost:11434 */
  url: string
  model: string
  backend?: Backend
  /** Ollama context window (num_ctx). */
  numCtx?: number
}

/**
 * Browser/service-worker-safe transport: talks to a local Ollama or llama.cpp
 * server over plain `fetch`. No Node dependencies. In an MV3 extension this must
 * run in the service worker (host_permissions bypass page CORS).
 */
export function createFetchTransport(config: FetchTransportConfig): ChatTransport {
  const backend = config.backend ?? 'ollama'
  return {
    async chat(systemPrompt: string, userContent: string, numCtx?: number): Promise<string> {
      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userContent },
      ]
      const effectiveNumCtx = numCtx ?? config.numCtx

      if (backend === 'llamacpp') {
        const res = await fetch(`${config.url}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: config.model,
            response_format: { type: 'json_object' },
            messages,
          }),
        })
        if (!res.ok) throw new Error(`llama.cpp ${res.status}: ${await res.text()}`)
        const data = (await res.json()) as {
          choices: Array<{ message: { content: string } }>
        }
        return data.choices[0]?.message?.content?.trim() ?? ''
      }

      const res = await fetch(`${config.url}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.model,
          format: 'json',
          stream: false,
          options: effectiveNumCtx ? { num_ctx: effectiveNumCtx } : undefined,
          messages,
        }),
      })
      if (!res.ok) throw new Error(`ollama ${res.status}: ${await res.text()}`)
      const data = (await res.json()) as { message?: { content?: string } }
      return data.message?.content?.trim() ?? ''
    },
  }
}
