import { Ollama } from 'ollama'
import type { ChatTransport } from './core.js'

export interface OllamaTransportConfig {
  url: string
  model: string
  numCtx?: number
}

/**
 * Node transport using the official `ollama` package. Intended for server-side
 * consumers (e.g. jobbot). `ollama` is an optional peer dependency — install it
 * in the consuming project. Browser/SW code should use ./browser instead.
 */
export function createOllamaTransport(config: OllamaTransportConfig): ChatTransport {
  const ollama = new Ollama({ host: config.url })
  return {
    async chat(systemPrompt: string, userContent: string): Promise<string> {
      const response = await ollama.chat({
        model: config.model,
        format: 'json',
        options: config.numCtx ? { num_ctx: config.numCtx } : undefined,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent },
        ],
      })
      return response.message.content.trim()
    },
  }
}
