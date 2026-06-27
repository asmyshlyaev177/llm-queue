import { Ollama } from 'ollama'
import { toOllamaOptions, type ChatMessage, type ChatOptions, type ChatTransport } from './core.js'

export interface OllamaTransportConfig {
  url: string
  /** Default model; per-call `ChatOptions.model` overrides it. */
  model: string
  numCtx?: number
}

/**
 * Node transport using the official `ollama` package. Intended for server-side
 * consumers. `ollama` is an optional peer dependency — install it in the
 * consuming project. Browser/SW code should use ./browser instead.
 */
export function createOllamaTransport(config: OllamaTransportConfig): ChatTransport {
  const ollama = new Ollama({ host: config.url })

  async function chatRaw(messages: ChatMessage[], opts: ChatOptions = {}): Promise<string> {
    const response = await ollama.chat({
      model: opts.model || config.model,
      format: opts.json ? 'json' : undefined,
      options: toOllamaOptions(opts.numCtx ?? config.numCtx, opts.params),
      messages,
    })
    return response.message.content.trim()
  }

  return {
    chatRaw,
    // Plain (system, user) chat. JSON mode is opt-in via chatRaw/ChatOptions —
    // callers that parse JSON (classify) pass `json: true` themselves.
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
