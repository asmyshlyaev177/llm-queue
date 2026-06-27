import { toOllamaOptions, type ChatMessage, type ChatOptions, type ChatTransport, type SamplingParams } from './core.js'

export type Backend = 'ollama' | 'llamacpp'

export interface FetchTransportConfig {
  /** Base URL, e.g. http://localhost:11434 */
  url: string
  /** Default model; per-call `ChatOptions.model` overrides it. */
  model: string
  backend?: Backend
  /** Ollama context window (num_ctx). */
  numCtx?: number
}

// Pick out the OpenAI-named sampling fields that are set (llama.cpp's OpenAI
// endpoint takes them verbatim). Ollama uses toOllamaOptions instead.
function openaiSampling(params?: SamplingParams): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  if (!params) return out
  const keys = [
    'temperature',
    'top_p',
    'max_tokens',
    'frequency_penalty',
    'presence_penalty',
    'stop',
    'seed',
  ] as const
  for (const k of keys) if (params[k] != null) out[k] = params[k]
  return out
}

/**
 * Browser/service-worker-safe transport: talks to a local Ollama or llama.cpp
 * server over plain `fetch`. No Node dependencies. In an MV3 extension this must
 * run in the service worker (host_permissions bypass page CORS).
 */
export function createFetchTransport(config: FetchTransportConfig): ChatTransport {
  const backend = config.backend ?? 'ollama'

  async function chatRaw(messages: ChatMessage[], opts: ChatOptions = {}): Promise<string> {
    const effectiveNumCtx = opts.numCtx ?? config.numCtx
    const model = opts.model || config.model

    if (backend === 'llamacpp') {
      const res = await fetch(`${config.url}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          response_format: opts.json ? { type: 'json_object' } : undefined,
          messages,
          ...openaiSampling(opts.params),
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
