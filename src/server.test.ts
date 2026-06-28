import { afterEach, describe, expect, it } from 'vitest'
import type { ChatMessage, ChatTransport, SamplingParams } from './core.js'
import { createLlmServer } from './server.js'

// POST an OpenAI-shaped chat request and return the completion content. Clients
// of the service are plain HTTP — this mirrors what they send.
async function postChat(url: string, body: Record<string, unknown>): Promise<string> {
  const res = await fetch(`${url}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> }
  return data.choices?.[0]?.message?.content ?? ''
}

const userMsg = (content: string): ChatMessage[] => [{ role: 'user', content }]

// A transport that records what each call received — messages reach `chatRaw`
// (the OpenAI/messages path), (system,user) reach `chat`. Lets us assert the
// messages array arrives intact and the JSON/model/params flow through.
interface Recorded {
  messages?: ChatMessage[]
  json?: boolean
  numCtx?: number
  model?: string
  params?: SamplingParams
}
function recordingTransport() {
  const calls: Recorded[] = []
  const transport: ChatTransport = {
    async chat(_system, user, numCtx) {
      calls.push({ numCtx })
      return `chat:${user}`
    },
    async chatRaw(messages, opts) {
      calls.push({
        messages,
        json: opts?.json,
        numCtx: opts?.numCtx,
        model: opts?.model,
        params: opts?.params,
      })
      return `raw:${messages.map((m) => m.content).join('|')}`
    },
  }
  return { transport, calls }
}

// A transport that records call order and concurrency, so we can assert the
// service serializes requests (one model call at a time) across HTTP clients.
function trackingTransport() {
  let active = 0
  let maxActive = 0
  const seen: string[] = []
  const numCtxSeen: Array<number | undefined> = []
  const transport: ChatTransport = {
    async chat(_sys, user, numCtx) {
      active++
      maxActive = Math.max(maxActive, active)
      await new Promise((r) => setTimeout(r, 10))
      active--
      seen.push(user)
      numCtxSeen.push(numCtx)
      return `echo:${user}`
    },
  }
  return { transport, get maxActive() { return maxActive }, seen, numCtxSeen }
}

// A transport that streams given pieces token-by-token (and records what each
// chatStream call received), so we can assert real SSE deltas over HTTP.
function streamingTransport(pieces: string[]) {
  const calls: Array<{ messages: ChatMessage[]; json?: boolean }> = []
  const transport: ChatTransport = {
    chat: async () => pieces.join(''),
    async *chatStream(messages, opts) {
      calls.push({ messages, json: opts?.json })
      for (const p of pieces) yield p
    },
  }
  return { transport, calls }
}

let close: (() => Promise<void>) | null = null
afterEach(async () => {
  if (close) await close()
  close = null
})

async function start(transport: ChatTransport) {
  const port = 11000 + Math.floor(Math.random() * 2000)
  const srv = createLlmServer({ transport, port, host: '127.0.0.1' })
  await srv.listen()
  close = srv.close
  return `http://127.0.0.1:${port}`
}

describe('llm-queue service', () => {
  it('round-trips a chat request and returns its content', async () => {
    const { transport } = trackingTransport()
    const url = await start(transport)
    expect(await postChat(url, { messages: userMsg('hello') })).toBe('echo:hello')
  })

  it('serializes concurrent clients through one queue', async () => {
    const t = trackingTransport()
    const url = await start(t.transport)
    await Promise.all(['a', 'b', 'c', 'd'].map((u) => postChat(url, { messages: userMsg(u) })))
    expect(t.maxActive).toBe(1)
    expect(t.seen.sort()).toEqual(['a', 'b', 'c', 'd'])
  })

  it('sends permissive CORS headers (browser-context clients)', async () => {
    const { transport } = trackingTransport()
    const url = await start(transport)
    const res = await fetch(`${url}/health`)
    expect(res.headers.get('access-control-allow-origin')).toBe('*')
  })

  it('runs at the high-water max numCtx and never drops below it', async () => {
    const t = trackingTransport()
    const url = await start(t.transport)

    await postChat(url, { messages: userMsg('a'), numCtx: 8192 }) // 8192
    await postChat(url, { messages: userMsg('b'), numCtx: 16384 }) // raises to 16384
    await postChat(url, { messages: userMsg('c'), numCtx: 8192 }) // still 16384 — never drops back

    expect(t.numCtxSeen).toEqual([8192, 16384, 16384])
  })

  it('returns a JSON error (not a crash) when the transport throws', async () => {
    const transport: ChatTransport = {
      chat: async () => {
        throw new Error('model exploded')
      },
    }
    const url = await start(transport)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'u' }] }),
    })
    expect(res.status).toBe(500)
    expect((await res.json()).error.message).toMatch(/model exploded/)
  })

  it('reads priority and numCtx as llm-queue body extension fields on /v1', async () => {
    const r = recordingTransport()
    const url = await start(r.transport)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'x' }],
        priority: true,
        numCtx: 4096,
      }),
    })
    expect(res.status).toBe(200)
    // numCtx negotiated to the high-water max and forwarded to the backend.
    expect(r.calls[0].numCtx).toBe(4096)
  })

  it('serves OpenAI-compatible /v1/chat/completions with the messages intact', async () => {
    const r = recordingTransport()
    const url = await start(r.transport)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'm',
        messages: [
          { role: 'system', content: 'sys' },
          { role: 'user', content: 'hi' },
        ],
      }),
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.object).toBe('chat.completion')
    expect(data.choices[0].message).toEqual({ role: 'assistant', content: 'raw:sys|hi' })
    expect(data.choices[0].finish_reason).toBe('stop')
    expect(data.usage.total_tokens).toBeGreaterThan(0)
    // messages reached the transport natively, not collapsed to a transcript
    expect(r.calls[0].messages).toEqual([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
    ])
  })

  it('only requests JSON mode when response_format asks for it', async () => {
    const r = recordingTransport()
    const url = await start(r.transport)
    const call = (extra: object) =>
      fetch(`${url}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ messages: [{ role: 'user', content: 'x' }], ...extra }),
      })
    await call({})
    await call({ response_format: { type: 'json_object' } })
    expect(r.calls.map((c) => c.json)).toEqual([false, true])
  })

  it('rejects a /v1 request with no messages (OpenAI error shape)', async () => {
    const r = recordingTransport()
    const url = await start(r.transport)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [] }),
    })
    expect(res.status).toBe(400)
    expect((await res.json()).error.type).toBe('invalid_request_error')
  })

  it('streams real token-by-token SSE deltas from a streaming transport', async () => {
    const { transport, calls } = streamingTransport(['Hel', 'lo', '!'])
    const url = await start(transport)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream: true }),
    })
    expect(res.headers.get('content-type')).toMatch(/text\/event-stream/)
    const text = await res.text()
    // The role-priming empty delta, then one delta per model token, in order.
    const contents = [...text.matchAll(/"content":"([^"]*)"/g)].map((m) => m[1])
    expect(contents).toEqual(['', 'Hel', 'lo', '!'])
    expect(text.trimEnd().endsWith('data: [DONE]')).toBe(true)
    expect(calls[0].messages).toEqual([{ role: 'user', content: 'hi' }])
  })

  it('passes JSON mode through to the streaming transport', async () => {
    const { transport, calls } = streamingTransport(['{}'])
    const url = await start(transport)
    await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'x' }],
        stream: true,
        response_format: { type: 'json_object' },
      }),
    }).then((r) => r.text())
    expect(calls[0].json).toBe(true)
  })

  it('closes the SSE stream cleanly when the transport errors mid-stream', async () => {
    const transport: ChatTransport = {
      chat: async () => '',
      async *chatStream() {
        yield 'partial'
        throw new Error('explode')
      },
    }
    const url = await start(transport)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'x' }], stream: true }),
    })
    // Headers already sent → still 200; the delta before the failure is kept and
    // the stream is closed with a normal [DONE] rather than hanging.
    expect(res.status).toBe(200)
    const text = await res.text()
    expect(text).toContain('partial')
    expect(text.trimEnd().endsWith('data: [DONE]')).toBe(true)
  })

  it('emits a whole-response SSE delta for a transport that cannot stream (fallback)', async () => {
    // recordingTransport has no chatStream, so the queue streams one whole chunk.
    const r = recordingTransport()
    const url = await start(r.transport)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream: true }),
    })
    expect(res.headers.get('content-type')).toMatch(/text\/event-stream/)
    const text = await res.text()
    expect(text).toContain('chat.completion.chunk')
    expect(text).toContain('raw:hi')
    expect(text.trimEnd().endsWith('data: [DONE]')).toBe(true)
  })

  it('routes /v1 requests through the same serialized queue', async () => {
    const t = trackingTransport()
    const url = await start(t.transport)
    await Promise.all(
      ['a', 'b', 'c', 'd'].map((u) =>
        fetch(`${url}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ messages: [{ role: 'user', content: u }] }),
        }),
      ),
    )
    expect(t.maxActive).toBe(1)
  })

  it('lists a model at /v1/models', async () => {
    const { transport } = trackingTransport()
    const url = await start(transport)
    const res = await fetch(`${url}/v1/models`)
    const data = await res.json()
    expect(data.object).toBe('list')
    expect(data.data[0].id).toBe('local')
  })

  it('respects the requested model and forwards sampling params on /v1', async () => {
    const r = recordingTransport()
    const url = await start(r.transport)
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3.1:8b',
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.2,
        max_tokens: 128,
        stop: ['\n\n'],
      }),
    })
    expect((await res.json()).model).toBe('llama3.1:8b') // echoed back
    expect(r.calls[0].model).toBe('llama3.1:8b') // forwarded to the backend
    expect(r.calls[0].params).toMatchObject({ temperature: 0.2, max_tokens: 128, stop: ['\n\n'] })
  })

  it('falls back to the configured model when none is requested', async () => {
    const r = recordingTransport()
    const url = await start(r.transport) // default model name is "local"
    const res = await fetch(`${url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: '', messages: [{ role: 'user', content: 'x' }] }),
    })
    expect((await res.json()).model).toBe('local')
    expect(r.calls[0].model).toBeUndefined() // transport uses its own default
  })
})
