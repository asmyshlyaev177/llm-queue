import { afterEach, describe, expect, it } from 'vitest'
import { createServiceTransport } from './client.js'
import type { ChatTransport } from './core.js'
import { createLlmServer } from './server.js'

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
    const client = createServiceTransport({ url })
    expect(await client.chat('sys', 'hello')).toBe('echo:hello')
  })

  it('serializes concurrent clients through one queue', async () => {
    const t = trackingTransport()
    const url = await start(t.transport)
    const client = createServiceTransport({ url })
    await Promise.all(['a', 'b', 'c', 'd'].map((u) => client.chat('s', u)))
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
    const small = createServiceTransport({ url, numCtx: 8192 })
    const large = createServiceTransport({ url, numCtx: 16384 })

    await small.chat('s', 'a') // 8192
    await large.chat('s', 'b') // raises to 16384
    await small.chat('s', 'c') // still 16384 — never drops back

    expect(t.numCtxSeen).toEqual([8192, 16384, 16384])
  })

  it('returns a JSON error (not a crash) when the transport throws', async () => {
    const transport: ChatTransport = {
      chat: async () => {
        throw new Error('model exploded')
      },
    }
    const url = await start(transport)
    const res = await fetch(`${url}/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ system: 's', user: 'u' }),
    })
    expect(res.status).toBe(500)
    expect((await res.json()).error).toMatch(/model exploded/)
  })
})
