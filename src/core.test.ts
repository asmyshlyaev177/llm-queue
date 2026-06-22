import { describe, expect, it, vi } from 'vitest'
import { boolField, createLlmQueue, type ChatTransport } from './core.js'

function transportOf(fn: (sys: string, user: string) => Promise<string> | string): ChatTransport {
  return { chat: async (sys, user) => fn(sys, user) }
}

describe('createLlmQueue.classify', () => {
  it('parses well-formed JSON', async () => {
    const { classify } = createLlmQueue(transportOf(() => '{"isSlop": true}'))
    const out = await classify('t', 'sys', 'user', (p) => ({ isSlop: boolField(p, 'isSlop') }))
    expect(out).toEqual({ isSlop: true })
  })

  it('repairs malformed JSON (markdown fence + trailing comma)', async () => {
    const { classify } = createLlmQueue(
      transportOf(() => '```json\n{"isSlop": true,}\n```'),
    )
    const out = await classify('t', 'sys', 'user', (p) => ({ isSlop: boolField(p, 'isSlop') }))
    expect(out).toEqual({ isSlop: true })
  })

  it('returns null on transport error (non-fatal)', async () => {
    const { classify } = createLlmQueue(
      transportOf(() => {
        throw new Error('ECONNREFUSED')
      }),
      { maxAttempts: 1 },
    )
    const out = await classify('t', 'sys', 'user', () => ({ x: 1 }))
    expect(out).toBeNull()
  })

  it('throws on fatal model error', async () => {
    const { classify } = createLlmQueue(
      transportOf(() => {
        throw new Error('model not found: foo')
      }),
      { maxAttempts: 1 },
    )
    await expect(classify('t', 'sys', 'user', () => ({}))).rejects.toThrow(/Fatal/)
  })

  it('serializes calls — never runs the transport concurrently', async () => {
    let active = 0
    let maxActive = 0
    const { classify } = createLlmQueue(
      transportOf(async () => {
        active++
        maxActive = Math.max(maxActive, active)
        await new Promise((r) => setTimeout(r, 5))
        active--
        return '{"ok": true}'
      }),
    )
    await Promise.all(
      Array.from({ length: 5 }, () => classify('t', 's', 'u', (p) => p)),
    )
    expect(maxActive).toBe(1)
  })

  it('runs priority items ahead of queued normal items', async () => {
    const order: string[] = []
    const transport = transportOf(async (_sys, user) => {
      await new Promise((r) => setTimeout(r, 1))
      order.push(user)
      return '{}'
    })
    const { classify } = createLlmQueue(transport)
    // First call occupies the worker; the next two queue behind it.
    const a = classify('a', 's', 'first', (p) => p)
    const b = classify('b', 's', 'normal', (p) => p)
    const c = classify('c', 's', 'priority', (p) => p, true)
    await Promise.all([a, b, c])
    expect(order).toEqual(['first', 'priority', 'normal'])
  })

  it('retries once then succeeds', async () => {
    const fn = vi
      .fn<(sys: string, user: string) => string>()
      .mockImplementationOnce(() => {
        throw new Error('flaky')
      })
      .mockImplementationOnce(() => '{"ok":true}')
    const { classify } = createLlmQueue(transportOf(fn), { maxAttempts: 2 })
    const out = await classify('t', 's', 'u', (p) => p)
    expect(out).toEqual({ ok: true })
    expect(fn).toHaveBeenCalledTimes(2)
  })
})
