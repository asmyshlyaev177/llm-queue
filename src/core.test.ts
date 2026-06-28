import { describe, expect, it, vi } from 'vitest'
import {
  collapseMessages,
  createLlmQueue,
  toOllamaOptions,
  type ChatMessage,
  type ChatTransport,
} from './core.js'

function transportOf(fn: (sys: string, user: string) => Promise<string> | string): ChatTransport {
  return { chat: async (sys, user) => fn(sys, user) }
}

/** Collect an async iterable into an array (test helper for streaming). */
async function collect(iter: AsyncIterable<string>): Promise<string[]> {
  const out: string[] = []
  for await (const piece of iter) out.push(piece)
  return out
}

describe('createLlmQueue', () => {
  it('returns the transport output verbatim', async () => {
    const { chat } = createLlmQueue(transportOf(() => 'hello'))
    expect(await chat('sys', 'user')).toBe('hello')
  })

  it('serializes calls — never runs the transport concurrently', async () => {
    let active = 0
    let maxActive = 0
    const { chat } = createLlmQueue(
      transportOf(async () => {
        active++
        maxActive = Math.max(maxActive, active)
        await new Promise((r) => setTimeout(r, 5))
        active--
        return 'ok'
      }),
    )
    await Promise.all(Array.from({ length: 5 }, () => chat('s', 'u')))
    expect(maxActive).toBe(1)
  })

  it('runs priority items ahead of queued normal items', async () => {
    const order: string[] = []
    const transport = transportOf(async (_sys, user) => {
      await new Promise((r) => setTimeout(r, 1))
      order.push(user)
      return ''
    })
    const { chat } = createLlmQueue(transport)
    // First call occupies the worker; the next two queue behind it.
    const a = chat('s', 'first')
    const b = chat('s', 'normal')
    const c = chat('s', 'priority', true)
    await Promise.all([a, b, c])
    expect(order).toEqual(['first', 'priority', 'normal'])
  })

  it('retries a non-fatal failure then succeeds', async () => {
    const fn = vi
      .fn<(sys: string, user: string) => string>()
      .mockImplementationOnce(() => {
        throw new Error('flaky')
      })
      .mockImplementationOnce(() => 'ok')
    const { chat } = createLlmQueue(transportOf(fn), { maxAttempts: 2 })
    expect(await chat('s', 'u')).toBe('ok')
    expect(fn).toHaveBeenCalledTimes(2)
  })

  it('stops retrying on a fatal model error', async () => {
    const fn = vi.fn<() => string>(() => {
      throw new Error('model not found: foo')
    })
    const { chat } = createLlmQueue(transportOf(fn), { maxAttempts: 3 })
    await expect(chat('s', 'u')).rejects.toThrow(/model not found/)
    expect(fn).toHaveBeenCalledTimes(1) // fatal short-circuits the retry loop
  })

  it('rejects after exhausting retries on a non-fatal error', async () => {
    const { chat } = createLlmQueue(
      transportOf(() => {
        throw new Error('ECONNREFUSED')
      }),
      { maxAttempts: 1 },
    )
    await expect(chat('s', 'u')).rejects.toThrow(/ECONNREFUSED/)
  })
})

describe('toOllamaOptions', () => {
  it('maps sampling params (+ num_ctx); max_tokens → num_predict', () => {
    expect(toOllamaOptions(8192, { temperature: 0.2, max_tokens: 100, seed: 7 })).toEqual({
      num_ctx: 8192,
      temperature: 0.2,
      num_predict: 100,
      seed: 7,
    })
  })

  it('returns undefined when nothing is set', () => {
    expect(toOllamaOptions()).toBeUndefined()
    expect(toOllamaOptions(0, {})).toBeUndefined()
  })
})

describe('collapseMessages', () => {
  it('joins system messages and keeps a single user turn verbatim', () => {
    expect(
      collapseMessages([
        { role: 'system', content: 'a' },
        { role: 'system', content: 'b' },
        { role: 'user', content: 'hi' },
      ]),
    ).toEqual({ system: 'a\n\nb', user: 'hi' })
  })

  it('renders multiple non-system turns as a labelled transcript', () => {
    expect(
      collapseMessages([
        { role: 'user', content: 'q' },
        { role: 'assistant', content: 'r' },
        { role: 'user', content: 'q2' },
      ]),
    ).toEqual({ system: '', user: 'user: q\n\nassistant: r\n\nuser: q2' })
  })
})

describe('createLlmQueue.chatMessages', () => {
  it('uses the transport chatRaw when present, messages intact', async () => {
    let seen: ChatMessage[] | undefined
    const transport: ChatTransport = {
      chat: async () => 'unused',
      chatRaw: async (messages) => {
        seen = messages
        return 'raw'
      },
    }
    const { chatMessages } = createLlmQueue(transport)
    const msgs: ChatMessage[] = [
      { role: 'system', content: 's' },
      { role: 'user', content: 'u' },
    ]
    expect(await chatMessages(msgs)).toBe('raw')
    expect(seen).toEqual(msgs)
  })

  it('collapses to chat(system, user) when the transport has no chatRaw', async () => {
    let sysUser: [string, string] | undefined
    const transport: ChatTransport = {
      chat: async (sys, user) => {
        sysUser = [sys, user]
        return `${sys}|${user}`
      },
    }
    const { chatMessages } = createLlmQueue(transport)
    expect(
      await chatMessages([
        { role: 'system', content: 's' },
        { role: 'user', content: 'u' },
      ]),
    ).toBe('s|u')
    expect(sysUser).toEqual(['s', 'u'])
  })
})

describe('createLlmQueue.chatMessagesStream', () => {
  function streamingTransport(pieces: string[]): ChatTransport {
    return {
      chat: async () => pieces.join(''),
      // eslint-disable-next-line require-yield
      async *chatStream() {
        for (const p of pieces) yield p
      },
    }
  }

  it('yields the transport stream deltas in order', async () => {
    const { chatMessagesStream } = createLlmQueue(streamingTransport(['Hel', 'lo', ' world']))
    expect(await collect(chatMessagesStream([{ role: 'user', content: 'hi' }]))).toEqual([
      'Hel',
      'lo',
      ' world',
    ])
  })

  it('falls back to a single whole-response chunk when the transport cannot stream', async () => {
    // transportOf has only `chat` — no chatStream — so the queue streams one chunk.
    const { chatMessagesStream } = createLlmQueue(transportOf(() => 'whole response'))
    expect(await collect(chatMessagesStream([{ role: 'user', content: 'hi' }]))).toEqual([
      'whole response',
    ])
  })

  it('propagates a mid-stream error to the consumer after delivering earlier deltas', async () => {
    const transport: ChatTransport = {
      chat: async () => '',
      async *chatStream() {
        yield 'partial'
        throw new Error('boom')
      },
    }
    const { chatMessagesStream } = createLlmQueue(transport)
    const got: string[] = []
    await expect(
      (async () => {
        for await (const p of chatMessagesStream([{ role: 'user', content: 'x' }])) got.push(p)
      })(),
    ).rejects.toThrow(/boom/)
    expect(got).toEqual(['partial'])
  })

  it('holds the single worker for the whole stream, serializing with a queued call', async () => {
    const order: string[] = []
    const transport: ChatTransport = {
      chat: async (_sys, user) => {
        order.push(`chat:${user}`)
        return 'x'
      },
      async *chatStream() {
        order.push('stream:start')
        await new Promise((r) => setTimeout(r, 10))
        yield 'a'
        order.push('stream:end')
      },
    }
    const { chatMessagesStream, chat } = createLlmQueue(transport)
    // Start the stream (occupies the worker), then enqueue a normal chat behind it.
    const streamDone = collect(chatMessagesStream([{ role: 'user', content: 's' }]))
    const chatDone = chat('sys', 'queued')
    await Promise.all([streamDone, chatDone])
    expect(order).toEqual(['stream:start', 'stream:end', 'chat:queued'])
  })
})
