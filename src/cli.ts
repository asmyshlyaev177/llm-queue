#!/usr/bin/env node
import { createFetchTransport } from './browser.js'
import { createLlmServer } from './server.js'

// `llm-queue serve` — start the shared queue service. The service talks to the
// model over HTTP (fetch transport), so no `ollama` npm package is needed.
//
//   OLLAMA_MODEL=granite4.1:8b llm-queue serve
//   OLLAMA_URL=http://my-host:11434 PORT=11500 llm-queue serve
async function main(): Promise<void> {
  const cmd = process.argv[2]
  if (cmd !== 'serve') {
    console.error(
      'usage: llm-queue serve   (env: OLLAMA_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX,\n' +
        '                              LLM_MAX_ATTEMPTS, LLM_TIMEOUT_MS, PORT, HOST)',
    )
    process.exit(1)
  }

  const url = process.env.OLLAMA_URL ?? 'http://localhost:11434'
  const model = process.env.OLLAMA_MODEL ?? 'granite4.1:8b'
  const numCtx = Number(process.env.OLLAMA_NUM_CTX ?? 8192)
  // Reliability knobs for the shared queue's per-attempt timeout + retry.
  const maxAttempts = Number(process.env.LLM_MAX_ATTEMPTS ?? 2)
  const timeoutMs = Number(process.env.LLM_TIMEOUT_MS ?? 90000)
  const port = Number(process.env.PORT ?? 11500)
  const host = process.env.HOST ?? '127.0.0.1'

  const transport = createFetchTransport({ url, model, numCtx })
  const { listen } = createLlmServer({
    transport,
    port,
    host,
    model,
    numCtx,
    maxAttempts,
    timeoutMs,
    logger: (m, lvl) => (lvl === 'error' ? console.error('[llm-queue]', m) : console.log('[llm-queue]', m)),
  })
  await listen()
  console.log(
    `llm-queue service on http://${host}:${port}  →  ollama ${model} @ ${url}\n` +
      `  num_ctx floor ${numCtx} (auto-raises to the largest a client requests; never drops back)\n` +
      `  ${maxAttempts} attempt(s) per request, ${timeoutMs}ms timeout each`,
  )
}

void main()
