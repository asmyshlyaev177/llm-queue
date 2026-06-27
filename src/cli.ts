#!/usr/bin/env node
import { createFetchTransport, type Backend } from './browser.js'
import { createLlmServer } from './server.js'

// `llm-queue serve` — start the shared queue service. The service talks to the
// model over HTTP (fetch transport), so no `ollama` npm package is needed.
//
//   OLLAMA_MODEL=granite4.1:8b llm-queue serve
//   LLM_BACKEND=llamacpp OLLAMA_URL=http://localhost:8080 PORT=11500 llm-queue serve
async function main(): Promise<void> {
  const cmd = process.argv[2]
  if (cmd !== 'serve') {
    console.error('usage: llm-queue serve   (env: OLLAMA_URL, OLLAMA_MODEL, LLM_BACKEND, OLLAMA_NUM_CTX, PORT, HOST)')
    process.exit(1)
  }

  const backend = (process.env.LLM_BACKEND ?? 'ollama') as Backend
  const url =
    process.env.OLLAMA_URL ??
    (backend === 'llamacpp' ? 'http://localhost:8080' : 'http://localhost:11434')
  const model = process.env.OLLAMA_MODEL ?? 'granite4.1:8b'
  const numCtx = Number(process.env.OLLAMA_NUM_CTX ?? 8192)
  const port = Number(process.env.PORT ?? 11500)
  const host = process.env.HOST ?? '127.0.0.1'

  const transport = createFetchTransport({ url, model, backend, numCtx })
  const { listen } = createLlmServer({
    transport,
    port,
    host,
    model,
    numCtx,
    logger: (m, lvl) => (lvl === 'error' ? console.error('[llm-queue]', m) : console.log('[llm-queue]', m)),
  })
  await listen()
  console.log(
    `llm-queue service on http://${host}:${port}  →  ${backend} ${model} @ ${url}\n` +
      `  num_ctx floor ${numCtx} (auto-raises to the largest a client requests; never drops back)`,
  )
}

void main()
