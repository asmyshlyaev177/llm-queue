# llm-queue

A single-worker priority queue + JSON `classify()` helper for a **local LLM**
(Ollama / llama.cpp), plus an optional **HTTP service** so multiple processes
share one serialized queue against one model.

Used by [li-slop-filter](../li-slop-filter) (browser extension) and
[jobbot](../jobbot) — both reach the model **only through the service's web API**,
never the `ollama` SDK directly.

## Why a service?

A local model serves one request at a time. If the extension's service worker
and jobbot's cron each kept their own in-process queue, they'd hit the model
concurrently and thrash. Running one `llm-queue serve` gives a single global
queue: every client funnels through it over HTTP.

```
extension SW ─┐
jobbot       ─┼─HTTP /chat──▶  llm-queue serve  ──fetch──▶  Ollama
bench        ─┘                (one queue, CORS)            (one model)
```

## Run the service

```bash
OLLAMA_MODEL=granite4.1:8b llm-queue serve     # → http://127.0.0.1:11500
# env: OLLAMA_URL, OLLAMA_MODEL, LLM_BACKEND (ollama|llamacpp), OLLAMA_NUM_CTX, PORT, HOST
```

The service owns the Ollama endpoint + model and sends permissive CORS headers
so browser-context clients (an MV3 service worker, a jsdom test) can read it —
the same role `OLLAMA_ORIGINS` plays for Ollama itself.

`POST /chat {system, user, priority?}` → `{content}` · `GET /health` → `ok`.

## Use from a client

```ts
import { createLlmQueue } from 'llm-queue/core'
import { createServiceTransport } from 'llm-queue/client'

const queue = createLlmQueue(createServiceTransport({ url: 'http://localhost:11500' }))
const out = await queue.classify('isRemote', SYSTEM_PROMPT, userText, (p) => ({
  isRemote: p.isRemote === true,
}))
```

`classify()` truncates to the context window, repairs malformed JSON
(`jsonrepair`), retries with a timeout, and maps the parsed object via your
builder. Returns `null` on a non-fatal failure; throws only on fatal model errors.

## Exports

| Subpath | What | Where |
|---|---|---|
| `llm-queue/core` | `createLlmQueue` (queue + `chat` + `classify`), `boolField` | anywhere |
| `llm-queue/client` | `createServiceTransport` (HTTP → service) | browser / SW / node |
| `llm-queue/server` | `createLlmServer` (the HTTP service) | node |
| `llm-queue/browser` | `createFetchTransport` (direct Ollama/llama.cpp HTTP) | service internals, benches |
| `llm-queue/node` | `createOllamaTransport` (the `ollama` SDK) | optional |

## Develop

```bash
pnpm install && pnpm build   # tsup → dist (ESM + d.ts)
pnpm test                    # vitest (core queue + service)
```
