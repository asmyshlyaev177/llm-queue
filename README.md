# llm-queue

A single-worker priority queue + JSON `classify()` helper for a **local LLM**
(Ollama / llama.cpp), plus an optional **OpenAI-compatible HTTP service** so
multiple processes share one serialized queue against one model.

## Why a service?

A local model serves one request at a time. When several processes — a cron job,
a browser extension's service worker, a benchmark, an OpenAI client — each keep
their own in-process queue, they hit the model concurrently and thrash. Running
one `llm-queue serve` gives a single global queue: every client funnels through
it over HTTP.

```text
process A  ─┐
process B  ─┼─ HTTP ─▶  llm-queue serve  ──fetch──▶  Ollama / llama.cpp
OpenAI SDK ─┘          (one queue, CORS)            (one model at a time)
```

## Run the service

```bash
OLLAMA_MODEL=granite4.1:8b llm-queue serve     # → http://127.0.0.1:11500
# env: OLLAMA_URL, OLLAMA_MODEL, LLM_BACKEND (ollama|llamacpp), OLLAMA_NUM_CTX, PORT, HOST
```

The service owns the Ollama endpoint + model and sends permissive CORS headers
so browser-context clients (an MV3 service worker, a jsdom test) can read it —
the same role `OLLAMA_ORIGINS` plays for Ollama itself.

Endpoints:

| Method | Path | Body / notes |
|---|---|---|
| `POST` | `/chat` | `{system, user, priority?, numCtx?, json?}` → `{content}` — the native API |
| `POST` | `/v1/chat/completions` | OpenAI-compatible (`messages`, `stream`, `response_format`) |
| `GET`  | `/v1/models` | OpenAI model list (single model) |
| `GET`  | `/health` | → `ok` |

Output is plain text by default; pass `json: true` (native) or
`response_format: { type: 'json_object' }` (OpenAI) to constrain to JSON.

## OpenAI-compatible

Point any OpenAI client at `<url>/v1` to get the **same single serialized queue**
in front of your local model — no SDK swap. The bearer token is ignored.

```ts
import OpenAI from 'openai'
const client = new OpenAI({ baseURL: 'http://localhost:11500/v1', apiKey: 'unused' })
const r = await client.chat.completions.create({
  model: 'granite4.1:8b', // any model your backend has — see GET /v1/models
  messages: [{ role: 'user', content: 'hello' }],
  temperature: 0.2,
  max_tokens: 256,
})
```

- **Model** — the requested `model` is passed through to the backend; omit it (or
  send an empty string) to use the service's configured `OLLAMA_MODEL`. All
  clients sharing one model is what keeps the runner from reloading — mixing
  models is allowed but reloads between calls (serialized, so never concurrent).
- **Sampling params** — `temperature`, `top_p`, `max_tokens` (→ Ollama
  `num_predict`), `frequency_penalty`, `presence_penalty`, `stop`, and `seed` are
  forwarded to the backend.
- **Streaming** — `stream: true` is supported, emitted as a single delta (the
  queue resolves whole responses — protocol-compatible, not token-by-token).
- **Priority / num_ctx** — standard clients can't express these, so they're
  accepted as optional `x-llmq-priority` / `x-llmq-num-ctx` request headers.

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
