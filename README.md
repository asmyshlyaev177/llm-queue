# llm-queue

> One local model, one serialized queue. Run `llm-queue serve` and every process
> — a cron job, a browser extension, an OpenAI client — funnels through a single
> **priority queue** over an **OpenAI-compatible HTTP API**, so the model is never
> hit concurrently and the runner never thrashes.

[![npm](https://img.shields.io/npm/v/llm-queue.svg)](https://www.npmjs.com/package/llm-queue)
[![CI](https://github.com/asmyshlyaev177/llm-queue/actions/workflows/test.yml/badge.svg)](https://github.com/asmyshlyaev177/llm-queue/actions/workflows/test.yml)
[![node](https://img.shields.io/node/v/llm-queue.svg)](https://www.npmjs.com/package/llm-queue)
[![license](https://img.shields.io/github/license/asmyshlyaev177/llm-queue.svg?style=flat-square)](./LICENSE)

A single-worker priority queue for a **local LLM** (Ollama), exposed as an
**OpenAI-compatible HTTP service** so multiple processes share one serialized
queue against one model.

## Install

```bash
npm i -g llm-queue   # the `llm-queue serve` CLI
# or embed the queue/server in your own Node app:
npm i llm-queue
```

Needs Node ≥ 20 and a running [Ollama](https://ollama.com).

## Why a service?

A local model serves one request at a time. When several processes — a cron job,
a browser extension's service worker, a benchmark, an OpenAI client — each keep
their own in-process queue, they hit the model concurrently and thrash. Running
one `llm-queue serve` gives a single global queue: every client funnels through
it over HTTP.

```text
process A  ─┐
process B  ─┼─ HTTP ─▶  llm-queue serve  ──fetch──▶  Ollama
OpenAI SDK ─┘          (one queue, CORS)            (one model at a time)
```

## Run the service

```bash
OLLAMA_MODEL=granite4.1:8b llm-queue serve     # → http://127.0.0.1:11500
# env: OLLAMA_URL (point at any Ollama host), OLLAMA_MODEL, OLLAMA_NUM_CTX,
#      LLM_MAX_ATTEMPTS (attempts incl. first; 1 = no retry), LLM_TIMEOUT_MS, PORT, HOST
```

The service owns the Ollama endpoint + model and sends permissive CORS headers
so browser-context clients (an MV3 service worker, a jsdom test) can read it —
the same role `OLLAMA_ORIGINS` plays for Ollama itself.

Endpoints:

| Method | Path | Body / notes |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible (`messages`, `stream`, `response_format`) + `priority?` / `numCtx?` extension fields |
| `GET` | `/v1/models` | OpenAI model list (single model) |
| `GET` | `/health` | → `ok` |

One endpoint: standard OpenAI `/v1/chat/completions`, plus two body fields the
queue needs that OpenAI has no slot for (`priority`, `numCtx`). Output is plain
text by default; pass `response_format: { type: 'json_object' }` to constrain it
to JSON.

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
- **Streaming** — `stream: true` streams real token-by-token SSE deltas straight
  from the model. The stream still goes through the one queue (it holds the single
  worker until it finishes, so requests never overlap). Omit it / `stream: false`
  for a whole-response JSON completion (what JSON-parsing clients want).
- **Priority / num_ctx** — standard clients can't express these, so they ride as
  optional `priority` (boolean) / `numCtx` (number) fields on the request body.
  Stock OpenAI servers ignore unknown body fields; ours read them.

## Use from a client

Clients are plain HTTP — no SDK or package needed. A browser service worker, a
cron job, and a benchmark all share the one queue just by POSTing to it:

```ts
const res = await fetch('http://localhost:11500/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userText },
    ],
    response_format: { type: 'json_object' }, // optional: constrain to JSON
    numCtx: 8192,                             // optional llm-queue extension
    stream: false,                            // whole completion (the default)
  }),
})
const { choices } = await res.json()
const out = choices[0].message.content
```

The queue serializes every client through the one model with a per-attempt
timeout and retry, and returns the model's raw string — JSON parsing, truncation,
and validation are the caller's job (this keeps it a queue, not a framework).

## Priority — two clients, one queue

When several clients share the queue, the `priority` body field lets a
latency-sensitive one jump ahead of a background one's backlog. A priority
request slots in front of any *waiting* normal requests — but never preempts the
one already running.

```ts
const url = 'http://localhost:11500/v1/chat/completions'
const post = (body: object) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

// Client A — a background batch job. Normal priority; fills the queue.
for (const row of rows) void post({ messages: [{ role: 'user', content: row.text }] }) // hundreds queued

// Client B — serves a user action. priority: true jumps A's backlog.
const res = await post({
  messages: [{ role: 'user', content: userQuestion }],
  priority: true, // llm-queue extension; runs next, before A's remaining backlog
})
```

Both processes funnel through the single `llm-queue serve` queue, so B's request
runs as soon as A's in-flight call finishes — ahead of A's remaining backlog. The
OpenAI SDK forwards the same field (stock OpenAI servers ignore unknown keys):

```ts
await client.chat.completions.create({
  model: 'granite4.1:8b',
  messages: [{ role: 'user', content: userQuestion }],
  // @ts-expect-error llm-queue extension; the SDK forwards it, OpenAI servers ignore it
  priority: true,
})
```

## Exports

| Subpath | What | Where |
|---|---|---|
| `llm-queue/core` | `createLlmQueue` (serialized queue: `chat`, `chatMessages`, `chatMessagesStream`) | anywhere |
| `llm-queue/browser` | `createFetchTransport` (direct Ollama HTTP) | service internals |
| `llm-queue/server` | `createLlmServer` (the HTTP service) | node |

Most consumers don't import the package at all — they just POST to the running
`llm-queue serve` over HTTP (see [Use from a client](#use-from-a-client)). The
subpaths above are the building blocks the CLI itself is made of.

## Develop

```bash
pnpm install                 # builds dist via the prepare hook (tsup)
pnpm typecheck               # tsc --noEmit
pnpm lint                    # oxlint
pnpm test                    # vitest (core queue + service)
pnpm build                   # tsup → dist (ESM + d.ts)
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, …) — release-please uses them to version, generate the
changelog, and publish to npm on merge to `master`.

## License

[MIT](./LICENSE) © asmyshlyaev177
