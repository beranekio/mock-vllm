# mock-vllm

Lightweight mock of the [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/online_serving/) for integration tests, local development, and CI. Runs as a small Go binary or Docker image—no GPU, no model weights.

## Features

- **OpenAI-compatible** endpoints on port `8000` (vLLM default):
  - `GET /v1/models`
  - `POST /v1/chat/completions` (JSON and SSE streaming)
  - `POST /v1/completions` (JSON and SSE streaming)
  - `POST /v1/embeddings`
  - `POST /v1/responses` (OpenAI Responses API shape)
  - `POST /v1/responses/input_tokens`
- **Anthropic Messages API** (as supported by vLLM):
  - `POST /v1/messages` (JSON and SSE streaming)
  - `POST /v1/messages/count_tokens`
- **Health / utility** (vLLM-style):
  - `GET /health`, `GET /healthz`, `GET /ping`, `GET /version`

Responses are deterministic: user text containing `hi`/`hello` → `hi`, `bye` → `bye`, otherwise `ok`. Optional slow responses when the request body contains configured markers (default: `otter`, `long story`).

Batched `/v1/completions` and `/v1/embeddings` requests return one choice or embedding per array element with matching `index` values, including when `stream: true` for completions. Embeddings accept string batches (`["a","b"]`), a single token array (`[1,2,3]`), or batched token arrays (`[[1,2],[3,4]]`); mock vectors are seeded deterministically from text or token IDs.

## Quick start

### Local binary

```bash
go run ./cmd/mock-vllm
```

### Docker

```bash
docker build -t mock-vllm .
docker run --rm -p 8000:8000 mock-vllm
```

Pushes to `main` publish a tested image to GitHub Container Registry after CI passes:

```bash
docker pull ghcr.io/beranekio/mock-vllm:latest
docker run --rm -p 8000:8000 ghcr.io/beranekio/mock-vllm:latest
```

Tags: `latest`, the short commit SHA, and `main`. On first publish, set the package visibility to **Public** under the repo’s **Packages** settings if you need anonymous pulls.

### Integration tests

**Anthropic Messages API** tests use the official [Anthropic Go SDK](https://github.com/anthropics/anthropic-sdk-go). By default they spin up an in-process `httptest` server; set `INTEGRATION_BASE_URL` to test a running instance (e.g. Docker):

```bash
go test -race ./integration/...

# against a container or local binary on port 8000:
INTEGRATION_BASE_URL=http://127.0.0.1:8000 go test -race ./integration/...
```

**OpenAI-compatible API** tests use the [openai-compatibility-tester](https://github.com/beranekio/openai-compatibility-tester) container to validate all supported OpenAI-compatible endpoints:

```bash
# Run the supported OpenAI compatibility tests against a local container or binary.
# The tester container runs in a separate network namespace from the host where
# mock-vllm is listening, so use --network host to share the host network and
# address it via the loopback.
docker run --rm --network host \
  -e OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  -e OPENAI_API_KEY=test-key \
  -e OPENAI_MODEL=mock-model \
  -e OPENAI_EMBEDDING_MODEL=mock-model \
  -e OPENAI_COMPLETION_MODEL=mock-model \
  -e TEST_SUITES=models,models_get,chat_completions,chat_completions_stream,completions,completions_stream,embeddings,embeddings_batch,responses,responses_input_tokens \
  -e REQUEST_TIMEOUT=30s \
  ghcr.io/beranekio/openai-compatibility-tester:latest
```

The supported suite list mirrors what CI runs (`.github/workflows/ci.yml`); the `extended` preset includes image, audio, and tool suites that mock-vllm does not implement, so use the explicit list above.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listen port |
| `DEFAULT_MODEL` | `mock-model` | Model id in `/v1/models` and when requests omit `model` |
| `SLOW_DELAY_SECONDS` | `30` | Sleep duration when slow markers match |
| `SLOW_MARKERS` | `otter,long story` | Comma-separated substrings (case-insensitive) |
| `RESPONSE_PREFIX` | *(empty)* | Prefix prepended to all reply text |
| `LOG_REQUESTS` | `true` | Log each HTTP request |

## Example requests

**OpenAI chat**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}'
```

**Anthropic messages** (e.g. Claude Code with `ANTHROPIC_BASE_URL=http://localhost:8000`)

```bash
curl -s http://localhost:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-api-key: dummy' \
  -d '{"model":"mock-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

**OpenAI responses** (streaming emits the typed Responses event sequence; the non-streaming response is a single `Response` object)

```bash
curl -sN http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","stream":true,"input":"hi"}'
```

### `/v1/responses` streaming contract

When `stream: true`, the mock emits the canonical OpenAI Responses event sequence as `text/event-stream` SSE, with one JSON payload per `data:` line:

1. `response.created` — empty `output`, `status: in_progress`
2. `response.in_progress` — same response object, still no output (mirrors `response.created`; emitted between created and the first item, matching the documented OpenAI Responses lifecycle)
3. `response.output_item.added` — assistant message skeleton, `status: in_progress`
4. `response.content_part.added` — first `output_text` part
5. `response.output_text.delta` (one or more) — accumulated reply chunks
6. `response.output_text.done` — final text
7. `response.content_part.done` — closing the part
8. `response.output_item.done` — closing the item, `status: completed`
9. `response.completed` — terminal event with the full `Response` (including `usage`)

Every event includes a monotonically increasing `sequence_number` starting at 0 (`response.created` is `sequence_number: 0`). Non-streaming responses follow the same envelope (`object: "response"`, `status: "completed"`, populated `usage` and per-part `annotations` and `logprobs`) so SDKs and tests can share a single shape check. Each `output_text` content part carries an empty `logprobs` array (the mock produces no logprobs); the `response.output_text.delta`/`.done` events carry only `delta`/`text` and do not include `logprobs`, matching the upstream stream shape.

## Development

```bash
go test -race ./...
gofmt -w .
```

CI runs unit tests, Anthropic SDK integration tests against a running mock-vllm container, and OpenAI compatibility tests via the [openai-compatibility-tester](https://github.com/beranekio/openai-compatibility-tester) container on every push/PR to `main`. Successful merges to `main` also trigger a GHCR publish workflow (`.github/workflows/publish-docker.yml`).

## License

See [LICENSE](LICENSE).