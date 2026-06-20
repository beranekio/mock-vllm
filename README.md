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

**OpenAI-compatible API** tests use the [openai-compatibility-tester](https://github.com/beranekio/openai-compatibility-tester) container to validate all OpenAI endpoints:

```bash
# Run all OpenAI compatibility tests against a local container or binary
docker run --rm \
  -e OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  -e OPENAI_API_KEY=test-key \
  -e OPENAI_MODEL=mock-model \
  -e TEST_SUITES=extended \
  ghcr.io/beranekio/openai-compatibility-tester:latest
```

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

## Development

```bash
go test -race ./...
gofmt -w .
```

CI runs unit tests, Anthropic SDK integration tests (in-process), and OpenAI compatibility tests via the [openai-compatibility-tester](https://github.com/beranekio/openai-compatibility-tester) container on every push/PR to `main`. Successful merges to `main` also trigger a GHCR publish workflow (`.github/workflows/publish-docker.yml`).

## License

See [LICENSE](LICENSE).