# AGENTS.md

Guidance for human and AI contributors working in this repository.

## Project overview

- **Purpose**: Provide a mock HTTP server that mimics key [vLLM online serving](https://docs.vllm.ai/en/stable/serving/online_serving/) APIs (OpenAI + Anthropic Messages) for tests and local stacks—without loading models or requiring GPUs.
- **Language**: Go 1.24+
- **Entrypoint**: `cmd/mock-vllm/main.go`
- **HTTP handlers**: `pkg/handler/`
- **Response logic**: `pkg/text/`
- **Container**: `Dockerfile` (distroless, non-root, port 8000)

## Recommended workflow

1. Read `README.md` and the files you will change.
2. Keep changes focused on the requested task.
3. Match existing package layout and naming.
4. Run validation commands for the areas you touched (see below).
5. Update `README.md` when behavior or configuration changes.

## Validation commands

Run from the repository root:

```bash
go mod download
gofmt -w .
gofmt -l .    # should print nothing after formatting
go vet ./...
go test -race -count=1 ./...
CGO_ENABLED=0 go build -trimpath -o /tmp/mock-vllm ./cmd/mock-vllm
```

**SDK integration tests** (`integration/`, official OpenAI + Anthropic Go clients):

```bash
go test -race -count=1 ./integration/...
```

**Docker integration tests**:

```bash
docker build -t mock-vllm:local .
docker run -d --name mock-vllm -p 8000:8000 mock-vllm:local
INTEGRATION_BASE_URL=http://127.0.0.1:8000 go test -race -count=1 ./integration/...
docker rm -f mock-vllm
```

CI (`.github/workflows/ci.yml`) runs the same checks on Ubuntu for pushes and PRs to `main`.

## Editing conventions

- Preserve vLLM-default port `8000` and path prefixes (`/v1/...`) unless there is a strong reason to diverge.
- Keep mock responses deterministic and documented in `README.md`.
- Avoid unrelated refactors in the same change.
- Keep runtime dependencies minimal (`github.com/google/uuid` only). SDK deps belong in integration tests (`github.com/openai/openai-go`, `github.com/anthropics/anthropic-sdk-go`).
- Streaming handlers must emit valid SSE (`text/event-stream`) for OpenAI and Anthropic event shapes.

## Adding endpoints

When adding a new compatible route:

1. Register the path in `pkg/handler/handler.go`.
2. Implement handler logic in `pkg/handler/openai.go` or `anthropic.go` (or a new file if large).
3. Add unit tests in `pkg/handler/handler_test.go`.
4. Extend `integration/` tests and document in `README.md`.

## Agent-specific notes

### Opening pull requests

When creating a PR, add a GitHub label that identifies the agent or tooling that authored it when the repo uses such labels (e.g. `grok`, `cursor`, `claude`, `codex`).

Include in the PR description:

- What changed and why
- How it was validated (exact commands)

### Scope boundaries

- This repo is a **mock** server: do not add real inference, model downloads, or GPU code.
- Prefer extending existing response helpers in `pkg/text/` over duplicating string-matching logic.