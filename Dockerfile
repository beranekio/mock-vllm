FROM golang:1.24 AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /mock-vllm ./cmd/mock-vllm

FROM gcr.io/distroless/static-debian13:nonroot
WORKDIR /app
COPY --from=builder /mock-vllm /app/mock-vllm

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEFAULT_MODEL=mock-model

EXPOSE 8000
USER nonroot:nonroot
ENTRYPOINT ["/app/mock-vllm"]