package handler

import (
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/beranekio/mock-vllm/pkg/config"
	"github.com/beranekio/mock-vllm/pkg/httpjson"
	"github.com/beranekio/mock-vllm/pkg/text"
)

type Server struct {
	cfg config.Config
}

func New(cfg config.Config) *Server {
	return &Server{cfg: cfg}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if s.cfg.LogRequests {
		log.Printf("%s %s %s", r.RemoteAddr, r.Method, r.URL.Path)
	}

	switch r.Method {
	case http.MethodGet:
		s.handleGet(w, r)
	case http.MethodPost:
		s.handlePost(w, r)
	default:
		httpjson.Write(w, http.StatusNotFound, map[string]string{"error": "not found"})
	}
}

func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/health", "/healthz":
		httpjson.Write(w, http.StatusOK, map[string]string{"status": "ok"})
	case "/ping":
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("pong"))
	case "/version":
		httpjson.Write(w, http.StatusOK, map[string]string{
			"version": config.Version,
			"service": "mock-vllm",
		})
	case "/v1/models":
		s.listModels(w)
	default:
		// Handle GET /v1/models/{id}
		if strings.HasPrefix(r.URL.Path, "/v1/models/") {
			modelID := strings.TrimPrefix(r.URL.Path, "/v1/models/")
			if modelID != "" {
				s.getModel(w, modelID)
				return
			}
		}
		httpjson.Write(w, http.StatusNotFound, map[string]string{"error": "not found"})
	}
}

func (s *Server) handlePost(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		httpjson.Error(w, http.StatusBadRequest, "invalid body")
		return
	}

	if text.ShouldDelay(raw, s.cfg.SlowMarkers) {
		select {
		case <-time.After(s.cfg.SlowDelay):
		case <-r.Context().Done():
			return
		}
	}

	payload, err := text.ParsePayload(raw)
	if err != nil {
		if isAnthropicPath(r.URL.Path) {
			httpjson.AnthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON or encoding")
			return
		}
		httpjson.Error(w, http.StatusBadRequest, "invalid JSON or encoding")
		return
	}

	switch r.URL.Path {
	case "/v1/chat/completions":
		s.chatCompletions(w, payload)
	case "/v1/completions":
		s.completions(w, payload)
	case "/v1/embeddings":
		s.embeddings(w, payload)
	case "/v1/responses":
		s.responses(w, payload)
	case "/v1/responses/input_tokens":
		s.inputTokens(w, payload)
	case "/v1/messages":
		s.messages(w, payload)
	case "/v1/messages/count_tokens":
		s.countTokens(w, payload)
	default:
		httpjson.Write(w, http.StatusNotFound, map[string]string{"error": "not found"})
	}
}

func isAnthropicPath(path string) bool {
	return path == "/v1/messages" || path == "/v1/messages/count_tokens"
}

func (s *Server) listModels(w http.ResponseWriter) {
	httpjson.Write(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{
			{
				"id":       s.cfg.DefaultModel,
				"object":   "model",
				"owned_by": "mock-vllm",
				"created":  time.Now().Unix(),
			},
		},
	})
}

func (s *Server) getModel(w http.ResponseWriter, modelID string) {
	httpjson.Write(w, http.StatusOK, map[string]any{
		"id":       modelID,
		"object":   "model",
		"owned_by": "mock-vllm",
		"created":  time.Now().Unix(),
	})
}

func (s *Server) inputTokens(w http.ResponseWriter, payload map[string]any) {
	in, _ := text.Usage(text.ExtractTokenCountText(payload), "")
	httpjson.Write(w, http.StatusOK, map[string]any{
		"object":       "response.input_tokens",
		"input_tokens": in,
	})
}

func (s *Server) countTokens(w http.ResponseWriter, payload map[string]any) {
	in, _ := text.Usage(text.ExtractTokenCountText(payload), "")
	httpjson.Write(w, http.StatusOK, map[string]int{"input_tokens": in})
}
