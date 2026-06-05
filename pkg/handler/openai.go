package handler

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/beranekio/mock-vllm/pkg/httpjson"
	"github.com/beranekio/mock-vllm/pkg/text"
)

func (s *Server) chatCompletions(w http.ResponseWriter, payload map[string]any) {
	model := text.Model(payload, s.cfg.DefaultModel)
	reply := text.Reply(payload, s.cfg.ResponsePrefix)
	input := text.ExtractInput(payload)
	inTok, outTok := text.Usage(input, reply)

	if text.StreamRequested(payload) {
		s.streamOpenAIChat(w, model, reply)
		return
	}

	httpjson.Write(w, http.StatusOK, map[string]any{
		"id":      "chatcmpl-" + idSuffix(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": reply,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     inTok,
			"completion_tokens": outTok,
			"total_tokens":      inTok + outTok,
		},
	})
}

func (s *Server) completions(w http.ResponseWriter, payload map[string]any) {
	model := text.Model(payload, s.cfg.DefaultModel)
	reply := text.Reply(payload, s.cfg.ResponsePrefix)
	input := text.ExtractInput(payload)
	inTok, outTok := text.Usage(input, reply)

	if text.StreamRequested(payload) {
		s.streamOpenAICompletion(w, model, reply)
		return
	}

	httpjson.Write(w, http.StatusOK, map[string]any{
		"id":      "cmpl-" + idSuffix(),
		"object":  "text_completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{
				"index":         0,
				"text":          reply,
				"finish_reason": "stop",
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     inTok,
			"completion_tokens": outTok,
			"total_tokens":      inTok + outTok,
		},
	})
}

func (s *Server) embeddings(w http.ResponseWriter, payload map[string]any) {
	model := text.Model(payload, s.cfg.DefaultModel)
	input := text.ExtractInput(payload)
	if input == "" {
		input = "mock"
	}

	dim := 8
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = float64((int(input[0])+i)%7) / 7.0
	}

	httpjson.Write(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{
			{
				"object":    "embedding",
				"index":     0,
				"embedding": vec,
			},
		},
		"model": model,
		"usage": map[string]int{
			"prompt_tokens": 4,
			"total_tokens":  4,
		},
	})
}

func (s *Server) responses(w http.ResponseWriter, payload map[string]any) {
	model := text.Model(payload, s.cfg.DefaultModel)
	reply := text.Reply(payload, s.cfg.ResponsePrefix)

	httpjson.Write(w, http.StatusOK, map[string]any{
		"id":     "resp_" + idSuffix(),
		"object": "response",
		"status": "completed",
		"model":  model,
		"output": []map[string]any{
			{
				"type": "message",
				"role": "assistant",
				"content": []map[string]string{
					{"type": "output_text", "text": reply},
				},
			},
		},
	})
}

func (s *Server) streamOpenAIChat(w http.ResponseWriter, model, reply string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		httpjson.Error(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	id := "chatcmpl-" + idSuffix()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	chunks := chunkText(reply, 4)
	if len(chunks) == 0 {
		chunks = []string{""}
	}

	writeSSE(w, map[string]any{
		"id": id, "object": "chat.completion.chunk", "model": model,
		"choices": []map[string]any{{
			"index": 0,
			"delta": map[string]string{"role": "assistant", "content": chunks[0]},
		}},
	})
	flusher.Flush()

	for _, part := range chunks[1:] {
		writeSSE(w, map[string]any{
			"id": id, "object": "chat.completion.chunk", "model": model,
			"choices": []map[string]any{{
				"index": 0, "delta": map[string]string{"content": part},
			}},
		})
		flusher.Flush()
	}

	writeSSE(w, map[string]any{
		"id": id, "object": "chat.completion.chunk", "model": model,
		"choices": []map[string]any{{
			"index": 0, "delta": map[string]any{}, "finish_reason": "stop",
		}},
	})
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) streamOpenAICompletion(w http.ResponseWriter, model, reply string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		httpjson.Error(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	id := "cmpl-" + idSuffix()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	for _, part := range chunkText(reply, 4) {
		writeSSE(w, map[string]any{
			"id": id, "object": "text_completion", "model": model,
			"choices": []map[string]any{{
				"index": 0, "text": part,
			}},
		})
		flusher.Flush()
	}

	writeSSE(w, map[string]any{
		"id": id, "object": "text_completion", "model": model,
		"choices": []map[string]any{{
			"index": 0, "text": "", "finish_reason": "stop",
		}},
	})
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func idSuffix() string {
	return strings.ReplaceAll(uuid.New().String(), "-", "")
}

func chunkText(s string, size int) []string {
	if s == "" {
		return nil
	}
	if size < 1 {
		size = 1
	}
	var chunks []string
	for i := 0; i < len(s); i += size {
		end := i + size
		if end > len(s) {
			end = len(s)
		}
		chunks = append(chunks, s[i:end])
	}
	return chunks
}
