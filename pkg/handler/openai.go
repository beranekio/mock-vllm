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
	prompts := text.ExtractCompletionPrompts(payload)

	if text.StreamRequested(payload) {
		s.streamOpenAICompletion(w, model, prompts)
		return
	}

	choices := make([]map[string]any, len(prompts))
	var totalIn, totalOut int
	for i, prompt := range prompts {
		reply := text.ReplyText(prompt, s.cfg.ResponsePrefix)
		inTok, outTok := text.Usage(prompt, reply)
		totalIn += inTok
		totalOut += outTok
		choices[i] = map[string]any{
			"index":         i,
			"text":          reply,
			"finish_reason": "stop",
		}
	}

	httpjson.Write(w, http.StatusOK, map[string]any{
		"id":      "cmpl-" + idSuffix(),
		"object":  "text_completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": choices,
		"usage": map[string]int{
			"prompt_tokens":     totalIn,
			"completion_tokens": totalOut,
			"total_tokens":      totalIn + totalOut,
		},
	})
}

func (s *Server) embeddings(w http.ResponseWriter, payload map[string]any) {
	model := text.Model(payload, s.cfg.DefaultModel)
	inputs := text.ExtractEmbeddingInputs(payload)

	dim := 8
	data := make([]map[string]any, len(inputs))
	for i, input := range inputs {
		vec := make([]float64, dim)
		seed := input.Seed()
		for j := range vec {
			vec[j] = float64((int(seed)+j)%7) / 7.0
		}
		data[i] = map[string]any{
			"object":    "embedding",
			"index":     i,
			"embedding": vec,
		}
	}

	httpjson.Write(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"model":  model,
		"usage": map[string]int{
			"prompt_tokens": len(inputs) * 4,
			"total_tokens":  len(inputs) * 4,
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

	chunks := text.Chunk(reply, 4)
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

func (s *Server) streamOpenAICompletion(w http.ResponseWriter, model string, prompts []string) {
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

	for i, prompt := range prompts {
		reply := text.ReplyText(prompt, s.cfg.ResponsePrefix)
		chunks := text.Chunk(reply, 4)
		if len(chunks) == 0 {
			chunks = []string{""}
		}
		for _, part := range chunks {
			writeSSE(w, map[string]any{
				"id": id, "object": "text_completion", "model": model,
				"choices": []map[string]any{{
					"index": i, "text": part,
				}},
			})
			flusher.Flush()
		}
		writeSSE(w, map[string]any{
			"id": id, "object": "text_completion", "model": model,
			"choices": []map[string]any{{
				"index": i, "text": "", "finish_reason": "stop",
			}},
		})
		flusher.Flush()
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func idSuffix() string {
	return strings.ReplaceAll(uuid.New().String(), "-", "")
}
