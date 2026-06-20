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
	var totalTokens int
	for i, input := range inputs {
		totalTokens += input.TokenCount()
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
			"prompt_tokens": totalTokens,
			"total_tokens":  totalTokens,
		},
	})
}

func (s *Server) responses(w http.ResponseWriter, payload map[string]any) {
	model := text.Model(payload, s.cfg.DefaultModel)
	input := text.ExtractInput(payload)
	reply := text.Reply(payload, s.cfg.ResponsePrefix)
	// Count the full prompt (system + all messages + all input items) for
	// usage; pick the latest user text for the reply choice.
	tokenInput := text.ExtractTokenCountText(payload)
	if tokenInput == "" {
		tokenInput = input
	}
	inTok, outTok := text.Usage(tokenInput, reply)

	if text.StreamRequested(payload) {
		s.streamResponses(w, model, reply, inTok, outTok)
		return
	}

	id := "resp_" + idSuffix()
	created := time.Now().Unix()
	itemID := "msg_" + idSuffix()
	item := map[string]any{
		"id":     itemID,
		"type":   "message",
		"role":   "assistant",
		"status": "completed",
		"content": []map[string]any{
			{
				"type":        "output_text",
				"text":        reply,
				"annotations": []any{},
				"logprobs":    []any{},
			},
		},
	}

	httpjson.Write(w, http.StatusOK, s.responseEnvelope(id, model, created, "completed", []map[string]any{item}, map[string]any{
		"input_tokens":          inTok,
		"input_tokens_details":  map[string]any{"cached_tokens": 0},
		"output_tokens":         outTok,
		"output_tokens_details": map[string]any{"reasoning_tokens": 0},
		"total_tokens":          inTok + outTok,
	}))
}

// defaultMaxOutputTokens is the value reported on the Response envelope's
// max_output_tokens field. The mock does not enforce a real cap; this mirrors
// the integer that vLLM populates from its resolved sampling params.
const defaultMaxOutputTokens = 4096

// responseEnvelope builds the /v1/responses Response object with the default
// optional fields populated. It mirrors the fields vLLM's ResponsesResponse
// model emits (vllm/entrypoints/openai/responses/protocol.py): every field
// vLLM marks required is present here, and fields vLLM omits (error, store,
// completed_at, conversation) are omitted too. status/output/usage vary
// across the response lifecycle (in_progress snapshots vs the terminal
// completed event), so they are parameters.
func (s *Server) responseEnvelope(id, model string, createdAt int64, status string, output []map[string]any, usage map[string]any) map[string]any {
	envelope := map[string]any{
		"id":                  id,
		"object":              "response",
		"created_at":          createdAt,
		"status":              status,
		"model":               model,
		"output":              output,
		"parallel_tool_calls": true,
		"tool_choice":         "auto",
		"tools":               []any{},
		"temperature":         1.0,
		"top_p":               1.0,
		"max_output_tokens":   defaultMaxOutputTokens,
		"background":          false,
		"truncation":          "disabled",
		"service_tier":        "auto",
		// Nullable fields vLLM emits as None by default.
		"instructions":       nil,
		"metadata":           nil,
		"incomplete_details": nil,
		"max_tool_calls":     nil,
		"user":               nil,
	}
	if usage != nil {
		envelope["usage"] = usage
	}
	return envelope
}

func (s *Server) streamResponses(w http.ResponseWriter, model, reply string, inTok, outTok int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		httpjson.Error(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	id := "resp_" + idSuffix()
	itemID := "msg_" + idSuffix()
	created := time.Now().Unix()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	chunks := text.Chunk(reply, 4)
	if len(chunks) == 0 {
		chunks = []string{""}
	}

	// Full message item used both in the lifecycle events and in the
	// terminal response.completed event.
	item := map[string]any{
		"id":     itemID,
		"type":   "message",
		"role":   "assistant",
		"status": "completed",
		"content": []map[string]any{
			{
				"type":        "output_text",
				"text":        reply,
				"annotations": []any{},
				"logprobs":    []any{},
			},
		},
	}

	// OpenAI Responses streams start response.created at sequence_number 0
	// and increment by one for each subsequent event.
	seq := 0
	next := func() int {
		n := seq
		seq++
		return n
	}

	// response.created
	writeSSE(w, map[string]any{
		"type":            "response.created",
		"sequence_number": next(),
		"response":        s.responseEnvelope(id, model, created, "in_progress", []map[string]any{}, nil),
	})
	flusher.Flush()

	// response.in_progress — emitted between created and the first
	// output_item.added, matching the documented OpenAI Responses lifecycle.
	// The payload mirrors response.created (status=in_progress, empty
	// output) so clients that subscribe to the in_progress transition can
	// observe the response still has no output yet.
	writeSSE(w, map[string]any{
		"type":            "response.in_progress",
		"sequence_number": next(),
		"response":        s.responseEnvelope(id, model, created, "in_progress", []map[string]any{}, nil),
	})
	flusher.Flush()

	// response.output_item.added
	itemInProgress := map[string]any{
		"id":      itemID,
		"type":    "message",
		"role":    "assistant",
		"status":  "in_progress",
		"content": []any{},
	}
	writeSSE(w, map[string]any{
		"type":            "response.output_item.added",
		"sequence_number": next(),
		"output_index":    0,
		"item":            itemInProgress,
	})
	flusher.Flush()

	// response.content_part.added
	partAdded := map[string]any{
		"type":        "output_text",
		"text":        "",
		"annotations": []any{},
		"logprobs":    []any{},
	}
	writeSSE(w, map[string]any{
		"type":            "response.content_part.added",
		"sequence_number": next(),
		"item_id":         itemID,
		"output_index":    0,
		"content_index":   0,
		"part":            partAdded,
	})
	flusher.Flush()

	// response.output_text.delta, one per chunk.
	for _, part := range chunks {
		writeSSE(w, map[string]any{
			"type":            "response.output_text.delta",
			"sequence_number": next(),
			"item_id":         itemID,
			"output_index":    0,
			"content_index":   0,
			"delta":           part,
		})
		flusher.Flush()
	}

	// response.output_text.done
	writeSSE(w, map[string]any{
		"type":            "response.output_text.done",
		"sequence_number": next(),
		"item_id":         itemID,
		"output_index":    0,
		"content_index":   0,
		"text":            reply,
		"logprobs":        []any{},
	})
	flusher.Flush()

	// response.content_part.done
	writeSSE(w, map[string]any{
		"type":            "response.content_part.done",
		"sequence_number": next(),
		"item_id":         itemID,
		"output_index":    0,
		"content_index":   0,
		"part": map[string]any{
			"type":        "output_text",
			"text":        reply,
			"annotations": []any{},
			"logprobs":    []any{},
		},
	})
	flusher.Flush()

	// response.output_item.done
	writeSSE(w, map[string]any{
		"type":            "response.output_item.done",
		"sequence_number": next(),
		"output_index":    0,
		"item":            item,
	})
	flusher.Flush()

	// response.completed
	writeSSE(w, map[string]any{
		"type":            "response.completed",
		"sequence_number": next(),
		"response": s.responseEnvelope(id, model, created, "completed", []map[string]any{item}, map[string]any{
			"input_tokens":          inTok,
			"input_tokens_details":  map[string]any{"cached_tokens": 0},
			"output_tokens":         outTok,
			"output_tokens_details": map[string]any{"reasoning_tokens": 0},
			"total_tokens":          inTok + outTok,
		}),
	})
	flusher.Flush()

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
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

	created := time.Now().Unix()
	chunks := text.Chunk(reply, 4)
	if len(chunks) == 0 {
		chunks = []string{""}
	}

	writeSSE(w, map[string]any{
		"id": id, "object": "chat.completion.chunk", "model": model, "created": created,
		"choices": []map[string]any{{
			"index": 0,
			"delta": map[string]string{"role": "assistant", "content": chunks[0]},
		}},
	})
	flusher.Flush()

	for _, part := range chunks[1:] {
		writeSSE(w, map[string]any{
			"id": id, "object": "chat.completion.chunk", "model": model, "created": created,
			"choices": []map[string]any{{
				"index": 0, "delta": map[string]string{"content": part},
			}},
		})
		flusher.Flush()
	}

	writeSSE(w, map[string]any{
		"id": id, "object": "chat.completion.chunk", "model": model, "created": created,
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
	created := time.Now().Unix()
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
				"id": id, "object": "text_completion", "model": model, "created": created,
				"choices": []map[string]any{{
					"index": i, "text": part,
				}},
			})
			flusher.Flush()
		}
		writeSSE(w, map[string]any{
			"id": id, "object": "text_completion", "model": model, "created": created,
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
