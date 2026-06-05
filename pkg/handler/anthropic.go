package handler

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/beranekio/mock-vllm/pkg/httpjson"
	"github.com/beranekio/mock-vllm/pkg/text"
)

func (s *Server) messages(w http.ResponseWriter, payload map[string]any) {
	model := text.Model(payload, s.cfg.DefaultModel)
	reply := text.Reply(payload, s.cfg.ResponsePrefix)
	input := text.ExtractInput(payload)
	inTok, outTok := text.Usage(input, reply)

	if text.StreamRequested(payload) {
		s.streamAnthropicMessages(w, model, reply, inTok, outTok)
		return
	}

	httpjson.Write(w, http.StatusOK, map[string]any{
		"id":            "msg_" + idSuffix(),
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"content": []map[string]string{
			{"type": "text", "text": reply},
		},
		"usage": map[string]int{
			"input_tokens":  inTok,
			"output_tokens": outTok,
		},
	})
}

func (s *Server) streamAnthropicMessages(w http.ResponseWriter, model, reply string, inTok, outTok int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		httpjson.AnthropicError(w, http.StatusInternalServerError, "api_error", "streaming not supported")
		return
	}

	msgID := "msg_" + idSuffix()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	writeAnthropicEvent(w, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":    msgID,
			"type":  "message",
			"role":  "assistant",
			"model": model,
			"usage": map[string]int{"input_tokens": inTok, "output_tokens": 0},
		},
	})
	flusher.Flush()

	writeAnthropicEvent(w, "content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": 0,
		"content_block": map[string]any{
			"type": "text",
			"text": "",
		},
	})
	flusher.Flush()

	for _, part := range text.Chunk(reply, 4) {
		writeAnthropicEvent(w, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]string{
				"type": "text_delta",
				"text": part,
			},
		})
		flusher.Flush()
	}

	writeAnthropicEvent(w, "content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": 0,
	})
	flusher.Flush()

	writeAnthropicEvent(w, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]string{
			"stop_reason":   "end_turn",
			"stop_sequence": "",
		},
		"usage": map[string]int{"output_tokens": outTok},
	})
	flusher.Flush()

	writeAnthropicEvent(w, "message_stop", map[string]any{
		"type": "message_stop",
	})
	flusher.Flush()
}

func writeAnthropicEvent(w http.ResponseWriter, event string, data map[string]any) {
	encoded, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, encoded)
}
