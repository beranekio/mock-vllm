package text

import (
	"encoding/json"
	"strings"
)

// Reply derives mock assistant text from request payload content.
func Reply(payload map[string]any, prefix string) string {
	raw := ExtractInput(payload)
	lowered := strings.ToLower(raw)
	switch {
	case strings.Contains(lowered, "bye"):
		return prefix + "bye"
	case strings.Contains(lowered, "hi"), strings.Contains(lowered, "hello"):
		return prefix + "hi"
	case raw != "":
		return prefix + "ok"
	default:
		return prefix + "ok"
	}
}

// ExtractInput collects text from the latest user turn in OpenAI or Anthropic bodies.
func ExtractInput(payload map[string]any) string {
	if v, ok := payload["input"].(string); ok && v != "" {
		return v
	}
	if v, ok := payload["prompt"].(string); ok && v != "" {
		return v
	}
	if msgs, ok := payload["messages"].([]any); ok {
		return extractLastUserText(msgs)
	}
	if input, ok := payload["input"].([]any); ok {
		return extractFromInputArray(input)
	}
	return ""
}

// ExtractEmbeddingInputs returns one string per embedding input (string or array).
func ExtractEmbeddingInputs(payload map[string]any) []string {
	switch v := payload["input"].(type) {
	case string:
		if v != "" {
			return []string{v}
		}
	case []any:
		var inputs []string
		for _, item := range v {
			if s, ok := item.(string); ok && s != "" {
				inputs = append(inputs, s)
			}
		}
		if len(inputs) > 0 {
			return inputs
		}
	}
	text := ExtractInput(payload)
	if text == "" {
		text = "mock"
	}
	return []string{text}
}

func extractFromInputArray(items []any) string {
	if len(items) == 0 {
		return ""
	}
	if _, ok := items[0].(map[string]any); ok {
		return extractLastUserText(items)
	}
	return ""
}

func extractLastUserText(msgs []any) string {
	var last string
	for _, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role != "user" {
			continue
		}
		if t := extractMessageContent(msg["content"]); t != "" {
			last = t
		}
	}
	return last
}

func extractMessageContent(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		return extractContentBlocks(c)
	default:
		return ""
	}
}

func extractContentBlocks(blocks []any) string {
	var parts []string
	for _, block := range blocks {
		b, ok := block.(map[string]any)
		if !ok {
			continue
		}
		switch blockType, _ := b["type"].(string); blockType {
		case "text", "input_text":
			if t, _ := b["text"].(string); t != "" {
				parts = append(parts, t)
			}
		}
	}
	return strings.Join(parts, "\n")
}

// ShouldDelay returns true when the raw body contains any configured slow marker.
func ShouldDelay(raw []byte, markers []string) bool {
	text := strings.ToLower(string(raw))
	for _, marker := range markers {
		if strings.Contains(text, strings.ToLower(marker)) {
			return true
		}
	}
	return false
}

// Model returns the model field from a payload or the default.
func Model(payload map[string]any, defaultModel string) string {
	if model, _ := payload["model"].(string); model != "" {
		return model
	}
	return defaultModel
}

// StreamRequested reports whether the client asked for SSE streaming.
func StreamRequested(payload map[string]any) bool {
	if v, ok := payload["stream"].(bool); ok {
		return v
	}
	return false
}

// Usage estimates token counts for mock usage blocks.
func Usage(input string, output string) (inputTokens, outputTokens int) {
	inputTokens = max(1, len(input)/4)
	outputTokens = max(1, len(output)/4)
	return inputTokens, outputTokens
}

// ParsePayload unmarshals JSON into a map, defaulting to empty on nil body.
func ParsePayload(raw []byte) (map[string]any, error) {
	if len(raw) == 0 {
		raw = []byte("{}")
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	return payload, nil
}
