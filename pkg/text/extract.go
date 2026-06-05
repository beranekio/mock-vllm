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

// ExtractInput collects user-visible text from OpenAI or Anthropic request bodies.
func ExtractInput(payload map[string]any) string {
	if v, ok := payload["input"].(string); ok && v != "" {
		return v
	}
	if v, ok := payload["prompt"].(string); ok && v != "" {
		return v
	}
	if msgs, ok := payload["messages"].([]any); ok {
		return extractFromMessages(msgs)
	}
	return ""
}

func extractFromMessages(msgs []any) string {
	var parts []string
	for _, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		content := msg["content"]
		switch c := content.(type) {
		case string:
			if c != "" {
				parts = append(parts, c)
			}
		case []any:
			for _, block := range c {
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
