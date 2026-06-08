package text

import (
	"encoding/json"
	"strings"
)

// Reply derives mock assistant text from request payload content.
func Reply(payload map[string]any, prefix string) string {
	return ReplyText(ExtractInput(payload), prefix)
}

// ReplyText applies deterministic mock reply rules to a single user text string.
func ReplyText(raw, prefix string) string {
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

// ExtractCompletionPrompts returns one string per completions prompt (string or array).
func ExtractCompletionPrompts(payload map[string]any) []string {
	switch v := payload["prompt"].(type) {
	case string:
		if v != "" {
			return []string{v}
		}
	case []any:
		var prompts []string
		for _, item := range v {
			if s, ok := item.(string); ok && s != "" {
				prompts = append(prompts, s)
			}
		}
		if len(prompts) > 0 {
			return prompts
		}
	}
	return []string{"mock"}
}

// EmbeddingInput is one element of an embeddings request (text or token IDs).
type EmbeddingInput struct {
	Text   string
	Tokens []int
}

// Seed returns a deterministic byte for mock vector generation.
func (e EmbeddingInput) Seed() byte {
	if e.Text != "" {
		return e.Text[0]
	}
	if len(e.Tokens) > 0 {
		return byte((e.Tokens[0] + len(e.Tokens)) % 256)
	}
	return 'm'
}

// ExtractEmbeddingInputs returns one element per embedding input (string, token, or batched).
func ExtractEmbeddingInputs(payload map[string]any) []EmbeddingInput {
	switch v := payload["input"].(type) {
	case string:
		if v != "" {
			return []EmbeddingInput{{Text: v}}
		}
	case []any:
		if len(v) == 0 {
			break
		}
		if _, ok := v[0].([]any); ok {
			if inputs := extractEmbeddingTokenBatches(v); len(inputs) > 0 {
				return inputs
			}
		}
		if tokens, ok := parseTokenSlice(v); ok {
			return []EmbeddingInput{{Tokens: tokens}}
		}
		var inputs []EmbeddingInput
		for _, item := range v {
			if s, ok := item.(string); ok && s != "" {
				inputs = append(inputs, EmbeddingInput{Text: s})
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
	return []EmbeddingInput{{Text: text}}
}

func extractEmbeddingTokenBatches(items []any) []EmbeddingInput {
	var inputs []EmbeddingInput
	for _, item := range items {
		tokens, ok := parseTokenSlice(item)
		if !ok {
			return nil
		}
		inputs = append(inputs, EmbeddingInput{Tokens: tokens})
	}
	return inputs
}

func parseTokenSlice(v any) ([]int, bool) {
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}
	tokens := make([]int, len(arr))
	for i, item := range arr {
		n, ok := item.(float64)
		if !ok {
			return nil, false
		}
		tokens[i] = int(n)
	}
	return tokens, true
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

// ExtractTokenCountText collects all prompt text for token-count endpoints.
func ExtractTokenCountText(payload map[string]any) string {
	var parts []string

	if sys := extractSystem(payload["system"]); sys != "" {
		parts = append(parts, sys)
	}
	if v, ok := payload["input"].(string); ok && v != "" {
		parts = append(parts, v)
	}
	if v, ok := payload["prompt"].(string); ok && v != "" {
		parts = append(parts, v)
	}
	if msgs, ok := payload["messages"].([]any); ok {
		if t := extractAllMessagesText(msgs); t != "" {
			parts = append(parts, t)
		}
	}
	if input, ok := payload["input"].([]any); ok {
		parts = append(parts, extractTokenCountFromInputArray(input)...)
	}
	if prompts, ok := payload["prompt"].([]any); ok {
		for _, item := range prompts {
			if s, ok := item.(string); ok && s != "" {
				parts = append(parts, s)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func extractTokenCountFromInputArray(items []any) []string {
	if len(items) == 0 {
		return nil
	}
	if _, ok := items[0].(map[string]any); ok {
		if t := extractAllMessagesText(items); t != "" {
			return []string{t}
		}
		return nil
	}
	var parts []string
	for _, item := range items {
		if s, ok := item.(string); ok && s != "" {
			parts = append(parts, s)
		}
	}
	return parts
}

func extractSystem(system any) string {
	switch s := system.(type) {
	case string:
		return s
	case []any:
		return extractContentBlocks(s)
	default:
		return ""
	}
}

func extractAllMessagesText(msgs []any) string {
	var parts []string
	for _, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if t := extractMessageContent(msg["content"]); t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, "\n")
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
