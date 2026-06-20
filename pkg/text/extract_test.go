package text

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestReply(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"say hi", "hi"},
		{"hello world", "hi"},
		{"goodbye", "bye"},
		{"random", "ok"},
		{"", "ok"},
	}
	for _, tc := range tests {
		got := Reply(map[string]any{"input": tc.input}, "")
		if got != tc.want {
			t.Errorf("Reply(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestExtractInput_messages(t *testing.T) {
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hello there"},
		},
	}
	if got := ExtractInput(payload); got != "hello there" {
		t.Fatalf("ExtractInput() = %q", got)
	}
}

func TestExtractInput_lastUserOnly(t *testing.T) {
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "assistant", "content": "goodbye"},
			map[string]any{"role": "user", "content": "hi"},
		},
	}
	if got := Reply(payload, ""); got != "hi" {
		t.Fatalf("Reply() = %q, want hi", got)
	}
}

func TestExtractInput_responsesStructured(t *testing.T) {
	payload := map[string]any{
		"input": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "hi"},
				},
			},
		},
	}
	if got := Reply(payload, ""); got != "hi" {
		t.Fatalf("Reply() = %q, want hi", got)
	}
}

func TestExtractCompletionPrompts_batch(t *testing.T) {
	payload := map[string]any{
		"prompt": []any{"hi", "bye"},
	}
	got := ExtractCompletionPrompts(payload)
	if len(got) != 2 || got[0] != "hi" || got[1] != "bye" {
		t.Fatalf("ExtractCompletionPrompts() = %v", got)
	}
	if ReplyText(got[0], "") != "hi" || ReplyText(got[1], "") != "bye" {
		t.Fatalf("unexpected replies for batch prompts")
	}
}

func TestExtractEmbeddingInputs_batch(t *testing.T) {
	payload := map[string]any{
		"input": []any{"a", "b"},
	}
	got := ExtractEmbeddingInputs(payload)
	if len(got) != 2 || got[0].Text != "a" || got[1].Text != "b" {
		t.Fatalf("ExtractEmbeddingInputs() = %v", got)
	}
}

func TestExtractEmbeddingInputs_tokenArray(t *testing.T) {
	payload := map[string]any{
		"input": []any{float64(1), int(2), int64(3)},
	}
	got := ExtractEmbeddingInputs(payload)
	if len(got) != 1 || len(got[0].Tokens) != 3 || got[0].Tokens[0] != 1 || got[0].Tokens[1] != 2 || got[0].Tokens[2] != 3 {
		t.Fatalf("ExtractEmbeddingInputs() = %v", got)
	}
	if got[0].TokenCount() != 3 {
		t.Fatalf("TokenCount() = %d, want 3", got[0].TokenCount())
	}
}

func TestExtractEmbeddingInputs_tokenBatches(t *testing.T) {
	payload := map[string]any{
		"input": []any{
			[]any{float64(1), float64(2)},
			[]any{float64(3), float64(4)},
		},
	}
	got := ExtractEmbeddingInputs(payload)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	if len(got[0].Tokens) != 2 || got[0].Tokens[0] != 1 || got[0].Tokens[1] != 2 {
		t.Fatalf("batch[0] = %v", got[0].Tokens)
	}
	if len(got[1].Tokens) != 2 || got[1].Tokens[0] != 3 || got[1].Tokens[1] != 4 {
		t.Fatalf("batch[1] = %v", got[1].Tokens)
	}
}

func TestChunk_utf8(t *testing.T) {
	const s = "日本語"
	chunks := Chunk(s, 2)
	if len(chunks) != 2 {
		t.Fatalf("len(chunks) = %d, want 2", len(chunks))
	}
	if chunks[0]+chunks[1] != s {
		t.Fatalf("chunks = %q + %q", chunks[0], chunks[1])
	}
	for _, c := range chunks {
		if !utf8.ValidString(c) {
			t.Fatalf("invalid UTF-8 chunk: %q", c)
		}
	}
}

func TestExtractTokenCountText_includesSystemAndHistory(t *testing.T) {
	longSystem := strings.Repeat("a", 400)
	payload := map[string]any{
		"system": longSystem,
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": "bye"},
			map[string]any{"role": "user", "content": "again"},
		},
	}
	got := ExtractTokenCountText(payload)
	if !strings.Contains(got, longSystem) {
		t.Fatal("missing system text")
	}
	if !strings.Contains(got, "bye") {
		t.Fatal("missing assistant history")
	}
	full, _ := Usage(got, "")
	lastUser, _ := Usage(ExtractInput(payload), "")
	if full <= lastUser {
		t.Fatalf("full count %d should exceed last-user-only %d", full, lastUser)
	}
}

func TestExtractTokenCountText_scalesWithInputSize(t *testing.T) {
	inShort, _ := Usage(ExtractTokenCountText(map[string]any{"input": "hi"}), "")
	inLong, _ := Usage(ExtractTokenCountText(map[string]any{"input": strings.Repeat("z", 200)}), "")
	if inLong <= inShort {
		t.Fatalf("long=%d short=%d", inLong, inShort)
	}
}

func TestExtractTokenCountText_includesOutputTextBlocks(t *testing.T) {
	// When a client passes a prior assistant message back in a structured
	// input, the part type is `output_text` (not `text` or `input_text`).
	// Token counting must include those blocks so multi-turn Responses
	// histories report the same usage as the input_tokens endpoint.
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "output_text", "text": strings.Repeat("y", 200)},
				},
			},
			map[string]any{"role": "user", "content": "again"},
		},
	}
	got := ExtractTokenCountText(payload)
	if !strings.Contains(got, strings.Repeat("y", 200)) {
		t.Fatalf("missing output_text block: %q", got)
	}
	full, _ := Usage(got, "")
	lastUser, _ := Usage(ExtractInput(payload), "")
	if full <= lastUser {
		t.Fatalf("full count %d should exceed last-user-only %d (output_text blocks are being skipped)", full, lastUser)
	}
}

func TestExtractTokenCountText_includesInstructions(t *testing.T) {
	// Responses API uses `instructions` (not `system`) for the developer
	// prompt. It can be a plain string or an array of content parts.
	// Token counting must include it so /v1/responses and
	// /v1/responses/input_tokens reflect the full prompt.
	long := strings.Repeat("i", 400)
	stringPayload := map[string]any{
		"instructions": long,
		"input":        "hi",
	}
	gotString := ExtractTokenCountText(stringPayload)
	if !strings.Contains(gotString, long) {
		t.Fatalf("missing string instructions: %q", gotString)
	}
	withInstructions, _ := Usage(gotString, "")
	withoutInstructions, _ := Usage(ExtractTokenCountText(map[string]any{"input": "hi"}), "")
	if withInstructions <= withoutInstructions {
		t.Fatalf("with instructions=%d, without=%d (want with > without)", withInstructions, withoutInstructions)
	}

	structuredPayload := map[string]any{
		"instructions": []any{
			map[string]any{"type": "input_text", "text": strings.Repeat("j", 300)},
		},
		"input": "hi",
	}
	gotStructured := ExtractTokenCountText(structuredPayload)
	if !strings.Contains(gotStructured, strings.Repeat("j", 300)) {
		t.Fatalf("missing structured instructions: %q", gotStructured)
	}
}

func TestShouldDelay(t *testing.T) {
	if !ShouldDelay([]byte(`{"input":"tell me about otters"}`), []string{"otter"}) {
		t.Fatal("expected slow delay marker match")
	}
	if ShouldDelay([]byte(`{"input":"fast"}`), []string{"otter"}) {
		t.Fatal("unexpected slow delay marker match")
	}
}

func TestModel_default(t *testing.T) {
	if got := Model(map[string]any{}, "default"); got != "default" {
		t.Fatalf("Model() = %q", got)
	}
}
