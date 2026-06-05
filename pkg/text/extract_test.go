package text

import (
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

func TestExtractEmbeddingInputs_batch(t *testing.T) {
	payload := map[string]any{
		"input": []any{"a", "b"},
	}
	got := ExtractEmbeddingInputs(payload)
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("ExtractEmbeddingInputs() = %v", got)
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
