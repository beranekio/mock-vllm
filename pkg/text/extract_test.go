package text

import "testing"

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
