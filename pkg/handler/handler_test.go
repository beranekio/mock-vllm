package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/beranekio/mock-vllm/pkg/config"
)

func newTestServer() *Server {
	return New(config.Config{DefaultModel: "test-model"})
}

func TestHealth(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestModels(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var body map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	data, ok := body["data"].([]any)
	if !ok || len(data) == 0 {
		t.Fatalf("unexpected models body: %v", body)
	}
}

func TestChatCompletions(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	choices, ok := resp["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("missing choices: %v", resp)
	}
}

func TestChatCompletions_multiTurn(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","messages":[
		{"role":"assistant","content":"goodbye"},
		{"role":"user","content":"hi"}
	]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	choice := resp["choices"].([]any)[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	if msg["content"] != "hi" {
		t.Fatalf("content = %v, want hi", msg["content"])
	}
}

func TestCompletions_batch(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","prompt":["hi","bye"]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	choices, ok := resp["choices"].([]any)
	if !ok || len(choices) != 2 {
		t.Fatalf("choices = %v, want 2", resp["choices"])
	}
	c0 := choices[0].(map[string]any)
	c1 := choices[1].(map[string]any)
	if c0["text"] != "hi" || c1["text"] != "bye" {
		t.Fatalf("texts = %v, %v", c0["text"], c1["text"])
	}
	if c0["index"].(float64) != 0 || c1["index"].(float64) != 1 {
		t.Fatalf("indices = %v, %v", c0["index"], c1["index"])
	}
}

func TestEmbeddings_batch(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","input":["a","b"]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	data, ok := resp["data"].([]any)
	if !ok || len(data) != 2 {
		t.Fatalf("data = %v, want 2 embeddings", resp["data"])
	}
}

func TestEmbeddings_tokenBatches(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","input":[[1,2],[3,4]]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	data, ok := resp["data"].([]any)
	if !ok || len(data) != 2 {
		t.Fatalf("data = %v, want 2 embeddings", resp["data"])
	}
	d0 := data[0].(map[string]any)
	d1 := data[1].(map[string]any)
	if d0["index"].(float64) != 0 || d1["index"].(float64) != 1 {
		t.Fatalf("indices = %v, %v", d0["index"], d1["index"])
	}
	usage := resp["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) != 4 || usage["total_tokens"].(float64) != 4 {
		t.Fatalf("usage = %v, want 4 prompt/total tokens", usage)
	}
}

func TestEmbeddings_tokenArray_usage(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","input":[1,2,3]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	usage := resp["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) != 3 || usage["total_tokens"].(float64) != 3 {
		t.Fatalf("usage = %v, want 3 prompt/total tokens", usage)
	}
}

func TestCompletionsStream_batch(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","stream":true,"prompt":["hi","bye"]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	out, _ := io.ReadAll(rec.Body)
	if !bytes.Contains(out, []byte("[DONE]")) {
		t.Fatalf("missing [DONE] in stream: %s", out)
	}

	var indices []float64
	for _, line := range strings.Split(string(out), "\n") {
		if !strings.HasPrefix(line, "data: ") || strings.Contains(line, "[DONE]") {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			continue
		}
		choices, ok := chunk["choices"].([]any)
		if !ok || len(choices) == 0 {
			continue
		}
		choice := choices[0].(map[string]any)
		if idx, ok := choice["index"].(float64); ok {
			indices = append(indices, idx)
		}
	}
	if !slicesContains(indices, 0) || !slicesContains(indices, 1) {
		t.Fatalf("stream indices = %v, want 0 and 1", indices)
	}
}

func slicesContains(vals []float64, target float64) bool {
	for _, v := range vals {
		if v == target {
			return true
		}
	}
	return false
}

func TestResponses_structuredInput(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	output := resp["output"].([]any)[0].(map[string]any)
	content := output["content"].([]any)[0].(map[string]any)
	if content["text"] != "hi" {
		t.Fatalf("text = %v, want hi", content["text"])
	}
}

func TestSlowDelay_respectsContextCancel(t *testing.T) {
	s := New(config.Config{
		DefaultModel: "test-model",
		SlowMarkers:  []string{"slow"},
		SlowDelay:    5 * time.Second,
		LogRequests:  false,
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	body := `{"model":"test-model","input":"trigger slow marker"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)).WithContext(ctx)
	rec := httptest.NewRecorder()

	start := time.Now()
	s.ServeHTTP(rec, req)
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Fatalf("handler blocked %v after context cancel", elapsed)
	}
}

func TestChatCompletionsStream_utf8(t *testing.T) {
	s := New(config.Config{DefaultModel: "test-model", ResponsePrefix: "日本語"})
	body := `{"model":"test-model","stream":true,"messages":[{"role":"user","content":"x"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	out, _ := io.ReadAll(rec.Body)
	if !utf8.Valid(out) {
		t.Fatal("stream body is not valid UTF-8")
	}
}

func TestResponsesInputTokens_fromPayload(t *testing.T) {
	s := newTestServer()
	short := `{"input":"hi"}`
	long := `{"input":"` + strings.Repeat("z", 200) + `"}`

	reqShort := httptest.NewRequest(http.MethodPost, "/v1/responses/input_tokens", strings.NewReader(short))
	recShort := httptest.NewRecorder()
	s.ServeHTTP(recShort, reqShort)

	reqLong := httptest.NewRequest(http.MethodPost, "/v1/responses/input_tokens", strings.NewReader(long))
	recLong := httptest.NewRecorder()
	s.ServeHTTP(recLong, reqLong)

	var respShort, respLong map[string]any
	_ = json.NewDecoder(recShort.Body).Decode(&respShort)
	_ = json.NewDecoder(recLong.Body).Decode(&respLong)

	if respLong["input_tokens"].(float64) <= respShort["input_tokens"].(float64) {
		t.Fatalf("long=%v short=%v", respLong["input_tokens"], respShort["input_tokens"])
	}
}

func TestMessagesCountTokens_includesSystem(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","system":"` + strings.Repeat("a", 400) + `","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var withSystem map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&withSystem); err != nil {
		t.Fatal(err)
	}

	bodyMinimal := `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`
	req2 := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(bodyMinimal))
	rec2 := httptest.NewRecorder()
	s.ServeHTTP(rec2, req2)

	var minimal map[string]any
	_ = json.NewDecoder(rec2.Body).Decode(&minimal)

	if withSystem["input_tokens"].(float64) <= minimal["input_tokens"].(float64) {
		t.Fatalf("with system=%v minimal=%v", withSystem["input_tokens"], minimal["input_tokens"])
	}
}

func TestMessagesAnthropic(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp["stop_reason"] != "end_turn" {
		t.Fatalf("stop_reason = %v", resp["stop_reason"])
	}
}

func TestChatCompletionsStream(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if !strings.Contains(rec.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("content-type = %q", rec.Header().Get("Content-Type"))
	}
	out, _ := io.ReadAll(rec.Body)
	if !bytes.Contains(out, []byte("[DONE]")) {
		t.Fatalf("missing [DONE] in stream: %s", out)
	}
}
