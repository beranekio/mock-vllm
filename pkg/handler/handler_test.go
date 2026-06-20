package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

func TestGetModel_singleSegment(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/v1/models/test-model", nil)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var body map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["id"] != "test-model" {
		t.Fatalf("id = %v, want test-model", body["id"])
	}
}

func TestGetModel_nestedPathReturns404(t *testing.T) {
	s := newTestServer()
	for _, path := range []string{"/v1/models/foo/bar", "/v1/models/foo/"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		s.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Fatalf("%s: status = %d, want 404", path, rec.Code)
		}
	}
}

func TestGetModel_encodedSlashInID(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/v1/models/meta-llama%2FLlama-3", nil)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["id"] != "meta-llama/Llama-3" {
		t.Fatalf("id = %v, want meta-llama/Llama-3", body["id"])
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

func TestResponses_nonStream_includesUsageAndAnnotations(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","input":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp["object"] != "response" || resp["status"] != "completed" {
		t.Fatalf("envelope = %v", resp)
	}
	usage, ok := resp["usage"].(map[string]any)
	if !ok {
		t.Fatalf("missing usage: %v", resp)
	}
	for _, k := range []string{"input_tokens", "input_tokens_details", "output_tokens", "output_tokens_details", "total_tokens"} {
		if _, ok := usage[k]; !ok {
			t.Fatalf("usage missing %q: %v", k, usage)
		}
	}
	if _, ok := resp["created_at"]; !ok {
		t.Fatalf("missing created_at: %v", resp)
	}
	output := resp["output"].([]any)[0].(map[string]any)
	content := output["content"].([]any)[0].(map[string]any)
	if _, ok := content["annotations"]; !ok {
		t.Fatalf("content missing annotations: %v", content)
	}
	if _, ok := content["logprobs"]; !ok {
		t.Fatalf("content missing logprobs: %v", content)
	}
}

func TestResponses_nonStream_usageCountsFullPrompt(t *testing.T) {
	s := newTestServer()
	// Multi-turn body: short final user turn, but a long system prompt and
	// prior conversation. usage.input_tokens should count the whole prompt
	// (matching /v1/responses/input_tokens), not just the last user text.
	body := `{"model":"test-model","system":"` + strings.Repeat("a", 400) + `","input":[{"role":"user","content":"` + strings.Repeat("b", 200) + `"},{"role":"assistant","content":"` + strings.Repeat("c", 200) + `"},{"role":"user","content":"hi"}]}`

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	usage := resp["usage"].(map[string]any)
	responsesIn := usage["input_tokens"].(float64)

	// Same body to the dedicated input_tokens endpoint.
	req2 := httptest.NewRequest(http.MethodPost, "/v1/responses/input_tokens", strings.NewReader(body))
	rec2 := httptest.NewRecorder()
	s.ServeHTTP(rec2, req2)
	var countResp map[string]any
	if err := json.NewDecoder(rec2.Body).Decode(&countResp); err != nil {
		t.Fatal(err)
	}
	directIn := countResp["input_tokens"].(float64)

	if responsesIn != directIn {
		t.Fatalf("responses input_tokens=%v, input_tokens endpoint=%v (want equal — both should count the full prompt)", responsesIn, directIn)
	}

	// Sanity: a minimal body should report fewer tokens than the long one.
	reqMin := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"hi"}`))
	recMin := httptest.NewRecorder()
	s.ServeHTTP(recMin, reqMin)
	var respMin map[string]any
	_ = json.NewDecoder(recMin.Body).Decode(&respMin)
	minimalIn := respMin["usage"].(map[string]any)["input_tokens"].(float64)
	if responsesIn <= minimalIn {
		t.Fatalf("long prompt usage=%v, minimal usage=%v (want long > minimal)", responsesIn, minimalIn)
	}
}

func TestResponses_nonStream_usageCountsInstructions(t *testing.T) {
	s := newTestServer()
	// Responses API uses `instructions` for the developer prompt. The
	// usage block on /v1/responses should count it, not just the input.
	body := `{"model":"test-model","instructions":"` + strings.Repeat("i", 400) + `","input":"hi"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	var resp map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	withIns := resp["usage"].(map[string]any)["input_tokens"].(float64)

	// Same body minus the instructions field.
	req2 := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"hi"}`))
	rec2 := httptest.NewRecorder()
	s.ServeHTTP(rec2, req2)
	var resp2 map[string]any
	_ = json.NewDecoder(rec2.Body).Decode(&resp2)
	withoutIns := resp2["usage"].(map[string]any)["input_tokens"].(float64)

	if withIns <= withoutIns {
		t.Fatalf("with instructions usage=%v, without=%v (want with > without)", withIns, withoutIns)
	}
}

// streamEvents parses the SSE body into a list of event payloads (data-only).
func streamEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	var events []map[string]any
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") || strings.Contains(line, "[DONE]") {
			continue
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev); err != nil {
			t.Fatalf("decode event %q: %v", line, err)
		}
		events = append(events, ev)
	}
	return events
}

func TestResponsesStream_eventOrderAndShape(t *testing.T) {
	s := newTestServer()
	body := `{"model":"test-model","stream":true,"input":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	out, _ := io.ReadAll(rec.Body)

	wantTypes := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}
	events := streamEvents(t, string(out))
	if len(events) != len(wantTypes) {
		t.Fatalf("got %d events, want %d: %v", len(events), len(wantTypes), eventTypes(events))
	}
	for i, want := range wantTypes {
		if got := events[i]["type"]; got != want {
			t.Fatalf("event %d type = %v, want %s", i, got, want)
		}
	}

	// sequence_number must be 0..N-1 (response.created starts at 0)
	for i, ev := range events {
		if got := ev["sequence_number"].(float64); int(got) != i {
			t.Fatalf("event %d sequence_number = %v, want %d", i, got, i)
		}
	}

	// response.completed carries the full aggregated response.
	completed := events[len(events)-1]
	resp := completed["response"].(map[string]any)
	if resp["object"] != "response" || resp["status"] != "completed" {
		t.Fatalf("response.completed.response = %v", resp)
	}
	usage, ok := resp["usage"].(map[string]any)
	if !ok {
		t.Fatalf("response.completed.response missing usage: %v", resp)
	}
	for _, k := range []string{"input_tokens", "output_tokens", "total_tokens"} {
		if _, ok := usage[k]; !ok {
			t.Fatalf("response.completed.response.usage missing %q: %v", k, usage)
		}
	}

	// response.output_text.delta carries a string delta and item/output/content indices.
	delta := events[4]
	if _, ok := delta["delta"].(string); !ok {
		t.Fatalf("delta event delta is not a string: %v", delta)
	}
	for _, k := range []string{"item_id", "output_index", "content_index"} {
		if _, ok := delta[k]; !ok {
			t.Fatalf("delta event missing %q: %v", k, delta)
		}
	}

	// response.in_progress mirrors response.created (status=in_progress, no output yet).
	inProgress := events[1]
	if inProgress["type"] != "response.in_progress" {
		t.Fatalf("event 1 type = %v, want response.in_progress", inProgress["type"])
	}
	inProgressResp := inProgress["response"].(map[string]any)
	if inProgressResp["status"] != "in_progress" {
		t.Fatalf("response.in_progress.response.status = %v", inProgressResp)
	}
	if out, ok := inProgressResp["output"].([]any); !ok || len(out) != 0 {
		t.Fatalf("response.in_progress.response.output = %v", inProgressResp["output"])
	}

	// response.created carries the bare response object (no usage, no output yet).
	created := events[0]
	createdResp := created["response"].(map[string]any)
	if createdResp["status"] != "in_progress" {
		t.Fatalf("response.created.response.status = %v", createdResp)
	}
	if out, ok := createdResp["output"].([]any); !ok || len(out) != 0 {
		t.Fatalf("response.created.response.output = %v", createdResp["output"])
	}

	// output_text content parts (content_part.added and content_part.done)
	// carry a logprobs array, matching the OpenAI Responses output_text shape.
	for _, idx := range []int{3, len(events) - 3} {
		part, ok := events[idx]["part"].(map[string]any)
		if !ok {
			t.Fatalf("event %d (%v) missing part: %v", idx, events[idx]["type"], events[idx])
		}
		if _, ok := part["logprobs"]; !ok {
			t.Fatalf("event %d part missing logprobs: %v", idx, part)
		}
	}
}

func eventTypes(events []map[string]any) []string {
	out := make([]string, 0, len(events))
	for _, e := range events {
		out = append(out, fmt.Sprint(e["type"]))
	}
	return out
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
