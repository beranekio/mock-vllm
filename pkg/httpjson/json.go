package httpjson

import (
	"encoding/json"
	"fmt"
	"net/http"
)

func Write(w http.ResponseWriter, status int, body any) {
	encoded, err := json.Marshal(body)
	if err != nil {
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(encoded)))
	w.WriteHeader(status)
	_, _ = w.Write(encoded)
}

func Error(w http.ResponseWriter, status int, message string) {
	Write(w, status, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "invalid_request_error",
		},
	})
}

func AnthropicError(w http.ResponseWriter, status int, errType, message string) {
	Write(w, status, map[string]any{
		"type": "error",
		"error": map[string]string{
			"type":    errType,
			"message": message,
		},
	})
}
