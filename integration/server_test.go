package integration

import (
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/beranekio/mock-vllm/pkg/config"
	"github.com/beranekio/mock-vllm/pkg/handler"
)

const testModel = "mock-model"

// serverRoot returns the mock server root URL (no path suffix). When
// INTEGRATION_BASE_URL is set (e.g. a Docker container in CI), tests target that
// server; otherwise an httptest server is started for the duration of the test.
func serverRoot(t *testing.T) string {
	t.Helper()
	if u := os.Getenv("INTEGRATION_BASE_URL"); u != "" {
		return strings.TrimRight(u, "/")
	}
	srv := httptest.NewServer(handler.New(config.Config{
		DefaultModel: testModel,
		LogRequests:  false,
	}))
	t.Cleanup(srv.Close)
	return strings.TrimRight(srv.URL, "/")
}

// anthropicBaseURL is the Anthropic SDK base URL (paths include v1/messages).
func anthropicBaseURL(t *testing.T) string {
	t.Helper()
	return serverRoot(t)
}
