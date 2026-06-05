package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const Version = "0.1.0"

type Config struct {
	Host           string
	Port           string
	DefaultModel   string
	SlowDelay      time.Duration
	SlowMarkers    []string
	ResponsePrefix string
	LogRequests    bool
}

func Load() Config {
	markers := strings.Split(envOr("SLOW_MARKERS", "otter,long story"), ",")
	trimmed := make([]string, 0, len(markers))
	for _, m := range markers {
		m = strings.TrimSpace(m)
		if m != "" {
			trimmed = append(trimmed, m)
		}
	}

	return Config{
		Host:           envOr("HOST", "0.0.0.0"),
		Port:           envOr("PORT", "8000"),
		DefaultModel:   envOr("DEFAULT_MODEL", "mock-model"),
		SlowDelay:      time.Duration(envFloat("SLOW_DELAY_SECONDS", 30) * float64(time.Second)),
		SlowMarkers:    trimmed,
		ResponsePrefix: envOr("RESPONSE_PREFIX", ""),
		LogRequests:    envBool("LOG_REQUESTS", true),
	}
}

func (c Config) Addr() string {
	return c.Host + ":" + c.Port
}

func (c Config) WriteTimeout() time.Duration {
	return c.SlowDelay + 30*time.Second
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return fallback
	}
	return f
}

func envBool(key string, fallback bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return fallback
	}
	return b
}

func EnvHelp() string {
	return fmt.Sprintf(`Environment variables:
  HOST                 Listen address (default: 0.0.0.0)
  PORT                 Listen port (default: 8000)
  DEFAULT_MODEL        Model id returned by /v1/models (default: mock-model)
  SLOW_DELAY_SECONDS   Delay when slow markers match (default: 30)
  SLOW_MARKERS         Comma-separated substrings triggering slow delay (default: otter,long story)
  RESPONSE_PREFIX      Prefix prepended to all generated text (default: empty)
  LOG_REQUESTS         Log each request (default: true)`)
}
