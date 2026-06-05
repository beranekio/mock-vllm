package main

import (
	"log"
	"net/http"
	"time"

	"github.com/beranekio/mock-vllm/pkg/config"
	"github.com/beranekio/mock-vllm/pkg/handler"
)

func main() {
	cfg := config.Load()
	addr := cfg.Addr()

	log.Printf("mock-vllm %s listening on %s (default_model=%s)", config.Version, addr, cfg.DefaultModel)

	srv := &http.Server{
		Addr:              addr,
		Handler:           handler.New(cfg),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      cfg.WriteTimeout(),
		IdleTimeout:       60 * time.Second,
	}

	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
