package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"emergent"
	"emergent/internal/ai"
	"emergent/internal/handlers"
)

func main() {
	// Structured JSON logging
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	// Initialize AI client
	ai.Init()

	// Create app
	app, err := handlers.NewApp(emergent.TemplateFS)
	if err != nil {
		slog.Error("failed to initialize app", "error", err)
		os.Exit(1)
	}

	// Routes
	mux := http.NewServeMux()

	// Pages
	mux.HandleFunc("GET /{$}", app.Index)
	mux.HandleFunc("POST /game/new", app.NewGame)
	mux.HandleFunc("POST /game/new-custom", app.NewCustomGame)
	mux.HandleFunc("POST /game/load", app.LoadGame)
	mux.HandleFunc("POST /game/exit", app.ExitGame)
	mux.HandleFunc("POST /scenario/import", app.ImportScenario)
	mux.HandleFunc("POST /settings/model", app.SetModel)

	// HTMX partials
	mux.HandleFunc("GET /partials/saves", app.PartialSaves)

	// API - Chat (returns streamed HTML)
	mux.HandleFunc("POST /api/chat", app.ChatSend)
	mux.HandleFunc("POST /api/chat/continue", app.ChatContinue)
	mux.HandleFunc("POST /api/chat/edit", app.EditMessage)
	mux.HandleFunc("POST /api/chat/rewind", app.RewindChat)
	mux.HandleFunc("POST /api/chat/regenerate", app.RegenerateChat)

	// Storage API (JSON, for compatibility)
	mux.HandleFunc("/api/storage", app.StorageHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "3000"
	}

	server := &http.Server{
		Addr:    ":" + port,
		Handler: handlers.LogRequest(mux),
	}

	// Graceful shutdown on SIGINT/SIGTERM
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	app.StartEviction(ctx)

	go func() {
		slog.Info("server starting", "addr", "http://localhost:"+port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	slog.Info("shutting down gracefully")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		slog.Error("shutdown error", "error", err)
		os.Exit(1)
	}

	slog.Info("server stopped")
}
