package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"emergent/internal/ai"
	"emergent/internal/handlers"
)

func main() {
	// Structured JSON logging
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	// Initialize AI client
	ai.Init()

	// Create app
	app, err := handlers.NewApp(TemplateFS)
	if err != nil {
		slog.Error("failed to initialize app", "error", err)
		os.Exit(1)
	}

	// Routes
	mux := http.NewServeMux()

	// Pages
	mux.HandleFunc("GET /{$}", app.WithSessionLock(app.Index))
	mux.HandleFunc("POST /game/new", app.WithSessionLock(app.NewGame))
	mux.HandleFunc("POST /game/new-custom", app.WithSessionLock(app.NewCustomGame))
	mux.HandleFunc("POST /game/load", app.WithSessionLock(app.LoadGame))
	mux.HandleFunc("POST /game/exit", app.WithSessionLock(app.ExitGame))
	mux.HandleFunc("POST /scenario/import", app.ImportScenario)
	mux.HandleFunc("POST /settings/model", app.WithSessionLock(app.SetModel))

	// HTMX partials
	mux.HandleFunc("GET /partials/saves", app.WithSessionLock(app.PartialSaves))

	// API - Chat (returns streamed HTML)
	mux.HandleFunc("POST /api/chat", app.WithSessionLock(app.ChatSend))
	mux.HandleFunc("POST /api/chat/continue", app.WithSessionLock(app.ChatContinue))
	mux.HandleFunc("POST /api/chat/edit", app.WithSessionLock(app.EditMessage))
	mux.HandleFunc("POST /api/chat/rewind", app.WithSessionLock(app.RewindChat))
	mux.HandleFunc("POST /api/chat/regenerate", app.WithSessionLock(app.RegenerateChat))

	// Storage API (JSON, for compatibility)
	mux.HandleFunc("GET /api/storage", app.StorageGet)
	mux.HandleFunc("POST /api/storage", app.StoragePost)
	mux.HandleFunc("DELETE /api/storage", app.StorageDelete)

	port := os.Getenv("PORT")
	if port == "" {
		port = "3001"
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
