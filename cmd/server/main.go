package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"emergent/internal/ai"
	"emergent/internal/handlers"
)

func main() {
	// Initialize AI client
	ai.Init()

	// Create app
	app, err := handlers.NewApp()
	if err != nil {
		log.Fatalf("Failed to initialize app: %v", err)
	}

	// Routes
	mux := http.NewServeMux()

	// Static files
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))

	// Pages
	mux.HandleFunc("/", app.Index)
	mux.HandleFunc("/game/new", app.NewGame)
	mux.HandleFunc("/game/new-custom", app.NewCustomGame)
	mux.HandleFunc("/game/load", app.LoadGame)
	mux.HandleFunc("/game/exit", app.ExitGame)
	mux.HandleFunc("/scenario/import", app.ImportScenario)
	mux.HandleFunc("/settings/model", app.SetModel)

	// HTMX partials
	mux.HandleFunc("/partials/saves", app.PartialSaves)

	// API - Chat (returns streamed HTML)
	mux.HandleFunc("/api/chat", app.ChatSend)
	mux.HandleFunc("/api/chat/continue", app.ChatContinue)

	// Storage API (JSON, for compatibility)
	mux.HandleFunc("/api/storage", app.StorageHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "3000"
	}

	fmt.Printf("Emergent World server starting on http://localhost:%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
