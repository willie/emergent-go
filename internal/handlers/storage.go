package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"emergent/internal/storage"
)

// StorageHandler provides a JSON API for storage (compatible with the JS frontend)
func (a *App) StorageHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		a.storageGet(w, r)
	case "POST":
		a.storagePost(w, r)
	case "DELETE":
		a.storageDelete(w, r)
	default:
		w.WriteHeader(405)
		json.NewEncoder(w).Encode(map[string]string{"error": "method not allowed"})
	}
}

func validStorageKey(key string) bool {
	for _, prefix := range []string{"surat-world-storage", "surat-chat-messages", "custom_scenarios"} {
		if key == prefix || strings.HasPrefix(key, prefix+"-") {
			return true
		}
	}
	return false
}

func (a *App) storageGet(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query().Get("key")
	list := r.URL.Query().Get("list")

	if list != "" {
		saves, err := storage.List()
		if err != nil {
			json.NewEncoder(w).Encode([]any{})
			return
		}
		json.NewEncoder(w).Encode(saves)
		return
	}

	if key == "" {
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Key is required"})
		return
	}

	if !validStorageKey(key) {
		w.WriteHeader(403)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid storage key"})
		return
	}

	data, err := storage.Get(key)
	if err != nil {
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to read data"})
		return
	}
	if data == nil {
		json.NewEncoder(w).Encode(nil)
		return
	}
	w.Write(data)
}

func (a *App) storagePost(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 5<<20)

	var body struct {
		Key   string          `json:"key"`
		Value json.RawMessage `json:"value"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid request body"})
		return
	}

	if body.Key == "" || body.Value == nil {
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Key and value are required"})
		return
	}

	if !validStorageKey(body.Key) {
		w.WriteHeader(403)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid storage key"})
		return
	}

	if err := storage.Set(body.Key, body.Value); err != nil {
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to save data"})
		return
	}

	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}

func (a *App) storageDelete(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query().Get("key")
	if key == "" {
		w.WriteHeader(400)
		json.NewEncoder(w).Encode(map[string]string{"error": "Key is required"})
		return
	}

	if !validStorageKey(key) {
		w.WriteHeader(403)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid storage key"})
		return
	}

	if err := storage.Delete(key); err != nil {
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(map[string]string{"error": "Failed to delete data"})
		return
	}

	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}
