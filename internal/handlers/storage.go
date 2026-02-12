package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"emergent/internal/storage"
)

func validStorageKey(key string) bool {
	for _, prefix := range []string{"surat-world-storage", "surat-chat-messages", "custom_scenarios"} {
		if key == prefix || strings.HasPrefix(key, prefix+"-") {
			return true
		}
	}
	return false
}

// StorageGet handles GET /api/storage
func (a *App) StorageGet(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
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

// StoragePost handles POST /api/storage
func (a *App) StoragePost(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
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

// StorageDelete handles DELETE /api/storage
func (a *App) StorageDelete(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
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
