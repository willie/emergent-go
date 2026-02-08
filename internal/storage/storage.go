package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

var dataDir = "data"
var mu sync.Mutex
var cleanKeyRe = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// SaveInfo represents metadata about a save file
type SaveInfo struct {
	ID        string    `json:"id"`
	UpdatedAt time.Time `json:"updatedAt"`
}

func ensureDataDir() error {
	return os.MkdirAll(dataDir, 0755)
}

func cleanKey(key string) string {
	return cleanKeyRe.ReplaceAllString(key, "")
}

// Get retrieves a value by key, returns nil if not found
func Get(key string) (json.RawMessage, error) {
	mu.Lock()
	defer mu.Unlock()

	if err := ensureDataDir(); err != nil {
		return nil, err
	}

	clean := cleanKey(key)
	path := filepath.Join(dataDir, clean+".json")

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	return json.RawMessage(data), nil
}

// Set stores a value by key
func Set(key string, value json.RawMessage) error {
	mu.Lock()
	defer mu.Unlock()

	if err := ensureDataDir(); err != nil {
		return err
	}

	clean := cleanKey(key)
	path := filepath.Join(dataDir, clean+".json")

	// Pretty-print for readability
	var parsed interface{}
	if err := json.Unmarshal(value, &parsed); err == nil {
		if pretty, err := json.MarshalIndent(parsed, "", "  "); err == nil {
			value = pretty
		}
	}

	return os.WriteFile(path, value, 0644)
}

// Delete removes a value by key
func Delete(key string) error {
	mu.Lock()
	defer mu.Unlock()

	clean := cleanKey(key)
	path := filepath.Join(dataDir, clean+".json")

	err := os.Remove(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// List returns all save files
func List() ([]SaveInfo, error) {
	mu.Lock()
	defer mu.Unlock()

	if err := ensureDataDir(); err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(dataDir)
	if err != nil {
		return nil, err
	}

	var saves []SaveInfo
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		id := strings.TrimSuffix(entry.Name(), ".json")
		saves = append(saves, SaveInfo{
			ID:        id,
			UpdatedAt: info.ModTime(),
		})
	}

	sort.Slice(saves, func(i, j int) bool {
		return saves[i].UpdatedAt.After(saves[j].UpdatedAt)
	})

	return saves, nil
}

// SetJSON stores a Go value as JSON by key
func SetJSON(key string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	return Set(key, json.RawMessage(data))
}

// GetJSON retrieves a JSON value and unmarshals into target
func GetJSON(key string, target interface{}) error {
	data, err := Get(key)
	if err != nil {
		return err
	}
	if data == nil {
		return nil
	}
	return json.Unmarshal(data, target)
}
