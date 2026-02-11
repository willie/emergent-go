package models

import "fmt"

// InitialLocation for scenario setup
type InitialLocation struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// CharacterConfig for scenario setup
type CharacterConfig struct {
	Name                string  `json:"name"`
	Description         string  `json:"description"`
	IsPlayer            bool    `json:"isPlayer"`
	InitialLocationName string  `json:"initialLocationName"`
	EncounterChance     float64 `json:"encounterChance"`
	Goals               string  `json:"goals,omitempty"`
}

// ScenarioConfig for starting a new world
type ScenarioConfig struct {
	Title                 string            `json:"title"`
	Description           string            `json:"description"`
	InitialNarrativeTime  string            `json:"initialNarrativeTime"`
	Locations             []InitialLocation `json:"locations"`
	Characters            []CharacterConfig `json:"characters"`
	PlayerStartingLocation string           `json:"playerStartingLocation"`
}

// WorldState is the entire world state
type WorldState struct {
	ID                   string            `json:"id"`
	Scenario             ScenarioConfig    `json:"scenario"`
	Time                 WorldTime         `json:"time"`
	Characters           []Character       `json:"characters"`
	LocationClusters     []LocationCluster `json:"locationClusters"`
	Locations            []Location        `json:"locations"`
	Events               []WorldEvent      `json:"events"`
	Conversations        []Conversation    `json:"conversations"`
	PlayerCharacterID    string            `json:"playerCharacterId"`
	MainConversationID   string            `json:"mainConversationId"`
}

// Validate checks if the scenario config is valid
func (s *ScenarioConfig) Validate() error {
	if s.Title == "" {
		return fmt.Errorf("title is required")
	}
	if s.Description == "" {
		return fmt.Errorf("description is required")
	}
	if s.InitialNarrativeTime == "" {
		return fmt.Errorf("initial narrative time is required")
	}
	if len(s.Locations) == 0 {
		return fmt.Errorf("at least one location is required")
	}
	if len(s.Characters) == 0 {
		return fmt.Errorf("at least one character is required")
	}
	hasPlayer := false
	for _, c := range s.Characters {
		if c.IsPlayer {
			hasPlayer = true
			break
		}
	}
	if !hasPlayer {
		return fmt.Errorf("at least one character must be a player")
	}
	if s.PlayerStartingLocation == "" {
		return fmt.Errorf("player starting location is required")
	}
	return nil
}

// ChatMessage represents a message in the main chat
type ChatMessage struct {
	ID      string `json:"id"`
	Role    string `json:"role"` // "user" or "assistant"
	Content string `json:"content"`
}
