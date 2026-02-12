package world

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"emergent/internal/models"
	"emergent/internal/storage"

	"github.com/google/uuid"
)

// SessionState manages the world state for a session
type SessionState struct {
	mu                  sync.RWMutex
	World               *models.WorldState
	ChatMessages        []models.ChatMessage
	ActiveSaveKey       string
	LastSimulationTick  int
	ModelID             string
	IsSimulating        bool
	LastAccessed        time.Time
}

// NewSessionState creates a new session state
func NewSessionState() *SessionState {
	return &SessionState{
		ActiveSaveKey: "surat-world-storage",
		LastAccessed:  time.Now(),
	}
}

// Touch updates the last access time
func (s *SessionState) Touch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LastAccessed = time.Now()
}

// GetLastAccessed returns the last access time
func (s *SessionState) GetLastAccessed() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.LastAccessed
}

// InitializeScenario sets up a new world from a scenario config
func (s *SessionState) InitializeScenario(config models.ScenarioConfig) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := config.Validate(); err != nil {
		return err
	}

	worldID := uuid.New().String()
	mainConversationID := uuid.New().String()

	var locationClusters []models.LocationCluster
	for _, loc := range config.Locations {
		locationClusters = append(locationClusters, models.LocationCluster{
			ID:                uuid.New().String(),
			CanonicalName:     loc.Name,
			CentroidEmbedding: []float64{},
		})
	}

	getLocationID := func(name string) string {
		for _, c := range locationClusters {
			if c.CanonicalName == name {
				return c.ID
			}
		}
		if len(locationClusters) > 0 {
			return locationClusters[0].ID
		}
		return ""
	}

	playerStartingLocationID := getLocationID(config.PlayerStartingLocation)
	if playerStartingLocationID == "" {
		return fmt.Errorf("player starting location %q not found", config.PlayerStartingLocation)
	}

	var playerCharacterID string
	var characters []models.Character
	for _, cc := range config.Characters {
		id := uuid.New().String()
		if cc.IsPlayer {
			playerCharacterID = id
		}
		locationID := getLocationID(cc.InitialLocationName)
		isAtPlayerLocation := locationID == playerStartingLocationID

		characters = append(characters, models.Character{
			ID:                       id,
			Name:                     cc.Name,
			Description:              cc.Description,
			IsPlayer:                 cc.IsPlayer,
			EncounterChance:          cc.EncounterChance,
			CurrentLocationClusterID: locationID,
			Knowledge:                []models.KnowledgeEntry{},
			Relationships:            []models.Relationship{},
			IsDiscovered:             cc.IsPlayer || isAtPlayerLocation,
			Goals:                    cc.Goals,
		})
	}

	var participantIDs []string
	for _, c := range characters {
		if c.IsDiscovered && c.CurrentLocationClusterID == playerStartingLocationID {
			participantIDs = append(participantIDs, c.ID)
		}
	}

	mainConversation := models.Conversation{
		ID:                mainConversationID,
		Type:              "main",
		LocationClusterID: playerStartingLocationID,
		ParticipantIDs:    participantIDs,
		Messages:          []models.Message{},
		IsActive:          true,
	}

	s.World = &models.WorldState{
		ID:       worldID,
		Scenario: config,
		Time: models.WorldTime{
			Tick:          0,
			NarrativeTime: config.InitialNarrativeTime,
		},
		Characters:         characters,
		LocationClusters:   locationClusters,
		Locations:          []models.Location{},
		Events:             []models.WorldEvent{},
		Conversations:      []models.Conversation{mainConversation},
		PlayerCharacterID:  playerCharacterID,
		MainConversationID: mainConversationID,
	}

	s.ChatMessages = nil
	s.LastSimulationTick = 0

	return s.persist()
}

// ResetWorld clears the world state
func (s *SessionState) ResetWorld() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.World = nil
	s.ChatMessages = nil
	s.IsSimulating = false
}

// AdvanceTime increments the world clock
func (s *SessionState) AdvanceTime(ticks int, narrativeTime string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.World == nil {
		return
	}
	s.World.Time.Tick += ticks
	if narrativeTime != "" {
		s.World.Time.NarrativeTime = narrativeTime
	}
}

// MoveCharacter changes a character's location
func (s *SessionState) MoveCharacter(characterID, locationClusterID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.World == nil {
		return
	}
	for i, c := range s.World.Characters {
		if c.ID == characterID {
			s.World.Characters[i].CurrentLocationClusterID = locationClusterID
			break
		}
	}
}

// DiscoverCharacter marks a character as discovered
func (s *SessionState) DiscoverCharacter(characterID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.World == nil {
		return
	}
	for i, c := range s.World.Characters {
		if c.ID == characterID {
			s.World.Characters[i].IsDiscovered = true
			break
		}
	}
}

// AddCharacter adds a new character to the world
func (s *SessionState) AddCharacter(character models.Character) models.Character {
	s.mu.Lock()
	defer s.mu.Unlock()
	if character.ID == "" {
		character.ID = uuid.New().String()
	}
	if character.Knowledge == nil {
		character.Knowledge = []models.KnowledgeEntry{}
	}
	if character.Relationships == nil {
		character.Relationships = []models.Relationship{}
	}
	if s.World != nil {
		s.World.Characters = append(s.World.Characters, character)
	}
	return character
}

// AddLocationCluster adds a new location cluster
func (s *SessionState) AddLocationCluster(canonicalName string) models.LocationCluster {
	s.mu.Lock()
	defer s.mu.Unlock()
	cluster := models.LocationCluster{
		ID:                uuid.New().String(),
		CanonicalName:     canonicalName,
		CentroidEmbedding: []float64{},
	}
	if s.World != nil {
		s.World.LocationClusters = append(s.World.LocationClusters, cluster)
	}
	return cluster
}

// AddEvent adds a world event
func (s *SessionState) AddEvent(event models.WorldEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if event.ID == "" {
		event.ID = uuid.New().String()
	}
	if s.World != nil {
		s.World.Events = append(s.World.Events, event)
	}
}

// AddConversation adds a conversation
func (s *SessionState) AddConversation(conversation models.Conversation) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if conversation.ID == "" {
		conversation.ID = uuid.New().String()
	}
	if s.World != nil {
		s.World.Conversations = append(s.World.Conversations, conversation)
	}
}

// UpdateCharacterKnowledge adds a knowledge entry to a character
func (s *SessionState) UpdateCharacterKnowledge(characterID string, content string, acquiredAt int, source string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.World == nil {
		return
	}
	entry := models.KnowledgeEntry{
		ID:         uuid.New().String(),
		Content:    content,
		AcquiredAt: acquiredAt,
		Source:     source,
	}
	for i, c := range s.World.Characters {
		if c.ID == characterID {
			s.World.Characters[i].Knowledge = append(s.World.Characters[i].Knowledge, entry)
			break
		}
	}
}

// GetPlayerCharacter returns the player character and true, or a zero value and false.
func (s *SessionState) GetPlayerCharacter() (models.Character, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return models.Character{}, false
	}
	for _, c := range s.World.Characters {
		if c.ID == s.World.PlayerCharacterID {
			return c, true
		}
	}
	return models.Character{}, false
}

// GetPlayerLocation returns the player's current location cluster and true, or a zero value and false.
func (s *SessionState) GetPlayerLocation() (models.LocationCluster, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return models.LocationCluster{}, false
	}
	var playerLocID string
	for _, c := range s.World.Characters {
		if c.ID == s.World.PlayerCharacterID {
			playerLocID = c.CurrentLocationClusterID
			break
		}
	}
	if playerLocID == "" {
		return models.LocationCluster{}, false
	}
	for _, l := range s.World.LocationClusters {
		if l.ID == playerLocID {
			return l, true
		}
	}
	return models.LocationCluster{}, false
}

// GetCharactersAtPlayerLocation returns discovered non-player characters at the player's location
func (s *SessionState) GetCharactersAtPlayerLocation() []models.Character {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return nil
	}
	var playerLocID string
	for _, c := range s.World.Characters {
		if c.ID == s.World.PlayerCharacterID {
			playerLocID = c.CurrentLocationClusterID
			break
		}
	}
	if playerLocID == "" {
		return nil
	}
	var result []models.Character
	for _, c := range s.World.Characters {
		if !c.IsPlayer && c.IsDiscovered && c.CurrentLocationClusterID == playerLocID {
			result = append(result, c)
		}
	}
	return result
}

// GetDiscoveredCharacters returns all discovered non-player characters
func (s *SessionState) GetDiscoveredCharacters() []models.Character {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return nil
	}
	var result []models.Character
	for _, c := range s.World.Characters {
		if c.IsDiscovered && !c.IsPlayer {
			result = append(result, c)
		}
	}
	return result
}

// GetOffscreenConversations returns active offscreen conversations
func (s *SessionState) GetOffscreenConversations() []models.Conversation {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return nil
	}
	var result []models.Conversation
	for _, c := range s.World.Conversations {
		if c.Type == "offscreen" && c.IsActive {
			result = append(result, c)
		}
	}
	return result
}

// GetCharacterByID returns a character by ID and true, or a zero value and false.
func (s *SessionState) GetCharacterByID(id string) (models.Character, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return models.Character{}, false
	}
	for _, c := range s.World.Characters {
		if c.ID == id {
			return c, true
		}
	}
	return models.Character{}, false
}

// GetLocationCluster returns a location cluster by ID and true, or a zero value and false.
func (s *SessionState) GetLocationCluster(id string) (models.LocationCluster, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return models.LocationCluster{}, false
	}
	for _, l := range s.World.LocationClusters {
		if l.ID == id {
			return l, true
		}
	}
	return models.LocationCluster{}, false
}

// FindBestCharacterMatch finds a character by name with fuzzy matching.
// Returns the character and true, or a zero value and false.
func (s *SessionState) FindBestCharacterMatch(name string) (models.Character, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return models.Character{}, false
	}
	normalizedSearch := normalizeName(name)

	// Exact match
	for _, c := range s.World.Characters {
		if strings.EqualFold(c.Name, name) {
			return c, true
		}
	}
	// Normalized exact
	for _, c := range s.World.Characters {
		if normalizeName(c.Name) == normalizedSearch {
			return c, true
		}
	}
	// Substring match
	for _, c := range s.World.Characters {
		normChar := normalizeName(c.Name)
		if strings.Contains(normChar, normalizedSearch) || strings.Contains(normalizedSearch, normChar) {
			return c, true
		}
	}
	return models.Character{}, false
}

func normalizeName(name string) string {
	name = strings.ToLower(name)
	// Remove non-word/space characters
	result := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == ' ' {
			return r
		}
		return -1
	}, name)
	return strings.TrimSpace(result)
}

// Persist saves the current state
func (s *SessionState) Persist() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.persist()
}

func (s *SessionState) persist() error {
	if s.World != nil {
		if err := storage.SetJSON(s.ActiveSaveKey, s.World); err != nil {
			return fmt.Errorf("save world: %w", err)
		}
	}
	if len(s.ChatMessages) > 0 {
		msgKey := s.getChatKey()
		if err := storage.SetJSON(msgKey, s.ChatMessages); err != nil {
			return fmt.Errorf("save messages: %w", err)
		}
	}
	return nil
}

func (s *SessionState) getChatKey() string {
	base := "surat-chat-messages"
	if s.ActiveSaveKey == "surat-world-storage" {
		return base
	}
	suffix := strings.TrimPrefix(s.ActiveSaveKey, "surat-world-storage")
	return base + suffix
}

// Load loads session state from storage
func (s *SessionState) Load(saveKey string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.ActiveSaveKey = saveKey

	var world models.WorldState
	data, err := storage.Get(saveKey)
	if err != nil {
		return err
	}
	if data != nil {
		if err := json.Unmarshal(data, &world); err != nil {
			return fmt.Errorf("unmarshal world: %w", err)
		}
		s.World = &world
	}

	msgKey := s.getChatKey()
	var msgs []models.ChatMessage
	if err := storage.GetJSON(msgKey, &msgs); err == nil && msgs != nil {
		s.ChatMessages = msgs
	}

	return nil
}

// AddChatMessage adds a message to chat history
func (s *SessionState) AddChatMessage(msg models.ChatMessage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if msg.ID == "" {
		msg.ID = uuid.New().String()
	}
	s.ChatMessages = append(s.ChatMessages, msg)
}

// GetWorld returns the world state pointer. Callers must hold the session
// mutex (via WithSessionLock middleware) to prevent concurrent access.
func (s *SessionState) GetWorld() *models.WorldState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.World
}

// GetModelID returns the current model ID under a read lock.
func (s *SessionState) GetModelID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ModelID
}

// SetModelID sets the model ID under a write lock.
func (s *SessionState) SetModelID(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ModelID = id
}

// GetIsSimulating returns the simulation flag under a read lock.
func (s *SessionState) GetIsSimulating() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.IsSimulating
}

// SetIsSimulating sets the simulation flag under a write lock.
func (s *SessionState) SetIsSimulating(v bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.IsSimulating = v
}

// GetLastSimulationTick returns the last simulation tick under a read lock.
func (s *SessionState) GetLastSimulationTick() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.LastSimulationTick
}

// SetLastSimulationTick sets the last simulation tick under a write lock.
func (s *SessionState) SetLastSimulationTick(tick int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LastSimulationTick = tick
}

// GetCurrentTick returns the current tick under a read lock.
func (s *SessionState) GetCurrentTick() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.World == nil {
		return 0
	}
	return s.World.Time.Tick
}

// GetActiveSaveKey returns the active save key under a read lock.
func (s *SessionState) GetActiveSaveKey() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ActiveSaveKey
}

// SetActiveSaveKey sets the active save key under a write lock.
func (s *SessionState) SetActiveSaveKey(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ActiveSaveKey = key
}

// GetChatMessagesCopy returns a defensive copy of the chat messages slice.
func (s *SessionState) GetChatMessagesCopy() []models.ChatMessage {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cp := make([]models.ChatMessage, len(s.ChatMessages))
	copy(cp, s.ChatMessages)
	return cp
}

// EditChatMessage updates the content of a chat message by ID.
func (s *SessionState) EditChatMessage(id, content string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, m := range s.ChatMessages {
		if m.ID == id {
			s.ChatMessages[i].Content = content
			return true
		}
	}
	return false
}

// TruncateChatMessages removes all messages at or after the given index.
func (s *SessionState) TruncateChatMessages(index int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= 0 && index < len(s.ChatMessages) {
		s.ChatMessages = s.ChatMessages[:index]
	}
}

// PopLastAssistantMessage removes and returns the last assistant message, if any.
func (s *SessionState) PopLastAssistantMessage() (models.ChatMessage, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.ChatMessages) - 1; i >= 0; i-- {
		if s.ChatMessages[i].Role == "assistant" {
			msg := s.ChatMessages[i]
			s.ChatMessages = append(s.ChatMessages[:i], s.ChatMessages[i+1:]...)
			return msg, true
		}
	}
	return models.ChatMessage{}, false
}

// ChatMessageCount returns the number of chat messages.
func (s *SessionState) ChatMessageCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.ChatMessages)
}

// UpdateCharacter updates character fields
func (s *SessionState) UpdateCharacter(characterID string, name, description string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.World == nil {
		return
	}
	for i, c := range s.World.Characters {
		if c.ID == characterID {
			if name != "" {
				s.World.Characters[i].Name = name
			}
			if description != "" {
				s.World.Characters[i].Description = description
			}
			break
		}
	}
}
