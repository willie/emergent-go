package models

// Location description with embedding
type Location struct {
	ID             string    `json:"id"`
	RawDescription string    `json:"rawDescription"`
	Embedding      []float64 `json:"embedding"`
	ClusterID      *string   `json:"clusterId"`
}

// LocationCluster is a cluster of similar locations
type LocationCluster struct {
	ID                string    `json:"id"`
	CanonicalName     string    `json:"canonicalName"`
	CentroidEmbedding []float64 `json:"centroidEmbedding"`
}

// KnowledgeEntry represents what a character knows
type KnowledgeEntry struct {
	ID                string `json:"id"`
	Content           string `json:"content"`
	AcquiredAt        int    `json:"acquiredAt"`
	Source            string `json:"source"` // "witnessed", "told", "inferred"
	SourceCharacterID string `json:"sourceCharacterId,omitempty"`
}

// Relationship between characters
type Relationship struct {
	CharacterID     string  `json:"characterId"`
	Sentiment       float64 `json:"sentiment"` // -1 to 1
	Description     string  `json:"description"`
	LastInteraction int     `json:"lastInteraction"`
}

// Character in the world
type Character struct {
	ID                       string           `json:"id"`
	Name                     string           `json:"name"`
	Description              string           `json:"description"`
	CurrentLocationClusterID string           `json:"currentLocationClusterId"`
	Knowledge                []KnowledgeEntry `json:"knowledge"`
	Relationships            []Relationship   `json:"relationships"`
	IsPlayer                 bool             `json:"isPlayer"`
	IsDiscovered             bool             `json:"isDiscovered"`
	EncounterChance          float64          `json:"encounterChance"`
	Goals                    string           `json:"goals,omitempty"`
	CreatedByMessageID       string           `json:"createdByMessageId,omitempty"`
}

// WorldTime tracks action-based time
type WorldTime struct {
	Tick          int    `json:"tick"`
	NarrativeTime string `json:"narrativeTime"`
}

// Message in a conversation
type Message struct {
	ID             string `json:"id"`
	ConversationID string `json:"conversationId"`
	Role           string `json:"role"` // "user", "assistant", "system"
	Content        string `json:"content"`
	Timestamp      int    `json:"timestamp"`
	SpeakerID      string `json:"speakerId,omitempty"`
}

// Conversation stream
type Conversation struct {
	ID                string    `json:"id"`
	Type              string    `json:"type"` // "main", "offscreen"
	LocationClusterID string    `json:"locationClusterId"`
	ParticipantIDs    []string  `json:"participantIds"`
	Messages          []Message `json:"messages"`
	IsActive          bool      `json:"isActive"`
}

// WorldEvent is an event that happened in the world
type WorldEvent struct {
	ID                   string   `json:"id"`
	Timestamp            int      `json:"timestamp"`
	LocationClusterID    string   `json:"locationClusterId"`
	InvolvedCharacterIDs []string `json:"involvedCharacterIds"`
	Description          string   `json:"description"`
	WitnessedByIDs       []string `json:"witnessedByIds"`
	IsOffscreen          bool     `json:"isOffscreen"`
	SourceMessageID      string   `json:"sourceMessageId,omitempty"`
}

// TimeCosts for different actions
var TimeCosts = map[string]int{
	"speak":   1,
	"move":    5,
	"examine": 2,
	"action":  3,
}
