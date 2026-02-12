package world

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"emergent/internal/ai"
	"emergent/internal/models"

	"github.com/google/uuid"
)

// SimulationResult from off-screen simulation
type SimulationResult struct {
	Events           []models.WorldEvent   `json:"events"`
	Conversations    []models.Conversation `json:"conversations"`
	CharacterUpdates []CharacterUpdate     `json:"characterUpdates"`
}

// CharacterUpdate is a character movement
type CharacterUpdate struct {
	CharacterID   string `json:"characterId"`
	NewLocationID string `json:"newLocationId"`
}

type simulationDepth string

const (
	depthFull    simulationDepth = "full"
	depthSummary simulationDepth = "summary"
	depthSkip    simulationDepth = "skip"
)

func determineSimulationDepth(timeSinceLastInteraction int, hasUnresolvedPlotPoints bool) simulationDepth {
	if timeSinceLastInteraction < 5 {
		return depthSkip
	}
	if hasUnresolvedPlotPoints || timeSinceLastInteraction > 20 {
		return depthFull
	}
	if timeSinceLastInteraction > 10 {
		return depthSummary
	}
	return depthSkip
}

func groupCharactersByLocation(characters []models.Character) map[string][]models.Character {
	groups := make(map[string][]models.Character)
	for _, c := range characters {
		groups[c.CurrentLocationClusterID] = append(groups[c.CurrentLocationClusterID], c)
	}
	return groups
}

func generateSummary(ctx context.Context, characters []models.Character, locationName string, timeElapsed int, world *models.WorldState, modelID string) (*models.WorldEvent, error) {
	var names []string
	for _, c := range characters {
		names = append(names, c.Name)
	}
	characterNames := strings.Join(names, " and ")

	var charDescriptions strings.Builder
	for _, c := range characters {
		fmt.Fprintf(&charDescriptions, "- %s: %s", c.Name, c.Description)
		if c.Goals != "" {
			fmt.Fprintf(&charDescriptions, "\n  Goal: %s", c.Goals)
		}
		charDescriptions.WriteString("\n")
	}

	messages := []ai.ChatMessage{
		{
			Role: "user",
			Content: fmt.Sprintf(`Summarize what likely happened between %s over %d time units at %s.

Characters:
%s
Scenario: %s

Write a brief 1-2 sentence summary of their interactions. Be specific but concise.`,
				characterNames, timeElapsed, locationName, charDescriptions.String(), world.Scenario.Description),
		},
	}

	resp, err := ai.GenerateText(ctx, modelID, messages, nil, nil)
	if err != nil {
		return nil, err
	}

	text := ""
	if len(resp.Choices) > 0 {
		text = strings.TrimSpace(resp.Choices[0].Message.Content)
	}

	var charIDs []string
	for _, c := range characters {
		charIDs = append(charIDs, c.ID)
	}

	return &models.WorldEvent{
		ID:                   uuid.New().String(),
		Timestamp:            world.Time.Tick,
		LocationClusterID:    characters[0].CurrentLocationClusterID,
		InvolvedCharacterIDs: charIDs,
		Description:          text,
		WitnessedByIDs:       charIDs,
		IsOffscreen:          true,
	}, nil
}

var dialogueLineRe = regexp.MustCompile(`^([A-Za-z][A-Za-z ]+):\s*(.+)$`)

func runFullSimulation(ctx context.Context, characters []models.Character, locationName string, timeElapsed int, world *models.WorldState, modelID string) (*SimulationResult, error) {
	var names []string
	for _, c := range characters {
		names = append(names, c.Name)
	}
	characterNames := strings.Join(names, " and ")
	turnCount := max(min(timeElapsed/2, 8), 1)

	var locNames []string
	for _, l := range world.LocationClusters {
		locNames = append(locNames, l.CanonicalName)
	}
	availableLocations := strings.Join(locNames, ", ")

	var charDescriptions strings.Builder
	for _, c := range characters {
		fmt.Fprintf(&charDescriptions, "- %s: %s", c.Name, c.Description)
		if c.Goals != "" {
			fmt.Fprintf(&charDescriptions, "\n  Goal: %s", c.Goals)
		}
		charDescriptions.WriteString("\n")
	}

	systemPrompt := fmt.Sprintf(`You are simulating a conversation between %s at %s.

Characters:
%s
Scenario: %s
Time: %s
Available Locations (for movement): %s

Write a natural dialogue between these characters. Each character should stay in character.
Format each line as: CHARACTER_NAME: "dialogue"
Include brief action descriptions in *asterisks* when appropriate.
If characters decide to go somewhere else, they should express it in dialogue.

Generate approximately %d exchanges.`,
		characterNames, locationName, charDescriptions.String(), world.Scenario.Description,
		world.Time.NarrativeTime, availableLocations, turnCount)

	messages := []ai.ChatMessage{
		{Role: "user", Content: systemPrompt},
	}

	resp, err := ai.GenerateText(ctx, modelID, messages, nil, nil)
	if err != nil {
		return &SimulationResult{}, err
	}

	text := ""
	if len(resp.Choices) > 0 {
		text = resp.Choices[0].Message.Content
	}

	// Parse dialogue into messages
	var convMessages []models.Message
	lines := strings.SplitSeq(text, "\n")
	for line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		matches := dialogueLineRe.FindStringSubmatch(line)
		if matches == nil {
			continue
		}
		speakerName := matches[1]
		content := strings.Trim(strings.TrimSpace(matches[2]), `"'`)

		var speakerID string
		for _, c := range characters {
			if strings.EqualFold(c.Name, speakerName) {
				speakerID = c.ID
				break
			}
		}

		convMessages = append(convMessages, models.Message{
			ID:             uuid.New().String(),
			ConversationID: "",
			Role:           "assistant",
			Content:        content,
			Timestamp:      world.Time.Tick,
			SpeakerID:      speakerID,
		})
	}

	// Extract events and movements using tool calling
	tools := []ai.Tool{
		{
			Type: "function",
			Function: ai.ToolFunction{
				Name:        "reportSimulation",
				Description: "Report events and movements from the conversation",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"events": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"description":   map[string]any{"type": "string"},
									"isSignificant": map[string]any{"type": "boolean"},
								},
								"required": []string{"description", "isSignificant"},
							},
						},
						"movements": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"characterName": map[string]any{"type": "string"},
									"destination":   map[string]any{"type": "string"},
								},
								"required": []string{"characterName", "destination"},
							},
						},
					},
					"required": []string{"events"},
				},
			},
		},
	}

	extractMessages := []ai.ChatMessage{
		{
			Role: "user",
			Content: fmt.Sprintf(`Analyze this conversation and extract significant events and any character movements:

%s

List any important events (agreements made, information shared, conflicts).
If any character EXPLICITLY decides to leave for another location, report it in movements. Matches must be from: %s`,
				text, availableLocations),
		},
	}

	extractResp, err := ai.GenerateText(ctx, modelID, extractMessages, tools, "required")

	var extractedEvents []struct {
		Description   string `json:"description"`
		IsSignificant bool   `json:"isSignificant"`
	}
	var extractedMovements []struct {
		CharacterName string `json:"characterName"`
		Destination   string `json:"destination"`
	}

	if err == nil && len(extractResp.Choices) > 0 && len(extractResp.Choices[0].Message.ToolCalls) > 0 {
		tc := extractResp.Choices[0].Message.ToolCalls[0]
		if tc.Function.Name == "reportSimulation" {
			var args struct {
				Events []struct {
					Description   string `json:"description"`
					IsSignificant bool   `json:"isSignificant"`
				} `json:"events"`
				Movements []struct {
					CharacterName string `json:"characterName"`
					Destination   string `json:"destination"`
				} `json:"movements"`
			}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil {
				extractedEvents = args.Events
				extractedMovements = args.Movements
			}
		}
	}

	var charIDs []string
	for _, c := range characters {
		charIDs = append(charIDs, c.ID)
	}

	var events []models.WorldEvent
	for _, e := range extractedEvents {
		if !e.IsSignificant {
			continue
		}
		events = append(events, models.WorldEvent{
			ID:                   uuid.New().String(),
			Timestamp:            world.Time.Tick,
			LocationClusterID:    characters[0].CurrentLocationClusterID,
			InvolvedCharacterIDs: charIDs,
			Description:          e.Description,
			WitnessedByIDs:       charIDs,
			IsOffscreen:          true,
		})
	}

	convID := uuid.New().String()
	conversation := models.Conversation{
		ID:                convID,
		Type:              "offscreen",
		LocationClusterID: characters[0].CurrentLocationClusterID,
		ParticipantIDs:    charIDs,
		Messages:          convMessages,
		IsActive:          true,
	}

	// Resolve movements
	var movements []CharacterUpdate
	for _, move := range extractedMovements {
		var charID string
		for _, c := range characters {
			if strings.EqualFold(c.Name, move.CharacterName) {
				charID = c.ID
				break
			}
		}
		if charID == "" {
			continue
		}

		var locID string
		// Prefer exact match
		for _, l := range world.LocationClusters {
			if strings.EqualFold(l.CanonicalName, move.Destination) {
				locID = l.ID
				break
			}
		}
		// Fall back to substring match
		if locID == "" {
			destLower := strings.ToLower(move.Destination)
			for _, l := range world.LocationClusters {
				nameLower := strings.ToLower(l.CanonicalName)
				if strings.Contains(nameLower, destLower) || strings.Contains(destLower, nameLower) {
					locID = l.ID
					break
				}
			}
		}
		if locID == "" {
			continue
		}

		// Don't move if already there
		for _, c := range characters {
			if c.ID == charID && c.CurrentLocationClusterID == locID {
				locID = ""
				break
			}
		}
		if locID != "" {
			movements = append(movements, CharacterUpdate{
				CharacterID:   charID,
				NewLocationID: locID,
			})
		}
	}

	return &SimulationResult{
		Events:           events,
		Conversations:    []models.Conversation{conversation},
		CharacterUpdates: movements,
	}, nil
}

// SimulateOffscreen runs off-screen simulation for non-player characters
func SimulateOffscreen(ctx context.Context, world *models.WorldState, playerLocationClusterID string, timeSinceLastSimulation int, modelID string) (*SimulationResult, error) {
	// Get absent, discovered, non-player characters
	var absentCharacters []models.Character
	for _, c := range world.Characters {
		if !c.IsPlayer && c.IsDiscovered && c.CurrentLocationClusterID != playerLocationClusterID {
			absentCharacters = append(absentCharacters, c)
		}
	}

	if len(absentCharacters) < 2 {
		return &SimulationResult{}, nil
	}

	byLocation := groupCharactersByLocation(absentCharacters)

	result := &SimulationResult{}

	for locationID, chars := range byLocation {
		if len(chars) < 2 {
			continue
		}

		locationName := "an unknown location"
		for _, l := range world.LocationClusters {
			if l.ID == locationID {
				locationName = l.CanonicalName
				break
			}
		}

		depth := determineSimulationDepth(timeSinceLastSimulation, false)
		if depth == depthSkip {
			continue
		}

		if depth == depthFull {
			simResult, err := runFullSimulation(ctx, chars, locationName, timeSinceLastSimulation, world, modelID)
			if err != nil {
				continue
			}
			result.Events = append(result.Events, simResult.Events...)
			result.Conversations = append(result.Conversations, simResult.Conversations...)
			result.CharacterUpdates = append(result.CharacterUpdates, simResult.CharacterUpdates...)
		} else if depth == depthSummary {
			event, err := generateSummary(ctx, chars, locationName, timeSinceLastSimulation, world, modelID)
			if err != nil {
				continue
			}
			result.Events = append(result.Events, *event)
		}
	}

	return result, nil
}
