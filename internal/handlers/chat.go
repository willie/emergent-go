package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html"
	"log/slog"
	"net/http"
	"strconv"
	"strings"

	"emergent/internal/ai"
	"emergent/internal/models"
	"emergent/internal/world"

	"github.com/google/uuid"
)

type toolCallResult struct {
	ToolCall ai.ToolCall
	Result   string
}

// writeSSE writes a single SSE event to the response
func writeSSE(w http.ResponseWriter, flusher http.Flusher, event, data string) {
	if event != "" {
		fmt.Fprintf(w, "event: %s\n", event)
	}
	// SSE data lines: split on newlines so each gets "data: " prefix
	for line := range strings.SplitSeq(data, "\n") {
		fmt.Fprintf(w, "data: %s\n", line)
	}
	fmt.Fprint(w, "\n")
	flusher.Flush()
}

// ChatSend handles a new chat message, returning an SSE stream
func (a *App) ChatSend(w http.ResponseWriter, r *http.Request) {
	if sid := a.getSessionID(r); sid != "" {
		mu := a.getSessionMutex(sid)
		mu.Lock()
		defer mu.Unlock()
	}

	if err := r.ParseForm(); err != nil {
		http.Error(w, "Bad request", 400)
		return
	}
	message := strings.TrimSpace(r.FormValue("message"))
	if message == "" {
		http.Error(w, "Empty message", 400)
		return
	}

	session := a.getSession(w, r)
	if session.GetWorld() == nil {
		http.Error(w, "No active game", 400)
		return
	}

	// Advance time by 1 tick for user action
	session.AdvanceTime(1, "")

	// Add user message
	userMsg := models.ChatMessage{
		ID:      uuid.New().String(),
		Role:    "user",
		Content: message,
	}
	session.AddChatMessage(userMsg)

	a.streamResponse(w, r, session, &userMsg)
}

// ChatContinue generates another assistant response
func (a *App) ChatContinue(w http.ResponseWriter, r *http.Request) {
	if sid := a.getSessionID(r); sid != "" {
		mu := a.getSessionMutex(sid)
		mu.Lock()
		defer mu.Unlock()
	}

	session := a.getSession(w, r)
	if session.GetWorld() == nil {
		http.Error(w, "No active game", 400)
		return
	}

	session.AdvanceTime(1, "")

	a.streamResponse(w, r, session, nil)
}

// streamResponse runs the LLM and streams SSE events
func (a *App) streamResponse(w http.ResponseWriter, r *http.Request, session *world.SessionState, userMsg *models.ChatMessage) {
	ctx := r.Context()

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", 500)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Send user message bubble if present
	if userMsg != nil {
		userHTML := fmt.Sprintf(
			`<div class="flex justify-end"><div class="max-w-[80%%] rounded-lg px-4 py-2 bg-blue-600 text-white"><div class="prose prose-invert max-w-none break-words whitespace-pre-wrap">%s</div></div></div>`,
			html.EscapeString(userMsg.Content))
		writeSSE(w, flusher, "user-message", userHTML)
	}

	// Build AI messages
	chatMessages := session.GetChatMessagesCopy()
	aiMessages := a.buildAIMessages(chatMessages)
	ws := session.GetWorld()
	systemPrompt := buildSystemPrompt(ws)
	tools := buildChatTools(ws)

	fullMessages := append([]ai.ChatMessage{
		{Role: "system", Content: systemPrompt},
	}, aiMessages...)

	model := session.GetModelID()
	if model == "" {
		model = ai.Models.MainConversation
	}

	const maxToolSteps = 5
	var finalContent strings.Builder

	for step := range maxToolSteps {
		// Stream tokens
		result, err := ai.StreamText(ctx, model, fullMessages, tools, func(content string) {
			writeSSE(w, flusher, "token", content)
		})

		if err != nil {
			slog.Error("AI streaming failed", "step", step, "error", err)
			writeSSE(w, flusher, "error", "Something went wrong. Please try again.")
			writeSSE(w, flusher, "done", "end")
			return
		}

		finalContent.WriteString(result.Content)

		if len(result.ToolCalls) == 0 {
			break
		}

		// Append assistant message with tool calls to conversation
		var content any = result.Content
		if result.Content == "" {
			content = nil // API expects null content for tool-call-only messages
		}
		fullMessages = append(fullMessages, ai.ChatMessage{
			Role:      "assistant",
			Content:   content,
			ToolCalls: result.ToolCalls,
		})

		// Process tool calls and get results
		notify := func(event, data string) { writeSSE(w, flusher, event, data) }
		results := a.processToolCalls(ctx, session, result.ToolCalls, "", notify)

		// Append tool result messages
		for _, tr := range results {
			fullMessages = append(fullMessages, ai.ChatMessage{
				Role:       "tool",
				Content:    tr.Result,
				ToolCallID: tr.ToolCall.ID,
				Name:       tr.ToolCall.Function.Name,
			})
		}

		// Send OOB updates for state changes from this step
		oobHTML := a.renderOOBUpdates(session)
		if oobHTML != "" {
			writeSSE(w, flusher, "oob", oobHTML)
		}

		// Rebuild system prompt since world state changed
		ws = session.GetWorld()
		systemPrompt = buildSystemPrompt(ws)
		tools = buildChatTools(ws)
		fullMessages[0] = ai.ChatMessage{Role: "system", Content: systemPrompt}
	}

	// Save assistant message
	assistantMsg := models.ChatMessage{
		ID:      uuid.New().String(),
		Role:    "assistant",
		Content: finalContent.String(),
	}
	session.AddChatMessage(assistantMsg)

	// Persist state
	if err := session.Persist(); err != nil {
		slog.Error("failed to persist state", "error", err)
	}

	writeSSE(w, flusher, "done", "")
}

// EditMessage updates a chat message's content
func (a *App) EditMessage(w http.ResponseWriter, r *http.Request) {
	if sid := a.getSessionID(r); sid != "" {
		mu := a.getSessionMutex(sid)
		mu.Lock()
		defer mu.Unlock()
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Bad request", 400)
		return
	}
	msgID := r.FormValue("message_id")
	content := r.FormValue("content")
	if msgID == "" || content == "" {
		http.Error(w, "Missing message_id or content", 400)
		return
	}

	session := a.getSession(w, r)
	if !session.EditChatMessage(msgID, content) {
		http.Error(w, "Message not found", 404)
		return
	}
	if err := session.Persist(); err != nil {
		slog.Error("failed to persist after edit", "error", err)
	}
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// RewindChat truncates chat history to before a given index
func (a *App) RewindChat(w http.ResponseWriter, r *http.Request) {
	if sid := a.getSessionID(r); sid != "" {
		mu := a.getSessionMutex(sid)
		mu.Lock()
		defer mu.Unlock()
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Bad request", 400)
		return
	}
	idxStr := r.FormValue("index")
	idx, err := strconv.Atoi(idxStr)
	if err != nil || idx < 0 {
		http.Error(w, "Invalid index", 400)
		return
	}

	session := a.getSession(w, r)
	session.TruncateChatMessages(idx)
	if err := session.Persist(); err != nil {
		slog.Error("failed to persist after rewind", "error", err)
	}
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// RegenerateChat removes the last assistant message and streams a new response
func (a *App) RegenerateChat(w http.ResponseWriter, r *http.Request) {
	if sid := a.getSessionID(r); sid != "" {
		mu := a.getSessionMutex(sid)
		mu.Lock()
		defer mu.Unlock()
	}

	session := a.getSession(w, r)
	if session.GetWorld() == nil {
		http.Error(w, "No active game", 400)
		return
	}

	if _, ok := session.PopLastAssistantMessage(); !ok {
		http.Error(w, "No assistant message to regenerate", 400)
		return
	}

	a.streamResponse(w, r, session, nil)
}

// renderOOBUpdates renders out-of-band swap HTML for header, sidebar, etc.
func (a *App) renderOOBUpdates(session *world.SessionState) string {
	data := a.buildGameData(session)
	var buf strings.Builder

	// Location name
	locName := "Unknown"
	if data.Location != nil {
		locName = data.Location.CanonicalName
	}
	fmt.Fprintf(&buf, `<span id="location-name" hx-swap-oob="innerHTML">%s</span>`, html.EscapeString(locName))

	// Narrative time
	fmt.Fprintf(&buf, `<span id="narrative-time" hx-swap-oob="innerHTML">%s</span>`, html.EscapeString(data.World.Time.NarrativeTime))

	// Tick count
	fmt.Fprintf(&buf, `<div id="tick-count" hx-swap-oob="innerHTML">tick %d</div>`, data.World.Time.Tick)

	// Present characters
	var names []string
	for _, c := range data.NearbyCharacters {
		names = append(names, html.EscapeString(c.Name))
	}
	fmt.Fprintf(&buf, `<span id="present-chars" hx-swap-oob="innerHTML">%s</span>`, strings.Join(names, ", "))

	// Offscreen conversations — use cached template
	fmt.Fprintf(&buf, `<div id="offscreen-content" hx-swap-oob="innerHTML">%s</div>`, singleLineHTML(a.renderPartial("offscreen_list", data)))

	// Characters list — use cached template
	fmt.Fprintf(&buf, `<div id="characters-content" hx-swap-oob="innerHTML">%s</div>`, singleLineHTML(a.renderPartial("character_list", data)))

	return buf.String()
}

// singleLineHTML collapses whitespace so multi-line HTML fits in a single SSE data: line
func singleLineHTML(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// renderPartial executes a named template block into a string
func (a *App) renderPartial(name string, data any) string {
	var buf bytes.Buffer
	if err := a.PageTemplates["game.html"].ExecuteTemplate(&buf, name, data); err != nil {
		slog.Error("render partial failed", "partial", name, "error", err)
		return ""
	}
	return buf.String()
}

func (a *App) buildAIMessages(chatMessages []models.ChatMessage) []ai.ChatMessage {
	var messages []ai.ChatMessage
	for _, msg := range chatMessages {
		messages = append(messages, ai.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	return messages
}

func buildSystemPrompt(ws *models.WorldState) string {
	player, hasPlayer := findPlayer(ws)
	var playerLocation *models.LocationCluster
	for _, l := range ws.LocationClusters {
		if hasPlayer && l.ID == player.CurrentLocationClusterID {
			playerLocation = &l
			break
		}
	}

	var presentChars []models.Character
	for _, c := range ws.Characters {
		if !c.IsPlayer && c.IsDiscovered && hasPlayer && c.CurrentLocationClusterID == player.CurrentLocationClusterID {
			presentChars = append(presentChars, c)
		}
	}

	var charDescriptions strings.Builder
	for _, c := range presentChars {
		knowledgeStr := ""
		if len(c.Knowledge) > 0 {
			last3 := c.Knowledge
			if len(last3) > 3 {
				last3 = last3[len(last3)-3:]
			}
			var parts []string
			for _, k := range last3 {
				parts = append(parts, k.Content)
			}
			knowledgeStr = "\n    Knows: " + strings.Join(parts, "; ")
		}
		fmt.Fprintf(&charDescriptions, "- %s: %s%s\n", c.Name, c.Description, knowledgeStr)
	}

	var recentEvents strings.Builder
	events := ws.Events
	if len(events) > 5 {
		events = events[len(events)-5:]
	}
	for _, e := range events {
		fmt.Fprintf(&recentEvents, "- %s\n", e.Description)
	}

	var undiscoveredHint string
	var undiscovered []string
	for _, c := range ws.Characters {
		if !c.IsPlayer && !c.IsDiscovered && hasPlayer && c.CurrentLocationClusterID == player.CurrentLocationClusterID {
			undiscovered = append(undiscovered, c.Name)
		}
	}
	if len(undiscovered) > 0 {
		undiscoveredHint = "\nHIDDEN (can be discovered if player looks around or circumstances arise): " + strings.Join(undiscovered, ", ")
	}

	var otherLocations []string
	for _, loc := range ws.LocationClusters {
		if hasPlayer && loc.ID != player.CurrentLocationClusterID {
			otherLocations = append(otherLocations, loc.CanonicalName)
		}
	}
	otherLocStr := strings.Join(otherLocations, ", ")
	if otherLocStr == "" {
		otherLocStr = "None yet"
	}

	locationName := "Unknown"
	if playerLocation != nil {
		locationName = playerLocation.CanonicalName
	}

	charDesc := charDescriptions.String()
	if charDesc == "" {
		charDesc = "(No one else is here)"
	}

	eventStr := recentEvents.String()
	eventsSection := ""
	if eventStr != "" {
		eventsSection = "RECENT EVENTS:\n" + eventStr + "\n"
	}

	return fmt.Sprintf(`You are the narrator and game master of an interactive narrative experience called "%s".

SCENARIO: %s

CURRENT LOCATION: %s
OTHER KNOWN LOCATIONS: %s
TIME: %s (tick %d)

CHARACTERS PRESENT (SYSTEM STATE):
%s(NOTE: If a character is participating in the conversation but is NOT listed above, they are not yet discovered. You MUST call discoverCharacter for them immediately.)
%s

%s
YOUR ROLE:
- Narrate the world and characters in response to what the player does
- Play the characters present - give them distinct voices and personalities
- Characters should only know what they have witnessed or been told
- When the player moves to a new location, describe it vividly
- Include sensory details and atmosphere
- Keep responses focused and not overly long
- Characters can suggest actions but never force the player

Tools:
- Use moveToLocation when the player goes somewhere new
- Use advanceTime when significant time passes (long conversations, waiting, etc.)
- Use discoverCharacter when introducing ANY new character (hidden or improvised)

IMPORTANT:
- Stay in character as the narrator
- Never break the fourth wall
- Don't explain game mechanics
- Let the player drive the story
- If you introduce or mention any character (whether from the "HIDDEN" list or a new one you create), you MUST call the discoverCharacter tool for them. Do not just describe them; use the tool to make them official.
- Check the recent history: if a character has been speaking or present but is NOT in the "CHARACTERS PRESENT" list above, call discoverCharacter for them immediately!
- You can call multiple tools in a single turn if needed (e.g. discovering two characters).

EXAMPLES:
- Player walks into a tavern with two unknown people → call discoverCharacter for each one AND write your narrative
- Player asks to go to the market → call moveToLocation with the destination, then narrate the arrival
- A long conversation happens → call advanceTime to reflect the passage of time`,
		ws.Scenario.Title, ws.Scenario.Description,
		locationName, otherLocStr,
		ws.Time.NarrativeTime, ws.Time.Tick,
		charDesc, undiscoveredHint, eventsSection)
}

func buildChatTools(ws *models.WorldState) []ai.Tool {
	return []ai.Tool{
		{
			Type: "function",
			Function: ai.ToolFunction{
				Name:        "moveToLocation",
				Description: "Call this when the player moves to a different location. This advances time and updates their position.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"destination": map[string]any{
							"type":        "string",
							"description": "Brief description of where they are going",
						},
						"narrativeTime": map[string]any{
							"type":        []string{"string", "null"},
							"description": "New narrative time if significant time passes",
						},
					},
					"required": []string{"destination"},
				},
			},
		},
		{
			Type: "function",
			Function: ai.ToolFunction{
				Name:        "advanceTime",
				Description: "Call this when significant time passes without movement",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"narrativeTime": map[string]any{
							"type":        "string",
							"description": "New narrative time description",
						},
						"ticks": map[string]any{
							"type":        "number",
							"description": "How many time units pass (default: 5)",
						},
					},
					"required": []string{"narrativeTime"},
				},
			},
		},
		{
			Type: "function",
			Function: ai.ToolFunction{
				Name:        "discoverCharacter",
				Description: "Call this when the player encounters or notices a new character. CALL THIS SEPARATELY FOR EACH CHARACTER.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"characterName": map[string]any{
							"type":        "string",
							"description": "Name of the character being discovered",
						},
						"introduction": map[string]any{
							"type":        "string",
							"description": "How they are introduced or noticed",
						},
						"goals": map[string]any{
							"type":        "string",
							"description": "Inferred or stated goals of the character",
						},
					},
					"required": []string{"characterName", "introduction"},
				},
			},
		},
	}
}

func (a *App) processToolCalls(ctx context.Context, session *world.SessionState, toolCalls []ai.ToolCall, messageID string, notify func(event, data string)) []toolCallResult {
	var results []toolCallResult
	for _, tc := range toolCalls {
		var result string
		switch tc.Function.Name {
		case "moveToLocation":
			r, err := a.handleMoveToLocation(ctx, session, tc, messageID, notify)
			if err != nil {
				slog.Error("moveToLocation failed", "error", err)
				r = "Failed to move."
			}
			result = r
		case "advanceTime":
			result = a.handleAdvanceTime(session, tc)
		case "discoverCharacter":
			result = a.handleDiscoverCharacter(session, tc, messageID)
		default:
			result = "Unknown tool."
		}
		results = append(results, toolCallResult{ToolCall: tc, Result: result})
	}
	return results
}

func (a *App) handleMoveToLocation(ctx context.Context, session *world.SessionState, tc ai.ToolCall, messageID string, notify func(event, data string)) (string, error) {
	var args struct {
		Destination   string  `json:"destination"`
		NarrativeTime *string `json:"narrativeTime"`
	}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("parse moveToLocation args: %w", err)
	}

	ws := session.GetWorld()
	if ws == nil {
		return "Could not move.", nil
	}

	player, ok := findPlayer(ws)
	if !ok {
		return "Could not move.", nil
	}
	previousLocationID := player.CurrentLocationClusterID

	modelID := session.GetModelID()
	resolved, err := world.ResolveLocation(ctx, args.Destination, ws.LocationClusters, modelID)
	if err != nil || resolved == nil {
		name := world.ExtractCanonicalName(args.Destination)
		resolved = &world.ResolveLocationResult{
			ClusterID:     nil,
			CanonicalName: name,
			IsNew:         true,
		}
	}

	var clusterID string
	if resolved.IsNew {
		newCluster := session.AddLocationCluster(resolved.CanonicalName)
		clusterID = newCluster.ID
	} else if resolved.ClusterID != nil {
		clusterID = *resolved.ClusterID
	}

	if clusterID != "" {
		session.MoveCharacter(ws.PlayerCharacterID, clusterID)

		timeSinceLastSim := ws.Time.Tick - session.GetLastSimulationTick()
		if timeSinceLastSim > 5 && previousLocationID != clusterID {
			session.SetIsSimulating(true)
			if notify != nil {
				notify("simulating", "Simulating the world...")
			}
			simResult, err := world.SimulateOffscreen(ctx, ws, clusterID, timeSinceLastSim, modelID)
			if err == nil && simResult != nil {
				for _, event := range simResult.Events {
					event.SourceMessageID = messageID
					session.AddEvent(event)
					for _, witnessID := range event.WitnessedByIDs {
						session.UpdateCharacterKnowledge(witnessID, event.Description, ws.Time.Tick, "witnessed")
					}
				}
				for _, conv := range simResult.Conversations {
					session.AddConversation(conv)
				}
				for _, update := range simResult.CharacterUpdates {
					session.MoveCharacter(update.CharacterID, update.NewLocationID)
				}
			}
			session.SetLastSimulationTick(ws.Time.Tick)
			session.SetIsSimulating(false)
			if notify != nil {
				notify("simulated", "")
			}
		}
	}

	narrativeTime := ""
	if args.NarrativeTime != nil {
		narrativeTime = *args.NarrativeTime
	}
	session.AdvanceTime(models.TimeCosts["move"], narrativeTime)

	destName := resolved.CanonicalName
	if destName == "" {
		destName = args.Destination
	}
	chars := session.GetCharactersAtPlayerLocation()
	if len(chars) > 0 {
		var names []string
		for _, c := range chars {
			names = append(names, c.Name)
		}
		return fmt.Sprintf("Moved to %s. Characters present: %s.", destName, strings.Join(names, ", ")), nil
	}
	return fmt.Sprintf("Moved to %s.", destName), nil
}

func (a *App) handleAdvanceTime(session *world.SessionState, tc ai.ToolCall) string {
	var args struct {
		NarrativeTime string `json:"narrativeTime"`
		Ticks         *int   `json:"ticks"`
	}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		slog.Error("failed to parse tool args", "tool", "advanceTime", "error", err)
		return "Failed to advance time."
	}

	ticks := 5
	if args.Ticks != nil {
		ticks = *args.Ticks
	}
	session.AdvanceTime(ticks, args.NarrativeTime)
	return fmt.Sprintf("Time advanced by %d ticks. It is now %s.", ticks, args.NarrativeTime)
}

func (a *App) handleDiscoverCharacter(session *world.SessionState, tc ai.ToolCall, messageID string) string {
	var args struct {
		CharacterName string `json:"characterName"`
		Introduction  string `json:"introduction"`
		Goals         string `json:"goals"`
	}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		slog.Error("failed to parse tool args", "tool", "discoverCharacter", "error", err)
		return "Failed to discover character."
	}

	match, found := session.FindBestCharacterMatch(args.CharacterName)
	if found {
		session.DiscoverCharacter(match.ID)
	} else {
		player, ok := session.GetPlayerCharacter()
		locationID := ""
		if ok {
			locationID = player.CurrentLocationClusterID
		}
		if locationID == "" {
			if ws := session.GetWorld(); ws != nil && len(ws.LocationClusters) > 0 {
				locationID = ws.LocationClusters[0].ID
			}
		}

		session.AddCharacter(models.Character{
			Name:                     args.CharacterName,
			Description:              args.Introduction,
			IsPlayer:                 false,
			EncounterChance:          0,
			CurrentLocationClusterID: locationID,
			Knowledge:                []models.KnowledgeEntry{},
			Relationships:            []models.Relationship{},
			IsDiscovered:             true,
			CreatedByMessageID:       messageID,
			Goals:                    args.Goals,
		})
	}
	return fmt.Sprintf("Character %s discovered.", args.CharacterName)
}

func findPlayer(ws *models.WorldState) (models.Character, bool) {
	for _, c := range ws.Characters {
		if c.ID == ws.PlayerCharacterID {
			return c, true
		}
	}
	return models.Character{}, false
}
