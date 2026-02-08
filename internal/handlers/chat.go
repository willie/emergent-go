package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"emergent/internal/ai"
	"emergent/internal/models"
	"emergent/internal/world"

	"github.com/google/uuid"
)

// writeSSE writes a single SSE event to the response
func writeSSE(w http.ResponseWriter, flusher http.Flusher, event, data string) {
	if event != "" {
		fmt.Fprintf(w, "event: %s\n", event)
	}
	// SSE data lines: split on newlines so each gets "data: " prefix
	for _, line := range strings.Split(data, "\n") {
		fmt.Fprintf(w, "data: %s\n", line)
	}
	fmt.Fprint(w, "\n")
	flusher.Flush()
}

// ChatSend handles a new chat message, returning an SSE stream
func (a *App) ChatSend(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	r.ParseForm()
	message := strings.TrimSpace(r.FormValue("message"))
	if message == "" {
		http.Error(w, "Empty message", 400)
		return
	}

	session := a.Session
	if session.World == nil {
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
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	session := a.Session
	if session.World == nil {
		http.Error(w, "No active game", 400)
		return
	}

	session.AdvanceTime(1, "")

	a.streamResponse(w, r, session, nil)
}

// streamResponse runs the LLM and streams SSE events
func (a *App) streamResponse(w http.ResponseWriter, r *http.Request, session *world.SessionState, userMsg *models.ChatMessage) {
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
		html := fmt.Sprintf(
			`<div class="flex justify-end"><div class="max-w-4/5 rounded-lg px-4 py-2 bg-blue-600 text-white"><div class="prose prose-invert max-w-none break-words whitespace-pre-wrap">%s</div></div></div>`,
			escapeHTML(userMsg.Content))
		writeSSE(w, flusher, "user-message", html)
	}

	// Build AI messages
	aiMessages := a.buildAIMessages(session)
	systemPrompt := buildSystemPrompt(session.World)
	tools := buildChatTools(session.World)

	fullMessages := append([]ai.ChatMessage{
		{Role: "system", Content: systemPrompt},
	}, aiMessages...)

	model := session.ModelID
	if model == "" {
		model = ai.Models.MainConversation
	}

	// Stream tokens
	result, err := ai.StreamText(model, fullMessages, tools, func(content string) {
		writeSSE(w, flusher, "token", escapeHTML(content))
	})

	if err != nil {
		writeSSE(w, flusher, "error", escapeHTML(err.Error()))
		writeSSE(w, flusher, "done", "")
		return
	}

	// Save assistant message
	assistantMsg := models.ChatMessage{
		ID:      uuid.New().String(),
		Role:    "assistant",
		Content: result.Content,
	}
	session.AddChatMessage(assistantMsg)

	// Process tool calls and build OOB updates
	if len(result.ToolCalls) > 0 {
		a.processToolCalls(session, result.ToolCalls, assistantMsg.ID)

		// Render OOB swap fragments for changed state
		oobHTML := a.renderOOBUpdates(session)
		if oobHTML != "" {
			writeSSE(w, flusher, "oob", oobHTML)
		}
	}

	// Persist state
	if err := session.Persist(); err != nil {
		log.Printf("Failed to persist state: %v", err)
	}

	writeSSE(w, flusher, "done", "")
}

// renderOOBUpdates renders out-of-band swap HTML for header, sidebar, etc.
func (a *App) renderOOBUpdates(session *world.SessionState) string {
	var buf strings.Builder

	// Location name
	location := session.GetPlayerLocation()
	locName := "Unknown"
	if location != nil {
		locName = location.CanonicalName
	}
	fmt.Fprintf(&buf, `<span id="location-name" hx-swap-oob="innerHTML">%s</span>`, escapeHTML(locName))

	// Narrative time
	fmt.Fprintf(&buf, `<span id="narrative-time" hx-swap-oob="innerHTML">%s</span>`, escapeHTML(session.World.Time.NarrativeTime))

	// Tick count
	fmt.Fprintf(&buf, `<div id="tick-count" hx-swap-oob="innerHTML">tick %d</div>`, session.World.Time.Tick)

	// Present characters
	nearby := session.GetCharactersAtPlayerLocation()
	var names []string
	for _, c := range nearby {
		names = append(names, escapeHTML(c.Name))
	}
	presentHTML := strings.Join(names, ", ")
	fmt.Fprintf(&buf, `<span id="present-chars" hx-swap-oob="innerHTML">%s</span>`, presentHTML)

	// Offscreen conversations
	offscreenHTML := a.renderOffscreenPartial(session)
	fmt.Fprintf(&buf, `<div id="offscreen-content" hx-swap-oob="innerHTML">%s</div>`, offscreenHTML)

	// Characters list
	charsHTML := a.renderCharactersPartial(session)
	fmt.Fprintf(&buf, `<div id="characters-content" hx-swap-oob="innerHTML">%s</div>`, charsHTML)

	return buf.String()
}

// renderOffscreenPartial renders the offscreen conversations HTML
func (a *App) renderOffscreenPartial(session *world.SessionState) string {
	convs := session.GetOffscreenConversations()
	if len(convs) == 0 {
		return `<div class="p-4 text-sm text-zinc-600 text-center"><p>No other conversations happening right now.</p><p class="mt-2 text-xs">When characters interact without you, their conversations will appear here.</p></div>`
	}

	var buf strings.Builder
	for _, conv := range convs {
		var names []string
		for _, pid := range conv.ParticipantIDs {
			c := session.GetCharacterByID(pid)
			if c != nil {
				names = append(names, escapeHTML(c.Name))
			}
		}
		locName := "Unknown"
		loc := session.GetLocationCluster(conv.LocationClusterID)
		if loc != nil {
			locName = loc.CanonicalName
		}

		fmt.Fprintf(&buf, `<div class="border-b border-zinc-800">`)
		fmt.Fprintf(&buf, `<button onclick="toggleOffscreen('%s')" class="w-full px-4 py-3 flex items-center justify-between hover:bg-zinc-900 transition-colors text-left">`, conv.ID)
		fmt.Fprintf(&buf, `<div class="flex flex-col gap-0.5"><span class="text-sm font-medium text-zinc-200">%s</span>`, strings.Join(names, " &amp; "))
		fmt.Fprintf(&buf, `<span class="text-xs text-zinc-500">%s</span></div>`, escapeHTML(locName))
		fmt.Fprintf(&buf, `<svg id="offscreen-arrow-%s" class="w-4 h-4 text-zinc-500 transition-transform rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>`, conv.ID)
		fmt.Fprintf(&buf, `</button>`)
		fmt.Fprintf(&buf, `<div id="offscreen-detail-%s" class="max-h-64 overflow-y-auto px-4 pb-4 space-y-2">`, conv.ID)
		if len(conv.Messages) == 0 {
			buf.WriteString(`<p class="text-xs text-zinc-600 italic">Nothing yet...</p>`)
		} else {
			for _, msg := range conv.Messages {
				buf.WriteString(`<div class="text-sm">`)
				if msg.SpeakerID != "" {
					speaker := session.GetCharacterByID(msg.SpeakerID)
					if speaker != nil {
						fmt.Fprintf(&buf, `<span class="font-medium text-zinc-400">%s: </span>`, escapeHTML(speaker.Name))
					}
				}
				fmt.Fprintf(&buf, `<span class="text-zinc-300">%s</span></div>`, escapeHTML(msg.Content))
			}
		}
		buf.WriteString(`</div></div>`)
	}
	return buf.String()
}

// renderCharactersPartial renders the characters list HTML
func (a *App) renderCharactersPartial(session *world.SessionState) string {
	chars := session.GetDiscoveredCharacters()
	if len(chars) == 0 {
		return `<div class="p-4 text-sm text-zinc-600 text-center">No characters discovered yet.</div>`
	}

	var buf bytes.Buffer
	tmpl := a.PageTemplates["game.html"]
	if tmpl == nil {
		// Fallback: build inline
		var sb strings.Builder
		sb.WriteString(`<div class="flex flex-col">`)
		for _, c := range chars {
			locName := "Unknown"
			loc := session.GetLocationCluster(c.CurrentLocationClusterID)
			if loc != nil {
				locName = loc.CanonicalName
			}
			fmt.Fprintf(&sb, `<div class="border-b border-zinc-800">
				<button onclick="toggleCharacter('%s')" class="w-full px-4 py-3 flex items-center justify-between hover:bg-zinc-900 transition-colors text-left">
				<div class="flex flex-col gap-0.5 flex-1 mr-4"><span class="text-sm font-medium text-zinc-200">%s</span>
				<span class="text-xs text-zinc-500">%s</span></div>
				<svg id="char-arrow-%s" class="w-4 h-4 text-zinc-500 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
				</button>
				<div id="char-detail-%s" class="hidden px-4 pb-4 space-y-3"><p class="text-sm text-zinc-400">%s</p></div>
				</div>`,
				c.ID, escapeHTML(c.Name), escapeHTML(locName), c.ID, c.ID, escapeHTML(c.Description))
		}
		sb.WriteString(`</div>`)
		return sb.String()
	}

	// Use the cached template's character_list block
	data := a.buildGameData(session)
	if err := tmpl.ExecuteTemplate(&buf, "character_list", data); err != nil {
		log.Printf("renderCharactersPartial error: %v", err)
		return `<div class="p-4 text-sm text-zinc-600 text-center">Error rendering characters.</div>`
	}
	return buf.String()
}

func (a *App) buildAIMessages(session *world.SessionState) []ai.ChatMessage {
	var messages []ai.ChatMessage
	for _, msg := range session.ChatMessages {
		messages = append(messages, ai.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	return messages
}

func buildSystemPrompt(ws *models.WorldState) string {
	player := findPlayer(ws)
	var playerLocation *models.LocationCluster
	for _, l := range ws.LocationClusters {
		if player != nil && l.ID == player.CurrentLocationClusterID {
			playerLocation = &l
			break
		}
	}

	var presentChars []models.Character
	for _, c := range ws.Characters {
		if !c.IsPlayer && c.IsDiscovered && player != nil && c.CurrentLocationClusterID == player.CurrentLocationClusterID {
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
		if !c.IsPlayer && !c.IsDiscovered && player != nil && c.CurrentLocationClusterID == player.CurrentLocationClusterID {
			undiscovered = append(undiscovered, c.Name)
		}
	}
	if len(undiscovered) > 0 {
		undiscoveredHint = "\nHIDDEN (can be discovered if player looks around or circumstances arise): " + strings.Join(undiscovered, ", ")
	}

	var otherLocations []string
	for _, loc := range ws.LocationClusters {
		if player != nil && loc.ID != player.CurrentLocationClusterID {
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
- If you introduce or mention any character (whether from the "HIDDEN" list or a new one you create), you MUST call the discoverCharacter tool for them.
- You can call multiple tools in a single turn if needed.`,
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
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"destination": map[string]interface{}{
							"type":        "string",
							"description": "Brief description of where they are going",
						},
						"narrativeTime": map[string]interface{}{
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
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"narrativeTime": map[string]interface{}{
							"type":        "string",
							"description": "New narrative time description",
						},
						"ticks": map[string]interface{}{
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
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"characterName": map[string]interface{}{
							"type":        "string",
							"description": "Name of the character being discovered",
						},
						"introduction": map[string]interface{}{
							"type":        "string",
							"description": "How they are introduced or noticed",
						},
						"goals": map[string]interface{}{
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

func (a *App) processToolCalls(session *world.SessionState, toolCalls []ai.ToolCall, messageID string) {
	for _, tc := range toolCalls {
		switch tc.Function.Name {
		case "moveToLocation":
			a.handleMoveToLocation(session, tc, messageID)
		case "advanceTime":
			a.handleAdvanceTime(session, tc)
		case "discoverCharacter":
			a.handleDiscoverCharacter(session, tc, messageID)
		}
	}
}

func (a *App) handleMoveToLocation(session *world.SessionState, tc ai.ToolCall, messageID string) {
	var args struct {
		Destination   string  `json:"destination"`
		NarrativeTime *string `json:"narrativeTime"`
	}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		log.Printf("Failed to parse moveToLocation args: %v", err)
		return
	}

	ws := session.World
	if ws == nil {
		return
	}

	player := findPlayer(ws)
	if player == nil {
		return
	}
	previousLocationID := player.CurrentLocationClusterID

	var clusters []struct {
		ID            string `json:"id"`
		CanonicalName string `json:"canonicalName"`
	}
	for _, c := range ws.LocationClusters {
		clusters = append(clusters, struct {
			ID            string `json:"id"`
			CanonicalName string `json:"canonicalName"`
		}{c.ID, c.CanonicalName})
	}

	resolved, err := world.ResolveLocation(args.Destination, clusters, session.ModelID)
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

		timeSinceLastSim := ws.Time.Tick - session.LastSimulationTick
		if timeSinceLastSim > 5 && previousLocationID != clusterID {
			session.IsSimulating = true
			simResult, err := world.SimulateOffscreen(ws, clusterID, timeSinceLastSim, session.ModelID)
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
			session.LastSimulationTick = ws.Time.Tick
			session.IsSimulating = false
		}
	}

	narrativeTime := ""
	if args.NarrativeTime != nil {
		narrativeTime = *args.NarrativeTime
	}
	session.AdvanceTime(models.TimeCosts["move"], narrativeTime)
}

func (a *App) handleAdvanceTime(session *world.SessionState, tc ai.ToolCall) {
	var args struct {
		NarrativeTime string `json:"narrativeTime"`
		Ticks         *int   `json:"ticks"`
	}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		log.Printf("Failed to parse advanceTime args: %v", err)
		return
	}

	ticks := 5
	if args.Ticks != nil {
		ticks = *args.Ticks
	}
	session.AdvanceTime(ticks, args.NarrativeTime)
}

func (a *App) handleDiscoverCharacter(session *world.SessionState, tc ai.ToolCall, messageID string) {
	var args struct {
		CharacterName string `json:"characterName"`
		Introduction  string `json:"introduction"`
		Goals         string `json:"goals"`
	}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		log.Printf("Failed to parse discoverCharacter args: %v", err)
		return
	}

	match := session.FindBestCharacterMatch(args.CharacterName)
	if match != nil {
		session.DiscoverCharacter(match.ID)
	} else {
		player := session.GetPlayerCharacter()
		locationID := ""
		if player != nil {
			locationID = player.CurrentLocationClusterID
		}
		if locationID == "" && len(session.World.LocationClusters) > 0 {
			locationID = session.World.LocationClusters[0].ID
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
}

func findPlayer(ws *models.WorldState) *models.Character {
	for _, c := range ws.Characters {
		if c.ID == ws.PlayerCharacterID {
			return &c
		}
	}
	return nil
}

func escapeHTML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "'", "&#39;")
	return s
}
