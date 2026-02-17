# Prompt Construction Pipeline

How the final LLM prompt is assembled from raw inputs to the API call in Emergent.

---

## Overview

Emergent is an interactive narrative system where an LLM acts as narrator/game master. The prompt is assembled on every user message (and re-assembled mid-turn when tool calls mutate world state). There are two distinct prompt pipelines:

1. **Main conversation** — the player-facing narrator (streaming, tool-calling loop)
2. **Off-screen simulation** — NPC conversations that happen while the player is elsewhere (non-streaming)

Both use the OpenRouter chat completions API in the OpenAI-compatible message format.

---

## 1. Entry Point: User Message to API Call

**File:** `internal/handlers/chat.go:40-65`

```go
func (a *App) ChatSend(w http.ResponseWriter, r *http.Request) {
    message := strings.TrimSpace(r.FormValue("message"))
    // ...
    session.AdvanceTime(1, "")        // tick +1 for user action
    userMsg := models.ChatMessage{
        ID:      uuid.New().String(),
        Role:    "user",
        Content: message,
    }
    session.AddChatMessage(userMsg)
    a.streamResponse(w, r, session, &userMsg)
}
```

The user's raw text from the HTML form is trimmed and stored verbatim — no preprocessing, macro substitution, or variable expansion. The message is added to `session.ChatMessages` as-is.

There are three entry points that trigger prompt assembly:
- `ChatSend` (line 40) — normal user message
- `ChatContinue` (line 68) — generate another response without user input
- `RegenerateChat` (line 273) — pop last assistant message, re-generate

All three converge on `streamResponse` (line 81).

---

## 2. System Prompt Construction

**File:** `internal/handlers/chat.go:382-510`
**Function:** `buildSystemPrompt(ws *models.WorldState) string`

The system prompt is a single `fmt.Sprintf` call with dynamic sections computed from the current `WorldState`. It is fully rebuilt on every API call — and rebuilt again within the same turn if tool calls change the world.

### Template (hardcoded, lines 463-505)

```
You are the narrator and game master of an interactive narrative experience called "{Title}".

SCENARIO: {Description}

CURRENT LOCATION: {locationName}
OTHER KNOWN LOCATIONS: {otherLocations}
TIME: {narrativeTime} (tick {tick})

CHARACTERS PRESENT (SYSTEM STATE):
{charDescriptions}
(NOTE: If a character is participating in the conversation but is NOT listed above, ...)
{undiscoveredHint}

{eventsSection}

YOUR ROLE:
- Narrate the world and characters in response to what the player does
- Play the characters present - give them distinct voices and personalities
- Characters should only know what they have witnessed or been told
- When the player moves to a new location, describe it vividly
- Include sensory details and atmosphere
- Keep responses to 2-4 paragraphs. Be vivid but concise
- Characters can suggest actions but never force the player

Tools:
- Use moveToLocation when the player goes somewhere new
- Use advanceTime when significant time passes (long conversations, waiting, etc.)
- Use discoverCharacter when introducing ANY new character (hidden or improvised)

CRITICAL RULE — DISCOVERING CHARACTERS:
[... 4 bullet points about character discovery ...]

IMPORTANT:
- Stay in character as the narrator
- Never break the fourth wall
- Don't explain game mechanics
- Let the player drive the story

EXAMPLES:
[... 3 concrete examples of tool usage ...]
```

### Dynamic sections and where they come from

#### 2a. Scenario Title & Description

**Source:** `ws.Scenario.Title` and `ws.Scenario.Description`
**Defined in:** `internal/world/scenarios.go` (built-in) or user-imported JSON
**Data model:** `internal/models/scenario.go:22-29`

```go
type ScenarioConfig struct {
    Title                 string            `json:"title"`
    Description           string            `json:"description"`
    InitialNarrativeTime  string            `json:"initialNarrativeTime"`
    Locations             []InitialLocation `json:"locations"`
    Characters            []CharacterConfig `json:"characters"`
    PlayerStartingLocation string           `json:"playerStartingLocation"`
}
```

Built-in scenarios are defined as Go literals in `internal/world/scenarios.go:6-45`. Custom scenarios are imported as JSON via `POST /scenario/import` (handlers.go:436-471) and stored in the `custom_scenarios` storage key.

Example (The Dusty Tankard, scenarios.go:8-9):
```go
Title:       "The Dusty Tankard",
Description: "A medieval fantasy tavern at a crossroads. Rumors of a dragon sighting...",
```

These strings are injected directly into the template at lines 506-507:
```go
ws.Scenario.Title, ws.Scenario.Description,
```

#### 2b. Current Location

**Source:** The player character's `CurrentLocationClusterID` resolved to a `LocationCluster.CanonicalName`
**Code path (chat.go:383-390):**
```go
player, hasPlayer := findPlayer(ws)
for _, l := range ws.LocationClusters {
    if hasPlayer && l.ID == player.CurrentLocationClusterID {
        playerLocation = &l
        break
    }
}
```

The location name falls back to `"Unknown"` if no match (line 447-450).

#### 2c. Other Known Locations

**Source:** All `LocationCluster` entries except the player's current one
**Code path (chat.go:436-445):**
```go
for _, loc := range ws.LocationClusters {
    if hasPlayer && loc.ID != player.CurrentLocationClusterID {
        otherLocations = append(otherLocations, loc.CanonicalName)
    }
}
otherLocStr := strings.Join(otherLocations, ", ")
if otherLocStr == "" {
    otherLocStr = "None yet"
}
```

New locations are added dynamically when the LLM calls `moveToLocation` with a destination that doesn't match existing clusters (via `ResolveLocation` in `internal/world/locations.go`).

#### 2d. Time

**Source:** `ws.Time.NarrativeTime` (string, e.g. "Late afternoon") and `ws.Time.Tick` (int)
**Mutated by:**
- `session.AdvanceTime(1, "")` on every user message (chat.go:54)
- `moveToLocation` tool: advances by `TimeCosts["move"]` = 5 ticks (chat.go:668)
- `advanceTime` tool: advances by specified ticks (chat.go:746)

#### 2e. Characters Present (the core character injection)

**Code path (chat.go:392-414):**
```go
// Collect discovered, non-player characters at the player's location
for _, c := range ws.Characters {
    if !c.IsPlayer && c.IsDiscovered && hasPlayer &&
       c.CurrentLocationClusterID == player.CurrentLocationClusterID {
        presentChars = append(presentChars, c)
    }
}

// Build descriptions with knowledge
for _, c := range presentChars {
    // Last 3 knowledge entries
    last3 := c.Knowledge
    if len(last3) > 3 {
        last3 = last3[len(last3)-3:]
    }
    // Format: "- CharName: Description\n    Knows: entry1; entry2; entry3"
    fmt.Fprintf(&charDescriptions, "- %s: %s%s\n", c.Name, c.Description, knowledgeStr)
}
```

Each character entry in the system prompt looks like:
```
- Grim: The grizzled barkeep of the Dusty Tankard. Knows every rumor that passes through.
    Knows: The merchant Bran arrived yesterday; A strange light was seen in the forest; ...
```

**Character model** (`internal/models/world.go:36-48`):
```go
type Character struct {
    ID                       string
    Name                     string
    Description              string
    CurrentLocationClusterID string
    Knowledge                []KnowledgeEntry
    Relationships            []Relationship
    IsPlayer                 bool
    IsDiscovered             bool
    EncounterChance          float64
    Goals                    string
    CreatedByMessageID       string
}
```

**Knowledge model** (`internal/models/world.go:19-25`):
```go
type KnowledgeEntry struct {
    ID                string
    Content           string
    AcquiredAt        int       // tick
    Source            string    // "witnessed", "told", "inferred"
    SourceCharacterID string
}
```

Only the last 3 knowledge entries per character are included. Knowledge is added when characters witness world events from off-screen simulation (`state.go:261-279`).

If no characters are present, the section reads `"(No one else is here)"` (chat.go:453-455).

#### 2f. Undiscovered Characters Hint

**Code path (chat.go:425-434):**
```go
for _, c := range ws.Characters {
    if !c.IsPlayer && !c.IsDiscovered && hasPlayer &&
       c.CurrentLocationClusterID == player.CurrentLocationClusterID {
        undiscovered = append(undiscovered, c.Name)
    }
}
if len(undiscovered) > 0 {
    undiscoveredHint = "\nHIDDEN (can be discovered if player looks around ...): " +
        strings.Join(undiscovered, ", ")
}
```

This tells the model which NPCs are at the location but haven't been formally introduced yet. It's injected after the character descriptions.

#### 2g. Recent Events Section

**Code path (chat.go:416-424, 457-461):**
```go
events := ws.Events
if len(events) > 5 {
    events = events[len(events)-5:]
}
for _, e := range events {
    fmt.Fprintf(&recentEvents, "- %s\n", e.Description)
}
// Only included if non-empty
if eventStr != "" {
    eventsSection = "RECENT EVENTS:\n" + eventStr + "\n"
}
```

Events come from off-screen simulation results and are capped at the 5 most recent. Each event has a description like "Grim and Sera discussed the dragon rumors." Events are created by the `reportSimulation` tool call during off-screen simulation (`internal/world/simulation.go:291-305`).

---

## 3. Chat History Assembly & Context Trimming

**File:** `internal/handlers/chat.go:337-380`

### Token Budget

```go
const maxMessageTokens = 100_000  // line 349
```

The system is budgeted for 100K tokens total across system prompt + chat history, leaving room for tool definitions (~1K) and the response (~2-4K) within the smallest supported model context (128K).

### Token Estimation

```go
func estimateTokens(s string) int {
    n := len(s) / 4    // ~4 chars per token approximation
    if n == 0 && len(s) > 0 {
        return 1
    }
    return n
}
```

No tokenizer library is used — it's a simple character-count heuristic (line 338-344).

### Trimming Algorithm

```go
func (a *App) buildAIMessages(systemPrompt string, chatMessages []models.ChatMessage) []ai.ChatMessage {
    budgetRemaining := maxMessageTokens - estimateTokens(systemPrompt)
    if budgetRemaining < 0 {
        budgetRemaining = 0
    }

    // Walk backwards to keep the most recent messages
    start := len(chatMessages)
    for i := len(chatMessages) - 1; i >= 0; i-- {
        cost := estimateTokens(chatMessages[i].Content)
        if cost > budgetRemaining {
            break
        }
        budgetRemaining -= cost
        start = i
    }

    // Convert to AI message format
    messages := make([]ai.ChatMessage, 0, len(chatMessages)-start)
    for _, msg := range chatMessages[start:] {
        messages = append(messages, ai.ChatMessage{
            Role:    msg.Role,
            Content: msg.Content,
        })
    }
    return messages
}
```

The algorithm is recency-biased: walk backward from the newest message, greedily including messages until the budget is exhausted. Older messages are silently dropped. There is no summarization — dropped messages are simply lost from context.

Chat messages are stored as `[]models.ChatMessage` with only `Role` ("user" or "assistant") and `Content` (plain text). Tool call messages (assistant with tool_calls, tool results) are NOT persisted to chat history — they only exist within the streaming loop's `fullMessages` slice.

---

## 4. Final Message Array Assembly

**File:** `internal/handlers/chat.go:124-133`

```go
chatMessages := session.GetChatMessagesCopy()
ws := session.GetWorld()
systemPrompt := buildSystemPrompt(ws)
aiMessages := a.buildAIMessages(systemPrompt, chatMessages)
tools := buildChatTools(ws)

fullMessages := append([]ai.ChatMessage{
    {Role: "system", Content: systemPrompt},
}, aiMessages...)
```

The final message array sent to the API is:

| Index | Role | Content |
|-------|------|---------|
| 0 | `system` | The full system prompt (see section 2) |
| 1 | `user` | Oldest retained chat message |
| ... | `user`/`assistant` | Alternating chat history |
| N | `user` | Most recent user message |

Within a tool-calling loop, the array grows with `assistant` (tool_calls) and `tool` (results) messages appended in-place (lines 165-183).

---

## 5. Tool Definitions

**File:** `internal/handlers/chat.go:512-587`
**Function:** `buildChatTools(ws *models.WorldState) []ai.Tool`

Three tools are defined as static JSON schemas:

### moveToLocation
```json
{
  "name": "moveToLocation",
  "description": "Call this when the player moves to a different location...",
  "parameters": {
    "type": "object",
    "properties": {
      "destination":    { "type": "string" },
      "narrativeTime":  { "type": "string" },
      "accompaniedBy":  { "type": "array", "items": {"type": "string"} }
    },
    "required": ["destination", "narrativeTime", "accompaniedBy"]
  }
}
```

### advanceTime
```json
{
  "name": "advanceTime",
  "parameters": {
    "properties": {
      "narrativeTime": { "type": "string" },
      "ticks":         { "type": "number" }
    },
    "required": ["narrativeTime", "ticks"]
  }
}
```

### discoverCharacter
```json
{
  "name": "discoverCharacter",
  "parameters": {
    "properties": {
      "characterName":  { "type": "string" },
      "introduction":   { "type": "string" },
      "goals":          { "type": "string" }
    },
    "required": ["characterName", "introduction", "goals"]
  }
}
```

These are passed to the API as the `tools` field. No `tool_choice` is set for main conversation (defaults to `"auto"`).

---

## 6. The Tool-Calling Loop

**File:** `internal/handlers/chat.go:140-196`

```go
const maxToolSteps = 5

for step := range maxToolSteps {
    result, err := ai.StreamText(ctx, model, fullMessages, tools, func(content string) {
        writeSSE(w, flusher, "token", content)
    })
    // ...
    if len(result.ToolCalls) == 0 {
        break  // no more tools, done
    }

    // Append assistant message with tool calls
    fullMessages = append(fullMessages, ai.ChatMessage{
        Role:      "assistant",
        Content:   content,  // or nil if empty
        ToolCalls: result.ToolCalls,
    })

    // Process tools, get results
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

    // CRITICAL: Rebuild system prompt since world state changed
    ws = session.GetWorld()
    systemPrompt = buildSystemPrompt(ws)
    tools = buildChatTools(ws)
    fullMessages[0] = ai.ChatMessage{Role: "system", Content: systemPrompt}
}
```

Key behaviors:
- Up to 5 iterations of: stream → process tool calls → re-prompt
- The system prompt at `fullMessages[0]` is **replaced in-place** after each tool call because the world state has changed
- Tool messages are appended to `fullMessages` but never persisted to `session.ChatMessages`
- Only the final text content is saved as the assistant message

### Tool Result Strings

Each tool handler returns a plain-text string that becomes the `tool` message content:

- **moveToLocation** (chat.go:615-730): `"Moved to {destination}. Characters here: A, B. ACTION REQUIRED: Call discoverCharacter for each of these undiscovered characters BEFORE narrating: C, D."`
- **advanceTime** (chat.go:732-748): `"Time advanced by 5 ticks. It is now Late evening."`
- **discoverCharacter** (chat.go:750-789): `"Character Sera discovered."`

---

## 7. Auto-Discovery Post-Processing

**File:** `internal/handlers/chat.go:198-216`

After the tool loop completes, a post-processing step scans the final response text for character names:

```go
responseText := finalContent.String()
for _, c := range ws.Characters {
    if !c.IsPlayer && !c.IsDiscovered && strings.Contains(responseText, c.Name) {
        session.DiscoverCharacter(c.ID)
    }
}
```

This is a safety net — if the model mentions an undiscovered character by name without calling `discoverCharacter`, the system discovers them automatically. This doesn't affect the prompt sent, but does affect future prompts (the character will appear in CHARACTERS PRESENT).

---

## 8. API Call Construction

**File:** `internal/ai/openrouter.go`

### Endpoint

```go
const apiURL = "https://openrouter.ai/api/v1/chat/completions"  // line 22
```

All requests go to OpenRouter's OpenAI-compatible endpoint.

### Authentication

```go
func Init() {
    apiKey = os.Getenv("OPENROUTER_API_KEY")  // line 47
}
// ...
httpReq.Header.Set("Authorization", "Bearer "+apiKey)
```

### Streaming Request (main conversation)

**Function:** `StreamText` (line 242)

```go
req := ChatRequest{
    Model:     model,
    Messages:  messages,
    Tools:     tools,
    MaxTokens: 2048,
    Stream:    true,
}
```

- `MaxTokens: 2048` — hard limit per streaming call
- `Stream: true` — SSE streaming
- No `ToolChoice` set (defaults to `"auto"`)
- Accept header: `text/event-stream`

### Non-Streaming Request (off-screen simulation, location resolution)

**Function:** `GenerateText` (line 174)

```go
req := ChatRequest{
    Model:      model,
    Messages:   messages,
    Tools:      tools,
    ToolChoice: toolChoice,
    MaxTokens:  4096,
    Stream:     false,
}
```

- `MaxTokens: 4096` — higher limit for non-streaming
- `ToolChoice` can be `"required"` (for extraction tasks) or `nil`

### Request JSON Structure

```go
type ChatRequest struct {
    Model      string        `json:"model"`
    Messages   []ChatMessage `json:"messages"`
    Tools      []Tool        `json:"tools,omitempty"`
    ToolChoice any           `json:"tool_choice,omitempty"`
    MaxTokens  int           `json:"max_tokens,omitempty"`
    Stream     bool          `json:"stream"`
}
```

### Message Format

```go
type ChatMessage struct {
    Role       string     `json:"role"`
    Content    any        `json:"content"`      // string or null
    ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
    ToolCallID string     `json:"tool_call_id,omitempty"`
    Name       string     `json:"name,omitempty"`
}
```

This is the OpenAI chat completions format — no ChatML tokens, no Alpaca formatting, no special tokens. The API handles model-specific formatting.

### Retry Logic

**Function:** `doWithRetry` (line 56)

```go
backoff := []time.Duration{0, 1 * time.Second, 2 * time.Second, 4 * time.Second}
```

Retries on HTTP 5xx and 429 (rate limit) with exponential backoff (0s, 1s, 2s, 4s).

---

## 9. Model Selection

**File:** `internal/ai/openrouter.go:27-44`

```go
var Models = struct {
    MainConversation    string
    OffscreenSimulation string
    Fast                string
}{
    MainConversation:    "z-ai/glm-4.6:exacto",
    OffscreenSimulation: "z-ai/glm-4.6:exacto",
    Fast:                "z-ai/glm-4.6:exacto",
}

var AvailableModels = []string{
    "deepseek/deepseek-v3.1-terminus:exacto",
    "openai/gpt-oss-120b:exacto",
    "qwen/qwen3-coder:exacto",
    "moonshotai/kimi-k2-0905:exacto",
    "z-ai/glm-4.6:exacto",
}
```

The user can select a model via the UI (`POST /settings/model`), stored in `session.ModelID`. The effective model is resolved at `internal/handlers/handlers.go:320-325`:

```go
func effectiveModelID(id string) string {
    if id == "" {
        return ai.Models.MainConversation
    }
    return id
}
```

The same model ID is used for main conversation, off-screen simulation, and location resolution within a session. The model choice only affects the `model` field in the API request — no prompt format changes per model.

---

## 10. Off-Screen Simulation Prompts

**File:** `internal/world/simulation.go`

When the player moves to a new location and >5 ticks have elapsed since the last simulation, the system simulates what NPCs were doing elsewhere. This makes **two separate LLM calls** with their own prompts.

Simulation depth is determined by `determineSimulationDepth` (simulation.go:37-48):
- **< 5 ticks** → skip (no simulation)
- **5-10 ticks** → skip (threshold not reached)
- **11-20 ticks** → summary mode (brief 1-2 sentence summary)
- **> 20 ticks** or unresolved plot points → full mode (dialogue + event extraction)

Note: `hasUnresolvedPlotPoints` is always passed as `false` (simulation.go:407), so in practice only elapsed time determines depth.

### 10a. Summary Mode (11-20 ticks elapsed, no unresolved plot points)

**Function:** `generateSummary` (line 58)

A single user message (no system prompt):

```
Summarize what likely happened between {characterNames} over {timeElapsed} time units at {locationName}.

Characters:
- CharName: Description
  Goal: ...

Scenario: {scenarioDescription}

Write a brief 1-2 sentence summary of their interactions. Be specific but concise.
```

Uses `GenerateText` (non-streaming, 4096 max tokens, no tools).

### 10b. Full Simulation (>20 ticks or unresolved plot points)

**Function:** `runFullSimulation` (line 116)

**Step 1 — Generate dialogue** (lines 139-161):

Single user message acting as system-level instruction:

```
You are simulating a conversation between {characterNames} at {locationName}.

Characters:
- CharName: Description
  Goal: ...

Scenario: {scenarioDescription}
Time: {narrativeTime}
Available Locations (for movement): {locationList}

Write a natural dialogue between these characters. Each character should stay in character.
Format each line as: CHARACTER_NAME: "dialogue"
Include brief action descriptions in *asterisks* when appropriate.
If characters decide to go somewhere else, they should express it in dialogue.

Generate approximately {turnCount} exchanges.
```

Turn count: `max(min(timeElapsed/2, 8), 1)` (line 122).

Uses `GenerateText` with no tools.

**Step 2 — Extract events and movements** (lines 242-284):

A separate LLM call with the `reportSimulation` tool:

```
Analyze this conversation and extract significant events and any character movements:

{dialogueText}

List any important events (agreements made, information shared, conflicts).
If any character EXPLICITLY decides to leave for another location, report it in movements.
Matches must be from: {availableLocations}
```

Uses `GenerateText` with `toolChoice: "required"` to force the model to call `reportSimulation`.

---

## 11. Location Resolution Prompts

**File:** `internal/world/locations.go:22-150`
**Function:** `ResolveLocation`

When the LLM calls `moveToLocation`, the destination string must be matched to an existing location cluster or identified as new. This uses an LLM call:

```
Given this location description: "{description}"

And these existing locations:
1. "The Dusty Tankard" (id: abc123)
2. "The Crossroads" (id: def456)
...

Determine if the description refers to one of the existing locations or is a new location.
Consider semantic similarity - "the cafe" matches "Coffee Shop", "town center" matches "Town Square", etc.

Call the resolveLocation tool with:
- matchedClusterId: the id of the matching location, or null if it's a new place
- canonicalName: the best canonical name for this location
- confidence: how confident you are in the match (0.0-1.0)
```

Uses `GenerateText` with `toolChoice: "required"` and the `resolveLocation` tool. Confidence threshold: >=0.6 to accept a match (line 125).

Fallback if LLM fails: `ExtractCanonicalName` (line 156-172) strips articles/prepositions and title-cases to 4 words max.

---

## 12. Streaming Response Format

**File:** `internal/ai/openrouter.go:242-359`

The streaming response is parsed as SSE events from the OpenRouter API:

```go
scanner := bufio.NewScanner(resp.Body)
for scanner.Scan() {
    line := scanner.Text()
    if !strings.HasPrefix(line, "data: ") { continue }
    data := strings.TrimPrefix(line, "data: ")
    if data == "[DONE]" { break }
    // Parse JSON, extract content delta and tool call fragments
}
```

Tool calls are accumulated across multiple SSE chunks using `ToolCallAccumulator` (line 228-233) which reassembles fragmented function arguments.

The server re-emits content deltas as its own SSE events to the browser (chat.go:142-143):
```go
result, err := ai.StreamText(ctx, model, fullMessages, tools, func(content string) {
    writeSSE(w, flusher, "token", content)
})
```

---

## 13. What's NOT in the System

Features common in roleplay/chat platforms that are **absent** here:

- **User persona/profile** — no user self-description is injected; the player character has a `Description` but it's not included in the system prompt
- **Author's note / editorial instructions** — no mid-conversation injection points
- **Few-shot examples** — the system prompt has behavioral examples but no in-character example exchanges
- **World info / lorebook with keyword triggers** — character knowledge exists but is injected per-character, not triggered by keyword matching
- **Prompt formatting templates** (ChatML, Alpaca, etc.) — the OpenAI message format is used directly; model-specific formatting is handled by OpenRouter
- **Token counting library** — uses a 4-chars-per-token heuristic
- **Chat summarization / compression** — old messages are silently dropped, never summarized
- **Regex/post-processing on model output** — responses are stored and displayed verbatim (rendered as Markdown in the browser)
- **Stop sequences** — none set; the model decides when to stop
- **Temperature / top_p / other sampling params** — not set; uses API defaults
- **Multiple prompt formats for different models** — all models receive the same message structure

---

## 14. Summary: Final Prompt Layout

The exact structure sent to OpenRouter for a main conversation turn:

```
┌─────────────────────────────────────────────────────┐
│ Message 0: role="system"                            │
│                                                     │
│   You are the narrator and game master of an        │
│   interactive narrative experience called "{Title}". │
│                                                     │
│   SCENARIO: {description}                           │
│                                                     │
│   CURRENT LOCATION: {location}                      │
│   OTHER KNOWN LOCATIONS: {list}                     │
│   TIME: {narrativeTime} (tick {N})                  │
│                                                     │
│   CHARACTERS PRESENT (SYSTEM STATE):                │
│   - CharA: description                              │
│       Knows: k1; k2; k3                             │
│   - CharB: description                              │
│   (NOTE: If a character is participating...)        │
│   HIDDEN (...): CharC, CharD                        │
│                                                     │
│   RECENT EVENTS:                                    │
│   - event1                                          │
│   - event2                                          │
│                                                     │
│   YOUR ROLE:                                        │
│   [behavioral instructions]                         │
│                                                     │
│   Tools:                                            │
│   [tool usage instructions]                         │
│                                                     │
│   CRITICAL RULE — DISCOVERING CHARACTERS:           │
│   [character discovery rules]                       │
│                                                     │
│   IMPORTANT:                                        │
│   [narrator constraints]                            │
│                                                     │
│   EXAMPLES:                                         │
│   [3 tool usage examples]                           │
├─────────────────────────────────────────────────────┤
│ Message 1: role="user"                              │
│   {oldest retained user message}                    │
├─────────────────────────────────────────────────────┤
│ Message 2: role="assistant"                         │
│   {assistant response}                              │
├─────────────────────────────────────────────────────┤
│ ...alternating user/assistant messages...           │
├─────────────────────────────────────────────────────┤
│ Message N: role="user"                              │
│   {latest user message}                             │
├─────────────────────────────────────────────────────┤
│                                                     │
│ tools: [moveToLocation, advanceTime,                │
│         discoverCharacter]                          │
│                                                     │
│ model: "z-ai/glm-4.6:exacto" (or user-selected)    │
│ max_tokens: 2048                                    │
│ stream: true                                        │
└─────────────────────────────────────────────────────┘
```

If the model responds with tool calls, the array extends:

```
├─────────────────────────────────────────────────────┤
│ Message N+1: role="assistant"                       │
│   content: null | partial text                      │
│   tool_calls: [{moveToLocation, args...}]           │
├─────────────────────────────────────────────────────┤
│ Message N+2: role="tool"                            │
│   tool_call_id: "call_abc"                          │
│   name: "moveToLocation"                            │
│   content: "Moved to Market Square. Characters..."  │
├─────────────────────────────────────────────────────┤
│ [system prompt at index 0 is REPLACED with          │
│  rebuilt version reflecting new world state]        │
├─────────────────────────────────────────────────────┤
│ → Next streaming call with extended array           │
│   (up to 5 iterations)                              │
└─────────────────────────────────────────────────────┘
```
