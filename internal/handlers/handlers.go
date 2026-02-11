package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html"
	"html/template"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"emergent/internal/ai"
	"emergent/internal/models"
	"emergent/internal/storage"
	"emergent/internal/world"

	"github.com/google/uuid"
	"github.com/yuin/goldmark"
	goldmarkhtml "github.com/yuin/goldmark/renderer/html"
)

const sessionCookieName = "emergent_session"

// App holds the application state and dependencies
type App struct {
	PageTemplates map[string]*template.Template
	FuncMap       template.FuncMap
	sessions      sync.Map // map[string]*world.SessionState
}

// NewApp creates a new app with templates compiled at startup
func NewApp() (*App, error) {
	md := goldmark.New(
		goldmark.WithRendererOptions(goldmarkhtml.WithHardWraps()),
	)

	funcMap := template.FuncMap{
		"lastN": func(items []models.KnowledgeEntry, n int) []models.KnowledgeEntry {
			if len(items) <= n {
				return items
			}
			return items[len(items)-n:]
		},
		"locationName": func(clusterID string, clusters []models.LocationCluster) string {
			for _, c := range clusters {
				if c.ID == clusterID {
					return c.CanonicalName
				}
			}
			return "Unknown"
		},
		"speakerName": func(id string, names map[string]string) string {
			if name, ok := names[id]; ok {
				return name
			}
			return id
		},
		"renderMarkdown": func(s string) template.HTML {
			var buf bytes.Buffer
			if err := md.Convert([]byte(s), &buf); err != nil {
				return template.HTML(template.HTMLEscapeString(s))
			}
			return template.HTML(buf.String())
		},
	}

	// Pre-compile all page templates at startup
	pages := map[string]*template.Template{}
	for _, page := range []string{"scenario_selector.html", "game.html"} {
		tmpl, err := compilePageTemplate(funcMap, page)
		if err != nil {
			return nil, fmt.Errorf("compile %s: %w", page, err)
		}
		pages[page] = tmpl
	}

	return &App{
		PageTemplates: pages,
		FuncMap:       funcMap,
	}, nil
}

// getSession retrieves the session for this request, creating one if needed.
// It reads the session ID from a cookie, looks it up in the map, and creates
// a fresh session (with cookie) if none exists.
func (a *App) getSession(w http.ResponseWriter, r *http.Request) *world.SessionState {
	if cookie, err := r.Cookie(sessionCookieName); err == nil {
		if val, ok := a.sessions.Load(cookie.Value); ok {
			return val.(*world.SessionState)
		}
	}

	// Create new session
	id := uuid.New().String()
	session := world.NewSessionState()

	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookieName,
		Value:    id,
		Path:     "/",
		MaxAge:   30 * 24 * 60 * 60, // 30 days
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
	})

	a.sessions.Store(id, session)
	return session
}

// compilePageTemplate parses layout + a specific page template + partials into one set
func compilePageTemplate(funcMap template.FuncMap, page string) (*template.Template, error) {
	files := []string{"templates/layout.html", "templates/" + page}
	tmpl, err := template.New("").Funcs(funcMap).ParseFiles(files...)
	if err != nil {
		return nil, err
	}
	partials, err := template.New("").Funcs(funcMap).ParseGlob("templates/partials/*.html")
	if err == nil {
		for _, t := range partials.Templates() {
			if _, addErr := tmpl.AddParseTree(t.Name(), t.Tree); addErr != nil {
				slog.Warn("failed to add partial", "template", t.Name(), "error", addErr)
			}
		}
	}
	return tmpl, nil
}

// Index serves the main page
func (a *App) Index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	session := a.getSession(w, r)
	if session.GetWorld() == nil {
		a.renderScenarioSelector(w, r)
		return
	}
	a.renderGame(w, r, session)
}

func (a *App) renderScenarioSelector(w http.ResponseWriter, r *http.Request) {
	saves, _ := storage.List()
	type saveDisplay struct {
		ID          string
		DisplayName string
		UpdatedAt   string
	}
	var displaySaves []saveDisplay
	for _, s := range saves {
		if !strings.HasPrefix(s.ID, "surat-world-storage") {
			continue
		}
		name := "Default"
		if s.ID != "surat-world-storage" {
			name = strings.ReplaceAll(strings.TrimPrefix(s.ID, "surat-world-storage-"), "-", " ")
		}
		displaySaves = append(displaySaves, saveDisplay{
			ID:          s.ID,
			DisplayName: name,
			UpdatedAt:   s.UpdatedAt.Format("Jan 2, 2006 3:04 PM"),
		})
	}

	var customScenarios []models.ScenarioConfig
	_ = storage.GetJSON("custom_scenarios", &customScenarios)

	data := map[string]interface{}{
		"Scenarios":       world.BuiltinScenarios,
		"CustomScenarios": customScenarios,
		"Saves":           displaySaves,
	}

	w.Header().Set("Content-Type", "text/html")
	if err := a.PageTemplates["scenario_selector.html"].ExecuteTemplate(w, "layout.html", data); err != nil {
		slog.Error("render failed", "template", "scenario_selector", "error", err)
	}
}

type gameData struct {
	World                  *models.WorldState
	Location               *models.LocationCluster
	NearbyCharacters       []models.Character
	ChatMessages           []models.ChatMessage
	OffscreenConversations []offscreenConvDisplay
	DiscoveredCharacters   []models.Character
	AvailableModels        []string
	ModelID                string
	IsSimulating           bool
}

type offscreenConvDisplay struct {
	ID                string
	ParticipantNames  []string
	ParticipantIDs    []string
	LocationName      string
	LocationClusterID string
	Messages          []models.Message
	SpeakerNames      map[string]string
}

// buildGameData constructs the gameData struct from session state
func (a *App) buildGameData(session *world.SessionState) gameData {
	var offscreenConvs []offscreenConvDisplay
	for _, conv := range session.GetOffscreenConversations() {
		var names []string
		speakerNames := make(map[string]string)
		for _, pid := range conv.ParticipantIDs {
			c, ok := session.GetCharacterByID(pid)
			if ok {
				names = append(names, c.Name)
				speakerNames[pid] = c.Name
			}
		}
		// Also resolve speaker IDs from messages that may not be participants
		for _, msg := range conv.Messages {
			if msg.SpeakerID != "" {
				if _, exists := speakerNames[msg.SpeakerID]; !exists {
					c, ok := session.GetCharacterByID(msg.SpeakerID)
					if ok {
						speakerNames[msg.SpeakerID] = c.Name
					}
				}
			}
		}
		locName := "Unknown"
		loc, ok := session.GetLocationCluster(conv.LocationClusterID)
		if ok {
			locName = loc.CanonicalName
		}
		offscreenConvs = append(offscreenConvs, offscreenConvDisplay{
			ID:                conv.ID,
			ParticipantNames:  names,
			ParticipantIDs:    conv.ParticipantIDs,
			LocationName:      locName,
			LocationClusterID: conv.LocationClusterID,
			Messages:          conv.Messages,
			SpeakerNames:      speakerNames,
		})
	}

	var playerLoc *models.LocationCluster
	if loc, ok := session.GetPlayerLocation(); ok {
		playerLoc = &loc
	}

	return gameData{
		World:                  session.GetWorld(),
		Location:               playerLoc,
		NearbyCharacters:       session.GetCharactersAtPlayerLocation(),
		ChatMessages:           session.GetChatMessagesCopy(),
		OffscreenConversations: offscreenConvs,
		DiscoveredCharacters:   session.GetDiscoveredCharacters(),
		AvailableModels:        ai.AvailableModels,
		ModelID:                session.GetModelID(),
		IsSimulating:           session.GetIsSimulating(),
	}
}

func (a *App) renderGame(w http.ResponseWriter, r *http.Request, session *world.SessionState) {
	data := a.buildGameData(session)

	w.Header().Set("Content-Type", "text/html")
	if err := a.PageTemplates["game.html"].ExecuteTemplate(w, "layout.html", data); err != nil {
		slog.Error("render failed", "template", "game", "error", err)
	}
}

// NewGame starts a new game with a built-in scenario
func (a *App) NewGame(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}
	r.ParseForm()
	idxStr := r.FormValue("scenario_index")
	idx, err := strconv.Atoi(idxStr)
	if err != nil || idx < 0 || idx >= len(world.BuiltinScenarios) {
		http.Error(w, "Invalid scenario", 400)
		return
	}

	session := a.getSession(w, r)
	scenario := world.BuiltinScenarios[idx]
	saveKey := fmt.Sprintf("surat-world-storage-game-%d", uuid.New().ID())
	session.SetActiveSaveKey(saveKey)

	if err := session.InitializeScenario(scenario); err != nil {
		http.Error(w, "Failed to initialize: "+err.Error(), 500)
		return
	}

	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// NewCustomGame starts a game with a custom scenario
func (a *App) NewCustomGame(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}
	r.ParseForm()
	idxStr := r.FormValue("scenario_index")
	idx, err := strconv.Atoi(idxStr)
	if err != nil {
		http.Error(w, "Invalid scenario", 400)
		return
	}

	var customScenarios []models.ScenarioConfig
	_ = storage.GetJSON("custom_scenarios", &customScenarios)

	if idx < 0 || idx >= len(customScenarios) {
		http.Error(w, "Invalid scenario", 400)
		return
	}

	session := a.getSession(w, r)
	scenario := customScenarios[idx]
	saveKey := fmt.Sprintf("surat-world-storage-game-%d", uuid.New().ID())
	session.SetActiveSaveKey(saveKey)

	if err := session.InitializeScenario(scenario); err != nil {
		http.Error(w, "Failed to initialize: "+err.Error(), 500)
		return
	}

	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// LoadGame loads an existing save
func (a *App) LoadGame(w http.ResponseWriter, r *http.Request) {
	saveID := r.URL.Query().Get("save")
	if saveID == "" {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	session := a.getSession(w, r)
	if err := session.Load(saveID); err != nil {
		slog.Error("failed to load save", "save", saveID, "error", err)
	}

	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// ExitGame exits to the main menu
func (a *App) ExitGame(w http.ResponseWriter, r *http.Request) {
	session := a.getSession(w, r)
	_ = session.Persist()
	session.ResetWorld()
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// ImportScenario handles scenario JSON import
func (a *App) ImportScenario(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	r.ParseMultipartForm(10 << 20)
	file, _, err := r.FormFile("scenario")
	if err != nil {
		http.Error(w, "Failed to read file", 400)
		return
	}
	defer file.Close()

	var scenario models.ScenarioConfig
	if err := json.NewDecoder(file).Decode(&scenario); err != nil {
		http.Error(w, "Invalid JSON", 400)
		return
	}

	if err := scenario.Validate(); err != nil {
		http.Error(w, "Invalid scenario: "+err.Error(), 400)
		return
	}

	var customScenarios []models.ScenarioConfig
	_ = storage.GetJSON("custom_scenarios", &customScenarios)
	customScenarios = append(customScenarios, scenario)
	_ = storage.SetJSON("custom_scenarios", customScenarios)

	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// SetModel updates the AI model
func (a *App) SetModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}
	r.ParseForm()
	session := a.getSession(w, r)
	model := r.FormValue("model")
	if model != "" {
		session.SetModelID(model)
	}
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// PartialSaves returns the saves list as HTML partial
func (a *App) PartialSaves(w http.ResponseWriter, r *http.Request) {
	session := a.getSession(w, r)
	saves, _ := storage.List()
	w.Header().Set("Content-Type", "text/html")

	for _, s := range saves {
		if !strings.HasPrefix(s.ID, "surat-world-storage") {
			continue
		}
		name := "Default"
		if s.ID != "surat-world-storage" {
			name = strings.ReplaceAll(strings.TrimPrefix(s.ID, "surat-world-storage-"), "-", " ")
		}
		isActive := s.ID == session.GetActiveSaveKey()

		activeClass := "bg-zinc-800/50 border-zinc-700 hover:bg-zinc-800"
		nameClass := "text-zinc-200"
		if isActive {
			activeClass = "bg-blue-900/20 border-blue-500/50"
			nameClass = "text-blue-400"
		}

		currentLabel := ""
		if isActive {
			currentLabel = ` <span class="ml-2 text-xs text-blue-500/80">(Current)</span>`
		}

		fmt.Fprintf(w, `<a href="/game/load?save=%s" class="group flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all %s">
			<div class="min-w-0">
				<p class="text-sm font-medium truncate %s">%s%s</p>
				<p class="text-xs text-zinc-500">%s</p>
			</div>
		</a>`,
			html.EscapeString(s.ID), activeClass, nameClass, html.EscapeString(name), currentLabel,
			s.UpdatedAt.Format("Jan 2, 2006 3:04 PM"))
	}
}
