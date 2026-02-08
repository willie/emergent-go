package handlers

import (
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"strconv"
	"strings"

	"emergent/internal/ai"
	"emergent/internal/models"
	"emergent/internal/storage"
	"emergent/internal/world"

	"github.com/google/uuid"
)

// App holds the application state and dependencies
type App struct {
	FuncMap  template.FuncMap
	Session  *world.SessionState
}

// NewApp creates a new app with templates loaded
func NewApp() (*App, error) {
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
	}

	session := world.NewSessionState()
	// Try to load default save
	_ = session.Load("surat-world-storage")

	return &App{
		FuncMap: funcMap,
		Session: session,
	}, nil
}

// parsePageTemplate parses the layout + a specific page template (+ partials)
func (a *App) parsePageTemplate(page string) (*template.Template, error) {
	files := []string{"templates/layout.html", "templates/" + page}
	tmpl, err := template.New("").Funcs(a.FuncMap).ParseFiles(files...)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", page, err)
	}
	// Also parse partials if they exist
	partialTmpl, err := template.New("").Funcs(a.FuncMap).ParseGlob("templates/partials/*.html")
	if err == nil {
		for _, t := range partialTmpl.Templates() {
			if _, err := tmpl.AddParseTree(t.Name(), t.Tree); err != nil {
				log.Printf("Warning: failed to add partial %s: %v", t.Name(), err)
			}
		}
	}
	return tmpl, nil
}

// Index serves the main page - either scenario selector or game
func (a *App) Index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	if a.Session.World == nil {
		a.renderScenarioSelector(w, r)
		return
	}

	a.renderGame(w, r)
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

	tmpl, err := a.parsePageTemplate("scenario_selector.html")
	if err != nil {
		http.Error(w, "template error", 500)
		log.Printf("Template error: %v", err)
		return
	}

	w.Header().Set("Content-Type", "text/html")
	if err := tmpl.ExecuteTemplate(w, "layout.html", data); err != nil {
		log.Printf("Render error: %v", err)
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
	ID               string
	ParticipantNames []string
	ParticipantIDs   []string
	LocationName     string
	LocationClusterID string
	Messages         []models.Message
}

func (a *App) renderGame(w http.ResponseWriter, r *http.Request) {
	session := a.Session

	var offscreenConvs []offscreenConvDisplay
	for _, conv := range session.GetOffscreenConversations() {
		var names []string
		for _, pid := range conv.ParticipantIDs {
			c := session.GetCharacterByID(pid)
			if c != nil {
				names = append(names, c.Name)
			}
		}
		locName := "Unknown"
		loc := session.GetLocationCluster(conv.LocationClusterID)
		if loc != nil {
			locName = loc.CanonicalName
		}
		offscreenConvs = append(offscreenConvs, offscreenConvDisplay{
			ID:                conv.ID,
			ParticipantNames:  names,
			ParticipantIDs:    conv.ParticipantIDs,
			LocationName:      locName,
			LocationClusterID: conv.LocationClusterID,
			Messages:          conv.Messages,
		})
	}

	discovered := session.GetDiscoveredCharacters()

	data := gameData{
		World:                  session.World,
		Location:               session.GetPlayerLocation(),
		NearbyCharacters:       session.GetCharactersAtPlayerLocation(),
		ChatMessages:           session.ChatMessages,
		OffscreenConversations: offscreenConvs,
		DiscoveredCharacters:   discovered,
		AvailableModels:        ai.AvailableModels,
		ModelID:                session.ModelID,
		IsSimulating:           session.IsSimulating,
	}

	tmpl, err := a.parsePageTemplate("game.html")
	if err != nil {
		http.Error(w, "template error", 500)
		log.Printf("Template error: %v", err)
		return
	}

	w.Header().Set("Content-Type", "text/html")
	if err := tmpl.ExecuteTemplate(w, "layout.html", data); err != nil {
		log.Printf("Render error: %v", err)
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

	scenario := world.BuiltinScenarios[idx]
	saveKey := fmt.Sprintf("surat-world-storage-game-%d", uuid.New().ID())
	a.Session.ActiveSaveKey = saveKey

	if err := a.Session.InitializeScenario(scenario); err != nil {
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

	scenario := customScenarios[idx]
	saveKey := fmt.Sprintf("surat-world-storage-game-%d", uuid.New().ID())
	a.Session.ActiveSaveKey = saveKey

	if err := a.Session.InitializeScenario(scenario); err != nil {
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

	if err := a.Session.Load(saveID); err != nil {
		log.Printf("Failed to load save %s: %v", saveID, err)
	}

	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// ExitGame exits to the main menu
func (a *App) ExitGame(w http.ResponseWriter, r *http.Request) {
	_ = a.Session.Persist()
	a.Session.ResetWorld()
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// ImportScenario handles scenario JSON import
func (a *App) ImportScenario(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	r.ParseMultipartForm(10 << 20) // 10MB
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
	model := r.FormValue("model")
	if model != "" {
		a.Session.ModelID = model
	}
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// PartialSaves returns the saves list as HTML partial
func (a *App) PartialSaves(w http.ResponseWriter, r *http.Request) {
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
		isActive := s.ID == a.Session.ActiveSaveKey

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
			s.ID, activeClass, nameClass, name, currentLabel,
			s.UpdatedAt.Format("Jan 2, 2006 3:04 PM"))
	}
}
