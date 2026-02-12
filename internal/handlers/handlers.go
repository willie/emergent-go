package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"io/fs"
	"log/slog"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"emergent/internal/ai"
	"emergent/internal/models"
	"emergent/internal/storage"
	"emergent/internal/world"

	"github.com/google/uuid"
	"github.com/yuin/goldmark"
	goldmarkhtml "github.com/yuin/goldmark/renderer/html"
)

// unsafeHrefRe matches href/src attributes with dangerous URL schemes in goldmark output.
var unsafeHrefRe = regexp.MustCompile(`(?i)(href|src)="(?:javascript|vbscript|data):[^"]*"`)

const sessionCookieName = "emergent_session"

// App holds the application state and dependencies
type App struct {
	PageTemplates  map[string]*template.Template
	FuncMap        template.FuncMap
	templateFS     fs.FS
	sessions       sync.Map // map[string]*world.SessionState
	sessionMutexes sync.Map // map[string]*sync.Mutex
}

// NewApp creates a new app with templates compiled at startup
func NewApp(templateFS fs.FS) (*App, error) {
	md := goldmark.New(
		goldmark.WithRendererOptions(goldmarkhtml.WithHardWraps()),
	)

	funcMap := template.FuncMap{
		"sub": func(a, b int) int { return a - b },
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
			return template.HTML(unsafeHrefRe.ReplaceAllString(buf.String(), `$1="#"`))
		},
	}

	// Pre-compile all page templates at startup
	pages := map[string]*template.Template{}
	for _, page := range []string{"scenario_selector.html", "game.html"} {
		tmpl, err := compilePageTemplate(templateFS, funcMap, page)
		if err != nil {
			return nil, fmt.Errorf("compile %s: %w", page, err)
		}
		pages[page] = tmpl
	}

	return &App{
		PageTemplates: pages,
		FuncMap:       funcMap,
		templateFS:    templateFS,
	}, nil
}

// getSession retrieves the session for this request, creating one if needed.
// Must be called from a handler wrapped with WithSessionLock.
func (a *App) getSession(r *http.Request) *world.SessionState {
	sid := r.Context().Value(sessionIDKey).(string)
	if val, ok := a.sessions.Load(sid); ok {
		session := val.(*world.SessionState)
		session.Touch()
		return session
	}
	session := world.NewSessionState()
	a.sessions.Store(sid, session)
	return session
}

func (a *App) getSessionMutex(sessionID string) *sync.Mutex {
	v, _ := a.sessionMutexes.LoadOrStore(sessionID, &sync.Mutex{})
	return v.(*sync.Mutex)
}

func (a *App) getSessionID(r *http.Request) string {
	if cookie, err := r.Cookie(sessionCookieName); err == nil {
		return cookie.Value
	}
	return ""
}

type contextKey string

const sessionIDKey contextKey = "sessionID"

// WithSessionLock returns middleware that acquires the session mutex for the
// duration of the request, creating a session ID and cookie if none exists.
func (a *App) WithSessionLock(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sid := a.getSessionID(r)
		if sid == "" {
			sid = uuid.New().String()
			http.SetCookie(w, &http.Cookie{
				Name:     sessionCookieName,
				Value:    sid,
				Path:     "/",
				MaxAge:   30 * 24 * 60 * 60,
				HttpOnly: true,
				Secure:   r.TLS != nil,
				SameSite: http.SameSiteLaxMode,
			})
		}
		mu := a.getSessionMutex(sid)
		mu.Lock()
		defer mu.Unlock()
		ctx := context.WithValue(r.Context(), sessionIDKey, sid)
		next(w, r.WithContext(ctx))
	}
}

// compilePageTemplate parses layout + a specific page template + partials into one set
func compilePageTemplate(templateFS fs.FS, funcMap template.FuncMap, page string) (*template.Template, error) {
	tmpl, err := template.New("").Funcs(funcMap).ParseFS(templateFS, "templates/layout.html", "templates/"+page)
	if err != nil {
		return nil, err
	}
	partials, _ := fs.Glob(templateFS, "templates/partials/*.html")
	if len(partials) > 0 {
		if _, err := tmpl.ParseFS(templateFS, "templates/partials/*.html"); err != nil {
			slog.Warn("failed to parse partials", "error", err)
		}
	}
	return tmpl, nil
}

// Index serves the main page
func (a *App) Index(w http.ResponseWriter, r *http.Request) {
	session := a.getSession(r)
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
	if err := storage.GetJSON("custom_scenarios", &customScenarios); err != nil {
		slog.Error("failed to load custom scenarios", "error", err)
	}

	data := map[string]any{
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
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Bad request", 400)
		return
	}
	idxStr := r.FormValue("scenario_index")
	idx, err := strconv.Atoi(idxStr)
	if err != nil || idx < 0 || idx >= len(world.BuiltinScenarios) {
		http.Error(w, "Invalid scenario", 400)
		return
	}

	session := a.getSession(r)
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
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Bad request", 400)
		return
	}
	idxStr := r.FormValue("scenario_index")
	idx, err := strconv.Atoi(idxStr)
	if err != nil {
		http.Error(w, "Invalid scenario", 400)
		return
	}

	var customScenarios []models.ScenarioConfig
	if err := storage.GetJSON("custom_scenarios", &customScenarios); err != nil {
		slog.Error("failed to load custom scenarios", "error", err)
		http.Error(w, "Failed to load scenarios", http.StatusInternalServerError)
		return
	}

	if idx < 0 || idx >= len(customScenarios) {
		http.Error(w, "Invalid scenario", 400)
		return
	}

	session := a.getSession(r)
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
	saveID := r.FormValue("save")
	if saveID == "" {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	if !validStorageKey(saveID) {
		http.Error(w, "Invalid save ID", http.StatusBadRequest)
		return
	}

	session := a.getSession(r)
	if err := session.Load(saveID); err != nil {
		slog.Error("failed to load save", "save", saveID, "error", err)
		http.Error(w, "Failed to load save", http.StatusInternalServerError)
		return
	}

	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// ExitGame exits to the main menu
func (a *App) ExitGame(w http.ResponseWriter, r *http.Request) {
	session := a.getSession(r)
	_ = session.Persist()
	session.ResetWorld()
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// ImportScenario handles scenario JSON import
func (a *App) ImportScenario(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(10 << 20); err != nil {
		http.Error(w, "Bad request", 400)
		return
	}
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
	if err := storage.GetJSON("custom_scenarios", &customScenarios); err != nil {
		slog.Info("no existing custom scenarios, starting fresh", "error", err)
	}
	customScenarios = append(customScenarios, scenario)
	if err := storage.SetJSON("custom_scenarios", customScenarios); err != nil {
		slog.Error("failed to save custom scenarios", "error", err)
		http.Error(w, "Failed to save scenario", 500)
		return
	}

	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// SetModel updates the AI model
func (a *App) SetModel(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Bad request", 400)
		return
	}
	session := a.getSession(r)
	model := r.FormValue("model")
	if model != "" {
		if !slices.Contains(ai.AvailableModels, model) {
			http.Error(w, "Invalid model", http.StatusBadRequest)
			return
		}
		session.SetModelID(model)
	}
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

// PartialSaves returns the saves list as HTML partial
func (a *App) PartialSaves(w http.ResponseWriter, r *http.Request) {
	session := a.getSession(r)
	saves, _ := storage.List()
	w.Header().Set("Content-Type", "text/html")

	activeSaveKey := session.GetActiveSaveKey()
	for _, s := range saves {
		if !strings.HasPrefix(s.ID, "surat-world-storage") {
			continue
		}
		name := "Default"
		if s.ID != "surat-world-storage" {
			name = strings.ReplaceAll(strings.TrimPrefix(s.ID, "surat-world-storage-"), "-", " ")
		}
		data := map[string]any{
			"ID":          s.ID,
			"DisplayName": name,
			"UpdatedAt":   s.UpdatedAt.Format("Jan 2, 2006 3:04 PM"),
			"IsActive":    s.ID == activeSaveKey,
		}
		if err := a.PageTemplates["game.html"].ExecuteTemplate(w, "save_item", data); err != nil {
			slog.Error("render save_item failed", "error", err)
		}
	}
}

// StartEviction starts a background goroutine that evicts idle sessions
func (a *App) StartEviction(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				a.evictSessions()
			}
		}
	}()
}

func (a *App) evictSessions() {
	now := time.Now()
	a.sessions.Range(func(key, value any) bool {
		session := value.(*world.SessionState)
		if now.Sub(session.GetLastAccessed()) > 1*time.Hour {
			mu := a.getSessionMutex(key.(string))
			if !mu.TryLock() {
				return true // in use, skip
			}
			if err := session.Persist(); err != nil {
				slog.Error("failed to persist session before eviction", "session", key, "error", err)
			}
			a.sessions.Delete(key)
			a.sessionMutexes.Delete(key)
			mu.Unlock()
			slog.Info("evicted idle session", "session", key)
		}
		return true
	})
}
