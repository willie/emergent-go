package world

import "emergent/internal/models"

// BuiltinScenarios contains the default scenarios
var BuiltinScenarios = []models.ScenarioConfig{
	{
		Title:                "The Dusty Tankard",
		Description:          "A medieval fantasy tavern at a crossroads. Rumors of a dragon sighting have brought travelers from all directions. The barkeep knows more than he lets on, and the hooded stranger in the corner has been watching you since you walked in.",
		InitialNarrativeTime: "Late afternoon",
		Locations: []models.InitialLocation{
			{Name: "The Dusty Tankard", Description: "A weathered tavern at the crossroads, filled with the smell of ale and woodsmoke."},
			{Name: "The Crossroads", Description: "A well-traveled intersection of two major roads, with a signpost pointing in four directions."},
			{Name: "The Forest Path", Description: "A narrow trail leading into dark, ancient woods."},
			{Name: "The Market Square", Description: "A bustling open-air market in the nearby village."},
		},
		Characters: []models.CharacterConfig{
			{Name: "You", Description: "A weary traveler seeking shelter and information.", IsPlayer: true, InitialLocationName: "The Dusty Tankard", EncounterChance: 1.0},
			{Name: "Grim", Description: "The grizzled barkeep of the Dusty Tankard. Knows every rumor that passes through.", IsPlayer: false, InitialLocationName: "The Dusty Tankard", EncounterChance: 1.0, Goals: "Keep the peace and profit from the increased traffic"},
			{Name: "Sera", Description: "A hooded stranger who watches the room with sharp eyes. Carries a worn leather journal.", IsPlayer: false, InitialLocationName: "The Dusty Tankard", EncounterChance: 0.3, Goals: "Investigate the dragon sighting without being noticed"},
			{Name: "Bran", Description: "A loud, boastful merchant who claims to have seen the dragon himself.", IsPlayer: false, InitialLocationName: "The Market Square", EncounterChance: 0.7, Goals: "Sell his 'dragon-proof' wares at inflated prices"},
			{Name: "Elda", Description: "An elderly herbalist gathering rare plants. Wise and soft-spoken.", IsPlayer: false, InitialLocationName: "The Forest Path", EncounterChance: 0.5, Goals: "Find the moonpetal flower before the frost comes"},
		},
		PlayerStartingLocation: "The Dusty Tankard",
	},
	{
		Title:                "Neon Shadows",
		Description:          "A cyberpunk noir detective story. You're a private investigator in Neo-Tokyo, 2087. A corporate exec has gone missing and their spouse has hired you to find them. The neon-lit streets hold secrets that powerful people want buried.",
		InitialNarrativeTime: "Late evening",
		Locations: []models.InitialLocation{
			{Name: "Your Office", Description: "A cramped office above a ramen shop in the lower district. Rain streaks the window."},
			{Name: "Kyoko Tower", Description: "The gleaming corporate headquarters of Kyoko Industries, 200 stories of glass and steel."},
			{Name: "The Undercity", Description: "A labyrinth of tunnels and illegal markets beneath the main streets."},
			{Name: "Club Zero", Description: "An exclusive nightclub frequented by corporate elites and information brokers."},
		},
		Characters: []models.CharacterConfig{
			{Name: "You", Description: "A jaded private investigator with a reputation for finding the unfindable.", IsPlayer: true, InitialLocationName: "Your Office", EncounterChance: 1.0},
			{Name: "Mika", Description: "The worried spouse of the missing exec. Seems genuine but is hiding something.", IsPlayer: false, InitialLocationName: "Your Office", EncounterChance: 1.0, Goals: "Find their missing spouse, protect a secret"},
			{Name: "Ghost", Description: "A legendary hacker who operates from the Undercity. Sells information to the highest bidder.", IsPlayer: false, InitialLocationName: "The Undercity", EncounterChance: 0.4, Goals: "Profit from the chaos of the missing exec situation"},
			{Name: "Director Tanaka", Description: "Head of security at Kyoko Industries. Cold, efficient, and politically connected.", IsPlayer: false, InitialLocationName: "Kyoko Tower", EncounterChance: 0.6, Goals: "Cover up what really happened to the missing exec"},
			{Name: "Zara", Description: "A bartender at Club Zero who hears everything. Has connections on both sides of the law.", IsPlayer: false, InitialLocationName: "Club Zero", EncounterChance: 0.8, Goals: "Stay neutral and alive in a dangerous game"},
		},
		PlayerStartingLocation: "Your Office",
	},
}
