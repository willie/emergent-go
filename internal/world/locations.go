package world

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"emergent/internal/ai"
)

// ResolveLocationResult from resolving a location description
type ResolveLocationResult struct {
	ClusterID     *string `json:"clusterId"`
	CanonicalName string  `json:"canonicalName"`
	IsNew         bool    `json:"isNew"`
}

// ResolveLocation resolves a location description to an existing cluster or new one
func ResolveLocation(description string, existingClusters []struct {
	ID            string `json:"id"`
	CanonicalName string `json:"canonicalName"`
}, modelID string) (*ResolveLocationResult, error) {
	if len(existingClusters) == 0 {
		name := ExtractCanonicalName(description)
		return &ResolveLocationResult{
			ClusterID:     nil,
			CanonicalName: name,
			IsNew:         true,
		}, nil
	}

	var clusterList strings.Builder
	for i, c := range existingClusters {
		fmt.Fprintf(&clusterList, "%d. \"%s\" (id: %s)\n", i+1, c.CanonicalName, c.ID)
	}

	tools := []ai.Tool{
		{
			Type: "function",
			Function: ai.ToolFunction{
				Name:        "resolveLocation",
				Description: "Match a location description to an existing location or indicate it is new",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"matchedClusterId": map[string]interface{}{
							"type":        []string{"string", "null"},
							"description": "The id of the matched cluster, or null if no match",
						},
						"canonicalName": map[string]interface{}{
							"type":        "string",
							"description": "The canonical name for this location",
						},
						"confidence": map[string]interface{}{
							"type":        "number",
							"description": "Confidence in the match (0-1)",
						},
					},
					"required": []string{"matchedClusterId", "canonicalName", "confidence"},
				},
			},
		},
	}

	messages := []ai.ChatMessage{
		{
			Role: "user",
			Content: fmt.Sprintf(`Given this location description: "%s"

And these existing locations:
%s
Determine if the description refers to one of the existing locations or is a new location.
Consider semantic similarity - "the cafe" matches "Coffee Shop", "town center" matches "Town Square", etc.

Call the resolveLocation tool with:
- matchedClusterId: the id of the matching location, or null if it's a new place
- canonicalName: the best canonical name for this location
- confidence: how confident you are in the match (0.0-1.0)`, description, clusterList.String()),
		},
	}

	resp, err := ai.GenerateText(modelID, messages, tools, "required")
	if err != nil {
		name := ExtractCanonicalName(description)
		return &ResolveLocationResult{
			ClusterID:     nil,
			CanonicalName: name,
			IsNew:         true,
		}, nil
	}

	if len(resp.Choices) == 0 || len(resp.Choices[0].Message.ToolCalls) == 0 {
		name := ExtractCanonicalName(description)
		return &ResolveLocationResult{
			ClusterID:     nil,
			CanonicalName: name,
			IsNew:         true,
		}, nil
	}

	tc := resp.Choices[0].Message.ToolCalls[0]
	if tc.Function.Name != "resolveLocation" {
		name := ExtractCanonicalName(description)
		return &ResolveLocationResult{
			ClusterID:     nil,
			CanonicalName: name,
			IsNew:         true,
		}, nil
	}

	var args struct {
		MatchedClusterID *string `json:"matchedClusterId"`
		CanonicalName    string  `json:"canonicalName"`
		Confidence       float64 `json:"confidence"`
	}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		name := ExtractCanonicalName(description)
		return &ResolveLocationResult{
			ClusterID:     nil,
			CanonicalName: name,
			IsNew:         true,
		}, nil
	}

	if args.MatchedClusterID != nil && args.Confidence >= 0.6 {
		// Find the canonical name from the existing cluster
		canonicalName := args.CanonicalName
		for _, c := range existingClusters {
			if c.ID == *args.MatchedClusterID {
				canonicalName = c.CanonicalName
				break
			}
		}
		return &ResolveLocationResult{
			ClusterID:     args.MatchedClusterID,
			CanonicalName: canonicalName,
			IsNew:         false,
		}, nil
	}

	name := args.CanonicalName
	if name == "" {
		name = ExtractCanonicalName(description)
	}
	return &ResolveLocationResult{
		ClusterID:     nil,
		CanonicalName: name,
		IsNew:         true,
	}, nil
}

var prefixRe = regexp.MustCompile(`(?i)^(the|a|an|my|your|their|our|to|towards?|into)\s+`)
var suffixRe = regexp.MustCompile(`(?i)\s+(area|place|spot|room|building)$`)

// ExtractCanonicalName extracts a short canonical name from a description
func ExtractCanonicalName(description string) string {
	cleaned := prefixRe.ReplaceAllString(description, "")
	cleaned = suffixRe.ReplaceAllString(cleaned, "")
	cleaned = strings.TrimSpace(cleaned)

	words := strings.Fields(cleaned)
	if len(words) > 4 {
		words = words[:4]
	}

	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + strings.ToLower(w[1:])
		}
	}
	return strings.Join(words, " ")
}
