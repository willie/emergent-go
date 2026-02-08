package ai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

var apiKey string

// Models by task
var Models = struct {
	MainConversation    string
	OffscreenSimulation string
	Fast                string
}{
	MainConversation:    "z-ai/glm-4.6:exacto",
	OffscreenSimulation: "z-ai/glm-4.6:exacto",
	Fast:                "z-ai/glm-4.6:exacto",
}

// AvailableModels is the list of selectable models
var AvailableModels = []string{
	"deepseek/deepseek-v3.1-terminus:exacto",
	"openai/gpt-oss-120b:exacto",
	"qwen/qwen3-coder:exacto",
	"moonshotai/kimi-k2-0905:exacto",
	"z-ai/glm-4.6:exacto",
}

func Init() {
	apiKey = os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("WARNING: OPENROUTER_API_KEY not set")
	}
}

// ToolFunction defines a tool the model can call
type ToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`
}

// Tool for the OpenRouter API
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ChatMessage for the API
type ChatMessage struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content"` // string or null for tool_calls
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
	Name       string      `json:"name,omitempty"`
}

// ToolCall from the model
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall details
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatRequest to OpenRouter
type ChatRequest struct {
	Model      string        `json:"model"`
	Messages   []ChatMessage `json:"messages"`
	Tools      []Tool        `json:"tools,omitempty"`
	ToolChoice interface{}   `json:"tool_choice,omitempty"`
	Stream     bool          `json:"stream"`
}

// ChatResponse (non-streaming)
type ChatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

// StreamDelta for SSE
type StreamDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// StreamChoice in SSE event
type StreamChoice struct {
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// StreamEvent is one SSE data payload
type StreamEvent struct {
	ID      string         `json:"id"`
	Choices []StreamChoice `json:"choices"`
}

// GenerateText makes a non-streaming chat completion
func GenerateText(model string, messages []ChatMessage, tools []Tool, toolChoice interface{}) (*ChatResponse, error) {
	if model == "" {
		model = Models.Fast
	}

	req := ChatRequest{
		Model:      model,
		Messages:   messages,
		Tools:      tools,
		ToolChoice: toolChoice,
		Stream:     false,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	return &chatResp, nil
}

// StreamCallback is called for each SSE content delta
type StreamCallback func(content string)

// ToolCallAccumulator collects streaming tool call fragments
type ToolCallAccumulator struct {
	ID       string
	Type     string
	Name     string
	ArgsBuf  strings.Builder
}

// StreamResult contains the final result of a streaming request
type StreamResult struct {
	Content   string
	ToolCalls []ToolCall
}

// StreamText makes a streaming chat completion, calling cb for each text chunk
func StreamText(model string, messages []ChatMessage, tools []Tool, cb StreamCallback) (*StreamResult, error) {
	if model == "" {
		model = Models.MainConversation
	}

	req := ChatRequest{
		Model:    model,
		Messages: messages,
		Tools:    tools,
		Stream:   true,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var fullContent strings.Builder
	toolAccumulators := make(map[int]*ToolCallAccumulator)

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var event StreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		for _, choice := range event.Choices {
			if choice.Delta.Content != "" {
				fullContent.WriteString(choice.Delta.Content)
				if cb != nil {
					cb(choice.Delta.Content)
				}
			}

			for _, tc := range choice.Delta.ToolCalls {
				idx := 0 // Use index from the tool call if available
				if _, ok := toolAccumulators[idx]; !ok {
					toolAccumulators[idx] = &ToolCallAccumulator{
						ID:   tc.ID,
						Type: tc.Type,
						Name: tc.Function.Name,
					}
				}
				acc := toolAccumulators[idx]
				if tc.ID != "" {
					acc.ID = tc.ID
				}
				if tc.Type != "" {
					acc.Type = tc.Type
				}
				if tc.Function.Name != "" {
					acc.Name = tc.Function.Name
				}
				acc.ArgsBuf.WriteString(tc.Function.Arguments)
			}
		}
	}

	var toolCalls []ToolCall
	for _, acc := range toolAccumulators {
		toolCalls = append(toolCalls, ToolCall{
			ID:   acc.ID,
			Type: acc.Type,
			Function: FunctionCall{
				Name:      acc.Name,
				Arguments: acc.ArgsBuf.String(),
			},
		})
	}

	return &StreamResult{
		Content:   fullContent.String(),
		ToolCalls: toolCalls,
	}, nil
}
