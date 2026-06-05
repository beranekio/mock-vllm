package integration

import (
	"context"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func anthropicClient(t *testing.T) anthropic.Client {
	t.Helper()
	return anthropic.NewClient(
		option.WithBaseURL(anthropicBaseURL(t)),
		option.WithAPIKey("test-key"),
	)
}

func TestAnthropic_Messages(t *testing.T) {
	ctx := context.Background()
	client := anthropicClient(t)

	msg, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     testModel,
		MaxTokens: 64,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
		},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	if msg.StopReason != anthropic.StopReasonEndTurn {
		t.Fatalf("stop_reason = %q, want end_turn", msg.StopReason)
	}
	if len(msg.Content) == 0 {
		t.Fatal("expected content blocks")
	}
	if got := msg.Content[0].AsText().Text; got != "hi" {
		t.Fatalf("text = %q, want hi", got)
	}
}

func TestAnthropic_MessagesStream(t *testing.T) {
	ctx := context.Background()
	client := anthropicClient(t)

	stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Model:     testModel,
		MaxTokens: 64,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
		},
	})

	var text string
	for stream.Next() {
		event := stream.Current()
		if event.Type != "content_block_delta" {
			continue
		}
		blockDelta := event.AsContentBlockDelta()
		if td := blockDelta.Delta.AsTextDelta(); td.Text != "" {
			text += td.Text
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if text != "hi" {
		t.Fatalf("streamed text = %q, want hi", text)
	}
}

func TestAnthropic_CountTokens(t *testing.T) {
	ctx := context.Background()
	client := anthropicClient(t)

	count, err := client.Messages.CountTokens(ctx, anthropic.MessageCountTokensParams{
		Model: testModel,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
		},
	})
	if err != nil {
		t.Fatalf("Messages.CountTokens: %v", err)
	}
	if count.InputTokens <= 0 {
		t.Fatalf("input_tokens = %d, want > 0", count.InputTokens)
	}
}
