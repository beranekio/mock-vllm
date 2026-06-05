package integration

import (
	"context"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func openAIClient(t *testing.T) openai.Client {
	t.Helper()
	return openai.NewClient(
		option.WithBaseURL(openAIBaseURL(t)),
		option.WithAPIKey("test-key"),
	)
}

func TestOpenAI_ListModels(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	page, err := client.Models.List(ctx)
	if err != nil {
		t.Fatalf("Models.List: %v", err)
	}
	if len(page.Data) == 0 {
		t.Fatal("expected at least one model")
	}
	if page.Data[0].ID != testModel {
		t.Fatalf("model id = %q, want %q", page.Data[0].ID, testModel)
	}
}

func TestOpenAI_ChatCompletions(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: testModel,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hi"),
		},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected choices")
	}
	if got := resp.Choices[0].Message.Content; got != "hi" {
		t.Fatalf("content = %q, want hi", got)
	}
}

func TestOpenAI_ChatCompletionsStream(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Model: testModel,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hi"),
		},
	})

	acc := openai.ChatCompletionAccumulator{}
	for stream.Next() {
		acc.AddChunk(stream.Current())
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(acc.Choices) == 0 {
		t.Fatal("expected accumulated choices")
	}
	if got := acc.Choices[0].Message.Content; got != "hi" {
		t.Fatalf("content = %q, want hi", got)
	}
}

func TestOpenAI_Completions(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Completions.New(ctx, openai.CompletionNewParams{
		Model: testModel,
		Prompt: openai.CompletionNewParamsPromptUnion{
			OfString: openai.String("hello"),
		},
	})
	if err != nil {
		t.Fatalf("Completions.New: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected choices")
	}
	if got := resp.Choices[0].Text; got != "hi" {
		t.Fatalf("text = %q, want hi", got)
	}
}

func TestOpenAI_Embeddings(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: testModel,
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String("test"),
		},
	})
	if err != nil {
		t.Fatalf("Embeddings.New: %v", err)
	}
	if len(resp.Data) == 0 || len(resp.Data[0].Embedding) == 0 {
		t.Fatalf("unexpected embedding response: %+v", resp)
	}
}
