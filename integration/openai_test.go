package integration

import (
	"context"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
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

func TestOpenAI_ChatCompletions_multiTurn(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: testModel,
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfAssistant: &openai.ChatCompletionAssistantMessageParam{
					Content: openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.String("goodbye"),
					},
				},
			},
			openai.UserMessage("hi"),
		},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
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

func TestOpenAI_Completions_batch(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Completions.New(ctx, openai.CompletionNewParams{
		Model: testModel,
		Prompt: openai.CompletionNewParamsPromptUnion{
			OfArrayOfStrings: []string{"hi", "bye"},
		},
	})
	if err != nil {
		t.Fatalf("Completions.New: %v", err)
	}
	if len(resp.Choices) != 2 {
		t.Fatalf("len(choices) = %d, want 2", len(resp.Choices))
	}
	if resp.Choices[0].Text != "hi" || resp.Choices[1].Text != "bye" {
		t.Fatalf("texts = %q, %q", resp.Choices[0].Text, resp.Choices[1].Text)
	}
	if resp.Choices[0].Index != 0 || resp.Choices[1].Index != 1 {
		t.Fatalf("indices = %d, %d", resp.Choices[0].Index, resp.Choices[1].Index)
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

func TestOpenAI_Responses(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: testModel,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String("hi"),
		},
	})
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	if resp.Status != responses.ResponseStatusCompleted {
		t.Fatalf("status = %q, want completed", resp.Status)
	}
	if got := resp.OutputText(); got != "hi" {
		t.Fatalf("output text = %q, want hi", got)
	}
}

func TestOpenAI_Responses_structuredInput(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: testModel,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage(
					responses.ResponseInputMessageContentListParam{
						responses.ResponseInputContentParamOfInputText("hi"),
					},
					responses.EasyInputMessageRoleUser,
				),
			},
		},
	})
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	if got := resp.OutputText(); got != "hi" {
		t.Fatalf("output text = %q, want hi", got)
	}
}

func TestOpenAI_Responses_multiTurn(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: testModel,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage("goodbye", responses.EasyInputMessageRoleAssistant),
				responses.ResponseInputItemParamOfMessage("hi", responses.EasyInputMessageRoleUser),
			},
		},
	})
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	if got := resp.OutputText(); got != "hi" {
		t.Fatalf("output text = %q, want hi", got)
	}
}

func TestOpenAI_Embeddings_batch(t *testing.T) {
	ctx := context.Background()
	client := openAIClient(t)

	resp, err := client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: testModel,
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: []string{"a", "b"},
		},
	})
	if err != nil {
		t.Fatalf("Embeddings.New: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("len(data) = %d, want 2", len(resp.Data))
	}
	if resp.Data[0].Index != 0 || resp.Data[1].Index != 1 {
		t.Fatalf("unexpected indices: %d, %d", resp.Data[0].Index, resp.Data[1].Index)
	}
}
