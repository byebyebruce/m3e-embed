package main

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

func main() {
	cfg := openai.DefaultConfig("")
	cfg.BaseURL = "http://127.0.0.1:5000/v1"
	cli := openai.NewClientWithConfig(cfg)

	ctx := context.Background()
	text := []string{
		"Once upon a time",
		"There was a frog",
		"Who lived in a well",
	}

	resp, err := cli.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: text,
		Model: openai.AdaEmbeddingV2,
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(resp)
}
