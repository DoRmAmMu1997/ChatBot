"""Decoder-only Transformer model for the chatbot.

This is a small LLM-style architecture: every token can only attend to earlier
tokens, and training asks the model to predict the next token in the dialogue.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class TransformerChatModel(nn.Module):
    """A compact GPT-like model built with PyTorch Transformer blocks.

    "Decoder-only" means the model reads the conversation from left to right and
    repeatedly answers the question: "what token should come next?" This is the
    same basic training objective used by larger language models.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings turn integer token ids into learnable vectors. The
        # model cannot work directly with words like "hello"; it works with
        # numbers, and embeddings give those numbers meaning during training.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # A Transformer sees all tokens in a sequence at once, so we add a
        # second learned vector that tells it where each token appears.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # Each Transformer layer contains self-attention plus a feed-forward
        # network. Self-attention lets every token compare itself with earlier
        # tokens, which is how the model uses conversation context.
        layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layer)
        self.final_norm = nn.LayerNorm(config.n_embd)

        # The language-model head converts hidden vectors back into vocabulary
        # scores. A score is produced for every token the model knows.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Reusing the token embedding matrix for the output layer is common in
        # language models. It saves parameters and usually helps small models.
        self.lm_head.weight = self.token_embedding.weight

        # The causal mask hides future tokens. Without this mask, the model
        # could "cheat" during training by looking at the answer token it is
        # supposed to predict.
        mask = torch.triu(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with stable defaults for small Transformers."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Run the model and optionally compute next-token loss.

        idx has shape [batch, time]. Each row is one training example, and each
        column is one token position in that example.
        """

        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} is larger than block_size "
                f"{self.config.block_size}."
            )

        # Create position ids [0, 1, 2, ...] so every token gets both a token
        # meaning and a location meaning.
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(positions)
        x = self.dropout(token_embeddings + position_embeddings)

        causal_mask = self.causal_mask[:seq_len, :seq_len]
        x = self.blocks(x, mask=causal_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross entropy compares the model's vocabulary scores with the
            # correct next token. Padding is ignored because it is only filler,
            # not real conversation text.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        """Generate new token ids after the prompt tokens in idx.

        Generation is a loop: predict one next token, append it to the prompt,
        then use the longer prompt to predict the following token.
        """

        for _ in range(max_new_tokens):
            # Keep only the latest block_size tokens so the prompt fits inside
            # the context window the model was trained for.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)

            # We only need the final position because that is the model's
            # prediction for the next token after the whole prompt.
            logits = logits[:, -1, :]

            if temperature <= 0:
                # Greedy decoding: always pick the highest-scoring token.
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature controls randomness. Lower values are safer and
                # more repetitive; higher values are more varied but less stable.
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    # top_k sampling keeps only the k most likely tokens before
                    # sampling, which prevents very unlikely words from popping
                    # into the response.
                    values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")
                probabilities = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

        return idx
