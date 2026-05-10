"""Original decoder-only Transformer model for ChatBot.

The model is inspired by patterns used in modern open LLMs, but it does not
load or copy their weights. The 10B config is our own dense decoder blueprint:
RoPE positions, RMSNorm, grouped-query attention, SwiGLU feed-forward layers,
and tied token/output embeddings.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .params import estimate_parameter_count


# A KV cache stores the key and value tensors from earlier generated tokens.
# Reusing those tensors makes chat generation much faster than re-reading the
# full prompt for every new token.
PastKeyValue = Tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by many modern LLMs.

    LayerNorm recenters and rescales activations. RMSNorm only rescales them,
    which is simpler and common in large decoder-only models.
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the average squared size of each hidden vector. This is the
        # "RMS" part: root mean square.
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        # rsqrt means "1 / sqrt". Multiplying by it rescales large activations
        # down and small activations up, which keeps training numerically calm.
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of hidden values for rotary position embeddings."""

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE).

    RoPE adds position information by rotating query/key vectors. Unlike a
    learned position table, it has no trainable parameters and works naturally
    with KV caching because every token position has a deterministic rotation.
    """

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        # inv_freq controls how quickly each pair of hidden dimensions rotates.
        # Early dimensions rotate quickly; later dimensions rotate more slowly.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids is [0, 1, 2, ...] for a fresh prompt, or continues after
        # the cached prompt length during generation.
        freqs = torch.einsum("t,d->td", position_ids.float(), self.inv_freq)

        # Each frequency is used for a pair of dimensions, so repeat_interleave
        # expands [d/2] values into [d] values that match the attention head.
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        # Add two singleton dimensions so cos/sin broadcast over batch and
        # attention heads: [1, 1, seq_len, head_dim].
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat key/value heads so grouped-query attention can use them."""

    if repeats == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape

    # GQA uses fewer key/value heads than query heads. Repeating is how each
    # query head gets a matching key/value head without storing extra weights.
    x = x[:, :, None, :, :].expand(batch, kv_heads, repeats, seq_len, head_dim)
    return x.reshape(batch, kv_heads * repeats, seq_len, head_dim)


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with grouped key/value heads."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.kv_repeats = config.n_head // config.n_kv_head

        # q_proj creates one query head per attention head. k_proj and v_proj
        # create fewer heads when grouped-query attention is enabled.
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.use_bias)

        # o_proj mixes all heads back into one hidden vector per token.
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PastKeyValue]]:
        batch, seq_len, _ = x.shape

        # Project hidden states into query/key/value tensors and reshape them
        # into [batch, heads, tokens, head_dim], the layout used by attention.
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE is applied to q and k because attention scores come from q*k.
        # Values carry content, so they are not position-rotated.
        q, k = self.rope(q, k, position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value

            # During generation, append the new token's k/v to the cached past.
            # This avoids recalculating old keys and values every step.
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        # After caching, repeat the smaller key/value head set so the tensor
        # has one key/value head for every query head.
        k = repeat_kv(k, self.kv_repeats)
        v = repeat_kv(v, self.kv_repeats)

        # Dot-product attention: high score means this query token should pay
        # more attention to that key token. Dividing by sqrt(head_dim) keeps
        # scores from getting too large as the head gets wider.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal masking stops a token from looking at future tokens. That is
        # what makes next-token prediction honest during training.
        total_len = k.size(2)
        key_positions = torch.arange(total_len, device=x.device)
        mask = key_positions[None, :] > position_ids[:, None]
        scores = scores.masked_fill(mask[None, None, :, :], torch.finfo(scores.dtype).min)

        # Softmax turns raw scores into probabilities that add up to 1.
        weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        weights = self.attn_dropout(weights)
        y = torch.matmul(weights, v)

        # Merge attention heads back into [batch, tokens, hidden_size].
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.config.n_embd)
        return self.resid_dropout(self.o_proj(y)), present


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    The gate decides what information should pass through, and the up/down
    projections expand then compress the hidden state.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden_size, bias=config.use_bias)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden_size, bias=config.use_bias)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.n_embd, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(gate) decides what information should pass through. Multiplying
        # by up_proj(x) is the "gated" part of SwiGLU.
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DecoderBlock(nn.Module):
    """One Transformer block: normalize, attend, normalize, feed forward."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PastKeyValue]]:
        attn_out, present = self.attn(
            self.attn_norm(x),
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        # Residual connections add the block's change back to the original
        # hidden state. They help gradients flow through many layers.
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class TransformerChatModel(nn.Module):
    """Original ChatBot decoder model.

    It is still trained with next-token prediction, but the internals now match
    the building blocks used by current open LLM families.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings are the model's lookup table from token id to a
        # dense vector the neural network can process.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # Stacking many decoder blocks is what gives a Transformer depth.
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Tying means the input embedding table is reused as the output
            # classifier. This saves parameters and is common in LLMs.
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # New weights start as small random numbers. Training gradually nudges
        # them toward values that predict the next token well.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: Optional[List[PastKeyValue]] = None,
        use_cache: bool = False,
    ):
        """Run the model and optionally compute next-token loss."""

        _, seq_len = idx.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)

        # block_size is the context window. The model cannot attend to more
        # tokens than this without changing its configuration.
        if past_len + seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {past_len + seq_len} is larger than block_size "
                f"{self.config.block_size}."
            )

        # Position ids tell RoPE where each token lives in the sequence.
        position_ids = torch.arange(past_len, past_len + seq_len, device=idx.device)
        x = self.dropout(self.token_embedding(idx))
        next_cache: List[PastKeyValue] = []

        for layer_index, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            x, present = block(
                x,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and present is not None:
                next_cache.append(present)

        x = self.final_norm(x)

        # logits are raw scores for every vocabulary token at every position.
        # The highest logit is the model's strongest next-token guess.
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy compares the logits with the correct next tokens.
            # Padding tokens are ignored so short examples do not affect loss.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )

        if use_cache:
            return logits, loss, next_cache
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate new token ids after the prompt tokens in ``idx``."""

        if num_beams > 1:
            return self._beam_search(idx, max_new_tokens=max_new_tokens, num_beams=num_beams)

        for _ in range(max_new_tokens):
            # Keep only the latest context-window tokens if the conversation is
            # longer than block_size.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)

            # We only sample from the final position because it predicts the
            # next token after the entire prompt.
            logits = logits[:, -1, :]
            logits = apply_repetition_penalty(logits, idx, repetition_penalty)
            next_id = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def _beam_search(self, idx: torch.Tensor, max_new_tokens: int, num_beams: int) -> torch.Tensor:
        """Small, readable beam search for single-prompt inference."""

        if idx.size(0) != 1:
            raise ValueError("Beam search currently supports batch size 1.")

        beams = [(idx, 0.0)]
        for _ in range(max_new_tokens):
            candidates = []
            for tokens, score in beams:
                logits, _ = self(tokens[:, -self.config.block_size :])
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

                # Each beam branches into its best next-token options. We keep
                # only the strongest num_beams candidates after sorting.
                values, token_ids = torch.topk(log_probs, k=num_beams, dim=-1)
                for value, token_id in zip(values[0], token_ids[0]):
                    next_tokens = torch.cat([tokens, token_id.view(1, 1)], dim=1)
                    candidates.append((next_tokens, score + float(value.item())))
            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:num_beams]
        return beams[0][0]


def apply_repetition_penalty(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    """Lower the score of tokens that already appeared in the prompt."""

    if repetition_penalty <= 1.0:
        return logits
    adjusted = logits.clone()
    for batch_index in range(previous_tokens.size(0)):
        for token_id in set(previous_tokens[batch_index].tolist()):
            # Dividing lowers the chance of repeating that token again.
            adjusted[batch_index, token_id] = adjusted[batch_index, token_id] / repetition_penalty
    return adjusted


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    do_sample: bool,
) -> torch.Tensor:
    """Pick one next token using greedy or filtered sampling."""

    if not do_sample or temperature <= 0:
        # Greedy decoding is deterministic: always choose the best-scoring id.
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Higher temperature makes the probability distribution flatter and more
    # random. Lower temperature makes it sharper and more conservative.
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        # Remove every token outside the best k scores.
        logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
    if top_p is not None and 0 < top_p < 1:
        # Nucleus sampling keeps the smallest high-probability set whose
        # combined probability reaches top_p.
        logits = top_p_filter(logits, top_p)

    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probability mass reaches top_p."""

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mark tokens after the probability mass has passed top_p, but shift the
    # mask right so the first token that crosses the threshold is still kept.
    sorted_remove = cumulative_probs > top_p
    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
    sorted_remove[:, 0] = False
    remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, -float("inf"))
