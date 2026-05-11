# Forge-250B architecture

A ~250B-total / ~25B-active-per-token Mixture-of-Experts decoder-only
Transformer specialized for code and software engineering tasks. Pairs
DeepSeek-V3-style fine-grained MoE with Multi-head Latent Attention (MLA)
so a 1M-token context is computationally viable.

## Big picture

```
   token ids ─▶ Embedding (vocab 200K, 6144-d)
                       │
              ┌── layers 0..2 ──┐         dense FFN warm-up
              │ MLA attention   │         (DeepSeek V3 trick: stabilizes
              │ SwiGLU FFN      │          early training)
              └─────────────────┘
                       │
              ┌── layers 3..63 ──────────────────────────┐
              │ MLA attention                            │
              │                                          │
              │ MoE FFN                                  │
              │   ├ router: top-8 of 128 routed experts  │
              │   ├ shared expert (always on)            │
              │   └ AL-free load balancing (bias trick)  │
              └──────────────────────────────────────────┘
                       │
                  RMSNorm → LM head → logits
```

## Why MoE?

A dense 250B Transformer is wildly expensive per token (every parameter
participates in every step). MoE keeps the same "knowledge" capacity but
only routes each token through a handful of specialist FFNs. Practical
shape for Forge:

* 128 routed expert FFNs per MoE layer, each FFN-hidden 1536.
* 8 routed experts active per token + 1 shared expert always on.
* Active params per token ≈ 25 B (out of 250 B total).
* The shared expert gives every token a basic "general" path so the
  router doesn't have to invent one.

## Why MLA?

The naive KV cache for a 1M-token context with normal GQA would be
hundreds of gigabytes per request — fiction, not engineering. MLA stores
a compressed latent (`kv_lora_rank = 512`) plus a tiny RoPE-only key
(`qk_rope_head_dim = 64`) per token. Re-expansion to per-head K and V
happens on the fly during attention:

```
cached_per_token ≈ kv_lora_rank + qk_rope_head_dim
                  = 512 + 64 = 576 floats
                  ≈ 1.1 KB in bf16

full-1M cache    ≈ 576 × 1_048_576 × 2 bytes
                  ≈ 1.1 GB per attention layer
                  × 64 layers
                  ≈ ~70 GB / request
```

That's the headline number. The same context in plain GQA would be ~10x
larger.

## Configuration knobs (highlights)

| Knob | Default | Why |
|---|---|---|
| `d_model` | 6144 | Residual width — narrower than Aurora because the FFN capacity is in MoE. |
| `n_layers` | 64 | 3 dense + 61 MoE. |
| `attention.variant` | `mla` | The whole point. |
| `attention.kv_lora_rank` | 512 | The "cache size per token". |
| `moe.num_routed_experts` | 128 | Pool of specialists. |
| `moe.num_active_experts` | 8 | Top-k routing. |
| `moe.expert_hidden` | 1536 | Each expert FFN is small. |
| `moe.load_balancing` | `aux_loss_free` | DeepSeek's bias trick. |
| `vocab_size` | 200032 | 200K BPE — generous because code has many rare tokens. |
| `max_position_embeddings` | 1048576 | 1M context — Opus-class. |
| `rope_base` | 50,000,000 | Tuned for the 1M window. |

## The plugin / skill / tool surface

Forge ships with a Claude-Code-style runtime (`src/chatbot/runtime/`). See
`docs/plugin-system.md` for the full reference. Highlights:

* Built-in tools: filesystem, shell, http, notebook.
* MCP client for connecting to any MCP server.
* Markdown skills auto-prefixed into the system prompt by trigger match.
* Pre/post-tool, on-message, and on-stop hooks for redaction / logging.
* Slash commands (`/help`, `/skills`, `/tools`, plus plugin-provided ones).
* Subagent spawning with isolated tool subsets.

## How to train Forge-250B

See `docs/training-guide.md` for full details. Forge's pipeline is the same
as Aurora's plus one extra stage at the end:

1. Train the 200K code-heavy BPE tokenizer.
2. Pretrain at 8K context (mostly code + math + a slice of general web).
3. Long-context extension to 1M (repo-packed sequences dominate).
4. SFT.
5. DPO.
6. Tool-use SFT — teaches the model to emit clean `<|tool_call|>` blocks
   that the runtime can dispatch.

The model code lives in `src/chatbot/models/forge_250b/`.
