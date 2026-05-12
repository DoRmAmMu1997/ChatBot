# Forge architecture (MoE coder, ~464B total / ~35.5B active)

A fine-grained Mixture-of-Experts decoder-only Transformer specialized
for code, software engineering, and DevOps tasks. Pairs DeepSeek-V3-style
fine-grained MoE with Multi-head Latent Attention (MLA) so a 1M-token
context is computationally viable. v2 adds vision (code screenshots) +
audio (voice descriptions). The config implies ~464B total parameters and
~35.5B active per token — squarely in the Opus-class MoE neighbourhood
(DeepSeek V3 is 671B/37B-active).

## Big picture

```
   screenshots / videos ─▶ small Vision Tower (224 px, 12-layer ViT)
                              │ patch features
                              ▼
                       MLP Connector → 256 image tokens / image
                              │
   voice descriptions ──▶ Audio Encoder (8-layer Conformer)
                              │ ~50 tokens / second
                              ▼
   token ids ─▶ Embedding (vocab 204K — text BPE + 4096 audio codes)
                              │
              ┌── layers 0..2 ──┐         dense FFN warm-up
              │ MLA attention   │         (DeepSeek V3 trick: stabilizes
              │ SwiGLU FFN      │          early training)
              └─────────────────┘
                              │
              ┌── layers 3..71 ──────────────────────────┐
              │ MLA attention                            │
              │                                          │
              │ MoE FFN                                  │
              │   ├ router: top-8 of 160 routed experts  │
              │   ├ shared expert (always on)            │
              │   └ AL-free load balancing (bias trick)  │
              └──────────────────────────────────────────┘
                              │
                         RMSNorm → LM head → logits
                              │
                      audio-code spans → shared AudioCodec
```

## Why MoE?

A dense 460B Transformer is wildly expensive per token (every parameter
participates in every step). MoE keeps the same "knowledge" capacity but
only routes each token through a handful of specialist FFNs. Practical
shape for Forge:

* **160 routed expert FFNs** per MoE layer, each FFN-hidden 2048
  (each expert ≈ `3 × 6656 × 2048 ≈ 40.89 M` params).
* **8 routed experts active per token + 1 shared expert** always on.
* **Active params per token ≈ 35.5 B** (out of ~464 B total).

## Why MLA?

A 1M-context naive GQA KV cache would be hundreds of gigabytes per
request — fiction, not engineering. MLA stores a compressed latent
(`kv_lora_rank = 512`) plus a tiny RoPE-only key
(`qk_rope_head_dim = 64`) per token. Re-expansion to per-head K and V
happens on the fly during attention:

```
cached_per_token ≈ kv_lora_rank + qk_rope_head_dim
                  = 512 + 64 = 576 floats
                  ≈ 1.1 KB in bf16

full-1M cache    ≈ 576 × 1_048_576 × 2 bytes
                  ≈ 1.1 GB per attention layer
                  × 72 layers
                  ≈ ~80 GB / request
```

That's the headline number. The same context in plain GQA would be ~10x
larger.

## Multimodal inputs

| Modality | How Forge ingests it |
|---|---|
| Screenshot | Vision tower → `<\|image\|>` splice. |
| Screen recording / GIF / WebP | `data.video_processing.sample_video_frames` extracts N frames; each becomes one `<\|image\|>`. |
| Voice description | Audio encoder → `<\|audio\|>` splice. |
| PDF / design doc | Runtime `parse_document` tool (uses `pypdf`) extracts text + image references and feeds them back through the chat. |
| Long pasted logs | Plain text. The DevOps tools below do the heavy parsing. |

## DevOps capabilities

Forge ships with a `builtin.devops` toolset (enabled by default in
`configs/runtime/forge-devops.yaml`):

| Tool | Use |
|---|---|
| `parse_logs` | Parse a payload (text or path) into structured records — JSON-lines, syslog, K8s, Apache combined, free-form. |
| `tail_log_file` | Last-N lines of a file with optional level filter. |
| `search_logs` | Regex search across a tree of log files with context windows. |
| `summarize_incidents` | Cluster records by normalized message (strip IDs / IPs / numbers); return top-K. |
| `query_metric` | PromQL-style metrics query — stub backend by default; set `CHATBOT_METRICS_URL` to forward to a real Prometheus. |

Plus a `log_triage` skill in `plugins_examples/log_triage/` that
auto-prefixes when the user message mentions logs / stack traces /
incidents.

## Configuration knobs (highlights)

| Knob | Default | Why |
|---|---|---|
| `d_model` | 6656 | Residual width — bumped from 6144. |
| `n_layers` | 72 | 3 dense + 69 MoE. |
| `attention.variant` | `mla` | The whole point. |
| `attention.kv_lora_rank` | 512 | The "cache size per token". |
| `moe.num_routed_experts` | 160 | Pool of specialists (was 128). |
| `moe.num_active_experts` | 8 | Top-k routing. |
| `moe.expert_hidden` | 2048 | Each expert FFN is small but slightly fatter than v1. |
| `moe.load_balancing` | `aux_loss_free` | DeepSeek's bias trick. |
| `vocab_size` | 204096 | 200K text BPE + 4096 audio codes. |
| `max_position_embeddings` | 1048576 | 1M context — Opus-class. |
| `vision.image_size` | 224 | Smaller than Aurora; code screenshots don't need 384. |
| `vision.num_image_tokens` | 256 | 16×16 patch grid. |
| `audio.encoder_layers` | 8 | Lighter than Aurora's 12. |

## How to train Forge

See `docs/training-guide.md` for full details. Forge's pipeline:

1. Tokenizer (200K BPE + 4096 audio codes).
2. Audio codec autoencoder pretrain (shared with Aurora).
3. Text pretrain at 8K (mostly code + math + a slice of general web).
4. Long-context extension to 1M (repo-packed sequences dominate).
5. Code-screenshot pretrain (vision tower warm-up).
6. SFT.
7. DPO.
8. Tool-use SFT.
9. **DevOps SFT** (new) — logs / incidents / postmortems.
10. **RLEF** (new) — reinforcement learning from execution feedback;
    the key SWE-Bench-Verified lever.
