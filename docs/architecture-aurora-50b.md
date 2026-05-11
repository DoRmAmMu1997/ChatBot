# Aurora-50B architecture

A ~50-billion-parameter dense multimodal Transformer (text + images),
Claude-Sonnet-class in spirit. Designed for clear, modern multimodal
interactions over a 256K-token context.

## Big picture

```
            ┌─────────────────────────────┐
images ──── │ Vision Tower (ViT-So400M)   │
            └────────────┬────────────────┘
                         │ patch features
                         ▼
                ┌─────────────────┐
                │ MLP Connector   │ → soft image tokens (729/img)
                └────────┬────────┘
                         │
   token ids ─────────▶  ▼
                ┌─────────────────────────────────┐
                │ Embedding (vocab 128K, 7168-d)   │
                └────────┬────────────────────────┘
                         │
            ┌──── for L in range(64) ────┐
            │  RMSNorm                   │
            │  GQA (56q / 8kv heads, RoPE)│
            │  ── residual ──            │
            │  RMSNorm                   │
            │  SwiGLU FFN (24576 hidden) │
            │  ── residual ──            │
            └────────────────────────────┘
                         │
                    RMSNorm
                         │
                    LM head (7168 → 128K)
                         │
                    next-token logits
```

## Configuration knobs

| Knob | Default | Why |
|---|---|---|
| `d_model` | 7168 | Residual-stream width. Wider = more capacity per token. |
| `n_layers` | 64 | Depth. More layers = more reasoning steps. |
| `n_heads` | 56 | Query heads. With `head_dim=128`, `n_heads * head_dim = d_model`. |
| `n_kv_heads` | 8 | KV heads (GQA). Each KV head is shared by 7 query heads → 7x KV-cache reduction. |
| `ffn_hidden` | 24576 | SwiGLU intermediate (~3.4x `d_model`). |
| `vocab_size` | 131072 | 128K BPE. Balances coverage with embedding/LM-head cost. |
| `max_position_embeddings` | 262144 | 256K context — Sonnet-class. Reached via YaRN extension. |
| `rope_base` | 2,000,000 | High base smoothly extends RoPE to long contexts. |
| `rope_scaling.factor` | 8.0 | YaRN: stretches an 8K-trained context up to 64K, with further extension during the long-context training stage. |

## Vision tower

* SigLIP2-style ViT, `dim=1152`, `depth=27`, `num_heads=16`, image size 384,
  patch size 14 → grid of 27×27 = 729 patch tokens.
* MLP connector (2 hidden layers, GELU activation) projects each patch into
  the LLM's 7168-d embedding space.
* During training and inference, images appear in the prompt as
  `<|image|>` placeholders. The model class splices the connector's
  outputs into the embedding stream exactly where those placeholders sat.

## Memory at full 256K

Standard GQA KV cache at 256K with bf16:
```
2 (K and V) × n_layers × n_kv_heads × head_dim × seq × 2 bytes
= 2 × 64 × 8 × 128 × 262144 × 2 ≈ 40 GB / request
```
So serving Aurora at its full context length needs serious GPU memory.
Most workloads will run shorter contexts and benefit from the 256K only
when the user explicitly asks for it.

## How to train Aurora-50B end to end

See `docs/training-guide.md` for the full step-by-step. The stages are:

1. Train the 128K BPE tokenizer on a representative text mix.
2. Pretrain at 8K context using `configs/training/pretrain.yaml`.
3. Long-context extension to 256K with `configs/training/long_context.yaml`.
4. SFT with `configs/training/sft.yaml`.
5. DPO with `configs/training/dpo.yaml`.

All five stages share the model code in `src/chatbot/models/aurora_50b/`.
