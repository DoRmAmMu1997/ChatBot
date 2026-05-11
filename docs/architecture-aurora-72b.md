# Aurora architecture (omni-modal, ~72B dense)

A dense, GPT-4o-class omni-modal Transformer. Text + image + audio in,
text + audio out. Designed for natural-feeling conversation over a 256K-
token context. The file is still called `architecture-aurora-50b.md` for
backwards compatibility, but the numbers below are the v2 (~72B) shape.

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
            ┌─────────────────────────────┐
audio  ──── │ Audio Encoder (Conformer)   │
            └────────────┬────────────────┘
                         │ soft audio tokens (~50/sec)
                         ▼
   token ids ─────────▶ Embedding (vocab 132K, 8192-d)
                         │
            ┌──── for L in range(72) ────┐
            │  RMSNorm                   │
            │  GQA (64q / 8kv heads, RoPE)│
            │  ── residual ──            │
            │  RMSNorm                   │
            │  SwiGLU FFN (28672 hidden) │
            │  ── residual ──            │
            └────────────────────────────┘
                         │
                    RMSNorm
                         │
                    LM head (8192 → 132K)
                         │
                    next-token logits
                         │
            audio-code spans → Audio Codec (decode) → waveform
```

## Configuration knobs

| Knob | Default | Why |
|---|---|---|
| `d_model` | 8192 | Residual-stream width — bumped from 7168 for capacity. |
| `n_layers` | 72 | Depth — bumped from 64. |
| `n_heads` | 64 | Query heads. With `head_dim=128`, `n_heads * head_dim = d_model`. |
| `n_kv_heads` | 8 | KV heads (GQA). Each KV head is shared by 8 query heads → 8x KV-cache reduction. |
| `ffn_hidden` | 28672 | SwiGLU intermediate (~3.5x `d_model`). |
| `vocab_size` | 132096 | 128K text BPE + 4096 audio codes + room for specials. |
| `max_position_embeddings` | 262144 | 256K context — Sonnet-class. Reached via YaRN extension. |
| `rope_base` | 2,000,000 | High base smoothly extends RoPE to long contexts. |
| `rope_scaling.factor` | 8.0 | YaRN: stretches an 8K-trained context up to 64K, then further during the long-context stage. |
| `vision.image_size` | 384 | SigLIP2-So400M default. |
| `vision.num_image_tokens` | 729 | 27×27 patch grid. |
| `audio.encoder_layers` | 12 | Conformer-ish encoder; depth tuned for open-domain speech. |
| `audio.num_audio_codes` | 4096 | Single-codebook EnCodec-style — adds 4096 entries to the LM vocab. |

## Vision tower

SigLIP2-style ViT, `dim=1152`, `depth=27`, `num_heads=16`, image size
384, patch size 14 → grid of 27×27 = 729 patch tokens. An MLP connector
(2 hidden layers, GELU activation) projects each patch into the LLM's
8192-d embedding space.

## Audio I/O

**Input** (`<|audio|>` placeholders):

* `LogMelSpectrogram` turns the raw waveform into 128 mel bands at 10 ms
  hop.
* Two strided Conv1d layers shrink the time axis 4×.
* 12 Transformer blocks (RMSNorm + MHA + SwiGLU) process the spectrogram.
* A 2-layer MLP projects each frame into the LLM's d_model.

Resulting frame rate: ~50 audio tokens per second of speech.

**Output** (audio code tokens in vocabulary):

* The LLM autoregressively emits tokens from the merged text+audio vocab.
* When it produces a span between `<|audio_start|>` and `<|audio_end|>`,
  the runtime looks up each `<audio:N>` token in `AudioCodec.codebook`,
  feeds the embedding sequence into the codec decoder, and returns a
  24 kHz waveform.
* `AudioCodec` is a small (~120 M) separate model, pretrained as an
  autoencoder via `configs/training/audio_codec_pretrain.yaml`.

## Memory at full 256K

Standard GQA KV cache at 256K with bf16:
```
2 (K and V) × n_layers × n_kv_heads × head_dim × seq × 2 bytes
= 2 × 72 × 8 × 128 × 262144 × 2 ≈ 45 GB / request
```

## How to train Aurora end to end

See `docs/training-guide.md` for the full step-by-step. Headline order:

1. Tokenizer (128K BPE + 4096 audio codes).
2. Audio codec autoencoder pretrain.
3. Text pretrain at 8K context.
4. Long-context extension to 256K.
5. Omni-modal pretrain (adds image + audio streams).
6. SFT.
7. Omni-SFT (voice-assistant instruction data).
8. DPO.
