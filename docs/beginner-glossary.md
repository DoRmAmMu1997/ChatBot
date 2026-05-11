# Beginner glossary

Short, plain-English definitions of the jargon used throughout this repo,
each with a pointer to the file that implements it. Read this once and the
rest of the codebase becomes much easier to navigate.

| Term | What it means | Where to look |
|---|---|---|
| **Token** | A small piece of text the model works with (often a word fragment). | `src/chatbot/tokenizer/bpe.py` |
| **BPE** | "Byte-Pair Encoding" — the algorithm that learns the token vocabulary by merging the most common byte pairs. | `src/chatbot/tokenizer/bpe.py` |
| **Embedding** | Each token id is mapped to a vector. The vector is what the model actually computes with. | `nn.Embedding` calls in `aurora_50b/model.py` and `forge_250b/model.py` |
| **RMSNorm** | A lightweight normalization layer. Keeps activations from blowing up or vanishing. Cheaper than LayerNorm. | `src/chatbot/models/common/normalization.py` |
| **RoPE** | "Rotary Position Embedding". A way of telling the attention layer where each token sits in the sequence by rotating its vectors. | `src/chatbot/models/common/rope.py` |
| **YaRN** | A trick on top of RoPE that lets a model trained at one context length generalize to a much longer one. We use it to reach 256K (Aurora) and 1M (Forge). | `src/chatbot/models/common/rope.py` |
| **GQA** | "Grouped-Query Attention". The model has many query heads but fewer key/value heads. Saves a lot of KV-cache memory. | `src/chatbot/models/common/attention.py` |
| **MLA** | "Multi-head Latent Attention". Compresses keys and values into a small latent so the KV cache scales sub-linearly in head count. Essential for Forge's 1M context. | `src/chatbot/models/common/attention.py` |
| **SwiGLU** | The feed-forward block used by every modern LLM: SiLU-gated linear units. | `src/chatbot/models/common/ffn.py` |
| **MoE** | "Mixture of Experts". Many specialized FFNs per layer; only a few run per token. Buys capacity without paying full FLOPs. | `src/chatbot/models/common/moe.py` |
| **Auxiliary-loss-free balancing** | A way to keep experts balanced (DeepSeek-V3): nudge a per-expert bias instead of adding an extra loss term. | `MixtureOfExperts.forward` in `moe.py` |
| **KV cache** | During generation, we cache the keys and values of past tokens so each new token only does a tiny amount of attention work. | `src/chatbot/models/common/kv_cache.py` |
| **Causal mask** | The "no peeking at the future" rule. Each token can only attend to tokens that came before it. Enforced automatically by SDPA's `is_causal=True`. | `attention.py` |
| **Sequence packing** | Glue many short examples back-to-back into a fixed-length training block. Hugely improves throughput vs. padding. | `src/chatbot/data/packing.py` |
| **FSDP** | "Fully Sharded Data Parallel" — splits model state across GPUs so a 50B / 250B model fits. | `src/chatbot/training/distributed.py` |
| **DeepSpeed ZeRO** | Microsoft's alternative to FSDP. Same idea, different code path. | `src/chatbot/training/distributed.py` |
| **SFT** | "Supervised Fine-Tuning". Teach the pretrained model to follow instructions on (prompt, ideal-response) pairs. | `src/chatbot/training/sft.py` |
| **DPO** | "Direct Preference Optimization". Teach the model to prefer A over B without a separate reward model. | `src/chatbot/training/dpo.py` |
| **LoRA** | "Low-Rank Adaptation". Freeze the base model and train tiny rank-r adapters. Lets you fine-tune big models on small hardware. | `src/chatbot/training/lora.py` |
| **QLoRA** | LoRA on top of a 4-bit-quantized base model. Lets 50B+ fit on consumer GPUs. | `configs/training/lora.yaml` |
| **MCP** | "Model Context Protocol". An open protocol for letting models talk to external tools over JSON-RPC. | `src/chatbot/runtime/plugins/mcp_client.py` |
| **Skill** | A markdown file with YAML frontmatter that gets prepended to the system prompt when its triggers match. | `src/chatbot/runtime/skills/loader.py` |
| **Hook** | A callback you register for lifecycle events (`pre_tool`, `post_tool`, `on_message`, `on_stop`). | `src/chatbot/runtime/hooks/registry.py` |
