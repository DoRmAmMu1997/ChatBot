# Chatbot — Aurora & Forge

Two from-scratch PyTorch large language models with a Claude-Code-style
plugin runtime, audio I/O, vision, and a DevOps tool layer. Every block
has beginner-friendly comments, every config knob is documented, and the
full training pipeline runs end-to-end on a laptop at the "tiny" tier
before you commit to renting GPUs.

| Model | Size | Modalities | Specialization | Context |
|---|---|---|---|---|
| **Aurora** | ~72 B dense | text + image + audio in, text + audio out | Omni-modal assistant (GPT-4o-class) | 256K |
| **Forge** | ~320 B MoE / ~31 B active | text + image + audio + video + PDF in, text out (audio optional) | Coding, software engineering, DevOps (Opus-class) | 1M |
| **Tiny (~50 M)** | Same architectures, miniaturised | All modalities | Smoke-testing & education | 4K |

## What ships in this repo

* **Original architectures** for both models. Every block (RMSNorm, RoPE
  + YaRN, GQA, MLA, SwiGLU, fine-grained MoE with auxiliary-loss-free
  load balancing, SigLIP-style vision tower, Whisper-style audio encoder,
  EnCodec-style audio codec) is implemented with `torch.nn` primitives.
  No `transformers` model imports.
* **Tokenizers**: byte-level BPE + 4096 audio code tokens trained via
  `scripts/train_tokenizer.py`.
* **Training pipelines**: text pretraining, long-context extension, audio
  codec pretrain, omni-modal pretrain, code-screenshot pretrain, SFT,
  omni-SFT, DPO, tool-use SFT, **DevOps SFT**, **RLEF (reinforcement
  learning from execution feedback)**, plus LoRA/QLoRA fine-tuning.
  FSDP2 and DeepSpeed-ZeRO-3 supported.
* **Inference**: KV-cache sampler (temperature / top-p / top-k / min-p /
  repetition penalty), multimodal chat with audio in/out + video frame
  sampling, minimal OpenAI-compatible HTTP server.
* **Claude-Code-style runtime** for Forge: agent loop, built-in tools
  (filesystem, shell, http, notebook, **PDF/document parser**,
  **DevOps log + metric tools**), MCP-protocol client, plugin manifests,
  markdown skills (including a ready-to-use `log_triage` skill),
  lifecycle hooks, slash commands (including `/postmortem`), sub-agents.
* **Benchmark runners**: HumanEval, MBPP, LiveCodeBench, MMLU, GSM8K,
  SWE-bench Lite.
* **No trained checkpoints**. The repo is functional code; training is
  the user's responsibility, with full instructions in `docs/training-guide.md`.

## Quick links

* [`docs/architecture-aurora-50b.md`](docs/architecture-aurora-50b.md)
* [`docs/architecture-forge-250b.md`](docs/architecture-forge-250b.md)
* [`docs/training-guide.md`](docs/training-guide.md) — end-to-end how-to-train
* [`docs/plugin-system.md`](docs/plugin-system.md) — Claude-Code-style runtime reference
* [`docs/datasets.md`](docs/datasets.md) — every dataset, license, fetch instructions
* [`docs/benchmarks.md`](docs/benchmarks.md) — how to run each eval
* [`docs/beginner-glossary.md`](docs/beginner-glossary.md) — RoPE, MoE, MLA, LoRA, … in plain English

## Repository layout

```
chatbot/
├── README.md
├── LICENSE                          # Apache-2.0
├── pyproject.toml
├── requirements.txt
├── requirements-train.txt
├── configs/
│   ├── models/                      # tiny, aurora-50b, forge-250b
│   ├── training/                    # pretrain, long_context, sft, dpo, lora, tool-use-sft
│   └── runtime/                     # default, aurora-chat, forge-coder
├── docs/                            # all the architecture / training / plugin docs
├── plugins_examples/                # two ready-to-copy plugins
├── scripts/                         # CLI entrypoints
├── src/chatbot/                     # the Python package itself
│   ├── models/{common, vision, aurora_50b, forge_250b}
│   ├── tokenizer/
│   ├── data/
│   ├── training/
│   ├── inference/
│   ├── runtime/
│   ├── eval/
│   └── utils/
├── tests/
└── legacy/v0-tiny-chatbot/          # the original ~1M-param educational chatbot, preserved
```

## Installation

```powershell
# Windows / PowerShell
git clone https://github.com/<you>/chatbot.git
cd chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
# For training:
pip install -r requirements-train.txt
```

```bash
# Linux / macOS
git clone https://github.com/<you>/chatbot.git
cd chatbot
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-train.txt   # for training
```

Verify the package imports and the tests pass:

```powershell
python -c "import chatbot; print(chatbot.__version__)"
pytest tests/ -q
```

Check that the model configs really hit ~50B / ~250B before you commit to
serious compute:

```powershell
python scripts/count_params.py --model tiny
python scripts/count_params.py --model aurora-50b
python scripts/count_params.py --model forge-250b
```

## Quick smoke test (no GPU required)

A four-command flow that exercises tokenizer training, pretraining, SFT,
and chat — all on the tiny config, on CPU, in minutes:

```powershell
# 1) Tokenizer.
python scripts/train_tokenizer.py `
    --files tests/conftest.py `
    --vocab-size 1024 `
    --output checkpoints/tiny-tok.json

# 2) Pretrain a few steps. (Loss should decrease on a fixed seed.)
python scripts/pretrain.py `
    --model tiny `
    --tokenizer checkpoints/tiny-tok.json `
    max_steps=10 micro_batch_size=2

# 3) SFT a few more steps.
python scripts/sft.py `
    --model tiny `
    --tokenizer checkpoints/tiny-tok.json `
    --resume-from outputs/pretrain/latest `
    max_steps=10 micro_batch_size=2

# 4) Chat with the tiny checkpoint.
python scripts/chat.py `
    --model tiny `
    --checkpoint outputs/sft/latest `
    --tokenizer checkpoints/tiny-tok.json
```

If those four commands succeed end-to-end, the bigger configs will run
end-to-end on a cluster — same code, just bigger numbers in YAML.

## Training the real models

Every stage is one script invocation. See
[`docs/training-guide.md`](docs/training-guide.md) for full
hyperparameter tables, hardware sizing, and the long-context extension
recipe. Headline commands:

```bash
# Aurora-50B
torchrun --nproc-per-node=8 scripts/pretrain.py \
    --model aurora-50b --training pretrain \
    --tokenizer checkpoints/aurora-tokenizer.json

torchrun --nproc-per-node=8 scripts/pretrain.py \
    --model aurora-50b --training long_context \
    --tokenizer checkpoints/aurora-tokenizer.json \
    resume_from=outputs/pretrain/latest

torchrun --nproc-per-node=8 scripts/sft.py    --model aurora-50b --tokenizer ...
torchrun --nproc-per-node=8 scripts/dpo.py    --model aurora-50b --tokenizer ...
```

Forge adds one extra stage (tool-use SFT) at the end. Compute budgets
are very real:

| Model        | Pretrain (full)    | SFT (full)      | LoRA   | QLoRA                |
|--------------|--------------------|-----------------|--------|----------------------|
| Tiny (~50M)  | 1 GPU              | 1 GPU           | 1 GPU  | 1 GPU                |
| Aurora-50B   | 256–1024× H100     | 32–64× H100     | 8× H100| 2–4× A100 / RTX 6000 |
| Forge-250B   | 1024–4096× H100    | 64–128× H100    | 16× H100 | 4–8× H100 (NF4)   |

LoRA / QLoRA paths (in `scripts/lora_finetune.py`) make continuation
training realistic on consumer hardware.

## Chatting with a trained model

```powershell
python scripts/chat.py `
    --model aurora-50b `
    --checkpoint outputs/dpo/latest `
    --tokenizer checkpoints/aurora-tokenizer.json `
    --runtime aurora-chat
```

The runtime config (`configs/runtime/aurora-chat.yaml`) controls system
prompt, temperature, top-p, top-k, min-p, repetition penalty, stop
sequences, max-new-tokens, and context window. Override any leaf from
the CLI:

```powershell
python scripts/chat.py ... runtime.temperature=0.3
```

## Agent mode (Forge)

```powershell
python scripts/agent.py `
    --model forge-250b `
    --checkpoint outputs/tool_use_sft/latest `
    --tokenizer checkpoints/forge-tokenizer.json `
    --runtime forge-coder
```

This launches the Claude-Code-style loop: the model can read the
filesystem, run shell commands (subject to allowlist), make HTTP
requests, execute Python notebook cells, and call any plugin or MCP
server discovered under `~/.chatbot/plugins/`.

Plugin authoring guide: [`docs/plugin-system.md`](docs/plugin-system.md).
Working examples: [`plugins_examples/`](plugins_examples/).

## Evaluation

```powershell
python scripts/eval.py `
    --bench humaneval `
    --model forge-250b `
    --checkpoint outputs/dpo/latest `
    --tokenizer checkpoints/forge-tokenizer.json `
    --limit 164
```

See [`docs/benchmarks.md`](docs/benchmarks.md) for the full benchmark
list and instructions for SWE-bench Lite (which needs a Docker-backed
sandbox per problem — we ship the patch-extraction half; you supply the
runner).

## Inspirations

Architectures are original, but the design ideas are openly inspired by:

* Llama 3.1 / 3.2-Vision, Gemma 3, Mistral — dense + SwiGLU + GQA + RoPE.
* SigLIP / SigLIP2 — vision encoder pattern.
* DeepSeek-V3 — fine-grained MoE, shared experts, AL-free balancing, MLA.
* Qwen3-Coder — long-context, repo-level packed training.
* Claude Code — plugin / skill / hook / slash-command / MCP runtime layer.

The code in this repository is our own. We use PyTorch (framework),
`tokenizers` (BPE library), `datasets` (data loading), and
`bitsandbytes` (quantization), but the model architectures themselves are
implemented from scratch.

## License

Apache-2.0, see [LICENSE](LICENSE). The model code is licensed as the
project's own work; *trained weights* (if you produce any) are governed by
the licenses of the datasets you trained on — see
[`docs/datasets.md`](docs/datasets.md) for license metadata on every
dataset entry in the registry.

## FAQ

**Q: Do you ship checkpoints?**
No. The repo is functional code. Training the real 50B / 250B models is
a compute commitment we don't make on your behalf — but you can do it
yourself with the scripts here, with the exact recipes documented in
[`docs/training-guide.md`](docs/training-guide.md).

**Q: Can I run the 50B / 250B models on a single GPU?**
For full inference: no. The KV cache alone at full context is tens of
gigabytes. For *fine-tuning*: yes, via QLoRA on the right hardware (see
the table above). For learning / experimentation: use the `tiny` config.

**Q: Why not `transformers`?**
We deliberately implement architectures from scratch so the code is
readable end-to-end. `transformers` is great for production, but it
hides the moving parts that this repo is designed to teach.

**Q: Do these models actually beat Claude / GPT?**
The repo is the model code, not the result of training it. We're not
claiming the resulting checkpoints will match production models — those
took thousands of GPU-years. We are claiming the architectures and
recipes are sound and faithfully reproduce the ideas behind those models.

## Help and contributing

Open an issue if anything is unclear; PRs welcome. The codebase is
deliberately small (about 5K lines of model / runtime code) so it's easy
to read and modify.
