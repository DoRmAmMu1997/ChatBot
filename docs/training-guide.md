# Training guide

Step-by-step instructions for taking either model from "no weights" to
"useful checkpoint". The repo ships **no checkpoints** — every weight
file is produced on your hardware by running the stages below.

## Training stages at a glance

Aurora (omni-modal):

1. Tokenizer training
2. Audio codec autoencoder pretrain
3. Text pretrain (short context, 8K)
4. Long-context extension (→ 256K)
5. Omni-modal pretrain (text + image + audio interleaved)
6. SFT
7. Omni-SFT (voice-instruction tuning)
8. DPO

Forge (coder + DevOps):

1. Tokenizer training
2. Audio codec autoencoder pretrain (shared with Aurora)
3. Text pretrain (short context)
4. Long-context extension (→ 1M)
5. Code-screenshot pretrain (vision tower warm-up)
6. SFT
7. DPO
8. Tool-use SFT
9. DevOps SFT (logs / incidents)
10. RLEF (reinforcement learning from execution feedback)

The same scripts under `scripts/` drive each stage; the config file is
what changes between them.

## Step 0 — set up your environment

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .                   # install chatbot in editable mode
pip install -r requirements-train.txt    # heavy training deps (flash-attn skipped on Windows)
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-train.txt
```

Verify everything imports:

```powershell
python -c "import chatbot; print(chatbot.__version__)"
pytest tests/ -q
```

Then sanity-check the model configs hit their target parameter counts:

```powershell
python scripts/count_params.py --model tiny
python scripts/count_params.py --model aurora-72b
python scripts/count_params.py --model forge-460b
```

Building the 250B graph on CPU takes a few minutes but only happens once.

## Step 1 — train a tokenizer

Pretraining needs a tokenizer. We train one with the HF `tokenizers`
library on a representative text corpus.

```powershell
python scripts/train_tokenizer.py `
    --files data/raw/sample-corpus/*.txt `
    --vocab-size 131072 `
    --output checkpoints/aurora-tokenizer.json
```

For Forge use `--vocab-size 200032` and feed code-heavy files in
`--files`.

## Step 2 — pretrain (short context)

Single-GPU dry run on the tiny config (great for verifying everything
works end-to-end before you rent H100s):

```powershell
python scripts/pretrain.py `
    --model tiny `
    --training pretrain `
    --tokenizer checkpoints/aurora-tokenizer.json `
    max_steps=50 micro_batch_size=2
```

Real Aurora-72B pretraining run (multi-node):

```bash
torchrun --nproc-per-node=8 --nnodes=$NNODES --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:29500 \
    scripts/pretrain.py \
    --model aurora-72b \
    --training pretrain \
    --tokenizer checkpoints/aurora-tokenizer.json
```

Hardware expectations:

| Model        | Pretrain (full)    | SFT (full)      | LoRA  | QLoRA                     |
|--------------|--------------------|-----------------|-------|---------------------------|
| Tiny (50M)   | 1 GPU              | 1 GPU           | 1 GPU | 1 GPU                     |
| Aurora-72B   | 256–1024× H100     | 32–64× H100     | 8× H100 | 2–4× A100 / RTX 6000     |
| Forge-460B   | 1024–4096× H100    | 64–128× H100    | 16× H100 | 4–8× H100 (NF4)         |

(Hours/days depending on cluster size — that's compute, not code, so we
don't try to estimate it precisely.)

## Step 3 — long-context extension (mid-training)

After pretrain, raise the context window with YaRN and continue training
on long packed documents:

```bash
torchrun --nproc-per-node=8 scripts/pretrain.py \
    --model aurora-72b \
    --training long_context \
    --tokenizer checkpoints/aurora-tokenizer.json \
    resume_from=outputs/pretrain/latest
```

Run this in stages: 8K → 32K → 128K → 256K (Aurora) or
8K → 32K → 256K → 1M (Forge). Each stage gets its own checkpoint.

## Step 4 — SFT

```bash
torchrun --nproc-per-node=8 scripts/sft.py \
    --model aurora-72b \
    --training sft \
    --tokenizer checkpoints/aurora-tokenizer.json \
    --resume-from outputs/long_context/latest
```

## Step 5 — DPO

```bash
torchrun --nproc-per-node=8 scripts/dpo.py \
    --model aurora-72b \
    --training dpo \
    --tokenizer checkpoints/aurora-tokenizer.json \
    --resume-from outputs/sft/latest
```

## Step 6 — tool-use SFT (Forge only)

```bash
torchrun --nproc-per-node=8 scripts/sft.py \
    --model forge-460b \
    --training tool-use-sft \
    --tokenizer checkpoints/forge-tokenizer.json \
    --resume-from outputs/dpo/latest
```

(Uses the SFT script with the tool-use config because the loss is the
same — only the data and masking differ.)

## LoRA / QLoRA fine-tunes

LoRA is the realistic option for end-users with modest hardware:

```powershell
python scripts/lora_finetune.py `
    --model aurora-72b `
    --training lora `
    --tokenizer checkpoints/aurora-tokenizer.json `
    --base-checkpoint outputs/sft/latest
```

For QLoRA add `qlora.enabled=true` and make sure `bitsandbytes` is
installed. A 50B QLoRA fine-tune typically fits on a single 48 GB GPU; a
250B QLoRA fine-tune needs 4-8× 80 GB GPUs.

## Smoke test the whole pipeline

A four-command flow to make sure every code path executes on the tiny
config — no internet required:

```powershell
python scripts/train_tokenizer.py --files tests/conftest.py --vocab-size 1024 --output checkpoints/tiny-tok.json
python scripts/pretrain.py       --model tiny --tokenizer checkpoints/tiny-tok.json max_steps=10 micro_batch_size=2
python scripts/sft.py            --model tiny --tokenizer checkpoints/tiny-tok.json max_steps=10 --resume-from outputs/pretrain/latest
python scripts/chat.py           --model tiny --checkpoint outputs/sft/latest --tokenizer checkpoints/tiny-tok.json
```

If that sequence runs end-to-end on a laptop, the bigger configs will
run end-to-end on a GPU cluster — same code, just bigger numbers in YAML.
