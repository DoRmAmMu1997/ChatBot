# ChatBot

ChatBot is an educational LLM project built with PyTorch. The repository now
contains two related paths:

- a tiny local model path for learning, tests, and CPU smoke runs
- an original untrained `ChatBot-10B` architecture for real large-scale training

The 10B model is not a fine-tune of Qwen, Gemma, DeepSeek, gpt-oss, or any other
external model. Those projects are used only as architectural inspiration for
modern decoder design choices such as RoPE, RMSNorm, SwiGLU, grouped-query
attention, KV caching, tied embeddings, and bf16-friendly training.

## What changed

- Added an original dense `ChatBot-10B` config at about `9.999B` parameters.
- Replaced the previous simple Transformer internals with modular decoder
  blocks: RMSNorm, RoPE, grouped-query attention, SwiGLU, and tied LM head.
- Added BPE tokenizer training through Hugging Face `tokenizers`.
- Added validation perplexity, top-p sampling, greedy decoding, beam search, and
  repetition penalty controls.
- Added dataset recipes for Cornell, DailyDialog, UltraChat, OpenAssistant
  OASST1, and Dolly 15k.
- Added tests and GitHub Actions CI for the tiny model path and parameter-count
  checks.

## Repository structure

```text
.
|-- chatbot.py
|-- chat_llm.py
|-- train_llm.py
|-- train_tokenizer.py
|-- train_10b.py
|-- configs/
|   |-- chatbot-10b.yaml
|   `-- chatbot-tiny.yaml
|-- scripts/
|   `-- estimate_params.py
|-- data/
|   |-- dataset_manifest.json
|   `-- cornell movie-dialogs corpus/
|-- tests/
|   |-- fixtures/
|   `-- test_*.py
`-- src/
    `-- chatbot/
        |-- chat.py
        |-- config.py
        |-- data.py
        |-- model.py
        |-- tokenizer.py
        |-- tokenizer_train.py
        |-- train.py
        `-- train_10b.py
```

## Model notes

`configs/chatbot-10b.yaml` defines the original 10B blueprint:

```text
vocab_size: 128000
n_layer: 36
n_embd: 5120
n_head: 40
n_kv_head: 8
ffn_hidden_size: 12800
block_size: 4096
```

With tied input/output embeddings, this reports about `9.999B` parameters:

```powershell
python scripts/estimate_params.py --config configs/chatbot-10b.yaml
```

Do not commit 10B checkpoints, random initialized weights, adapters, tokenizer
outputs, or training runs. They are intentionally ignored by `.gitignore`.

## Setup

```powershell
pip install -r requirements.txt
```

For full 10B training, use a Linux multi-GPU environment with recent PyTorch,
bf16-capable GPUs, and a distributed training launcher such as FSDP or DeepSpeed.
This repository provides the model and data pipeline, but ordinary laptops are
not expected to train the 10B config.

## Tiny local training

Use the tiny config to verify the code path on CPU:

```powershell
python train_llm.py --dataset cornell --max-pairs 64 --steps 5 --batch-size 4 --config configs/chatbot-tiny.yaml --cpu
```

The checkpoint stores model config, tokenizer metadata, train args, metrics, and
model weights.

## BPE tokenizer

Train a BPE tokenizer from the configured data mix:

```powershell
python train_tokenizer.py --dataset mixed --max-pairs 50000 --vocab-size 128000 --output tokenizers/chatbot-bpe.json
```

For tiny experiments, lower `--vocab-size` and `--max-pairs`.

## ChatBot-10B training

After creating a BPE tokenizer, launch training with the 10B config:

```powershell
python train_10b.py --config configs/chatbot-10b.yaml --tokenizer bpe --tokenizer-path tokenizers/chatbot-bpe.json --dataset mixed
```

For real training, run the same entrypoint through your distributed launcher and
set batch size, gradient accumulation, precision, checkpointing, and output
locations for the target cluster. The full datasets are downloaded during
training; they are not stored in this repository.

## Dataset notes

The repo keeps Cornell Movie Dialogues bundled for offline experiments. The
other recipes are downloaded through Hugging Face `datasets` when requested:

- `OpenRL/daily_dialog`
- `HuggingFaceH4/ultrachat_200k`
- `OpenAssistant/oasst1`
- `databricks/databricks-dolly-15k`

Always review each dataset license and terms before training or publishing
weights. The test suite uses tiny synthetic fixtures, not full external data.

## Chatting

After training a checkpoint:

```powershell
python chatbot.py --checkpoint checkpoints/chatbot-small-llm.pt --temperature 0.8 --top-p 0.9
```

Useful inference controls:

- `--greedy`: always choose the highest-scoring token
- `--top-k`: keep only the top k tokens before sampling
- `--top-p`: nucleus sampling threshold
- `--num-beams`: beam search width
- `--repetition-penalty`: reduce repeated tokens

## Tests

```powershell
python -m compileall chatbot.py chat_llm.py train_llm.py train_tokenizer.py train_10b.py src scripts
pytest
python scripts/estimate_params.py --config configs/chatbot-10b.yaml
```

CI runs these checks on Python 3.11 and verifies the tiny CPU training path.
