# ChatBot

ChatBot is a small LLM-style chatbot project built with PyTorch. It uses a
decoder-only Transformer trained with next-token prediction, which is the same
basic idea used by larger language models.

## What changed

- The model is a compact GPT-like Transformer.
- Cornell Movie Dialogues still works as the default offline dataset.
- DailyDialog is available as an optional cleaner conversational dataset when
  the `datasets` package can download it from Hugging Face.
- The project is modular: data loading, tokenization, modeling, training, and
  chatting live in separate files.
- Previous experimental artifacts were removed so the repository now points
  clearly at the upgraded architecture.

## Repository structure

```text
.
|-- chatbot.py
|-- chat_llm.py
|-- train_llm.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- cornell movie-dialogs corpus/
|   |   |-- movie_lines.txt
|   |   |-- movie_conversations.txt
|   |   |-- movie_titles_metadata.txt
|   |   |-- movie_characters_metadata.txt
|   |   |-- raw_script_urls.txt
|   |   |-- chameleons.pdf
|   |   `-- README.txt
`-- src/
    `-- chatbot/
        |-- __init__.py
        |-- chat.py
        |-- config.py
        |-- data.py
        |-- model.py
        |-- tokenizer.py
        `-- train.py
```

### Top-level scripts

`train_llm.py` starts training. It is intentionally tiny because the real
training logic lives in `src/chatbot/train.py`.

`chatbot.py` and `chat_llm.py` both start an interactive chat session from a
trained checkpoint. `chatbot.py` is kept as the familiar main entrypoint.

### `src/chatbot/config.py`

Contains the `ModelConfig` dataclass. This stores model size settings such as
embedding size, number of Transformer layers, number of attention heads, and
maximum context length.

### `src/chatbot/data.py`

Loads training examples.

- `load_cornell_pairs()` reads Cornell files already stored under `data/`.
- `load_dailydialog_pairs()` optionally downloads DailyDialog through Hugging
  Face datasets.
- `ConversationDataset` converts text into `(input_tokens, target_tokens)` pairs
  for next-token prediction.

### `src/chatbot/tokenizer.py`

Implements a small word-level tokenizer. It lowercases text, separates words and
punctuation, keeps special tokens like `<user>` and `<bot>`, and maps uncommon
words to `<unk>`.

This is simpler than a production BPE tokenizer, but it keeps the project easy
to understand and removes the need for extra tokenizer files.

### `src/chatbot/model.py`

Defines `TransformerChatModel`, the new small LLM. It uses:

- token embeddings
- positional embeddings
- causal self-attention through Transformer encoder layers
- a language-model head that predicts the next token

The causal mask is what makes it LLM-like: each token can only look backward at
earlier tokens, never forward at the answer it is supposed to predict.

### `src/chatbot/train.py`

Handles the full training loop:

- reads the selected dataset
- builds the tokenizer
- creates train/validation splits
- trains the Transformer
- saves a checkpoint containing model weights, tokenizer vocabulary, config, and
  basic metrics

### `src/chatbot/chat.py`

Loads a checkpoint and runs terminal inference. User messages are formatted as:

```text
<bos> <user> your message <bot>
```

The model then generates the bot side until it reaches a stop token or the
maximum response length.

## Setup

Create and activate a virtual environment if you want one, then install the
dependencies:

```powershell
pip install -r requirements.txt
```

`torch` is required. `datasets` is only required for DailyDialog.

## Training

Train on the bundled Cornell Movie Dialogues data:

```powershell
python train_llm.py --dataset cornell --steps 2000
```

For a quick CPU smoke run:

```powershell
python train_llm.py --dataset cornell --max-pairs 256 --steps 5 --batch-size 16 --cpu
```

Train with the optional DailyDialog dataset:

```powershell
python train_llm.py --dataset dailydialog --steps 2000
```

Useful knobs:

- `--max-pairs`: limit examples for experiments
- `--block-size`: maximum token context length
- `--n-layer`: number of Transformer layers
- `--n-head`: number of attention heads
- `--n-embd`: embedding size
- `--steps`: number of optimizer steps
- `--batch-size`: training batch size

The default checkpoint path is:

```text
checkpoints/chatbot-small-llm.pt
```

## Chatting

After training, start a chat session:

```powershell
python chatbot.py --checkpoint checkpoints/chatbot-small-llm.pt
```

You can also use:

```powershell
python chat_llm.py --checkpoint checkpoints/chatbot-small-llm.pt
```

Type `quit`, `q`, or `exit` to stop.

## Dataset notes

Cornell Movie Dialogues is large and already present in this repository, so it
is the most reliable default. It contains movie dialogue, which can be dramatic,
noisy, or old-fashioned.

DailyDialog is usually better for simple everyday conversation because it is
made of short multi-turn daily-life dialogues. It is optional because it needs
an internet download and the Hugging Face `datasets` package.

## Model notes

This is a small educational LLM, not a production assistant. It learns from the
dataset you train it on and does not include instruction tuning, safety tuning,
retrieval, or external knowledge. Better responses usually require:

- more training steps
- a cleaner dataset
- a larger model
- a subword tokenizer
- validation examples that match the chat style you want

## Suggested next upgrades

- Add a byte-pair encoding tokenizer for better handling of rare words.
- Add perplexity tracking over a held-out validation file.
- Add beam search or nucleus sampling controls for inference.
- Add a small test suite for data loading, tokenization, and checkpoint loading.
