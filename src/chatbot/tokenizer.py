"""Tokenizers for the chatbot.

The original project used only ``SimpleTokenizer`` because it is easy to read.
That tokenizer stays available for beginners and tiny experiments. The 10B
training path adds ``BPETokenizer``, a byte-pair encoding tokenizer like the
subword tokenizers used by real LLM training pipelines.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


# Special tokens are not normal words. They give the model structure:
# <pad> fills short examples, <unk> stands for unknown words, <bos>/<eos> mark
# sequence boundaries, and <user>/<bot> mark who is speaking.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
USER_TOKEN = "<user>"
BOT_TOKEN = "<bot>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    BOT_TOKEN,
]

# This regex keeps special tokens like <user>, normal words, numbers, and
# punctuation as separate tokens.
TOKEN_PATTERN = re.compile(r"<[^>\s]+>|[a-zA-Z]+(?:'[a-z]+)?|[0-9]+|[^\s\w]")


def normalize_text(text: str) -> str:
    """Convert text into a clean, lowercase ASCII form."""

    # NFKD separates accents from letters. The ASCII encode/decode step then
    # drops those accents so accented and unaccented words behave consistently.
    ascii_text = unicodedata.normalize("NFKD", text)
    ascii_text = ascii_text.encode("ascii", "ignore").decode("ascii")

    # Lowercasing keeps the beginner tokenizer small: "Hello" and "hello"
    # become the same token instead of two separate vocabulary entries.
    ascii_text = ascii_text.lower().strip()
    return re.sub(r"\s+", " ", ascii_text)


class SimpleTokenizer:
    """A small vocabulary tokenizer used by the Transformer model.

    A tokenizer is the bridge between text and tensors. It turns strings into
    integer ids for the model, and later turns generated ids back into text.
    """

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = dict(token_to_id)

        # The reverse lookup is needed during chat, when model output ids need
        # to become readable words again.
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.user_id = self.token_to_id[USER_TOKEN]
        self.bot_id = self.token_to_id[BOT_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Number of tokens the model can predict."""

        return len(self.token_to_id)

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        max_vocab_size: int = 12000,
        min_freq: int = 2,
    ) -> "SimpleTokenizer":
        """Learn a vocabulary from training texts.

        Common tokens are kept first. Rare tokens are mapped to <unk> during
        training and chatting, which keeps the model small and fast.
        """

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        # Special tokens are added first so their ids are stable and easy to
        # find later. Normal vocabulary starts after them.
        token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        room_left = max_vocab_size - len(token_to_id)

        # Keep the most common tokens up to max_vocab_size. min_freq filters out
        # words that appear too rarely to learn useful patterns from.
        for token, count in counter.most_common(max(room_left, 0)):
            if count < min_freq:
                continue
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        return cls(token_to_id)

    def encode(self, text: str) -> List[int]:
        """Turn text into token ids."""

        # Any token missing from the learned vocabulary becomes <unk>. This
        # prevents crashes when the user types a word the model never saw.
        return [self.token_to_id.get(token, self.unk_id) for token in tokenize(text)]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Turn token ids back into readable text."""

        tokens: List[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), UNK_TOKEN)
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)

        text = " ".join(tokens)

        # The tokenizer separates punctuation as its own token. These two small
        # cleanup rules make decoded text look natural again.
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\s+'", "'", text)
        return text.strip()

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Return a checkpoint-friendly representation."""

        return {"kind": "simple", "token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, int]]) -> "SimpleTokenizer":
        """Load a tokenizer from a saved checkpoint."""

        return cls(data["token_to_id"])


def tokenize(text: str) -> List[str]:
    """Split text into tokens using the project regex."""

    return TOKEN_PATTERN.findall(normalize_text(text))


class BPETokenizer:
    """Byte-pair encoding tokenizer backed by Hugging Face ``tokenizers``.

    BPE learns common chunks of text instead of whole words. That matters for a
    large model because rare words, names, and typos can be represented as
    smaller pieces instead of becoming one useless ``<unk>`` token.
    """

    kind = "bpe"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._refresh_special_ids()

    def _refresh_special_ids(self) -> None:
        # Store special token ids as attributes so the chat code can stop on
        # <eos>, ignore <pad>, and recognize speaker markers quickly.
        self.pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
        self.unk_id = self.tokenizer.token_to_id(UNK_TOKEN)
        self.bos_id = self.tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self.user_id = self.tokenizer.token_to_id(USER_TOKEN)
        self.bot_id = self.tokenizer.token_to_id(BOT_TOKEN)

    @property
    def token_to_id(self) -> Dict[str, int]:
        """Expose a simple mapping for code shared with ``SimpleTokenizer``."""

        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self) -> int:
        """Number of BPE tokens available to the model."""

        return self.tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train a BPE tokenizer from an iterator of text strings."""

        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as exc:  # pragma: no cover - exercised by users
            raise ImportError("BPE tokenization requires the 'tokenizers' package.") from exc

        # The BPE model starts with bytes and repeatedly learns useful merges
        # like "ing" or common word pieces from the training corpus.
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))

        # ByteLevel keeps the tokenizer robust: any text can be represented,
        # even if it contains unusual characters or rare names.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        # The trainer decides vocabulary size and guarantees our special tokens
        # are present before normal learned tokens.
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    def encode(self, text: str) -> List[int]:
        """Turn text into BPE token ids."""

        return self.tokenizer.encode(text).ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Turn BPE token ids back into text."""

        return self.tokenizer.decode(list(map(int, ids)), skip_special_tokens=skip_special).strip()

    def save(self, path: str | Path) -> None:
        """Save the tokenizer JSON that future model runs must reuse."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised by users
            raise ImportError("Loading a BPE tokenizer requires the 'tokenizers' package.") from exc

        return cls(Tokenizer.from_file(str(path)))

    def to_dict(self) -> Dict[str, str]:
        """Store the tokenizer JSON inside a checkpoint if needed."""

        return {"kind": "bpe", "tokenizer_json": self.tokenizer.to_str()}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised by users
            raise ImportError("Loading a BPE tokenizer requires the 'tokenizers' package.") from exc

        return cls(Tokenizer.from_str(data["tokenizer_json"]))


def tokenizer_from_dict(data):
    """Load either tokenizer type from checkpoint metadata."""

    kind = data.get("kind", "simple")

    # Checkpoints can come from older simple-tokenizer runs or newer BPE runs.
    # Dispatching here keeps chat.py independent of tokenizer details.
    if kind == "bpe":
        return BPETokenizer.from_dict(data)
    if kind == "simple":
        return SimpleTokenizer.from_dict(data)
    raise ValueError(f"Unknown tokenizer kind: {kind}")


def tokenizer_metadata(tokenizer, tokenizer_path: str | None = None) -> Dict[str, str]:
    """Return checkpoint metadata for a tokenizer.

    When a tokenizer file path is supplied, we store the path for clarity and
    also embed the tokenizer payload so the checkpoint remains self-contained.
    """

    data = tokenizer.to_dict()
    if tokenizer_path:
        data["path"] = tokenizer_path
    return data


def write_tokenizer_manifest(path: str | Path, tokenizer_path: str, vocab_size: int) -> None:
    """Write a tiny JSON manifest describing a trained tokenizer."""

    manifest = {
        "kind": "bpe",
        "tokenizer_path": tokenizer_path,
        "vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
