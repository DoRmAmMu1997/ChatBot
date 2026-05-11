"""A tiny word-level tokenizer for the chatbot.

This project intentionally avoids a heavy tokenizer dependency. The tokenizer
below is simple enough for beginners to read: it lowercases text, splits words
and punctuation, and keeps a fixed vocabulary learned from the training data.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
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

    ascii_text = unicodedata.normalize("NFKD", text)
    ascii_text = ascii_text.encode("ascii", "ignore").decode("ascii")
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

        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, int]]) -> "SimpleTokenizer":
        """Load a tokenizer from a saved checkpoint."""

        return cls(data["token_to_id"])


def tokenize(text: str) -> List[str]:
    """Split text into tokens using the project regex."""

    return TOKEN_PATTERN.findall(normalize_text(text))
