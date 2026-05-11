"""Byte-level BPE tokenizer wrapper.

We wrap the Hugging Face `tokenizers` library — it's a small, fast,
well-tested C++ implementation that we don't need to rebuild from scratch.
The model architecture is ours; the tokenizer is just plumbing.

The wrapper:
    * registers our project-specific special tokens (chat roles, image/tool
      markers),
    * exposes a minimal interface (`encode`, `decode`, `save`, `from_file`)
      so the rest of the codebase doesn't depend on `tokenizers` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


@dataclass
class SpecialTokens:
    """The set of special markers every model uses (or ignores)."""

    pad: str = "<|pad|>"
    bos: str = "<|bos|>"
    eos: str = "<|eos|>"
    unk: str = "<|unk|>"
    # Chat role markers.
    system: str = "<|system|>"
    user: str = "<|user|>"
    assistant: str = "<|assistant|>"
    # Vision (Aurora only).
    image: str = "<|image|>"
    image_start: str = "<|image_start|>"
    image_end: str = "<|image_end|>"
    # Tool calling (Forge).
    tool_call: str = "<|tool_call|>"
    tool_call_end: str = "<|/tool_call|>"
    tool_result: str = "<|tool_result|>"
    tool_result_end: str = "<|/tool_result|>"

    extra: List[str] = field(default_factory=list)

    def all(self) -> List[str]:
        return [
            self.pad, self.bos, self.eos, self.unk,
            self.system, self.user, self.assistant,
            self.image, self.image_start, self.image_end,
            self.tool_call, self.tool_call_end,
            self.tool_result, self.tool_result_end,
            *self.extra,
        ]


class BPETokenizer:
    """Thin wrapper around a Hugging Face byte-level BPE tokenizer."""

    def __init__(self, hf_tokenizer: Tokenizer, specials: SpecialTokens):
        self._tk = hf_tokenizer
        self.specials = specials

    # -------- factories --------

    @classmethod
    def train(
        cls,
        files: Sequence[str],
        *,
        vocab_size: int,
        specials: Optional[SpecialTokens] = None,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train a fresh byte-level BPE tokenizer from text files."""

        specials = specials or SpecialTokens()
        tk = Tokenizer(models.BPE(unk_token=specials.unk))
        # Byte-level pre-tokenization: every byte is a starting unit. Same
        # approach as GPT-2 / Llama 3. Robust against arbitrary unicode.
        tk.normalizer = normalizers.NFKC()
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=specials.all(),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tk.train(files=list(files), trainer=trainer)
        return cls(tk, specials)

    @classmethod
    def from_file(cls, path: str | Path, specials: Optional[SpecialTokens] = None) -> "BPETokenizer":
        specials = specials or SpecialTokens()
        tk = Tokenizer.from_file(str(path))
        return cls(tk, specials)

    # -------- interface --------

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tk.token_to_id(token)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        result = self._tk.encode(text, add_special_tokens=add_special_tokens)
        return list(result.ids)

    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return [list(e.ids) for e in self._tk.encode_batch(list(texts))]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self._tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tk.save(str(path))

    # -------- convenience: special-token ids --------

    def pad_id(self) -> int:
        return self.token_to_id(self.specials.pad) or 0

    def bos_id(self) -> int:
        return self.token_to_id(self.specials.bos) or 1

    def eos_id(self) -> int:
        return self.token_to_id(self.specials.eos) or 2
