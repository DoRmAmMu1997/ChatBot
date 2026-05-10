"""Configuration objects used by training and inference.

The project now has two model scales:

* tiny configs that can run in tests or on a laptop, and
* the original untrained ChatBot-10B config used for real training plans.

Keeping both in one dataclass makes the large model easy to describe without
ever allocating its weights during normal tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Settings for the original decoder-only ChatBot architecture.

    The names intentionally stay close to common LLM papers and configs:
    ``n_embd`` is the hidden size, ``n_head`` is the number of query heads, and
    ``n_kv_head`` is the smaller number of key/value heads used by grouped-query
    attention. Beginners can read this file as the model's blueprint.
    """

    vocab_size: int
    block_size: int = 96
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    n_kv_head: int | None = None
    ffn_hidden_size: int | None = None
    dropout: float = 0.1
    attention_dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = 0
    tie_embeddings: bool = True
    use_bias: bool = False
    model_name: str = "chatbot-small"

    def __post_init__(self) -> None:
        """Fill derived defaults and validate shape choices."""

        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.ffn_hidden_size is None:
            # SwiGLU uses three projections, so the hidden size can be smaller
            # than the classic Transformer 4x MLP while still being expressive.
            self.ffn_hidden_size = self.n_embd * 4
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head for GQA.")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE.")

    @property
    def head_dim(self) -> int:
        """Number of hidden units handled by one attention head."""

        return self.n_embd // self.n_head

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly copy of the configuration."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Build a config from the dictionary stored in a checkpoint."""

        # Older checkpoints do not have the newer architecture fields. The
        # dataclass defaults fill them in, so old tiny checkpoints still load.
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "ModelConfig":
        """Load a model config from a YAML file."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised by users
            raise ImportError("Reading YAML configs requires PyYAML.") from exc

        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        return cls.from_dict(data)


def chatbot_10b_config() -> ModelConfig:
    """Return the original untrained ChatBot-10B blueprint.

    This config is intentionally dense, not MoE, because the user wanted the
    10B model to be ours and ours only. The parameter count is approximately
    9.999B when the input embedding and output head are tied.
    """

    return ModelConfig(
        model_name="chatbot-10b",
        vocab_size=128000,
        block_size=4096,
        n_embd=5120,
        n_head=40,
        n_kv_head=8,
        n_layer=36,
        ffn_hidden_size=12800,
        dropout=0.0,
        attention_dropout=0.0,
        rope_theta=1000000.0,
        norm_eps=1e-5,
        initializer_range=0.02,
        tie_embeddings=True,
        use_bias=False,
    )


def tiny_config(vocab_size: int = 128) -> ModelConfig:
    """Return a tiny config used by tests and CPU smoke runs."""

    return ModelConfig(
        model_name="chatbot-tiny",
        vocab_size=vocab_size,
        block_size=32,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        ffn_hidden_size=128,
        dropout=0.0,
        attention_dropout=0.0,
    )
