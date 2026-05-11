"""Configuration objects used by training and inference.

Dataclasses keep the settings in one readable place. They are also easy to
save into a checkpoint because they can be converted to plain dictionaries.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Size settings for the small decoder-only Transformer."""

    vocab_size: int
    block_size: int = 96
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    pad_token_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly copy of the configuration."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Build a config from the dictionary stored in a checkpoint."""

        return cls(**data)
