"""Small LLM chatbot package."""

from .config import ModelConfig, chatbot_10b_config, tiny_config
from .params import estimate_parameter_count
from .tokenizer import BPETokenizer, SimpleTokenizer

__all__ = [
    "BPETokenizer",
    "ModelConfig",
    "SimpleTokenizer",
    "TransformerChatModel",
    "chatbot_10b_config",
    "estimate_parameter_count",
    "tiny_config",
]


def __getattr__(name):
    """Load PyTorch model objects only when they are requested."""

    if name == "TransformerChatModel":
        from .model import TransformerChatModel

        return TransformerChatModel
    raise AttributeError(name)
