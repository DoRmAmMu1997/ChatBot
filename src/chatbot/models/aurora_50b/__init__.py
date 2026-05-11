"""Aurora-50B: dense multimodal (text + images) decoder-only Transformer."""

from .config import AuroraConfig, aurora_config_from_yaml
from .model import AuroraForCausalLM

__all__ = ["AuroraConfig", "aurora_config_from_yaml", "AuroraForCausalLM"]
