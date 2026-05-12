"""Forge (~464B total / ~35.5B active): MoE coder + DevOps assistant.

Multi-head Latent Attention (MLA) lets the 1M-token context fit in
memory; fine-grained MoE keeps the active per-token cost manageable; a
small vision tower handles code screenshots; an audio encoder handles
voice descriptions. The package directory keeps its historical
``forge_250b`` name for backwards-compatible imports.
"""

from .config import ForgeConfig, forge_config_from_yaml
from .model import ForgeForCausalLM

__all__ = ["ForgeConfig", "forge_config_from_yaml", "ForgeForCausalLM"]
