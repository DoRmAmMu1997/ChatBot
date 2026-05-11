"""Aurora (~72B): dense omni-modal decoder-only Transformer.

Text + image + audio in, text + audio out. The package directory keeps
its historical ``aurora_50b`` name so existing imports don't break; the
config and the docs use the current size.
"""

from .config import AuroraConfig, aurora_config_from_yaml
from .model import AuroraForCausalLM

__all__ = ["AuroraConfig", "aurora_config_from_yaml", "AuroraForCausalLM"]
