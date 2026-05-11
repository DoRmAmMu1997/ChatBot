"""Config loader.

Every YAML config in ``configs/`` is parsed into an OmegaConf ``DictConfig``.
A small extension we add on top: a ``defaults:`` key listing other configs
to merge in first. This mimics Hydra's defaults system but without pulling
the whole of Hydra in (we only need a tiny slice of its behavior).

Example::

    # configs/runtime/forge-coder.yaml
    defaults:
      - default          # → merges configs/runtime/default.yaml first
    temperature: 0.3

The function ``load_config`` resolves the path, walks any ``defaults`` chain,
merges layer by layer, and applies CLI overrides at the end.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf


# Where the repo keeps its YAML configs. We resolve relative to the package
# install so the loader works whether the user runs scripts from the repo
# root or invokes the installed CLI from somewhere else.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIGS_DIR = _PACKAGE_ROOT / "configs"


def _resolve_config_path(name: str, base_dir: Path) -> Path:
    """Find a config file by short name or path.

    A ``name`` can be:
    * an absolute or relative path (``./my.yaml`` or ``/etc/x.yaml``),
    * a path relative to ``configs/`` (``models/tiny`` → ``configs/models/tiny.yaml``),
    * a bare name within ``base_dir`` (``default`` in ``configs/runtime/``).
    """

    if name.endswith(".yaml") or name.endswith(".yml"):
        candidate = Path(name)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if (Path.cwd() / candidate).exists():
            return Path.cwd() / candidate
        # fall through to the package configs search
        if (DEFAULT_CONFIGS_DIR / candidate).exists():
            return DEFAULT_CONFIGS_DIR / candidate

    # Try adding `.yaml`.
    for suffix in (".yaml", ".yml"):
        local = base_dir / f"{name}{suffix}"
        if local.exists():
            return local
        rooted = DEFAULT_CONFIGS_DIR / f"{name}{suffix}"
        if rooted.exists():
            return rooted

    raise FileNotFoundError(
        f"Could not resolve config '{name}'. Looked in {base_dir} and {DEFAULT_CONFIGS_DIR}."
    )


def load_config(name_or_path: str, *, base_dir: Optional[Path] = None) -> DictConfig:
    """Load a YAML config, resolving any ``defaults:`` chain.

    The returned ``DictConfig`` is a normal OmegaConf object: dot-access,
    interpolations, structured merges, etc.
    """

    base_dir = base_dir or DEFAULT_CONFIGS_DIR
    path = _resolve_config_path(name_or_path, base_dir)

    # Parent dir of this file is where its `defaults:` siblings live.
    own_base = path.parent

    cfg = OmegaConf.load(path)
    defaults: Sequence[str] = []
    if isinstance(cfg, DictConfig) and "defaults" in cfg:
        defaults = list(cfg.pop("defaults"))

    # Resolve defaults in order, layering them under the current config.
    merged: DictConfig = OmegaConf.create({})
    for default_name in defaults:
        merged = merge_configs(merged, load_config(default_name, base_dir=own_base))
    merged = merge_configs(merged, cfg)
    return merged


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Right-most config wins (deep merge), same semantics as OmegaConf.merge."""

    if not configs:
        return OmegaConf.create({})
    result = configs[0]
    for cfg in configs[1:]:
        result = OmegaConf.merge(result, cfg)  # type: ignore[assignment]
    return result  # type: ignore[return-value]


def override_from_cli(cfg: DictConfig, overrides: Iterable[str]) -> DictConfig:
    """Apply ``--key.path=value`` style overrides to a config.

    Each override is parsed by OmegaConf so types ('123' → 123) are coerced.
    """

    overrides = list(overrides)
    if not overrides:
        return cfg
    cli = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, cli)  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: os.PathLike[str] | str) -> None:
    """Persist a resolved config alongside checkpoints for reproducibility."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))


def to_container(cfg: DictConfig) -> dict:
    """Return a plain ``dict`` for code that doesn't want OmegaConf objects."""

    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
