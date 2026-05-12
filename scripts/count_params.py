"""Sanity-check a model config by instantiating it and counting parameters.

Use this BEFORE you commit to a training run — it's a one-second check that
a 50B config really hits ~50B and a 250B config really hits ~250B total /
~25B active. If your numbers are off, fix the config first.

Examples (PowerShell):
    python scripts\\count_params.py --model tiny
    python scripts\\count_params.py --model aurora-72b
    python scripts\\count_params.py --model forge-460b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable without installation, so this script works on a
# fresh checkout before ``pip install -e .`` has been run.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chatbot.utils.config import load_config           # noqa: E402
from chatbot.utils.memory import (                     # noqa: E402
    count_parameters,
    count_active_parameters,
    format_param_count,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count parameters in a model config.")
    parser.add_argument("--model", required=True, help="Model config name, e.g. tiny / aurora-72b / forge-460b.")
    args = parser.parse_args()

    model_cfg = load_config(f"models/{args.model}")
    family = str(model_cfg.get("family", "aurora")).lower()

    print(f"Building {args.model} ({family})…  (CPU instantiation can take a minute on large configs)")
    if family == "aurora":
        from chatbot.models.aurora_50b import AuroraForCausalLM, aurora_config_from_yaml

        cfg = aurora_config_from_yaml(model_cfg)
        model = AuroraForCausalLM(cfg)
    elif family == "forge":
        from chatbot.models.forge_250b import ForgeForCausalLM, forge_config_from_yaml

        cfg = forge_config_from_yaml(model_cfg)
        model = ForgeForCausalLM(cfg)
    else:
        raise SystemExit(f"Unknown family: {family!r}")

    total, trainable = count_parameters(model)
    active = count_active_parameters(model)
    print(f"Total parameters:     {format_param_count(total):>10}")
    print(f"Trainable parameters: {format_param_count(trainable):>10}")
    print(f"Active per token:     {format_param_count(active):>10}")


if __name__ == "__main__":
    main()
