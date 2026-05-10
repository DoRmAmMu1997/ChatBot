"""Estimate model parameters without allocating weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Running scripts/estimate_params.py directly sets sys.path to scripts/.
    # Adding the repo root lets Python import src.chatbot without installation.
    sys.path.insert(0, str(ROOT))

from src.chatbot.config import ModelConfig
from src.chatbot.params import estimate_parameter_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ChatBot model parameter counts.")
    parser.add_argument("--config", required=True, help="Path to a YAML model config.")
    args = parser.parse_args()

    # The estimator reads the blueprint only. It never creates a 10B model in
    # memory, so it is safe to run on a laptop or in CI.
    config = ModelConfig.from_yaml_file(args.config)
    report = estimate_parameter_count(config)
    print(f"model: {config.model_name}")
    print(f"total_parameters: {report.total}")
    print(f"total_billions: {report.total / 1_000_000_000:.3f}B")
    print(f"embeddings: {report.embeddings}")
    print(f"layers: {report.layers}")
    print(f"final_norm: {report.final_norm}")
    print(f"lm_head: {report.lm_head}")


if __name__ == "__main__":
    main()
