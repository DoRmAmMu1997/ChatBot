"""Pre-download the datasets the user picks for a training stage.

This is convenience tooling: it walks the data registry, calls HF's
``load_dataset`` for each entry the user requests, and tells you where the
cached files landed. The training scripts themselves stream lazily, so
pre-downloading is optional — useful when you want to bake a dataset
mirror into a slow-network environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chatbot.data.registry import DATASETS, available_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download datasets via Hugging Face.")
    parser.add_argument("--stage", choices=["pretrain", "sft", "dpo", "tool_use",
                                            "image_text", "long_context", "all"],
                        default="all")
    parser.add_argument("--names", nargs="*", default=None,
                        help="Override stage filter — fetch these specific datasets only.")
    parser.add_argument("--limit", type=int, default=100,
                        help="Iterate first N rows to force the cache to materialize.")
    args = parser.parse_args()

    names = args.names or available_datasets(stage=None if args.stage == "all" else args.stage)
    print(f"Downloading {len(names)} datasets: {names}")
    for name in names:
        spec = DATASETS[name]
        if spec.hf_repo is None:
            print(f"[skip] {name} — locally-generated, no remote source.")
            continue
        try:
            iterator = iter(spec.loader())
            for _ in range(args.limit):
                next(iterator)
            print(f"[ok]   {name}")
        except StopIteration:
            print(f"[ok]   {name} (smaller than {args.limit} rows)")
        except Exception as exc:  # noqa: BLE001 — we want a friendly summary
            print(f"[fail] {name}: {exc}")


if __name__ == "__main__":
    main()
