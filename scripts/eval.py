"""Run a benchmark against a trained checkpoint.

Currently supports: humaneval, mbpp, mmlu, gsm8k, livecodebench.
SWE-bench Lite is supported via the library API but needs user-supplied
patch-apply / test-run callbacks (see chatbot/eval/swe_bench_lite.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from chatbot.eval import EvalResult, run_benchmark             # noqa: E402
from chatbot.inference.generate import _build_model             # noqa: E402
from chatbot.tokenizer.bpe import BPETokenizer                  # noqa: E402
from chatbot.training.checkpoint import load_checkpoint         # noqa: E402
from chatbot.utils.config import load_config                    # noqa: E402


def _load_dataset(bench: str, limit: int):
    """Load a small slice of the right HF dataset for the chosen benchmark."""

    from datasets import load_dataset

    if bench == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        return list(ds.select(range(min(limit, len(ds)))))
    if bench == "mbpp":
        ds = load_dataset("mbpp", split="test")
        return list(ds.select(range(min(limit, len(ds)))))
    if bench == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        return list(ds.select(range(min(limit, len(ds)))))
    if bench == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return list(ds.select(range(min(limit, len(ds)))))
    if bench == "livecodebench":
        ds = load_dataset("livecodebench/code_generation", split="test")
        return list(ds.select(range(min(limit, len(ds)))))
    raise ValueError(f"Unknown benchmark: {bench}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark on a checkpoint.")
    parser.add_argument("--bench", required=True,
                        choices=["humaneval", "mbpp", "mmlu", "gsm8k", "livecodebench"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--limit", type=int, default=164,
                        help="Maximum number of examples (some benchmarks have 1000+).")
    args = parser.parse_args()

    model_cfg = load_config(f"models/{args.model}")
    model = _build_model(model_cfg)
    load_checkpoint(args.checkpoint, model=model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    tokenizer = BPETokenizer.from_file(args.tokenizer)

    examples = _load_dataset(args.bench, args.limit)
    key = "problems" if args.bench in {"humaneval", "mbpp", "gsm8k", "livecodebench"} else "questions"
    result: EvalResult = run_benchmark(args.bench, model=model, tokenizer=tokenizer, **{key: examples})
    print(f"=== {result.benchmark} ===")
    print(f"Score:  {result.score:.4f}")
    print(f"N:      {result.num_examples}")
    if result.details:
        print(f"Details: {result.details}")
    if result.notes:
        for note in result.notes:
            print(f"Note: {note}")


if __name__ == "__main__":
    main()
