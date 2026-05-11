# Benchmarks

The repo ships runners for the headline benchmarks. **All numbers are
zero by default** — that's what you get when no training has happened.
Fill in the table after you finish training; we leave it blank so
nobody mistakes the harness for a result.

## How to run

```powershell
python scripts/eval.py `
    --bench humaneval `
    --model forge-320b `
    --checkpoint outputs/dpo/latest `
    --tokenizer checkpoints/forge-tokenizer.json `
    --limit 164
```

Supported `--bench` values:

| Name             | Type                 | Implementation file                              |
|------------------|----------------------|--------------------------------------------------|
| `humaneval`      | code (pass@1)        | `src/chatbot/eval/humaneval.py`                  |
| `mbpp`           | code (pass@1)        | `src/chatbot/eval/mbpp.py`                       |
| `livecodebench`  | code (pass@1)        | `src/chatbot/eval/livecodebench.py`              |
| `mmlu`           | multiple-choice      | `src/chatbot/eval/mmlu.py`                       |
| `gsm8k`          | math word problems   | `src/chatbot/eval/gsm8k.py`                      |
| `swe_bench_lite` | agentic patching     | `src/chatbot/eval/swe_bench_lite.py` (library only — needs your own apply/test callbacks) |

## Result table (fill in after training)

| Benchmark         | Tiny | Aurora-72B  | Forge-320B  |
|-------------------|------|-------------|-------------|
| HumanEval         | TBD  | TBD         | TBD         |
| MBPP              | TBD  | TBD         | TBD         |
| LiveCodeBench     | TBD  | TBD         | TBD         |
| MMLU              | TBD  | TBD         | TBD         |
| GSM8K             | TBD  | TBD         | TBD         |
| SWE-bench Lite    | TBD  | TBD         | TBD         |

## SWE-bench Lite — usage notes

`swe_bench_lite` needs a sandbox per problem (each instance ships a
container image), so we don't run the actual tests for you. Instead, the
runner is a library-only entrypoint:

```python
from chatbot.eval.swe_bench_lite import run_swe_bench_lite

result = run_swe_bench_lite(
    agent=my_built_agent,
    problems=swebench_problems,
    apply_patch_fn=my_apply_patch_fn,   # checks out repo, applies the diff
    run_tests_fn=my_run_tests_fn,       # returns True if all gating tests pass
)
```

This keeps containerization concerns out of the LLM repo; users with a
running SWE-bench Docker setup wire the callbacks in 20 lines of glue.

## Reproducibility

* `--temperature 0` is the default for code/math benchmarks so a single
  run is deterministic.
* For MMLU we use log-prob scoring across the four-letter answer tokens,
  so temperature is moot.
* The exact dataset versions are pinned via Hugging Face dataset IDs in
  `scripts/eval.py`.
