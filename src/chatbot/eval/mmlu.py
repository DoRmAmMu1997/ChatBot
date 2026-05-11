"""MMLU runner — 4-way multiple choice across 57 academic subjects."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .runner import EvalResult, register_benchmark


def _question_to_prompt(question: Dict[str, Any]) -> str:
    choices = question.get("choices", [])
    letters = ["A", "B", "C", "D"]
    rendered = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    return (
        f"Question: {question['question']}\n"
        f"{rendered}\n"
        f"Answer:"
    )


@register_benchmark("mmlu")
def run_mmlu(
    *,
    model,
    tokenizer,
    questions: List[Dict[str, Any]],
) -> EvalResult:
    """Compare log-probs of " A", " B", " C", " D" and pick the highest."""

    if not questions:
        return EvalResult(benchmark="mmlu", score=0.0, num_examples=0,
                          notes=["No questions supplied — load cais/mmlu."])

    device = next(model.parameters()).device
    letter_ids = {l: tokenizer.encode(f" {l}")[-1] for l in ["A", "B", "C", "D"]}
    correct = 0
    for q in questions:
        prompt = _question_to_prompt(q)
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids)
        logits = out["logits"][0, -1, :]
        scores = {l: float(F.log_softmax(logits, dim=-1)[t]) for l, t in letter_ids.items()}
        pick = max(scores, key=scores.get)
        gold = ["A", "B", "C", "D"][int(q["answer"])] if isinstance(q.get("answer"), int) else str(q["answer"]).strip()
        if pick == gold:
            correct += 1
    score = correct / len(questions)
    return EvalResult(
        benchmark="mmlu", score=score, num_examples=len(questions),
        details={"correct": correct},
    )
