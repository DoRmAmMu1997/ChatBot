"""Batched generation for evaluation / bulk inference.

This is intentionally simple: it does not support left-padding or KV-cache
sharing across the batch. For a benchmark sweep on a small model that's
plenty; production workloads should use a serving runtime like vLLM.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .sampling import sample_token


@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 0,
    eos_token_ids: Optional[Sequence[int]] = None,
) -> List[str]:
    """Generate completions for many prompts. Returns text in input order."""

    eos_ids = set(eos_token_ids or [tokenizer.eos_id()])
    device = next(model.parameters()).device
    outputs: List[str] = []

    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated_ids: List[int] = []
        # Prefill.
        out = model(ids)
        next_logits = out["logits"][:, -1, :]
        for _ in range(max_new_tokens):
            next_token = sample_token(
                next_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                previous_tokens=torch.tensor([generated_ids], device=device)
                if generated_ids else None,
            )
            tok_id = int(next_token.item())
            if tok_id in eos_ids:
                break
            generated_ids.append(tok_id)
            ids = torch.cat([ids, next_token], dim=1)
            out = model(ids)
            next_logits = out["logits"][:, -1, :]
        outputs.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return outputs
