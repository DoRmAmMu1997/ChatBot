"""Token-sampling helpers.

We support the standard knobs:

* ``temperature`` — divide logits by this before softmax. ``< 1`` sharpens,
  ``> 1`` flattens, ``0`` collapses to argmax (greedy).
* ``top_k`` — keep only the K highest-probability tokens.
* ``top_p`` — keep the smallest set of tokens whose cumulative probability
  is at least ``top_p`` (nucleus sampling).
* ``min_p`` — drop tokens whose probability is < ``min_p`` × max prob.
* ``repetition_penalty`` — divide the logit of every already-generated
  token by this factor; > 1 reduces repetition.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample the next token from a [batch, vocab] logit tensor.

    Returns a tensor of shape ``[batch, 1]`` with the sampled token ids.
    """

    if temperature <= 0:
        # Greedy decoding bypass — pick the argmax. No randomness. Useful for
        # benchmarks where we want deterministic, reproducible outputs.
        return logits.argmax(dim=-1, keepdim=True)

    # Apply repetition penalty to already-generated tokens.
    # Standard trick from CTRL (Keskar et al. 2019): if a token has already
    # been produced, push its logit toward zero so the softmax probability
    # for it shrinks — discourages obvious loops like "the the the …".
    if repetition_penalty != 1.0 and previous_tokens is not None:
        for batch_idx in range(logits.shape[0]):
            seen = previous_tokens[batch_idx]
            seen_logits = logits[batch_idx, seen]
            # Dividing positive logits and multiplying negative ones both push
            # toward zero in log space, so the sign flip is intentional.
            logits[batch_idx, seen] = torch.where(
                seen_logits > 0,
                seen_logits / repetition_penalty,
                seen_logits * repetition_penalty,
            )

    # Temperature. Dividing the logits before softmax sharpens (T<1) or
    # flattens (T>1) the distribution. We floor T at a tiny positive number so
    # we never divide by zero when callers pass small temperatures.
    logits = logits / max(temperature, 1.0e-5)

    # Top-K filtering — keep only the K most likely tokens by zeroing out
    # everything below the K-th largest logit. Setting `-inf` makes those
    # tokens have zero probability after softmax.
    if top_k > 0:
        kth_largest = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_largest, torch.full_like(logits, float("-inf")), logits)

    # Top-P (nucleus) filtering: keep the smallest set of tokens whose
    # cumulative probability is >= top_p. This adapts to the model's
    # confidence — a peaked distribution will keep very few tokens; a flat
    # one keeps many. Implementation steps:
    #   1) Sort logits descending; compute softmax probs over the sorted list.
    #   2) Cumulative-sum those probs.
    #   3) Mask any sorted positions where the *previous* cumulative was
    #      already past top_p (i.e. we already had enough mass without them).
    #   4) Scatter the mask back to the original token order.
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative > top_p
        # Shift right by one so we always keep the very first token (the
        # max-prob one) — guarantees the nucleus is non-empty.
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    # Min-P filtering: drop any token whose probability is less than
    # `min_p * max_prob`. Unlike top-p, this *requires* a token to be at
    # least somewhat plausible relative to the best alternative, regardless
    # of how many other tokens compete with it.
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        cutoff = min_p * max_prob
        logits = torch.where(probs < cutoff, torch.full_like(logits, float("-inf")), logits)

    # Convert filtered logits to a probability distribution and draw one
    # token per batch row.
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
