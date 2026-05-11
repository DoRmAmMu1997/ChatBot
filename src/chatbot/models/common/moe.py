"""Mixture-of-Experts (MoE) feed-forward block.

The high-level picture:

  * Each layer has **E routed experts** plus **S shared experts**.
  * For each input token, a tiny router scores all E routed experts.
  * The top-K routed experts (typically K=8 out of 128) are activated for
    this token; their outputs are weighted by the router probabilities and
    summed.
  * The shared expert(s) always run for every token — they handle patterns
    that aren't worth specializing.

This buys two things at once:

  * **Capacity** — total parameters scale with E, so the model "knows"
    more without making each forward pass more expensive.
  * **Active compute** — only K of E expert FFNs run per token, so the
    FLOP cost is closer to a much smaller dense model.

We implement two load-balancing strategies and pick via config:

  * ``aux_loss`` — classic auxiliary loss that pushes the router toward
    even expert usage.
  * ``aux_loss_free`` — DeepSeek-V3's trick: maintain a per-expert *bias*
    added to routing logits and update it after each batch so the slowest
    expert speeds up and the busiest expert slows down. No extra loss term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import SwiGLU


@dataclass
class MoEOutput:
    hidden_states: torch.Tensor
    # Auxiliary metrics surfaced to the training loop. ``aux_loss`` is added
    # to the language-modeling loss; ``router_z_loss`` discourages routing
    # logits from blowing up; ``load_balance`` is monitoring-only.
    aux_loss: torch.Tensor
    router_z_loss: torch.Tensor
    load_balance: torch.Tensor


class _ExpertGroup(nn.Module):
    """A bag of E parallel SwiGLU experts.

    The straightforward implementation (a Python list of SwiGLU modules,
    iterated per token) is slow on GPUs. We instead pack the parameters into
    big tensors and dispatch tokens via ``torch.index_select``. This is the
    "loop over experts in a single Python call" pattern that is much
    cheaper than per-expert Python overhead.
    """

    def __init__(self, num_experts: int, d_model: int, hidden: int):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.hidden = hidden

        # Concatenated weights: one big tensor per projection so we can
        # gather a per-token expert weight in one go.
        self.gate_weight = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.up_weight = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.down_weight = nn.Parameter(torch.empty(num_experts, d_model, hidden))
        nn.init.normal_(self.gate_weight, std=0.02)
        nn.init.normal_(self.up_weight, std=0.02)
        nn.init.normal_(self.down_weight, std=0.02)

    def forward_tokens(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Process tokens routed to specific experts.

        Args:
            x: ``[N, d_model]`` flat tokens.
            expert_ids: ``[N]`` integer expert assignment per token.

        Returns: ``[N, d_model]`` output.
        """

        # Gather the per-token expert weights.
        gate_w = self.gate_weight[expert_ids]   # [N, hidden, d_model]
        up_w = self.up_weight[expert_ids]
        down_w = self.down_weight[expert_ids]

        # Manual matmul per token: (1, d_model) @ (d_model, hidden) → (1, hidden).
        # Using ``einsum`` makes shape intent explicit; PyTorch will dispatch
        # this efficiently with BMM under the hood.
        gate = torch.einsum("nd,nhd->nh", x, gate_w)
        up = torch.einsum("nd,nhd->nh", x, up_w)
        hidden = F.silu(gate) * up
        return torch.einsum("nh,ndh->nd", hidden, down_w)


class MixtureOfExperts(nn.Module):
    """Top-K routed MoE FFN block.

    Args:
        d_model: residual dimension.
        num_routed_experts: ``E`` — pool of specialists.
        num_shared_experts: ``S`` — always-on experts (typically 1).
        num_active_experts: ``K`` — how many routed experts run per token.
        expert_hidden: SwiGLU intermediate dim of each expert.
        router_jitter: small noise added to routing logits during training
            for exploration.
        load_balancing: ``"aux_loss"`` or ``"aux_loss_free"``.
        bias_update_speed: step size for the AL-free bias.
        router_z_loss_coef: coefficient for the z-loss (router stability).
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_routed_experts: int,
        num_shared_experts: int,
        num_active_experts: int,
        expert_hidden: int,
        router_jitter: float = 0.0,
        load_balancing: str = "aux_loss_free",
        bias_update_speed: float = 1.0e-3,
        router_z_loss_coef: float = 1.0e-4,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        assert load_balancing in {"aux_loss", "aux_loss_free"}, load_balancing
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_active_experts = num_active_experts
        self.router_jitter = router_jitter
        self.load_balancing = load_balancing
        self.bias_update_speed = bias_update_speed
        self.router_z_loss_coef = router_z_loss_coef
        self.aux_loss_coef = aux_loss_coef

        # The router itself is just a Linear: each token gets a score per expert.
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)

        # Auxiliary-loss-free balancing keeps a non-trainable bias vector that
        # nudges underused experts up and overused experts down each step.
        self.register_buffer(
            "router_bias", torch.zeros(num_routed_experts), persistent=True
        )

        # Routed experts (packed) + shared experts (plain).
        self.experts = _ExpertGroup(num_routed_experts, d_model, expert_hidden)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [SwiGLU(d_model, expert_hidden) for _ in range(num_shared_experts)]
            )
        else:
            self.shared_experts = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> MoEOutput:
        batch, seq, d_model = x.shape
        flat = x.reshape(-1, d_model)
        n_tokens = flat.shape[0]

        # ---- Router ----
        # Standard logits, then optionally jittered during training.
        logits = self.router(flat)                # [N, E]
        if self.training and self.router_jitter > 0:
            logits = logits + self.router_jitter * torch.randn_like(logits)
        # Bias used by the AL-free strategy. Gradients do NOT flow through
        # this bias — it's only used to *pick* experts, not weight them.
        biased_logits = logits + self.router_bias.detach()

        # Top-K experts by the biased logits.
        topk_vals, topk_idx = torch.topk(biased_logits, k=self.num_active_experts, dim=-1)
        # Routing weights come from a softmax over the *unbiased* top-K logits,
        # selected by the same indices. This is the DeepSeek-V3 recipe.
        topk_logits = torch.gather(logits, dim=-1, index=topk_idx)
        topk_weights = F.softmax(topk_logits.float(), dim=-1).to(x.dtype)  # [N, K]

        # ---- Aux losses ----
        # Router z-loss: ``log(sum(exp(logits))) ** 2`` averaged. Keeps the
        # softmax temperature stable so the router doesn't saturate.
        router_z = torch.logsumexp(logits.float(), dim=-1)
        router_z_loss = router_z.pow(2).mean() * self.router_z_loss_coef

        # Sequence-level load balance loss (standard MoE aux):
        # ``E * sum_i (f_i * p_i)`` where f_i = fraction of tokens routed to
        # expert i and p_i = mean router probability for expert i.
        with torch.no_grad():
            full_probs = F.softmax(logits.float(), dim=-1)              # [N, E]
        p_i = full_probs.mean(dim=0)                                    # [E]
        # f_i computed from the actual top-K dispatch.
        mask = torch.zeros_like(full_probs).scatter_(1, topk_idx, 1.0)  # [N, E]
        f_i = mask.mean(dim=0)                                          # [E]
        load_balance = (p_i * f_i).sum() * self.num_routed_experts

        if self.load_balancing == "aux_loss":
            aux_loss = load_balance * self.aux_loss_coef
        else:
            aux_loss = x.new_zeros(())

        # AL-free balancing: nudge the per-expert bias so under-used
        # experts (f_i below target) get a *higher* bias next step and
        # over-used experts a *lower* one. This is a closed-loop control
        # signal — no gradients involved — that keeps token allocation
        # roughly uniform without an explicit auxiliary loss term.
        if self.training and self.load_balancing == "aux_loss_free":
            target_load = 1.0 / self.num_routed_experts
            with torch.no_grad():
                # Subtraction direction: f_i > target → delta > 0 → bias
                # decreases → expert becomes less attractive next step.
                delta = (f_i - target_load) * self.bias_update_speed
                self.router_bias.sub_(delta)

        # ---- Dispatch tokens to experts ----
        # We flatten the (token, slot) pairs into a bag of assignments.
        # If token 5 was routed to experts [13, 47, 91], we'd add three
        # entries: (5, 13), (5, 47), (5, 91). All N*K such entries are
        # processed in a single batched call to the expert group.
        flat_expert_ids = topk_idx.reshape(-1)                          # [N*K]
        # Each token id is repeated K times because it goes to K experts.
        flat_token_ids = torch.arange(n_tokens, device=x.device).repeat_interleave(self.num_active_experts)
        # Gather the input vector for each (token, expert) assignment.
        # We could sort by expert id for slightly better cache locality,
        # but the unsorted gather is correct and easier to read.
        token_inputs = flat[flat_token_ids]                             # [N*K, d_model]
        # Run each assignment through the right expert in parallel.
        expert_outputs = self.experts.forward_tokens(token_inputs, flat_expert_ids)

        # Weight each expert output by its router probability. The K
        # weights for a single token sum to 1, so the final per-token
        # routed output is a convex combination of the K expert outputs.
        flat_weights = topk_weights.reshape(-1, 1)                      # [N*K, 1]
        expert_outputs = expert_outputs * flat_weights

        # Scatter back to per-token outputs.
        routed = torch.zeros_like(flat)
        routed.index_add_(0, flat_token_ids, expert_outputs)

        # ---- Shared experts (always on) ----
        shared = torch.zeros_like(flat)
        for shared_expert in self.shared_experts:
            shared = shared + shared_expert(flat)

        hidden_states = (routed + shared).view(batch, seq, d_model)
        return MoEOutput(
            hidden_states=hidden_states,
            aux_loss=aux_loss,
            router_z_loss=router_z_loss,
            load_balance=load_balance.detach(),
        )
