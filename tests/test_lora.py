"""LoRA correctness tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from chatbot.training.lora import LoRALinear, apply_lora_to_model


def test_lora_zero_alpha_matches_base():
    base = nn.Linear(8, 8, bias=False)
    base.weight.data.normal_()
    # With B initialized to zero, the adapter has zero contribution at init.
    wrapped = LoRALinear(base, rank=4, alpha=8, init="kaiming")
    x = torch.randn(2, 8)
    out_base = base(x)
    out_wrapped = wrapped(x)
    assert torch.allclose(out_base, out_wrapped, atol=1e-6)


def test_lora_only_adapter_trains():
    base = nn.Linear(8, 8, bias=False)
    wrapped = LoRALinear(base, rank=4, alpha=8)
    # Base weight should be frozen.
    assert not wrapped.base.weight.requires_grad
    # Adapter A and B should be trainable.
    assert wrapped.lora_A.requires_grad
    assert wrapped.lora_B.requires_grad


def test_apply_lora_to_nested_module_picks_targets():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.unused = nn.Linear(4, 4)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Inner()

    model = Outer()
    apply_lora_to_model(model, target_modules=["q_proj"], rank=2, alpha=4)
    assert isinstance(model.attn.q_proj, LoRALinear)
    assert not isinstance(model.attn.unused, LoRALinear)
