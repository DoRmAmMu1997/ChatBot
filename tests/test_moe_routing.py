"""MoE routing correctness tests."""

from __future__ import annotations

import torch

from chatbot.models.common.moe import MixtureOfExperts


def test_moe_forward_shape():
    moe = MixtureOfExperts(
        d_model=32,
        num_routed_experts=4,
        num_shared_experts=1,
        num_active_experts=2,
        expert_hidden=64,
    )
    x = torch.randn(2, 5, 32)
    out = moe(x)
    assert out.hidden_states.shape == x.shape
    assert out.aux_loss.shape == ()
    assert out.router_z_loss.shape == ()


def test_moe_aux_free_bias_updates():
    moe = MixtureOfExperts(
        d_model=16,
        num_routed_experts=4,
        num_shared_experts=0,
        num_active_experts=2,
        expert_hidden=32,
        load_balancing="aux_loss_free",
        bias_update_speed=1.0,  # very large step so the change is visible
    )
    moe.train()
    initial_bias = moe.router_bias.clone()
    x = torch.randn(8, 4, 16)
    moe(x)
    assert not torch.allclose(initial_bias, moe.router_bias), (
        "Aux-loss-free balancing should adjust the per-expert bias during training."
    )


def test_moe_no_bias_update_in_eval():
    moe = MixtureOfExperts(
        d_model=16,
        num_routed_experts=4,
        num_shared_experts=0,
        num_active_experts=2,
        expert_hidden=32,
        load_balancing="aux_loss_free",
        bias_update_speed=1.0,
    )
    moe.eval()
    initial_bias = moe.router_bias.clone()
    x = torch.randn(8, 4, 16)
    moe(x)
    assert torch.allclose(initial_bias, moe.router_bias), (
        "Bias should not change while the module is in eval mode."
    )
