"""Shape tests for the vision tower + connector."""

from __future__ import annotations

import torch

from chatbot.models.vision.connector import MLPConnector
from chatbot.models.vision.patch_embed import PatchEmbed
from chatbot.models.vision.vit_encoder import ViTEncoder


def test_patch_embed_shape():
    pe = PatchEmbed(image_size=224, patch_size=14, in_channels=3, embed_dim=64)
    x = torch.randn(2, 3, 224, 224)
    out = pe(x)
    assert out.shape == (2, (224 // 14) ** 2, 64)


def test_vit_encoder_shape():
    vit = ViTEncoder(image_size=224, patch_size=14, dim=64, depth=2, num_heads=4, ffn_hidden=128)
    x = torch.randn(1, 3, 224, 224)
    out = vit(x)
    assert out.shape == (1, (224 // 14) ** 2, 64)


def test_connector_projects_to_llm_dim():
    conn = MLPConnector(vision_dim=64, hidden_dim=96, llm_dim=128)
    tokens = torch.randn(1, 49, 64)
    out = conn(tokens)
    assert out.shape == (1, 49, 128)
