"""Audio codec — encoder/quantizer/decoder round-trip + loss shape."""

from __future__ import annotations

import torch

from chatbot.models.audio.codec import AudioCodec, AudioCodecConfig


def test_codec_forward_shapes():
    cfg = AudioCodecConfig(codebook_size=128, codebook_dim=64, enc_dim=64)
    codec = AudioCodec(cfg)
    waveform = torch.randn(1, 24000)         # 1 second of fake audio
    out = codec(waveform)
    assert "recon" in out
    assert "codes" in out
    assert "recon_loss" in out
    assert "vq_loss" in out
    assert out["recon_loss"].dim() == 0
    assert out["vq_loss"].dim() == 0


def test_codec_decode_roundtrip_shape():
    cfg = AudioCodecConfig(codebook_size=128, codebook_dim=64, enc_dim=64)
    codec = AudioCodec(cfg)
    codes = torch.zeros(1, 32, dtype=torch.long)
    waveform = codec.decode(codes)
    assert waveform.dim() == 2
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0
