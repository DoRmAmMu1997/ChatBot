"""Single-codebook EnCodec-style audio codec.

The codec has two halves:

* **Encoder** — turns a waveform into a sequence of discrete codebook
  indices (one int per ~20 ms frame). The LLM doesn't normally use this
  side directly; it's used during training to *teach* the LLM what audio
  tokens look like.
* **Decoder** — turns those indices back into a waveform. The LLM's
  inference path uses this half: when the model emits audio tokens
  between ``<|audio_start|>`` and ``<|audio_end|>``, we look up each
  token's embedding from the codec's codebook and feed the sequence
  through the decoder.

Why single-codebook (not RVQ-8 like full EnCodec): keeping a single
codebook means the LLM's vocabulary stays flat — text tokens and audio
tokens are interchangeable from the loss's perspective. Audio fidelity at
24 kHz with 4096 codes per 20 ms is intelligible if not studio-quality;
swapping in multi-codebook RVQ later is a contained change to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioCodecConfig:
    sample_rate: int = 24000
    codebook_size: int = 4096
    codebook_dim: int = 128
    # Frame rate after the encoder strides — 24000 / (8 * 60) ≈ 50 fps
    # (this is approximate; the exact value depends on the stride product
    # below).
    encoder_strides: tuple = (2, 4, 5, 8)
    enc_dim: int = 256


class _ResidualBlock1d(nn.Module):
    """Small residual block used inside the codec encoder and decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        return F.gelu(x + residual)


class _Encoder(nn.Module):
    """Waveform → continuous frames (pre-quantization)."""

    def __init__(self, cfg: AudioCodecConfig):
        super().__init__()
        layers = [nn.Conv1d(1, cfg.enc_dim, kernel_size=7, padding=3), nn.GELU()]
        channels = cfg.enc_dim
        for stride in cfg.encoder_strides:
            # Each strided conv halves the feature rate; combined they reduce
            # the 24 kHz waveform down to the codec's frame rate.
            layers.append(nn.Conv1d(channels, channels, kernel_size=stride * 2,
                                    stride=stride, padding=stride))
            layers.append(_ResidualBlock1d(channels))
        layers.append(nn.Conv1d(channels, cfg.codebook_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [batch, samples] → [batch, 1, samples] → [batch, codebook_dim, frames]
        return self.net(waveform.unsqueeze(1))


class _Decoder(nn.Module):
    """Quantized frames → waveform."""

    def __init__(self, cfg: AudioCodecConfig):
        super().__init__()
        channels = cfg.enc_dim
        layers = [nn.Conv1d(cfg.codebook_dim, channels, kernel_size=1), nn.GELU()]
        for stride in reversed(cfg.encoder_strides):
            layers.append(_ResidualBlock1d(channels))
            layers.append(nn.ConvTranspose1d(channels, channels, kernel_size=stride * 2,
                                              stride=stride, padding=stride // 2))
            layers.append(nn.GELU())
        layers.append(nn.Conv1d(channels, 1, kernel_size=7, padding=3))
        layers.append(nn.Tanh())  # keep waveform in [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [batch, codebook_dim, frames] → [batch, samples]
        return self.net(frames).squeeze(1)


class _VectorQuantizer(nn.Module):
    """Pick the nearest codebook vector for each input frame.

    Training uses straight-through gradient estimation: in the forward pass we
    snap to the nearest codebook entry; in the backward pass we pretend the
    quantization was identity so gradients flow into the encoder.
    """

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.normal_(self.codebook.weight, std=0.02)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

    def forward(self, x: torch.Tensor):
        """Args: ``[batch, codebook_dim, frames]``. Returns (quantized, codes, vq_loss)."""

        batch, dim, frames = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, dim)  # [batch*frames, dim]

        # Squared distance to each codebook entry.
        # ||x - e||^2 = ||x||^2 - 2 x·e + ||e||^2
        x_sq = flat.pow(2).sum(dim=-1, keepdim=True)
        c = self.codebook.weight
        c_sq = c.pow(2).sum(dim=-1)
        dists = x_sq - 2 * flat @ c.t() + c_sq

        codes = dists.argmin(dim=-1)              # [batch*frames]
        quantized = self.codebook(codes)         # [batch*frames, dim]

        # VQ loss has two pieces: the codebook moves toward the encoder
        # (commitment), and the encoder moves toward the codebook
        # (codebook loss). Standard formulation from VQ-VAE.
        commitment = F.mse_loss(quantized.detach(), flat)
        codebook_loss = F.mse_loss(quantized, flat.detach())
        vq_loss = codebook_loss + 0.25 * commitment

        # Straight-through estimator.
        quantized = flat + (quantized - flat).detach()
        quantized = quantized.view(batch, frames, dim).permute(0, 2, 1)
        codes = codes.view(batch, frames)
        return quantized, codes, vq_loss


class AudioCodec(nn.Module):
    """End-to-end audio codec: encode→quantize→decode.

    Three modes you'll actually use:

    * :meth:`encode` (training) — waveform → codes + vq_loss.
    * :meth:`decode` (inference) — codes → waveform.
    * :meth:`forward` (codec-only autoencoder pretrain) — runs the full
      pipeline and returns the reconstruction + loss components.
    """

    def __init__(self, cfg: AudioCodecConfig | None = None):
        super().__init__()
        cfg = cfg or AudioCodecConfig()
        self.cfg = cfg
        self.encoder = _Encoder(cfg)
        self.quantizer = _VectorQuantizer(cfg.codebook_size, cfg.codebook_dim)
        self.decoder = _Decoder(cfg)

    def encode(self, waveform: torch.Tensor):
        z = self.encoder(waveform)
        quantized, codes, vq_loss = self.quantizer(z)
        return codes, quantized, vq_loss

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [batch, frames] integers → codebook lookups → decoder.
        embeds = self.quantizer.codebook(codes)             # [batch, frames, dim]
        frames = embeds.permute(0, 2, 1)                    # [batch, dim, frames]
        return self.decoder(frames)

    def forward(self, waveform: torch.Tensor):
        codes, quantized, vq_loss = self.encode(waveform)
        recon = self.decoder(quantized)
        # Reconstruction loss is on the waveform itself — same length as input
        # (within a few samples; we trim to the shorter length).
        min_len = min(recon.shape[-1], waveform.shape[-1])
        recon_loss = F.l1_loss(recon[..., :min_len], waveform[..., :min_len])
        return {"recon": recon, "codes": codes, "recon_loss": recon_loss, "vq_loss": vq_loss}
