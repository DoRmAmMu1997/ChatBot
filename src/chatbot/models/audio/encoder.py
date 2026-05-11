"""Audio encoder (speech-in side).

Takes a log-mel spectrogram and produces a sequence of vectors in the
LLM's embedding space. Conformer-ish: a stack of pre-norm Transformer
blocks fronted by two strided Conv1d layers that subsample the time axis
so the LLM doesn't get drowned in audio frames.

Output sequence length math (with the defaults):

    16 kHz waveform → 10 ms mel frames → 80 ms tokens after 2x conv stride
                                       → 320 ms tokens with extra 2x pool

So a 1-second audio clip becomes ~3 soft tokens for the LLM. A 10-second
clip becomes ~30 soft tokens. The LLM treats them exactly like text token
embeddings, so they can be interleaved with text via the ``<|audio|>``
placeholder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.ffn import SwiGLU
from ..common.normalization import RMSNorm
from .mel import LogMelSpectrogram


class _MHAAudio(nn.Module):
    """Plain multi-head self-attention used inside the audio encoder.

    Audio attention is non-causal — every frame can look at every other
    frame, same as the vision tower. So no RoPE / no causal mask.
    """

    def __init__(self, dim: int, num_heads: int, *, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.n_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.o_proj(out)


class _AudioEncoderBlock(nn.Module):
    """One Transformer block inside the audio encoder."""

    def __init__(self, dim: int, num_heads: int, ffn_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = _MHAAudio(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AudioEncoder(nn.Module):
    """Whisper-style audio encoder.

    Args:
        llm_dim: target dimension to project into (the LLM's ``d_model``).
        n_mels: mel-band count from the spectrogram.
        dim: width of the encoder's internal residual stream.
        depth: number of Transformer blocks.
        num_heads: attention heads.
        ffn_hidden: SwiGLU intermediate dimension inside each block.
        sample_rate / n_fft / hop_length: feature-extraction settings.

    Forward signature: takes a 1-D waveform tensor ``[batch, samples]`` and
    returns a sequence of LLM-dimension embeddings ``[batch, frames, llm_dim]``.
    """

    def __init__(
        self,
        *,
        llm_dim: int,
        n_mels: int = 128,
        dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        ffn_hidden: int | None = None,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()
        if ffn_hidden is None:
            ffn_hidden = dim * 4

        self.mel = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Two strided convs downsample mel frames 4x in the time axis so the
        # LLM doesn't get hundreds of audio tokens per second of speech.
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_mels, dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # Trainable absolute position embeddings — audio is short enough that
        # we don't need RoPE here. We size for ~30 s of audio = ~94 frames at
        # 320 ms each; bumping this if you want longer per-utterance clips.
        self.max_frames = 4096
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_AudioEncoderBlock(dim, num_heads, ffn_hidden) for _ in range(depth)]
        )
        self.final_norm = RMSNorm(dim)

        # MLP connector projects into the LLM's embedding dimension. Same
        # design as the vision connector.
        self.connector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, llm_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Step 1: raw waveform → log-mel spectrogram. Shape goes from
        # [batch, samples] to [batch, n_mels, frames]. Each mel frame is
        # 10 ms of audio (default ``hop_length=160`` at 16 kHz).
        mel = self.mel(waveform)
        # Step 2: two strided Conv1ds shrink the time axis 4x. A 1-second
        # clip with 100 mel frames becomes ~25 conv-frames. Without this
        # downsample the LLM would receive hundreds of audio tokens per
        # second of speech, which is far more than it actually needs.
        x = self.input_proj(mel)
        # Step 3: re-layout to [batch, frames', dim] which is what the
        # Transformer blocks expect (time as the sequence axis).
        x = x.transpose(1, 2)
        seq = x.shape[1]
        # Step 4: trim very long inputs to ``max_frames`` rather than
        # crashing. In production you'd chunk long audio into overlapping
        # windows *before* the encoder; this guard is a safety net.
        if seq > self.max_frames:
            x = x[:, : self.max_frames]
            seq = self.max_frames
        # Step 5: add the learned positional embedding for each frame.
        # Audio is short enough that absolute positions work fine; no
        # RoPE here.
        x = x + self.pos_embed[:, :seq]

        # Step 6: stack of Transformer blocks (RMSNorm + MHA + SwiGLU)
        # acts on the frame sequence.
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # Step 7: 2-layer MLP projects each frame into the LLM's
        # embedding space. The result is what the LLM's `<|audio|>`
        # placeholder tokens get replaced with.
        return self.connector(x)
