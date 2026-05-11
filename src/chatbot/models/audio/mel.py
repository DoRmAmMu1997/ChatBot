"""Log-mel spectrogram feature extraction.

A neural network can't eat a raw waveform efficiently — it would be too
long and too noisy. Instead we slice the waveform into short windows
(25 ms), Fourier-transform each window into a frequency spectrum, project
that spectrum onto a perceptual ("mel") scale, and take the log of the
result. The output is a 2-D image of shape ``[n_mels, n_frames]`` that
captures the information humans use to recognize speech.

This module implements the same recipe Whisper / wav2vec / EnCodec use.
We avoid the heavyweight ``torchaudio`` dependency by computing the mel
filterbank manually — it's a few hundred lines of NumPy worth of code in
the public domain.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    """Slaney-style hz → mel mapping (same scale used by librosa / Whisper)."""

    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_hz_to_mel`."""

    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    """Build a triangular mel filterbank matrix of shape ``[n_mels, n_fft//2 + 1]``.

    Each row picks out a small frequency band on the mel scale. Multiplying
    a power spectrum by this matrix gives a per-band energy.
    """

    if fmax is None:
        fmax = sample_rate / 2.0

    # Equally spaced mel breakpoints, then convert each back to hz.
    mel_min = _hz_to_mel(torch.tensor(fmin, dtype=torch.float32))
    mel_max = _hz_to_mel(torch.tensor(fmax, dtype=torch.float32))
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)

    # Snap to FFT bin centres so we can index into the spectrum directly.
    bin_freqs = torch.linspace(0, sample_rate / 2.0, n_fft // 2 + 1)

    fb = torch.zeros(n_mels, n_fft // 2 + 1, dtype=torch.float32)
    for m in range(n_mels):
        f_lo, f_mid, f_hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        # Rising edge from f_lo → f_mid.
        rise = (bin_freqs - f_lo) / (f_mid - f_lo + 1e-9)
        # Falling edge from f_mid → f_hi.
        fall = (f_hi - bin_freqs) / (f_hi - f_mid + 1e-9)
        triangle = torch.clamp(torch.minimum(rise, fall), min=0.0)
        fb[m] = triangle

    # Energy-normalize each filter so the sum across bins is ~1.
    fb_sums = fb.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return fb / fb_sums


class LogMelSpectrogram(nn.Module):
    """Convert a 1-D waveform tensor into a log-mel spectrogram tensor.

    Args:
        sample_rate: audio sample rate in Hz (default 16000).
        n_fft: FFT window size in samples (default 400 = 25 ms at 16 kHz).
        hop_length: stride between windows in samples (default 160 = 10 ms).
        n_mels: number of mel bands (default 128 — same as Whisper-large).
        fmin / fmax: frequency clipping range, in Hz.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Mel filterbank is constant — store as a non-trainable buffer.
        fb = _build_mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)
        self.register_buffer("mel_fb", fb, persistent=False)
        # Hann window dampens FFT edge artifacts.
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Args: ``[batch, samples]`` waveform → ``[batch, n_mels, frames]``."""

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # ``torch.stft`` returns complex coefficients; ``.abs() ** 2`` is the
        # power spectrum at each (frame, frequency) bin.
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(waveform.device),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        power = spec.abs().pow(2)

        # Apply the mel filterbank along the frequency axis.
        mel = torch.einsum("mf,bft->bmt", self.mel_fb.to(waveform.device), power)

        # Log-compression. A small floor avoids ``log(0)``.
        return torch.log(torch.clamp(mel, min=1e-10))
