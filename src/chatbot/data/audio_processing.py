"""Waveform loading + mel feature framing for training and inference.

A *very* light wrapper around stdlib + ``soundfile`` (if available) for
loading audio files. Falls back to a raw-PCM read for headerless inputs.
The model itself owns the mel-spectrogram and codec maths; this module
just gets us from "path on disk" to "tensor on a device".
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import torch


def load_waveform(path: str | Path, *, target_sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Read a WAV (or PCM) file into a 1-D float tensor in ``[-1, 1]``.

    Args:
        path: file path on disk. ``.wav`` and ``.pcm`` are supported with the
            stdlib alone. For richer formats (FLAC, MP3, OGG) install
            ``soundfile``; this module will pick it up automatically.
        target_sample_rate: if set, naively resamples by linear interpolation
            to the target. For real training, prefer ``librosa.resample``
            or ``soxr`` — they're higher quality.

    Returns:
        ``(waveform, sample_rate)`` where waveform is 1-D float32.
    """

    path = Path(path)
    waveform: torch.Tensor
    sample_rate: int

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as f:
            sample_rate = f.getframerate()
            n_frames = f.getnframes()
            frames = f.readframes(n_frames)
            sample_width = f.getsampwidth()
            channels = f.getnchannels()
        # Decode raw frames into int samples then scale to [-1, 1].
        if sample_width == 2:
            samples = torch.frombuffer(frames, dtype=torch.int16).float() / 32768.0
        elif sample_width == 4:
            samples = torch.frombuffer(frames, dtype=torch.int32).float() / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        if channels > 1:
            samples = samples.view(-1, channels).mean(dim=-1)
        waveform = samples.contiguous()
    else:
        # Try ``soundfile`` for non-WAV formats.
        try:
            import soundfile as sf

            data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=-1)
            waveform = torch.from_numpy(data)
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot read {path.suffix} without the optional 'soundfile' package."
            ) from exc

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = _linear_resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def _linear_resample(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """Naive linear-interpolation resampling. Good enough for smoke tests."""

    if src_sr == dst_sr:
        return waveform
    duration = waveform.shape[-1] / src_sr
    n_dst = int(round(duration * dst_sr))
    src_positions = torch.arange(n_dst, dtype=torch.float32) * (src_sr / dst_sr)
    floor_idx = src_positions.floor().clamp(max=waveform.shape[-1] - 1).long()
    frac = (src_positions - floor_idx.float()).clamp(0.0, 1.0)
    ceil_idx = (floor_idx + 1).clamp(max=waveform.shape[-1] - 1)
    return waveform[floor_idx] * (1 - frac) + waveform[ceil_idx] * frac


def save_waveform(path: str | Path, waveform: torch.Tensor, *, sample_rate: int = 24000) -> None:
    """Write a 1-D float waveform out as 16-bit PCM WAV."""

    wf = waveform.detach().cpu().float().clamp(-1.0, 1.0)
    pcm = (wf * 32767.0).to(torch.int16).contiguous().numpy().tobytes()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm)
