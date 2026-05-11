"""Shape tests for the audio encoder + mel feature extractor."""

from __future__ import annotations

import torch

from chatbot.models.audio.encoder import AudioEncoder
from chatbot.models.audio.mel import LogMelSpectrogram


def test_log_mel_shape():
    mel = LogMelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
    waveform = torch.randn(2, 16000)         # 1 second per item in batch
    out = mel(waveform)
    assert out.shape[0] == 2
    assert out.shape[1] == 80                # n_mels
    assert out.shape[2] > 0                  # some number of frames


def test_audio_encoder_projects_to_llm_dim():
    enc = AudioEncoder(
        llm_dim=64, n_mels=80, dim=64, depth=2, num_heads=4, ffn_hidden=128,
        sample_rate=16000, n_fft=400, hop_length=160,
    )
    waveform = torch.randn(1, 16000)
    out = enc(waveform)
    assert out.shape[0] == 1
    assert out.shape[2] == 64                # llm_dim
    assert out.shape[1] > 0                  # some number of tokens
