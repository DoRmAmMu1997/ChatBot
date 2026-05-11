"""Pretrain the audio codec (waveform → discrete codes → waveform).

Loss = L1 reconstruction on the waveform + VQ commitment. Once the codec
is good enough that its codes are meaningful, the LLM is taught to emit
those same codes (during the omni-SFT stage), and at inference time we
look up codebook entries and run the codec's decoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..models.audio.codec import AudioCodec, AudioCodecConfig
from ..utils.config import load_config, override_from_cli, save_config
from ..utils.logging import get_logger, setup_logging
from ..utils.seeding import set_seed
from .checkpoint import save_checkpoint
from .metrics import RollingMean
from .optim import build_optimizer, build_scheduler

logger = get_logger(__name__)


def _iter_waveforms(cfg, target_sample_rate: int):
    """Stream waveforms from the configured dataset mix."""

    from ..data.registry import get_dataset

    for entry in cfg.data.mix:
        try:
            spec = get_dataset(str(entry["name"]))
        except KeyError:
            continue
        for row in spec.loader():
            audio = row.get("audio")
            if audio is None:
                continue
            # HF audio datasets return ``{'array': np.ndarray, 'sampling_rate': int}``.
            if isinstance(audio, dict) and "array" in audio:
                wave = torch.tensor(audio["array"], dtype=torch.float32)
                sr = int(audio["sampling_rate"])
            else:
                continue
            if sr != target_sample_rate:
                # Quick linear resample; production should use a proper resampler.
                from ..data.audio_processing import _linear_resample

                wave = _linear_resample(wave, sr, target_sample_rate)
            yield wave


def run_codec_pretrain(cfg) -> Path:
    setup_logging(level="INFO")
    set_seed(int(cfg.get("seed", 42)))

    codec_cfg = AudioCodecConfig()  # tweak via CLI if needed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = AudioCodec(codec_cfg).to(device)

    optimizer = build_optimizer(codec.parameters(), cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps=int(cfg.max_steps))
    rolling = RollingMean()
    output_dir = Path(cfg.get("output_dir", "outputs/codec_pretrain"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "train_config.yaml")

    codec.train()
    step = 0
    for waveform in _iter_waveforms(cfg, codec_cfg.sample_rate):
        if step >= int(cfg.max_steps):
            break
        # Crop/pad to a fixed length so we can batch easily.
        target_len = codec_cfg.sample_rate * int(cfg.get("clip_seconds", 4))
        if waveform.shape[-1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[:target_len]
        batch = waveform.unsqueeze(0).to(device)

        out = codec(batch)
        loss = out["recon_loss"] + 0.25 * out["vq_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), float(cfg.grad_clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        rolling.update(float(loss.item()))
        step += 1
        if step % int(cfg.get("log_every", 25)) == 0:
            logger.info("codec step %d | loss %.4f", step, rolling.mean)
        if step % int(cfg.get("save_every", 1000)) == 0:
            save_checkpoint(output_dir, step=step, model=codec, optimizer=optimizer,
                            scheduler=scheduler, config=cfg)

    save_checkpoint(output_dir, step=step, model=codec, optimizer=optimizer,
                    scheduler=scheduler, config=cfg)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the audio codec autoencoder.")
    parser.add_argument("--training", default="audio_codec_pretrain")
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()
    cfg = load_config(f"training/{args.training}")
    cfg = override_from_cli(cfg, args.override)
    run_codec_pretrain(cfg)
