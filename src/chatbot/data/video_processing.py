"""Sample N frames from a video file so it can be fed through the vision tower.

We deliberately keep this dependency-light: ``imageio`` is the preferred
reader (it lives behind ``pillow`` already), and we fall back to PIL's
multi-frame iterator for animated GIF / WebP / PNG. Real video files (MP4,
MOV) need ``imageio`` with ``imageio-ffmpeg`` — install if you'll feed
those, but they're not required to import this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from PIL import Image

from .image_processing import ImagePreprocessor, batch_images


def _uniform_indices(total: int, k: int) -> List[int]:
    """Return ``k`` evenly-spaced indices in ``[0, total - 1]``."""

    if total <= 0:
        return []
    if k >= total:
        return list(range(total))
    return [int(round(i * (total - 1) / max(k - 1, 1))) for i in range(k)]


def sample_video_frames(path: str | Path, *, num_frames: int = 8) -> List[Image.Image]:
    """Return ``num_frames`` PIL images sampled uniformly from a video file."""

    path = Path(path)

    # First try ``imageio``. It handles MP4/MOV/AVI/WebM via ffmpeg if
    # ``imageio-ffmpeg`` is installed; it handles GIF/WebP/animated PNG
    # standalone.
    try:
        import imageio.v3 as iio

        all_frames = []
        for frame in iio.imiter(str(path)):
            all_frames.append(frame)
        indices = _uniform_indices(len(all_frames), num_frames)
        return [Image.fromarray(all_frames[i]) for i in indices]
    except (ImportError, Exception):  # noqa: BLE001 — fall back to PIL multi-frame
        pass

    # PIL fallback: works for animated GIF / WebP / animated PNG.
    with Image.open(path) as img:
        n_frames = getattr(img, "n_frames", 1)
        indices = _uniform_indices(n_frames, num_frames)
        frames: List[Image.Image] = []
        for idx in indices:
            img.seek(idx)
            frames.append(img.convert("RGB").copy())
        return frames


def video_to_tensor(
    path: str | Path,
    *,
    num_frames: int = 8,
    image_size: int = 224,
) -> torch.Tensor:
    """Sample frames from a video file and return a tensor ``[N, 3, H, W]``."""

    frames = sample_video_frames(path, num_frames=num_frames)
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    pre = ImagePreprocessor(image_size=image_size)
    return batch_images(frames, pre)
