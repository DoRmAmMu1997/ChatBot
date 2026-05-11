"""Video frame sampling — synthetic animated GIF round-trip."""

from __future__ import annotations

from PIL import Image

from chatbot.data.video_processing import _uniform_indices, sample_video_frames


def test_uniform_indices_evenly_spaced():
    # Banker's rounding makes the midpoints favour the even integer, so
    # _uniform_indices(10, 5) gives 0, 2, 4, 7, 9 rather than 0, 2, 5, 7, 9.
    assert _uniform_indices(10, 5) == [0, 2, 4, 7, 9]
    assert _uniform_indices(3, 8) == [0, 1, 2]
    assert _uniform_indices(0, 5) == []
    # First and last index always present.
    five = _uniform_indices(100, 5)
    assert five[0] == 0 and five[-1] == 99
    assert len(five) == 5


def test_sample_video_frames_from_gif(tmp_path):
    # Build a tiny animated GIF with 4 frames so we can test sampling.
    frames = []
    for tone in (0, 80, 160, 240):
        frames.append(Image.new("RGB", (32, 32), color=(tone, tone, tone)))
    gif_path = tmp_path / "demo.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    sampled = sample_video_frames(gif_path, num_frames=2)
    assert len(sampled) == 2
    assert all(isinstance(f, Image.Image) for f in sampled)
