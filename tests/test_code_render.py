"""Sanity check for the synthetic code-screenshot renderer."""

from __future__ import annotations

from PIL import Image

from chatbot.data.code_render import render_code_image


def test_render_returns_image_with_size():
    code = "def hello():\n    print('hi')\n"
    img = render_code_image(code, max_lines=2, width=320)
    assert isinstance(img, Image.Image)
    assert img.size[0] == 320
    assert img.size[1] >= 32
