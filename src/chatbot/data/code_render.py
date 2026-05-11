"""Render code strings into PNG images for Forge's screenshot-pretraining stage.

We use PIL's built-in TrueType engine and a simple "monospace, line-numbered,
syntax-coloured" theme. Real code IDEs render much prettier images, but for
the model's pretraining we just need a believable approximation of what a
screenshot looks like.

Two modes:

* :func:`render_code_image` — synchronous, returns a PIL image.
* :func:`iter_synthetic_screenshots` — generator that pairs code source
  files with rendered images, suitable for streaming into the training
  loop alongside text.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

from PIL import Image, ImageDraw, ImageFont


# Tiny built-in palette: (background, foreground, keyword, string, comment).
_THEMES = [
    {"bg": (30, 30, 30), "fg": (220, 220, 220), "kw": (190, 134, 198),
     "str": (152, 195, 121), "com": (128, 128, 128), "num": (209, 154, 102)},
    {"bg": (255, 255, 255), "fg": (60, 60, 60), "kw": (192, 39, 134),
     "str": (102, 153, 0), "com": (170, 170, 170), "num": (153, 102, 0)},
    {"bg": (38, 50, 56), "fg": (236, 239, 244), "kw": (143, 188, 187),
     "str": (235, 203, 139), "com": (105, 121, 130), "num": (180, 142, 173)},
]

_KEYWORDS = {
    "def", "class", "return", "if", "else", "elif", "for", "while", "import",
    "from", "as", "in", "not", "is", "and", "or", "with", "try", "except",
    "finally", "raise", "yield", "lambda", "pass", "break", "continue",
    "True", "False", "None", "self", "async", "await",
}

_TOKEN_RE = re.compile(r"(#[^\n]*|\".*?\"|'.*?'|\b\w+\b|\d+\.?\d*|\S)", re.DOTALL)


def _classify(token: str) -> str:
    """Return the theme-key under which a token should be coloured.

    Returns one of: ``"com"`` (comment), ``"str"`` (string literal),
    ``"kw"`` (Python keyword), ``"num"`` (numeric literal), or ``"fg"``
    (default foreground). The theme dict picks an RGB tuple from the key.
    This is a deliberately simple tokenizer — good enough to make rendered
    code *look* highlighted, not a real Python parser.
    """

    if token.startswith("#"):
        return "com"
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        return "str"
    if token in _KEYWORDS:
        return "kw"
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return "num"
    return "fg"


def _safe_font(size: int) -> ImageFont.ImageFont:
    """Return a TrueType monospace font if we can find one, else a bitmap fallback.

    PIL doesn't know what fonts are installed on the host; we try the
    common monospace TrueType names in order. If none are present (e.g.
    on a minimal Linux container without fontconfig) we fall back to
    PIL's built-in bitmap font, which is small but always works.
    """

    for name in ("Consolas.ttf", "DejaVuSansMono.ttf", "Menlo.ttc", "Courier.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_code_image(
    code: str,
    *,
    max_lines: int = 30,
    font_size: int = 14,
    line_height: int = 18,
    padding: int = 12,
    width: int = 720,
    theme: dict | None = None,
) -> Image.Image:
    """Render a code string into a PIL image."""

    theme = theme or random.choice(_THEMES)
    font = _safe_font(font_size)
    lines = code.splitlines()[:max_lines]
    height = padding * 2 + line_height * len(lines)
    img = Image.new("RGB", (width, max(height, 64)), color=theme["bg"])
    draw = ImageDraw.Draw(img)
    for row, line in enumerate(lines):
        y = padding + row * line_height
        # Line number gutter.
        gutter = f"{row + 1:>3} "
        draw.text((padding, y), gutter, font=font, fill=theme["com"])
        # Tokenize and colourise.
        gutter_width = font_size * len(gutter) // 2
        x = padding + gutter_width
        cursor = 0
        for match in _TOKEN_RE.finditer(line):
            start, end = match.span()
            # Draw whitespace between tokens.
            ws = line[cursor:start]
            if ws:
                draw.text((x, y), ws, font=font, fill=theme["fg"])
                x += int(font.getlength(ws)) if hasattr(font, "getlength") else font_size // 2 * len(ws)
            token = match.group(0)
            kind = _classify(token)
            draw.text((x, y), token, font=font, fill=theme[kind])
            x += int(font.getlength(token)) if hasattr(font, "getlength") else font_size // 2 * len(token)
            cursor = end
        if cursor < len(line):
            tail = line[cursor:]
            draw.text((x, y), tail, font=font, fill=theme["fg"])
    return img


def iter_synthetic_screenshots(
    source_dir: str | Path,
    *,
    glob: str = "**/*.py",
    max_lines: int = 30,
    image_size: int = 224,
) -> Iterator[Tuple[str, Image.Image]]:
    """Walk a directory of source code and yield ``(code_string, rendered_image)`` pairs."""

    for path in Path(source_dir).glob(glob):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        image = render_code_image(text, max_lines=max_lines)
        if image_size and image.size != (image_size, image_size):
            image = image.resize((image_size, image_size))
        yield text, image
