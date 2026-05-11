"""Helpers for audio token sequences inside chat messages.

The contract for audio output:

  * The model emits ``<|audio_start|>`` followed by a run of audio
    token ids (each in ``<audio:0>`` … ``<audio:4095>``) followed by
    ``<|audio_end|>``.
  * The inference runtime collects those token ids, looks them up in
    :class:`chatbot.models.audio.AudioCodec`'s codebook, and decodes
    the resulting frame sequence into a waveform.

For audio input we use a different shape — the ``<|audio|>`` placeholder
in the prompt text is replaced *by the runtime* with the audio
encoder's projected tokens (one placeholder = many tokens, like images).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

AUDIO_INPUT_PLACEHOLDER = "<|audio|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"


def audio_token_name(index: int) -> str:
    """Return the codebook-id token name, e.g. ``"<audio:42>"``."""

    return f"<audio:{index}>"


def audio_token_index(token: str) -> Optional[int]:
    """Inverse of :func:`audio_token_name`. Returns ``None`` if not an audio token."""

    m = re.fullmatch(r"<audio:(\d+)>", token)
    return int(m.group(1)) if m else None


def extract_audio_spans(
    decoded_text: str,
    audio_token_to_index: dict,
) -> Tuple[str, List[List[int]]]:
    """Split a decoded model output into (clean_text, list_of_audio_id_sequences).

    ``decoded_text`` here is the *non-special-skipped* decoded output of the
    LLM — so it contains the ``<|audio_start|>`` / ``<|audio_end|>`` markers
    and individual ``<audio:N>`` tokens as plain text.

    Returns:
        clean_text: same text but with each audio span replaced by a small
        ``[audio: N samples]`` marker.
        audio_id_sequences: per-span list of ``[code0, code1, ...]`` integers
        ready to feed into :meth:`AudioCodec.decode`.
    """

    spans: List[List[int]] = []
    pieces: List[str] = []
    cursor = 0
    while True:
        start = decoded_text.find(AUDIO_START, cursor)
        if start == -1:
            pieces.append(decoded_text[cursor:])
            break
        pieces.append(decoded_text[cursor:start])
        end = decoded_text.find(AUDIO_END, start)
        if end == -1:
            # Unterminated audio span — keep the rest as text and stop.
            pieces.append(decoded_text[start:])
            break
        span_body = decoded_text[start + len(AUDIO_START) : end]
        codes: List[int] = []
        for token in re.findall(r"<audio:\d+>", span_body):
            idx = audio_token_to_index.get(token)
            if idx is None:
                continue
            # Re-derive the codebook index from the token name — far cheaper
            # than another lookup table at runtime.
            num = audio_token_index(token)
            if num is not None:
                codes.append(num)
        spans.append(codes)
        pieces.append(f"[audio: {len(codes)} frames]")
        cursor = end + len(AUDIO_END)

    return "".join(pieces), spans
