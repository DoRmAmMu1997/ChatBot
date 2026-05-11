"""Audio subsystem — encoder (for speech in) and codec (for speech out).

Two halves, both shared between Aurora and Forge:

* :class:`AudioEncoder` — turns a log-mel spectrogram into a sequence of
  soft embeddings that the LLM treats just like text token embeddings
  (analogous to the vision tower's role for images).
* :class:`AudioCodec` — a small generative decoder that turns *discrete
  audio tokens* (which the LLM emits between ``<|audio_start|>`` and
  ``<|audio_end|>``) back into a waveform. Single-codebook design so the
  LLM's output vocabulary stays uniform — text tokens and audio tokens
  share one cross-entropy loss.

Both are deliberately small relative to the LLM (~150 M for the encoder,
~120 M for the codec). The LLM does the heavy lifting; these modules just
translate between waveform and token space.
"""

from .codec import AudioCodec
from .encoder import AudioEncoder
from .mel import LogMelSpectrogram

__all__ = ["AudioEncoder", "AudioCodec", "LogMelSpectrogram"]
