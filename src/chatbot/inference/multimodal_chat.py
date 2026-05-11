"""Multimodal generation for Aurora and Forge.

Handles four input shapes, in any combination:

* text     — plain string content.
* images   — PIL ``Image`` objects; consumed by the vision tower at the
  positions where ``<|image|>`` appears.
* videos   — file paths; we sample N frames and route each frame as an
  ``<|image|>``. Aurora and Forge both have a vision tower so both
  support video.
* audio    — file paths or float waveform tensors; consumed by the audio
  encoder at the positions where ``<|audio|>`` appears.

For *audio output*: :func:`generate_with_audio_output` runs the LLM, then
collects any audio-code span the model emits between ``<|audio_start|>``
and ``<|audio_end|>`` and decodes each span through :class:`AudioCodec`
to produce a waveform. Returns both the cleaned-up text and the decoded
waveforms in span order.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image

from ..data.image_processing import ImagePreprocessor, batch_images
from ..data.video_processing import sample_video_frames
from ..tokenizer.audio_template import extract_audio_spans
from ..tokenizer.chat_template import format_messages

AUDIO_MARKER = "<|audio|>"
IMAGE_MARKER = "<|image|>"


def _waveform_to_audio_tokens(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """Pad / resample a 1-D waveform to the audio encoder's expected rate."""

    if sample_rate != target_sample_rate:
        from ..data.audio_processing import _linear_resample

        waveform = _linear_resample(waveform, sample_rate, target_sample_rate)
    return waveform.contiguous()


def _expand_modality_markers(
    messages: List[dict],
    image_count: int,
    image_tokens_each: int,
    audio_count: int,
    audio_tokens_each: int,
) -> List[dict]:
    """Inflate each ``<|image|>`` / ``<|audio|>`` placeholder to N copies.

    Each modality embedding produced by the encoders takes ``N`` token
    slots in the LLM stream. We pre-write that many placeholder tokens
    so the splice step has a target to overwrite.
    """

    rendered: List[dict] = []
    image_seen = 0
    audio_seen = 0
    for msg in messages:
        content = msg.get("content", "")
        while IMAGE_MARKER in content and image_seen < image_count:
            content = content.replace(IMAGE_MARKER, IMAGE_MARKER * image_tokens_each, 1)
            image_seen += 1
        while AUDIO_MARKER in content and audio_seen < audio_count:
            content = content.replace(AUDIO_MARKER, AUDIO_MARKER * audio_tokens_each, 1)
            audio_seen += 1
        rendered.append({**msg, "content": content})
    return rendered


def generate_multimodal(
    model,
    tokenizer,
    messages: List[dict],
    *,
    images: Optional[Sequence[Image.Image]] = None,
    videos: Optional[Sequence[str]] = None,
    audio: Optional[Sequence[torch.Tensor]] = None,
    image_size: Optional[int] = None,
    video_frames_per_clip: int = 8,
    audio_sample_rate: int = 16000,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 0,
) -> str:
    """Generate a text reply conditioned on any mix of modalities.

    See :func:`generate_with_audio_output` for the variant that also returns
    decoded audio.
    """

    device = next(model.parameters()).device

    # Step 1: expand any videos into individual frames (each frame = one image
    # marker). This keeps the rest of the code path identical for images vs
    # videos.
    extra_images: List[Image.Image] = []
    if videos:
        for path in videos:
            extra_images.extend(sample_video_frames(path, num_frames=video_frames_per_clip))
    all_images: List[Image.Image] = list(images or []) + extra_images

    # Step 2: figure out per-modality token counts from the model config.
    image_tokens_each = int(getattr(model.config.vision, "num_image_tokens", 0))
    image_size = image_size or int(getattr(model.config.vision, "image_size", 384))

    audio_clips: List[torch.Tensor] = []
    audio_tokens_each = 0
    if audio:
        # Rough estimate: the encoder downsamples 16 kHz audio by ~4x in conv
        # plus the conformer doesn't change time length. So ~50 frames/sec
        # of audio. We compute the exact count by running the encoder once.
        # For prompt-formatting purposes we need a number up front; we
        # generously over-allocate to the *max* clip length.
        target_sr = int(model.config.audio.sample_rate)
        max_samples = max(c.numel() for c in audio)
        # Padding to the longest clip means each <|audio|> marker reserves
        # ``ceil(max_samples / (sample_rate / 50))`` tokens approximately.
        audio_tokens_each = max(1, max_samples * 50 // target_sr)
        for clip in audio:
            audio_clips.append(_waveform_to_audio_tokens(clip, audio_sample_rate, target_sr))

    # Step 3: build the prompt with inflated markers.
    rendered_messages = _expand_modality_markers(
        messages,
        image_count=len(all_images), image_tokens_each=image_tokens_each,
        audio_count=len(audio_clips), audio_tokens_each=audio_tokens_each,
    )
    prompt = format_messages(rendered_messages, add_generation_prompt=True)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    # Step 4: build modality tensors (if any).
    image_tensor = None
    if all_images:
        pre = ImagePreprocessor(image_size=image_size)
        image_tensor = batch_images(list(all_images), pre).to(device)

    audio_tensor = None
    if audio_clips:
        # Pad all clips to the same length (so they can be stacked into a
        # single tensor). Trailing zeros are fine; the encoder ignores
        # them.
        max_len = max(c.numel() for c in audio_clips)
        padded = [torch.nn.functional.pad(c, (0, max_len - c.numel())) for c in audio_clips]
        audio_tensor = torch.stack(padded, dim=0).to(device)

    # Step 5: generate.
    out = model.generate(
        prompt_ids,
        images=image_tensor,
        audio=audio_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    new_ids = out[0, prompt_ids.shape[1] :].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def generate_with_audio_output(
    model,
    tokenizer,
    codec,
    messages: List[dict],
    *,
    images: Optional[Sequence[Image.Image]] = None,
    videos: Optional[Sequence[str]] = None,
    audio: Optional[Sequence[torch.Tensor]] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 0,
) -> Tuple[str, List[torch.Tensor]]:
    """Generate a reply that may include spoken audio.

    Returns:
        ``(text, audio_waveforms)`` — ``text`` has each audio span replaced
        by a short ``[audio: K frames]`` marker; ``audio_waveforms`` is a list
        of 1-D float tensors (one per audio span emitted by the model).
    """

    device = next(model.parameters()).device

    # Reuse the multimodal prompt-building logic but ask for the raw decode
    # so we can spot the audio markers ourselves.
    # We replicate the early steps inline so we can keep special tokens in
    # the decoded string.
    image_tokens_each = int(getattr(model.config.vision, "num_image_tokens", 0))
    image_size = int(getattr(model.config.vision, "image_size", 384))
    extra_images: List[Image.Image] = []
    if videos:
        for path in videos:
            extra_images.extend(sample_video_frames(path, num_frames=8))
    all_images: List[Image.Image] = list(images or []) + extra_images

    audio_clips: List[torch.Tensor] = []
    audio_tokens_each = 0
    if audio:
        target_sr = int(model.config.audio.sample_rate)
        max_samples = max(c.numel() for c in audio)
        audio_tokens_each = max(1, max_samples * 50 // target_sr)
        for clip in audio:
            audio_clips.append(_waveform_to_audio_tokens(clip, 16000, target_sr))

    rendered_messages = _expand_modality_markers(
        messages,
        image_count=len(all_images), image_tokens_each=image_tokens_each,
        audio_count=len(audio_clips), audio_tokens_each=audio_tokens_each,
    )
    prompt = format_messages(rendered_messages, add_generation_prompt=True)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    image_tensor = batch_images(list(all_images), ImagePreprocessor(image_size=image_size)).to(device) if all_images else None
    audio_tensor = None
    if audio_clips:
        max_len = max(c.numel() for c in audio_clips)
        padded = [torch.nn.functional.pad(c, (0, max_len - c.numel())) for c in audio_clips]
        audio_tensor = torch.stack(padded, dim=0).to(device)

    out = model.generate(
        prompt_ids,
        images=image_tensor,
        audio=audio_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    new_ids = out[0, prompt_ids.shape[1] :].tolist()
    decoded_full = tokenizer.decode(new_ids, skip_special_tokens=False)

    # Extract audio spans. We pass in a `{token_name: codebook_index}` dict
    # so the function can normalize without re-querying the tokenizer.
    audio_token_map = {f"<audio:{i}>": i for i in range(int(model.config.audio.num_audio_codes))}
    clean_text, code_sequences = extract_audio_spans(decoded_full, audio_token_map)

    # Decode each audio span through the codec.
    waveforms: List[torch.Tensor] = []
    for codes in code_sequences:
        if not codes:
            continue
        code_tensor = torch.tensor([codes], dtype=torch.long, device=device)
        waveforms.append(codec.decode(code_tensor)[0].detach().cpu())

    return clean_text.strip(), waveforms
