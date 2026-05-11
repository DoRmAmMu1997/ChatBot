"""Multimodal generation for Aurora.

You pass a list of messages where text content may contain ``<|image|>``
markers; for every marker we expect exactly one PIL image in ``images``
(in order). The image is preprocessed and routed through the vision tower
before the LLM ever sees it.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image

from ..data.image_processing import ImagePreprocessor, batch_images
from ..tokenizer.chat_template import format_messages


def generate_multimodal(
    model,
    tokenizer,
    messages: List[dict],
    images: Optional[List[Image.Image]] = None,
    *,
    image_size: int = 384,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 0,
) -> str:
    """Generate a reply that conditions on text + (optional) images."""

    device = next(model.parameters()).device

    if not images:
        # No images? Fall back to pure-text path.
        prompt = format_messages(messages, add_generation_prompt=True)
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        new_ids = out[0, prompt_ids.shape[1] :].tolist()
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    # Tile the prompt with N copies of the special <|image|> token so every
    # image has ``num_image_tokens`` reserved slots in the embedding stream.
    num_image_tokens = int(model.config.vision.num_image_tokens)
    image_marker = "<|image|>"
    replacement = image_marker * num_image_tokens
    rendered_messages = []
    img_counter = 0
    for msg in messages:
        content = msg.get("content", "")
        # Replace each "<|image|>" with the inflated version.
        while image_marker in content and img_counter < len(images):
            content = content.replace(image_marker, replacement, 1)
            img_counter += 1
        rendered_messages.append({**msg, "content": content})
    prompt = format_messages(rendered_messages, add_generation_prompt=True)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    pre = ImagePreprocessor(image_size=image_size)
    img_tensor = batch_images(images, pre).to(device)

    out = model.generate(
        prompt_ids,
        images=img_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    new_ids = out[0, prompt_ids.shape[1] :].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)
