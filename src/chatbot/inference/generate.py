"""Text generation entry point used by ``scripts/chat.py``.

Wraps a tokenizer + model + runtime config to provide a one-call interface:

    text = generate_text(model, tokenizer, prompt, runtime_cfg)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch

from ..models.aurora_50b import AuroraForCausalLM, aurora_config_from_yaml
from ..models.forge_250b import ForgeForCausalLM, forge_config_from_yaml
from ..tokenizer.bpe import BPETokenizer
from ..tokenizer.chat_template import format_messages
from ..training.checkpoint import load_checkpoint
from ..utils.config import load_config


def _build_model(model_cfg) -> torch.nn.Module:
    family = str(model_cfg.get("family", "aurora")).lower()
    if family == "aurora":
        return AuroraForCausalLM(aurora_config_from_yaml(model_cfg))
    if family == "forge":
        return ForgeForCausalLM(forge_config_from_yaml(model_cfg))
    raise ValueError(f"Unknown model family: {family!r}")


def generate_text(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 0,
    eos_token_ids: Optional[List[int]] = None,
) -> str:
    """Sample text from a model given a string prompt."""

    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_ids=eos_token_ids or [tokenizer.eos_id()],
    )
    new_ids = out[0, ids.shape[1] :].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def cli() -> None:
    """``chatbot-chat`` script entry. Interactive single-turn loop."""

    parser = argparse.ArgumentParser(description="Chat with a checkpoint.")
    parser.add_argument("--model", required=True, help="Model config name (e.g. tiny).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--runtime", default="default", help="Runtime config name.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    model_cfg = load_config(f"models/{args.model}")
    runtime_cfg = load_config(f"runtime/{args.runtime}")

    model = _build_model(model_cfg)
    load_checkpoint(args.checkpoint, model=model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    tokenizer = BPETokenizer.from_file(args.tokenizer)

    system_prompt = str(runtime_cfg.get("system_prompt", "")).strip()
    history: List[dict] = [{"role": "system", "content": system_prompt}] if system_prompt else []

    print("Type a message, or 'quit' to exit.")
    while True:
        try:
            user = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if user.lower() in {"quit", "exit", "q"}:
            break
        history.append({"role": "user", "content": user})
        prompt = format_messages(history, add_generation_prompt=True)
        reply = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens or int(runtime_cfg.get("max_new_tokens", 256)),
            temperature=args.temperature if args.temperature is not None else float(runtime_cfg.get("temperature", 0.7)),
            top_p=args.top_p if args.top_p is not None else float(runtime_cfg.get("top_p", 0.95)),
            top_k=args.top_k if args.top_k is not None else int(runtime_cfg.get("top_k", 0)),
        )
        history.append({"role": "assistant", "content": reply})
        print(f"bot> {reply}")


if __name__ == "__main__":
    cli()
