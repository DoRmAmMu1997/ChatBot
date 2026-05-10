"""Interactive inference for the trained Transformer chatbot."""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from .config import ModelConfig
from .model import TransformerChatModel
from .tokenizer import BOT_TOKEN, BOS_TOKEN, EOS_TOKEN, USER_TOKEN, tokenizer_from_dict


History = List[Tuple[str, str]]


def load_chatbot(checkpoint_path: str, cpu: bool = False):
    """Load the model, tokenizer, and device from a checkpoint file.

    Inference must recreate the same model and tokenizer used during training.
    If the vocabulary or model size changes, the saved weights no longer line
    up with the code.
    """

    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # The checkpoint contains both the learned weights and the "recipe" needed
    # to rebuild the model class before loading those weights.
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = TransformerChatModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer, device


def build_prompt(message: str, history: History, max_history: int = 3) -> str:
    """Create the text prompt that is sent into the language model.

    The model was trained on text containing <user> and <bot> markers, so chat
    uses the same format. A little recent history gives the model context while
    keeping the prompt short enough for the context window.
    """

    pieces = [BOS_TOKEN]
    for user_text, bot_text in history[-max_history:]:
        pieces.extend([USER_TOKEN, user_text, BOT_TOKEN, bot_text])
    pieces.extend([USER_TOKEN, message, BOT_TOKEN])
    return " ".join(pieces)


@torch.no_grad()
def generate_reply(
    model: TransformerChatModel,
    tokenizer,
    message: str,
    device: torch.device,
    history: History | None = None,
    max_new_tokens: int = 48,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
    do_sample: bool = True,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate one response for a user message."""

    history = history or []
    prompt = build_prompt(message, history)

    # If the prompt is longer than the model's context window, keep the most
    # recent tokens. Recent conversation is usually the most relevant context.
    prompt_ids = tokenizer.encode(prompt)[-model.config.block_size :]

    # PyTorch models expect batches, so wrap the one prompt in a batch of size 1.
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )[0].tolist()

    new_ids = output_ids[len(prompt_ids) :]

    # Stop when the model generates a boundary marker. For example, <user>
    # usually means it is about to start inventing the next user turn.
    stop_ids = {
        tokenizer.eos_id,
        tokenizer.bos_id,
        tokenizer.user_id,
        tokenizer.bot_id,
        tokenizer.pad_id,
    }

    response_ids = []
    for token_id in new_ids:
        if token_id in stop_ids:
            break
        response_ids.append(token_id)

    response = tokenizer.decode(response_ids)
    # Very early checkpoints can generate only stop tokens. Return a friendly
    # fallback instead of printing a blank line.
    return response or "I am not sure how to answer that yet."


def chat_loop(
    model: TransformerChatModel,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    do_sample: bool,
    num_beams: int,
    repetition_penalty: float,
) -> None:
    """Run a terminal chat loop until the user quits."""

    history: History = []
    print("Chatbot ready. Type 'quit' or 'q' to stop.")
    while True:
        message = input("> ").strip()
        if message.lower() in {"q", "quit", "exit"}:
            break

        # Keep the conversation history in memory so each next reply can see a
        # few recent turns. This is not long-term memory; it is prompt context.
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            message=message,
            device=device,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        history.append((message, reply))
        print(f"Bot: {reply}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the chat CLI."""

    parser = argparse.ArgumentParser(description="Chat with a trained small LLM checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/chatbot-small-llm.pt")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA is available.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling and always choose the highest-scoring token.")
    return parser


def main() -> None:
    """CLI entrypoint used by chatbot.py and chat_llm.py."""

    args = build_arg_parser().parse_args()
    model, tokenizer, device = load_chatbot(args.checkpoint, cpu=args.cpu)
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
