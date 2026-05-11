"""LoRA / QLoRA fine-tuning entrypoint.

Loads a frozen base checkpoint, wraps the model with LoRA adapters on the
target modules listed in the config, and runs an SFT-style training loop.
Only the adapters get gradients — the base model stays frozen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch                                                    # noqa: E402
from torch.utils.data import DataLoader                         # noqa: E402

from chatbot.data.sft_loader import SFTDataset, SFTMixEntry, collate_sft_batch   # noqa: E402
from chatbot.tokenizer.bpe import BPETokenizer                  # noqa: E402
from chatbot.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from chatbot.training.lora import apply_lora_to_model, save_lora_adapters  # noqa: E402
from chatbot.training.metrics import RollingMean                # noqa: E402
from chatbot.training.optim import build_optimizer, build_scheduler  # noqa: E402
from chatbot.training.pretrain import build_model               # noqa: E402
from chatbot.utils.config import load_config, override_from_cli # noqa: E402
from chatbot.utils.logging import get_logger, setup_logging     # noqa: E402
from chatbot.utils.seeding import set_seed                      # noqa: E402

logger = get_logger("lora_finetune")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA / QLoRA fine-tune.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--training", default="lora")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-checkpoint", required=True,
                        help="Base model checkpoint to attach adapters to.")
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    setup_logging(level="INFO")
    model_cfg = load_config(f"models/{args.model}")
    train_cfg = load_config(f"training/{args.training}")
    train_cfg = override_from_cli(train_cfg, args.override)
    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_cfg).to(device)
    load_checkpoint(args.base_checkpoint, model=model)

    # Apply LoRA to the configured target modules. The function freezes the
    # base parameters and adds two small trainable matrices per target.
    apply_lora_to_model(
        model,
        target_modules=list(train_cfg.lora.target_modules),
        rank=int(train_cfg.lora.rank),
        alpha=int(train_cfg.lora.alpha),
        dropout=float(train_cfg.lora.dropout),
        init=str(train_cfg.lora.get("init_lora_weights", "kaiming")),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("LoRA trainable parameters: %.2fM", sum(p.numel() for p in trainable) / 1e6)

    tokenizer = BPETokenizer.from_file(args.tokenizer)
    mix = [SFTMixEntry(name=str(e["name"]), weight=float(e["weight"])) for e in train_cfg.data.mix]
    dataset = SFTDataset(
        mix=mix,
        tokenizer=tokenizer,
        block_size=int(train_cfg.sequence_length),
        pad_token_id=tokenizer.pad_id(),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.micro_batch_size),
        collate_fn=collate_sft_batch,
    )

    optimizer = build_optimizer(trainable, train_cfg.optimizer)
    scheduler = build_scheduler(optimizer, train_cfg.scheduler, total_steps=int(train_cfg.max_steps))

    output_dir = Path(train_cfg.get("output_dir", "outputs/lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rolling = RollingMean()
    model.train()
    step = 0
    for batch in loader:
        if step >= int(train_cfg.max_steps):
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(train_cfg.grad_clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        rolling.update(float(loss.item()))
        step += 1
        if step % int(train_cfg.get("log_every", 25)) == 0:
            logger.info("lora step %d | loss %.4f", step, rolling.mean)
        if step % int(train_cfg.get("save_every", 250)) == 0:
            save_lora_adapters(model, str(output_dir / f"step_{step:09d}"))

    save_lora_adapters(model, str(output_dir / "final"))
    save_checkpoint(output_dir=str(output_dir), step=step, model=model, optimizer=None, scheduler=None,
                    config=model_cfg)


if __name__ == "__main__":
    main()
