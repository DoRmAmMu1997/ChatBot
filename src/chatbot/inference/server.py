"""Minimal HTTP server that mimics the OpenAI Chat Completions schema.

Not a vLLM replacement — single worker, single request at a time, no
continuous batching. Useful for local tooling that already speaks the
OpenAI wire format.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

import torch

from ..tokenizer.bpe import BPETokenizer
from ..tokenizer.chat_template import format_messages
from ..training.checkpoint import load_checkpoint
from ..utils.config import load_config
from .generate import _build_model


def _build_handler(model, tokenizer, runtime_cfg):
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — required name
            if self.path != "/v1/chat/completions":
                self.send_error(404, "Unknown path")
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload: Dict[str, Any] = json.loads(body)
            messages = payload.get("messages") or []
            prompt = format_messages(messages, add_generation_prompt=True)
            ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long,
                               device=next(model.parameters()).device)
            out = model.generate(
                ids,
                max_new_tokens=int(payload.get("max_tokens", runtime_cfg.get("max_new_tokens", 256))),
                temperature=float(payload.get("temperature", runtime_cfg.get("temperature", 0.7))),
                top_p=float(payload.get("top_p", runtime_cfg.get("top_p", 0.95))),
                top_k=int(payload.get("top_k", runtime_cfg.get("top_k", 0))),
            )
            new_ids = out[0, ids.shape[1] :].tolist()
            reply = tokenizer.decode(new_ids, skip_special_tokens=True)
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "chatbot"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": reply},
                    }
                ],
            }
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):  # silence default logger
            pass

    return _Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a chatbot model over HTTP (OpenAI-style).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--runtime", default="default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_cfg = load_config(f"models/{args.model}")
    runtime_cfg = load_config(f"runtime/{args.runtime}")
    model = _build_model(model_cfg)
    load_checkpoint(args.checkpoint, model=model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    tokenizer = BPETokenizer.from_file(args.tokenizer)
    handler = _build_handler(model, tokenizer, runtime_cfg)
    print(f"Listening on http://{args.host}:{args.port}")
    HTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
