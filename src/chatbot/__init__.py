"""Chatbot — two original PyTorch LLMs (Aurora-50B + Forge-250B) plus runtime.

This is the top-level package. Sub-packages:

* ``chatbot.models``     — model code (Aurora-50B, Forge-250B, shared blocks).
* ``chatbot.tokenizer``  — BPE tokenizer + chat / tool templates.
* ``chatbot.data``       — pretrain / SFT / DPO / tool-use data loaders.
* ``chatbot.training``   — pretrain, SFT, DPO, LoRA training pipelines.
* ``chatbot.inference``  — generation, multimodal chat, OpenAI-style HTTP server.
* ``chatbot.runtime``    — Claude-Code-style plugin / skill / hook layer.
* ``chatbot.eval``       — HumanEval, MBPP, SWE-bench, MMLU, GSM8K runners.
* ``chatbot.utils``      — config loader, logging, seeding, memory accounting.

Nothing here imports torch at the top level; sub-packages do, so importing
``chatbot`` itself is cheap.
"""

__version__ = "1.0.0"
