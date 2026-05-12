"""Chatbot — two original PyTorch LLMs (Aurora + Forge) plus runtime.

Aurora (~72B dense omni-modal: text + image + audio in, text + audio out)
and Forge (~460B / ~35B-active MoE coder + DevOps assistant with vision
and audio inputs) are the two flagship models. A tiny (~30M) config
shares the same architecture for smoke tests.

Sub-packages:

* ``chatbot.models``     — model code (Aurora, Forge, shared blocks,
  vision tower, audio encoder + codec).
* ``chatbot.tokenizer``  — BPE tokenizer + chat / tool / audio templates.
* ``chatbot.data``       — pretrain / SFT / DPO / tool-use / omni /
  DevOps data loaders, plus synthetic-data generators.
* ``chatbot.training``   — pretrain, long-context, SFT, DPO, tool-use,
  DevOps, RLEF, LoRA pipelines.
* ``chatbot.inference``  — generation, multimodal chat (text + image +
  audio + video), OpenAI-style HTTP server.
* ``chatbot.runtime``    — Claude-Code-style plugin / skill / hook layer
  with filesystem, shell, http, notebook, document, and DevOps tools.
* ``chatbot.eval``       — HumanEval, MBPP, LiveCodeBench, SWE-bench,
  MMLU, GSM8K runners.
* ``chatbot.utils``      — config loader, logging, seeding, memory
  accounting.

Nothing here imports torch at the top level; sub-packages do, so importing
``chatbot`` itself is cheap.
"""

__version__ = "2.0.0"
