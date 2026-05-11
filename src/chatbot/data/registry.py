"""Catalogue of supported datasets.

Each entry says where a dataset lives, what license governs it, what stage
(pretrain / sft / dpo / tool_use / image_text / long_context) it belongs to,
and which loader function knows how to convert it into ``{prompt, response}``
or raw text records.

Loaders are lazy — we only import Hugging Face ``datasets`` when one of these
loaders is actually called. That keeps ``import chatbot.data`` snappy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_repo: Optional[str]
    license: str
    stage: str                                  # "pretrain" | "sft" | "dpo" | "tool_use" | "image_text" | "long_context"
    description: str
    loader: Callable[..., Iterable[dict]]
    default_split: str = "train"
    tags: List[str] = field(default_factory=list)


# ---------- lazy loader factories ----------------------------------------

def _hf_text_loader(repo: str, text_key: str = "text"):
    def _load(split: str = "train", streaming: bool = True, **kwargs):
        from datasets import load_dataset

        ds = load_dataset(repo, split=split, streaming=streaming, **kwargs)
        for row in ds:
            text = row.get(text_key)
            if text:
                yield {"text": text}
    return _load


def _hf_chat_loader(repo: str, messages_key: str = "messages"):
    def _load(split: str = "train", streaming: bool = True, **kwargs):
        from datasets import load_dataset

        ds = load_dataset(repo, split=split, streaming=streaming, **kwargs)
        for row in ds:
            yield {"messages": row.get(messages_key) or []}
    return _load


def _hf_preference_loader(repo: str):
    def _load(split: str = "train", streaming: bool = True, **kwargs):
        from datasets import load_dataset

        ds = load_dataset(repo, split=split, streaming=streaming, **kwargs)
        for row in ds:
            yield {
                "prompt": row.get("prompt") or row.get("question") or "",
                "chosen": row.get("chosen") or row.get("response_a") or "",
                "rejected": row.get("rejected") or row.get("response_b") or "",
            }
    return _load


# ---------- the actual registry ------------------------------------------

DATASETS: Dict[str, DatasetSpec] = {}


def _register(spec: DatasetSpec) -> None:
    DATASETS[spec.name] = spec


# Pretrain text.
_register(DatasetSpec(
    name="fineweb_edu",
    hf_repo="HuggingFaceFW/fineweb-edu",
    license="ODC-By 1.0",
    stage="pretrain",
    description="Filtered, educational-quality slice of CommonCrawl. The Aurora workhorse.",
    loader=_hf_text_loader("HuggingFaceFW/fineweb-edu", "text"),
))
_register(DatasetSpec(
    name="the_stack_v2",
    hf_repo="bigcode/the-stack-v2-dedup",
    license="permissive (per-file)",
    stage="pretrain",
    description="Source code from GitHub, deduplicated and license-filtered.",
    loader=_hf_text_loader("bigcode/the-stack-v2-dedup", "content"),
))
_register(DatasetSpec(
    name="wikipedia",
    hf_repo="wikimedia/wikipedia",
    license="CC-BY-SA 3.0",
    stage="pretrain",
    description="High-quality encyclopedic prose.",
    loader=_hf_text_loader("wikimedia/wikipedia", "text"),
))
_register(DatasetSpec(
    name="books",
    hf_repo="manu/project_gutenberg",
    license="public domain",
    stage="pretrain",
    description="Project Gutenberg corpus (English).",
    loader=_hf_text_loader("manu/project_gutenberg", "text"),
))
_register(DatasetSpec(
    name="arxiv",
    hf_repo="ccdv/arxiv-classification",
    license="various",
    stage="pretrain",
    description="Scientific papers from arXiv.",
    loader=_hf_text_loader("ccdv/arxiv-classification", "text"),
))

# Multimodal pretrain.
_register(DatasetSpec(
    name="laion_recaptioned",
    hf_repo="laion/laion-coco",
    license="CC-BY 4.0",
    stage="image_text",
    description="LAION subset with synthetic re-captions for cleaner alignment.",
    loader=_hf_chat_loader("laion/laion-coco", "messages"),
))

# Long-context.
_register(DatasetSpec(
    name="long_books",
    hf_repo="emozilla/long_books",
    license="various",
    stage="long_context",
    description="Long-form books for context-extension training.",
    loader=_hf_text_loader("emozilla/long_books", "text"),
))
_register(DatasetSpec(
    name="arxiv_long",
    hf_repo="THUDM/LongBench",
    license="MIT",
    stage="long_context",
    description="Long arXiv articles for context-extension.",
    loader=_hf_text_loader("THUDM/LongBench", "text"),
))
_register(DatasetSpec(
    name="github_repo_packed",
    hf_repo="bigcode/the-stack-v2-dedup-repos",
    license="permissive",
    stage="long_context",
    description="Repo-level packed sequences from GitHub (Forge-heavy).",
    loader=_hf_text_loader("bigcode/the-stack-v2-dedup-repos", "content"),
))
_register(DatasetSpec(
    name="wikipedia_long_articles",
    hf_repo="wikimedia/wikipedia",
    license="CC-BY-SA 3.0",
    stage="long_context",
    description="Top-decile-length Wikipedia articles.",
    loader=_hf_text_loader("wikimedia/wikipedia", "text"),
))

# SFT.
_register(DatasetSpec(
    name="tulu3_sft",
    hf_repo="allenai/tulu-3-sft-mixture",
    license="ODC-By 1.0",
    stage="sft",
    description="Curated SFT mix used to train Tulu 3 — strong general assistant data.",
    loader=_hf_chat_loader("allenai/tulu-3-sft-mixture", "messages"),
))
_register(DatasetSpec(
    name="openorca",
    hf_repo="Open-Orca/OpenOrca",
    license="MIT",
    stage="sft",
    description="GPT-4-explained FLAN tasks.",
    loader=_hf_chat_loader("Open-Orca/OpenOrca", "messages"),
))
_register(DatasetSpec(
    name="magicoder_evol",
    hf_repo="ise-uiuc/Magicoder-Evol-Instruct-110K",
    license="MIT",
    stage="sft",
    description="Code-instruction data used by Magicoder. Forge-heavy.",
    loader=_hf_chat_loader("ise-uiuc/Magicoder-Evol-Instruct-110K", "messages"),
))
_register(DatasetSpec(
    name="llava_next_instruct",
    hf_repo="lmms-lab/LLaVA-NeXT-Data",
    license="research-only",
    stage="sft",
    description="Multimodal SFT (image + dialogue). Aurora-heavy.",
    loader=_hf_chat_loader("lmms-lab/LLaVA-NeXT-Data", "messages"),
))
_register(DatasetSpec(
    name="openmathinstruct2",
    hf_repo="nvidia/OpenMathInstruct-2",
    license="CC-BY 4.0",
    stage="sft",
    description="Math-reasoning SFT (chain-of-thought).",
    loader=_hf_chat_loader("nvidia/OpenMathInstruct-2", "messages"),
))

# DPO.
_register(DatasetSpec(
    name="ultrafeedback",
    hf_repo="HuggingFaceH4/ultrafeedback_binarized",
    license="MIT",
    stage="dpo",
    description="Binarized UltraFeedback — strong general preference data.",
    loader=_hf_preference_loader("HuggingFaceH4/ultrafeedback_binarized"),
))
_register(DatasetSpec(
    name="hh_rlhf",
    hf_repo="Anthropic/hh-rlhf",
    license="MIT",
    stage="dpo",
    description="Helpfulness + harmlessness preference pairs from Anthropic.",
    loader=_hf_preference_loader("Anthropic/hh-rlhf"),
))
_register(DatasetSpec(
    name="code_preference_pairs",
    hf_repo="Vezora/Code-Preference-Pairs",
    license="MIT",
    stage="dpo",
    description="Pairs of better / worse code completions.",
    loader=_hf_preference_loader("Vezora/Code-Preference-Pairs"),
))

# Tool-use / agent.
_register(DatasetSpec(
    name="toolbench",
    hf_repo="ToolBench/ToolBench",
    license="MIT",
    stage="tool_use",
    description="Real-world API tool-call traces.",
    loader=_hf_chat_loader("ToolBench/ToolBench", "messages"),
))
_register(DatasetSpec(
    name="toolllm",
    hf_repo="ToolLLM/ToolLLM",
    license="MIT",
    stage="tool_use",
    description="Diverse multi-step tool dialogues.",
    loader=_hf_chat_loader("ToolLLM/ToolLLM", "messages"),
))
_register(DatasetSpec(
    name="opencodeinterpreter",
    hf_repo="m-a-p/OpenCodeInterpreter",
    license="Apache-2.0",
    stage="tool_use",
    description="Code + execution interpreter traces.",
    loader=_hf_chat_loader("m-a-p/OpenCodeInterpreter", "messages"),
))
_register(DatasetSpec(
    name="synthetic_agent_traces",
    hf_repo=None,
    license="self-generated",
    stage="tool_use",
    description="Locally-generated agent traces using the runtime's built-in tools.",
    # Loader supplied at runtime in scripts/download_datasets.py.
    loader=lambda *a, **k: iter(()),
))


# --- Audio / speech (Aurora omni-pretrain + omni-SFT) ---------------------

def _hf_audio_loader(repo: str, text_key: str = "text", audio_key: str = "audio"):
    """Yield ``{text, audio_path}`` dicts so the omni loader can splice both."""

    def _load(split: str = "train", streaming: bool = True, **kwargs):
        from datasets import load_dataset

        ds = load_dataset(repo, split=split, streaming=streaming, **kwargs)
        for row in ds:
            text = row.get(text_key) or ""
            audio = row.get(audio_key)
            yield {"text": text, "audio": audio}
    return _load


_register(DatasetSpec(
    name="librispeech",
    hf_repo="openslr/librispeech_asr",
    license="CC-BY 4.0",
    stage="omni_pretrain",
    description="Read English audiobooks with transcripts. Audio + text alignment.",
    loader=_hf_audio_loader("openslr/librispeech_asr", "text", "audio"),
))
_register(DatasetSpec(
    name="commonvoice",
    hf_repo="mozilla-foundation/common_voice_17_0",
    license="CC0",
    stage="omni_pretrain",
    description="Multilingual crowdsourced speech with transcripts.",
    loader=_hf_audio_loader("mozilla-foundation/common_voice_17_0", "sentence", "audio"),
))
_register(DatasetSpec(
    name="peoples_speech",
    hf_repo="MLCommons/peoples_speech",
    license="CC-BY-SA 4.0",
    stage="omni_pretrain",
    description="30k+ hours of diverse English speech.",
    loader=_hf_audio_loader("MLCommons/peoples_speech", "text", "audio"),
))
_register(DatasetSpec(
    name="libritts",
    hf_repo="mythicinfinity/libritts",
    license="CC-BY 4.0",
    stage="audio_codec_pretrain",
    description="Multi-speaker English read speech with clean recordings — used to train the codec autoencoder.",
    loader=_hf_audio_loader("mythicinfinity/libritts", "text", "audio"),
))
_register(DatasetSpec(
    name="librilight",
    hf_repo="LibriLight/librilight",
    license="open",
    stage="audio_codec_pretrain",
    description="60k hours of audio (audiobooks) — bulk codec pretraining.",
    loader=_hf_audio_loader("LibriLight/librilight", "text", "audio"),
))

# --- Omni SFT (voice instruction tuning) ----------------------------------

_register(DatasetSpec(
    name="vocalbench",
    hf_repo="WillHeld/vocalbench",
    license="MIT",
    stage="omni_sft",
    description="Voice-instruction QA — what users actually say to a speech assistant.",
    loader=_hf_chat_loader("WillHeld/vocalbench", "messages"),
))
_register(DatasetSpec(
    name="voicebench",
    hf_repo="hlt-mt/voicebench",
    license="CC-BY 4.0",
    stage="omni_sft",
    description="Voice instruction evaluation benchmark — also used as SFT fodder.",
    loader=_hf_chat_loader("hlt-mt/voicebench", "messages"),
))
_register(DatasetSpec(
    name="audiollm_instruct",
    hf_repo="THUDM/AudioLLM-Instruct",
    license="MIT",
    stage="omni_sft",
    description="Mixed audio-instruction dataset for omni-modal SFT.",
    loader=_hf_chat_loader("THUDM/AudioLLM-Instruct", "messages"),
))


# --- Code-screenshot pretrain (Forge vision tower warm-up) ----------------

_register(DatasetSpec(
    name="synthetic_code_screenshots",
    hf_repo=None,
    license="self-generated",
    stage="code_screenshot_pretrain",
    description="Locally rendered code screenshots paired with their source. "
                "See src/chatbot/data/code_render.py.",
    loader=lambda *a, **k: iter(()),
))


# --- SWE-bench-Verified train fold + augmentation -------------------------

_register(DatasetSpec(
    name="swe_bench_verified_train",
    hf_repo="princeton-nlp/SWE-bench_Verified",
    license="MIT",
    stage="rlef",
    description="Real GitHub bug-fix problems with grading test suites.",
    loader=_hf_chat_loader("princeton-nlp/SWE-bench_Verified", "messages"),
    default_split="train",
))
_register(DatasetSpec(
    name="taskbench_code",
    hf_repo="microsoft/TaskBench",
    license="MIT",
    stage="rlef",
    description="Multi-step coding tasks with executable graders.",
    loader=_hf_chat_loader("microsoft/TaskBench", "messages"),
))
_register(DatasetSpec(
    name="repofix",
    hf_repo="bigcode/repofix",
    license="permissive",
    stage="rlef",
    description="Repo-level fix-the-bug problems.",
    loader=_hf_chat_loader("bigcode/repofix", "messages"),
))
_register(DatasetSpec(
    name="synthetic_swebench_traces",
    hf_repo=None,
    license="self-generated",
    stage="rlef",
    description="Locally-generated agent traces against synthetic bug-fix problems. "
                "See src/chatbot/data/synthetic_swebench.py.",
    loader=lambda *a, **k: iter(()),
))


# --- DevOps SFT ----------------------------------------------------------

_register(DatasetSpec(
    name="logqa",
    hf_repo="microsoft/LogQA",
    license="MIT",
    stage="devops_sft",
    description="Question-answer pairs over real log files.",
    loader=_hf_chat_loader("microsoft/LogQA", "messages"),
))
_register(DatasetSpec(
    name="bugrepo_logs",
    hf_repo="bugrepo/bug-logs",
    license="permissive",
    stage="devops_sft",
    description="Bug reports paired with log payloads.",
    loader=_hf_chat_loader("bugrepo/bug-logs", "messages"),
))
_register(DatasetSpec(
    name="incidentbench",
    hf_repo="IncidentBench/IncidentBench",
    license="research-only",
    stage="devops_sft",
    description="Incident postmortems + remediation traces.",
    loader=_hf_chat_loader("IncidentBench/IncidentBench", "messages"),
))
_register(DatasetSpec(
    name="kube_postmortems",
    hf_repo="kubernetes-incident-postmortems",
    license="permissive",
    stage="devops_sft",
    description="K8s incident postmortems in a structured form.",
    loader=_hf_chat_loader("kubernetes-incident-postmortems", "messages"),
))
_register(DatasetSpec(
    name="synthetic_log_triage",
    hf_repo=None,
    license="self-generated",
    stage="devops_sft",
    description="Locally-generated log payloads + analyses. "
                "See src/chatbot/data/synthetic_log_triage.py.",
    loader=lambda *a, **k: iter(()),
))


def available_datasets(stage: Optional[str] = None) -> List[str]:
    """List the datasets we know about, optionally filtered by stage."""

    return [
        name for name, spec in DATASETS.items()
        if stage is None or spec.stage == stage
    ]


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Known: {sorted(DATASETS)}"
        )
    return DATASETS[name]
