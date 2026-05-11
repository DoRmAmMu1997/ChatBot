# Datasets

Every dataset the training pipeline knows about is listed in
`src/chatbot/data/registry.py`. The training scripts only use the ones
referenced from their YAML mix; nothing is downloaded silently.

## Aurora-50B mix

| Dataset                | Stage          | License        | Notes |
|------------------------|----------------|----------------|-------|
| `fineweb_edu`          | pretrain       | ODC-By 1.0     | Filtered educational-quality CommonCrawl. The bulk of Aurora's pretraining diet. |
| `the_stack_v2`         | pretrain       | per-file       | Light code slice — Aurora needs some code competence but is text-first. |
| `wikipedia`            | pretrain       | CC-BY-SA 3.0   | Encyclopedic prose. |
| `books`                | pretrain       | public domain  | Project Gutenberg English-language books. |
| `arxiv`                | pretrain       | various        | Scientific writing. |
| `laion_recaptioned`    | image_text     | CC-BY 4.0      | Image-text pairs with synthetic re-captions. |
| `long_books`           | long_context   | various        | Long-form books for the context-extension stage. |
| `arxiv_long`           | long_context   | MIT            | Long arXiv articles. |
| `wikipedia_long_articles` | long_context | CC-BY-SA 3.0  | Top-decile Wikipedia. |
| `tulu3_sft`            | sft            | ODC-By 1.0     | Strong general SFT mixture (used by Tulu 3). |
| `openorca`             | sft            | MIT            | FLAN-explanations data. |
| `magicoder_evol`       | sft            | MIT            | Light code instructions (Aurora is mostly text but still touches code). |
| `llava_next_instruct`  | sft            | research-only  | Multimodal SFT (images + dialogue). |
| `openmathinstruct2`    | sft            | CC-BY 4.0      | Math chain-of-thought. |
| `ultrafeedback`        | dpo            | MIT            | Binarized preferences. |
| `hh_rlhf`              | dpo            | MIT            | Anthropic's helpfulness/harmlessness preferences. |

## Forge-250B mix

| Dataset                | Stage          | License        | Notes |
|------------------------|----------------|----------------|-------|
| `the_stack_v2`         | pretrain       | per-file       | Full Forge pretrain workhorse. |
| `github_repo_packed`   | long_context   | per-file       | Repo-level packed sequences. The 1M-context magic ingredient. |
| `openmathinstruct2`    | sft / pretrain | CC-BY 4.0      | Reasoning data. |
| `magicoder_evol`       | sft            | MIT            | Code-instruction pairs. |
| `opencodeinterpreter`  | tool_use       | Apache-2.0     | Code + execution traces. |
| `toolbench` / `toolllm`| tool_use       | MIT            | API tool-call traces. |
| `synthetic_agent_traces` | tool_use     | self-generated | Locally produced traces using the runtime's built-in tools. |
| `code_preference_pairs`| dpo            | MIT            | Code-quality preference pairs. |

## Pre-downloading

```powershell
python scripts/download_datasets.py --stage pretrain --limit 1000
```

Pre-downloading is optional — the training scripts stream lazily — but
handy when you want to bake a dataset mirror into an offline environment.

## License compliance

Every entry above lists its upstream license. If you redistribute a
trained checkpoint, make sure the *training data licenses* allow it. Some
research-only datasets (`llava_next_instruct`, parts of LAION) require
extra care; the registry surfaces the license string so it's hard to
forget. We do not redistribute any dataset content; users fetch directly
from Hugging Face.
