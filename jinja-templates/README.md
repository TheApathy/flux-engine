<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Chat &amp; Tool-Call Jinja Templates (<code>jinja-templates/</code>)</b></p>

# Jinja Template Overrides

Place `.jinja` files here named by `model_type` to override the chat template
from a checkpoint's `tokenizer_config.json`. This allows applying formatting or
tool-calling fixes without modifying local model weight directories.

## Checked-In Templates & Upstream Attribution

The filename matches the model's `model_type` from `config.json` (with optional
API-specific overrides under subdirectories such as `openai/`):

| Model Family | `model_type` | Override File | Upstream Origin | Upstream License / Model Terms |
|---|---|---|---|---|
| **Google Gemma 4** | `gemma4` | `gemma4.jinja` | [`google`](https://huggingface.co/google) model cards | Google Gemma Terms of Use |
| **Zhipu AI GLM-5 / GLM-4.7** | `glm5_next` | `glm5_next.jinja` | [`zai-org`](https://huggingface.co/zai-org) model cards | `MIT` |
| **Mistral** | `mistral` | `mistral.jinja` | [`mistralai`](https://huggingface.co/mistralai) model cards | `Apache-2.0` |
| **NVIDIA Nemotron-H** | `nemotron_h` | `nemotron_h.jinja` | [`nvidia`](https://huggingface.co/nvidia) model cards | NVIDIA Open Model License |
| **Qwen 3.5 Dense** | `qwen3_5` | `qwen3_5.jinja` | [`Qwen`](https://huggingface.co/Qwen) model cards | `Apache-2.0` |
| **Qwen 3.5 / 3.8 MoE** | `qwen3_5_moe` | `qwen3_5_moe.jinja` | [`Qwen`](https://huggingface.co/Qwen) model cards | `Apache-2.0` |
| **Qwen 3.5 / 3.8 MoE (OpenAI API)** | `qwen3_5_moe` | `openai/qwen3_5_moe.jinja` | [`Qwen`](https://huggingface.co/Qwen) model cards (OpenAI toggle variant) | `Apache-2.0` |

> **Provenance, Compile-Time Embedding & Model Terms Notice:** Each `.jinja`
> file above implements the chat-formatting specification published with its
> corresponding upstream model family (see
> [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)). When serving
> checkpoints governed by model-specific terms (such as Google Gemma or NVIDIA
> Nemotron models), the respective upstream model license and terms of use
> apply.
>
> All 7 checked-in `.jinja` files above are compiled into `spark-server` via
> `include_str!` (`crates/spark-server/src/tokenizer/jinja_helpers.rs`), so
> deleting a file from disk at runtime falls back to the compile-time embedded
> copy rather than `tokenizer_config.json` (and deleting one before building
> breaks `cargo build`). To supply a custom template at runtime, place a
> `<model_type>.jinja` file in `$ATLAS_TEMPLATE_DIR` (or `FLUX_TEMPLATE_DIR`).
>
> Other supported model families (such as DeepSeek-V4.1 and MiniMax) use either
> built-in Rust prompt/tool-call handlers or the template shipped in the
> checkpoint's `tokenizer_config.json`.

## Resolution Priority

Override file lookup (`read_override` in `crates/spark-server/src/tokenizer/jinja_helpers.rs`)
checks four locations in order:

1. `$ATLAS_TEMPLATE_DIR/<rel>` (highest priority on disk)
2. `<repo_root>/jinja-templates/<rel>`
3. `./jinja-templates/<rel>`
4. Compile-time embedded copy (`<embedded>/jinja-templates/<rel>`) for the 7 bundled templates above

Template selection (`select_chat_template`) then resolves in this order:

1. **Dense Qwen3.8 checkpoint-contract exception:** For `model_type == "qwen3_5"`, if the checkpoint's `tokenizer_config.json` template declares both `resolved_reasoning_effort` and `preserve_thinking`, the checkpoint's own template takes priority over `qwen3_5.jinja`.
2. **Resolved override template** from the 4-tier lookup above (when one exists for `model_type`).
3. **Checkpoint template** from the model directory's `tokenizer_config.json`.
4. **Default ChatML fallback** if no template is found.

## Usage

```bash
# Example: apply a custom runtime override without modifying checked-in files
mkdir -p /path/to/my-templates
cp /path/to/custom_chat_template.jinja /path/to/my-templates/qwen3_5_moe.jinja
ATLAS_TEMPLATE_DIR=/path/to/my-templates flux-server serve ...
```

At startup, the server logs which override source was loaded:

```text
Using override Jinja template from /path/to/my-templates/qwen3_5_moe.jinja (7800 chars)
```
