<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>API Client Examples (<code>examples/</code>)</b></p>

# API Client Examples (`examples/`)

Ready-to-run `curl` and Python client examples targeting the OpenAI-compatible (`/v1/chat/completions`) HTTP endpoint exposed by `flux-server`.

---

## Contents

| File | Description |
|---|---|
| [`curl/streaming.sh`](curl/streaming.sh) | Bash script demonstrating Server-Sent Events (SSE) streaming chat completions with safe JSON payload encoding. |
| [`python/chat_completions.py`](python/chat_completions.py) | Python `openai` SDK script demonstrating blocking completions, SSE streaming (`temperature=0.0`), and function/tool calling. |

For Anthropic Messages API (`/v1/messages`) and container endpoint examples, see [`docs/CONTAINER_GUIDE.md`](../docs/CONTAINER_GUIDE.md).
