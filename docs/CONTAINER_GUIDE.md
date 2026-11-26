# Flux Engine — Production Container & GHCR Reference (`v1.0.3`)

> **Image Registry**: `ghcr.io/theapathy/flux-engine:v1.0.3` · `ghcr.io/theapathy/flux-engine:gb10` · `ghcr.io/theapathy/flux-engine:latest`  
> **Source Repository**: [https://github.com/TheApathy/flux-engine](https://github.com/TheApathy/flux-engine)  
> **License**: `AGPL-3.0-only` (Corresponding Source and license notices included; see [`SOURCE_AND_LICENSES.md`](../SOURCE_AND_LICENSES.md))  
> **Target Architectures**: `linux/arm64` NVIDIA GB10 Grace Blackwell (`sm_121a`, 128 GB unified memory) and `linux/amd64` NVIDIA Blackwell (`sm_120a` / RTX PRO 6000)

---

## 1. Container Overview & Building from Source

Flux Engine production images are built on `nvidia/cuda:13.0.0-runtime-ubuntu24.04` and package the compiled `flux-server` binary, the `/usr/local/bin/flux` model auto-detection launcher, and benchmark tooling.

- **AGPL-3.0 Source Provenance**:
  - Every release container image is built directly from the public [`TheApathy/flux-engine`](https://github.com/TheApathy/flux-engine) repository and ships `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `SOURCE_AND_LICENSES.md` under `/usr/share/licenses/flux-engine/`.
  - Container labels carry OCI source and version metadata (`org.opencontainers.image.source`, `org.opencontainers.image.version`) so operators can always trace a running container back to its exact Corresponding Source.
- **Dual-Mode CUDA Driver Binding (GPU Serving + CPU-Only Inspection)**:
  - When launched with `--gpus all`, the container binds to the host NVIDIA driver (`libcuda.so.1`).
  - When launched without `--gpus all` (for example on CI runners or login nodes), compatibility libraries allow `--help`, `--version`, `flux inspect`, `flux list-models`, `flux diagnose`, and `flux model-card` to exit cleanly with status `0`.

### Building Locally from Source

```bash
# NVIDIA GB10 / DGX Spark (linux/arm64, sm_121a):
docker build -f docker/gb10/Dockerfile -t flux-engine:gb10 .

# NVIDIA Blackwell RTX PRO 6000 / SM120 (linux/amd64, sm_120a):
docker build -f docker/sm120/Dockerfile -t flux-engine:sm120 .
```

See [`docker/`](../docker/) for model-specific Dockerfiles and [`docs/SM120_SM103_BRINGUP.md`](SM120_SM103_BRINGUP.md) for discrete SM120 bring-up notes.

---

## 2. Quickstart & Serving Modes

### 2.1 One-Command Server Launch (Auto-Detects Any Model in `/models`)

```bash
# Recommended on DGX Spark before loading 20–70 GB checkpoints:
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

docker run --rm -it --gpus all \
  --ipc=host --ulimit memlock=-1 \
  -p 8899:8899 \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:v1.0.3
```

### 2.2 Selecting a Specific Local Model & Execution Mode (`FLUX_MODE`)

```bash
docker run --rm -it --gpus all \
  --ipc=host --ulimit memlock=-1 \
  -p 8899:8899 \
  -e FLUX_MODEL=/models/Qwen3.8-Flash-Next-NVFP4 \
  -e FLUX_MODE=auto \
  -e FLUX_CONCURRENCY=4 \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:v1.0.3
```

| `FLUX_MODE` | Primary Use Case | Behavior & Key Controls |
| :--- | :--- | :--- |
| `auto` *(default)* | Balanced high-speed prefill + lossless decode | Selects the qualified hardware/model recipe (`recipes/` + `MODEL.toml` defaults, 135,168-token context on Qwen3.8-Flash-Next) |
| `max-prefill` | Peak long-prompt ingestion throughput | Enables FP8 prefill projections and QSA FP8 prefill path (`ATLAS_QWEN4_PREFILL_QSA_FP8=1`) |
| `decode` | Interactive / speculative decode throughput | Enables qualified K=5 native MTP or DFlash2 (`gamma=15`) speculation with CUDA graph capture |
| `long-context` / `136k` | 136.6K document synthesis & Estonia benchmark | Raises `--max-seq-len` to `139264` (`136K`) |
| `256k` *(Qwen3.8-Flash-Next)* | Requests exceeding ~135K tokens (up to 262,144) | Native 262K BF16-operand QSA profile (`ATLAS_QWEN4_PREFILL_QSA_FP8=0`, `ATLAS_QWEN4_QSA_VERIFY_ROWS=0`, `--max-seq-len 262144`) |
| `exact` | Reference correctness validation | Disables optional speculative approximations and uses conservative chunking (`flux-exact-reference.env`) |

Run `auto` (the default on Qwen3.8-Flash-Next, 135,168-token context) unless a workload explicitly requires contexts beyond ~135K tokens, in which case select `256k`.

> **GB10 unified memory guidance:** Keep `--gpu-memory-utilization` at `0.55` or lower on 128 GB GB10 systems so OS page cache and CUDA runtime allocations do not contend. See [`FLAGS.md`](../FLAGS.md) and [`docs/OPERATOR-GUIDE.md`](OPERATOR-GUIDE.md).

---

## 3. Universal Local Model Auto-Detection (`/usr/local/bin/flux`)

When you mount a model directory under `/models`, `flux` inspects `config.json`, `hf_quant_config.json`, `ple/` shards, `vision_config`, and sibling speculative drafter folders:

| Model Family | Supported Quantizations | Auto-Selected `--kernel-target` | Speculative / MTP Engine |
| :--- | :--- | :--- | :--- |
| **Qwen3.8-Flash-Next (`qwen4_exp`)** | `NVFP4`, `FP8`, `BF16` | `qwen3.8-flash-next` | Native MTP (`K=5`) or 5-layer DFlash2 (`gamma=15`) |
| **Qwen3.5 / Qwen3.6 35B-A3B MoE** | `NVFP4`, `FP8`, `BF16` | `qwen3.5-35b-a3b` | Native MTP (`K=2`) |
| **Qwen3.8-27B Dense Hybrid DeltaNet** | `NVFP4`, `FP8`, `BF16` | `qwen3.8-27b` | 5-layer DFlash2 (`gamma=15`) or 8-layer DFlash v1 |
| **Qwen3.6-27B Dense Hybrid DeltaNet** | `NVFP4`, `FP8`, `BF16` | `qwen3.6-27b` | 8-layer DFlash (`gamma=15`) |
| **DeepSeek-V3.2 / V4.1 MoE** | `NVFP4`, `FP8` | `deepseek-v4.1` | Native Multi-Token Prediction (`K=5`) |
| **GLM-5.3 Flash MoE** | `EXL3` (2.05 bpw) | `glm5.3-flash` | Target-only (speculation disabled at load) |
| **Gemma-4 26B-A4B / 31B Multimodal** | `NVFP4`, `FP8`, `BF16` | `gemma-4-26b-a4b` / `gemma-4-31b` | Target-only; auto-detects `ple-offload/` (`ATLAS_PLE_OFFLOAD_DIR`) & Vision tower |
| **Nemotron-3 Nano 30B / Super 120B** | `NVFP4`, `FP8` | `nemotron-3-nano-30b-a3b` | Native Hybrid Mamba-MoE |

### Inspecting Models Without a GPU

```bash
# List every model, quantization, and companion drafter in your local directory:
docker run --rm -v /path/to/your/models:/models:ro ghcr.io/theapathy/flux-engine:v1.0.3 list-models

# Inspect a specific model and preview the resolved binary, flags, and environment variables:
docker run --rm -v /path/to/your/models:/models:ro ghcr.io/theapathy/flux-engine:v1.0.3 inspect /models/Qwen3.8-Flash-Next-NVFP4
```

---

## 4. Connecting Endpoints, SDKs & Agentic Tools

Flux Engine exposes standard OpenAI and Anthropic HTTP endpoints on port `8899` (configurable via `--port` / `FLUX_PORT`; the default raw `flux-server` bind address is `127.0.0.1` for local safety — pass `--bind 0.0.0.0` explicitly with `--api-key` when exposing across a network):

- `POST /v1/chat/completions`: OpenAI Chat Completions (streaming SSE, tool calls, `reasoning_content`, `logprobs`)
- `POST /v1/completions`: OpenAI Text Completions
- `POST /v1/responses`: OpenAI Responses API
- `POST /v1/messages`: Anthropic Messages API (compatible with Anthropic SDKs)
- `GET /v1/models`: Model discovery endpoint
- `POST /tokenize`: Prompt tokenization & token counting
- `GET /health` & `GET /metrics`: Health readiness & live performance counters

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8899/v1", api_key="flux-local")
resp = client.chat.completions.create(
    model="Qwen3.8-Flash-Next-NVFP4",
    messages=[{"role": "user", "content": "Explain warp-specialized persistent kernels on Blackwell."}],
    max_tokens=512,
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

### Python (Anthropic SDK / `/v1/messages`)

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8899", api_key="flux-local")
msg = client.messages.create(
    model="Qwen3.8-Flash-Next-NVFP4",
    max_tokens=512,
    messages=[{"role": "user", "content": "Write a CUDA cooperative group reduction example."}],
)
print(msg.content[0].text)
```

### Built-In Endpoint Probe (`flux verify-endpoint`) & CLI Chat (`flux chat`)

```bash
# Verify /health, /v1/models, and /v1/chat/completions latency against a running server:
docker exec -it <container_name> flux verify-endpoint http://127.0.0.1:8899

# Send a direct prompt from the terminal:
docker exec -it <container_name> flux chat "Summarize the key differences between NVFP4 and FP8."
```

---

## 5. Reproducing Benchmarks (`lil-bench`, Estonia 136.6K & Concurrency Sweeps)

Container images can run [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) presets and generate SVG/PNG + Markdown summary reports under `/results/`:

### 5.1 All-Context Prefill & Decode Matrix (`0, 8k, 16k, 32k, 64k, 128k`)

```bash
# Against a running server:
docker exec -it <container_name> lil-bench matrix

# Custom context list or multiple repetitions (computes mean ± SD across reps):
docker exec -it <container_name> lil-bench matrix --contexts 0,8k,32k,128k --reps 3
```

**Outputs written to `/results/`**:
- `/results/repetition-1.json` (raw `llm-inference-bench` JSON)
- `/results/card.svg` & `/results/card.png` (visual benchmark summary cards)
- `/results/MODEL_CARD.md` & `/results/summary.md` (GitHub-flavored Markdown tables with Mean ± SD, Min/Max, TTFT, and decode throughput)

### 5.2 Multi-Stream Concurrency Sweeps (`C=1,2,4,8` or Custom Concurrencies)

```bash
# 1. Turnkey single command (starts server with FLUX_CONCURRENCY=8, runs C=1,2,4,8 sweep, writes cards):
docker run --rm --gpus all --ipc=host \
  -e FLUX_CONCURRENCY=8 \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:v1.0.3 \
  serve --with-concurrency 1,2,4,8 --with-card

# 2. Against a running server with custom concurrency levels and context depths:
docker exec -it <container_name> lil-bench concurrency 1,2,4,8,16 --contexts 0,8k,32k

# 3. Decode-only concurrency sweep (skips the initial prefill sweep):
docker exec -it <container_name> lil-bench concurrency 1,2,4,8 --contexts 0,8k,32k --decode-only
```

### 5.3 Estonia 136.6K Long-Context Needle Benchmark (`--test-profile estonia`)

Tests 136,645-token prompt processing and multi-needle historical synthesis across the 105,650-word Estonia corpus:

```bash
# 1. Turnkey single command (auto-configures FLUX_MODE=long-context and max_seq_len=139264):
docker run --rm --gpus all --ipc=host \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:v1.0.3 \
  serve --with-estonia --with-card

# 2. Against an already-running server (started with FLUX_MODE=long-context):
docker exec -it <container_name> lil-bench estonia
```

### 5.4 Coding Throughput & Accuracy (`GSM8K` / `MMLU-Pro` / `GPQA-Diamond`)

```bash
# Coding throughput (5 runs x 2000 tokens at temperature=0):
docker exec -it <container_name> lil-bench coding

# Standardized accuracy benchmarks:
docker exec -it <container_name> lil-bench accuracy gsm8k
docker exec -it <container_name> lil-bench accuracy mmlu-pro
docker exec -it <container_name> lil-bench accuracy gpqa-diamond
```

---

## 6. Generating Model Cards, Logs & Diagnostics (`flux diagnose` & `flux model-card`)

| Artifact in `/results/` | Command to Generate / Refresh | Purpose |
| :--- | :--- | :--- |
| `flux-diagnostics.json` | `flux diagnose` *(or auto on startup)* | Snapshot of GPU memory, `libcuda.so.1` binding, `/dev/shm` size, UMA page cache, detected models, broken symlinks, and resolved launch plan |
| `flux-server.log` | Automatic during `flux serve` | Full server stdout/stderr log including kernel initialization, CUDA graph capture, and prefill/decode timings |
| `MODEL_CARD.md` | `flux model-card` or `lil-bench card` | Shareable Markdown Model Card combining hardware/runtime config, model profile, and benchmark matrices |
| `card.svg` / `card.png` | `lil-bench matrix` or `lil-bench card` | Visual benchmark summary cards |

---

## 7. Privacy & Telemetry Opt-Out Controls (`FLUX_OPT_OUT`)

- **Zero External Telemetry by Design**: Flux Engine and `lil-bench` execute locally with no analytics, crash reporting, or outbound network connections.
- **Granular Redaction Controls**: Before sharing `MODEL_CARD.md`, `card.png`, or `flux-diagnostics.json` publicly, set `FLUX_OPT_OUT` or pass CLI flags to redact host metadata and prompt contents:

```bash
# Redact hostname, kernel version, GPU driver version, and hardware IDs from all cards & JSONs:
docker run --rm --gpus all \
  -e FLUX_OPT_OUT=host,logs \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:v1.0.3 \
  serve --with-lil matrix --with-card
```

| Environment / Flag | Redaction Behavior |
| :--- | :--- |
| `FLUX_OPT_OUT=telemetry` / `--opt-out-telemetry` | *(Default)* Offline local execution, no outbound requests |
| `FLUX_OPT_OUT=host` / `--opt-out-host-info` | Replaces hostname, kernel release, and driver version with `REDACTED` in `flux-diagnostics.json`, `repetition-*.json`, `card.svg`, `card.png`, and `MODEL_CARD.md` |
| `FLUX_OPT_OUT=logs` / `--opt-out-prompt-logs` | Disables `/results/flux-server.log` file creation and redacts raw prompt/completion text from JSON outputs |
| `FLUX_OPT_OUT=all` / `--opt-out-all` | Enables all privacy redactions (`telemetry + host + logs`) |
