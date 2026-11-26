# Flux Engine — Production Container (`ghcr.io/theapathy/flux-engine:v1.0.2` / `:gb10` / `:latest`)

**Native Rust/CUDA (`sm_121a` / `sm_120a`) Inference Engine for NVIDIA Blackwell & DGX Spark (GB10)**  
*Zero-Config Local Model Auto-Detection · OpenAI & Anthropic Compatible Server · Built-in `lil-bench` (Matrix, Concurrency Sweeps & Estonia 136.6K) · Automated SVG/PNG & Markdown Model Cards · Full Diagnostics & Privacy Opt-Out*

---

## 1. Container Image Tags & Installation

```bash
# Primary v1.0.2 GB10 / DGX Spark production tags
docker pull ghcr.io/theapathy/flux-engine:v1.0.2
docker pull ghcr.io/theapathy/flux-engine:gb10
docker pull ghcr.io/theapathy/flux-engine:latest
```

| Image Reference | Architecture | CUDA / OS Runtime | Included Capabilities |
| :--- | :--- | :--- | :--- |
| `ghcr.io/theapathy/flux-engine:v1.0.2` (`:1.0.2`, `:gb10`, `:latest`) | `linux/arm64` (`sm_121a`) | CUDA 13.0 / Ubuntu 24.04 | GB10 production release (`repro-kit/Dockerfile` packages `flux-server-real` aliased as `flux-server`, plus `flux` CLI, `flux-bench`, `flux-lens`, `lil-bench`) |
| `flux-engine:latest` | `linux/arm64` (`sm_121a`) | CUDA 13.0 / Ubuntu 24.04 | Local convenience alias |

See the [qualification runbook](QWEN4_R9_QUALIFICATION.md) and
[exact-image release wrapper](R15_EXACT_IMAGE_RELEASE.md) for receipt verification
and container composition details.

### Flux-Native Trials

[Flux Lens](FLUX_LENS.md) is an original, standard-library-only trial runner
for exact-token context/concurrency matrices and paired output comparisons.
Use `python3 repro-kit/flux_lens.py --help` from a source checkout, or
`flux lens --help` / `flux-lens --help` inside the container image.

Plans are prepared offline. Only an explicit `run` contacts the selected
completion endpoint. Raw requests and responses remain in private local
artifacts; `report` recomputes a reusable report without prompt text, local
paths, endpoint URLs, or declared identity strings. Its initial throughput
metric includes the whole request, not decode-only time. Client concurrency
does not attest native engine batching, and exact corpus matches do not
replace model-accuracy or exact-image release qualification.

---

## 2. 60-Second Quickstart

### A. Start the OpenAI / Anthropic Compatible Server (Auto-Detects Any Model in `/models`)
Mount your local models directory to `/models:ro` and an output folder to `/results`. The container automatically inspects `config.json`, `hf_quant_config.json`, PLE shards, and companion speculative drafters:

```bash
# Optional on DGX Spark UMA systems before loading large checkpoints:
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

docker run --rm -it --gpus all \
  --ipc=host --ulimit memlock=-1 \
  -p 8899:8899 \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:gb10
```

*(Note: `--security-opt seccomp=unconfined` is **optional** and only needed if enabling async `io_uring` NVMe PLE offload; under Docker's default seccomp profile, the engine automatically falls back to parallel `pread64`.)*

### B. Select a Specific Model, Execution Mode (`FLUX_MODE`), or Custom Recipe Knobs (`--set`)
```bash
docker run --rm -it --gpus all \
  --ipc=host --ulimit memlock=-1 \
  -p 8899:8899 \
  -e FLUX_MODEL=/models/Qwen3.8-Flash-Next-NVFP4-Offload \
  -e FLUX_MODE=max-prefill \
  -e FLUX_CONCURRENCY=4 \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:gb10 \
  serve --set FLUX_QWEN4_PREFILL_WIDE=4096
```

### C. Repacking Qwen3.8-Flash-Next 51B PLE Weights to NVFP4 (`repack-ple` / `export-repacker`)
The container ships a self-contained NVFP4 PLE repacker (`/opt/flux/scripts/repack_qwen38_flash_next_ple_nvfp4.py`, aliased to `/usr/local/bin/repack-ple` and `flux repack-ple`) that converts `Apathy-Qwen3.8-Flash-Next-NVFP4` (167 GB raw BF16 PLE table) into either:
- **`--mode offload` (Recommended for 1x DGX Spark 128 GB UMA)**: Keeps backbone + router + MTP in NVFP4 (`~22 GB`) and writes 48 `io_uring`-aligned NVFP4 PLE shards (`ple_layer_XX.bin`, `~28.8 GB` total) into `<dst>/ple-offload/`.
- **`--mode resident` (For >=64 GB free GPU RAM / multi-Spark)**: Quantizes `ple_embed_tokens.weight` to ModelOpt NVFP4 tensors inside `.safetensors` (`~51 GB` total).

```bash
# 1. Run the PLE repacker directly inside the container (zero host dependencies required):
docker run --rm \
  -v /path/to/your/models:/models:rw \
  ghcr.io/theapathy/flux-engine:gb10 \
  repack-ple \
    /models/Apathy-Qwen3.8-Flash-Next-NVFP4 \
    /models/Qwen3.8-Flash-Next-NVFP4-Offload \
    --mode offload

# 2. Or export the standalone Python repacker script to your host:
docker run --rm ghcr.io/theapathy/flux-engine:gb10 export-repacker > repack_qwen38_flash_next_ple_nvfp4.py
chmod +x repack_qwen38_flash_next_ple_nvfp4.py
python3 repack_qwen38_flash_next_ple_nvfp4.py --self-test
```

---

## 3. Querying the Engine with Your Own Endpoints & SDKs

Once the container is running, query it from your host, web UI (OpenWebUI, Cursor, Continue, Claude Code), or custom application on port `8899`:

| Endpoint | Method | Protocol Compatibility | Description |
| :--- | :--- | :--- | :--- |
| `/v1/chat/completions` | `POST` | OpenAI Chat Completions | Streaming SSE (`stream: true`), tool/function calling, `reasoning_content`, `logprobs`, structured outputs |
| `/v1/completions` | `POST` | OpenAI Completions | Raw prompt completion & exact token-targeted benchmarking |
| `/v1/responses` | `POST` | OpenAI Responses API | Modern OpenAI Responses schema |
| `/v1/messages` | `POST` | Anthropic Messages API | Native Anthropic SDK / Claude Code compatible endpoint |
| `/v1/models` | `GET` | OpenAI Models API | Returns loaded model ID, context window, and active configuration |
| `/tokenize` | `POST` | Tokenizer Utility | Returns exact token IDs and token count for any prompt string |
| `/health` | `GET` | Readiness Probe | Returns HTTP `200 OK` when weights and CUDA graphs are ready |
| `/metrics` | `GET` | Telemetry Snapshot | Local real-time prefill/decode tok/s, KV utilization, and speculative acceptance rate |

### Example: cURL Chat Completion
```bash
curl http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "Write a lock-free ring buffer in C++20."}],
    "max_tokens": 512,
    "temperature": 0.2
  }'
```

> **Qwen3.8-Flash-Next defaults (since `9.15.3.1`):** with no sampling fields, a request decodes greedily (`temperature 0`, `presence_penalty 0`) with thinking off. Send `"chat_template_kwargs": {"enable_thinking": true}` or an explicit `"temperature"` to opt back in; both lower MTP acceptance and decode tok/s.

### Example: Built-In Endpoint Verifier (`flux verify-endpoint`)
Validate any running container or remote endpoint (`/health`, `/v1/models`, `/v1/chat/completions`) and measure round-trip latency:
```bash
docker exec -it <container_name> flux verify-endpoint http://127.0.0.1:8899
```

---

## 4. Running `lil-bench`, Estonia 136.6K, Concurrency Sweeps & Model Cards

The container bundles **`local-inference-lab/llm-inference-bench` (`v0.7.9`)** at `/opt/flux/llm-inference-bench` and `/usr/local/bin/lil-bench`, plus an SVG/PNG + Markdown **Model Card Generator** (`rsvg-convert` pre-installed).

### A. Standard All-Context Matrix (`0, 8k, 16k, 32k, 64k, 128k`) + Auto-Generated Cards
Replicates our exact single-stream evaluation (`--token-targeting exact --request-seed 20261011 --duration 30 --max-tokens 2048`) and automatically writes `/results/repetition-1.json`, `/results/card.svg`, `/results/card.png`, and `/results/MODEL_CARD.md`:
```bash
# Against an already-running container:
docker exec -it <container_name> lil-bench matrix

# Or one-command turnkey (starts server -> runs matrix -> generates cards -> exits):
docker run --rm --gpus all --ipc=host \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:gb10 \
  serve --with-lil matrix --with-card
```

### B. Multi-Stream Concurrency Sweeps (`C=1,2,4,8` or Custom Concurrencies)
Test how the engine scales across parallel client streams (`--concurrency 1,2,4,8`, `--show-capacity-limited-values`) and generate a dedicated Concurrency Heatmap Card (`concurrency-card.svg` / `concurrency-card.png`):
```bash
# 1. Start the server with multi-stream concurrency enabled (e.g. FLUX_CONCURRENCY=8):
docker run --rm --gpus all --ipc=host \
  -e FLUX_CONCURRENCY=8 \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:gb10 \
  serve --with-concurrency 1,2,4,8 --with-card

# 2. Or against a running server, test any custom concurrency list & context depths:
docker exec -it <container_name> lil-bench concurrency 1,2,4,8,16 --contexts 0,8k,32k
```

### C. Estonia 136.6K Long-Context Needle-in-a-Haystack Benchmark (`--test-profile estonia`)
Runs the 105,650-word (~136,645-token) Estonia history synthesis & embedded needle retrieval benchmark:
```bash
# Against a running server (ensure server was started with FLUX_MODE=long-context or FLUX_MAX_SEQ_LEN=143360):
docker exec -it <container_name> lil-bench estonia

# Or one-command turnkey (automatically sets FLUX_MODE=long-context & min_seq_len=143360):
docker run --rm --gpus all --ipc=host \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:gb10 \
  serve --with-estonia --with-card
```

### D. Coding Peak, Prefill-Only, Accuracy Suite, or Full Unified Suite
```bash
# Coding peak throughput (5 runs x 2000 tokens at ctx=0):
docker exec -it <container_name> lil-bench coding

# Dedicated prefill-only sweep (8k, 16k, 32k, 64k, 128k):
docker exec -it <container_name> lil-bench prefill

# Accuracy evaluation (gsm8k, mmlu-pro, gpqa-diamond, or all):
docker exec -it <container_name> lil-bench accuracy gsm8k

# Full unified suite (matrix + coding + concurrency 1,2,4,8 + estonia + cards):
docker exec -it <container_name> lil-bench suite

# Re-render dark-mode SVG/PNG + Markdown Model Cards from any JSON files in /results:
docker exec -it <container_name> lil-bench card /results/card /results/repetition-1.json
```

An accuracy command completing successfully is not by itself a release pass.
R15 requires full, matched target-only and native-MTP K=5 runs from separate
fresh server processes, followed by receipt validation as described in the
[R15 qualification runbook](QWEN4_R9_QUALIFICATION.md#gsm8k-and-mmlu-pro).

---

## 5. Universal Local Model Auto-Detection Matrix

Run `flux list-models` or `flux inspect` to verify how any local directory in `/models` is mapped. Note that the release-packaging `repro-kit/Dockerfile` copies only `flux-server-real` (aliased to `flux-server`); source-built multi-model deployments can also stage `flux-server-universal` or `flux-server-glm53` alongside it when serving non-primary targets:

| Model Family | Detected Quantizations | Selected `--kernel-target` | Launcher Binary Preference | Speculative / MTP Acceleration |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3.8-Flash-Next Hybrid DeltaNet-MoE (QSA/PLE)** | `NVFP4`, `FP8`, `BF16` | `qwen3.8-flash-next` | `flux-server-real` | Native Multi-Token Prediction (`K=5`, `gamma=4`) or DFlash2 + NVMe `ple-offload/` |
| **Qwen3.8 / Qwen3.6 / Qwen3.5 35B-A3B MoE** | `NVFP4`, `FP8`, `BF16` | `qwen3.6-35b-a3b` *(or `qwen3.5-35b-a3b` for 3.5)* | `flux-server-universal` | Auto-pairs DFlash (`gamma=block_size-1`) or Native MTP (`K=5`) |
| **Qwen3.8-27B Dense Hybrid DeltaNet** | `NVFP4`, `FP8`, `BF16` | `qwen3.8-27b` | `flux-server-universal` | Auto-pairs `Apathy-...-DFlash-drafter-v2/v3` (`gamma=15`), `z-lab/Qwen3.8-27B-DFlash` (`gamma=15`), or `incoai/Qwen3.8-27B-DFlash2` (`gamma=7`) |
| **Qwen3.6-27B Dense Hybrid DeltaNet** | `NVFP4`, `FP8`, `BF16` | `qwen3.6-27b` | `flux-server-qwen36-27b` / `flux-server-universal` | Auto-pairs Qwen3.6-27B DFlash / DFlash2 (`gamma = block_size - 1`) |
| **DeepSeek-V4.1 Flash MoE** | `NVFP4`, `FP8` | `deepseek-v4.1` | `flux-server-universal` | None: target-only (`speculation: "none"`) |
| **GLM-5.3 Flash MoE** | `EXL3` (2.05 bpw) | `glm5.3-flash` | `flux-server-glm53` | None: target-only (the engine refuses DFlash2 / MTP speculation for GLM at load) |
| **Gemma-4 26B-A4B / 31B Multimodal** | `NVFP4`, `FP8`, `BF16` | `gemma-4-26b-a4b` / `gemma-4-31b` | `flux-server-universal` | None: target-only; auto-detects `ple/` shards & Vision tower |
| **Nemotron-3 Nano 30B / Super 120B** | `NVFP4`, `FP8` | `nemotron-3-nano-30b-a3b` / `nemotron-super-120b-a12b` | `flux-server-universal` | Native Hybrid Mamba-MoE |

---

## 6. Execution Mode Recipes (`FLUX_MODE`)

| `FLUX_MODE` | Prefill / Attention Profile | Target Workload | Key Tuning Parameters |
| :--- | :--- | :--- | :--- |
| `auto` *(default)* | Adaptive (`W=16384` / `W=4096`) | General serving & standard benchmarks | `recipes/flux-nextgen-max-prefill-decode.env` |
| `nextgen` | Unified SM121a (`W=16384`) | Peak v1.0+ single-Spark prefill + 0-ULP speculative decode | `recipes/flux-nextgen-max-prefill-decode.env`, `min_seq_len=262144` |
| `agent` *(or `claude-code`, `ohmypi`, `codex`, `antigravity`)* | Interactive (`W=4096`) | Multi-turn agentic coding harnesses & tool use | `--enable-prefix-caching true --ssm-cache-slots 16`, `--tool-max-tokens` defaults to `max(65536, max_seq_len)` (`135168+`; override via `FLUX_TOOL_MAX_TOKENS`) |
| `fast` / `v8` | Wide Prefill (`W=16384`) | Fast long-context prefill | `recipes/flux-fast-prefill.env`, `FLUX_QWEN4_PREFILL_WIDE=16384`, speculation disabled |
| `max-prefill` / `v9` | Wide Prefill (`W=16384`) | Maximum 128K prefill throughput | `recipes/flux-v9-max-prefill.env`, `FLUX_QWEN4_PREFILL_WIDE=16384`, `FLUX_QWEN4_PREFILL_MOE_V28Q=2` |
| `interactive` / `w4096` | Interactive (`W=4096`) | Interactive coding & deep needle retrieval | `FLUX_QWEN4_PREFILL_WIDE=4096`, `recipes/flux-v9-w4096-interactive.env` |
| `decode` | Exact Full | Lossless full-attention prefill + max decode | `recipes/flux-k5-lossless-decode.env`, decode-optimized recipe |
| `long-context` / `136k` | Long Context (`W=4096`) | Estonia 136.6K benchmark & >128K documents | `recipes/flux-estonia-136k.env`, raises `--max-seq-len` to at least `143360` |
| `256k` | Native 262K (`W=4096`) | Native context capacity (`262,144` tokens) | Qwen4 QSA: `--max-seq-len 262144 --kv-cache-dtype bf16` by default; compressed KV is opt-in |
| `1m-nvfp4` | Extended 1M NVFP4 KV (`W=4096`) | Single-Spark 1M (`1,048,576`) context on Qwen3.8-Flash-Next | `recipes/flux-nextgen-1m-nvfp4.env`, `--max-seq-len 1048576 --rope-yarn-factor 4`, NVFP4 KV |
| `c128` | High Concurrency (`W=2048`) | Multi-stream serving up to `C=128` | `recipes/flux-nextgen-c128-concurrency.env`, `min_seq_len=32768`, SLO interleave |
| `high-kv` | Mixed NVFP4/BF16 KV | Long-context high concurrency (`4/12` BF16 + `8/12` NVFP4 KV) | `recipes/flux-nextgen-max-prefill-decode.env`, `min_seq_len=262144` |
| `2x-spark` | Dual-Spark TP=2+EP=2 / Disagg | 2x DGX Spark ConnectX-7 RoCEv2 cluster | `recipes/2x-spark-tp2-disagg.env`, `min_seq_len=262144` |
| `2x-spark-1m` | Dual-Spark 1M BF16 KV | 2x DGX Spark 1M (`1,048,576`) lossless BF16 context | `recipes/2x-spark-tp2-disagg.env`, `--max-seq-len 1048576 --rope-yarn-factor 4` |
| `3x-spark` | Triple-Spark PP=3 / Disagg | 3x DGX Spark pipeline ring or 1P+2D disaggregation | `recipes/3x-spark-pp3-disagg.env`, `min_seq_len=262144` |
| `4x-spark` | Quad-Spark TP=4+EP=4 / PP=4 | 4x DGX Spark cluster (`FLUX_PP_SIZE=4` for PP=4 ring) | `recipes/4x-spark-cluster.env`, `min_seq_len=262144` |
| `512k` | Extended 512K (`W=4096`) | Extended 512K RoPE/YaRN context for non-QSA models *(refused on `Qwen3.8-Flash-Next`)* | `--max-seq-len 524288` with automatic YaRN scaling |
| `1m` / `yarn` | Extended 1M (`W=4096`) | Extended 1M RoPE/YaRN context for non-QSA models *(refused on `Qwen3.8-Flash-Next`; use `1m-nvfp4` or `2x-spark-1m`)* | `--max-seq-len 1048576 --rope-yarn-factor 4` |
| `exact` | Exact Full | Strict reference validation | `recipes/flux-exact-reference.env`, full `O(N^2)` attention, speculation disabled |

For Qwen3.8-Flash-Next, `auto` and `nextgen` load `recipes/flux-nextgen-max-prefill-decode.env`, while `max-prefill` and `256k` load `recipes/flux-v9-max-prefill.env` (`FLUX_MODE=512k` and `FLUX_MODE=1m` are rejected on Qwen3.8-Flash-Next; use `FLUX_MODE=1m-nvfp4` on 1x Spark or `FLUX_MODE=2x-spark-1m` on 2x Spark). When `FLUX_QWEN4_MTP_CONF` / `FLUX_QWEN4_MTP_CONF_EARLY` are enabled, the launcher keeps them enabled for `FLUX_CONCURRENCY > 1` because they are gated per-cohort at runtime in Rust (solo `c=1` cohorts keep full parity while `live >= 2` cohorts stay protected). To disable them explicitly, pass `-e FLUX_QWEN4_MTP_CONF=0 -e FLUX_QWEN4_MTP_CONF_EARLY=0`.

---

## 7. Using Flux Engine with Agentic Coding Harnesses (`Claude Code`, `ohmypi`, `Codex`, `Antigravity`, `OpenCode`)

Run `flux agents` inside the container at any time to print copy-pasteable harness configurations tailored to your detected model:
```bash
docker run --rm -v /path/to/your/models:/models:ro ghcr.io/theapathy/flux-engine:gb10 agents
```

### Recommended Server Launch for Agentic Harnesses (`FLUX_MODE=agent`)
Launching with `-e FLUX_MODE=agent` enables **multi-turn prefix caching (`--enable-prefix-caching true`)**, **16 recurrent SSM/DeltaNet state cache slots (`--ssm-cache-slots 16`)**, a **wide `W=4096` attention window**, and **`--tool-max-tokens` defaulting to `max(65536, max_seq_len)`** (`135168+` tokens; override via `FLUX_TOOL_MAX_TOKENS`) so large multi-file edits never truncate:
```bash
docker run --rm -it --gpus all --ipc=host --shm-size 16g -p 8899:8899 \
  -v /path/to/your/models:/models:ro \
  -e FLUX_MODE=agent \
  ghcr.io/theapathy/flux-engine:gb10
```

| Harness | Protocol / Endpoint | Quick Configuration |
| :--- | :--- | :--- |
| **Claude Code (`claude`)** | Anthropic `/v1/messages` | `export ANTHROPIC_BASE_URL=http://localhost:8899 ANTHROPIC_API_KEY=flux-local ANTHROPIC_MODEL=Qwen3.8-35B-A3B-NVFP4 && claude` |
| **ohmypi (`pi` / `omp`)** | OpenAI `/v1/chat/completions` | Add provider `"baseUrl": "http://localhost:8899/v1"`, `"api": "openai-completions"`, `"maxTokens": 16384` in `~/.pi/agent/models.json` |
| **OpenAI Codex CLI (`codex`)** | OpenAI `/v1/chat/completions` or `/v1/responses` | Set `base_url = "http://localhost:8899/v1"`, `wire_api = "chat"` in `~/.codex/config.toml` |
| **Antigravity (`agy`) / OpenCode / Cline** | OpenAI `/v1` or Anthropic `/` | Point Base URL to `http://localhost:8899/v1` (OpenAI) or `http://localhost:8899` (Anthropic) |

---

## 8. Hardware Topology Auto-Detection & Extended Context Scaling (`256K`, `512K`, `1M` via RoPE / YaRN)

Flux Engine automatically inspects both your **model checkpoint (`max_position_embeddings`, `rope_theta`, `rope_scaling`)** and your **host GPU topology (`detect_hardware_topology()` — VRAM/UMA capacity, SM architecture `sm_121a`/`sm_120`/`sm_100`/`sm_90a`, and multi-GPU count)**:
- **Native `256K` Capacity (`FLUX_MODE=256k` or `--max-seq-len 262144`)**: At the checkpoint's native limit, Qwen4 QSA defaults to BF16 KV and does not enable YaRN. Set `FLUX_KV_CACHE_DTYPE=nvfp4` (or pass `--kv-cache-dtype nvfp4`) only when you explicitly want the memory/quality tradeoff. This mode configures capacity; it is not a 262K retrieval qualification. The currently published long-context retrieval proof ends at 136,576 tokens.
- **Qwen4 concurrent sessions (`FLUX_CONCURRENCY=2..128`; engine native row ceiling `256`)**: The launcher enables native batching by default (`ATLAS_QWEN4_NATIVE_BATCH` unset) up to `QWEN4_NATIVE_MAX_ROWS=128` (the live Rust engine hard ceiling is `256` in `crates/spark-model/src/layers/mod.rs`). Setting `ATLAS_QWEN4_NATIVE_BATCH=0` forces the serialized C1-per-row path; setting `=1` refuses to start unless the native route is available. For speculation under concurrency, batched DFlash2 supports `C2-C16` on GB10 UMA (`C2-C32` on discrete Blackwell), fixed K5 native MTP supports `C2-C32`, adaptive-K native MTP supports up to `C64` (or `C128` with `FLUX_SPEC_MODE=mtp`), and wider concurrency falls back to native target-only batching. Every serialized C2+ frame is counted by reason in `atlas_qwen4_decode_frames_total`, and each serialized request is logged once as `QWEN4_REQUEST_ROUTE ... route=serial reason=<reason>`. A recipe without `ATLAS_MOE_T_LANES=1`, or with `ATLAS_SSM_H_FP16=1`, `ATLAS_LM_HEAD_TC=1`, `ATLAS_QWEN4_LMHEAD_BF16=1` or `--profile`, serializes with that reason unless the selector is `1`.
- **Native MTP beyond 2K**: K=5 remains enabled. The Issue #13 v5 qualifier pins `ATLAS_QWEN4_QSA_VERIFY_ROWS=1`, checks 2,043 through 2,050 plus every disclosed long-context length, and requires the source-pinned exact row fallback beyond the safe dense endpoint. Every K5 request must carry its own dispatch, verify-frame, completed-cycle, and runtime-request-ID evidence. The normal K5 arm must emit a verify-graph capture/use marker. A separate forced-eager K5 arm uses `ATLAS_DFLASH_DEBUG_NO_GRAPH=1`, proves completed partial-accept rollback between exact sole-slot canaries, and consumes request-scoped executed-route receipts for the exact twelve-layer QSA set. Plain-only admission probes prove 262,143 plus one is accepted, 262,144 plus one is rejected before scheduler-owned prefill counters move, and post-rejection health, response geometry, and prefill counters recover exactly; lazy chat/TUI counters are not required. A final whole-log census excludes late overflow dispatch. A stopped 262,080 semantic request rejects contradictory answer text, while a separate forced 64-token row reaches 262,144 and compares plain/K5 text and scheduler token IDs without treating fill text as semantics. Strict duplicate-key, parsed non-finite, and exponent-overflow JSON rejection plus manifest and artifact revalidation close receipt ambiguity and TOCTOU gaps. These are harness requirements, not a runtime claim. The direct binary and immutable container must each produce a PASS receipt before Issue #13 is updated or closed.
- **Extended `1M` Context via Dynamic YaRN (`FLUX_MODE=1m-nvfp4` or `FLUX_MODE=2x-spark-1m`)**: For `Qwen3.8-Flash-Next`, use `FLUX_MODE=1m-nvfp4` (1x Spark, 4-bit NVFP4 KV) or `FLUX_MODE=2x-spark-1m` (2x Spark TP=2+EP=2, lossless BF16 KV) — `flux-autodetect` rejects legacy `FLUX_MODE=512k`/`1m` on Qwen4 QSA models. Non-QSA checkpoints accept `FLUX_MODE=512k`/`1m` or explicit `--rope-yarn-factor <F>` / `--rope-theta-override <THETA>`.
- **Automatic Hardware Context Scaling (`-e FLUX_AUTO_SCALE_CONTEXT=1`)**: Automatically raises the default sequence length ceiling on high-VRAM / multi-GPU nodes (e.g. RTX PRO 6000 Blackwell 96GB, B200 192GB, H200 141GB, or multi-GPU `--tp-size 2|4|8`).

For a source-build native concurrency parity run, keep the native route isolated from long-context and speculative qualification. Set the selector to `1` so an incompatible recipe refuses to start instead of serializing:

```bash
FLUX_CONCURRENCY=32 \
FLUX_MAX_SEQ_LEN=2048 \
FLUX_KV_CACHE_DTYPE=bf16 \
FLUX_SPEC_MODE=none \
FLUX_PREFIX_CACHE=0 \
ATLAS_QWEN4_NATIVE_BATCH=1 \
flux serve /models/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid
```

The required GPU gate compares each tested active width against isolated C1 using identical prompts and positions. It checks token IDs and full logits, then repeats after row removal, slot reuse, and replacement. Run each width in a fresh process so `QWEN4_NATIVE_BATCH_ENGAGED rows=N` identifies the tested rung. That marker proves selection only. It is not a parity or speed receipt. The launcher caps `FLUX_CONCURRENCY` at `128` (`QWEN4_NATIVE_MAX_ROWS=128`); use `FLUX_MODE=c128` for `C=1..128` deployments.

```bash
# Launch with 256K native context window:
docker run --rm -it --gpus all --ipc=host -p 8899:8899 \
  -v /path/to/your/models:/models:ro \
  -e FLUX_MODE=256k \
  ghcr.io/theapathy/flux-engine:gb10

# Launch with 1M context window on 1x Spark (NVFP4 KV, YaRN factor=4):
docker run --rm -it --gpus all --ipc=host -p 8899:8899 \
  -v /path/to/your/models:/models:ro \
  -e FLUX_MODE=1m-nvfp4 \
  ghcr.io/theapathy/flux-engine:gb10

# Launch with 1M context window on 2x Spark (lossless BF16 KV, TP=2+EP=2):
docker run --rm -it --gpus all --ipc=host -p 8899:8899 \
  -v /path/to/your/models:/models:ro \
  -e FLUX_MODE=2x-spark-1m -e FLUX_TP_SIZE=2 \
  ghcr.io/theapathy/flux-engine:gb10
```

---

## 9. Privacy-First Architecture, Diagnostics & Community Uploads (`TheApathy/flux-engine`)

### A. Privacy-First by Default (`FLUX_OPT_OUT=all`, `FLUX_PRIVACY_FIRST=1`, `HF_HUB_OFFLINE=1`)
The container ships **Privacy-First out of the box**:
- **Zero External Telemetry (`--opt-out-telemetry`)**: 100% air-gapped local execution, zero outbound network calls.
- **Automatic Host & Path Redaction (`--opt-out-host-info`)**: `hostname`, kernel release, driver serials, and user home-directory (`$HOME`) paths are automatically redacted in all generated JSONs, logs, and Model Cards.
- **Zero Prompt Logging (`--opt-out-prompt-logs`)**: Live API endpoints never log user prompts/completions to disk, and raw response text is scrubbed from benchmark artifacts by default.
- *(To explicitly include raw host/driver info or prompt logs for private local debugging, pass `--include-host-info` or `--include-prompt-logs`.)*

### B. Package & Upload Diagnostics / Model Cards to `https://github.com/TheApathy/flux-engine/issues`
Run `flux bundle` to package a 100% privacy-scrubbed submission folder in `/results/submission-bundle/` (`MODEL_CARD.md`, `card.svg`, `card.png`, `flux-diagnostics.json`, `flux-server.sanitized.log`, and `SUBMISSION_MANIFEST.json`):
```bash
docker run --rm --gpus all \
  -v /path/to/your/models:/models:ro \
  -v $(pwd)/results:/results \
  ghcr.io/theapathy/flux-engine:gb10 \
  bundle
```
You can then attach `/results/submission-bundle/` to an Issue or Discussion on **[`https://github.com/TheApathy/flux-engine/issues`](https://github.com/TheApathy/flux-engine/issues)**!
