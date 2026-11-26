# DeepSeek-V4 Single-Node Benchmarks & Probes (`bench/deepseek-v4/`)

Benchmark scripts, accuracy/long-generation quality gates, and an offline DSpark drafter acceptance probe for single-node **DeepSeek-V4-Flash** (`model_type=deepseek_v4`, REAP FP8/MXFP4 checkpoints) on NVIDIA GB10 / Blackwell GPUs.

## Quick Start

Set `MODEL_DIR` (or `FLUX_MODELS_DIR`) to your local DeepSeek-V4-Flash checkpoint directory:

```bash
# 1. Start a single-node server (defaults to 127.0.0.1:8899, FP8 KV cache, unified MoE layout)
MODEL_DIR=/path/to/DeepSeek-V4-Flash-162B bash bench/deepseek-v4/serve_single.sh

# 2. First-token coherence smoke check
PORT=8899 bash bench/deepseek-v4/smoke.sh

# 3. Single-stream decode throughput (3 reps after warmup)
PORT=8899 NTOK=256 bash bench/deepseek-v4/decode_bench.sh

# 4. Concurrent aggregate decode throughput (start server with MAX_BATCH >= CONC)
PORT=8899 CONC=12 NTOK=256 bash bench/deepseek-v4/concurrent_bench.sh
```

## Scripts & Files

| File | Purpose |
| :--- | :--- |
| [`serve_single.sh`](serve_single.sh) | Launches `flux-server` / `spark` in single-node (`EP=1`) mode with FP8 KV cache (`KV_DTYPE=fp8`), unified MoE layout (`ATLAS_UNIFIED_MOE_LAYOUT=1`), and configurable `MODEL_DIR`, `PORT`, `MAX_SEQ`, `MAX_BATCH`, and `EXTRA_ARGS`. |
| [`smoke.sh`](smoke.sh) | Sends a greedy sanity prompt to `/v1/chat/completions` and verifies coherent completion output. |
| [`decode_bench.sh`](decode_bench.sh) | Measures single-stream wall-clock decode throughput and extracts server-reported per-request `tok/s` and TTFT from `LOG`. |
| [`concurrent_bench.sh`](concurrent_bench.sh) | Measures aggregate and per-stream decode throughput across distinct prompts at `CONC` concurrent requests (or comma-separated `SWEEP`). |
| [`mtp_bench.py`](mtp_bench.py) | 9-workload speculative/plain wall-clock benchmark (512 tokens each) with optional log parsing for per-step acceptance rate and tokens/step, plus `--table` comparison mode. |
| [`quality_probe.py`](quality_probe.py) | Greedy 4-prompt coherence check + 12-item arithmetic reasoning accuracy probe (`--min-acc 0.80`). |
| [`longgen_gate.py`](longgen_gate.py) | Long-generation structural & anti-degeneration quality gate (`numbered-20`, `structured-essay`, `long-chain-arithmetic`, `code-block`), supporting `--save-baseline` and regression gating via `--baseline` (see [`longgen_baseline_bf16.json`](longgen_baseline_bf16.json)). |
| [`dspark_probe/`](dspark_probe/) | Offline DSpark drafter acceptance probe (`probe.py`) replaying captured target hidden-state dumps (`dspark_dump.bin`) through the DeepSeek reference `model.py` and pure-PyTorch `kernel.py` shim. |

## Environment Variables

- `FLUX_MODELS_DIR` — Base directory for default model paths (`MODEL_DIR`, `MODEL`, `DSPARK_DRAFTER_DIR`, `DSPARK_TARGET_DIR`).
- `DS4_BIN` — Path to the compiled server binary (defaults to `<repo>/target/release/spark`).
- `HOST` / `PORT` — Target server address and port (default: `127.0.0.1:8899`).
- `LOG` — Path to the server log file inspected by `decode_bench.sh` / `mtp_bench.py`.
