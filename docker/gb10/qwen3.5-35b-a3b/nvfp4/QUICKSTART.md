# Flux Engine — Qwen3.5-35B-A3B Quick Start

Run Qwen3.5-35B-A3B on a single NVIDIA GB10 GPU with NVFP4 weights and MTP speculative decoding.

## Requirements

- NVIDIA DGX Spark (GB10, SM121)
- Docker with `--gpus` support (NVIDIA Container Toolkit)
- ~22 GB disk for model weights

## 1. Download the model

```bash
pip install -U "huggingface_hub"
hf download Kbenkhaled/Qwen3.5-35B-A3B-NVFP4
```

This caches to `~/.cache/huggingface/hub/` (~22 GB).

## 2. Build or pull the Docker image

Pull the published multi-target GB10 image from GHCR:

```bash
docker pull ghcr.io/theapathy/flux-engine:gb10
```

Or build the dedicated per-model image from the repository root:

```bash
docker build -f docker/gb10/qwen3.5-35b-a3b/nvfp4/Dockerfile -t flux-qwen35-35b .
```

## 3. Run

```bash
docker run --gpus all --ipc=host -p 8888:8888 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  flux-qwen35-35b \
  serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
  --speculative --kv-cache-dtype nvfp4 --mtp-quantization nvfp4 \
  --scheduling-policy slai --max-seq-len 131072
```

The server starts on port 8888 after ~90 seconds of model loading.

## 4. Test

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"m","messages":[{"role":"user","content":"Explain how sparse Mixture-of-Experts routing reduces inference latency."}],"max_tokens":256}'
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--speculative` | off | Enable MTP speculative decoding |
| `--kv-cache-dtype` | `fp8` | KV cache format (`fp8`, `nvfp4`, `bf16`) |
| `--mtp-quantization` | `bf16` | MTP head quantization (`nvfp4` saves memory) |
| `--scheduling-policy` | `fifo` | Scheduler (`fifo` or `slai` for SLO-aware) |
| `--max-seq-len` | `32768` | Maximum sequence length (tokens) |
| `--port` | `8888` | HTTP port |
| `--max-batch-size` | `8` | Max sequences per GPU decode step |
| `--max-num-seqs` | `128` | Max concurrent sequences in flight |
| `--max-prefill-tokens` | `8192` | Chunked prefill size (0 = process entire prompt at once) |

## OpenAI-compatible API

The server exposes `/v1/chat/completions` and `/v1/models`. Point any OpenAI-compatible client at `http://localhost:8888/v1` with any API key.
