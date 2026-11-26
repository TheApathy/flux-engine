# Flux Engine — Discrete Blackwell SM120 (`linux/amd64`) Container

Multi-stage, privacy-hardened `linux/amd64` container build for discrete NVIDIA Blackwell (`sm_120a`) GPUs:

- **16 GB**: RTX 5060 Ti 16GB / RTX 5070 Ti / RTX 5080 (<=14B NVFP4 models)
- **32 GB**: RTX 5090 (`Qwen3.8-27B-NVFP4` with ~10+ GiB free VRAM for large KV pool)
- **48 GB / 72 GB**: RTX PRO 5000 Blackwell (`Qwen3.8-27B-NVFP4` with >=262K KV pool; `2x` for `Qwen3.8-Flash-Next`)
- **96 GB**: RTX PRO 6000 Blackwell (`Qwen3.8-Flash-Next-NVFP4` + DFlash2 + >=262K BF16 KV on 1 GPU; `2x`/`4x` for multi-million-token pools)

## Build

Run from the **repository root**:

```bash
docker build --platform linux/amd64 \
  -f docker/sm120/Dockerfile \
  --build-arg FLUX_TARGET_HW=rtx6000 \
  --build-arg FLUX_TARGET_MODEL='*' \
  --build-arg FLUX_TARGET_QUANT='*' \
  -t ghcr.io/theapathy/flux-engine:sm120 .
```

To cross-build `linux/amd64` from an `arm64` host (e.g. DGX Spark) with Buildx:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/sm120/Dockerfile \
  --build-arg FLUX_TARGET_HW=rtx6000 \
  -t ghcr.io/theapathy/flux-engine:sm120 \
  --load .
```

## Run

### Single Discrete GPU (`1x`)

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=16g -p 8899:8899 \
  -v /path/to/models:/models:ro \
  ghcr.io/theapathy/flux-engine:sm120
```

### Single-Host Multi-GPU (`2x` / `4x` TP+EP)

Inspect the auto-generated single-host multi-GPU launch plan (including per-rank `--gpu-ordinal`, `--world-size`, `127.0.0.1:29500`, and `NCCL_P2P_LEVEL=SYS` snippet):

```bash
docker run --rm --gpus all -v /path/to/models:/models:ro \
  ghcr.io/theapathy/flux-engine:sm120 inspect
```

Direct `spark` CLI invocation is also available via the same image:

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g --network=host \
  -v /path/to/models:/models:ro \
  ghcr.io/theapathy/flux-engine:sm120 spark serve --model-from-path /models/<model> --port 8899
```

## Privacy & Build Hygiene

- **Air-gapped by default**: `FLUX_PRIVACY_FIRST=1`, `FLUX_OPT_OUT=all`, `DO_NOT_TRACK=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- **Strict allowlist `COPY`**: Only `build-support/`, Rust crates, `kernels/`, `jinja-templates/`, `repro-kit/flux-autodetect`, `repro-kit/recipes/`, `LICENSE`, `LICENSES/`, `CREDITS.md`, `THIRD_PARTY_NOTICES.md`, and `SOURCE_AND_LICENSES.md` are copied into the image; workspace notes and local artifacts are excluded both by `.dockerignore` and by explicit `COPY` lists.
- **NCCL floor**: Enforces `libnccl2 >= 2.28` at build time for symmetric memory multi-GPU collectives.

## Full Bringup & Tuning Guide

For bare-metal / cloud spinup, `FLUX_TARGET_HW=rtx6000|gb300` native builds, PLE `/dev/shm` `mmap` offload, SM103 (`B300`/`GB300`) notes, and troubleshooting, see [docs/SM120_SM103_BRINGUP.md](../../docs/SM120_SM103_BRINGUP.md).
