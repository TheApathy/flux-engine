# syntax=docker/dockerfile:1
# Flux Engine — Discrete NVIDIA Blackwell SM120 (linux/amd64)
# Supports: RTX 5060 Ti 16GB / RTX 5070 Ti / RTX 5080 / RTX 5090 32GB /
#           RTX PRO 5000 48-72GB / RTX PRO 6000 Blackwell 96GB (1x, 2x, 4x)
#
# Build from repository root:
#   docker build --platform linux/amd64 \
#     -f docker/sm120/Dockerfile \
#     -t ghcr.io/theapathy/flux-engine:sm120 .
#
# Single-model slim build example:
#   docker build --platform linux/amd64 \
#     --build-arg FLUX_TARGET_HW=rtx6000 \
#     --build-arg FLUX_TARGET_MODEL=qwen3.8-27b \
#     --build-arg FLUX_TARGET_QUANT=nvfp4 \
#     -f docker/sm120/Dockerfile \
#     -t ghcr.io/theapathy/flux-engine:sm120 .
#
# Run (1x discrete GPU, auto-detect model & VRAM budget):
#   docker run --rm --gpus all --ipc=host --shm-size=16g -p 8899:8899 \
#     -v /path/to/models:/models:ro \
#     ghcr.io/theapathy/flux-engine:sm120
#
# Run (2x / 4x single-host multi-GPU dry-run to inspect TP=2/TP=4 launch snippet):
#   docker run --rm --gpus all -v /path/to/models:/models:ro \
#     ghcr.io/theapathy/flux-engine:sm120 inspect

# ── Build stage (linux/amd64, CUDA 13.0 devel) ───────────────────
FROM --platform=linux/amd64 nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder

ARG FLUX_TARGET_HW=rtx6000
ARG FLUX_TARGET_MODEL=*
ARG FLUX_TARGET_QUANT=*
ARG FLUX_ENGINE_VERSION=1.0.3

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      ca-certificates curl build-essential pkg-config git cmake libclang-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=rust:1.93.1-bookworm@sha256:7c4ae649a84014c467d79319bbf17ce2632ae8b8be123ac2fb2ea5be46823f31 /usr/local/cargo /usr/local/cargo
COPY --from=rust:1.93.1-bookworm@sha256:7c4ae649a84014c467d79319bbf17ce2632ae8b8be123ac2fb2ea5be46823f31 /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}" \
    RUSTFLAGS="-C target-cpu=x86-64-v3"

WORKDIR /build

# Strict allowlist COPY: only build inputs are copied into the builder context.
# Local workspace notes, git history, and temporary build artifacts are never copied.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY build-support/ build-support/
COPY crates/ crates/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/

ENV FLUX_TARGET_HW=${FLUX_TARGET_HW} \
    FLUX_TARGET_MODEL=${FLUX_TARGET_MODEL} \
    FLUX_TARGET_QUANT=${FLUX_TARGET_QUANT} \
    ATLAS_TARGET_HW=${FLUX_TARGET_HW} \
    ATLAS_TARGET_MODEL=${FLUX_TARGET_MODEL} \
    ATLAS_TARGET_QUANT=${FLUX_TARGET_QUANT}

# Pinned Apache-2.0 FlashInfer + 3rdparty/cccl headers for build-time static
# compilation of `qwen4_fi_attn.cu` (`ATLAS_QWEN4_PREFILL_ATTN_FI=1`).
# Statically linked into `spark`; zero runtime stage bloat.
ARG FLASHINFER_REV=74fb3fb8a64cd36b8f4452a5ef1a7f865eef013e
RUN git init -q /opt/fi_src && \
    git -C /opt/fi_src remote add origin https://github.com/flashinfer-ai/flashinfer.git && \
    git -C /opt/fi_src fetch --depth 1 origin "${FLASHINFER_REV}" && \
    git -C /opt/fi_src checkout -q --detach FETCH_HEAD && \
    git -C /opt/fi_src submodule update --init --depth 1 3rdparty/cccl
ENV FLASHINFER_HOME=/opt/fi_src

# Pinned MIT ExLlamaV3 source required by crates/atlas-kernels/build.rs when
# building universal (`FLUX_TARGET_MODEL=*`) images including GLM-5.3-Flash EXL3.
ARG EXLLAMAV3_REV=e648f1a131365aae15920073e761a3fa5a527654
RUN git init -q /tmp/exllamav3 && \
    git -C /tmp/exllamav3 remote add origin https://github.com/turboderp-org/exllamav3.git && \
    git -C /tmp/exllamav3 fetch --depth 1 origin "${EXLLAMAV3_REV}" && \
    git -C /tmp/exllamav3 checkout -q FETCH_HEAD
ENV ATLAS_EXLLAMAV3_SOURCE=/tmp/exllamav3

RUN cargo build --release --locked -p spark-server

# ── Runtime stage (linux/amd64, CUDA 13.0 runtime) ───────────────
FROM --platform=linux/amd64 nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

ARG FLUX_TARGET_HW=rtx6000
ARG FLUX_ENGINE_VERSION=1.0.3

LABEL org.opencontainers.image.title="Flux Engine (Blackwell SM120 / linux-amd64)" \
      org.opencontainers.image.description="Native Rust/CUDA inference engine for discrete NVIDIA Blackwell SM120 GPUs (RTX 50-series & RTX PRO 5000/6000), with FlashInfer FA2 prefill, single- and multi-GPU NCCL serving, and local qualification tools." \
      org.opencontainers.image.version="${FLUX_ENGINE_VERSION}" \
      org.opencontainers.image.vendor="TheApathy" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/TheApathy/flux-engine" \
      org.opencontainers.image.url="https://github.com/TheApathy/flux-engine" \
      com.theapathy.flux.version="${FLUX_ENGINE_VERSION}" \
      com.theapathy.flux.arch="sm_120a" \
      com.theapathy.flux.target-hw="${FLUX_TARGET_HW}" \
      com.theapathy.flux.privacy="privacy-first-airgapped"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FLUX_ENGINE_VERSION="${FLUX_ENGINE_VERSION}" \
    FLUX_ARCH_TARGET="sm_120a" \
    FLUX_MODE="auto" \
    FLUX_HOST="0.0.0.0" \
    FLUX_PORT="8899" \
    FLUX_CONCURRENCY="1" \
    FLUX_MAX_SEQ_LEN="131072" \
    FLUX_PRIVACY_FIRST="1" \
    FLUX_OPT_OUT="all" \
    DO_NOT_TRACK="1" \
    HF_HUB_OFFLINE="1" \
    TRANSFORMERS_OFFLINE="1" \
    CUDA_MODULE_LOADING="LAZY" \
    NCCL_P2P_DISABLE="0" \
    NCCL_P2P_LEVEL="SYS" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    LIL_BENCH_DIR="/opt/flux/llm-inference-bench" \
    LLM_BENCH_NO_UPDATE_CHECK="1" \
    RUST_LOG="info" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-13.0/compat"

# NCCL >= 2.28 for single-host multi-GPU (TP/EP/PP) symmetric memory windows,
# RDMA userspace libs, CUDA 13.0 runtime libraries, libturbojpeg for multimodal
# JPEG decoding, and Python 3 + stdlib/bench helpers for flux-autodetect.
RUN apt-mark unhold libcublas-13-0 libnccl2 2>/dev/null || true && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      ca-certificates \
      curl \
      jq \
      python3 \
      python3-httpx \
      python3-psutil \
      python3-rich \
      python3-numpy \
      procps \
      libgomp1 \
      libnuma1 \
      libssl3 \
      libturbojpeg \
      libnccl2 \
      libibverbs1 \
      librdmacm1 \
      ibverbs-providers \
      libnl-3-200 \
      libnl-route-3-200 \
      libcublas-13-0 \
      cuda-cudart-13-0 \
      cuda-nvrtc-13-0 \
      libnvjitlink-13-0 \
      libnvfatbin-13-0 \
      libcufile-13-0 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" >&2; exit 1; } && \
    echo "/usr/local/cuda-13.0/compat" > /etc/ld.so.conf.d/zz-cuda-compat.conf && \
    ldconfig && \
    mkdir -p /opt/flux/bin /opt/flux/scripts /opt/flux/repro-kit/recipes /opt/flux/llm-inference-bench /models /results && \
    chmod 1777 /results

# Explicit allowlist COPY into runtime stage — never copies local workspace notes.
COPY --from=builder /build/target/release/spark /opt/flux/bin/flux-server-real
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY repro-kit/flux-autodetect /usr/local/bin/flux
COPY repro-kit/flux-bench /usr/local/bin/flux-bench
COPY repro-kit/flux_lens.py /usr/local/bin/flux-lens
COPY repro-kit/lil-bench /usr/local/bin/lil-bench
COPY repro-kit/scripts/repack_qwen38_flash_next_ple_nvfp4.py /opt/flux/scripts/repack_qwen38_flash_next_ple_nvfp4.py
COPY repro-kit/llm-inference-bench /opt/flux/llm-inference-bench
COPY repro-kit/recipes/ /opt/flux/repro-kit/recipes/
COPY repro-kit/benchmark-config.yaml repro-kit/model-manifest.json repro-kit/METHODOLOGY.md repro-kit/README.md repro-kit/SOURCE_AND_LICENSES.md repro-kit/FLUX_LENS.md /opt/flux/repro-kit/
COPY LICENSE /LICENSE
COPY LICENSE CREDITS.md THIRD_PARTY_NOTICES.md SOURCE_AND_LICENSES.md /usr/share/licenses/flux-engine/
COPY LICENSES/ /usr/share/licenses/flux-engine/LICENSES/

RUN chmod 0755 \
      /opt/flux/bin/flux-server-real \
      /usr/local/bin/flux \
      /usr/local/bin/flux-bench \
      /usr/local/bin/flux-lens \
      /usr/local/bin/lil-bench \
      /opt/flux/scripts/repack_qwen38_flash_next_ple_nvfp4.py && \
    ln -sf /opt/flux/bin/flux-server-real /opt/flux/bin/flux-server-universal && \
    ln -sf /opt/flux/bin/flux-server-real /opt/flux/bin/flux-server-glm53 && \
    ln -sf /opt/flux/bin/flux-server-real /opt/flux/bin/flux-server-qwen38-27b && \
    ln -sf /opt/flux/bin/flux-server-real /usr/local/bin/flux-server && \
    ln -sf /opt/flux/bin/flux-server-real /usr/local/bin/spark && \
    ln -sf /usr/local/bin/flux /usr/local/bin/flux-autodetect && \
    ln -sf /opt/flux/scripts/repack_qwen38_flash_next_ple_nvfp4.py /usr/local/bin/repack-ple

WORKDIR /opt/flux
EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --start-period=150s --retries=3 \
    CMD curl -sf http://127.0.0.1:${FLUX_PORT:-8899}/health || exit 1

ENTRYPOINT ["/usr/local/bin/flux"]
CMD ["serve"]
