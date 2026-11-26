# Flux Engine (derived from Atlas) — Qwen3.6-27B-FP8 (Dense Hybrid SSM+Attn) on GB10
#
# Build from repo root:
#   docker build -f docker/gb10/qwen3.6-27b/nvfp4/Dockerfile -t atlas-gb10:qwen3.6-27b .
#
# Run (target only, no DFlash):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-gb10:qwen3.6-27b serve Qwen/Qwen3.6-27B-FP8 \
#       --max-seq-len 16384
#
# Run (with DFlash drafter pairing):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-gb10:qwen3.6-27b serve Qwen/Qwen3.6-27B-FP8 \
#       --dflash --draft-model z-lab/Qwen3.6-27B-DFlash --max-seq-len 16384

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=rust:1.93.1-bookworm@sha256:7c4ae649a84014c467d79319bbf17ce2632ae8b8be123ac2fb2ea5be46823f31 /usr/local/cargo /usr/local/cargo
COPY --from=rust:1.93.1-bookworm@sha256:7c4ae649a84014c467d79319bbf17ce2632ae8b8be123ac2fb2ea5be46823f31 /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
# nvcc must be ON PATH, not merely findable via CUDA_HOME: cudarc 0.19.2's
# build.rs:115 shells out to a bare `nvcc --version` and panics if it is not
# resolvable, regardless of CUDA_HOME/CUDA_PATH/CUDA_ROOT. The nvidia/cuda
# devel images happen to set this themselves, so today the build works by
# INHERITANCE rather than by declaration — which breaks silently on a base
# image bump. Declare it.
ENV PATH="/usr/local/cuda/bin:${PATH}"

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY build-support/ build-support/
COPY crates/ crates/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=qwen3.6-27b
ENV ATLAS_TARGET_QUANT=nvfp4

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

LABEL org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/TheApathy/flux-engine"

# NCCL for multi-GPU (EP=2), RDMA userspace for RoCE transport
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null); \
    if [ -z "$NCCL_VER" ]; then \
      echo "ERROR: libnccl2 is not installed — the install step above failed"; exit 1; \
    fi && \
    echo "Linked NCCL: $NCCL_VER" && \
    if ! dpkg --compare-versions "$NCCL_VER" ge "2.28"; then \
      echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable"; exit 1; \
    fi

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE
COPY LICENSE CREDITS.md THIRD_PARTY_NOTICES.md SOURCE_AND_LICENSES.md /usr/share/licenses/flux-engine/
COPY LICENSES/ /usr/share/licenses/flux-engine/LICENSES/

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
