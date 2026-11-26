# Flux Engine — All models on GB10 (multi-target binary)
#
# Build from repo root:
#   docker build -f docker/gb10/Dockerfile -t flux-gb10 .
#
# Run (auto-selects kernels based on model config):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     flux-gb10 serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
#
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     flux-gb10 serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
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
ENV ATLAS_TARGET_MODEL=*
ENV ATLAS_TARGET_QUANT=*

# Pinned MIT ExLlamaV3 source required by crates/atlas-kernels/build.rs when
# building universal (`ATLAS_TARGET_MODEL=*`) images including GLM-5.3-Flash EXL3.
ARG EXLLAMAV3_REV=e648f1a131365aae15920073e761a3fa5a527654
RUN git init -q /tmp/exllamav3 && \
    git -C /tmp/exllamav3 remote add origin https://github.com/turboderp-org/exllamav3.git && \
    git -C /tmp/exllamav3 fetch --depth 1 origin "${EXLLAMAV3_REV}" && \
    git -C /tmp/exllamav3 checkout -q FETCH_HEAD
ENV ATLAS_EXLLAMAV3_SOURCE=/tmp/exllamav3

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

LABEL org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/TheApathy/flux-engine"

# NCCL for multi-GPU (EP=2), RDMA userspace for RoCE transport.
# NCCL ≥ 2.28 is required for `ncclMemAlloc`/`ncclMemFree` symmetric memory
# windows (substrate for device-API fused allreduce kernels). The base
# `nvidia/cuda:13.0.0-runtime-ubuntu24.04` image has the NVIDIA CUDA APT
# repo enabled, which carries libnccl2 2.29+ (>=2.28 satisfied). Install
# the latest available, then verify the resolved version meets the floor.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 \
      # DeepSeek-V4.1 JPEG input: libjpeg-turbo decodes bit-identically to the
      # Python engine's Pillow (crates/spark-server/src/dsv41/turbojpeg.rs,
      # dlopen'd at runtime; without it JPEGs fall back to zune-jpeg).
      libturbojpeg && \
    rm -rf /var/lib/apt/lists/* && \
    # Sanity-check the linked NCCL is ≥ 2.28 — fail-fast if a future repo
    # change downgrades the available version below the symmetric-memory floor.
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" >&2; exit 1; }

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE
COPY LICENSE CREDITS.md THIRD_PARTY_NOTICES.md SOURCE_AND_LICENSES.md /usr/share/licenses/flux-engine/
COPY LICENSES/ /usr/share/licenses/flux-engine/LICENSES/

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
