# syntax=docker/dockerfile:1
# Release-packaging runtime Dockerfile (build context: repro-kit/; expects pre-staged bin/flux-server-real).
# To build from source (which executes `COPY build-support/ build-support/` from repository root),
# use docker/gb10/Dockerfile or docker/sm120/Dockerfile.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

ARG FLUX_ENGINE_VERSION="1.0.2"
ARG FLUX_SOURCE_REVISION
ARG FLUX_SOURCE_TREE
ARG FLUX_BINARY_SHA256
ARG FLUX_RECIPE_SHA256
ARG FLUX_INTERACTIVE_RECIPE_SHA256
ARG FLUX_HIGH_PREFILL_RECIPE_SHA256
ARG FLUX_BINARY_NAME=flux-server-real
ARG FLUX_RECIPE_NAME=flux-nextgen-max-prefill-decode.env

LABEL org.opencontainers.image.title="Flux Engine (GB10 / SM121 Blackwell)" \
      org.opencontainers.image.description="Native Rust/CUDA inference engine for NVIDIA DGX Spark (GB10), with model auto-detection, OpenAI/Anthropic endpoints, and local qualification tools." \
      org.opencontainers.image.version="${FLUX_ENGINE_VERSION}" \
      org.opencontainers.image.ref.name="1.0.2" \
      org.opencontainers.image.revision="${FLUX_SOURCE_REVISION}" \
      org.opencontainers.image.vendor="TheApathy" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/TheApathy/flux-engine" \
      org.opencontainers.image.url="https://github.com/TheApathy/flux-engine" \
      org.opencontainers.image.documentation="https://github.com/TheApathy/flux-engine/blob/main/README.md" \
      com.theapathy.flux.version="${FLUX_ENGINE_VERSION}" \
      com.theapathy.flux.source-tree="${FLUX_SOURCE_TREE}" \
      com.theapathy.flux.binary-sha256="${FLUX_BINARY_SHA256}" \
      com.theapathy.flux.recipe-sha256="${FLUX_RECIPE_SHA256}" \
      com.theapathy.flux.binary-path="/opt/flux/bin/${FLUX_BINARY_NAME}" \
      com.theapathy.flux.recipe-path="/opt/flux/repro-kit/recipes/${FLUX_RECIPE_NAME}" \
      com.theapathy.flux.performance.interactive.recipe-path="/opt/flux/repro-kit/recipes/flux-v9-w4096-interactive.env" \
      com.theapathy.flux.performance.interactive.recipe-sha256="${FLUX_INTERACTIVE_RECIPE_SHA256}" \
      com.theapathy.flux.performance.high-prefill.recipe-path="/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env" \
      com.theapathy.flux.performance.high-prefill.recipe-sha256="${FLUX_HIGH_PREFILL_RECIPE_SHA256}" \
      com.theapathy.flux.target="qwen3.8-flash-next" \
      com.theapathy.flux.privacy="privacy-first-airgapped" \
      com.theapathy.flux.diagnostics-repo="https://github.com/TheApathy/flux-engine/issues"

# FLUX_MAX_SEQ_LEN=131072 is a compatibility sentinel.  The launcher may raise
# it to the selected model or mode minimum.  Set another value to force an
# explicit KV-cache budget instead of opting every image into a heavy 262K
# allocation at startup.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FLUX_ENGINE_VERSION="${FLUX_ENGINE_VERSION}" \
    FLUX_TARGET="qwen3.8-flash-next" \
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
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    LIL_BENCH_DIR="/opt/flux/llm-inference-bench" \
    FLASHINFER_HOME="/opt/flux/fi_home" \
    LLM_BENCH_NO_UPDATE_CHECK="1" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-13.0/compat"

RUN apt-mark unhold libcublas-13-0 libnccl2 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        jq \
        python3 \
        python3-httpx \
        python3-psutil \
        python3-rich \
        python3-numpy \
        librsvg2-bin \
        fonts-dejavu-core \
        procps \
        libgomp1 \
        libnuma1 \
        libibverbs1 \
        libssl3 \
        libcublas-13-0 \
        cuda-cudart-13-0 \
        cuda-nvrtc-13-0 \
        libnvjitlink-13-0 \
        libnvfatbin-13-0 \
        libcufile-13-0 \
        libnccl2 \
    && rm -rf /var/lib/apt/lists/* \
    && echo "/usr/local/cuda-13.0/compat" > /etc/ld.so.conf.d/zz-cuda-compat.conf \
    && ldconfig \
    && mkdir -p /opt/flux/bin /opt/flux/cache /opt/flux/fi_home /opt/flux/models/Qwen3.8-Flash-Next-Inferact-MTP /opt/flux/repro-kit/recipes /opt/flux/llm-inference-bench /models /results

# Copy the pre-staged serving binary and pre-seeded cuBLASLt autotune cache.
# Optional external MTP sidecar checkpoints should be mounted at runtime under /models or /opt/flux/models/.
COPY bin/flux-server-real /opt/flux/bin/flux-server-real
COPY cache/cublaslt_plans_v2.json /opt/flux/cache/cublaslt_plans_v2.json

# Copy universal CLI, benchmark runners, recipes, and bundled local-inference-lab (llm-inference-bench v0.7.9)
COPY flux-autodetect /usr/local/bin/flux
COPY flux-bench /usr/local/bin/flux-bench
COPY flux_lens.py /usr/local/bin/flux-lens
COPY lil-bench /usr/local/bin/lil-bench
COPY scripts/repack_qwen38_flash_next_ple_nvfp4.py /opt/flux/scripts/repack_qwen38_flash_next_ple_nvfp4.py
COPY llm-inference-bench /opt/flux/llm-inference-bench
COPY recipes /opt/flux/repro-kit/recipes
COPY validate-image-provenance.py /opt/flux/repro-kit/validate-image-provenance.py
COPY environment.json benchmark-config.yaml results.json model-manifest.json METHODOLOGY.md README.md SOURCE_AND_LICENSES.md /opt/flux/repro-kit/
COPY FLUX_LENS.md /opt/flux/repro-kit/FLUX_LENS.md

# Bind release labels to the files copied into this image, rather than accepting
# digest strings for an unrelated host binary or recipe. Performance profiles
# are separately bound, not an assertion that their runtime gates have passed.
# The outer inspector also verifies copied bytes and source provenance.
RUN python3 /opt/flux/repro-kit/validate-image-provenance.py \
        --version "$FLUX_ENGINE_VERSION" \
        --source-revision "$FLUX_SOURCE_REVISION" \
        --source-tree "$FLUX_SOURCE_TREE" \
        --binary-name "$FLUX_BINARY_NAME" \
        --binary-sha256 "$FLUX_BINARY_SHA256" \
        --recipe-name "$FLUX_RECIPE_NAME" \
        --recipe-sha256 "$FLUX_RECIPE_SHA256" \
        --interactive-recipe-sha256 "$FLUX_INTERACTIVE_RECIPE_SHA256" \
        --high-prefill-recipe-sha256 "$FLUX_HIGH_PREFILL_RECIPE_SHA256"

RUN chmod 0755 \
        /usr/local/bin/flux \
        /usr/local/bin/flux-bench \
        /usr/local/bin/flux-lens \
        /usr/local/bin/lil-bench \
        /opt/flux/scripts/repack_qwen38_flash_next_ple_nvfp4.py \
    && chmod 1777 /results \
    && ln -sf /opt/flux/bin/flux-server-real /usr/local/bin/flux-server \
    && ln -sf /usr/local/bin/flux /usr/local/bin/flux-autodetect \
    && ln -sf /opt/flux/scripts/repack_qwen38_flash_next_ple_nvfp4.py /usr/local/bin/repack-ple

WORKDIR /opt/flux/repro-kit
EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --start-period=150s --retries=3 \
    CMD curl -sf http://127.0.0.1:${FLUX_PORT:-8899}/health || exit 1

ENTRYPOINT ["/usr/local/bin/flux"]
CMD ["serve"]
