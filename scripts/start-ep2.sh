#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Launch Flux Engine with Expert Parallelism (EP=2) across two GB10 nodes.
#
# Usage:
#   HEAD_IP=<head-ip> WORKER_IP=<worker-ip> ./scripts/start-ep2.sh [MODEL]
#
# Default model: Sehyo/Qwen3.5-122B-A10B-NVFP4
#
# Prerequisites:
#   - Two GB10 nodes connected via RoCEv2 (configure NCCL_SOCKET_IFNAME / NCCL_IB_HCA)
#   - Passwordless SSH from head (HEAD_IP env) to worker (WORKER_IP env)
#   - Identical Flux Engine Docker image tag on BOTH nodes (mixing versions
#     across ranks can hang NCCL during ncclCommInitRank)
#   - Model weights cached on both nodes (~/.cache/huggingface)
#   - RDMA kernel support on host (IB device at /dev/infiniband)
#
# The head node (rank 0) runs the HTTP server + scheduler.
# The worker node (rank 1) mirrors model operations for EP collectives.
#
# Recommended GPU_MEM_UTIL for 122B-class models: 0.90 (0.70 can be too tight
# for MoE weight transposition). For MiniMax M2.x checkpoints or target-only
# runs, set SPECULATIVE=0 (and pass any additional flags via EXTRA_SERVE_ARGS).

set -euo pipefail

MODEL="${1:-Sehyo/Qwen3.5-122B-A10B-NVFP4}"
IMAGE="${IMAGE:-ghcr.io/theapathy/flux-engine:gb10}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
HEAD_IP="${HEAD_IP:?Set HEAD_IP to the head node IPv4 address}"
WORKER_IP="${WORKER_IP:?Set WORKER_IP to the worker node IPv4 address}"
MASTER_PORT="${MASTER_PORT:-29500}"
PORT="${PORT:-8888}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SPECULATIVE="${SPECULATIVE:-1}"
MTP_QUANT="${MTP_QUANT:-nvfp4}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
DEV_BINARY="${DEV_BINARY:-}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

echo "=== Flux Engine EP=2 Launch (RDMA-enabled) ==="
echo "Image:  $IMAGE"
echo "Model:  $MODEL"
echo "Head:   $HEAD_IP (rank 0, HTTP on port $PORT)"
echo "Worker: $WORKER_IP (rank 1)"
echo "GPU mem utilization: $GPU_MEM_UTIL"
echo "MTP quantization:   $MTP_QUANT"
echo ""

# Helper: safely quote an argv array into a single remote shell command string.
quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf '%q ' "$arg")"
  done
  printf '%s' "${out% }"
}

# Stop any existing containers
echo "Cleaning up old containers..."
"$DOCKER_BIN" rm -f flux-ep0 2>/dev/null || true
ssh "$WORKER_IP" "$(quote_cmd "$DOCKER_BIN" rm -f flux-ep1) 2>/dev/null || true"

# Optional dev binary mount (overrides container's /usr/local/bin/spark)
DEV_MOUNT=()
DEV_MOUNT_WORKER=()
if [[ -n "$DEV_BINARY" ]]; then
  [[ -f "$DEV_BINARY" ]] || { echo "ERROR: DEV_BINARY not found: $DEV_BINARY" >&2; exit 1; }
  DEV_MOUNT=(-v "${DEV_BINARY}:/usr/local/bin/spark:ro")
  scp "$DEV_BINARY" "${WORKER_IP}:/tmp/spark-dev"
  DEV_MOUNT_WORKER=(-v "/tmp/spark-dev:/usr/local/bin/spark:ro")
  echo "Using dev binary: $DEV_BINARY"
fi

# RDMA device and capability flags
RDMA_FLAGS=(
  --device=/dev/infiniband
  --cap-add=IPC_LOCK
  --ulimit memlock=-1
)

# NCCL environment for RoCEv2 transport over IPv4
NCCL_ENV=(
  -e NCCL_IB_DISABLE=0
  -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_TIMEOUT=22
  -e NCCL_IB_RETRY_CNT=7
  -e NCCL_NET_GDR_LEVEL=0
  -e NCCL_NET_GDR_C2C=0
  -e NCCL_DMABUF_ENABLE=0
  -e NCCL_NVLS_ENABLE=0
  -e NCCL_CUMEM_HOST_ENABLE=0
  -e NCCL_PROTO=Simple
  -e NCCL_ALGO=Ring
  -e NCCL_MIN_NCHANNELS=1
  -e NCCL_MAX_NCHANNELS=2
  -e NCCL_DEBUG=INFO
  -e NCCL_DEBUG_SUBSYS=INIT,NET
)
if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
  NCCL_ENV+=(-e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
fi
if [[ -n "$NCCL_IB_HCA" ]]; then
  NCCL_ENV+=(-e "NCCL_IB_HCA=${NCCL_IB_HCA}")
fi

# Speculative decoding flags (set SPECULATIVE=0 for target-only serving)
SPEC_FLAGS=()
if [[ "$SPECULATIVE" == "1" ]]; then
  SPEC_FLAGS=(--speculative --mtp-quantization "$MTP_QUANT")
fi

# Optional extra CLI arguments forwarded to both ranks (word-split into array)
EXTRA_ARGS=()
if [[ -n "$EXTRA_SERVE_ARGS" ]]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA_SERVE_ARGS"
fi

# Start rank 0 (head) — HTTP server + scheduler
echo "Starting rank 0 on $HEAD_IP..."
"$DOCKER_BIN" run -d \
  --name flux-ep0 \
  --gpus all \
  --ipc=host \
  --network host \
  "${RDMA_FLAGS[@]}" \
  "${NCCL_ENV[@]}" \
  -e RUST_LOG=info \
  "${DEV_MOUNT[@]}" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "$IMAGE" serve "$MODEL" \
    --rank 0 \
    --world-size 2 \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    --port "$PORT" \
    --max-batch-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype nvfp4 \
    --enable-prefix-caching \
    "${SPEC_FLAGS[@]}" \
    "${EXTRA_ARGS[@]}"

# Start rank 1 (worker) — EP worker loop only (quoted safely via printf %q)
REMOTE_HF_CACHE="${REMOTE_HF_CACHE:-$HF_CACHE_DIR}"
REMOTE_CMD=(
  "$DOCKER_BIN" run -d
  --name flux-ep1
  --gpus all
  --ipc=host
  --network host
  "${RDMA_FLAGS[@]}"
  "${NCCL_ENV[@]}"
  -e RUST_LOG=info
  "${DEV_MOUNT_WORKER[@]}"
  -v "${REMOTE_HF_CACHE}:/root/.cache/huggingface"
  "$IMAGE" serve "$MODEL"
    --rank 1
    --world-size 2
    --master-addr "$HEAD_IP"
    --master-port "$MASTER_PORT"
    --port 0
    --max-batch-size 1
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --kv-cache-dtype nvfp4
    --enable-prefix-caching
    "${SPEC_FLAGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "Starting rank 1 on $WORKER_IP..."
ssh "$WORKER_IP" "$(quote_cmd "${REMOTE_CMD[@]}")"

echo ""
echo "=== Both ranks starting ==="
echo "Monitor rank 0: $DOCKER_BIN logs -f flux-ep0"
echo "Monitor rank 1: ssh $WORKER_IP '$DOCKER_BIN logs -f flux-ep1'"
echo "API endpoint:   http://$HEAD_IP:$PORT/v1/chat/completions"
echo ""
echo "Wait for both ranks to complete NCCL init before sending requests."
echo "Look for 'NCCL initialized' and 'EP worker ready' in logs."
echo "Check for 'NET/IB' in NCCL_DEBUG output to confirm RDMA transport."
