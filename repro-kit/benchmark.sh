#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# ==============================================================================
# Flux Reproducibility Kit v0.1 — One-Command Benchmark & Verification Script
# Works both inside the OCI container (/opt/flux/benchmark.sh) and directly on
# a host (1x DGX Spark, 2x/4x DGX Spark cluster, or SM90/SM100/SM120 GPU).
# Sanitization Policy: FLUX-SEC-0001
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUX_BENCH="${SCRIPT_DIR}/flux-bench"
if [[ ! -x "${FLUX_BENCH}" ]] && command -v flux-bench >/dev/null 2>&1; then
  FLUX_BENCH="$(command -v flux-bench)"
fi

CONFIG_PATH="${FLUX_CONFIG:-${SCRIPT_DIR}/benchmark-config.yaml}"
MANIFEST_PATH="${FLUX_MANIFEST:-${SCRIPT_DIR}/model-manifest.json}"
MODELS_DIR="${MODELS_ROOT:-/models}"
OUTPUT_DIR="${RESULTS_DIR:-/results}"
MODE="${FLUX_MODE:-max-prefill}"
TOPOLOGY="${FLUX_TOPOLOGY:-auto}"
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --topology)
      TOPOLOGY="$2"
      shift 2
      ;;
    --verify-reference)
      VERIFY_ONLY=1
      shift
      ;;
    *)
      echo "[benchmark.sh] Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo " Flux Reproducibility Kit v0.1 — Benchmark Execution Suite"
echo " Mode: ${MODE} | Topology: ${TOPOLOGY} | Output: ${OUTPUT_DIR}"
echo "======================================================================"

# Step 1: Hardware & UMA Floor Inspection
python3 -B "${FLUX_BENCH}" inspect-hw --topology "${TOPOLOGY}"
python3 -B "${FLUX_BENCH}" inspect-hw --topology "${TOPOLOGY}" --json > "${OUTPUT_DIR}/environment.json"

# Step 2: Model Manifest & SHA-256 Pin Verification
python3 -B "${FLUX_BENCH}" verify-manifest \
  --models-dir "${MODELS_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --allow-offline-manifest

# Step 3: Run Prefill Sweep (8K..128K x N=3), Same-Prompt Bit-Identity Gate,
#         and 10-Prompt Lossless K=5 Speculative Decode Gate
EXTRA_ARGS=()
if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
  EXTRA_ARGS+=("--verify-reference")
fi

python3 -B "${FLUX_BENCH}" run \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_DIR}/results.json" \
  --mode "${MODE}" \
  --topology "${TOPOLOGY}" \
  "${EXTRA_ARGS[@]}"

echo "[benchmark.sh] PASS: Wrote ${OUTPUT_DIR}/results.json and ${OUTPUT_DIR}/environment.json"
