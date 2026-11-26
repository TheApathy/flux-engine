#!/usr/bin/env bash
# Rebuild the three SM121a FI-GDN AOT objects from their preferred source.
# This consumes a content-pinned FlashInfer wheel and an already provisioned
# Python/CUDA environment matching toolchain.lock. It does not fetch secrets.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PYTHON FLASHINFER_WHEEL OUTPUT_DIR" >&2
  exit 64
fi

python_bin=$1
flashinfer_wheel=$2
output_dir=$3
kit_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
expected_wheel_sha=d124369346a3d48eac67e31c42f7a3c813bcc0abc10e2e36db413b7b3dfd97df
actual_wheel_sha=$(sha256sum "$flashinfer_wheel" | awk '{print $1}')
if [[ "$actual_wheel_sha" != "$expected_wheel_sha" ]]; then
  echo "FlashInfer wheel hash mismatch" >&2
  exit 65
fi

work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT
mkdir -p "$work_dir/site" "$output_dir/bf16" "$output_dir/fp16" "$output_dir/nf8"
unzip -q "$flashinfer_wheel" -d "$work_dir/site"
install -m 0644 \
  "$kit_dir/flashinfer-0.6.14-modified/delta_rule_sm120.py" \
  "$work_dir/site/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py"

PYTHONPATH="$work_dir/site${PYTHONPATH:+:$PYTHONPATH}" \
  FLUX_GDN_AOT_OUT="$output_dir/bf16" \
  "$python_bin" "$kit_dir/bf16/export_init.py"

PYTHONPATH="$work_dir/site${PYTHONPATH:+:$PYTHONPATH}" \
  FLUX_GDN_AOT_OUT="$output_dir/fp16" \
  "$python_bin" "$kit_dir/fp16/export_fp16.py"

PYTHONPATH="$kit_dir/nf8:$work_dir/site${PYTHONPATH:+:$PYTHONPATH}" \
  NF8_NORM_WARPS=1 NF8_LDST_REGS=56 \
  "$python_bin" "$kit_dir/nf8/make_variant.py" \
  "$kit_dir/flashinfer-0.6.14-modified/delta_rule_sm120.py" \
  "$work_dir/delta_rule_sm120_nf8.py"
cmp "$work_dir/delta_rule_sm120_nf8.py" "$kit_dir/nf8/nw1/delta_rule_sm120_nf8.py"

PYTHONPATH="$kit_dir/nf8:$work_dir/site${PYTHONPATH:+:$PYTHONPATH}" \
  NF8_NORM_WARPS=1 NF8_LDST_REGS=56 \
  "$python_bin" "$kit_dir/nf8/export_nf8.py" "$work_dir" "$output_dir/nf8"

sha256sum \
  "$output_dir/bf16/atlas_fi_gdn_sm121a_prefill.o" \
  "$output_dir/fp16/atlas_fi_gdn_sm121a_prefill_f16.o" \
  "$output_dir/nf8/atlas_fi_gdn_sm121a_prefill_f16_nf8.o"
