#!/usr/bin/env bash
# Row-batched EXL3 kernel equivalence: compile AND execute rowbatch_equiv, then write a
# bound receipt (receipt.json) for make_evidence.py --flux. Needs a GB10 (sm_121) GPU.
#
#   harness/run.sh [out_dir]              # default out_dir = harness/out
#   ATLAS_EXLLAMAV3_SOURCE=<exllamav3 checkout @ e648f1a1>  (default ${TMPDIR:-/tmp}/exllamav3)
#   COMPILE_ONLY=1 harness/run.sh         # build + hash only, no GPU; receipt marks executed=false
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=${1:-$HERE/out}; mkdir -p "$OUT"
EXL=${ATLAS_EXLLAMAV3_SOURCE:-${TMPDIR:-/tmp}/exllamav3}
NVCC=${NVCC:-${CUDA_HOME:+${CUDA_HOME}/bin/nvcc}}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
KERNEL=kernels/gb10/glm5.3-flash/exl3/glm53_exl3_k4_cb2.cu
BIN=$OUT/rowbatch_equiv
# A receipt must describe THIS invocation only: never let an earlier binary or log stand in for it.
if ! rm -f "$BIN" "$OUT/build.log" "$OUT/run.log" "$OUT/receipt.json" "$OUT/receipt.json.tmp"; then
  printf 'Cannot invalidate prior harness artifacts in %s; aborting. Do not use a remaining receipt. Retry in a fresh output directory.\n' "$OUT" >&2
  exit 1
fi
set +e
"$NVCC" -arch sm_121f -O3 --fmad false -DTQ_PLUS_SIGNS -std c++17 \
  -I"$EXL/exllamav3/exllamav3_ext" -Xcudafe --diag_suppress=20012 \
  -o "$BIN" "$HERE/rowbatch_equiv.cu" > "$OUT/build.log" 2>&1
build_rc=$?
run_rc=null; executed=false
if [ $build_rc = 0 ] && [ "${COMPILE_ONLY:-0}" != 1 ]; then
  "$BIN" > "$OUT/run.log" 2>&1; run_rc=$?; executed=true
fi
set -e
python3 - "$OUT" "$REPO" "$HERE" "$KERNEL" "$EXL" "$NVCC" "$build_rc" "$run_rc" "$executed" <<'PY'
import hashlib, json, pathlib, re, subprocess, sys
out, repo, here, kernel, exl, nvcc, build_rc, run_rc, executed = sys.argv[1:]
out, repo, here = map(pathlib.Path, (out, repo, here))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if pathlib.Path(p).exists() else None
log = (out / "run.log").read_text() if executed == "true" and (out / "run.log").exists() else ""
results = re.findall(r"^RESULT: (BIT-IDENTICAL|DIFFER) \((\d+)", log, re.M)
m = results[0] if len(results) == 1 else None  # exactly one final summary, or no result


def cmd(*argv):
    """Metadata probes never abort the receipt: a missing tool is recorded as None."""
    try:
        r = subprocess.run(list(argv), capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
sys.path.insert(0, str(here.parent))
import ts_identity
exl_rev = cmd("git", "-C", exl, "rev-parse", "HEAD")
exl_status = cmd("git", "-C", exl, "status", "--porcelain", "--untracked-files=no")
gpu = cmd("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
nvcc_v = cmd(nvcc, "--version")
receipt = {
    "schema": "atlas.glm53.rowbatch_harness.v1",
    "harness_source_sha256": sha(here / "rowbatch_equiv.cu"),
    "kernel_source": kernel, "kernel_source_sha256": sha(repo / kernel),
    "spec_source_fnv1a64": ts_identity.spec_source_fnv1a64(repo),
    "exllamav3_rev": exl_rev,
    # the harness #includes exllamav3 headers: an edited checkout at the pinned rev is not the pinned source
    "exllamav3_dirty": exl_status is None or exl_status != "",
    "nvcc": nvcc_v.splitlines()[-1:] if nvcc_v else None,
    "binary_sha256": sha(out / "rowbatch_equiv") if int(build_rc) == 0 else None,
    "build_rc": int(build_rc), "executed": executed == "true",
    "run_rc": None if run_rc == "null" else int(run_rc),
    "result": m[0] if m else None, "failing_cases": int(m[1]) if m else None,
    "result_lines": len(results),
    "gpu": gpu if executed == "true" else None,
}
tmp = out / "receipt.json.tmp"
tmp.write_text(json.dumps(receipt, indent=1) + "\n")
tmp.replace(out / "receipt.json")  # atomic: a reader sees the old (deleted) or the complete new receipt
print(json.dumps(receipt))
PY
[ "$build_rc" = 0 ] && { [ "${COMPILE_ONLY:-0}" = 1 ] || [ "$run_rc" = 0 ]; }
