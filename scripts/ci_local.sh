#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Run Flux Engine's core CI hygiene, lint, and no-GPU Rust/Python checks locally
# without needing `act` or Docker. Mirrors `.github/workflows/ci.yml`.
#
# Usage:
#   ./scripts/ci_local.sh          # full local check (fast lint + clippy + tests)
#   ./scripts/ci_local.sh --fast   # skip cargo clippy/test (hygiene + fmt + links + typos only)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FAST_ONLY=0
if [ "${1:-}" = "--fast" ]; then
  FAST_ONLY=1
elif [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,10p' "$0"
  exit 0
fi

echo "==> [1/8] Privacy & PII leak check"
python3 -B scripts/verify_no_leaks.py --root .

echo "==> [2/8] Offline relative Markdown links"
python3 -B scripts/check_markdown_links.py --root .

echo "==> [3/8] .gitignore vs tracked files guard"
IGNORED_TRACKED="$(git ls-files -i -c --exclude-standard)"
if [ -n "$IGNORED_TRACKED" ]; then
  echo "ERROR: tracked files match .gitignore:" >&2
  echo "$IGNORED_TRACKED" >&2
  exit 1
fi

echo "==> [4/8] Shell & Python syntax checks"
bash -n scripts/*.sh bench/qwen/*.sh
python3 -m py_compile scripts/*.py

echo "==> [5/8] Optional local tools (typos, gitleaks, actionlint)"
if command -v typos >/dev/null 2>&1; then
  typos --config _typos.toml
else
  echo "  (skip: typos not installed)"
fi
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --config .github/gitleaks.toml --source . --no-git --redact --exit-code 1
else
  echo "  (skip: gitleaks not installed)"
fi
if command -v actionlint >/dev/null 2>&1; then
  actionlint
else
  echo "  (skip: actionlint not installed)"
fi

echo "==> [6/8] Rust formatting (cargo fmt --all -- --check)"
cargo fmt --all -- --check

if [ "$FAST_ONLY" -eq 1 ]; then
  echo "[PASS] ci_local.sh (--fast) complete."
  exit 0
fi

echo "==> [7/8] No-GPU Rust clippy (ATLAS_SKIP_BUILD=1)"
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests -- -D warnings

echo "==> [8/8] Offline Python tests (tests/ + repro/tests/)"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest -q tests/ repro/tests

echo "[PASS] ci_local.sh complete."
