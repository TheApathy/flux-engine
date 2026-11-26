#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
exec python3 -B "$(dirname -- "${BASH_SOURCE[0]}")/check_ignored_sources.py" "$@"
