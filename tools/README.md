<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Developer &amp; Profiling Tools (<code>tools/</code>)</b></p>

# Developer Tools (`tools/`)

Standalone utilities for fast local builds, recipe evaluation, and hardware profiling.

| Path | Purpose |
|---|---|
| [`fast_cargo.sh`](fast_cargo.sh) | Bash wrapper for local Cargo builds with optional `sccache` auto-detection and isolated per-worktree `CARGO_TARGET_DIR` seeding. |
| [`recipe_bench/`](recipe_bench/) | Python + Bash harness (`generate.py`, `run.sh`) for launching built-in YAML model recipes and recording structured prefill/decode telemetry. |
