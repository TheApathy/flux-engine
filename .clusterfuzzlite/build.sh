#!/bin/bash -eu
# SPDX-License-Identifier: AGPL-3.0-only
# Build ClusterFuzzLite libFuzzer targets for Flux Engine's pure-Rust parsers.

cd "$SRC/flux-engine/fuzz"
cargo +nightly fuzz build --release --debug-assertions
cp target/x86_64-unknown-linux-gnu/release/xgrammar_schema "$OUT/"
