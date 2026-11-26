# Milestone 0 Qualification Receipt: Worktree & Integration Foundation

**Date**: 2026-10-24T11:05:35Z  
**Branch**: `release`  
**Base Commit**: `295a6aaa1d4d1766ccacf5c8e5c6fd9916cfe371`  
**HEAD Commit**: `90ce7ffa261c816f9c2ba9faeb36672b136d07ed`  
**Binary Location**: `target/release/flux-server`  
**Binary Size**: 92,208,544 bytes (88 MB)  
**Binary SHA256**: `3ad909084080a530195cf920388025e64a824f428a13fbe3ba3e0f4163fd043d`  

## Verified Items

1. **Worktree Creation**: Created `.` from `295a6aaa` with dedicated evidence folder `qualification/`.
2. **Repro Dockerfile Fix**: Cherry-picked commit `dfb43612` (`c53c1091`) replacing `libcudart13` with `cuda-cudart-13-0` in `repro-kit/Dockerfile:80`.
3. **Answer Prefix Whitespace**: Confirmed `crates/spark-server/src/api/chat_blocking.rs:344` preserves raw answer prefix whitespace without `trim_start()`, preventing `trim_answer_prefix` contamination in the intervention ledger.
4. **Engine Compiler Fixes**:
   - `crates/spark-model/src/model/trait_impl/async_chkpt.rs`: Replaced `pub(super)` with `pub(in crate::model)` for `commit_qwen4_qsa_k5_only_dispatch`, `abort_verify_state_dispatch`, `commit_verify_state_async_dispatch`.
   - `crates/spark-model/src/model/trait_impl/sequence.rs`: Replaced `pub(super)` with `pub(in crate::model)` for `free_sequence_dispatch`.
   - `crates/spark-model/src/model/trait_impl/speculative.rs`: Replaced `pub(super)` with `pub(in crate::model)` for `trim_proposer_state_dispatch`.
   - `crates/spark-model/src/model/verify_batch_frame.rs`: Fixed `PrefixCache` trait object handling (`!self.prefix_cache.is_active()` and `self.prefix_cache.as_ref()`).
   - `crates/spark-model/src/layers/qwen4_mtp_rows.rs`: Added `#[allow(dead_code)]` to staged frame statics and functions.
   - `crates/spark-model/src/traits/verify_batch_owner.rs`: Added `#[allow(dead_code)]` to `ranges`.
5. **Prefill Budget Wide Override Fix**:
   - Implemented `resolve_prefill_budget_with_explicit_overrides` in `crates/spark-server/src/main_modules/serve_phases/kv_cache.rs` to honor `ATLAS_QWEN4_PREFILL_WIDE` (4096, 8192, 16384, 32768) when `--max-prefill-tokens` is omitted, bypassing the 1024 auto-clamp so that fast wide-prefill runs natively.
   - Added unit test `prefill_budget_honors_explicit_wide_when_max_prefill_tokens_omitted`.
6. **Host Binary Build & Symbols**:
   - Built under `flock -w 7200 $FLUX_BUILD_LOCK_FILE` with `CARGO_BUILD_JOBS=2` and `CARGO_TARGET_DIR=.-target`.
   - Compiled 173 kernels for target 0 (gb10, qwen3.8-27b, nvfp4) and 203 kernels for target 1 (gb10, qwen3.8-flash-next, nvfp4).
   - Verified symbols: `flux_server::scheduler::verify_qwen4_batched_step::step_verify_qwen4_batched`, `flux_server::main_modules::serve_phases::kv_cache::resolve_prefill_budget_with_explicit_overrides`.
   - Tested binary `--help` successfully exiting with code 0.
