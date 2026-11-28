// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler routing policy for `ATLAS_DETERMINISTIC_BATCH` (`ZERO_OR_ONE`, default off).
//!
//! When `ATLAS_DETERMINISTIC_BATCH=1`, every multi-sequence cohort (`B >= 2`)
//! executes the exact same canonical `B=1` single-sequence prefill, decode,
//! speculative verify, and DFlash2/MTP propose path sequentially per slot with
//! a secondary-stream barrier between slots. That guarantees bit-exact
//! (`0-ULP`) equivalence between `B=1` and `B=2/4/8/16` across Qwen3.8-Flash-Next
//! target-only, native MTP, and DFlash2 while leaving default (`Disabled`)
//! scheduling 100% untouched.

use spark_model::layers::deterministic_batch::{
    DeterministicBatchEnvError, DeterministicBatchPolicy,
};

/// Read and validate the process-wide `ATLAS_DETERMINISTIC_BATCH` policy.
///
/// In production (`cfg(not(test))`), delegates to the process-cached
/// `spark_model::layers::deterministic_batch::deterministic_batch_policy()`.
/// Under `spark-server` unit tests (`cfg(test)`), reads the env var live via
/// `parse_deterministic_batch_os` so test cases can toggle `"0"` / `"1"` /
/// invalid values without poisoning `spark-model`'s non-test `OnceLock`.
///
/// # Errors
/// Returns [`DeterministicBatchEnvError`] on malformed or non-UTF-8 values so
/// scheduler entry points fail closed instead of silently running batched paths.
#[inline]
pub(super) fn scheduler_deterministic_batch_policy()
-> std::result::Result<DeterministicBatchPolicy, DeterministicBatchEnvError> {
    #[cfg(not(test))]
    {
        spark_model::layers::deterministic_batch::deterministic_batch_policy()
    }
    #[cfg(test)]
    {
        if let Some(override_res) = test_support::thread_override() {
            return override_res;
        }
        let raw =
            std::env::var_os(spark_model::layers::deterministic_batch::DETERMINISTIC_BATCH_ENV);
        spark_model::layers::deterministic_batch::parse_deterministic_batch_os(raw.as_deref())
    }
}

/// Whether Qwen4 multi-prompt short-prefill bursts (`run_qwen4_short_prefill_bursts`)
/// may run under `det_batch`.
#[inline]
#[must_use]
pub(super) const fn prefill_short_batch_allowed(
    det_batch: DeterministicBatchPolicy,
    env_enabled: bool,
) -> bool {
    !det_batch.is_enabled() && env_enabled
}

/// Whether Q12 multi-stream batched prefill (`can_batch_prefill_only` /
/// `can_batch_mixed`) may run under `det_batch`.
#[inline]
#[must_use]
pub(super) const fn prefill_q12_batch_allowed(
    det_batch: DeterministicBatchPolicy,
    q12_enabled: bool,
) -> bool {
    !det_batch.is_enabled() && q12_enabled
}

/// Whether single-stream `mixed_forward` (prefill + decode fusion) may run
/// under `det_batch`.
#[inline]
#[must_use]
pub(super) const fn prefill_mixed_forward_allowed(
    det_batch: DeterministicBatchPolicy,
    can_mix: bool,
) -> bool {
    !det_batch.is_enabled() && can_mix
}

/// Whether prefill chunk-size planners (`slo_interleave`, `legacy_interleave`,
/// `prefill_chunk_policy`) observe active decode rows under `det_batch`.
///
/// Returning `false` under `DeterministicBatchPolicy::Enabled` pins every
/// prompt's chunk boundaries to the exact `B=1` idle-box chunking regardless
/// of staggered concurrent arrivals.
#[inline]
#[must_use]
pub(super) const fn prefill_chunk_sees_active_decode(
    det_batch: DeterministicBatchPolicy,
    decode_rows_active: bool,
) -> bool {
    !det_batch.is_enabled() && decode_rows_active
}

/// Whether a decode/speculative cohort of `active_len` sequences must execute
/// sequentially per slot through the canonical `B=1` single-sequence path.
#[inline]
#[must_use]
pub(super) const fn decode_runs_canonical_per_slot(
    det_batch: DeterministicBatchPolicy,
    active_len: usize,
) -> bool {
    det_batch.is_enabled() && active_len >= 2
}

/// Whether Phase A batched native `K=1` bootstrap (`try_batched_native_k1_bootstrap`)
/// may batch `>= 2` sequences under `det_batch`.
#[inline]
#[must_use]
pub(super) const fn mtp_batched_bootstrap_allowed(
    det_batch: DeterministicBatchPolicy,
    eligible: bool,
) -> bool {
    !det_batch.is_enabled() && eligible
}

/// Whether Phase B cross-sequence batched verify or cohort-width coupling
/// (`step_verify_qwen4_batched`, `step_verify_k2_batched_csk`,
/// `step_verify_k3_batched_csk`, `step_verify_dflash_batched`,
/// `narrow_pending_drafts`, `harmonize_cohort_widths`) may run under `det_batch`.
#[inline]
#[must_use]
pub(super) const fn mtp_cross_seq_verify_allowed(
    det_batch: DeterministicBatchPolicy,
    eligible: bool,
) -> bool {
    !det_batch.is_enabled() && eligible
}

/// Effective cohort size observed by MTP/DFlash draft-depth & loop guards.
///
/// Under `DeterministicBatchPolicy::Enabled`, every live sequence sees a
/// singleton cohort (`live_seqs.min(1)`) so draft width `K` and `p_cum`
/// thresholds never vary with concurrency.
#[inline]
#[must_use]
pub(super) const fn mtp_effective_live_seqs(
    det_batch: DeterministicBatchPolicy,
    live_seqs: usize,
) -> usize {
    if det_batch.is_enabled() {
        if live_seqs > 0 { 1 } else { 0 }
    } else {
        live_seqs
    }
}

/// Whether cross-request history-ring grafting is permitted under `det_batch`.
///
/// Disabled under `DeterministicBatchPolicy::Enabled` so `B=1` and `B>=2`
/// never couple draft proposals across concurrent or prior requests even if
/// `HISTORY_DRAFTS=1` is set.
#[inline]
#[must_use]
pub(super) const fn history_graft_allowed_with_policy(
    det_batch: DeterministicBatchPolicy,
    base_allowed: bool,
) -> bool {
    !det_batch.is_enabled() && base_allowed
}

#[cfg(test)]
pub(super) mod test_support {
    use std::cell::RefCell;

    use spark_model::layers::deterministic_batch::{
        DeterministicBatchEnvError, DeterministicBatchPolicy, parse_deterministic_batch_str,
    };

    type PolicyOverride =
        Option<std::result::Result<DeterministicBatchPolicy, DeterministicBatchEnvError>>;

    thread_local! {
        static TEST_POLICY_OVERRIDE: RefCell<PolicyOverride> = const { RefCell::new(None) };
    }

    pub(super) fn thread_override() -> PolicyOverride {
        TEST_POLICY_OVERRIDE.with(|cell| cell.borrow().clone())
    }

    /// Thread-local guard overriding `ATLAS_DETERMINISTIC_BATCH` for the
    /// current test thread without mutating process-wide environment state.
    pub(in crate::scheduler) struct ScopedDetBatchEnv {
        prev: PolicyOverride,
    }

    impl ScopedDetBatchEnv {
        pub(in crate::scheduler) fn set(value: Option<&str>) -> Self {
            let next = Some(parse_deterministic_batch_str(value));
            let prev = TEST_POLICY_OVERRIDE.with(|cell| cell.replace(next));
            Self { prev }
        }
    }

    impl Drop for ScopedDetBatchEnv {
        fn drop(&mut self) {
            TEST_POLICY_OVERRIDE.with(|cell| {
                cell.replace(self.prev.take());
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_policy_preserves_all_baseline_scheduler_routes() {
        let p = DeterministicBatchPolicy::Disabled;
        assert!(prefill_short_batch_allowed(p, true));
        assert!(!prefill_short_batch_allowed(p, false));
        assert!(prefill_q12_batch_allowed(p, true));
        assert!(!prefill_q12_batch_allowed(p, false));
        assert!(prefill_mixed_forward_allowed(p, true));
        assert!(!prefill_mixed_forward_allowed(p, false));
        assert!(prefill_chunk_sees_active_decode(p, true));
        assert!(!prefill_chunk_sees_active_decode(p, false));
        assert!(!decode_runs_canonical_per_slot(p, 1));
        assert!(!decode_runs_canonical_per_slot(p, 2));
        assert!(!decode_runs_canonical_per_slot(p, 8));
        assert!(!decode_runs_canonical_per_slot(p, 16));
        assert!(mtp_batched_bootstrap_allowed(p, true));
        assert!(mtp_cross_seq_verify_allowed(p, true));
        assert_eq!(mtp_effective_live_seqs(p, 0), 0);
        assert_eq!(mtp_effective_live_seqs(p, 1), 1);
        assert_eq!(mtp_effective_live_seqs(p, 8), 8);
        assert!(history_graft_allowed_with_policy(p, true));
    }

    #[test]
    fn enabled_policy_forces_canonical_b1_across_prefill_decode_and_spec_verify() {
        let p = DeterministicBatchPolicy::Enabled;
        assert!(!prefill_short_batch_allowed(p, true));
        assert!(!prefill_q12_batch_allowed(p, true));
        assert!(!prefill_mixed_forward_allowed(p, true));
        assert!(!prefill_chunk_sees_active_decode(p, true));
        assert!(!decode_runs_canonical_per_slot(p, 0));
        assert!(!decode_runs_canonical_per_slot(p, 1));
        for b in [2, 4, 8, 16, 32] {
            assert!(
                decode_runs_canonical_per_slot(p, b),
                "B={b} must run canonical per-slot B=1"
            );
            assert_eq!(
                mtp_effective_live_seqs(p, b),
                1,
                "B={b} must observe canonical singleton cohort size"
            );
        }
        assert_eq!(mtp_effective_live_seqs(p, 0), 0);
        assert_eq!(mtp_effective_live_seqs(p, 1), 1);
        assert!(!mtp_batched_bootstrap_allowed(p, true));
        assert!(!mtp_cross_seq_verify_allowed(p, true));
        assert!(!history_graft_allowed_with_policy(p, true));
    }

    #[test]
    fn decode_only_and_mtp_execute_canonical_b1_per_slot_when_enabled_and_fail_closed_on_invalid_env()
     {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use anyhow::Result;
        use spark_model::layers::deterministic_batch::DETERMINISTIC_BATCH_ENV;
        use spark_model::traits::{Model, SequenceState};
        use spark_runtime::gpu::DevicePtr;

        use super::test_support::ScopedDetBatchEnv;
        use crate::scheduler::test_fixtures::active_seq;
        use crate::scheduler::{ResponseSink, ThinkSpecCtx, step_decode_only, step_mtp};

        struct BatchAuditModel {
            decode_batch_widths: Mutex<Vec<usize>>,
            single_decode_calls: AtomicUsize,
            batched_verify_preflights: AtomicUsize,
            batched_verify_calls: AtomicUsize,
            sync_secondary_calls: AtomicUsize,
            frees: AtomicUsize,
        }

        impl BatchAuditModel {
            fn new() -> Self {
                Self {
                    decode_batch_widths: Mutex::new(Vec::new()),
                    single_decode_calls: AtomicUsize::new(0),
                    batched_verify_preflights: AtomicUsize::new(0),
                    batched_verify_calls: AtomicUsize::new(0),
                    sync_secondary_calls: AtomicUsize::new(0),
                    frees: AtomicUsize::new(0),
                }
            }
        }

        impl Model for BatchAuditModel {
            fn vocab_size(&self) -> usize {
                64
            }

            fn qwen4_batched_k5_armed(&self) -> bool {
                true
            }

            fn proposer_is_dflash(&self) -> bool {
                true
            }

            fn decode(
                &self,
                _token: u32,
                seq: &mut SequenceState,
                _stream: u64,
            ) -> Result<DevicePtr> {
                self.single_decode_calls.fetch_add(1, Ordering::SeqCst);
                seq.seq_len += 1;
                Ok(DevicePtr(0x1000))
            }

            fn decode_batch(
                &self,
                tokens: &[u32],
                seqs: &mut [&mut SequenceState],
                _stream: u64,
            ) -> Result<DevicePtr> {
                self.decode_batch_widths.lock().unwrap().push(tokens.len());
                for seq in seqs.iter_mut() {
                    seq.seq_len += 1;
                }
                Ok(DevicePtr(0x1000))
            }

            fn argmax_on_device(&self, _logits: DevicePtr, _stream: u64) -> Result<u32> {
                Ok(42)
            }

            fn argmax_batch(
                &self,
                _logits: DevicePtr,
                batch_size: usize,
                _stream: u64,
            ) -> Result<Vec<u32>> {
                Ok(vec![42; batch_size])
            }

            fn sync_secondary(&self) -> Result<()> {
                self.sync_secondary_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }

            fn preflight_verify_qwen4_batched(
                &self,
                _tokens_per_seq: &[Vec<u32>],
                _seqs: &[&SequenceState],
            ) -> Result<()> {
                self.batched_verify_preflights
                    .fetch_add(1, Ordering::SeqCst);
                Ok(())
            }

            fn decode_verify_qwen4_batched(
                &self,
                _tokens_per_seq: &[Vec<u32>],
                _seqs: &mut [&mut SequenceState],
            ) -> Result<Vec<Vec<u32>>> {
                self.batched_verify_calls.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::new())
            }

            fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
                self.frees.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }

            fn prefill(
                &self,
                _tokens: &[u32],
                _seq: &mut SequenceState,
                _stream: u64,
            ) -> Result<DevicePtr> {
                unreachable!("prefill not used")
            }

            fn prefill_chunk(
                &self,
                _tokens: &[u32],
                _seq: &mut SequenceState,
                _chunk_start: usize,
                _chunk_len: usize,
                _is_last: bool,
                _stream: u64,
            ) -> Result<DevicePtr> {
                unreachable!("prefill_chunk not used")
            }

            fn alloc_sequence(&self) -> Result<SequenceState> {
                unreachable!("alloc_sequence not used")
            }

            fn logits_buffer_ptr(&self) -> DevicePtr {
                DevicePtr(0x1000)
            }

            fn hidden_after_norm(&self) -> DevicePtr {
                DevicePtr::NULL
            }

            fn decode_verify(
                &self,
                _tokens: &[u32],
                _seq: &mut SequenceState,
                _stream: u64,
            ) -> Result<Vec<u32>> {
                unreachable!("decode_verify not used")
            }

            fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
                Ok(())
            }

            fn rollback_ssm_states(&self, _seq: &mut SequenceState, _pos: usize) -> Result<()> {
                Ok(())
            }

            fn generate_speculative(
                &self,
                _prompt: &[u32],
                _params: &spark_runtime::sampler::SamplingParams,
                _num_drafts: usize,
            ) -> Result<spark_model::engine::GenerateResult> {
                unreachable!("generate_speculative not used")
            }

            fn has_proposer(&self) -> bool {
                true
            }

            fn has_self_speculative(&self) -> bool {
                false
            }

            fn decode_draft(
                &self,
                _token: u32,
                _seq: &mut SequenceState,
                _stream: u64,
            ) -> Result<DevicePtr> {
                unreachable!("decode_draft not used")
            }

            fn cache_sequence(&self, _seq: &SequenceState) {}

            fn compact_sequence(&self, _seq: &mut SequenceState, _keep: usize) -> Result<bool> {
                Ok(true)
            }

            fn decode_verify_graphed(
                &self,
                _tokens: &[u32; 2],
                _seq: &mut SequenceState,
                _stream: u64,
            ) -> Result<[u32; 2]> {
                unreachable!("decode_verify_graphed not used")
            }

            fn decode_verify_graphed_k3(
                &self,
                _tokens: &[u32; 3],
                _seq: &mut SequenceState,
                _stream: u64,
            ) -> Result<[u32; 3]> {
                unreachable!("decode_verify_graphed_k3 not used")
            }

            fn decode_verify_graphed_k4(
                &self,
                _tokens: &[u32; 4],
                _seq: &mut SequenceState,
                _stream: u64,
            ) -> Result<[u32; 4]> {
                unreachable!("decode_verify_graphed_k4 not used")
            }

            fn save_hidden_for_mtp(&self, _row: usize, _stream: u64) -> Result<()> {
                Ok(())
            }

            fn run_mtp_propose(
                &self,
                _tok: u32,
                _pos: usize,
                _seq: &mut SequenceState,
                _stream: u64,
            ) -> Result<Option<u32>> {
                Ok(None)
            }

            fn run_mtp_propose_multi(
                &self,
                _tok: u32,
                _pos: usize,
                _count: usize,
                _seq: &mut SequenceState,
                _stream: u64,
                _mask: Option<&[i32]>,
            ) -> Result<Vec<u32>> {
                Ok(vec![11, 12, 13, 14, 15, 16, 17])
            }

            fn trim_proposer_state(
                &self,
                _seq: &mut SequenceState,
                _accepted: usize,
                _stream: u64,
            ) -> Result<()> {
                Ok(())
            }

            fn copy_logits_to_host(&self, _logits: DevicePtr, buf: &mut [u8]) -> Result<()> {
                buf.fill(0);
                let row_bytes = self.vocab_size() * 2;
                let win_bits = (10.0f32.to_bits() >> 16) as u16;
                for row in buf.chunks_exact_mut(row_bytes) {
                    row[42 * 2..42 * 2 + 2].copy_from_slice(&win_bits.to_le_bytes());
                }
                Ok(())
            }

            fn num_free_blocks(&self) -> usize {
                64
            }

            fn num_total_blocks(&self) -> usize {
                64
            }

            fn create_stream(&self) -> Result<u64> {
                Ok(0)
            }

            fn create_event(&self) -> Result<u64> {
                Ok(0)
            }

            fn record_event(&self, _event: u64, _stream: u64) -> Result<()> {
                Ok(())
            }

            fn stream_wait_event(&self, _stream: u64, _event: u64) -> Result<()> {
                Ok(())
            }

            fn synchronize_stream(&self, _stream: u64) -> Result<()> {
                Ok(())
            }

            fn default_stream(&self) -> u64 {
                0
            }

            fn bind_gpu_to_thread(&self) -> Result<()> {
                Ok(())
            }
        }

        // 1. Baseline (Disabled): B=4 runs a single batched decode_batch(4).
        {
            let _det = ScopedDetBatchEnv::set(None);
            let m = BatchAuditModel::new();
            let mut active = Vec::new();
            let mut _rxs = Vec::new();
            for slot in 0..4 {
                let (tx, rx) = tokio::sync::oneshot::channel();
                active.push(active_seq(ResponseSink::Blocking(Some(tx)), slot));
                _rxs.push(rx);
            }
            let mut cache = vec![true; 4];
            step_decode_only(
                &m,
                &mut active,
                &mut cache,
                None,
                None,
                None,
                None,
                None,
                &[],
                false,
            );
            assert_eq!(*m.decode_batch_widths.lock().unwrap(), vec![4]);
            assert_eq!(active.len(), 4);
            assert_eq!(cache.len(), 4);
            for a in &active {
                assert_eq!(a.last_token, 42);
            }
        }

        // 2. Enabled: B=4 runs 4 canonical B=1 decode_batch(1) calls with sync_secondary barriers.
        {
            let _det = ScopedDetBatchEnv::set(Some("1"));
            let m = BatchAuditModel::new();
            let mut active = Vec::new();
            let mut _rxs = Vec::new();
            for slot in 0..4 {
                let (tx, rx) = tokio::sync::oneshot::channel();
                active.push(active_seq(ResponseSink::Blocking(Some(tx)), slot));
                _rxs.push(rx);
            }
            let mut cache = vec![true; 4];
            step_decode_only(
                &m,
                &mut active,
                &mut cache,
                None,
                None,
                None,
                None,
                None,
                &[],
                false,
            );
            assert_eq!(
                *m.decode_batch_widths.lock().unwrap(),
                vec![1, 1, 1, 1],
                "ATLAS_DETERMINISTIC_BATCH=1 must execute canonical B=1 decode per slot"
            );
            assert_eq!(m.sync_secondary_calls.load(Ordering::SeqCst), 4);
            assert_eq!(active.len(), 4);
            assert_eq!(cache.len(), 4);
            for a in &active {
                assert_eq!(a.last_token, 42);
            }
        }

        // 3. Enabled on step_mtp: B=4 bootstrap uses canonical B=1 decode (never batched K=1
        //    bootstrap or batched Qwen4 verify preflight/exec).
        {
            let _det = ScopedDetBatchEnv::set(Some("1"));
            let m = BatchAuditModel::new();
            let mut active = Vec::new();
            let mut _rxs = Vec::new();
            for slot in 0..4 {
                let (tx, rx) = tokio::sync::oneshot::channel();
                active.push(active_seq(ResponseSink::Blocking(Some(tx)), slot));
                _rxs.push(rx);
            }
            let mut cache = vec![true; 4];
            let think = ThinkSpecCtx {
                enabled: false,
                code_fence_token: None,
                reflection_suppress_ids: &[],
                adaptive_sampling: false,
                honor_eos_inside_thinking: Some(false),
            };
            step_mtp(&m, &mut active, 7, &think, &mut cache);
            assert_eq!(
                m.single_decode_calls.load(Ordering::SeqCst),
                4,
                "every slot must bootstrap via canonical single-seq Model::decode"
            );
            assert!(
                m.decode_batch_widths.lock().unwrap().is_empty(),
                "batched native K=1 bootstrap must not fire under ATLAS_DETERMINISTIC_BATCH=1"
            );
            assert_eq!(m.batched_verify_preflights.load(Ordering::SeqCst), 0);
            assert_eq!(m.batched_verify_calls.load(Ordering::SeqCst), 0);
            assert!(m.sync_secondary_calls.load(Ordering::SeqCst) >= 4);

            // Direct defense-in-depth check on step_verify_qwen4_batched.
            for a in &mut active {
                a.pending_drafts = vec![1, 2, 3, 4, 5, 6, 7];
            }
            let frames = vec![vec![0, 1, 2, 3]];
            assert!(
                !crate::scheduler::verify_qwen4_batched_step::step_verify_qwen4_batched(
                    &m,
                    &mut active,
                    &frames,
                    7,
                    &think,
                ),
                "step_verify_qwen4_batched must refuse cohort under ATLAS_DETERMINISTIC_BATCH=1"
            );
            assert_eq!(m.batched_verify_preflights.load(Ordering::SeqCst), 0);
        }

        // 4. Invalid env: step_decode_only and step_mtp both fail closed cleanly.
        {
            let _det = ScopedDetBatchEnv::set(Some("true"));
            let m = BatchAuditModel::new();
            let (tx0, mut rx0) = tokio::sync::oneshot::channel();
            let (tx1, mut rx1) = tokio::sync::oneshot::channel();
            let mut active = vec![
                active_seq(ResponseSink::Blocking(Some(tx0)), 0),
                active_seq(ResponseSink::Blocking(Some(tx1)), 1),
            ];
            let mut cache = vec![true; 2];
            step_decode_only(
                &m,
                &mut active,
                &mut cache,
                None,
                None,
                None,
                None,
                None,
                &[],
                false,
            );
            assert!(active.is_empty());
            assert!(cache.is_empty());
            assert!(m.decode_batch_widths.lock().unwrap().is_empty());
            for rx in [&mut rx0, &mut rx1] {
                match rx.try_recv() {
                    Ok(Err(err)) => {
                        assert!(err.to_string().contains(DETERMINISTIC_BATCH_ENV), "{err:#}")
                    }
                    Ok(Ok(_)) => panic!("expected fail-closed env error, got Ok response"),
                    Err(_) => panic!("expected fail-closed env error, channel closed"),
                }
            }

            let (tx2, _rx2) = tokio::sync::oneshot::channel();
            let mut mtp_active = vec![active_seq(ResponseSink::Blocking(Some(tx2)), 0)];
            let mut mtp_cache = vec![true];
            let think = ThinkSpecCtx {
                enabled: false,
                code_fence_token: None,
                reflection_suppress_ids: &[],
                adaptive_sampling: false,
                honor_eos_inside_thinking: Some(false),
            };
            step_mtp(&m, &mut mtp_active, 7, &think, &mut mtp_cache);
            assert!(mtp_active[0].finished);
            assert!(!mtp_cache[0]);
            assert_eq!(m.single_decode_calls.load(Ordering::SeqCst), 0);
        }
    }
}
