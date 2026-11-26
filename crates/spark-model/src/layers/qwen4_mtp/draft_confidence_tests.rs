// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};

use super::*;
use crate::layers::qwen4_mtp_gate::{self, GatePolicy};
use crate::traits::{ImageSpan, RotaryPositions};

#[test]
fn stops_before_first_low_confidence_draft() {
    assert_eq!(confident_prefix(&[0.9, 0.8, 0.5, 0.95], 0.6), 2);
    assert_eq!(confident_prefix(&[0.9, 0.8, 0.7, 0.65], 0.6), 4);
}

#[test]
fn always_keeps_one_draft() {
    assert_eq!(confident_prefix(&[0.1, 0.9], 0.6), 1);
    assert_eq!(confident_prefix(&[0.59], 0.6), 1);
}

#[test]
fn threshold_is_inclusive() {
    assert_eq!(confident_prefix(&[0.6, 0.6], 0.6), 2);
}

#[test]
fn cohort_uniform_batch_drafting_never_truncates_to_zero_under_gumbel() {
    // Batched (C >= 2) proposals always keep at least 1 draft even on hopeless
    // step-0 probabilities (< 0.12) and keep all drafts when base threshold is 0.
    assert_eq!(confident_prefix_batch(&[0.01, 0.9, 0.9, 0.9], 0.0), 4);
    assert_eq!(confident_prefix_batch(&[0.01, 0.9, 0.9, 0.9], 0.6), 1);
    assert_eq!(confident_prefix_batch(&[0.05], 0.85), 1);

    let conf_off = GatePolicy {
        threshold: None,
        max_drafts: None,
        early_stop: false,
        log: false,
    };
    assert!(!conf_off.wants_probabilities_batch());
    assert_eq!(conf_off.effective_base_threshold_batch(), None);
    assert_eq!(conf_off.early_threshold_batch(), None);
}

fn image_positions() -> RotaryPositions {
    RotaryPositions::from_image_spans(
        13,
        64,
        &[
            ImageSpan {
                start: 1,
                height: 2,
                width: 2,
            },
            ImageSpan {
                start: 6,
                height: 2,
                width: 3,
            },
        ],
    )
    .unwrap()
}

#[test]
fn image_rotary_admission_is_fail_closed_before_metadata() {
    let positions = image_positions();
    assert_eq!(
        rotary_axes(&positions, 2, true, true, 0.0).unwrap(),
        [1, 1, 2]
    );
    assert!(rotary_axes(&positions, 2, false, true, 0.0).is_err());
    assert!(rotary_axes(&positions, 2, true, false, 0.0).is_err());
    assert!(rotary_axes(&positions, 2, true, true, 4.0).is_err());
    assert!(rotary_axes(&positions, 64, true, true, 0.0).is_err());
    assert!(validate_rotary_span(&positions, 62, 2, true, true, 0.0).is_ok());
    assert!(validate_rotary_span(&positions, 63, 2, true, true, 0.0).is_err());
    assert!(
        validate_rotary_span(&positions, 0, MAX_ROTARY_METADATA_ROWS + 1, true, true, 0.0).is_err()
    );
    assert!(validate_rotary_span(&positions, usize::MAX, 2, true, true, 0.0).is_err());
}

#[test]
fn metadata_preserves_physical_cache_fields_and_maps_all_rotary_axes() {
    let positions = image_positions();
    let mut packed = vec![0xa5; 32];
    write_rotary_metadata(&positions, 2, &mut packed).unwrap();
    assert_eq!(&packed[0..4], &1u32.to_le_bytes());
    assert_eq!(
        &packed[POSITION_H_OFFSET..POSITION_H_OFFSET + 4],
        &1u32.to_le_bytes()
    );
    assert_eq!(
        &packed[POSITION_W_OFFSET..POSITION_W_OFFSET + 4],
        &2u32.to_le_bytes()
    );
    assert!(
        packed[4..POSITION_H_OFFSET]
            .iter()
            .all(|byte| *byte == 0xa5)
    );

    let mut identity = vec![0xa5; 32];
    write_rotary_metadata(&RotaryPositions::identity(), 9, &mut identity).unwrap();
    assert_eq!(&identity[0..4], &9u32.to_le_bytes());
    assert!(
        identity[POSITION_H_OFFSET..]
            .iter()
            .all(|byte| *byte == 0xa5)
    );
    assert!(write_rotary_metadata(&positions, 2, &mut [0u8; 27]).is_err());
}

#[test]
fn image_metadata_uses_distinct_axes_but_text_keeps_legacy_aliases() {
    let base = DevicePtr(4096);
    let image = image_positions();
    assert_eq!(
        rotary_metadata_ptrs(base, &image),
        (
            base.offset(POSITION_H_OFFSET),
            base.offset(POSITION_W_OFFSET)
        )
    );
    assert_eq!(
        rotary_metadata_ptrs(base, &RotaryPositions::identity()),
        (base, base)
    );
}

#[test]
fn prompt_tail_maps_predict_into_positions_without_wrapping() {
    let positions = image_positions();
    let start = prompt_tail_start(12, 4).unwrap();
    assert_eq!(start, 10);
    assert_eq!(
        positions.range(start, 4).unwrap(),
        [[4, 5, 5], [4, 5, 6], [7; 3], [8; 3]]
    );
    assert!(prompt_tail_start(2, 4).is_err());
    assert!(prompt_tail_start(usize::MAX, 1).is_err());
}

#[test]
fn propose_batch_metadata_and_scratch_spans_are_disjoint_up_to_c64() {
    for b in 2..=64usize {
        let (pos_h_off, pos_w_off, last_tok_off, slot_off, seq_len_off, hidden_ptrs_off, bt_off) =
            if b <= 8 {
                (64, 128, 192, 256, 512, 576, 768)
            } else {
                (256, 512, 768, 1024, 1536, 2048, 2560)
            };
        assert_eq!(slot_off % 8, 0, "b={b}");
        assert_eq!(hidden_ptrs_off % 8, 0, "b={b}");
        let bt_row_bytes = 64 * 4;
        let stride = (bt_off + b * bt_row_bytes).next_multiple_of(256);
        let spans = [
            (0usize, b * 4),
            (pos_h_off, pos_h_off + b * 4),
            (pos_w_off, pos_w_off + b * 4),
            (last_tok_off, last_tok_off + b * 4),
            (slot_off, slot_off + b * 8),
            (seq_len_off, seq_len_off + b * 4),
            (hidden_ptrs_off, hidden_ptrs_off + b * 8),
            (bt_off, bt_off + b * bt_row_bytes),
        ];
        for window in spans.windows(2) {
            assert!(window[0].1 <= window[1].0, "b={b}: {window:?}");
        }
        assert!(spans.last().unwrap().1 <= stride, "b={b}");

        let (prob_offset, prob_span, token_offset) = if b <= 8 {
            (CHAIN_PROB_OFFSET, CHAIN_PROB_SPAN, CHAIN_TOKEN_OFFSET)
        } else {
            (CHAIN_ARGMAX_OFFSET + 2048, 2048, CHAIN_ARGMAX_OFFSET + 4096)
        };
        for num_drafts in 1..=7usize {
            let bytes = num_drafts * b * 4;
            assert!(CHAIN_ARGMAX_OFFSET + bytes <= prob_offset, "b={b}");
            assert!(bytes <= prob_span, "b={b}");
            assert!(prob_offset + bytes <= token_offset, "b={b}");
            assert!(token_offset + bytes <= CHAIN_META_OFFSET, "b={b}");
        }
    }
}

#[test]
fn mtp_qsa_mode_parsing_and_fail_closed_invalid_values() {
    assert_eq!(
        parse_mtp_qsa_mode(None, None).unwrap(),
        MtpQsaMode::Disabled
    );
    assert_eq!(
        parse_mtp_qsa_mode(Some("0"), None).unwrap(),
        MtpQsaMode::Disabled
    );
    assert_eq!(
        parse_mtp_qsa_mode(Some("0"), Some("1")).unwrap(),
        MtpQsaMode::Disabled
    );
    assert_eq!(
        parse_mtp_qsa_mode(Some("0"), Some("bogus")).unwrap(),
        MtpQsaMode::Disabled
    );
    assert_eq!(
        parse_mtp_qsa_mode(Some("1"), None).unwrap(),
        MtpQsaMode::DraftSelectionReuse
    );
    assert_eq!(
        parse_mtp_qsa_mode(Some(" 1 "), Some(" 1 ")).unwrap(),
        MtpQsaMode::DraftSelectionReuse
    );
    assert_eq!(
        parse_mtp_qsa_mode(Some("1"), Some("0")).unwrap(),
        MtpQsaMode::PerStep
    );
    for bad in ["", "2", "true", "yes", "on", "-1", "reuse"] {
        assert!(
            parse_mtp_qsa_mode(Some(bad), None).is_err(),
            "expected invalid ATLAS_QWEN4_MTP_QSA={bad:?} to error"
        );
        assert_eq!(
            resolve_mtp_qsa_mode(Some(bad), Some("1")),
            MtpQsaMode::Disabled,
            "invalid ATLAS_QWEN4_MTP_QSA={bad:?} must fail closed"
        );
        assert!(
            parse_mtp_qsa_mode(Some("1"), Some(bad)).is_err(),
            "expected invalid ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE={bad:?} to error"
        );
        assert_eq!(
            resolve_mtp_qsa_mode(Some("1"), Some(bad)),
            MtpQsaMode::Disabled,
            "invalid ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE={bad:?} must fail closed"
        );
    }
}

#[test]
fn mtp_qsa_reused_index_width_bounds() {
    let base = crate::layers::qwen4_qsa::QSA_OUTPUT_WIDTH;
    assert_eq!(base, 2_051);
    for step in 0..=8 {
        let width = checked_mtp_reused_index_width(step, 8).unwrap();
        assert_eq!(width, base + step);
        assert!(width <= 2_059);
    }
    assert_eq!(checked_mtp_reused_index_width(8, 8).unwrap(), 2_059);
    assert_eq!(checked_mtp_reused_index_width(16, 16).unwrap(), 2_067);
    assert!(checked_mtp_reused_index_width(9, 8).is_err());
    assert!(checked_mtp_reused_index_width(1, 17).is_err());
    assert!(checked_mtp_reused_index_width(usize::MAX, 8).is_err());
}

#[test]
fn mtp_draft_step_qsa_plan_routes_snapshot_reuse_and_dense_prefix() {
    // Disabled or no QSA side cache -> always off.
    assert_eq!(
        draft_step_qsa_plan(MtpQsaMode::Disabled, true, 4096, 0, 4),
        (false, None)
    );
    assert_eq!(
        draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, false, 4096, 0, 4),
        (false, None)
    );
    // PerStep -> always QSA required without reuse guard.
    assert_eq!(
        draft_step_qsa_plan(MtpQsaMode::PerStep, true, 4096, 2, 4),
        (true, None)
    );
    // DraftSelectionReuse at long context (>= 2048): step 0 snapshots, steps 1..K-1 reuse.
    assert_eq!(
        draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 4096, 0, 8),
        (true, Some(DraftStepReuseAction::SnapshotStep0))
    );
    for step in 1..8 {
        assert_eq!(
            draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 4096, step, 8),
            (
                true,
                Some(DraftStepReuseAction::ReuseStep {
                    base_seq_len: 4096,
                    step_offset: step,
                    max_drafts: 7,
                })
            )
        );
    }
    // Dense prefix (< 2048): step 0 updates side cache, speculative drafts skip side cache.
    assert_eq!(
        draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 512, 0, 4),
        (true, None)
    );
    assert_eq!(
        draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 512, 1, 4),
        (false, None)
    );
    // 2048-straddling boundary chain falls back to PerStep cleanly.
    assert_eq!(
        draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 2046, 2, 4),
        (true, None)
    );
}

#[test]
fn chain_confidence_gate_preserves_greedy_and_enables_gumbel_running_product() {
    let active_policy = GatePolicy {
        threshold: Some(0.85),
        max_drafts: None,
        early_stop: true,
        log: false,
    };
    // Greedy (`gumbel_active = false`) keeps early-stop syncs and truncation intact.
    assert_eq!(
        resolve_chain_confidence_gate(active_policy, true, Some(0.85), true, false),
        ChainConfidenceGate {
            want_prob: true,
            early: Some(0.85),
            apply_truncate: true,
        }
    );
    // Greedy without argmax_prob kernel disables prob/early/truncate cleanly.
    assert_eq!(
        resolve_chain_confidence_gate(active_policy, true, Some(0.85), false, false),
        ChainConfidenceGate {
            want_prob: false,
            early: None,
            apply_truncate: false,
        }
    );
    // Non-greedy Gumbel-CRN (`gumbel_active = true`) enables running-product
    // MTP_CONF_CUM with 0.0 per-step floor so single T>0 tokens survive greedy
    // per-step floors while low-cumulative chains stop cleanly.
    assert_eq!(
        resolve_chain_confidence_gate_with_cum(
            ChainConfidenceScope::SingleSequence,
            active_policy,
            true,
            Some(0.85),
            true,
            true,
            Some(qwen4_mtp_gate::DEFAULT_GUMBEL_CUM_THRESHOLD),
            false,
        ),
        ChainConfidenceGate {
            want_prob: true,
            early: Some(0.0),
            apply_truncate: true,
        }
    );
    // Explicit `ATLAS_QWEN4_MTP_CONF_CUM=0` (`gumbel_cum = None`) disables gating.
    assert_eq!(
        resolve_chain_confidence_gate_with_cum(
            ChainConfidenceScope::SingleSequence,
            active_policy,
            true,
            Some(0.85),
            true,
            true,
            None,
            false,
        ),
        ChainConfidenceGate {
            want_prob: false,
            early: None,
            apply_truncate: false,
        }
    );
    // Diagnostic logging with `gumbel_cum = None` still collects probabilities without truncating.
    let logged_policy = GatePolicy {
        log: true,
        ..active_policy
    };
    assert_eq!(
        resolve_chain_confidence_gate_with_cum(
            ChainConfidenceScope::SingleSequence,
            logged_policy,
            true,
            Some(0.85),
            true,
            true,
            None,
            false,
        ),
        ChainConfidenceGate {
            want_prob: true,
            early: None,
            apply_truncate: false,
        }
    );
    // BatchedCohort (`b >= 2`) never enables early-stop or truncation even under Gumbel.
    assert_eq!(
        resolve_chain_confidence_gate_with_cum(
            ChainConfidenceScope::BatchedCohort,
            active_policy,
            true,
            Some(0.85),
            true,
            true,
            Some(qwen4_mtp_gate::DEFAULT_GUMBEL_CUM_THRESHOLD),
            false,
        ),
        ChainConfidenceGate {
            want_prob: false,
            early: None,
            apply_truncate: false,
        }
    );
}

#[test]
fn mtp_pinned_h2d_ring_aligns_wraps_and_copies() {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut ring = MtpPinnedH2dRing::new(&gpu, 64).expect("mock pinned ring");
    let dst_a = gpu.alloc(20).unwrap();
    let dst_b = gpu.alloc(36).unwrap();
    let payload_a = [0x5Au8; 20];
    let payload_b = [0xA5u8; 36];

    assert!(
        ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&payload_a, dst_a)], 0)
            .unwrap()
    );
    assert_eq!(gpu.read_alloc(dst_a).unwrap(), payload_a);

    // Next 36B copy aligns to 24 (24 + 40 = 64 <= 64).
    assert!(
        ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&payload_b, dst_b)], 0)
            .unwrap()
    );
    assert_eq!(gpu.read_alloc(dst_b).unwrap(), payload_b);

    // Next 20B copy wraps cleanly back to offset 0.
    assert!(
        ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&payload_a, dst_a)], 0)
            .unwrap()
    );
    // Oversized payload (> 64 B) returns Ok(false) to trigger fallback.
    let oversized = [0x11u8; 72];
    assert!(
        !ring
            .copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&oversized, dst_a)], 0)
            .unwrap()
    );
    ring.destroy(&gpu);
    assert!(
        !ring
            .copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&payload_a, dst_a)], 0)
            .unwrap()
    );
}

#[test]
fn mtp_pinned_h2d_ring_fences_wrap_and_destroys_events() {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    struct EventFenceGpu {
        inner: spark_runtime::gpu::mock::MockGpuBackend,
        next_event: AtomicU64,
        ready: AtomicBool,
        destroyed: AtomicUsize,
    }

    impl GpuBackend for EventFenceGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc_managed(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.inner.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.inner.copy_d2h(src, dst)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }
        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.inner
                .launch(func, grid, block, shared_mem, stream, params)
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.inner.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            self.inner.default_stream()
        }
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            self.inner.kernel(module, func_name)
        }
        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(ptr, value, bytes)
        }
        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }
        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
        fn create_event(&self) -> Result<u64> {
            Ok(self.next_event.fetch_add(1, Ordering::Relaxed))
        }
        fn record_event(&self, _event: u64, _stream: u64) -> Result<()> {
            Ok(())
        }
        fn poll_event(&self, _event: u64) -> Result<bool> {
            Ok(self.ready.load(Ordering::Acquire))
        }
        fn destroy_event(&self, _event: u64) -> Result<()> {
            self.destroyed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    let gpu = EventFenceGpu {
        inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
        next_event: AtomicU64::new(1),
        ready: AtomicBool::new(false),
        destroyed: AtomicUsize::new(0),
    };
    let mut ring = MtpPinnedH2dRing::new(&gpu, 64).expect("event-fenced ring");
    let dst = gpu.alloc(32).unwrap();
    let half0 = [0x11u8; 24];
    let half1 = [0x22u8; 32];
    let wrap = [0x33u8; 24];

    // Fill half 0 and half 1 on stream 1 while events remain busy.
    assert!(
        ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&half0, dst)], 1)
            .unwrap()
    );
    assert!(
        ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&half1, dst)], 1)
            .unwrap()
    );
    // Wrap back into busy half 0 must decline (fallback to synchronous H2D).
    assert!(
        !ring
            .copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&wrap, dst)], 1)
            .unwrap()
    );
    // Once prior DMA completes, wrap succeeds and copies the new payload.
    gpu.ready.store(true, Ordering::Release);
    assert!(
        ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&wrap, dst)], 1)
            .unwrap()
    );
    assert_eq!(gpu.inner.read_alloc(dst).unwrap()[..24], wrap);

    ring.destroy(&gpu);
    assert_eq!(gpu.destroyed.load(Ordering::Relaxed), 2);
}

#[test]
fn mtp_lm_head_tc_pipe_defaults_off_and_admits_aligned_b_ge_8_only_on_opt_in() {
    assert!(!parse_mtp_lm_head_tc_pipe_env(None));
    assert!(!parse_mtp_lm_head_tc_pipe_env(Some("0")));
    assert!(!parse_mtp_lm_head_tc_pipe_env(Some("auto")));
    assert!(parse_mtp_lm_head_tc_pipe_env(Some("1")));
    assert!(parse_mtp_lm_head_tc_pipe_env(Some(" 1 ")));

    let p = DevicePtr(256);
    assert!(!should_use_mtp_lm_head_pipe(true, 4, 2560, true, p, p, p));
    assert!(!should_use_mtp_lm_head_pipe(true, 7, 2560, true, p, p, p));
    assert!(should_use_mtp_lm_head_pipe(true, 8, 2560, true, p, p, p));
    assert!(should_use_mtp_lm_head_pipe(true, 16, 2560, true, p, p, p));
    assert!(!should_use_mtp_lm_head_pipe(false, 16, 2560, true, p, p, p));
    assert!(!should_use_mtp_lm_head_pipe(true, 16, 2560, false, p, p, p));
    assert!(!should_use_mtp_lm_head_pipe(true, 16, 2544, true, p, p, p));
    assert!(!should_use_mtp_lm_head_pipe(
        true,
        16,
        2560,
        true,
        DevicePtr(264),
        p,
        p,
    ));
}

#[test]
fn batched_cohort_confidence_gate_never_early_stops_or_truncates_per_sequence() {
    let active_policy = GatePolicy {
        threshold: Some(0.6),
        max_drafts: None,
        early_stop: true,
        log: false,
    };
    // Even with MTP_CONF=0.6 + MTP_CONF_EARLY=1 + greedy, b >= 2 cohorts never
    // enable prob readback, mid-chain early-stop syncs, or per-sequence truncation.
    assert_eq!(
        resolve_chain_confidence_gate_for_scope(
            ChainConfidenceScope::BatchedCohort,
            active_policy,
            true,
            Some(0.6),
            true,
            false,
        ),
        ChainConfidenceGate {
            want_prob: false,
            early: None,
            apply_truncate: false,
        }
    );
    // Diagnostic logging on b >= 2 still skips early-stop and truncation.
    let logged_policy = GatePolicy {
        log: true,
        ..active_policy
    };
    assert_eq!(
        resolve_chain_confidence_gate_for_scope(
            ChainConfidenceScope::BatchedCohort,
            logged_policy,
            true,
            Some(0.6),
            true,
            false,
        ),
        ChainConfidenceGate {
            want_prob: true,
            early: None,
            apply_truncate: false,
        }
    );
    // Missing argmax_prob_k disables want_prob even with log=true.
    assert_eq!(
        resolve_chain_confidence_gate_for_scope(
            ChainConfidenceScope::BatchedCohort,
            logged_policy,
            true,
            Some(0.6),
            false,
            false,
        ),
        ChainConfidenceGate {
            want_prob: false,
            early: None,
            apply_truncate: false,
        }
    );
}

#[test]
fn packed_meta_scratch_reuses_capacity_and_zeroes_active_span() {
    let mut buf = Vec::new();
    let s1 = prepare_packed_meta_scratch(&mut buf, 4096);
    assert_eq!(s1.len(), 4096);
    assert!(s1.iter().all(|&b| b == 0));
    s1.fill(0xAB);
    let ptr_after_grow = buf.as_ptr();

    // Shrinking reuses the exact allocation and zeroes the active prefix.
    let s2 = prepare_packed_meta_scratch(&mut buf, 1024);
    assert_eq!(s2.len(), 1024);
    assert!(s2.iter().all(|&b| b == 0));
    assert_eq!(buf.as_ptr(), ptr_after_grow);

    // Growing back up to high-water mark also zeroes previously written tail bytes.
    let s3 = prepare_packed_meta_scratch(&mut buf, 4096);
    assert_eq!(s3.len(), 4096);
    assert!(s3.iter().all(|&b| b == 0));
    assert_eq!(buf.as_ptr(), ptr_after_grow);
}

#[test]
fn batch_gumbel_meta_layout_is_aligned_and_disjoint_up_to_c64() {
    for b in 2..=64usize {
        let bt_off = if b <= 8 { 768 } else { 2560 };
        let stride = (bt_off + b * 64 * 4).next_multiple_of(256);
        for num_drafts in 1..=7usize {
            let base_off = num_drafts * stride;
            let layout = BatchGumbelMetaLayout::compute(base_off, b, num_drafts)
                .expect("layout fits without overflow");
            assert_eq!(layout.seeds_off % 8, 0, "b={b} num_drafts={num_drafts}");
            assert!(layout.seeds_off >= base_off);
            assert_eq!(layout.pos_base_off, layout.seeds_off + b * 8);
            assert_eq!(layout.pos_base_off % 8, 0, "b={b} num_drafts={num_drafts}");
            assert_eq!(layout.total_bytes % 256, 0);
            for step in 0..num_drafts {
                let step_off = layout.step_pos_off(step, b);
                assert_eq!(step_off % 8, 0, "b={b} step={step}");
                assert!(step_off + b * 8 <= base_off + layout.total_bytes);
            }
        }
    }
}

#[test]
fn mtp_coalesced_nvfp4_lm_head_and_fc_gate_admits_b_ge_6_and_enforces_bounds() {
    // LM head coalesced NVFP4 engages for b in 6..=QWEN4_NATIVE_MAX_ROWS when TC dense is active.
    for b in [2usize, 4, 5] {
        assert!(!should_use_mtp_lm_head_coalesced(true, b, 2560), "b={b}");
    }
    for b in [
        6usize,
        7,
        8,
        16,
        32,
        64,
        128,
        crate::layers::QWEN4_NATIVE_MAX_ROWS,
    ] {
        assert!(should_use_mtp_lm_head_coalesced(true, b, 2560), "b={b}");
        assert!(!should_use_mtp_lm_head_coalesced(false, b, 2560), "b={b}");
    }
    assert!(!should_use_mtp_lm_head_coalesced(
        true,
        crate::layers::QWEN4_NATIVE_MAX_ROWS + 1,
        2560
    ));
    assert!(!should_use_mtp_lm_head_coalesced(true, 16, 0));
    assert!(!should_use_mtp_lm_head_coalesced(true, 16, 2544));

    // Pre-FC coalesced NVFP4 engages for b in 6..=64 (4*b <= 256) with adequate staging.
    let qkv_cap = 64 * 2560 * 2;
    assert!(!should_use_mtp_fc_coalesced(
        true, 5, 4, 2560, qkv_cap, true
    ));
    for b in [6usize, 7, 8, 16, 32, 33, 48, 64] {
        assert!(
            should_use_mtp_fc_coalesced(true, b, 4, 2560, qkv_cap, true),
            "b={b}"
        );
        assert!(
            !should_use_mtp_fc_coalesced(false, b, 4, 2560, qkv_cap, true),
            "b={b}"
        );
        assert!(
            !should_use_mtp_fc_coalesced(true, b, 4, 2560, qkv_cap, false),
            "b={b}"
        );
    }
    // b=65 -> 4*65 = 260 > 256 fails closed; insufficient qkv_output or bad hc fails closed.
    assert!(!should_use_mtp_fc_coalesced(
        true,
        65,
        4,
        2560,
        128 * 2560 * 2,
        true
    ));
    assert!(!should_use_mtp_fc_coalesced(
        true,
        64,
        4,
        2560,
        64 * 2560 * 2 - 2,
        true
    ));
    assert!(!should_use_mtp_fc_coalesced(
        true,
        16,
        4,
        2560,
        16 * 2560 * 2 - 2,
        true
    ));
    assert!(!should_use_mtp_fc_coalesced(
        true, 16, 2, 2560, qkv_cap, true
    ));
    assert!(!should_use_mtp_fc_coalesced(
        true,
        usize::MAX,
        4,
        2560,
        usize::MAX,
        true
    ));
}

#[test]
fn batched_argmax_prob_and_ragged_gpu_chain_gates_enforce_invariants() {
    assert!(!should_use_batched_argmax_prob(0, true));
    assert!(!should_use_batched_argmax_prob(1, true));
    assert!(should_use_batched_argmax_prob(2, true));
    assert!(should_use_batched_argmax_prob(64, true));
    assert!(!should_use_batched_argmax_prob(64, false));

    let lanes_64: Vec<usize> = (0..64).collect();
    let targets_homog = vec![4usize; 64];
    assert!(can_gpu_chain_ragged_steps(
        0.0,
        &lanes_64,
        &targets_homog,
        4,
        2048,
        2048,
        262_144,
        524_288
    ));
    // Positive pcum_thresh requires host compaction between steps.
    assert!(!can_gpu_chain_ragged_steps(
        0.30,
        &lanes_64,
        &targets_homog,
        4,
        2048,
        2048,
        262_144,
        524_288
    ));
    // Single-step chain already performs one final D2H.
    assert!(!can_gpu_chain_ragged_steps(
        0.0,
        &lanes_64,
        &vec![1usize; 64],
        1,
        2048,
        2048,
        262_144,
        524_288
    ));
    // Heterogeneous active target depths require per-step host compaction.
    let mut targets_hetero = targets_homog.clone();
    targets_hetero[17] = 2;
    assert!(!can_gpu_chain_ragged_steps(
        0.0,
        &lanes_64,
        &targets_hetero,
        4,
        2048,
        2048,
        262_144,
        524_288
    ));
    // Inactive zero-target lane excluded from active_lanes still allows GPU chaining.
    let active_subset: Vec<usize> = (0..64).filter(|&i| i != 17).collect();
    targets_hetero[17] = 0;
    assert!(can_gpu_chain_ragged_steps(
        0.0,
        &active_subset,
        &targets_hetero,
        4,
        2048,
        2048,
        262_144,
        524_288
    ));
    // Scratch / span overflow fails closed to per-step path.
    assert!(!can_gpu_chain_ragged_steps(
        0.0,
        &lanes_64,
        &vec![9usize; 64],
        9,
        2048,
        2048,
        262_144,
        524_288
    ));
    assert!(!can_gpu_chain_ragged_steps(
        0.0,
        &lanes_64,
        &targets_homog,
        4,
        2048,
        2048,
        524_289,
        524_288
    ));
}

#[test]
fn ragged_mtp_env_parsers_and_c1_sync_mid_chain_defaults() {
    // Ragged MTP defaults ON when adaptive_k=true, OFF when adaptive_k=false or =0.
    assert!(parse_batched_ragged_mtp(None, true));
    assert!(parse_batched_ragged_mtp(Some("1"), true));
    assert!(!parse_batched_ragged_mtp(Some("0"), true));
    assert!(!parse_batched_ragged_mtp(Some("false"), true));
    assert!(!parse_batched_ragged_mtp(Some("off"), true));
    assert!(!parse_batched_ragged_mtp(None, false));
    assert!(!parse_batched_ragged_mtp(Some("1"), false));

    // p_cum threshold defaults to 0.30, honors valid [0, 1), falls back on invalid.
    assert!((parse_mtp_ragged_pcum_thresh(None) - 0.30).abs() < 1e-6);
    assert!((parse_mtp_ragged_pcum_thresh(Some("0.25")) - 0.25).abs() < 1e-6);
    assert!((parse_mtp_ragged_pcum_thresh(Some("0")) - 0.0).abs() < 1e-6);
    assert!((parse_mtp_ragged_pcum_thresh(Some("1.0")) - 0.30).abs() < 1e-6);
    assert!((parse_mtp_ragged_pcum_thresh(Some("-0.1")) - 0.30).abs() < 1e-6);
    assert!((parse_mtp_ragged_pcum_thresh(Some("nan")) - 0.30).abs() < 1e-6);

    // C=1 sync_mid_chain defaults ON so MTP_CONF early-stops low-confidence drafts,
    // while honoring explicit =0/=1 overrides.
    for n in 1..=8 {
        assert!(should_sync_mid_chain(None, n), "n={n}");
        assert!(should_sync_mid_chain(Some("1"), n), "n={n}");
        assert!(!should_sync_mid_chain(Some("0"), n), "n={n}");
    }
}

#[test]
fn hidden_compaction_copies_are_non_overlapping_and_preserve_exact_rows() {
    let cases: &[(&[usize], &[usize])] = &[
        (&[0, 1, 2, 3, 4, 5], &[0, 1, 2, 3, 4, 5]),
        (&[0, 1, 2, 3, 4, 5], &[1, 2, 3, 5]),
        (&[0, 1, 2, 3, 4, 5], &[3, 4, 5]),
        (&[0, 1, 2, 3, 4, 5], &[0, 1, 4, 5]),
        (&[0, 1, 2, 3, 4, 5], &[5]),
        (&[1, 3, 5, 7], &[7]),
        (&[0, 1, 2, 3], &[]),
    ];
    for &(prev, next) in cases {
        let copies = plan_hidden_compaction_copies(prev, next);
        let mut rows: Vec<usize> = prev.to_vec();
        for &(src, dst, count) in &copies {
            assert!(count > 0);
            assert!(
                dst + count <= src,
                "overlapping copy src={src} dst={dst} count={count} for {prev:?}->{next:?}"
            );
            let tmp: Vec<usize> = rows[src..src + count].to_vec();
            rows[dst..dst + count].copy_from_slice(&tmp);
        }
        assert_eq!(
            &rows[..next.len()],
            next,
            "compaction mismatch for {prev:?}->{next:?} copies={copies:?}"
        );
    }
}

#[test]
fn active_lane_pcum_compaction_expands_confident_lanes_and_stops_uncertain_lanes() {
    // 4 lanes with target_drafts = [6, 6, 6, 3], threshold = 0.15:
    // - lane 0 (hopeless prose): p0 = 0.10 < 0.15 -> stops at step 0 with 0 drafts (K=1), steps_run=1
    // - lane 1 (medium prose):   p = [0.60, 0.50, 0.40] -> p_cum = [0.60, 0.30, 0.12 < 0.15] -> 2 drafts (K=3), steps_run=3
    // - lane 2 (confident code): p = [0.95; 6] -> 6 drafts (K=7), steps_run=6 (runs last 3 steps with b_act==1!)
    // - lane 3 (capped target):  p = [0.90; 3], target=3 -> 3 drafts (K=4), steps_run=3
    let target_drafts = [6usize, 6, 6, 3];
    let max_steps = 6usize;
    let thresh = 0.15f32;
    let probs_by_lane: [&[f32]; 4] = [
        &[0.10],
        &[0.60, 0.50, 0.40],
        &[0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
        &[0.90, 0.90, 0.90],
    ];
    for allow_zero in [false, true] {
        let mut active = vec![0usize, 1, 2, 3];
        let mut p_cum = vec![1.0f32; 4];
        let mut steps_run = vec![0usize; 4];
        let mut computed = vec![Vec::new(); 4];
        let mut emitted = vec![Vec::new(); 4];
        let mut cur_tokens = vec![10u32, 20, 30, 40];
        let mut active_counts = Vec::new();

        for step in 0..max_steps {
            if active.is_empty() {
                break;
            }
            active_counts.push(active.len());
            let step_toks: Vec<u32> = active
                .iter()
                .map(|&s| 100 * (s as u32 + 1) + step as u32)
                .collect();
            let step_probs: Vec<f32> = active.iter().map(|&s| probs_by_lane[s][step]).collect();
            active = advance_ragged_mtp_step(
                step,
                max_steps,
                thresh,
                allow_zero,
                &active,
                &step_toks,
                &step_probs,
                &target_drafts,
                &mut p_cum,
                &mut steps_run,
                &mut computed,
                &mut emitted,
                &mut cur_tokens,
                None,
            );
        }

        // Compaction progression: 4 -> 3 -> 3 -> 1 -> 1 -> 1 (zero wasted work on stopped lanes!).
        assert_eq!(active_counts, vec![4, 3, 3, 1, 1, 1]);
        assert!(active.is_empty());
        assert_eq!(steps_run, vec![1, 3, 6, 3]);
        assert_eq!(
            emitted[0].len(),
            usize::from(!allow_zero),
            "lane 0 -> K=1 when allow_zero else K=2 without extra steps"
        );
        assert_eq!(emitted[1].len(), 2, "lane 1 -> K=3");
        assert_eq!(emitted[2].len(), 6, "lane 2 -> K=7");
        assert_eq!(emitted[3].len(), 3, "lane 3 -> K=4 (target cap)");
        assert_eq!(computed[0].len(), 1, "KV steps match last_num_drafted");
        assert_eq!(computed[1].len(), 3, "KV steps match last_num_drafted");
        assert_eq!(computed[2].len(), 6, "KV steps match last_num_drafted");
        assert_eq!(computed[3].len(), 3, "KV steps match last_num_drafted");
    }
}

#[test]
fn ragged_graph_bucket_quantization_trims_lowest_tail_pcum_and_preserves_k2_floor() {
    assert_eq!(ragged_row_bucket_step(2), 2);
    assert_eq!(ragged_row_bucket_step(4), 4);
    assert_eq!(ragged_row_bucket_step(8), 8);
    assert_eq!(ragged_row_bucket_step(16), 8);
    assert_eq!(ragged_row_bucket_step(32), 16);

    // Straggler stop engages for b >= 4, step >= 2, and <= 25% surviving.
    assert!(!ragged_mtp_should_stop_stragglers(3, 3, 1));
    assert!(ragged_mtp_should_stop_stragglers(4, 2, 1));
    assert!(!ragged_mtp_should_stop_stragglers(4, 2, 2));
    assert!(!ragged_mtp_should_stop_stragglers(8, 1, 1));
    assert!(ragged_mtp_should_stop_stragglers(8, 2, 1));
    assert!(ragged_mtp_should_stop_stragglers(8, 2, 2));
    assert!(!ragged_mtp_should_stop_stragglers(8, 2, 3));
    assert!(ragged_mtp_should_stop_stragglers(16, 2, 4));
    assert!(!ragged_mtp_should_stop_stragglers(16, 2, 5));

    // C=4 (step=4): 19 total rows -> trims 3 lowest-tail-pcum drafts to 16 rows
    // without dropping lane 0 below 1 draft (K=2).
    let mut drafts = vec![
        vec![1],
        vec![2, 3, 4],
        vec![5, 6, 7, 8, 9],
        vec![10, 11, 12, 13, 14, 15],
    ];
    let mut pcums = vec![
        vec![0.05],
        vec![0.70, 0.40, 0.12],
        vec![0.80, 0.55, 0.35, 0.20, 0.10],
        vec![0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
    ];
    quantize_ragged_drafts_to_graph_bucket(&mut drafts, &mut pcums, false);
    let total: usize = drafts.iter().map(|d| d.len() + 1).sum();
    assert_eq!(total, 16);
    assert_eq!(drafts[0].len(), 1, "lane 0 preserved at K=2 floor");
    assert_eq!(drafts[1].len(), 2, "lane 1 trimmed 0.12 tail");
    assert_eq!(drafts[2].len(), 3, "lane 2 trimmed 0.10 and 0.20 tails");
    assert_eq!(drafts[3].len(), 6, "lane 3 confident drafts untouched");
}

#[test]
fn mtp_graph_policy_auto_enables_above_64_sms_and_keeps_gb10_off() {
    assert_eq!(parse_mtp_graph_policy(None), MtpGraphPolicy::Auto);
    assert_eq!(parse_mtp_graph_policy(Some("auto")), MtpGraphPolicy::Auto);
    assert_eq!(parse_mtp_graph_policy(Some("0")), MtpGraphPolicy::Disabled);
    assert_eq!(
        parse_mtp_graph_policy(Some("false")),
        MtpGraphPolicy::Disabled
    );
    assert_eq!(
        parse_mtp_graph_policy(Some("OFF")),
        MtpGraphPolicy::Disabled
    );
    assert_eq!(parse_mtp_graph_policy(Some("1")), MtpGraphPolicy::Enabled);
    assert_eq!(
        parse_mtp_graph_policy(Some("true")),
        MtpGraphPolicy::Enabled
    );
    assert_eq!(parse_mtp_graph_policy(Some("ON")), MtpGraphPolicy::Enabled);

    // GB10 (<= 64 SMs): default OFF under Auto; explicit "1" opts in.
    for sm in [48u32, 64u32] {
        assert!(!resolve_mtp_graph_enabled(MtpGraphPolicy::Auto, sm));
        assert!(!resolve_mtp_graph_enabled(MtpGraphPolicy::Disabled, sm));
        assert!(resolve_mtp_graph_enabled(MtpGraphPolicy::Enabled, sm));
    }
    // SM120 / SM103 (> 64 SMs): auto-ON under Auto; explicit "0" opts out.
    for sm in [128u32, 188u32] {
        assert!(resolve_mtp_graph_enabled(MtpGraphPolicy::Auto, sm));
        assert!(!resolve_mtp_graph_enabled(MtpGraphPolicy::Disabled, sm));
        assert!(resolve_mtp_graph_enabled(MtpGraphPolicy::Enabled, sm));
    }
}

#[test]
fn mtp_graph_bt_bucket_and_cache_lifecycle() {
    let scratch = 16 * 1024 * 1024;
    assert_eq!(mtp_graph_bt_blocks_cap(1, 32, 3, 2560, scratch), 16);
    assert_eq!(mtp_graph_bt_blocks_cap(16, 32, 3, 2560, scratch), 16);
    assert_eq!(mtp_graph_bt_blocks_cap(17, 32, 3, 2560, scratch), 32);
    assert_eq!(mtp_graph_bt_blocks_cap(65, 64, 3, 2560, scratch), 128);
    // Tiny scratch fails closed to exact max_mb instead of overflowing.
    assert_eq!(
        mtp_graph_bt_blocks_cap(17, 64, 4, 2560, CHAIN_META_OFFSET + 32_768),
        17
    );

    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut cache = MtpGraphCache::default();
    let make_key = |batch_size: usize, step_count: usize, want_prob: bool| MtpGraphKey {
        batch_size,
        step_count,
        want_prob,
        bt_blocks_cap: 16,
        vocab: 248_320,
        write_step_token_out: true,
        tc_dense_scope: true,
        coalesced_fc: true,
        coalesced_lm_head: true,
        scratch_ptr: 0x1000,
        hidden_ptr: 0x2000,
        residual_ptr: 0x3000,
        logits_ptr: 0x4000,
        k_pool_ptr: 0x5000,
        v_pool_ptr: 0x6000,
        stream: 1,
    };

    let k1 = make_key(32, 3, false);
    assert!(cache.get(&k1).is_none());
    assert!(cache.mark_warmed(k1));
    assert!(!cache.mark_warmed(k1));
    cache.insert(&gpu, k1, GraphHandle(42));
    assert_eq!(cache.get(&k1).map(|h| h.0), Some(42));

    let k_prob = make_key(32, 3, true);
    assert!(cache.get(&k_prob).is_none());
    cache.mark_poisoned(k_prob);
    assert!(cache.is_poisoned(&k_prob));

    // FIFO eviction enforces MTP_GRAPH_CACHE_CAP cleanly.
    for i in 0..(MTP_GRAPH_CACHE_CAP + 4) {
        let k = make_key(i + 2, 1, false);
        cache.insert(&gpu, k, GraphHandle((i + 100) as u64));
    }
    assert_eq!(cache.graphs.len(), MTP_GRAPH_CACHE_CAP);
    assert!(cache.get(&k1).is_none(), "oldest entry evicted at cap");

    cache.clear(&gpu);
    assert!(cache.graphs.is_empty());
    assert!(cache.order.is_empty());
    assert!(cache.warmed.is_empty());
    assert!(cache.poisoned.is_empty());
}
