// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn selector_is_strict() {
    assert!(!parse(None).unwrap());
    assert!(!parse(Some("0")).unwrap());
    assert!(parse(Some("1")).unwrap());
    assert!(parse(Some("true")).is_err());
}

#[test]
fn score_v2_selector_is_strict() {
    assert_eq!(parse_score_v2(None).unwrap(), None);
    assert_eq!(parse_score_v2(Some("0")).unwrap(), None);
    assert_eq!(parse_score_v2(Some("1")).unwrap(), Some(ScoreV2Mode::Auto));
    assert_eq!(
        parse_score_v2(Some("2")).unwrap(),
        Some(ScoreV2Mode::Tiles2)
    );
    assert_eq!(
        parse_score_v2(Some("4")).unwrap(),
        Some(ScoreV2Mode::Tiles4)
    );
    assert_eq!(
        parse_score_v2(Some("8")).unwrap(),
        Some(ScoreV2Mode::Tiles8)
    );
    assert_eq!(
        parse_score_v2(Some("16")).unwrap(),
        Some(ScoreV2Mode::Tiles16)
    );
    assert!(parse_score_v2(Some("3")).is_err());
    assert_eq!(ScoreV2Mode::Auto.blocks_per_cta(2048), 128);
    assert_eq!(ScoreV2Mode::Auto.blocks_per_cta(4096), 256);
    assert_eq!(ScoreV2Mode::Auto.blocks_per_cta(8192), 512);
    assert_eq!(ScoreV2Mode::Auto.blocks_per_cta(16384), 1024);
    assert_eq!(ScoreV2Mode::Auto.blocks_per_cta(32768), 1024);
    assert_eq!(ScoreV2Mode::Tiles2.blocks_per_cta(32768), 128);
    assert_eq!(ScoreV2Mode::Tiles4.blocks_per_cta(2048), 256);
    assert_eq!(ScoreV2Mode::Tiles8.blocks_per_cta(2048), 512);
    assert_eq!(ScoreV2Mode::Tiles16.blocks_per_cta(2048), 1024);
}

#[test]
fn dense_window_matches_the_reference_boundary() {
    // n_p = (p+1)/4 <= 512 exactly for p <= 2050 (HF topk(min(512, n))).
    assert_eq!(DENSE_LAST_POS, 2050);
    assert_eq!((DENSE_LAST_POS + 1) / COMPRESS_RATIO, BLOCK_TOPK);
    assert_eq!((DENSE_LAST_POS + 2) / COMPRESS_RATIO, BLOCK_TOPK + 1);
    assert!(!chunk_is_sparse(0, 2048));
    assert!(!chunk_is_sparse(0, 2051));
    assert!(chunk_is_sparse(0, 2052));
    assert!(chunk_is_sparse(2048, 2048));
    assert!(chunk_is_sparse(8192, 16));
}

#[test]
fn plan_splits_dense_and_sparse_rows() {
    let first = SparsePlan::new(8192, 0).unwrap();
    assert_eq!(first.sparse_row0, 2051);
    assert_eq!(first.blocks, 2048);
    let cont = SparsePlan::new(8192, 8192).unwrap();
    assert_eq!(cont.sparse_row0, 0);
    assert_eq!(cont.blocks, 4096);
    let straddle = SparsePlan::new(2048, 2048).unwrap();
    assert_eq!(straddle.sparse_row0, 3);
    let dense = SparsePlan::new(16, 0).unwrap();
    assert_eq!(dense.sparse_row0, 16);
}

#[test]
fn plan_masks_cover_every_tail_block() {
    for (rows, start) in [(8192, 24576), (2048, 6144), (17, 2040), (1, 2051)] {
        let p = SparsePlan::new(rows, start).unwrap();
        let last = start + rows - 1;
        let tail_block = (last + 1) / COMPRESS_RATIO;
        assert!(tail_block < p.mask_words * 32, "{rows}@{start}");
        assert!(p.blocks <= p.ld && p.ld.is_multiple_of(64));
    }
}

#[test]
fn plan_scratch_regions_are_disjoint_and_aligned() {
    let p = SparsePlan::new(8192, 24576).unwrap();
    let offs = [
        p.iq_off,
        p.mask_off,
        p.entry_off,
        p.count_off,
        p.score_off,
        p.bytes,
    ];
    assert!(offs.windows(2).all(|w| w[0] < w[1]));
    assert!(offs.iter().all(|o| o % 256 == 0));
    assert_eq!(p.tiles, 820);
    // 32K context: one 2048-row score batch of 8192 blocks.
    assert_eq!(p.bytes - p.score_off, 2048 * 8192 * 4);
}

#[test]
fn topk_switches_to_global_memory_past_the_smem_row() {
    // 88 KB of fp32 scores = 22528 blocks = 90112 tokens staged in shared memory.
    assert!(topk_staged(0));
    assert!(topk_staged(22_528));
    assert!(!topk_staged(22_529));
    // Every context up to 32K stays on the staged kernel (no behaviour change).
    assert!(topk_staged(SparsePlan::new(8192, 24_576).unwrap().blocks));
    assert!(SparsePlan::new(8192, 81_924).is_ok());
    assert!(SparsePlan::new(0, 0).is_err());
}

#[test]
fn score_scratch_is_bounded_for_long_contexts() {
    // Up to 32K: one 2048-row batch, as before.
    for start in [0, 8192, 16_384, 24_576] {
        assert_eq!(SparsePlan::new(8192, start).unwrap().score_batch, 2048);
    }
    // 128K / 256K / 1M: fewer rows per batch, scratch stays at 64 MB.
    for (start, batch) in [(122_880, 512), (253_952, 256), (1_040_384, 64)] {
        let p = SparsePlan::new(8192, start).unwrap();
        assert_eq!(p.score_batch, batch, "{start}");
        assert!(p.bytes - p.score_off <= 64 << 20, "{start}");
        assert!(p.score_batch.is_multiple_of(64));
    }
}

#[test]
fn attention_launch_geometry_matches_the_kernel() {
    // qwen4_qsa_prefill_attn_t10: M = 120 rows -> 8 warps; kCap = 5132.
    assert_eq!(ATTN_THREADS, 256);
    assert_eq!(ATTN_CAP, 5132);
    const {
        assert!(ATTN_SMEM <= SMEM_OPTIN_MAX);
    }
}

#[test]
fn shared_memory_requests_fit_the_gb10_opt_in_limit() {
    // cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) rejects static + dynamic
    // above the opt-in limit (spark-w6 failed with topk 8,240 + 98,304).
    const {
        assert!(TOPK_STATIC_SMEM + MAX_TOPK_SMEM <= SMEM_OPTIN_MAX);
        assert!(ATTN_SMEM <= SMEM_OPTIN_MAX);
    }
}

#[test]
fn global_topk_caps_ctas_per_sm_to_keep_rows_in_l2() {
    // 2 CTAs/SM (40 KB + 8,240 static) through 128K tokens, 1 CTA/SM (88 KB) beyond.
    assert_eq!(topk_global_occupancy_smem(22_529), 40 * 1024);
    assert_eq!(topk_global_occupancy_smem(32_768), 40 * 1024);
    assert_eq!(topk_global_occupancy_smem(32_769), MAX_TOPK_SMEM);
    assert_eq!(topk_global_occupancy_smem(65_536), MAX_TOPK_SMEM);
    assert!(TOPK_STATIC_SMEM + topk_global_occupancy_smem(22_529) <= SMEM_OPTIN_MAX);
}

#[test]
fn scratch_grows_in_few_steps_over_a_long_request() {
    // 8192-row chunks of a 256K request: the plan grows every chunk, the
    // allocation only when it crosses a 64 MB step.
    let mut cap = 0;
    let mut reallocs = 0;
    for start in (0..262_144).step_by(8192) {
        let need = SparsePlan::new(8192, start).unwrap().bytes;
        if need > cap {
            cap = scratch_capacity(need);
            reallocs += 1;
        }
    }
    assert!(reallocs <= 4, "{reallocs} reallocations");
    assert!(cap <= 256 << 20, "{cap} B at 256K");
}

#[test]
fn v2_score_batch_fits_l2() {
    // V2 keeps the fp32 score batch at <= 32 MB from 8K through 128K contexts.
    for start in [0, 8192, 24_576, 57_344, 122_880] {
        let p = SparsePlan::v2(8192, start).unwrap();
        assert!(
            p.score_batch * p.ld * 4 <= 32 << 20 || p.score_batch == 64,
            "{start}"
        );
        assert!(p.score_batch.is_multiple_of(64));
        assert!(p.bytes <= SparsePlan::new(8192, start).unwrap().bytes);
    }
    assert_eq!(SparsePlan::v2(8192, 24_576).unwrap().score_batch, 1024);
    assert!(parse(Some("2")).is_err());
}

#[test]
fn reserve_covers_every_sparse_chunk_up_to_the_context_limit() {
    // The serve recipe's geometry: 16384-row chunks, --max-seq-len 135168.
    let (max_rows, max_seq_len) = (16_384, 135_168);
    for v2 in [false, true] {
        let reserve = reserve_bytes(max_rows, max_seq_len, v2, false).unwrap();
        for rows in [16, 2048, 2051, 8192, 12_000, 16_383, 16_384] {
            let starts = (0..=max_seq_len - rows)
                .step_by(997)
                .chain([max_seq_len - rows]);
            for start in starts.filter(|&s| chunk_is_sparse(s, rows)) {
                let p = if v2 {
                    SparsePlan::v2(rows, start)
                } else {
                    SparsePlan::new(rows, start)
                };
                let bytes = p.unwrap().bytes;
                assert!(
                    bytes <= reserve,
                    "v2={v2} rows={rows} start={start}: {bytes} > {reserve}"
                );
            }
        }
    }
    // The last chunk is not the largest under V2 at this limit.
    let last = SparsePlan::v2(max_rows, max_seq_len - max_rows)
        .unwrap()
        .bytes;
    assert!(reserve_bytes(max_rows, max_seq_len, true, false).unwrap() >= last);
    assert_eq!(
        reserve_bytes(max_rows, DENSE_LAST_POS + 1, true, false).unwrap(),
        0
    );
}

#[test]
fn fused_topk_plan_only_appends_the_group_maxima() {
    // ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK: every offset of the unfused plan is
    // unchanged; the maxima follow the scores, ld/8 words per score row.
    for (rows, start) in [
        (16_384, 0),
        (16_384, 16_384),
        (16_384, 114_688),
        (8192, 24_576),
        (17, 2040),
    ] {
        for v2 in [false, true] {
            let plain = SparsePlan::for_chunk(rows, start, v2, false).unwrap();
            let fused = SparsePlan::for_chunk(rows, start, v2, true).unwrap();
            let base = if v2 {
                SparsePlan::v2(rows, start).unwrap()
            } else {
                SparsePlan::new(rows, start).unwrap()
            };
            assert_eq!(plain, base, "{rows}@{start} v2={v2}");
            assert_eq!((plain.gmax_off, plain.ldg), (0, 0));
            let strip = SparsePlan {
                gmax_off: 0,
                ldg: 0,
                bytes: plain.bytes,
                ..fused
            };
            assert_eq!(strip, plain, "{rows}@{start} v2={v2}");
            let score_rows = (rows - plain.sparse_row0).min(plain.score_batch);
            assert_eq!(fused.ldg * 8, fused.ld);
            assert!(fused.gmax_off >= plain.score_off + score_rows * plain.ld * 4);
            assert!(fused.gmax_off.is_multiple_of(256));
            assert!(fused.bytes >= fused.gmax_off + score_rows * fused.ldg * 4);
            assert!(
                fused.bytes - plain.bytes <= (score_rows * fused.ldg * 4).next_multiple_of(256)
            );
        }
    }
}

#[test]
fn fused_topk_reserve_is_memory_neutral_at_the_serve_geometry() {
    // V2 batches keep the maxima at <= 4 MiB; the 64 MiB scratch step absorbs them.
    let (max_rows, max_seq_len) = (16_384, 135_168);
    let plain = reserve_bytes(max_rows, max_seq_len, true, false).unwrap();
    let fused = reserve_bytes(max_rows, max_seq_len, true, true).unwrap();
    assert!(
        fused > plain && fused - plain <= 4 << 20,
        "{plain} -> {fused}"
    );
    assert_eq!(scratch_capacity(fused), scratch_capacity(plain));
    for rows in [2048, 8192, 16_384] {
        for start in (0..=max_seq_len - rows)
            .step_by(4093)
            .filter(|&s| chunk_is_sparse(s, rows))
        {
            let p = SparsePlan::for_chunk(rows, start, true, true).unwrap();
            assert!(p.bytes <= fused, "rows={rows} start={start}");
        }
    }
}
