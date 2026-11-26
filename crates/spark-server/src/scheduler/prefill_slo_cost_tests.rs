// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn buckets_and_bands() {
    assert_eq!(bucket(1), MIN_BUCKET);
    assert_eq!(bucket(128), 128);
    assert_eq!(bucket(129), 256);
    assert_eq!(bucket(368), 512);
    assert_eq!(bucket(512), 512);
    assert_eq!(bucket(4096), 4096);
    assert_eq!(band(0), 0);
    assert_eq!(band(8_191), 0);
    assert_eq!(band(8_192), 1);
    assert_eq!(band(32_768), 2);
    assert_eq!(band(131_072), 3);
    assert_eq!(band(usize::MAX), 3);
    assert_eq!(BAND_LABELS.len(), BAND_UPPER.len() + 1);
}

#[test]
fn static_table_is_monotonic_in_rows_and_offset_and_calibrated() {
    let mut prev = 0.0;
    for b in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768] {
        let ms = static_chunk_ms(b, 0);
        assert!(ms > prev, "{b}");
        prev = ms;
        for band_index in 1..4 {
            assert!(static_chunk_ms(b, band_index) > static_chunk_ms(b, band_index - 1));
        }
    }
    // Calibration point: the measured w4096 chunk (1555 ms per 8K).
    assert!((static_chunk_ms(4096, 0) - 777.0).abs() < 10.0);
}

#[test]
fn unmeasured_cells_fall_back_to_the_table() {
    let model = ChunkCostModel::default();
    assert_eq!(model.predict_ms(512, 0), static_chunk_ms(512, 0));
    assert_eq!(model.predict_ms(300, 40_000), static_chunk_ms(512, 2));
    assert!(!model.is_measured(512, 0));
}

#[test]
fn ewma_replaces_the_table_after_warmup_and_min_samples() {
    let mut model = ChunkCostModel::default();
    // Warm-up sample of the bucket is dropped.
    assert_eq!(model.observe(512, 0, 5_000.0, 350.0), None);
    let first = model.observe(512, 0, 200.0, 350.0).unwrap();
    assert_eq!((first.bucket, first.band), (512, 0));
    assert_eq!(first.ewma_ms, 200.0);
    assert_eq!(first.error_ms, -150.0);
    // One sample is not enough: the (ratio-scaled) table still predicts.
    assert!(!model.is_measured(512, 0));
    let second = model.observe(512, 0, 300.0, 200.0).unwrap();
    assert!((second.ewma_ms - (200.0 + EWMA_ALPHA * 100.0)).abs() < 1e-9);
    assert!(model.is_measured(512, 0));
    assert!((model.predict_ms(512, 0) - 230.0).abs() < 1e-9);
    // The band is its own cell: a deep-offset chunk has no samples yet.
    assert!(!model.is_measured(512, 50_000));
}

#[test]
fn global_ratio_scales_unmeasured_cells() {
    let mut model = ChunkCostModel::default();
    model.observe(1024, 0, 0.0, 0.0); // warm-up
    // A box running at half the table's time.
    let half = static_chunk_ms(1024, 0) / 2.0;
    model.observe(1024, 0, half, 0.0);
    let scaled = model.predict_ms(2048, 0);
    assert!((scaled - static_chunk_ms(2048, 0) * 0.5).abs() < 1e-6);
    // The ratio is clamped.
    let mut hot = ChunkCostModel::default();
    hot.observe(512, 0, 0.0, 0.0);
    hot.observe(512, 0, 1e9, 0.0);
    assert!((hot.predict_ms(2048, 0) - static_chunk_ms(2048, 0) * 4.0).abs() < 1e-6);
}

#[test]
fn invalid_samples_are_ignored() {
    let mut model = ChunkCostModel::default();
    model.observe(512, 0, 1.0, 0.0); // warm-up
    assert_eq!(model.observe(0, 0, 100.0, 0.0), None);
    assert_eq!(model.observe(512, 0, f64::NAN, 0.0), None);
    assert_eq!(model.observe(512, 0, -1.0, 0.0), None);
    assert_eq!(model.predict_ms(512, 0), static_chunk_ms(512, 0));
}

#[test]
fn decode_clock_defaults_then_tracks_an_ewma() {
    let mut clock = DecodeStepClock::default();
    assert_eq!(clock.measured_ms(), None);
    assert_eq!(clock.step_ms(), DecodeStepClock::DEFAULT_MS);
    clock.observe(40.0);
    clock.observe(f64::INFINITY);
    assert_eq!(clock.step_ms(), 40.0);
    clock.observe(60.0);
    assert!((clock.step_ms() - (40.0 + EWMA_ALPHA * 20.0)).abs() < 1e-9);
}
