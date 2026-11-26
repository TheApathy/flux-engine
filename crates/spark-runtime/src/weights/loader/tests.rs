// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const MOCK_FREE: usize = 120 * 1024 * 1024 * 1024;
static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "atlas-indexed-admission-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn header(&self, filename: &str, header: serde_json::Value, payload: &[u8]) {
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(self.0.join(filename)).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(payload).unwrap();
    }

    fn shard(&self, filename: &str, tensors: &[(&str, usize)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for &(name, bytes) in tensors {
            let start = data.len();
            data.resize(start + bytes, name.as_bytes()[0]);
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": "U8", "shape": [bytes], "data_offsets": [start, data.len()]
                }),
            );
        }
        self.header(filename, header.into(), &data);
    }

    fn index(&self, filename: &str, weights: &[(&str, &str)]) {
        let map: HashMap<_, _> = weights.iter().copied().collect();
        std::fs::write(
            self.0.join(filename),
            serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap(),
        )
        .unwrap();
    }

    fn three_tensor_index(&self, filename: &str) {
        self.shard("part.safetensors", &[("a", 64), ("b", 8), ("c", 16)]);
        self.index(
            filename,
            &[
                ("a", "part.safetensors"),
                ("b", "part.safetensors"),
                ("c", "part.safetensors"),
            ],
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn loader(construction: usize) -> SafetensorsLoader {
    let mut loader = SafetensorsLoader::new();
    loader.peak_memory_multiplier = Some(1.0);
    loader.construction_overhead_bytes = construction;
    loader
}

#[test]
fn indexed_combined_peak_rejects_the_old_split_admission_gap() {
    for filename in [
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ] {
        let fixture = Fixture::new();
        fixture.three_tensor_index(filename);
        let gpu = MockGpuBackend::new();
        // Both 88 bytes of weights and 13 construction bytes fit separately,
        // but their combined 101-byte peak does not fit the 100-byte budget.
        let error = loader(13)
            .load(&fixture.0, &gpu, MOCK_FREE - 100)
            .err()
            .expect("combined admission must fail");
        assert!(format!("{error:#}").contains("OOM pre-flight"));
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn indexed_exact_fit_counts_shared_shards_and_construction_once() {
    let fixture = Fixture::new();
    fixture.three_tensor_index("model.safetensors.index.json");
    let gpu = MockGpuBackend::new();
    let store = loader(12)
        .load(&fixture.0, &gpu, MOCK_FREE - 100)
        .expect("88 weight bytes + 12 construction bytes fit exactly");
    assert_eq!(store.len(), 3);
    assert_eq!(gpu.alloc_count(), 3);
    assert_eq!(
        gpu.read_alloc(store.get("a").unwrap().ptr).unwrap(),
        vec![b'a'; 64]
    );
}

#[test]
fn indexed_combined_admission_preserves_the_full_reserve() {
    let fixture = Fixture::new();
    fixture.three_tensor_index("model.safetensors.index.json");
    let gpu = MockGpuBackend::new();
    let error = loader(12)
        .load(&fixture.0, &gpu, MOCK_FREE - 99)
        .err()
        .expect("one-byte reserve deficit must reject");
    assert!(format!("{error:#}").contains("OOM pre-flight"));
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn indexed_explicit_multiplier_is_not_reduced_or_applied_to_construction() {
    let fixture = Fixture::new();
    fixture.shard("part.safetensors", &[("a", 64)]);
    fixture.index("model.safetensors.index.json", &[("a", "part.safetensors")]);
    let mut candidate = loader(21);
    candidate.peak_memory_multiplier = Some(1.25);
    let gpu = MockGpuBackend::new();
    assert!(candidate.load(&fixture.0, &gpu, MOCK_FREE - 100).is_err());
    assert_eq!(gpu.alloc_count(), 0);
    candidate.construction_overhead_bytes = 20;
    candidate
        .load(&fixture.0, &gpu, MOCK_FREE - 100)
        .expect("64 * 1.25 + 20 fits exactly");
}

#[test]
fn indexed_fp8_default_multiplier_uses_actual_shard_headers() {
    let fixture = Fixture::new();
    fixture.header(
        "part.safetensors",
        serde_json::json!({
            "a": { "dtype": "F8_E4M3", "shape": [64], "data_offsets": [0, 64] }
        }),
        &[0; 64],
    );
    fixture.index("model.safetensors.index.json", &[("a", "part.safetensors")]);
    let mut candidate = loader(5);
    candidate.peak_memory_multiplier = None;
    let gpu = MockGpuBackend::new();
    assert!(candidate.load(&fixture.0, &gpu, MOCK_FREE - 100).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn indexed_filter_is_shared_by_admission_and_upload() {
    let fixture = Fixture::new();
    fixture.three_tensor_index("model.safetensors.index.json");
    let mut candidate = loader(76);
    candidate.extra_skip = Some(std::sync::Arc::new(|name| name == "a"));
    let gpu = MockGpuBackend::new();
    let store = candidate.load(&fixture.0, &gpu, MOCK_FREE - 100).unwrap();
    assert_eq!(store.len(), 2);
    assert!(!store.contains("a"));
    assert_eq!(gpu.alloc_count(), 2);
    candidate.extra_skip = None;
    let control = MockGpuBackend::new();
    assert!(
        candidate
            .load(&fixture.0, &control, MOCK_FREE - 100)
            .is_err()
    );
    assert_eq!(control.alloc_count(), 0);
}

#[test]
fn indexed_multi_shard_peak_is_combined_before_any_allocation() {
    let fixture = Fixture::new();
    fixture.shard("first.safetensors", &[("a", 64)]);
    fixture.shard("second.safetensors", &[("b", 24)]);
    fixture.index(
        "model.safetensors.index.json",
        &[("a", "first.safetensors"), ("b", "second.safetensors")],
    );
    let gpu = MockGpuBackend::new();
    assert!(loader(13).load(&fixture.0, &gpu, MOCK_FREE - 100).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn missing_indexed_shard_fails_before_any_allocation() {
    let fixture = Fixture::new();
    fixture.shard("first.safetensors", &[("a", 64)]);
    fixture.index(
        "model.safetensors.index.json",
        &[("a", "first.safetensors"), ("b", "missing.safetensors")],
    );
    let gpu = MockGpuBackend::new();
    assert!(loader(0).load(&fixture.0, &gpu, 0).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn single_and_unindexed_shards_keep_combined_admission() {
    for filename in [
        "model.safetensors",
        "model.safetensors-00001-of-00001.safetensors",
    ] {
        let fixture = Fixture::new();
        fixture.shard(filename, &[("a", 88)]);
        let gpu = MockGpuBackend::new();
        assert!(loader(13).load(&fixture.0, &gpu, MOCK_FREE - 100).is_err());
        assert_eq!(gpu.alloc_count(), 0);
        loader(12).load(&fixture.0, &gpu, MOCK_FREE - 100).unwrap();
        assert_eq!(gpu.alloc_count(), 1);
    }
}

#[test]
fn combined_accounting_rejects_invalid_multipliers_and_overflow() {
    for multiplier in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0, 0.5] {
        assert!(combined_admission_bytes(1, multiplier, 0, 0).is_err());
    }
    assert!(combined_admission_bytes(usize::MAX, 2.0, 0, 0).is_err());
    assert!(combined_admission_bytes(1, f64::MAX, 0, 0).is_err());
    assert!(combined_admission_bytes(1, 1.0, usize::MAX, 0).is_err());
    assert!(combined_admission_bytes(1, 1.0, 0, usize::MAX).is_err());
    assert_eq!(combined_admission_bytes(3, 1.5, 7, 11).unwrap(), (12, 23));
    assert_eq!(
        combined_admission_bytes(usize::MAX, 1.0, 0, 0).unwrap(),
        (usize::MAX, usize::MAX)
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn combined_accounting_does_not_round_large_integer_byte_counts_down() {
    let bytes = (1usize << 53) + 1;
    assert_eq!(
        combined_admission_bytes(bytes, 1.0, 0, 0).unwrap(),
        (bytes, bytes)
    );
}

#[test]
fn header_estimation_rejects_shape_and_byte_overflow() {
    assert!(estimated_tensor_bytes(&[usize::MAX, 2], safetensors::Dtype::U8).is_err());
    assert!(estimated_tensor_bytes(&[usize::MAX], safetensors::Dtype::BF16).is_err());
    let fixture = Fixture::new();
    fixture.header(
        "part.safetensors",
        serde_json::json!({
            "a": { "dtype": "U8", "shape": [usize::MAX], "data_offsets": [0, 0] },
            "b": { "dtype": "U8", "shape": [1], "data_offsets": [0, 0] }
        }),
        &[],
    );
    let paths = [fixture.0.join("part.safetensors")];
    assert!(estimate_load_bytes(&paths, &|_| false).is_err());
    assert!(estimate_has_fp8(&paths, &|_| false).is_err());
}
