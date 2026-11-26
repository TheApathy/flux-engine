// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::model::seqstate::format::BlobReader;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};

fn exportable() -> ExportFacts {
    ExportFacts {
        rotary_identity: true,
        proposer_state: true,
        native_mtp_proposer: true,
        qsa_required: true,
        qsa_cache_present: true,
        ..ExportFacts::default()
    }
}

#[test]
fn quiescent_native_mtp_qsa_sequence_is_exportable() {
    check_exportable(&exportable()).unwrap();
    check_exportable(&ExportFacts {
        rotary_identity: true,
        ..ExportFacts::default()
    })
    .unwrap();
}

/// Each predicate input alone must flip the verdict: a guard that cannot
/// fail is not a guard.
#[test]
fn every_refusal_fires_on_its_own() {
    let cases: [(&str, fn(&mut ExportFacts)); 9] = [
        ("prefix cache", |f| f.prefix_cache_active = true),
        ("rotary", |f| f.rotary_identity = false),
        ("verify frame", |f| f.verify_frame_bound = true),
        ("deferred replay", |f| f.deferred_mtp_replay = true),
        ("disk blocks", |f| f.disk_blocks = true),
        ("alt proposer", |f| f.alt_proposer_state = true),
        ("foreign proposer", |f| f.native_mtp_proposer = false),
        ("mixed h", |f| f.mixed_h_representation = true),
        ("qsa cache missing", |f| f.qsa_cache_present = false),
    ];
    for (name, mutate) in cases {
        let mut f = exportable();
        mutate(&mut f);
        assert!(check_exportable(&f).is_err(), "{name} was accepted");
    }
}

fn kv_cache(gpu: &MockGpuBackend, dtype: KvCacheDtype, blocks: usize) -> PagedKvCache {
    PagedKvCache::new(
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 2,
            head_dim: 32,
            num_layers: 3,
            dtype,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        blocks,
        gpu,
    )
    .unwrap()
}

fn fill_block(kv: &PagedKvCache, gpu: &MockGpuBackend, block: u32, seed: u8) {
    for layer in 0..kv.num_layers() {
        let k: Vec<u8> = (0..kv.k_block_stride_bytes_for_layer(layer))
            .map(|i| (i as u8) ^ seed ^ ((layer as u8) << 4))
            .collect();
        let v: Vec<u8> = k.iter().map(|b| b.wrapping_add(seed)).collect();
        kv.write_block(layer, block, &k, &v, gpu).unwrap();
    }
}

/// The KV section of a blob is the cache's bytes as stored: a sequence on
/// scattered blocks restores onto different blocks, every layer and side
/// byte-identical, for a plain and a block-scaled (NVFP4) dtype.
#[test]
fn kv_blocks_round_trip_exactly_onto_other_physical_blocks() {
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Nvfp4] {
        let gpu = MockGpuBackend::new();
        let kv = kv_cache(&gpu, dtype, 12);
        let src = [7u32, 2, 3, 9];
        for (i, &b) in src.iter().enumerate() {
            fill_block(&kv, &gpu, b, 0x11 * (i as u8 + 1));
        }
        let dst = [0u32, 1, 5, 6];
        let header = SeqStateHeader {
            fingerprint: [0u8; DIGEST_BYTES],
            seq_len: 50,
            block_size: 16,
            num_blocks: 4,
            flags: 0,
            prompt_len: 50,
            mtp_num_blocks: 0,
            mtp_hidden_selected_row: 0,
            mtp_seq_len: 0,
            session_hash: 0,
            mtp_last_num_drafted: 0,
            mtp_hidden_position: 0,
            layout_digest: [0u8; DIGEST_BYTES],
            tokens: (0..50).collect(),
            mtp_last_drafts: vec![],
        };
        let regions = kv_block_regions(&kv, &src, SectionKind::KvK, SectionKind::KvV).unwrap();
        assert_eq!(regions.len(), 2 * kv.num_layers());
        let mut blob = Vec::new();
        engine::capture(&gpu, &header, &regions, &mut blob).unwrap();

        let dst_regions = kv_block_regions(&kv, &dst, SectionKind::KvK, SectionKind::KvV).unwrap();
        let mut r: &[u8] = &blob;
        engine::apply(&gpu, BlobReader::open(&mut r).unwrap(), &dst_regions).unwrap();
        for (s, d) in src.iter().zip(dst) {
            for layer in 0..kv.num_layers() {
                assert_eq!(
                    kv.read_block(layer, *s, &gpu).unwrap(),
                    kv.read_block(layer, d, &gpu).unwrap(),
                    "{dtype} layer {layer} block {s}->{d}"
                );
            }
        }
    }
}

#[test]
fn kv_regions_refuse_blocks_outside_the_pool() {
    let gpu = MockGpuBackend::new();
    let kv = kv_cache(&gpu, KvCacheDtype::Bf16, 4);
    assert!(kv_block_regions(&kv, &[1, 4], SectionKind::KvK, SectionKind::KvV).is_err());
}

/// A destination pool of another dtype has another geometry: the import is
/// refused before a byte is written.
#[test]
fn kv_dtype_change_is_refused_by_geometry() {
    let gpu = MockGpuBackend::new();
    let bf16 = kv_cache(&gpu, KvCacheDtype::Bf16, 4);
    let fp4 = kv_cache(&gpu, KvCacheDtype::Nvfp4, 4);
    fill_block(&bf16, &gpu, 1, 3);
    let header = SeqStateHeader {
        fingerprint: [0u8; DIGEST_BYTES],
        seq_len: 10,
        block_size: 16,
        num_blocks: 1,
        flags: 0,
        prompt_len: 10,
        mtp_num_blocks: 0,
        mtp_hidden_selected_row: 0,
        mtp_seq_len: 0,
        session_hash: 0,
        mtp_last_num_drafted: 0,
        mtp_hidden_position: 0,
        layout_digest: [0u8; DIGEST_BYTES],
        tokens: (0..10).collect(),
        mtp_last_drafts: vec![],
    };
    let mut blob = Vec::new();
    let src = kv_block_regions(&bf16, &[1], SectionKind::KvK, SectionKind::KvV).unwrap();
    engine::capture(&gpu, &header, &src, &mut blob).unwrap();
    let dst = kv_block_regions(&fp4, &[2], SectionKind::KvK, SectionKind::KvV).unwrap();
    let before = fp4.read_block(0, 2, &gpu).unwrap();
    let mut r: &[u8] = &blob;
    assert!(engine::apply(&gpu, BlobReader::open(&mut r).unwrap(), &dst).is_err());
    assert_eq!(fp4.read_block(0, 2, &gpu).unwrap(), before);
}

#[test]
fn numerics_env_keeps_selectors_and_drops_paths_and_its_own_flag() {
    let vars = [
        ("ATLAS_QWEN4_QSA_SCORE_V2", "1"),
        ("ATLAS_HOME", "/x"),
        ("ATLAS_LOG_LEVEL", "debug"),
        ("ATLAS_QWEN4_SEQSTATE", "1"),
        ("ATLAS_QWEN4_SEQSTATE_CACHE_RAM_MARGIN_GIB", "8"),
        ("ATLAS_SSM_H_FP16", "1"),
        ("PATH", "/bin"),
    ]
    .map(|(k, v)| (k.to_string(), v.to_string()));
    let kept: Vec<String> = numerics_env(vars.into_iter())
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(kept, ["ATLAS_QWEN4_QSA_SCORE_V2", "ATLAS_SSM_H_FP16"]);
}
