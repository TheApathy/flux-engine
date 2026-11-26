// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Barrier};

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_comm::CommBackend;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::prefix_cache::NoPrefixCaching;

use super::TransformerModel;
use crate::layers::MtpQuantization;
use crate::traits::{EP_CMD_VERIFY_KGAMMA, EP_VERIFY_KGAMMA_ABORT, Model};
use crate::weight_map::DenseWeight;

struct InMemoryRankComm {
    rank: usize,
    world_size: usize,
    gpu: Arc<MockGpuBackend>,
    bcast_tx: Vec<SyncSender<Vec<u8>>>,
    bcast_rx: Vec<parking_lot::Mutex<Receiver<Vec<u8>>>>,
    p2p_tx: Vec<SyncSender<Vec<u8>>>,
    p2p_rx: Vec<parking_lot::Mutex<Receiver<Vec<u8>>>>,
    barrier: Arc<Barrier>,
}

impl InMemoryRankComm {
    fn make_world(world_size: usize, gpus: &[Arc<MockGpuBackend>]) -> Vec<Arc<dyn CommBackend>> {
        let mut bcast_txs = vec![Vec::new(); world_size];
        let mut bcast_rxs: Vec<Vec<Option<Receiver<Vec<u8>>>>> = (0..world_size)
            .map(|_| (0..world_size).map(|_| None).collect())
            .collect();
        let mut p2p_txs = vec![Vec::new(); world_size];
        let mut p2p_rxs: Vec<Vec<Option<Receiver<Vec<u8>>>>> = (0..world_size)
            .map(|_| (0..world_size).map(|_| None).collect())
            .collect();

        for src in 0..world_size {
            for dst in 0..world_size {
                let (btx, brx) = sync_channel(128);
                bcast_txs[src].push(btx);
                bcast_rxs[dst][src] = Some(brx);

                let (ptx, prx) = sync_channel(128);
                p2p_txs[src].push(ptx);
                p2p_rxs[dst][src] = Some(prx);
            }
        }

        let barrier = Arc::new(Barrier::new(world_size));
        (0..world_size)
            .map(|rank| {
                Arc::new(Self {
                    rank,
                    world_size,
                    gpu: Arc::clone(&gpus[rank]),
                    bcast_tx: bcast_txs[rank].clone(),
                    bcast_rx: bcast_rxs[rank]
                        .iter_mut()
                        .map(|rx| parking_lot::Mutex::new(rx.take().unwrap()))
                        .collect(),
                    p2p_tx: p2p_txs[rank].clone(),
                    p2p_rx: p2p_rxs[rank]
                        .iter_mut()
                        .map(|rx| parking_lot::Mutex::new(rx.take().unwrap()))
                        .collect(),
                    barrier: Arc::clone(&barrier),
                }) as Arc<dyn CommBackend>
            })
            .collect()
    }
}

impl CommBackend for InMemoryRankComm {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }
    fn all_gather(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }
    fn reduce_scatter(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        if self.rank == root {
            let mut payload = vec![0u8; bytes];
            if bytes > 0 {
                self.gpu.copy_d2h(DevicePtr(ptr), &mut payload)?;
            }
            for dst in 0..self.world_size {
                if dst != self.rank {
                    self.bcast_tx[dst]
                        .send(payload.clone())
                        .map_err(|_| anyhow::anyhow!("bcast send closed"))?;
                }
            }
        } else {
            let payload = self.bcast_rx[root]
                .lock()
                .recv_timeout(std::time::Duration::from_secs(2))
                .map_err(|e| {
                    anyhow::anyhow!(
                        "bcast recv timeout/closed on rank {} waiting for root {root} ({bytes}B): {e}",
                        self.rank
                    )
                })?;
            anyhow::ensure!(
                payload.len() == bytes,
                "bcast size mismatch on rank {}: expected {bytes}, got {}",
                self.rank,
                payload.len()
            );
            if bytes > 0 {
                self.gpu.copy_h2d(&payload, DevicePtr(ptr))?;
            }
        }
        Ok(())
    }
    fn barrier(&self) -> Result<()> {
        self.barrier.wait();
        Ok(())
    }
    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, _stream: u64) -> Result<()> {
        let mut payload = vec![0u8; bytes];
        if bytes > 0 {
            self.gpu.copy_d2h(DevicePtr(ptr), &mut payload)?;
        }
        self.p2p_tx[dest_rank]
            .send(payload)
            .map_err(|_| anyhow::anyhow!("p2p send closed"))
    }
    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, _stream: u64) -> Result<()> {
        let payload = self.p2p_rx[src_rank]
            .lock()
            .recv_timeout(std::time::Duration::from_secs(2))
            .map_err(|e| {
                anyhow::anyhow!(
                    "p2p recv timeout/closed on rank {} waiting for src {src_rank} ({bytes}B): {e}",
                    self.rank
                )
            })?;
        anyhow::ensure!(
            payload.len() == bytes,
            "p2p size mismatch on rank {} from {src_rank}: expected {bytes}, got {}",
            self.rank,
            payload.len()
        );
        if bytes > 0 {
            self.gpu.copy_h2d(&payload, DevicePtr(ptr))?;
        }
        Ok(())
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world_size
    }
}

fn build_synthetic_rank_model(
    mut cfg: ModelConfig,
    gpu: Arc<MockGpuBackend>,
    comm: Arc<dyn CommBackend>,
    max_batch_tokens: usize,
    max_seq_len: usize,
    max_batch_size: usize,
) -> TransformerModel {
    cfg.model_type = "qwen3_next".to_string();
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.moe_intermediate_size = 64;
    cfg.vocab_size = 128;
    cfg.num_hidden_layers = 0;
    cfg.layer_types = Vec::new();
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 16;
    cfg.hc_count = 0;
    cfg.hc_mult = 0;

    let embed = DenseWeight {
        weight: gpu.alloc(cfg.vocab_size * cfg.hidden_size * 2).unwrap(),
    };
    let final_norm = DenseWeight {
        weight: gpu.alloc(cfg.hidden_size * 2).unwrap(),
    };
    let lm_head = DenseWeight {
        weight: gpu.alloc(cfg.vocab_size * cfg.hidden_size * 2).unwrap(),
    };
    let buffers = BufferArena::new(
        &cfg,
        max_batch_tokens,
        max_seq_len,
        16,
        max_batch_size,
        gpu.as_ref(),
    )
    .unwrap();
    let kv_cfg = KvCacheConfig {
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 16,
        block_size: 16,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: Vec::new(),
        layer_dims: Vec::new(),
        cache_blocks_per_seq: None,
    };
    let kv_cache = PagedKvCache::new(kv_cfg, 64, gpu.as_ref()).unwrap();
    let is_head = cfg.tp_rank == 0 && cfg.ep_rank == 0 && cfg.pp_rank == 0;
    TransformerModel::new(
        cfg,
        embed,
        final_norm,
        None,
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        None,
        lm_head,
        None,
        vec![],
        buffers,
        kv_cache,
        vec![],
        None,
        None,
        gpu as Arc<dyn GpuBackend>,
        max_seq_len,
        max_batch_size,
        MtpQuantization::Bf16,
        true,
        Box::new(NoPrefixCaching),
        0,
        Some(comm),
        is_head,
        4,
        None,
        0,
        0,
    )
    .unwrap()
}

#[test]
fn pack_and_unpack_ep_slot_meta_roundtrips_all_flags() {
    for slot in [0usize, 1, 7, 127, 4095] {
        for qsa in [false, true] {
            for fresh in [false, true] {
                let packed = TransformerModel::pack_ep_slot_meta(slot, qsa, fresh);
                assert_eq!(
                    TransformerModel::unpack_ep_slot_meta(packed),
                    (slot, qsa, fresh)
                );
            }
        }
    }
}

#[test]
fn tp2_multi_slot_prefill_decode_batch_and_selective_free_lockstep() {
    let gpus: Vec<Arc<MockGpuBackend>> = (0..2).map(|_| Arc::new(MockGpuBackend::new())).collect();
    let comms = InMemoryRankComm::make_world(2, &gpus);

    let mut cfg0 = ModelConfig::qwen3_next_80b_nvfp4();
    cfg0.tp_world_size = 2;
    cfg0.tp_rank = 0;
    let mut cfg1 = cfg0.clone();
    cfg1.tp_rank = 1;

    let model0 =
        build_synthetic_rank_model(cfg0, Arc::clone(&gpus[0]), comms[0].clone(), 16, 128, 4);
    let model1 =
        build_synthetic_rank_model(cfg1, Arc::clone(&gpus[1]), comms[1].clone(), 16, 128, 4);
    drop(comms);

    // Also test scratch-chunked `ep_broadcast_tokens` with a payload larger than `scratch / 4`.
    let big_tokens: Vec<u32> = (0..(model0.buffers.sizes().scratch / 4 + 37))
        .map(|i| (i as u32) ^ 0x5A5A)
        .collect();
    let big_len = big_tokens.len();

    let worker = std::thread::spawn(move || {
        let received_big = model1.ep_broadcast_tokens(&vec![0u32; big_len]).unwrap();
        let mut wseq = model1.alloc_sequence().unwrap();
        assert_eq!(wseq.slot_idx, 0);
        while model1.ep_worker_step(&mut wseq).unwrap() {}
        let extra_remaining = model1.worker_extra_slots.lock().len();
        let snapshot = (
            wseq.slot_idx,
            wseq.seq_len,
            wseq.qwen4_qsa_required,
            wseq.tokens.last().copied(),
        );
        let _ = model1.free_sequence(&mut wseq);
        (received_big, snapshot, extra_remaining)
    });

    let sent_big = model0.ep_broadcast_tokens(&big_tokens).unwrap();
    assert_eq!(sent_big, big_tokens);

    let mut seq0 = model0.alloc_sequence().unwrap();
    let mut seq1 = model0.alloc_sequence().unwrap();
    assert_eq!((seq0.slot_idx, seq1.slot_idx), (0, 1));
    seq0.qwen4_qsa_required = true;
    seq1.qwen4_qsa_required = false;

    // 1. Prefill slot 1 FIRST (out-of-order slot activation) across 2 chunks.
    let prompt1 = [10u32, 11, 12, 13, 14, 15];
    for (start, len, last) in [(0usize, 4usize, false), (4usize, 2usize, true)] {
        model0.ep_broadcast_cmd(0xFFFFFFF0).unwrap();
        model0.ep_broadcast_cmd(len as u32).unwrap();
        model0.ep_broadcast_cmd(start as u32).unwrap();
        model0.ep_broadcast_cmd(prompt1.len() as u32).unwrap();
        let _ = model0.ep_broadcast_tokens(&prompt1).unwrap();
        let _ = model0
            .prefill_chunk(&prompt1, &mut seq1, start, len, last, 0)
            .unwrap();
    }
    assert_eq!(seq1.seq_len, 6);

    // 2. Prefill slot 0 (`qwen4_qsa_required = true`) via single-shot `prefill`.
    let prompt0 = [20u32, 21, 22, 23];
    model0.ep_broadcast_cmd(0xFFFFFFF0).unwrap();
    model0.ep_broadcast_cmd(prompt0.len() as u32).unwrap();
    model0.ep_broadcast_cmd(0).unwrap();
    model0.ep_broadcast_cmd(prompt0.len() as u32).unwrap();
    let _ = model0.ep_broadcast_tokens(&prompt0).unwrap();
    let _ = model0.prefill(&prompt0, &mut seq0, 0).unwrap();
    assert_eq!(seq0.seq_len, 4);

    // 3. C=2 lockstep `decode_batch` (previously deadlocked NCCL & clobbered row 0).
    let _ = model0
        .decode_batch(&[30, 40], &mut [&mut seq0, &mut seq1], 0)
        .unwrap();
    assert_eq!((seq0.seq_len, seq1.seq_len), (5, 7));

    // 4. Verify `EP_VERIFY_KGAMMA_ABORT` (`0xFFFFFFFF`) inside `VerifyResultPending`
    // aborts only the verify frame on slot 0 without shutting down the worker!
    let verify_win = [31u32, 32, 33];
    model0.ep_broadcast_cmd(EP_CMD_VERIFY_KGAMMA).unwrap();
    model0.ep_broadcast_cmd(verify_win.len() as u32).unwrap();
    let _ = model0.ep_broadcast_tokens(&verify_win).unwrap();
    let _ = model0
        .decode_verify_dflash(&verify_win, &mut seq0, 0)
        .unwrap();
    seq0.seq_len = 5;
    seq0.tokens.truncate(5);
    model0
        .abort_verify_state(&mut seq0, verify_win.len())
        .unwrap();
    model0.ep_broadcast_cmd(EP_VERIFY_KGAMMA_ABORT).unwrap();

    // 5. Selective free of slot 1 via `0xFFFFFFF1` while slot 0 remains live!
    model0.free_sequence(&mut seq1).unwrap();
    model0.ep_broadcast_cmd(0xFFFFFFF1).unwrap();

    // 6. Single-row decode on surviving slot 0 (`n == 1` path).
    model0.ep_broadcast_cmd(35).unwrap();
    let _ = model0.decode_batch(&[35], &mut [&mut seq0], 0).unwrap();
    assert_eq!(seq0.seq_len, 6);
    assert_eq!(seq0.tokens.last().copied(), Some(35));

    // 7. Clean worker shutdown (`0xFFFFFFFF` from Idle).
    model0.ep_broadcast_cmd(0xFFFFFFFF).unwrap();
    model0.free_sequence(&mut seq0).unwrap();
    drop(model0);

    let (received_big, (wslot, wlen, wqsa, wlast), extra_remaining) = worker.join().unwrap();
    assert_eq!(received_big, big_tokens);
    assert_eq!(extra_remaining, 0);
    assert_eq!(wslot, 0);
    assert_eq!(wlen, 6);
    assert!(wqsa);
    assert_eq!(wlast, Some(35));
}

#[test]
fn pp3_three_stage_ring_prefill_decode_and_verify_lockstep() {
    let gpus: Vec<Arc<MockGpuBackend>> = (0..3).map(|_| Arc::new(MockGpuBackend::new())).collect();
    let comms = InMemoryRankComm::make_world(3, &gpus);

    let models: Vec<TransformerModel> = (0..3)
        .map(|stage| {
            let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
            cfg.pp_world_size = 3;
            cfg.pp_rank = stage;
            build_synthetic_rank_model(
                cfg,
                Arc::clone(&gpus[stage]),
                comms[stage].clone(),
                16,
                128,
                4,
            )
        })
        .collect();
    drop(comms);
    let [m0, m1, m2] = models.try_into().ok().unwrap();

    let w1 = std::thread::spawn(move || {
        let mut s = m1.alloc_sequence().unwrap();
        while m1.ep_worker_step(&mut s).unwrap() {}
        let snap = (s.seq_len, s.qwen4_qsa_required, s.tokens.clone());
        let _ = m1.free_sequence(&mut s);
        snap
    });
    let w2 = std::thread::spawn(move || {
        let mut s = m2.alloc_sequence().unwrap();
        while m2.ep_worker_step(&mut s).unwrap() {}
        let snap = (s.seq_len, s.qwen4_qsa_required, s.tokens.clone());
        let _ = m2.free_sequence(&mut s);
        snap
    });

    let mut seq = m0.alloc_sequence().unwrap();
    seq.qwen4_qsa_required = true;

    // 1. Chunked prefill across Stage 0 -> Stage 1 -> Stage 2 -> Stage 0.
    let prompt = [5u32, 6, 7, 8, 9];
    m0.ep_broadcast_cmd(0xFFFFFFF0).unwrap();
    m0.ep_broadcast_cmd(prompt.len() as u32).unwrap();
    m0.ep_broadcast_cmd(0).unwrap();
    m0.ep_broadcast_cmd(prompt.len() as u32).unwrap();
    let _ = m0.ep_broadcast_tokens(&prompt).unwrap();
    let _ = m0
        .prefill_chunk(&prompt, &mut seq, 0, prompt.len(), true, 0)
        .unwrap();
    assert_eq!(seq.seq_len, 5);

    // 2. Single-token decode across the 3-stage PP ring.
    m0.ep_broadcast_cmd(42).unwrap();
    let _ = m0.decode(42, &mut seq, 0).unwrap();
    assert_eq!(seq.seq_len, 6);

    // 3. K=gamma verify + commit (1 draft accepted -> total_accepted = 2) across PP=3.
    let win = [43u32, 44, 45];
    m0.ep_broadcast_cmd(EP_CMD_VERIFY_KGAMMA).unwrap();
    m0.ep_broadcast_cmd(win.len() as u32).unwrap();
    let _ = m0.ep_broadcast_tokens(&win).unwrap();
    let _ = m0.decode_verify_dflash(&win, &mut seq, 0).unwrap();
    seq.seq_len = 8;
    seq.tokens.truncate(8);
    m0.commit_verify_state_async(&mut seq, 2, win.len())
        .unwrap();
    m0.ep_broadcast_cmd(1).unwrap(); // num_accepted = 1 -> total_accepted = 2

    m0.ep_broadcast_cmd(0xFFFFFFFF).unwrap();
    let expected_tokens = seq.tokens.clone();
    m0.free_sequence(&mut seq).unwrap();
    drop(m0);

    let (len1, qsa1, tok1) = w1.join().unwrap();
    let (len2, qsa2, tok2) = w2.join().unwrap();
    assert_eq!((len1, len2), (8, 8));
    assert!(qsa1 && qsa2);
    assert_eq!(tok1, expected_tokens);
    assert_eq!(tok2, expected_tokens);
}
