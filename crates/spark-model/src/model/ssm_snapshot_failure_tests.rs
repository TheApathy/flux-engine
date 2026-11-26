// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

struct FailingCopyGpu {
    fail_at: Option<usize>,
    copies: AtomicUsize,
    widener_kernel: KernelHandle,
    launches: AtomicUsize,
}

impl FailingCopyGpu {
    fn new(fail_at: Option<usize>) -> Self {
        Self {
            fail_at,
            copies: AtomicUsize::new(0),
            widener_kernel: KernelHandle(0),
            launches: AtomicUsize::new(0),
        }
    }

    fn with_widener(widener_kernel: KernelHandle) -> Self {
        Self {
            fail_at: None,
            copies: AtomicUsize::new(0),
            widener_kernel,
            launches: AtomicUsize::new(0),
        }
    }

    fn copy_count(&self) -> usize {
        self.copies.load(Ordering::SeqCst)
    }

    fn launch_count(&self) -> usize {
        self.launches.load(Ordering::SeqCst)
    }
}

impl GpuBackend for FailingCopyGpu {
    fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
        Ok(DevicePtr(0x1000))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }

    fn free(&self, _ptr: DevicePtr) -> Result<()> {
        Ok(())
    }

    fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
        Ok(())
    }

    fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        Ok(())
    }

    fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        _src: DevicePtr,
        _dst: DevicePtr,
        _bytes: usize,
        _stream: u64,
    ) -> Result<()> {
        let call = self.copies.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_at == Some(call) {
            bail!("injected D2D failure at copy {call}");
        }
        Ok(())
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        self.launches.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        if module == "ssm_h_dtype" && func_name == "ssm_h_state_f16_to_f32" {
            Ok(self.widener_kernel)
        } else {
            Ok(KernelHandle(0))
        }
    }

    fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn memset_async(&self, _ptr: DevicePtr, _value: u8, _bytes: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(1 << 30)
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(1 << 30)
    }
}

fn state_pool() -> SsmStatePool {
    SsmStatePool {
        allocation_geometry: Default::default(),
        h_state_pools: vec![DevicePtr(0x1000), DevicePtr(0x2000)],
        conv_state_pools: vec![DevicePtr(0x3000), DevicePtr(0x4000)],
        h_intermediate_pools: Vec::new(),
        conv_intermediate_pools: Vec::new(),
        h_verify_scratch: DevicePtr::NULL,
        h_checkpoint_pools: Vec::new(),
        conv_checkpoint_pools: Vec::new(),
        wy17_kv_retain_pools: Vec::new(),
        wy17_gate_retain_pools: Vec::new(),
        kv_retain_bytes: 0,
        gate_retain_bytes: 0,
        h_bytes: 8,
        conv_bytes: 4,
        max_slots: 1,
        spec_slots: 2,
        num_ssm_layers: 2,
        has_mtp: false,
        compact_lazy_h: false,
        num_intermediates: 0,
        free_slots: Mutex::new(Vec::new()),
    }
}

fn snapshot_pool(free_slots: Vec<usize>) -> SsmSnapshotPool {
    SsmSnapshotPool {
        h_snapshots: vec![DevicePtr(0x5000), DevicePtr(0x6000)],
        conv_snapshots: vec![DevicePtr(0x7000), DevicePtr(0x8000)],
        free_slots: Mutex::new(free_slots),
        num_slots: 1,
        h_bytes: 8,
        conv_bytes: 4,
        num_ssm_layers: 2,
        // Unused by FP32 tests below (`h_is_f16 = false`); FP16 widening is
        // exercised via `SsmSnapshotPool::new` in `enabled_pool_resolves_f16_widener_and_saves_f16_h_state`.
        h_f16_to_f32_k: KernelHandle(0),
        session_tags: Mutex::new(HashMap::new()),
        decode_h_snapshots: Vec::new(),
        decode_conv_snapshots: Vec::new(),
        decode_ring_slots: 0,
        decode_max_seqs: 0,
        ple_conv_layers: Vec::new(),
        ple_conv_snapshots: Vec::new(),
        decode_ple_conv_snapshots: Vec::new(),
        ple_conv_bytes: 0,
        slot_history: None,
    }
}

#[test]
fn every_copy_failure_restores_slot_and_clears_owner() {
    let main = state_pool();
    for fail_at in 1..=4 {
        let gpu = FailingCopyGpu::new(Some(fail_at));
        let snapshots = snapshot_pool(vec![0]);
        snapshots.session_tags.lock().insert(0, 0xAABB);

        let error = snapshots
            .save(0, 0xCCDD, false, &main, &gpu, 7)
            .unwrap_err();

        assert!(error.to_string().contains(&format!("copy {fail_at}")));
        assert_eq!(gpu.copy_count(), fail_at);
        assert_eq!(*snapshots.free_slots.lock(), vec![0]);
        assert!(!snapshots.session_tags.lock().contains_key(&0));
    }
}

#[test]
fn successful_save_and_free_keep_existing_semantics() {
    let main = state_pool();
    let gpu = FailingCopyGpu::new(None);
    let snapshots = snapshot_pool(vec![0]);

    assert_eq!(
        snapshots.save(0, 0xCCDD, false, &main, &gpu, 7).unwrap(),
        Some(0)
    );
    assert_eq!(gpu.copy_count(), 4);
    assert!(snapshots.free_slots.lock().is_empty());
    assert_eq!(snapshots.session_tags.lock().get(&0), Some(&0xCCDD));

    snapshots.free(0);
    assert_eq!(*snapshots.free_slots.lock(), vec![0]);
    assert!(!snapshots.session_tags.lock().contains_key(&0));
}

#[test]
fn disabled_or_exhausted_pool_still_returns_none_without_copying() {
    let main = state_pool();
    let gpu = FailingCopyGpu::new(Some(1));
    let mut disabled = snapshot_pool(vec![0]);
    disabled.num_slots = 0;
    assert_eq!(disabled.save(0, 1, false, &main, &gpu, 7).unwrap(), None);

    let exhausted = snapshot_pool(Vec::new());
    assert_eq!(exhausted.save(0, 1, false, &main, &gpu, 7).unwrap(), None);
    assert_eq!(gpu.copy_count(), 0);
}

#[test]
fn ple_conv_save_restore_and_decode_rollback_round_trip_with_failure_recovery() {
    let main = state_pool();
    let gpu = FailingCopyGpu::new(None);
    let mut snapshots = snapshot_pool(vec![0]);
    snapshots.decode_h_snapshots = vec![DevicePtr(0x11_000), DevicePtr(0x12_000)];
    snapshots.decode_conv_snapshots = vec![DevicePtr(0x13_000), DevicePtr(0x14_000)];
    snapshots.decode_ring_slots = 2;
    snapshots.decode_max_seqs = 1;
    snapshots.ple_conv_layers = vec![PleConvLayerBinding {
        conv_state: DevicePtr(0x20_000),
        conv_checkpoint: DevicePtr(0x21_000),
        max_slots: 1,
    }];
    snapshots.ple_conv_snapshots = vec![DevicePtr(0x22_000)];
    snapshots.decode_ple_conv_snapshots = vec![DevicePtr(0x23_000)];
    snapshots.ple_conv_bytes = 184_320;

    // 2 SSM layers (h + conv = 4 copies) + 1 PLE layer (1 copy) = 5 D2D copies on save.
    assert_eq!(
        snapshots.save(0, 0xBEEF, false, &main, &gpu, 9).unwrap(),
        Some(0)
    );
    assert_eq!(gpu.copy_count(), 5);

    // Restore copies 4 SSM + 1 PLE live + 1 PLE checkpoint mirror = 6 D2D copies.
    snapshots.restore(0, 0, &main, &gpu, 9).unwrap();
    assert_eq!(gpu.copy_count(), 11);

    // Decode ring save + restore: respects `replay_ssm_enabled()` for SSM h_state,
    // while ALWAYS saving/restoring PLE `conv_state` (+ checkpoint mirror on restore).
    let ssm_decode_copies = if crate::layers::qwen3_ssm::replay_ssm_enabled() {
        2
    } else {
        4
    };
    snapshots.save_decode(0, 1, &main, &gpu, 9).unwrap();
    assert_eq!(gpu.copy_count(), 11 + ssm_decode_copies + 1);
    snapshots.restore_decode(0, 1, &main, &gpu, 9).unwrap();
    assert_eq!(
        gpu.copy_count(),
        11 + (ssm_decode_copies + 1) + (ssm_decode_copies + 2)
    );

    snapshots.free(0);

    // Inject failure right on the 5th copy (the PLE conv_state copy): slot 0 must be freed!
    let failing_gpu = FailingCopyGpu::new(Some(5));
    let err = snapshots
        .save(0, 0xDEAD, false, &main, &failing_gpu, 9)
        .unwrap_err();
    assert!(err.to_string().contains("copy 5"));
    assert_eq!(*snapshots.free_slots.lock(), vec![0]);
    assert!(snapshots.session_tags.lock().is_empty());
}

#[test]
fn enabled_pool_resolves_f16_widener_and_saves_f16_h_state() {
    let main = state_pool();
    let gpu = FailingCopyGpu::with_widener(KernelHandle(0x42));

    let disabled = SsmSnapshotPool::new(0, 8, 4, 2, 0, 0, &gpu).unwrap();
    assert_eq!(disabled.h_f16_to_f32_k.0, 0);

    let enabled = SsmSnapshotPool::new(1, 8, 4, 2, 0, 0, &gpu).unwrap();
    assert_eq!(enabled.h_f16_to_f32_k.0, 0x42);
    assert_eq!(
        enabled.save(0, 0x1234, true, &main, &gpu, 7).unwrap(),
        Some(0)
    );
    assert_eq!(gpu.launch_count(), 2);
    assert_eq!(gpu.copy_count(), 2);
}

#[test]
fn attach_ple_rejects_enabling_regions_when_ssm_slabs_are_unallocated() {
    let gpu = FailingCopyGpu::new(None);
    let binding = || {
        vec![PleConvLayerBinding {
            conv_state: DevicePtr(0x9000),
            conv_checkpoint: DevicePtr(0xA000),
            max_slots: 1,
        }]
    };

    // Pool with SSM layers but Marconi disabled (num_slots == 0, h_snapshots empty).
    let mut no_marconi = SsmSnapshotPool::new(0, 8, 4, 2, 2, 1, &gpu).unwrap();
    let err = no_marconi
        .attach_ple_conv_layers(binding(), 16, 2, 1, 4, &gpu)
        .unwrap_err();
    assert!(
        err.to_string().contains("unallocated SSM Marconi slabs"),
        "unexpected error: {err:#}"
    );
    assert!(!no_marconi.is_enabled());

    // Pool with SSM layers but decode ring disabled (decode_ring_slots == 0, decode_h_snapshots empty).
    let mut no_decode = SsmSnapshotPool::new(4, 8, 4, 2, 0, 1, &gpu).unwrap();
    let err = no_decode
        .attach_ple_conv_layers(binding(), 16, 2, 1, 4, &gpu)
        .unwrap_err();
    assert!(
        err.to_string().contains("unallocated SSM decode slabs"),
        "unexpected error: {err:#}"
    );
    assert!(!no_decode.decode_rollback_enabled());
}
