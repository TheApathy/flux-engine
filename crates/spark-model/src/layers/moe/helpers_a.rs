// SPDX-License-Identifier: AGPL-3.0-only

//! Setters + transposes + transpose_for_prefill_unified_inner.

use super::*;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::{RoutedOriginalGroups, RoutedProjection};

const TS4E128_ALIGN_BYTES: usize = 16;

/// Number of `u32` `page_bases` entries for `num_bkts` buckets (always a multiple of 4
/// so `bkt_off` starts 16-byte aligned; equals `4` for `num_bkts <= 511`, e.g. Qwen4's 400).
#[inline]
pub(crate) const fn ts4e128_page_bases_len(num_bkts: usize) -> usize {
    ((num_bkts >> 7) + 4) & !3
}

/// Byte layout metadata for one contiguous 16B-aligned Atlas-TS4E-128 per-expert scale blob.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ts4e128Layout {
    pub(crate) num_k32: usize,
    pub(crate) n_tiles: usize,
    pub(crate) num_bkts: usize,
    pub(crate) nibbles_bytes: usize,
    pub(crate) page_bases_len: usize,
    pub(crate) page_bases_off: usize,
    pub(crate) bkt_off_off: usize,
    pub(crate) bkt_off_bytes_aligned: usize,
    pub(crate) exc_off: usize,
}

/// Computes the exact section offsets of one Atlas-TS4E-128 scale blob for `[K/16, N]`.
pub(crate) fn ts4e128_layout(k: usize, n: usize) -> Result<Ts4e128Layout> {
    anyhow::ensure!(
        k > 0 && k.is_multiple_of(32) && n > 0 && n.is_multiple_of(128),
        "Atlas-TS4E-128 requires K > 0 (K % 32 == 0) and N > 0 (N % 128 == 0), got K={k}, N={n}"
    );
    let num_k32 = k / 32;
    let n_tiles = n / 128;
    let num_bkts = num_k32
        .checked_mul(n_tiles)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 bucket count overflow"))?;
    let nibbles_bytes = num_k32
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 nibbles byte size overflow"))?;
    let page_bases_len = ts4e128_page_bases_len(num_bkts);
    let page_bases_off = nibbles_bytes
        .checked_add(n)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 page_bases offset overflow"))?;
    let page_bases_bytes = page_bases_len
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 page_bases byte size overflow"))?;
    let bkt_off_off = page_bases_off
        .checked_add(page_bases_bytes)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 bkt_off offset overflow"))?;
    let bkt_off_bytes = num_bkts
        .checked_add(1)
        .and_then(|entries| entries.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 bkt_off byte size overflow"))?;
    let bkt_off_bytes_aligned = bkt_off_bytes
        .checked_add(TS4E128_ALIGN_BYTES - 1)
        .map(|bytes| bytes & !(TS4E128_ALIGN_BYTES - 1))
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 bkt_off alignment overflow"))?;
    let exc_off = bkt_off_off
        .checked_add(bkt_off_bytes_aligned)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 exceptions offset overflow"))?;
    Ok(Ts4e128Layout {
        num_k32,
        n_tiles,
        num_bkts,
        nibbles_bytes,
        page_bases_len,
        page_bases_off,
        bkt_off_off,
        bkt_off_bytes_aligned,
        exc_off,
    })
}

/// Returns the 16-byte-aligned total byte size of one Atlas-TS4E-128 blob with `num_exc` exceptions.
pub(crate) fn ts4e128_blob_bytes(k: usize, n: usize, num_exc: usize) -> Result<usize> {
    let layout = ts4e128_layout(k, n)?;
    let exc_bytes_aligned = num_exc
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(TS4E128_ALIGN_BYTES - 1))
        .map(|bytes| bytes & !(TS4E128_ALIGN_BYTES - 1))
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 exceptions byte size overflow"))?;
    layout
        .exc_off
        .checked_add(exc_bytes_aligned)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 total blob byte size overflow"))
}

/// Pure-Rust reference Atlas-TS4E-128 encoder (byte-identical to GPU `atlas_ts4e128_encode.cu`).
///
/// Accepts untransposed row-major `[N, K/16]` raw E4M3 scale bytes and returns the
/// 16-byte-aligned contiguous TS4E-128 blob alongside its exact exception count.
#[allow(dead_code)]
pub(crate) fn ts4e128_encode_host(
    src_row_major: &[u8],
    k: usize,
    n: usize,
) -> Result<(Vec<u8>, usize)> {
    let layout = ts4e128_layout(k, n)?;
    let k16_count = k / 16;
    let expected_len = n
        .checked_mul(k16_count)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 input scale size overflow"))?;
    anyhow::ensure!(
        src_row_major.len() == expected_len,
        "TS4E-128 encode expected {expected_len} input bytes for [N={n}, K/16={k16_count}], got {}",
        src_row_major.len()
    );

    let mut bases = vec![0u8; n];
    let mut total_exc = 0usize;
    for (col, base_slot) in bases.iter_mut().enumerate() {
        let mut hist = [0u32; 256];
        let row_slice = &src_row_major[col * k16_count..(col + 1) * k16_count];
        for &val in row_slice {
            hist[usize::from(val)] += 1;
        }
        let mut win: u32 = hist[..15].iter().sum();
        let mut best_cnt = win;
        let mut best_base = 0u8;
        for b in 1usize..=241 {
            win = win + hist[b + 14] - hist[b - 1];
            if win > best_cnt {
                best_cnt = win;
                best_base = b as u8;
            }
        }
        *base_slot = best_base;
        total_exc += k16_count - best_cnt as usize;
    }

    let blob_bytes = ts4e128_blob_bytes(k, n, total_exc)?;
    let mut blob = vec![0u8; blob_bytes];
    blob[layout.nibbles_bytes..layout.nibbles_bytes + n].copy_from_slice(&bases);

    let mut exc_cursor = 0usize;
    let mut page_base = 0usize;
    for k32 in 0..layout.num_k32 {
        for nt in 0..layout.n_tiles {
            let bkt = k32 * layout.n_tiles + nt;
            if bkt.is_multiple_of(128) {
                page_base = exc_cursor;
                let pb_u32 = u32::try_from(page_base)
                    .map_err(|_| anyhow::anyhow!("TS4E-128 page_base exceeds u32"))?;
                let off = layout.page_bases_off + (bkt >> 7) * 4;
                blob[off..off + 4].copy_from_slice(&pb_u32.to_le_bytes());
            }
            let rel_u16 = u16::try_from(exc_cursor - page_base)
                .map_err(|_| anyhow::anyhow!("TS4E-128 relative bucket offset exceeds u16"))?;
            let bkt_pos = layout.bkt_off_off + bkt * 2;
            blob[bkt_pos..bkt_pos + 2].copy_from_slice(&rel_u16.to_le_bytes());

            for k_bit in 0usize..2 {
                let k16 = k32 * 2 + k_bit;
                for ln in 0usize..128 {
                    let col = nt * 128 + ln;
                    let val = src_row_major[col * k16_count + k16];
                    let base = bases[col];
                    // Carry-safety: base <= 241 implies base + 14 <= 255 (zero u8 overflow).
                    let (code, is_exc) = if (base..=base + 14).contains(&val) {
                        (val - base, false)
                    } else {
                        (15u8, true)
                    };
                    let nib_idx = k32 * n + col;
                    if k_bit == 0 {
                        blob[nib_idx] |= code;
                    } else {
                        blob[nib_idx] |= code << 4;
                    }
                    if is_exc {
                        let local_pos = ((k_bit << 7) | ln) as u16;
                        let entry = (u16::from(val) << 8) | local_pos;
                        let exc_pos = layout.exc_off + exc_cursor * 2;
                        blob[exc_pos..exc_pos + 2].copy_from_slice(&entry.to_le_bytes());
                        exc_cursor += 1;
                    }
                }
            }
        }
    }

    anyhow::ensure!(
        exc_cursor == total_exc,
        "TS4E-128 exception count mismatch: packed {exc_cursor}, expected {total_exc}"
    );
    if layout.num_bkts.is_multiple_of(128) {
        page_base = exc_cursor;
        let pb_u32 = u32::try_from(page_base)
            .map_err(|_| anyhow::anyhow!("TS4E-128 terminal page_base exceeds u32"))?;
        let off = layout.page_bases_off + (layout.num_bkts >> 7) * 4;
        blob[off..off + 4].copy_from_slice(&pb_u32.to_le_bytes());
    }
    let end_rel = u16::try_from(exc_cursor - page_base)
        .map_err(|_| anyhow::anyhow!("TS4E-128 terminal relative bucket offset exceeds u16"))?;
    let end_pos = layout.bkt_off_off + layout.num_bkts * 2;
    blob[end_pos..end_pos + 2].copy_from_slice(&end_rel.to_le_bytes());

    Ok((blob, total_exc))
}

/// Pure-Rust reference Atlas-TS4E-128 decoder producing transposed flat `[K/16, N]` E4M3 scales.
#[allow(dead_code)]
pub(crate) fn ts4e128_decode_host(blob: &[u8], k: usize, n: usize) -> Result<Vec<u8>> {
    let layout = ts4e128_layout(k, n)?;
    anyhow::ensure!(
        blob.len() >= layout.exc_off && blob.len().is_multiple_of(TS4E128_ALIGN_BYTES),
        "TS4E-128 blob length {} invalid (min {}, 16B aligned)",
        blob.len(),
        layout.exc_off
    );

    let read_u32 = |off: usize| -> Result<u32> {
        let bytes: [u8; 4] = blob
            .get(off..off + 4)
            .ok_or_else(|| anyhow::anyhow!("TS4E-128 u32 read out of bounds at {off}"))?
            .try_into()
            .map_err(|_| anyhow::anyhow!("TS4E-128 u32 slice conversion failed"))?;
        Ok(u32::from_le_bytes(bytes))
    };
    let read_u16 = |off: usize| -> Result<u16> {
        let bytes: [u8; 2] = blob
            .get(off..off + 2)
            .ok_or_else(|| anyhow::anyhow!("TS4E-128 u16 read out of bounds at {off}"))?
            .try_into()
            .map_err(|_| anyhow::anyhow!("TS4E-128 u16 slice conversion failed"))?;
        Ok(u16::from_le_bytes(bytes))
    };

    let k16_count = k / 16;
    let out_len = k16_count
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 decoded scale size overflow"))?;
    let mut out = vec![0u8; out_len];

    for bkt in 0..layout.num_bkts {
        let k32 = bkt / layout.n_tiles;
        let nt = bkt % layout.n_tiles;
        let beg = read_u32(layout.page_bases_off + (bkt >> 7) * 4)? as usize
            + usize::from(read_u16(layout.bkt_off_off + bkt * 2)?);
        let end = read_u32(layout.page_bases_off + ((bkt + 1) >> 7) * 4)? as usize
            + usize::from(read_u16(layout.bkt_off_off + (bkt + 1) * 2)?);
        anyhow::ensure!(
            beg <= end && layout.exc_off + end * 2 <= blob.len(),
            "TS4E-128 bucket {bkt} exception range [{beg}..{end}) out of bounds"
        );

        for k_bit in 0usize..2 {
            let k16 = k32 * 2 + k_bit;
            for ln in 0usize..128 {
                let col = nt * 128 + ln;
                let nib_pair = blob[k32 * n + col];
                let code = if k_bit == 0 {
                    nib_pair & 0x0f
                } else {
                    nib_pair >> 4
                };
                let scale_val = if code != 15 {
                    let base = blob[layout.nibbles_bytes + col];
                    anyhow::ensure!(
                        base <= 241,
                        "TS4E-128 column {col} base {base} exceeds carry-safe max 241"
                    );
                    base + code
                } else {
                    let target = ((k_bit << 7) | ln) as u8;
                    let mut lo = beg;
                    let mut hi = end;
                    while lo < hi {
                        let mid = lo + ((hi - lo) >> 1);
                        let pos = (read_u16(layout.exc_off + mid * 2)? & 0xff) as u8;
                        if pos < target {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    anyhow::ensure!(
                        lo < end,
                        "TS4E-128 missing exception for bkt={bkt} local_pos={target}"
                    );
                    let entry = read_u16(layout.exc_off + lo * 2)?;
                    anyhow::ensure!(
                        (entry & 0xff) as u8 == target,
                        "TS4E-128 unsorted/missing exception for bkt={bkt} local_pos={target}"
                    );
                    (entry >> 8) as u8
                };
                out[k16 * n + col] = scale_val;
            }
        }
    }

    Ok(out)
}

struct Ts4eProjectionResult {
    table: ExpertPtrTable,
    scale_slab_bytes: usize,
    total_exc: u64,
}

#[allow(clippy::too_many_arguments)]
fn transpose_packed_and_encode_ts4e_projection(
    experts: &[ExpertWeight],
    proj: impl Fn(&ExpertWeight) -> &QuantizedWeight,
    src_table: &ExpertPtrTable,
    packed_slab: DevicePtr,
    first_packed_offset: usize,
    packed_stride: usize,
    ts4e_scale_slabs: &mut Vec<DevicePtr>,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    transpose_k: KernelHandle,
    stream: u64,
) -> Result<Ts4eProjectionResult> {
    let num_experts = experts.len();
    anyhow::ensure!(
        num_experts > 0,
        "Atlas-TS4E-128 requires at least one expert"
    );
    let num_experts_u32 = u32::try_from(num_experts)
        .map_err(|_| anyhow::anyhow!("TS4E-128 expert count exceeds u32"))?;
    let n_u32 = u32::try_from(n).map_err(|_| anyhow::anyhow!("TS4E-128 N exceeds u32"))?;
    let k_u32 = u32::try_from(k).map_err(|_| anyhow::anyhow!("TS4E-128 K exceeds u32"))?;

    // 1. Build destination packed pointer table inside the packed-only slab and transpose [N, K/2] -> [K/2, N] on GPU.
    // Pack only non-null (local) experts densely so EP>1 slabs stay 1/ep_world_size in VRAM.
    let mut dst_packed_bytes = Vec::with_capacity(num_experts * 8);
    let mut scale2_bytes = Vec::with_capacity(num_experts * 4);
    let mut local_slot = 0usize;
    for expert in experts {
        let w = proj(expert);
        let dst_ptr = if w.is_null() {
            DevicePtr::NULL
        } else {
            let off = first_packed_offset
                .checked_add(
                    packed_stride
                        .checked_mul(local_slot)
                        .ok_or_else(|| anyhow::anyhow!("TS4E-128 packed offset overflow"))?,
                )
                .ok_or_else(|| anyhow::anyhow!("TS4E-128 packed offset overflow"))?;
            local_slot += 1;
            packed_slab.offset(off)
        };
        dst_packed_bytes.extend_from_slice(&dst_ptr.0.to_le_bytes());
        scale2_bytes.extend_from_slice(&w.weight_scale_2.to_le_bytes());
    }
    let packed_ptrs = gpu.alloc(dst_packed_bytes.len())?;
    gpu.copy_h2d(&dst_packed_bytes, packed_ptrs)?;
    let scale2_vals = gpu.alloc(scale2_bytes.len())?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    ops::moe_transpose_u8_batched(
        gpu,
        transpose_k,
        src_table.packed_ptrs,
        packed_ptrs,
        n_u32,
        k_u32 / 2,
        num_experts_u32,
        stream,
    )?;

    // 2. Two-pass GPU Atlas-TS4E-128 scale encoding from original [N, K/16] flat scales.
    let count_k = gpu.kernel("atlas_ts4e128_encode", "atlas_ts4e128_count_batched")?;
    let pack_k = gpu.kernel("atlas_ts4e128_encode", "atlas_ts4e128_pack_batched")?;
    let bases_bytes = num_experts
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 bases staging size overflow"))?
        .max(TS4E128_ALIGN_BYTES);
    let exc_totals_bytes = num_experts
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("TS4E-128 exc_totals staging size overflow"))?
        .max(TS4E128_ALIGN_BYTES);

    let bases_tmp = match gpu.alloc(bases_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            match gpu.synchronize(stream) {
                Ok(()) => {
                    let _ = gpu.free(scale2_vals);
                    let _ = gpu.free(packed_ptrs);
                }
                Err(sync_err) => tracing::error!(
                    "TS4E-128 staging alloc error drain failed; leaking ptr tables to avoid GPU UAF: {sync_err:#}"
                ),
            }
            return Err(err);
        }
    };
    let exc_totals_dev = match gpu.alloc(exc_totals_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            let _ = gpu.free(bases_tmp);
            match gpu.synchronize(stream) {
                Ok(()) => {
                    let _ = gpu.free(scale2_vals);
                    let _ = gpu.free(packed_ptrs);
                }
                Err(sync_err) => tracing::error!(
                    "TS4E-128 exc_totals alloc error drain failed; leaking ptr tables to avoid GPU UAF: {sync_err:#}"
                ),
            }
            return Err(err);
        }
    };

    let mut dst_offsets_dev = DevicePtr::NULL;
    let encode_result: Result<(DevicePtr, usize, u64)> = (|| {
        KernelLaunch::new(gpu, count_k)
            .grid([num_experts_u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src_table.scale_ptrs)
            .arg_ptr(bases_tmp)
            .arg_ptr(exc_totals_dev)
            .arg_u32(k_u32)
            .arg_u32(n_u32)
            .arg_u32(num_experts_u32)
            .launch(stream)?;

        let mut exc_bytes = vec![0u8; num_experts * 4];
        gpu.copy_d2h_on_stream(exc_totals_dev, &mut exc_bytes, stream)?;

        let mut dst_offsets = Vec::with_capacity(num_experts);
        let mut total_scale_bytes = 0usize;
        let mut total_exc = 0u64;
        for (expert_idx, chunk) in exc_bytes.chunks_exact(4).enumerate() {
            if proj(&experts[expert_idx]).is_null() {
                dst_offsets.push(0u64);
                continue;
            }
            let exc = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize;
            total_exc += exc as u64;
            let blob_bytes = ts4e128_blob_bytes(k, n, exc)?;
            dst_offsets.push(
                u64::try_from(total_scale_bytes)
                    .map_err(|_| anyhow::anyhow!("TS4E-128 slab offset exceeds u64"))?,
            );
            total_scale_bytes = total_scale_bytes
                .checked_add(blob_bytes)
                .ok_or_else(|| anyhow::anyhow!("TS4E-128 total scale slab size overflow"))?;
        }

        let dst_slab = gpu.alloc(total_scale_bytes.max(TS4E128_ALIGN_BYTES))?;
        anyhow::ensure!(
            !dst_slab.is_null(),
            "TS4E-128 scale slab allocation is null"
        );
        ts4e_scale_slabs.push(dst_slab);

        let dst_offsets_bytes: Vec<u8> = dst_offsets
            .iter()
            .flat_map(|off| off.to_le_bytes())
            .collect();
        dst_offsets_dev = gpu.alloc(dst_offsets_bytes.len().max(TS4E128_ALIGN_BYTES))?;
        gpu.copy_h2d(&dst_offsets_bytes, dst_offsets_dev)?;

        KernelLaunch::new(gpu, pack_k)
            .grid([num_experts_u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src_table.scale_ptrs)
            .arg_ptr(bases_tmp)
            .arg_ptr(dst_slab)
            .arg_ptr(dst_offsets_dev)
            .arg_u32(k_u32)
            .arg_u32(n_u32)
            .arg_u32(num_experts_u32)
            .launch(stream)?;

        gpu.synchronize(stream)?;

        let mut scale_ptr_bytes = Vec::with_capacity(num_experts * 8);
        for (expert_idx, &off) in dst_offsets.iter().enumerate() {
            let ptr = if proj(&experts[expert_idx]).is_null() {
                DevicePtr::NULL
            } else {
                let off_usize = usize::try_from(off)
                    .map_err(|_| anyhow::anyhow!("TS4E-128 expert offset exceeds usize"))?;
                dst_slab.offset(off_usize)
            };
            scale_ptr_bytes.extend_from_slice(&ptr.0.to_le_bytes());
        }
        let scale_ptrs = gpu.alloc(scale_ptr_bytes.len())?;
        gpu.copy_h2d(&scale_ptr_bytes, scale_ptrs)?;

        Ok((scale_ptrs, total_scale_bytes, total_exc))
    })();

    // Release temporary staging buffers only once stream completion is guaranteed.
    // On error, gate all frees on `gpu.synchronize(stream)` succeeding so in-flight
    // transpose/count/pack kernels can never observe a GPU use-after-free.
    if encode_result.is_err() {
        match gpu.synchronize(stream) {
            Ok(()) => {
                if !dst_offsets_dev.is_null() {
                    let _ = gpu.free(dst_offsets_dev);
                }
                let _ = gpu.free(exc_totals_dev);
                let _ = gpu.free(bases_tmp);
                let _ = gpu.free(scale2_vals);
                let _ = gpu.free(packed_ptrs);
            }
            Err(sync_err) => tracing::error!(
                "TS4E-128 error drain failed; leaking temporary staging buffers to avoid GPU UAF: {sync_err:#}"
            ),
        }
    } else {
        if !dst_offsets_dev.is_null() {
            let _ = gpu.free(dst_offsets_dev);
        }
        let _ = gpu.free(exc_totals_dev);
        let _ = gpu.free(bases_tmp);
    }

    let (scale_ptrs, scale_slab_bytes, total_exc) = encode_result?;
    Ok(Ts4eProjectionResult {
        table: ExpertPtrTable {
            packed_ptrs,
            scale_ptrs,
            scale2_vals,
        },
        scale_slab_bytes,
        total_exc,
    })
}

pub(super) const ROUTED_TRANSPOSE_SLAB_ALIGNMENT: usize = 256;
const QWEN4_ROUTED_SLAB_LAYERS: usize = 48;
const QWEN4_ROUTED_SLAB_EXPERTS: usize = 512;
const QWEN4_ROUTED_SLAB_HIDDEN: usize = 2560;
const QWEN4_ROUTED_SLAB_INTER: usize = 640;
const QWEN4_ROUTED_SLAB_PROJECTIONS: usize = 3;
const QWEN4_ROUTED_SLAB_QUANTIZED_WEIGHTS: usize = 73_728 / QWEN4_ROUTED_SLAB_LAYERS;
const QWEN4_ROUTED_SLAB_COMPONENT_VIEWS: usize = 147_456 / QWEN4_ROUTED_SLAB_LAYERS;
const QWEN4_ROUTED_SLAB_PACKED_PAYLOAD: usize = 1_258_291_200;
const QWEN4_ROUTED_SLAB_SCALE_PAYLOAD: usize = 157_286_400;
const QWEN4_ROUTED_SLAB_PAYLOAD: usize = 1_415_577_600;
const QWEN4_ROUTED_SLAB_ALLOCATION: usize =
    QWEN4_ROUTED_SLAB_PAYLOAD + 2 * ROUTED_TRANSPOSE_SLAB_ALIGNMENT;
const QWEN4_PHASE_B_QUANTIZED_WEIGHTS: usize = 1_024;
const QWEN4_PHASE_B_COMPONENT_BUFFERS: usize = 2_048;
const QWEN4_PHASE_B_REQUESTED_BYTES: usize = 943_718_400;
const QWEN4_PHASE_D_QUANTIZED_WEIGHTS: usize = 512;
const QWEN4_PHASE_D_COMPONENT_BUFFERS: usize = 1_024;
const QWEN4_PHASE_D_REQUESTED_BYTES: usize = 471_859_200;

fn routed_slab_align_up(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(ROUTED_TRANSPOSE_SLAB_ALIGNMENT - 1)
        .map(|value| value & !(ROUTED_TRANSPOSE_SLAB_ALIGNMENT - 1))
        .ok_or_else(|| anyhow::anyhow!("routed transpose slab alignment overflow"))
}

pub(super) fn transpose_component_bytes(n: usize, k: usize) -> Result<(usize, usize)> {
    let n_pad = n
        .checked_add(63)
        .map(|value| value & !63)
        .ok_or_else(|| anyhow::anyhow!("routed transpose padded row count overflow"))?;
    let packed = (k / 2)
        .checked_mul(n_pad)
        .ok_or_else(|| anyhow::anyhow!("routed transpose packed size overflow"))?;
    let scale = (k / 16)
        .checked_mul(n_pad)
        .ok_or_else(|| anyhow::anyhow!("routed transpose scale size overflow"))?;
    Ok((packed, scale))
}

#[derive(Clone, Copy, Debug)]
struct RoutedProjectionSlabLayout {
    first_weight_offset: usize,
    scale_offset_in_pair: usize,
    pair_stride: usize,
    expert_count: usize,
    packed_bytes: usize,
    scale_bytes: usize,
    payload_bytes: usize,
    end_offset: usize,
}

impl RoutedProjectionSlabLayout {
    fn build(
        cursor: usize,
        expert_count: usize,
        packed_bytes: usize,
        scale_bytes: usize,
    ) -> Result<Self> {
        let first_weight_offset = routed_slab_align_up(cursor)?;
        let scale_offset_in_pair = routed_slab_align_up(packed_bytes)?;
        let pair_stride = routed_slab_align_up(
            scale_offset_in_pair
                .checked_add(scale_bytes)
                .ok_or_else(|| anyhow::anyhow!("routed transpose pair size overflow"))?,
        )?;
        let end_offset = first_weight_offset
            .checked_add(
                pair_stride
                    .checked_mul(expert_count)
                    .ok_or_else(|| anyhow::anyhow!("routed transpose projection size overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("routed transpose projection end overflow"))?;
        let payload_bytes = packed_bytes
            .checked_add(scale_bytes)
            .and_then(|bytes| bytes.checked_mul(expert_count))
            .ok_or_else(|| anyhow::anyhow!("routed transpose projection payload overflow"))?;
        Ok(Self {
            first_weight_offset,
            scale_offset_in_pair,
            pair_stride,
            expert_count,
            packed_bytes,
            scale_bytes,
            payload_bytes,
            end_offset,
        })
    }

    fn offsets(self, expert: usize) -> Result<(usize, usize)> {
        anyhow::ensure!(
            expert < self.expert_count,
            "routed transpose expert index {expert} exceeds {}",
            self.expert_count
        );
        let weight = self
            .first_weight_offset
            .checked_add(
                self.pair_stride
                    .checked_mul(expert)
                    .ok_or_else(|| anyhow::anyhow!("routed transpose expert offset overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("routed transpose weight offset overflow"))?;
        let scale = weight
            .checked_add(self.scale_offset_in_pair)
            .ok_or_else(|| anyhow::anyhow!("routed transpose scale offset overflow"))?;
        Ok((weight, scale))
    }
}

#[derive(Clone, Copy, Debug)]
struct RoutedExpertSlabLayout {
    gate: RoutedProjectionSlabLayout,
    up: RoutedProjectionSlabLayout,
    down: RoutedProjectionSlabLayout,
    packed_payload_bytes: usize,
    scale_payload_bytes: usize,
    payload_bytes: usize,
    gate_up_allocation_bytes: usize,
    down_allocation_bytes: usize,
    allocation_bytes: usize,
}

impl RoutedExpertSlabLayout {
    fn canonical_qwen4() -> Result<Self> {
        let layout = Self::build(
            QWEN4_ROUTED_SLAB_EXPERTS,
            QWEN4_ROUTED_SLAB_HIDDEN,
            QWEN4_ROUTED_SLAB_INTER,
        )?;
        anyhow::ensure!(
            layout.packed_payload_bytes == QWEN4_ROUTED_SLAB_PACKED_PAYLOAD
                && layout.scale_payload_bytes == QWEN4_ROUTED_SLAB_SCALE_PAYLOAD
                && layout.payload_bytes == QWEN4_ROUTED_SLAB_PAYLOAD
                && layout.allocation_bytes == QWEN4_ROUTED_SLAB_ALLOCATION,
            "canonical Qwen4 routed transpose slab byte contract changed"
        );
        Ok(layout)
    }

    fn build(expert_count: usize, hidden: usize, inter: usize) -> Result<Self> {
        let (gate_up_packed, gate_up_scale) = transpose_component_bytes(inter, hidden)?;
        let (down_packed, down_scale) = transpose_component_bytes(hidden, inter)?;
        let gate = RoutedProjectionSlabLayout::build(
            ROUTED_TRANSPOSE_SLAB_ALIGNMENT,
            expert_count,
            gate_up_packed,
            gate_up_scale,
        )?;
        let up = RoutedProjectionSlabLayout::build(
            gate.end_offset,
            expert_count,
            gate_up_packed,
            gate_up_scale,
        )?;
        let down = RoutedProjectionSlabLayout::build(
            ROUTED_TRANSPOSE_SLAB_ALIGNMENT,
            expert_count,
            down_packed,
            down_scale,
        )?;
        let gate_up_allocation_bytes = routed_slab_align_up(up.end_offset)?;
        let down_allocation_bytes = routed_slab_align_up(down.end_offset)?;
        let allocation_bytes = gate_up_allocation_bytes
            .checked_add(down_allocation_bytes)
            .ok_or_else(|| anyhow::anyhow!("routed transpose slab allocation sum overflow"))?;
        let packed_payload_bytes = gate
            .packed_bytes
            .checked_add(up.packed_bytes)
            .and_then(|bytes| bytes.checked_add(down.packed_bytes))
            .and_then(|bytes| bytes.checked_mul(expert_count))
            .ok_or_else(|| anyhow::anyhow!("routed transpose packed payload overflow"))?;
        let scale_payload_bytes = gate
            .scale_bytes
            .checked_add(up.scale_bytes)
            .and_then(|bytes| bytes.checked_add(down.scale_bytes))
            .and_then(|bytes| bytes.checked_mul(expert_count))
            .ok_or_else(|| anyhow::anyhow!("routed transpose scale payload overflow"))?;
        let payload_bytes = gate
            .payload_bytes
            .checked_add(up.payload_bytes)
            .and_then(|bytes| bytes.checked_add(down.payload_bytes))
            .ok_or_else(|| anyhow::anyhow!("routed transpose total payload overflow"))?;
        Ok(Self {
            gate,
            up,
            down,
            packed_payload_bytes,
            scale_payload_bytes,
            payload_bytes,
            gate_up_allocation_bytes,
            down_allocation_bytes,
            allocation_bytes,
        })
    }
}

fn allocate_routed_transpose_slab(
    owner: &mut Option<DevicePtr>,
    allocation_bytes: usize,
    allocate: impl FnOnce(usize) -> Result<DevicePtr>,
) -> Result<DevicePtr> {
    anyhow::ensure!(
        owner.is_none(),
        "routed transpose slab is already installed"
    );
    let slab = allocate(allocation_bytes)?;
    anyhow::ensure!(!slab.is_null(), "routed transpose slab allocation is null");
    *owner = Some(slab);
    Ok(slab)
}

pub(super) fn canonical_qwen4_routed_slab_selected(
    config: &atlas_core::config::ModelConfig,
    keep_originals: bool,
) -> bool {
    !keep_originals
        && config.is_qwen4_exp()
        && config.num_hidden_layers == QWEN4_ROUTED_SLAB_LAYERS
        && config.num_experts == QWEN4_ROUTED_SLAB_EXPERTS
        && config.hidden_size == QWEN4_ROUTED_SLAB_HIDDEN
        && config.moe_intermediate_size == QWEN4_ROUTED_SLAB_INTER
        && config.ep_world_size == 1
        && config.ep_rank == 0
}

fn validate_canonical_qwen4_routed_experts(experts: &[ExpertWeight]) -> Result<()> {
    anyhow::ensure!(
        experts.len() == QWEN4_ROUTED_SLAB_EXPERTS,
        "canonical Qwen4 routed transpose requires {} experts, got {}",
        QWEN4_ROUTED_SLAB_EXPERTS,
        experts.len()
    );
    for (expert_index, expert) in experts.iter().enumerate() {
        for (projection, weight) in [
            ("gate", &expert.gate_proj),
            ("up", &expert.up_proj),
            ("down", &expert.down_proj),
        ] {
            anyhow::ensure!(
                !weight.weight.is_null() && !weight.weight_scale.is_null(),
                "canonical Qwen4 routed transpose expert {expert_index} {projection} is incomplete"
            );
        }
    }
    Ok(())
}

fn moe_slab_transpose_gpu_enabled() -> bool {
    std::env::var("ATLAS_MOE_TRANSPOSE_GPU")
        .map(|v| {
            let trimmed = v.trim();
            !(trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off"))
        })
        .unwrap_or(true)
}

fn can_use_gpu_slab_transpose(
    enabled: bool,
    transpose_k: KernelHandle,
    src_table: &ExpertPtrTable,
    n: usize,
    k: usize,
    num_experts: usize,
) -> bool {
    enabled
        && transpose_k.0 != 0
        && !src_table.packed_ptrs.is_null()
        && !src_table.scale_ptrs.is_null()
        && num_experts > 0
        && n > 0
        && k > 0
        && n.is_multiple_of(64)
        && k.is_multiple_of(16)
        && u32::try_from(num_experts).is_ok()
        && u32::try_from(n).is_ok()
        && u32::try_from(k).is_ok()
}

fn free_expert_ptr_table(table: ExpertPtrTable, gpu: &dyn GpuBackend) {
    if !table.packed_ptrs.is_null() {
        let _ = gpu.free(table.packed_ptrs);
    }
    if !table.scale_ptrs.is_null() {
        let _ = gpu.free(table.scale_ptrs);
    }
    if !table.scale2_vals.is_null() {
        let _ = gpu.free(table.scale2_vals);
    }
}

/// GPU-batched NVFP4 slab transpose for canonical `N % 64 == 0` routed experts.
///
/// Produces the exact byte-for-byte layout of [`transpose_projection_into_slab`]
/// (`dst[c * N + r] = src[r * C + c]` for both packed `[N, K/2]` and scale
/// `[N, K/16]`) using 2 `moe_transpose_u8_batched` kernel launches per
/// projection instead of `512 × 2` serial D2H/CPU/H2D round-trips, cutting the
/// 48-layer Qwen4 transpose pass from ~49 s to ~1 s on GB10.
#[allow(clippy::too_many_arguments)]
fn transpose_projection_into_slab_gpu(
    experts: &[ExpertWeight],
    projection: impl Fn(&ExpertWeight) -> &QuantizedWeight,
    src_table: &ExpertPtrTable,
    layout: RoutedProjectionSlabLayout,
    slab: DevicePtr,
    gpu: &dyn GpuBackend,
    transpose_k: KernelHandle,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<ExpertPtrTable> {
    let n_u32 =
        u32::try_from(n).map_err(|_| anyhow::anyhow!("GPU slab transpose N exceeds u32"))?;
    let k_u32 =
        u32::try_from(k).map_err(|_| anyhow::anyhow!("GPU slab transpose K exceeds u32"))?;
    let num_experts_u32 = u32::try_from(experts.len())
        .map_err(|_| anyhow::anyhow!("GPU slab transpose expert count exceeds u32"))?;
    let mut transposed = Vec::with_capacity(experts.len());
    for (expert_index, expert) in experts.iter().enumerate() {
        let src = projection(expert);
        if src.is_null() {
            transposed.push(QuantizedWeight::null());
        } else {
            let (weight_offset, scale_offset) = layout.offsets(expert_index)?;
            transposed.push(QuantizedWeight {
                weight: slab.offset(weight_offset),
                weight_scale: slab.offset(scale_offset),
                weight_scale_2: src.weight_scale_2,
                input_scale: src.input_scale,
            });
        }
    }
    let dst_table = build_ptr_table_from_qw(&transposed, gpu)?;
    if let Err(err) = ops::moe_transpose_u8_batched(
        gpu,
        transpose_k,
        src_table.packed_ptrs,
        dst_table.packed_ptrs,
        n_u32,
        k_u32 / 2,
        num_experts_u32,
        stream,
    ) {
        free_expert_ptr_table(dst_table, gpu);
        return Err(err);
    }
    if let Err(err) = ops::moe_transpose_u8_batched(
        gpu,
        transpose_k,
        src_table.scale_ptrs,
        dst_table.scale_ptrs,
        n_u32,
        k_u32 / 16,
        num_experts_u32,
        stream,
    ) {
        // Wait for the already-launched packed transpose kernel to finish before
        // freeing `dst_table` or returning to a caller that may tear down the slab.
        match gpu.synchronize(stream) {
            Ok(()) => free_expert_ptr_table(dst_table, gpu),
            Err(sync_err) => tracing::error!(
                "GPU slab transpose error drain failed; leaking dst_table to avoid GPU UAF: {sync_err:#}"
            ),
        }
        return Err(err);
    }
    Ok(dst_table)
}

fn transpose_projection_into_slab(
    experts: &[ExpertWeight],
    projection: impl Fn(&ExpertWeight) -> &QuantizedWeight,
    layout: RoutedProjectionSlabLayout,
    slab: DevicePtr,
    gpu: &dyn GpuBackend,
    n: usize,
    k: usize,
) -> Result<Vec<QuantizedWeight>> {
    let mut transposed = Vec::with_capacity(experts.len());
    for (expert_index, expert) in experts.iter().enumerate() {
        let (weight_offset, scale_offset) = layout.offsets(expert_index)?;
        transposed.push(projection(expert).transpose_for_gemm_into(
            gpu,
            n,
            k,
            slab.offset(weight_offset),
            slab.offset(scale_offset),
        )?);
    }
    Ok(transposed)
}

impl MoeLayer {
    /// True when the router runs on its NVFP4 copy, so no BF16 router arm is
    /// reached and the BF16 router source may be released.
    pub(crate) fn has_nvfp4_router(&self) -> bool {
        self.gate_nvfp4.is_some()
    }

    pub fn attach_routed_original_groups(&mut self, groups: RoutedOriginalGroups) -> Result<()> {
        anyhow::ensure!(
            self.routed_original_groups.is_none(),
            "routed group owners already attached"
        );
        for projection in [
            RoutedProjection::Gate,
            RoutedProjection::Up,
            RoutedProjection::Down,
        ] {
            let pairs: Vec<_> = self
                .weights
                .experts
                .iter()
                .map(|expert| {
                    let weight = match projection {
                        RoutedProjection::Gate => &expert.gate_proj,
                        RoutedProjection::Up => &expert.up_proj,
                        RoutedProjection::Down => &expert.down_proj,
                    };
                    (weight.weight, weight.weight_scale)
                })
                .collect();
            groups.validate_projection(projection, &pairs)?;
        }
        self.routed_original_groups = Some(groups);
        Ok(())
    }

    /// Pick the FP8 grouped GEMM kernel handle — v2 (coalesced thread remap)
    /// is default-ON (`ATLAS_FP8_MOE_COALESCED=0` opts out to restore legacy v1)
    /// and falls back to v1 only when v2 is absent from the PTX image.
    #[inline]
    pub(super) fn fp8_grouped_kernel(&self) -> KernelHandle {
        let k = if self.fp8_moe_coalesced_enabled && self.moe_fp8_grouped_gemm_v2_k.0 != 0 {
            self.moe_fp8_grouped_gemm_v2_k
        } else {
            self.moe_fp8_grouped_gemm_k
        };
        // One-shot log to verify selection at runtime
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "ATLAS_FP8_GROUPED_KERNEL: coalesced_enabled={} v2_handle={} v1_handle={} selected_handle={}",
                self.fp8_moe_coalesced_enabled,
                self.moe_fp8_grouped_gemm_v2_k.0,
                self.moe_fp8_grouped_gemm_k.0,
                k.0
            );
        });
        k
    }

    /// Set pre-expert norm (Gemma-4 26B: pre_feedforward_layernorm_2).
    /// Applied to input AFTER routing but BEFORE expert dispatch.
    pub fn set_pre_expert_norm(&mut self, norm: crate::weight_map::DenseWeight) {
        self.pre_expert_norm = Some(norm);
    }

    /// Set GeGLU activation for MoE experts (Gemma-4 26B).
    /// Replaces SiLU with GELU in the sorted/unfused path and gates Qwen4
    /// SiLU-only prefill admission.
    pub fn set_gelu_activation(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.moe_act_mul = gpu.kernel("gelu", "gelu_mul")?;
        self.gelu_activation = true;
        Ok(())
    }

    /// Transpose MoE weights for coalesced prefill GEMM reads.
    ///
    /// Transposes per-expert routed weights `[N, K/2] → [K/2, N]` to enable
    /// the cp.async pipelined FP8-MMA K64 kernels. This doubles expert
    /// memory (~17 GB for 35B, ~30 GB for 122B) but eliminates the
    /// catastrophic uncoalesced B reads in the fallback grouped GEMM,
    /// cutting MoE prefill time by ~2x.
    pub fn transpose_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, true)
    }

    /// Transpose only the gate+up routed weights, leaving the down projection
    /// in its original layout. Cuts the transpose memory cost from ~3×
    /// (gate+up+down) to ~2× per expert. Used by MiniMax M2.7-NVFP4 EP=2
    /// when the full transpose doesn't fit but gate+up does — the fused
    /// `moe_w4a16_fused_gate_up_k64_n128` kernel still runs (capturing the
    /// dominant gate+up bandwidth savings), while down stays on the
    /// uncoalesced grouped-GEMM path.
    pub fn transpose_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, false)
    }

    pub(super) fn transpose_for_prefill_impl(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        include_down: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;

        // Transpose per-expert routed weights for coalesced prefill GEMM reads.
        let num_experts = self.weights.experts.len();
        let mut gate_t = Vec::with_capacity(num_experts);
        let mut up_t = Vec::with_capacity(num_experts);
        let mut down_t = Vec::with_capacity(num_experts);

        for expert in &self.weights.experts {
            if expert.gate_proj.is_null() {
                gate_t.push(QuantizedWeight::null());
                up_t.push(QuantizedWeight::null());
                if include_down {
                    down_t.push(QuantizedWeight::null());
                }
            } else {
                gate_t.push(expert.gate_proj.transpose_for_gemm(gpu, inter, h)?);
                up_t.push(expert.up_proj.transpose_for_gemm(gpu, inter, h)?);
                if include_down {
                    down_t.push(expert.down_proj.transpose_for_gemm(gpu, h, inter)?);
                }
            }
        }

        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        if include_down {
            self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        }

        // Transpose shared expert weights (tiny: ~5 MB per layer).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            if include_down {
                self.shared_down_t =
                    Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                        gpu,
                        h,
                        shared_inter,
                    )?);
            }
        }

        Ok(())
    }

    /// Phase 8a unified-layout transpose pass: build persistent transposed
    /// gate/up/down for all experts, freeing the untransposed copies between
    /// phases so the entire pass fits in tight memory budgets that the
    /// non-unified `transpose_for_prefill_impl(true)` would reject.
    ///
    /// Phased flow (memory math for MiniMax M2.7-NVFP4 EP=2 ≈ 47 GB free):
    ///   A. Transpose gate+up               (allocs +39 GB; free ≈ 8 GB)
    ///   B. Free gate+up untransposed       (frees 39 GB; free ≈ 47 GB)
    ///   C. Transpose down                  (allocs +20 GB; free ≈ 27 GB)
    ///   D. Free down untransposed          (frees 20 GB; free ≈ 47 GB)
    ///
    /// Net memory: same as starting point, but layout is now unified
    /// (transposed-only) — the `[N, K/2]` decode kernels can no longer
    /// run; dispatch must use the `_t` decode kernels (which do).
    ///
    /// Caller responsibilities:
    ///   1. Set `ATLAS_UNIFIED_MOE_LAYOUT=1` so `MoeLayer::use_t_layout_for_decode()`
    ///      returns true at dispatch time.
    ///   2. Call this method INSTEAD of `transpose_for_prefill` /
    ///      `transpose_gate_up_for_prefill`.
    pub fn transpose_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, false)
    }

    /// Hybrid-layout transpose pass — analogue of `transpose_for_prefill_unified`
    /// that **keeps** the untransposed originals so decode + MTP verify dispatch
    /// can continue using the warp-reduction kernels. Allocates ~58 GB
    /// transposed alongside the existing ~58 GB originals on MiniMax M2.7-NVFP4
    /// EP=2; fits in 122 GB GB10 with KV-cache headroom up to ~32K context.
    /// Caller is responsible for memory-fit gating (factory checks free memory
    /// before invoking this).
    pub fn transpose_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, true)
    }

    /// Phased build of the transposed weight set. When `keep_originals` is true
    /// (hybrid-layout mode), Phase B and Phase D frees are skipped so decode
    /// paths still find the untransposed weights. When false (unified-layout
    /// mode), the originals are freed between phases — current Phase 8a
    /// behavior.
    pub(super) fn transpose_for_prefill_unified_inner(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        keep_originals: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;
        let num_experts = self.weights.experts.len();
        anyhow::ensure!(
            self.routed_bounded_transpose_owners.is_empty(),
            "bounded routed transpose owners are already installed"
        );
        let bounded = bounded_transpose::selected_for_layer(config, keep_originals)?;
        let use_ts4e = atlas_moe_scale_ts4e_enabled()
            && !keep_originals
            && inter > 0
            && h > 0
            && inter.is_multiple_of(128)
            && h.is_multiple_of(128)
            && inter.is_multiple_of(32)
            && h.is_multiple_of(32);
        if use_ts4e {
            anyhow::ensure!(
                bounded.is_none(),
                "ATLAS_MOE_SCALE_TS4E=1 conflicts with ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS=1"
            );
            anyhow::ensure!(
                self.routed_ts4e_scale_slabs.is_empty()
                    && self.routed_gate_up_transpose_slab.is_none()
                    && self.routed_down_transpose_slab.is_none(),
                "routed transpose slabs are already installed"
            );
            return self.transpose_for_prefill_unified_ts4e(gpu, config);
        }
        if bounded.is_some() {
            validate_canonical_qwen4_routed_experts(&self.weights.experts)?;
            anyhow::ensure!(
                self.routed_bounded_transpose_owners.is_empty()
                    && self.routed_gate_up_transpose_slab.is_none()
                    && self.routed_down_transpose_slab.is_none(),
                "routed transpose destination owners are already installed"
            );
        }
        let routed_slab_layout =
            if bounded.is_none() && canonical_qwen4_routed_slab_selected(config, keep_originals) {
                validate_canonical_qwen4_routed_experts(&self.weights.experts)?;
                anyhow::ensure!(
                    self.routed_gate_up_transpose_slab.is_none()
                        && self.routed_down_transpose_slab.is_none(),
                    "canonical Qwen4 routed transpose slabs are already installed"
                );
                Some(RoutedExpertSlabLayout::canonical_qwen4()?)
            } else {
                None
            };
        let routed_gate_up_slab = if let Some(layout) = routed_slab_layout {
            Some(allocate_routed_transpose_slab(
                &mut self.routed_gate_up_transpose_slab,
                layout.gate_up_allocation_bytes,
                |bytes| gpu.alloc(bytes),
            )?)
        } else {
            None
        };

        let stream = gpu.default_stream();
        let gpu_slab_enabled = moe_slab_transpose_gpu_enabled();

        // ── Phase A: transpose gate+up routed experts ──
        let mut used_gpu_gate_up = false;
        if let (Some(layout), Some(slab)) = (routed_slab_layout, routed_gate_up_slab)
            && can_use_gpu_slab_transpose(
                gpu_slab_enabled,
                self.moe_transpose_u8_batched_k,
                &self.gate_ptrs,
                inter,
                h,
                num_experts,
            )
            && can_use_gpu_slab_transpose(
                gpu_slab_enabled,
                self.moe_transpose_u8_batched_k,
                &self.up_ptrs,
                inter,
                h,
                num_experts,
            )
        {
            let gate_table = transpose_projection_into_slab_gpu(
                &self.weights.experts,
                |expert| &expert.gate_proj,
                &self.gate_ptrs,
                layout.gate,
                slab,
                gpu,
                self.moe_transpose_u8_batched_k,
                inter,
                h,
                stream,
            )?;
            let up_table = match transpose_projection_into_slab_gpu(
                &self.weights.experts,
                |expert| &expert.up_proj,
                &self.up_ptrs,
                layout.up,
                slab,
                gpu,
                self.moe_transpose_u8_batched_k,
                inter,
                h,
                stream,
            ) {
                Ok(table) => table,
                Err(err) => {
                    // Drain in-flight gate transpose kernels before freeing
                    // `gate_table` or returning to a caller that may free the slab.
                    match gpu.synchronize(stream) {
                        Ok(()) => free_expert_ptr_table(gate_table, gpu),
                        Err(sync_err) => tracing::error!(
                            "GPU gate transpose error drain failed; leaking gate_table to avoid GPU UAF: {sync_err:#}"
                        ),
                    }
                    return Err(err);
                }
            };
            self.gate_ptrs_t = Some(gate_table);
            self.up_ptrs_t = Some(up_table);
            used_gpu_gate_up = true;
        } else {
            let (gate_t, up_t) =
                if let (Some(layout), Some(slab)) = (routed_slab_layout, routed_gate_up_slab) {
                    (
                        transpose_projection_into_slab(
                            &self.weights.experts,
                            |expert| &expert.gate_proj,
                            layout.gate,
                            slab,
                            gpu,
                            inter,
                            h,
                        )?,
                        transpose_projection_into_slab(
                            &self.weights.experts,
                            |expert| &expert.up_proj,
                            layout.up,
                            slab,
                            gpu,
                            inter,
                            h,
                        )?,
                    )
                } else if let Some(owner_cap) = bounded {
                    (
                        bounded_transpose::transpose_projection(
                            &self.weights.experts,
                            |expert| &expert.gate_proj,
                            &mut self.routed_bounded_transpose_owners,
                            gpu,
                            inter,
                            h,
                            owner_cap,
                            "gate",
                        )?,
                        bounded_transpose::transpose_projection(
                            &self.weights.experts,
                            |expert| &expert.up_proj,
                            &mut self.routed_bounded_transpose_owners,
                            gpu,
                            inter,
                            h,
                            owner_cap,
                            "up",
                        )?,
                    )
                } else {
                    let mut gate_t = Vec::with_capacity(num_experts);
                    let mut up_t = Vec::with_capacity(num_experts);
                    for expert in &self.weights.experts {
                        if expert.gate_proj.is_null() {
                            gate_t.push(QuantizedWeight::null());
                            up_t.push(QuantizedWeight::null());
                        } else {
                            gate_t.push(expert.gate_proj.transpose_for_gemm(gpu, inter, h)?);
                            up_t.push(expert.up_proj.transpose_for_gemm(gpu, inter, h)?);
                        }
                    }
                    (gate_t, up_t)
                };
            self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
            self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        }
        // Shared expert (tiny, do unconditionally — fits regardless).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
        }

        if !keep_originals {
            // ── Phase B: free gate+up untransposed ──
            // The previous gate_ptrs / up_ptrs device-side pointer tables now
            // contain stale addresses, but the unified dispatch never reads
            // them (gated by `use_t_layout_for_decode()`).
            if used_gpu_gate_up || self.routed_original_groups.is_some() {
                // Explicit completion is required before freeing original
                // buffers read by GPU transpose kernels or backing groups
                // shared by many original tensor views.
                gpu.synchronize(gpu.default_stream())?;
            }
            if let Some(groups) = self.routed_original_groups.as_ref() {
                groups.release_projection(RoutedProjection::Gate, gpu)?;
                for expert in &mut self.weights.experts {
                    expert.gate_proj.weight = DevicePtr::NULL;
                    expert.gate_proj.weight_scale = DevicePtr::NULL;
                }
                groups.release_projection(RoutedProjection::Up, gpu)?;
                for expert in &mut self.weights.experts {
                    expert.up_proj.weight = DevicePtr::NULL;
                    expert.up_proj.weight_scale = DevicePtr::NULL;
                }
            } else {
                for expert in &mut self.weights.experts {
                    if !expert.gate_proj.weight.is_null() {
                        gpu.free(expert.gate_proj.weight)?;
                        gpu.free(expert.gate_proj.weight_scale)?;
                        expert.gate_proj.weight = DevicePtr::NULL;
                        expert.gate_proj.weight_scale = DevicePtr::NULL;
                    }
                    if !expert.up_proj.weight.is_null() {
                        gpu.free(expert.up_proj.weight)?;
                        gpu.free(expert.up_proj.weight_scale)?;
                        expert.up_proj.weight = DevicePtr::NULL;
                        expert.up_proj.weight_scale = DevicePtr::NULL;
                    }
                }
            }
            if !self.weights.shared_expert.gate_proj.weight.is_null() && shared_inter > 0 {
                gpu.free(self.weights.shared_expert.gate_proj.weight)?;
                gpu.free(self.weights.shared_expert.gate_proj.weight_scale)?;
                self.weights.shared_expert.gate_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.gate_proj.weight_scale = DevicePtr::NULL;
                gpu.free(self.weights.shared_expert.up_proj.weight)?;
                gpu.free(self.weights.shared_expert.up_proj.weight_scale)?;
                self.weights.shared_expert.up_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.up_proj.weight_scale = DevicePtr::NULL;
            }
        }

        // ── Phase C: transpose down routed experts ──
        // Delay this allocation until every Phase-B original has been freed.
        let routed_down_slab = if let Some(layout) = routed_slab_layout {
            Some(allocate_routed_transpose_slab(
                &mut self.routed_down_transpose_slab,
                layout.down_allocation_bytes,
                |bytes| gpu.alloc(bytes),
            )?)
        } else {
            None
        };
        let mut used_gpu_down = false;
        if let (Some(layout), Some(slab)) = (routed_slab_layout, routed_down_slab)
            && can_use_gpu_slab_transpose(
                gpu_slab_enabled,
                self.moe_transpose_u8_batched_k,
                &self.down_ptrs,
                h,
                inter,
                num_experts,
            )
        {
            let down_table = transpose_projection_into_slab_gpu(
                &self.weights.experts,
                |expert| &expert.down_proj,
                &self.down_ptrs,
                layout.down,
                slab,
                gpu,
                self.moe_transpose_u8_batched_k,
                h,
                inter,
                stream,
            )?;
            self.down_ptrs_t = Some(down_table);
            used_gpu_down = true;
        } else {
            let down_t = if let (Some(layout), Some(slab)) = (routed_slab_layout, routed_down_slab)
            {
                transpose_projection_into_slab(
                    &self.weights.experts,
                    |expert| &expert.down_proj,
                    layout.down,
                    slab,
                    gpu,
                    h,
                    inter,
                )?
            } else if let Some(owner_cap) = bounded {
                bounded_transpose::transpose_projection(
                    &self.weights.experts,
                    |expert| &expert.down_proj,
                    &mut self.routed_bounded_transpose_owners,
                    gpu,
                    h,
                    inter,
                    owner_cap,
                    "down",
                )?
            } else {
                let mut down_t = Vec::with_capacity(num_experts);
                for expert in &self.weights.experts {
                    if expert.down_proj.is_null() {
                        down_t.push(QuantizedWeight::null());
                    } else {
                        down_t.push(expert.down_proj.transpose_for_gemm(gpu, h, inter)?);
                    }
                }
                down_t
            };
            self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        }
        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                h,
                shared_inter,
            )?);
        }

        if !keep_originals {
            // ── Phase D: free down untransposed ──
            if used_gpu_down || self.routed_original_groups.is_some() {
                gpu.synchronize(gpu.default_stream())?;
            }
            if let Some(groups) = self.routed_original_groups.as_ref() {
                groups.release_projection(RoutedProjection::Down, gpu)?;
                for expert in &mut self.weights.experts {
                    expert.down_proj.weight = DevicePtr::NULL;
                    expert.down_proj.weight_scale = DevicePtr::NULL;
                }
            } else {
                for expert in &mut self.weights.experts {
                    if !expert.down_proj.weight.is_null() {
                        gpu.free(expert.down_proj.weight)?;
                        gpu.free(expert.down_proj.weight_scale)?;
                        expert.down_proj.weight = DevicePtr::NULL;
                        expert.down_proj.weight_scale = DevicePtr::NULL;
                    }
                }
            }
            if !self.weights.shared_expert.down_proj.weight.is_null() && shared_inter > 0 {
                gpu.free(self.weights.shared_expert.down_proj.weight)?;
                gpu.free(self.weights.shared_expert.down_proj.weight_scale)?;
                self.weights.shared_expert.down_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.down_proj.weight_scale = DevicePtr::NULL;
            }
        } else if used_gpu_gate_up || used_gpu_down {
            gpu.synchronize(stream)?;
        }

        if let Some(layout) = routed_slab_layout {
            tracing::info!(
                target: "atlas_transposed_expert_slab",
                schema = "atlas.moe-transposed-expert-slab.v2",
                layout = "qwen4-routed-nvfp4",
                keep_originals = false,
                managed = false,
                slab_count = 2,
                experts = QWEN4_ROUTED_SLAB_EXPERTS,
                projections = QWEN4_ROUTED_SLAB_PROJECTIONS,
                quantized_weights = QWEN4_ROUTED_SLAB_QUANTIZED_WEIGHTS,
                component_views = QWEN4_ROUTED_SLAB_COMPONENT_VIEWS,
                packed_payload_bytes = layout.packed_payload_bytes,
                scale_payload_bytes = layout.scale_payload_bytes,
                payload_bytes = layout.payload_bytes,
                allocation_bytes = layout.allocation_bytes,
                gate_up_allocation_bytes = layout.gate_up_allocation_bytes,
                down_allocation_bytes = layout.down_allocation_bytes,
                down_allocation_after_phase_b = true,
                alignment_bytes = ROUTED_TRANSPOSE_SLAB_ALIGNMENT,
                base_unexposed = true,
                individual_pair_allocations = 0,
                phase_b_freed_quantized_weights = QWEN4_PHASE_B_QUANTIZED_WEIGHTS,
                phase_b_freed_component_buffers = QWEN4_PHASE_B_COMPONENT_BUFFERS,
                phase_b_freed_requested_bytes = QWEN4_PHASE_B_REQUESTED_BYTES,
                phase_d_freed_quantized_weights = QWEN4_PHASE_D_QUANTIZED_WEIGHTS,
                phase_d_freed_component_buffers = QWEN4_PHASE_D_COMPONENT_BUFFERS,
                phase_d_freed_requested_bytes = QWEN4_PHASE_D_REQUESTED_BYTES,
                "TRANSPOSED_EXPERT_SLAB MOE_TRANSPOSE_ROUTED_SLAB_READY"
            );
        }
        if let Some(owner_cap) = bounded {
            bounded_transpose::record_ready(
                config,
                self.routed_bounded_transpose_owners.len(),
                owner_cap,
            )?;
        }

        Ok(())
    }

    fn transpose_for_prefill_unified_ts4e(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;
        let num_experts = self.weights.experts.len();
        let stream = gpu.default_stream();
        anyhow::ensure!(
            self.weights
                .experts
                .iter()
                .all(|expert| expert.gate_proj.is_null() == expert.up_proj.is_null()),
            "TS4E-128 routed experts must have matching gate_proj/up_proj nullity before sizing gate_up_slab"
        );
        let active_gu = self
            .weights
            .experts
            .iter()
            .filter(|expert| !expert.gate_proj.is_null())
            .count();
        let active_dn = self
            .weights
            .experts
            .iter()
            .filter(|expert| !expert.down_proj.is_null())
            .count();

        // ── Phase A: packed-only gate+up slab + 2-pass GPU TS4E-128 scale encoding ──
        let (gu_packed_bytes, _) = transpose_component_bytes(inter, h)?;
        let gu_stride = routed_slab_align_up(gu_packed_bytes)?;
        let gate_first = ROUTED_TRANSPOSE_SLAB_ALIGNMENT;
        let gu_proj_bytes = gu_stride
            .checked_mul(active_gu.max(1))
            .ok_or_else(|| anyhow::anyhow!("TS4E-128 gate/up packed slab size overflow"))?;
        let up_first = gate_first
            .checked_add(gu_proj_bytes)
            .ok_or_else(|| anyhow::anyhow!("TS4E-128 up packed slab offset overflow"))?;
        let gu_alloc_bytes = routed_slab_align_up(
            up_first
                .checked_add(gu_proj_bytes)
                .ok_or_else(|| anyhow::anyhow!("TS4E-128 gate/up packed allocation overflow"))?,
        )?;
        let gate_up_slab = allocate_routed_transpose_slab(
            &mut self.routed_gate_up_transpose_slab,
            gu_alloc_bytes,
            |bytes| gpu.alloc(bytes),
        )?;
        let gate_res = transpose_packed_and_encode_ts4e_projection(
            &self.weights.experts,
            |expert| &expert.gate_proj,
            &self.gate_ptrs,
            gate_up_slab,
            gate_first,
            gu_stride,
            &mut self.routed_ts4e_scale_slabs,
            inter,
            h,
            gpu,
            self.moe_transpose_u8_batched_k,
            stream,
        )?;
        let up_res = transpose_packed_and_encode_ts4e_projection(
            &self.weights.experts,
            |expert| &expert.up_proj,
            &self.up_ptrs,
            gate_up_slab,
            up_first,
            gu_stride,
            &mut self.routed_ts4e_scale_slabs,
            inter,
            h,
            gpu,
            self.moe_transpose_u8_batched_k,
            stream,
        )?;
        self.gate_ptrs_t = Some(gate_res.table);
        self.up_ptrs_t = Some(up_res.table);

        // Shared expert always stays flat E4M3 (~5 MB/layer; decode reads sh_*_t_scale flat).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
        }

        // ── Phase B: free gate+up untransposed originals ──
        if let Some(groups) = self.routed_original_groups.as_ref() {
            gpu.synchronize(stream)?;
            groups.release_projection(RoutedProjection::Gate, gpu)?;
            for expert in &mut self.weights.experts {
                expert.gate_proj.weight = DevicePtr::NULL;
                expert.gate_proj.weight_scale = DevicePtr::NULL;
            }
            groups.release_projection(RoutedProjection::Up, gpu)?;
            for expert in &mut self.weights.experts {
                expert.up_proj.weight = DevicePtr::NULL;
                expert.up_proj.weight_scale = DevicePtr::NULL;
            }
        } else {
            for expert in &mut self.weights.experts {
                if !expert.gate_proj.weight.is_null() {
                    gpu.free(expert.gate_proj.weight)?;
                    gpu.free(expert.gate_proj.weight_scale)?;
                    expert.gate_proj.weight = DevicePtr::NULL;
                    expert.gate_proj.weight_scale = DevicePtr::NULL;
                }
                if !expert.up_proj.weight.is_null() {
                    gpu.free(expert.up_proj.weight)?;
                    gpu.free(expert.up_proj.weight_scale)?;
                    expert.up_proj.weight = DevicePtr::NULL;
                    expert.up_proj.weight_scale = DevicePtr::NULL;
                }
            }
        }
        if !self.weights.shared_expert.gate_proj.weight.is_null() && shared_inter > 0 {
            gpu.free(self.weights.shared_expert.gate_proj.weight)?;
            gpu.free(self.weights.shared_expert.gate_proj.weight_scale)?;
            self.weights.shared_expert.gate_proj.weight = DevicePtr::NULL;
            self.weights.shared_expert.gate_proj.weight_scale = DevicePtr::NULL;
            gpu.free(self.weights.shared_expert.up_proj.weight)?;
            gpu.free(self.weights.shared_expert.up_proj.weight_scale)?;
            self.weights.shared_expert.up_proj.weight = DevicePtr::NULL;
            self.weights.shared_expert.up_proj.weight_scale = DevicePtr::NULL;
        }

        // ── Phase C: packed-only down slab + 2-pass GPU TS4E-128 scale encoding ──
        let (dn_packed_bytes, _) = transpose_component_bytes(h, inter)?;
        let dn_stride = routed_slab_align_up(dn_packed_bytes)?;
        let dn_first = ROUTED_TRANSPOSE_SLAB_ALIGNMENT;
        let dn_proj_bytes = dn_stride
            .checked_mul(active_dn.max(1))
            .ok_or_else(|| anyhow::anyhow!("TS4E-128 down packed slab size overflow"))?;
        let dn_alloc_bytes = routed_slab_align_up(
            dn_first
                .checked_add(dn_proj_bytes)
                .ok_or_else(|| anyhow::anyhow!("TS4E-128 down packed allocation overflow"))?,
        )?;
        let down_slab = allocate_routed_transpose_slab(
            &mut self.routed_down_transpose_slab,
            dn_alloc_bytes,
            |bytes| gpu.alloc(bytes),
        )?;
        let down_res = transpose_packed_and_encode_ts4e_projection(
            &self.weights.experts,
            |expert| &expert.down_proj,
            &self.down_ptrs,
            down_slab,
            dn_first,
            dn_stride,
            &mut self.routed_ts4e_scale_slabs,
            h,
            inter,
            gpu,
            self.moe_transpose_u8_batched_k,
            stream,
        )?;
        self.down_ptrs_t = Some(down_res.table);

        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                h,
                shared_inter,
            )?);
        }

        // ── Phase D: free down untransposed originals ──
        if let Some(groups) = self.routed_original_groups.as_ref() {
            gpu.synchronize(stream)?;
            groups.release_projection(RoutedProjection::Down, gpu)?;
            for expert in &mut self.weights.experts {
                expert.down_proj.weight = DevicePtr::NULL;
                expert.down_proj.weight_scale = DevicePtr::NULL;
            }
        } else {
            for expert in &mut self.weights.experts {
                if !expert.down_proj.weight.is_null() {
                    gpu.free(expert.down_proj.weight)?;
                    gpu.free(expert.down_proj.weight_scale)?;
                    expert.down_proj.weight = DevicePtr::NULL;
                    expert.down_proj.weight_scale = DevicePtr::NULL;
                }
            }
        }
        if !self.weights.shared_expert.down_proj.weight.is_null() && shared_inter > 0 {
            gpu.free(self.weights.shared_expert.down_proj.weight)?;
            gpu.free(self.weights.shared_expert.down_proj.weight_scale)?;
            self.weights.shared_expert.down_proj.weight = DevicePtr::NULL;
            self.weights.shared_expert.down_proj.weight_scale = DevicePtr::NULL;
        }

        self.routed_scale_layout = RoutedScaleLayout::Ts4e128;

        let flat_scale_bytes = active_gu * 2 * (h / 16) * inter + active_dn * (inter / 16) * h;
        let ts4e_scale_bytes =
            gate_res.scale_slab_bytes + up_res.scale_slab_bytes + down_res.scale_slab_bytes;
        let reclaimed_bytes = flat_scale_bytes.saturating_sub(ts4e_scale_bytes);
        let total_exc = gate_res.total_exc + up_res.total_exc + down_res.total_exc;
        tracing::info!(
            experts = num_experts,
            flat_scale_bytes,
            ts4e_scale_bytes,
            reclaimed_bytes,
            total_exc,
            "ATLAS_TS4E128_MOE_SCALES_READY"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn small_weight(
        gpu: &MockGpuBackend,
        rows: usize,
        columns: usize,
        seed: u8,
    ) -> Result<QuantizedWeight> {
        let packed: Vec<u8> = (0..rows * (columns / 2))
            .map(|index| (index as u8).wrapping_add(seed))
            .collect();
        let scales: Vec<u8> = (0..rows * (columns / 16))
            .map(|index| (index as u8).wrapping_add(seed.wrapping_mul(3)))
            .collect();
        let weight = gpu.alloc(packed.len())?;
        gpu.copy_h2d(&packed, weight)?;
        let weight_scale = gpu.alloc(scales.len())?;
        gpu.copy_h2d(&scales, weight_scale)?;
        let input_scale = gpu.alloc(4)?;
        gpu.copy_h2d(&(seed as f32).to_le_bytes(), input_scale)?;
        Ok(QuantizedWeight {
            weight,
            weight_scale,
            weight_scale_2: seed as f32,
            input_scale,
        })
    }

    fn assert_transposed_weight(
        gpu: &MockGpuBackend,
        original: QuantizedWeight,
        transposed: QuantizedWeight,
        rows: usize,
        columns: usize,
    ) -> Result<()> {
        let padded_rows = rows.div_ceil(64) * 64;
        for (source, destination, width) in [
            (original.weight, transposed.weight, columns / 2),
            (original.weight_scale, transposed.weight_scale, columns / 16),
        ] {
            let mut input = vec![0u8; rows * width];
            gpu.copy_d2h(source, &mut input)?;
            let mut expected = vec![0u8; padded_rows * width];
            for row in 0..rows {
                for column in 0..width {
                    expected[column * padded_rows + row] = input[row * width + column];
                }
            }
            let mut actual = vec![0u8; expected.len()];
            gpu.copy_d2h(destination, &mut actual)?;
            assert_eq!(actual, expected);
        }
        assert_eq!(transposed.weight_scale_2, original.weight_scale_2);
        assert_eq!(transposed.input_scale, original.input_scale);
        Ok(())
    }

    fn assert_transposed_table(gpu: &MockGpuBackend, weights: &[QuantizedWeight]) -> Result<()> {
        let table = build_ptr_table_from_qw(weights, gpu)?;
        for (pointer, expected) in [
            (
                table.packed_ptrs,
                weights
                    .iter()
                    .flat_map(|weight| weight.weight.0.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                table.scale_ptrs,
                weights
                    .iter()
                    .flat_map(|weight| weight.weight_scale.0.to_le_bytes())
                    .collect(),
            ),
            (
                table.scale2_vals,
                weights
                    .iter()
                    .flat_map(|weight| weight.weight_scale_2.to_le_bytes())
                    .collect(),
            ),
        ] {
            let mut actual = vec![0u8; expected.len()];
            gpu.copy_d2h(pointer, &mut actual)?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn phase_slab_owners_fail_closed_without_overwriting_prior_owner() -> Result<()> {
        let gpu = MockGpuBackend::new();
        let mut gate_up_owner = None;
        let mut down_owner = None;
        let first = allocate_routed_transpose_slab(&mut gate_up_owner, 512, |bytes| {
            assert_eq!(bytes, 512);
            gpu.alloc(bytes)
        })?;
        assert_eq!(gate_up_owner, Some(first));
        assert!(
            allocate_routed_transpose_slab(&mut down_owner, 256, |_| {
                anyhow::bail!("injected down allocation failure")
            })
            .is_err()
        );
        assert_eq!(gate_up_owner, Some(first));
        assert_eq!(down_owner, None);
        assert_eq!(gpu.alloc_count(), 1);
        assert!(gpu.read_alloc(first).is_some());
        assert!(
            allocate_routed_transpose_slab(&mut gate_up_owner, 512, |_| {
                panic!("must refuse reinstall before allocation")
            })
            .is_err()
        );
        assert!(
            allocate_routed_transpose_slab(&mut down_owner, 256, |_| { Ok(DevicePtr::NULL) })
                .is_err()
        );
        assert_eq!(down_owner, None);
        gpu.free(first)?;
        assert_eq!(gpu.alloc_count(), 0);
        Ok(())
    }

    #[test]
    fn small_phase_slabs_preserve_padded_bytes_tables_and_original_lifetimes() -> Result<()> {
        let gpu = MockGpuBackend::new();
        let layout = RoutedExpertSlabLayout::build(2, 64, 32)?;
        let mut experts = Vec::new();
        for seed in [1u8, 7] {
            experts.push(ExpertWeight {
                gate_proj: small_weight(&gpu, 32, 64, seed)?,
                up_proj: small_weight(&gpu, 32, 64, seed + 1)?,
                down_proj: small_weight(&gpu, 64, 32, seed + 2)?,
            });
        }
        let mut gate_up_owner = None;
        let mut down_owner = None;
        let gate_up = allocate_routed_transpose_slab(
            &mut gate_up_owner,
            layout.gate_up_allocation_bytes,
            |bytes| gpu.alloc(bytes),
        )?;
        let gate = transpose_projection_into_slab(
            &experts,
            |expert| &expert.gate_proj,
            layout.gate,
            gate_up,
            &gpu,
            32,
            64,
        )?;
        let up = transpose_projection_into_slab(
            &experts,
            |expert| &expert.up_proj,
            layout.up,
            gate_up,
            &gpu,
            32,
            64,
        )?;
        assert_eq!(down_owner, None);
        for (index, expert) in experts.iter().enumerate() {
            assert_transposed_weight(&gpu, expert.gate_proj, gate[index], 32, 64)?;
            assert_transposed_weight(&gpu, expert.up_proj, up[index], 32, 64)?;
            for original in [expert.gate_proj, expert.up_proj] {
                gpu.free(original.weight)?;
                gpu.free(original.weight_scale)?;
                assert!(gpu.read_alloc(original.weight).is_none());
                assert!(gpu.read_alloc(original.input_scale).is_some());
            }
            assert!(gpu.read_alloc(expert.down_proj.weight).is_some());
        }
        let down_slab = allocate_routed_transpose_slab(
            &mut down_owner,
            layout.down_allocation_bytes,
            |bytes| gpu.alloc(bytes),
        )?;
        let down = transpose_projection_into_slab(
            &experts,
            |expert| &expert.down_proj,
            layout.down,
            down_slab,
            &gpu,
            64,
            32,
        )?;
        for (index, expert) in experts.iter().enumerate() {
            assert_transposed_weight(&gpu, expert.down_proj, down[index], 64, 32)?;
            gpu.free(expert.down_proj.weight)?;
            gpu.free(expert.down_proj.weight_scale)?;
        }
        assert_transposed_table(&gpu, &gate)?;
        assert_transposed_table(&gpu, &up)?;
        assert_transposed_table(&gpu, &down)?;
        assert!(gpu.read_alloc(gate_up).is_some());
        assert!(gpu.read_alloc(down_slab).is_some());
        assert_ne!(gate[0].weight, gate_up);
        assert_ne!(down[0].weight, down_slab);
        assert_eq!(gpu.alloc_count(), 17); // two slabs, nine tables, six scalar originals
        Ok(())
    }

    #[test]
    fn phase_slab_sizing_rejects_overflow() {
        assert!(RoutedExpertSlabLayout::build(usize::MAX, 64, 64).is_err());
        assert!(RoutedExpertSlabLayout::build(2, usize::MAX, 64).is_err());
        assert!(RoutedExpertSlabLayout::build(2, 64, usize::MAX).is_err());
    }

    fn canonical_qwen4_config() -> atlas_core::config::ModelConfig {
        let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.num_hidden_layers = QWEN4_ROUTED_SLAB_LAYERS;
        config.num_experts = QWEN4_ROUTED_SLAB_EXPERTS;
        config.hidden_size = QWEN4_ROUTED_SLAB_HIDDEN;
        config.moe_intermediate_size = QWEN4_ROUTED_SLAB_INTER;
        config.ep_world_size = 1;
        config.ep_rank = 0;
        config
    }

    #[test]
    fn canonical_qwen4_routed_slab_layout_is_exact_and_nonoverlapping() -> Result<()> {
        let layout = RoutedExpertSlabLayout::canonical_qwen4()?;
        assert_eq!(layout.gate.first_weight_offset, 256);
        assert_eq!(layout.up.first_weight_offset, 471_859_456);
        assert_eq!(layout.down.first_weight_offset, 256);
        assert_eq!(layout.gate.pair_stride, 921_600);
        assert_eq!(layout.up.pair_stride, 921_600);
        assert_eq!(layout.down.pair_stride, 921_600);
        assert_eq!(layout.packed_payload_bytes, 1_258_291_200);
        assert_eq!(layout.scale_payload_bytes, 157_286_400);
        assert_eq!(layout.payload_bytes, 1_415_577_600);
        assert_eq!(layout.gate_up_allocation_bytes, 943_718_656);
        assert_eq!(layout.down_allocation_bytes, 471_859_456);
        assert_eq!(layout.allocation_bytes, 1_415_578_112);

        let gate_up_projections = [layout.gate, layout.up];
        let down_projections = [layout.down];
        let mut component_views = 0usize;
        for (projections, allocation_bytes) in [
            (&gate_up_projections[..], layout.gate_up_allocation_bytes),
            (&down_projections[..], layout.down_allocation_bytes),
        ] {
            let mut ranges = Vec::new();
            for projection in projections {
                for expert in 0..QWEN4_ROUTED_SLAB_EXPERTS {
                    let (weight, scale) = projection.offsets(expert)?;
                    ranges.push((weight, weight + projection.packed_bytes));
                    ranges.push((scale, scale + projection.scale_bytes));
                }
            }
            component_views += ranges.len();
            ranges.sort_unstable();
            assert!(ranges.iter().all(|(start, end)| {
                *start >= ROUTED_TRANSPOSE_SLAB_ALIGNMENT
                    && *start % ROUTED_TRANSPOSE_SLAB_ALIGNMENT == 0
                    && start < end
                    && *end <= allocation_bytes
            }));
            assert!(ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        }
        assert_eq!(component_views, QWEN4_ROUTED_SLAB_COMPONENT_VIEWS);
        assert_eq!(
            QWEN4_ROUTED_SLAB_QUANTIZED_WEIGHTS * QWEN4_ROUTED_SLAB_LAYERS,
            73_728
        );
        assert_eq!(
            QWEN4_ROUTED_SLAB_COMPONENT_VIEWS * QWEN4_ROUTED_SLAB_LAYERS,
            147_456
        );
        assert_eq!(
            layout.payload_bytes * QWEN4_ROUTED_SLAB_LAYERS,
            67_947_724_800
        );
        assert_eq!(
            layout.allocation_bytes * QWEN4_ROUTED_SLAB_LAYERS,
            67_947_749_376
        );
        Ok(())
    }

    #[test]
    fn routed_slab_selection_is_target_only_and_unified_only() {
        let mut config = canonical_qwen4_config();
        assert!(canonical_qwen4_routed_slab_selected(&config, false));
        assert!(!canonical_qwen4_routed_slab_selected(&config, true));

        config.ep_world_size = 2;
        assert!(!canonical_qwen4_routed_slab_selected(&config, false));
        config = canonical_qwen4_config();
        config.model_type = "minimax_m2".to_string();
        assert!(!canonical_qwen4_routed_slab_selected(&config, false));
        config = canonical_qwen4_config();
        config.num_experts -= 1;
        assert!(!canonical_qwen4_routed_slab_selected(&config, false));
    }

    #[test]
    fn canonical_routed_slab_rejects_partial_expert_components() {
        let weight = QuantizedWeight {
            weight: DevicePtr(0x1000),
            weight_scale: DevicePtr(0x2000),
            weight_scale_2: 1.0,
            input_scale: DevicePtr(0x3000),
        };
        let mut experts = vec![
            ExpertWeight {
                gate_proj: weight,
                up_proj: weight,
                down_proj: weight,
            };
            QWEN4_ROUTED_SLAB_EXPERTS
        ];
        assert!(validate_canonical_qwen4_routed_experts(&experts).is_ok());
        experts[17].up_proj.weight_scale = DevicePtr::NULL;
        let error = validate_canonical_qwen4_routed_experts(&experts).unwrap_err();
        assert!(error.to_string().contains("expert 17 up is incomplete"));
    }

    fn transpose_expected_scales(src_row_major: &[u8], k: usize, n: usize) -> Vec<u8> {
        let k16 = k / 16;
        let mut expected = vec![0u8; k16 * n];
        for col in 0..n {
            for r in 0..k16 {
                expected[r * n + col] = src_row_major[col * k16 + r];
            }
        }
        expected
    }

    #[test]
    fn ts4e128_roundtrip_bit_exact_all_256_bytes_and_boundaries() -> Result<()> {
        // 1. Exhaustive all 256 byte values (0x00..=0xFF) + base=241 carry-safety proof.
        {
            let k = 4096usize; // k16 = 256 -> every column can hold all 256 values
            let n = 128usize;
            let k16 = k / 16;
            let mut src = vec![0u8; n * k16];
            for col in 0..n {
                for r in 0..k16 {
                    src[col * k16 + r] = match col {
                        // Col 0: all 256 distinct byte values in ascending order (max exceptions: 241/256 = 94.1%).
                        0 => r as u8,
                        // Col 1: all 256 distinct byte values in reverse order.
                        1 => (255 - r) as u8,
                        // Col 2: optimal window forced to base = 241 ([241..=255]) plus escapes 0x00 and 0xF0.
                        2 => {
                            if r < 240 {
                                241 + (r % 15) as u8
                            } else if r.is_multiple_of(2) {
                                0x00
                            } else {
                                0xF0
                            }
                        }
                        // Col 3: tie-break lowest base (two disjoint windows of equal count at 0x20 and 0x80).
                        3 => {
                            if r < 128 {
                                0x20 + (r % 15) as u8
                            } else {
                                0x80 + (r % 15) as u8
                            }
                        }
                        // Remaining columns: pseudo-random full-byte spread exercising all 256 values.
                        _ => ((r * 131 + col * 17 + 7) & 0xff) as u8,
                    };
                }
            }
            let layout = ts4e128_layout(k, n)?;
            assert_eq!(layout.num_bkts, 128);
            assert_eq!(layout.page_bases_len, 4);
            let (blob, num_exc) = ts4e128_encode_host(&src, k, n)?;
            assert_eq!(blob.len(), ts4e128_blob_bytes(k, n, num_exc)?);
            assert!(blob.len().is_multiple_of(TS4E128_ALIGN_BYTES));
            // Verify col 2 picked exact carry-safe max base 241 and col 3 broke ties to 0x20.
            assert_eq!(blob[layout.nibbles_bytes + 2], 241);
            assert_eq!(blob[layout.nibbles_bytes + 3], 0x20);
            // Every column base must satisfy base <= 241.
            assert!(
                blob[layout.nibbles_bytes..layout.nibbles_bytes + n]
                    .iter()
                    .all(|&b| b <= 241)
            );
            // Verify bit-exact 0-ULP roundtrip across all 32,768 scales.
            let decoded = ts4e128_decode_host(&blob, k, n)?;
            assert_eq!(decoded, transpose_expected_scales(&src, k, n));
        }

        // 2. Multi-page boundary test (num_bkts = 512 -> 4 full 128-bucket pages, hitting page_bases[4])
        //    with high-exception columns proving u16 bkt_off never overflows (<= 127 * 256 = 32,512).
        {
            let k = 4096usize; // num_k32 = 128
            let n = 512usize; // n_tiles = 4 => num_bkts = 512
            let k16 = k / 16;
            let layout = ts4e128_layout(k, n)?;
            assert_eq!(layout.num_bkts, 512);
            assert_eq!(layout.page_bases_len, 8);
            let mut src = vec![0u8; n * k16];
            for col in 0..n {
                for r in 0..k16 {
                    // Stride 16 hits every 16th byte -> at most 1 value per 15-code window (240/256 = 93.75% exc!).
                    src[col * k16 + r] = ((r * 17 + col * 3) & 0xff) as u8;
                }
            }
            let (blob, num_exc) = ts4e128_encode_host(&src, k, n)?;
            assert!(
                num_exc > 110_000,
                "expected >110k exceptions, got {num_exc}"
            );
            assert_eq!(blob.len(), ts4e128_blob_bytes(k, n, num_exc)?);
            // Verify terminal page_bases[4] == num_exc and bkt_off[512] == 0.
            let pb4_off = layout.page_bases_off + 4 * 4;
            let pb4 = u32::from_le_bytes(blob[pb4_off..pb4_off + 4].try_into()?);
            assert_eq!(pb4 as usize, num_exc);
            let end_bkt_off_pos = layout.bkt_off_off + layout.num_bkts * 2;
            let end_bkt_off =
                u16::from_le_bytes(blob[end_bkt_off_pos..end_bkt_off_pos + 2].try_into()?);
            assert_eq!(end_bkt_off, 0);
            let decoded = ts4e128_decode_host(&blob, k, n)?;
            assert_eq!(decoded, transpose_expected_scales(&src, k, n));
        }

        // 3. Explicit 4-lane SWAR carry-safety proof for base = 241, code = 14 / 15.
        {
            const ADD1: u32 = 0x0101_0101;
            const HI4: u32 = 0x1010_1010;
            // All lanes at max safe base=241 and max non-escape code=14 -> 255 in every byte, zero carry.
            let base_word = u32::from_le_bytes([241, 241, 241, 241]);
            let codes_safe = u32::from_le_bytes([14, 14, 14, 14]);
            assert_eq!((codes_safe.wrapping_add(ADD1)) & HI4, 0);
            assert_eq!(
                base_word.wrapping_add(codes_safe),
                u32::from_le_bytes([255, 255, 255, 255])
            );
            // Any lane with code=15 MUST trigger the escape mask (prevents 241+15=256 carry).
            for lane in 0..4usize {
                let mut lanes = [14u8; 4];
                lanes[lane] = 15;
                let codes_esc = u32::from_le_bytes(lanes);
                assert_ne!((codes_esc.wrapping_add(ADD1)) & HI4, 0);
            }
        }

        // 4. Canonical Qwen4 shapes: gate/up [K=2560, N=640] and down [K=640, N=2560].
        for (k, n, expected_exc_off) in [(2560usize, 640usize, 52_672usize), (640, 2560, 54_592)] {
            let layout = ts4e128_layout(k, n)?;
            assert_eq!(layout.nibbles_bytes, 51_200);
            assert_eq!(layout.num_bkts, 400);
            assert_eq!(layout.page_bases_len, 4);
            assert_eq!(layout.bkt_off_bytes_aligned, 816);
            assert_eq!(layout.exc_off, expected_exc_off);

            let k16 = k / 16;
            let mut src = vec![0u8; n * k16];
            for col in 0..n {
                // Typical clustered E4M3 scales with ~2% extreme outliers across full [0x00..=0xFF].
                let col_base = (48 + (col % 194)) as u8; // spans up to 241
                for r in 0..k16 {
                    src[col * k16 + r] = if (col * 31 + r * 13).is_multiple_of(50) {
                        ((col * 7 + r * 29) & 0xff) as u8
                    } else {
                        col_base + ((r + col) % 15) as u8
                    };
                }
            }
            let (blob, num_exc) = ts4e128_encode_host(&src, k, n)?;
            assert_eq!(blob.len(), ts4e128_blob_bytes(k, n, num_exc)?);
            assert!(blob.len().is_multiple_of(TS4E128_ALIGN_BYTES));
            assert!(
                blob.len() < 60_000,
                "expected significant compression below 102,400 B flat, got {}",
                blob.len()
            );
            let decoded = ts4e128_decode_host(&blob, k, n)?;
            assert_eq!(decoded, transpose_expected_scales(&src, k, n));
        }

        Ok(())
    }

    #[test]
    fn gpu_slab_transpose_gate_requires_kernel_tables_and_64_row_alignment() {
        let valid_table = ExpertPtrTable {
            packed_ptrs: DevicePtr(0x1000),
            scale_ptrs: DevicePtr(0x2000),
            scale2_vals: DevicePtr(0x3000),
        };
        let valid_k = KernelHandle(42);
        // Qwen3.8-Flash-Next gate/up (640 x 2560) and down (2560 x 640) qualify.
        assert!(can_use_gpu_slab_transpose(
            true,
            valid_k,
            &valid_table,
            640,
            2560,
            512
        ));
        assert!(can_use_gpu_slab_transpose(
            true,
            valid_k,
            &valid_table,
            2560,
            640,
            512
        ));
        // Opt-out, missing kernel, null pointer table, or unpadded N % 64 != 0 must fall back.
        assert!(!can_use_gpu_slab_transpose(
            false,
            valid_k,
            &valid_table,
            640,
            2560,
            512
        ));
        assert!(!can_use_gpu_slab_transpose(
            true,
            KernelHandle(0),
            &valid_table,
            640,
            2560,
            512
        ));
        assert!(!can_use_gpu_slab_transpose(
            true,
            valid_k,
            &ExpertPtrTable {
                packed_ptrs: DevicePtr::NULL,
                ..valid_table
            },
            640,
            2560,
            512
        ));
        assert!(!can_use_gpu_slab_transpose(
            true,
            valid_k,
            &ExpertPtrTable {
                scale_ptrs: DevicePtr::NULL,
                ..valid_table
            },
            640,
            2560,
            512
        ));
        assert!(!can_use_gpu_slab_transpose(
            true,
            valid_k,
            &valid_table,
            32,
            64,
            2
        ));
        assert!(!can_use_gpu_slab_transpose(
            true,
            valid_k,
            &valid_table,
            640,
            24,
            512
        ));
        assert!(!can_use_gpu_slab_transpose(
            true,
            valid_k,
            &valid_table,
            640,
            2560,
            0
        ));
    }
}
