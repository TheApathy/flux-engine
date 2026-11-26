// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in FlashInfer FA2 attention core for Flash-Next wide-chunk prefill
//! (`ATLAS_QWEN4_PREFILL_ATTN_FI=1`, default off; needs a build with
//! `FLASHINFER_HOME` set, see `crates/spark-model/build.rs`).
//!
//! One causal ragged launch per attention layer and chunk: a first chunk
//! attends over its own compact Q/K/V; a continuation chunk attends over the
//! whole prefix plus itself, gathered from the paged BF16 cache. Replaces the
//! in-tree flash core (first chunk) and `inferspark_prefill_paged_64`
//! (continuations, ~25 TF/s). Not bit-exact versus either: different tiling
//! and accumulation order.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_FI";

pub(crate) fn selected() -> bool {
    static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| std::env::var(SELECTOR).ok().as_deref() == Some("1"))
}

#[cfg(atlas_qwen4_fi_attn)]
mod ffi {
    unsafe extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn atlas_qwen4_fi_attn_prefill(
            q: *const core::ffi::c_void,
            k_chunk: *const core::ffi::c_void,
            v_chunk: *const core::ffi::c_void,
            k_pool: *const core::ffi::c_void,
            v_pool: *const core::ffi::c_void,
            block_table: *const u32,
            block_size: u32,
            start: u32,
            rows: u32,
            nq: u32,
            nkv: u32,
            sm_scale: f32,
            o: *mut core::ffi::c_void,
            stream: *mut core::ffi::c_void,
        ) -> i32;
    }
}

/// Inputs of one attention-core call. `q`/`k`/`v` are the chunk's compact
/// `[rows, heads, 256]` slabs; the pools and `block_table` describe the
/// sequence's paged BF16 cache, which already holds the chunk's own K/V.
#[cfg_attr(not(atlas_qwen4_fi_attn), allow(dead_code))]
pub(crate) struct Call {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub block_table: DevicePtr,
    pub block_size: usize,
    pub start: usize,
    pub rows: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sm_scale: f32,
    pub output: DevicePtr,
}

fn validate_call(call: &Call) -> Result<()> {
    anyhow::ensure!(
        !call.q.is_null()
            && !call.k.is_null()
            && !call.v.is_null()
            && !call.k_pool.is_null()
            && !call.v_pool.is_null()
            && !call.output.is_null(),
        "{SELECTOR}: requires non-null Q/K/V, paged-KV, and output buffers"
    );
    anyhow::ensure!(
        call.start == 0 || !call.block_table.is_null(),
        "{SELECTOR}: continuation requires a non-null block table"
    );
    anyhow::ensure!(
        call.block_size > 0
            && call.rows > 0
            && call.num_q_heads > 0
            && call.num_kv_heads > 0
            && call.head_dim == 256,
        "{SELECTOR}: requires positive block/head/row dimensions and head_dim 256"
    );
    anyhow::ensure!(
        call.sm_scale.is_finite() && call.sm_scale > 0.0,
        "{SELECTOR}: requires a positive finite attention scale"
    );
    // The CUDA ABI narrows every dimension to u32. Validate even in a host
    // build without FlashInfer so a requested-but-unavailable path reports
    // malformed input before its build-capability error.
    u32::try_from(call.block_size)?;
    u32::try_from(call.start)?;
    u32::try_from(call.rows)?;
    u32::try_from(call.num_q_heads)?;
    u32::try_from(call.num_kv_heads)?;
    Ok(())
}

pub(crate) fn run(call: &Call, stream: u64) -> Result<()> {
    validate_call(call)?;
    #[cfg(atlas_qwen4_fi_attn)]
    {
        // SAFETY: `validate_call` verified all required device pointers are non-null
        // (including `block_table` when `start > 0`), all dimensions are positive and
        // fit in `u32`, `head_dim == 256`, and `sm_scale` is positive and finite. The
        // caller guarantees the device allocations cover the declared shapes and
        // `stream` is a valid CUDA stream handle.
        let rc = unsafe {
            ffi::atlas_qwen4_fi_attn_prefill(
                call.q.0 as usize as *const core::ffi::c_void,
                call.k.0 as usize as *const core::ffi::c_void,
                call.v.0 as usize as *const core::ffi::c_void,
                call.k_pool.0 as usize as *const core::ffi::c_void,
                call.v_pool.0 as usize as *const core::ffi::c_void,
                call.block_table.0 as usize as *const u32,
                u32::try_from(call.block_size)?,
                u32::try_from(call.start)?,
                u32::try_from(call.rows)?,
                u32::try_from(call.num_q_heads)?,
                u32::try_from(call.num_kv_heads)?,
                call.sm_scale,
                call.output.0 as usize as *mut core::ffi::c_void,
                stream as usize as *mut core::ffi::c_void,
            )
        };
        anyhow::ensure!(
            rc == 0,
            "{SELECTOR}: atlas_qwen4_fi_attn_prefill failed rc={rc} start={} rows={}",
            call.start,
            call.rows
        );
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "ATTN_PREFILL_FI_ENGAGED selector={SELECTOR} start={} rows={} core=flashinfer_fa2_ragged_hd256 non_bit_exact",
                call.start,
                call.rows
            )
        });
        Ok(())
    }
    #[cfg(not(atlas_qwen4_fi_attn))]
    {
        let _ = stream;
        anyhow::bail!(
            "{SELECTOR}=1 but this build has no FlashInfer attention (set FLASHINFER_HOME)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_call() -> Call {
        Call {
            q: DevicePtr(2),
            k: DevicePtr(4),
            v: DevicePtr(6),
            k_pool: DevicePtr(8),
            v_pool: DevicePtr(10),
            block_table: DevicePtr::NULL,
            block_size: 64,
            start: 0,
            rows: 16,
            num_q_heads: 32,
            num_kv_heads: 4,
            head_dim: 256,
            sm_scale: 0.0625,
            output: DevicePtr(12),
        }
    }

    #[test]
    fn first_chunk_accepts_a_null_block_table() {
        validate_call(&valid_call()).unwrap();
    }

    #[test]
    fn continuation_requires_a_block_table() {
        let mut call = valid_call();
        call.start = 16;
        assert!(
            validate_call(&call)
                .unwrap_err()
                .to_string()
                .contains("continuation requires a non-null block table")
        );
        call.block_table = DevicePtr(16);
        validate_call(&call).unwrap();
    }

    #[test]
    fn pointer_and_scale_contracts_fail_closed() {
        let mut call = valid_call();
        call.output = DevicePtr::NULL;
        assert!(validate_call(&call).is_err());

        call.output = DevicePtr(12);
        call.sm_scale = f32::NAN;
        assert!(validate_call(&call).is_err());
    }
}
