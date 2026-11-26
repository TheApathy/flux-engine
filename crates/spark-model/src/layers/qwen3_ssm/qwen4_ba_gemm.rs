// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in GEMM route for the Flash-Next prefill GDN gates
//! (`ATLAS_QWEN4_PREFILL_BA_GEMM=1`, default off).
//!
//! `dense_gemv_ba_gates_batchn` launches one GEMV block per (row, 4 outputs)
//! and re-reads the [96, 2560] BA weight per row: ~0.56 ms per SSM layer at
//! 2048 rows, ~80 ms per 8192-row chunk over 36 layers. Here one cuBLASLt
//! GEMM writes the FP32 [rows, 96] BA projection and one small kernel applies
//! the same gate transforms. Not bit-exact (GEMM accumulation order).

use std::sync::{Mutex, OnceLock};

use anyhow::Context;
use spark_runtime::cublaslt::GemmDtype;
use spark_runtime::kernel_args::KernelLaunch;

use super::*;

pub(super) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_BA_GEMM";
pub(super) const FUSED_ACT_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_BA_FUSED_ACT";
pub(super) const DENSE_SELECTOR: &str = "ATLAS_PREFILL_BA_GEMM";
const DENSE_BA_GEMM_SM_THRESHOLD: u32 = 64;
const DENSE_BA_GEMM_MIN_ROWS: usize = 32;

pub(super) fn selected() -> bool {
    static SEL: OnceLock<bool> = OnceLock::new();
    *SEL.get_or_init(|| std::env::var(SELECTOR).ok().as_deref() == Some("1"))
}

pub(super) fn fused_act_selected() -> bool {
    static SEL: OnceLock<bool> = OnceLock::new();
    selected()
        && *SEL.get_or_init(|| std::env::var(FUSED_ACT_SELECTOR).ok().as_deref() != Some("0"))
}

/// Resolve `ATLAS_PREFILL_BA_GEMM` for dense prefill (`Qwen3.8-27B` etc.):
/// explicit `"1"`/`"true"` enables, explicit `"0"`/`"false"` disables, and
/// unset auto-enables on discrete high-SM GPUs (`sm_count > 64`) while staying
/// default-OFF on GB10 (`<= 64`).
pub(super) fn resolve_dense_ba_gemm_enabled(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        Some(_) => false,
        None => sm_count > DENSE_BA_GEMM_SM_THRESHOLD,
    }
}

pub(super) fn dense_ba_gemm_enabled() -> bool {
    static SEL: OnceLock<bool> = OnceLock::new();
    *SEL.get_or_init(|| {
        let raw = std::env::var(DENSE_SELECTOR).ok();
        resolve_dense_ba_gemm_enabled(raw.as_deref(), atlas_core::device::active_sm_count())
    })
}

/// Pure eligibility predicate for routing dense SSM prefill BA projection +
/// GDN gate transform through cuBLASLt Tensor Core GEMM instead of the
/// per-token `dense_gemm_ba_gates_prefill` GEMV kernel.
#[expect(
    clippy::too_many_arguments,
    reason = "pure const policy predicate mirroring every BA GEMM alignment and shape check"
)]
pub(super) const fn dense_ba_gemm_eligible(
    enabled: bool,
    rows: usize,
    hidden_size: usize,
    ba_size: usize,
    nk: usize,
    nv: usize,
    input_ptr: u64,
    weight_ptr: u64,
) -> bool {
    enabled
        && rows >= DENSE_BA_GEMM_MIN_ROWS
        && hidden_size > 0
        && hidden_size.is_multiple_of(8)
        && nk > 0
        && nv > 0
        && nv.is_multiple_of(nk)
        && ba_size == nv * 2
        && input_ptr != 0
        && weight_ptr != 0
        && input_ptr.is_multiple_of(16)
        && weight_ptr.is_multiple_of(16)
}

fn kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static V2_ON: OnceLock<bool> = OnceLock::new();
    let v2 = *V2_ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PREFILL_BA_GEMM_V2")
            .ok()
            .as_deref()
            != Some("0")
    });
    if v2 {
        static K_V2: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
        return K_V2
            .get_or_init(|| {
                gpu.kernel("qwen4_wide_glue", "qwen4_ba_gates_from_f32_v2")
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map_err(|e| anyhow::anyhow!("{SELECTOR}: v2 kernel unavailable: {e}"));
    }
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_wide_glue", "qwen4_ba_gates_from_f32")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: kernel unavailable: {e}"))
}

fn planar_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_wide_glue", "qwen4_ba_gates_from_f32_planar")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: planar kernel unavailable: {e}"))
}

/// True when `qwen4_ba_gates_gemm` writes planar `[gate(rows * 48) | beta(rows * 48)]`
/// directly for the FlashInfer GDN bridge, skipping `k_deinterleave_gate_beta`.
pub(super) fn uses_planar_fi_gates(rows: usize) -> bool {
    selected()
        && super::qwen4_gdn_fi_policy::use_fi(super::qwen4_gdn_fi::selected(), rows)
        && std::env::var_os("ATLAS_QWEN4_GDN_FI_DUMP_DIR").is_none()
}

/// Grow-only FP32 [rows, GATES] scratch, stream-ordered on the prefill stream.
fn scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static S: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
    let mut guard = S
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: scratch poisoned"))?;
    if let Some((ptr, cap)) = *guard
        && cap >= bytes
    {
        return Ok(ptr);
    }
    // The old block is leaked on purpose (grow-only, in-flight safety).
    let cap = bytes.next_multiple_of(1 << 20);
    let ptr = gpu
        .alloc(cap)
        .with_context(|| format!("{SELECTOR}: alloc {cap} B"))?;
    *guard = Some((ptr, cap));
    Ok(ptr)
}

impl Qwen3SsmLayer {
    /// Writes `[gate(nv) | beta(nv)]` FP32 rows (stride `2 * nv`) into `gates`
    /// (or planar `[gate(rows * nv) | beta(rows * nv)]` when
    /// [`uses_planar_fi_gates`] is active). Parameterized by
    /// `ctx.config.hidden_size` so both Qwen4 (`H=2560`) and dense Qwen3.8
    /// (`H=5120`) share the same cuBLASLt Tensor Core path.
    pub(super) fn qwen4_ba_gates_gemm(
        &self,
        input: DevicePtr,
        gates: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let s = super::qwen4_prefill_exact_plan::SsmShape::from_heads(
            ctx.config.linear_num_key_heads,
            ctx.config.linear_num_value_heads,
        );
        let h = u32::try_from(ctx.config.hidden_size)?;
        let planar = uses_planar_fi_gates(rows);
        let k = if planar {
            planar_kernel(ctx.gpu)?
        } else {
            kernel(ctx.gpu)?
        };
        let ba = scratch(ctx.gpu, rows * s.gates * 4)?;
        let rows_u32 = u32::try_from(rows)?;
        spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
            input.0,
            h,
            self.ssm.in_proj_ba.weight.0,
            ba.0,
            s.gates as u32,
            rows_u32,
            s.gates as u32,
            h,
            GemmDtype::Bf16,
            GemmDtype::F32,
            false,
            rows_u32,
            stream,
        )?;
        let last_arg = if planar { rows_u32 } else { s.gates as u32 };
        KernelLaunch::new(ctx.gpu, k)
            .grid([rows_u32, 1, 1])
            .block([s.gates as u32, 1, 1])
            .arg_ptr(ba)
            .arg_ptr(self.ssm.a_log.weight)
            .arg_ptr(self.ssm.dt_bias.weight)
            .arg_ptr(gates)
            .arg_u32(s.gates as u32)
            .arg_u32((s.nv / s.nk) as u32)
            .arg_u32(last_arg)
            .launch(stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "SSM_PREFILL_BA_GEMM_ENGAGED selector={SELECTOR} rows={rows} hidden={h} planar={planar} non_bit_exact"
            );
        });
        Ok(())
    }

    /// Tensor-core BA projection + GDN gate transform for dense SSM prefill
    /// (e.g. Qwen3.8-27B `H=5120, nv=48, nk=16` on SM120/SM103).
    ///
    /// Priority 1: `qwen4_wide_glue::qwen4_ba_gates_from_f32[_v2]` with
    /// cuBLASLt `Bf16 -> F32` (when shipped in the bundle).
    /// Priority 2: cuBLASLt `Bf16 -> Bf16` + `common/ssm_preprocess_multi_seq`'s
    /// `compute_gdn_gates_multi_seq` (always present in dense bundles), writing
    /// the exact `[gate(nv) | beta(nv)]` FP32 layout with stride `2 * nv`.
    /// Returns `Ok(false)` when disabled, ineligible, or missing kernels so
    /// `trait_prefill` falls back cleanly to `ops::dense_gemm_ba_gates_prefill`.
    pub(super) fn try_dense_ba_gates_gemm(
        &self,
        input: DevicePtr,
        gates: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let h = ctx.config.hidden_size;
        let nk = ctx.config.linear_num_key_heads;
        let nv = ctx.config.linear_num_value_heads;
        let ba_size = ctx.config.ssm_ba_size();
        if !dense_ba_gemm_eligible(
            dense_ba_gemm_enabled(),
            rows,
            h,
            ba_size,
            nk,
            nv,
            input.0,
            self.ssm.in_proj_ba.weight.0,
        ) {
            return Ok(false);
        }
        let h_u32 = u32::try_from(h)?;
        let rows_u32 = u32::try_from(rows)?;
        let ba_u32 = u32::try_from(ba_size)?;
        let vpg_u32 = u32::try_from(nv / nk)?;

        if let Ok(wide_k) = kernel(ctx.gpu) {
            let ba = scratch(ctx.gpu, rows * ba_size * std::mem::size_of::<f32>())?;
            spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
                input.0,
                h_u32,
                self.ssm.in_proj_ba.weight.0,
                ba.0,
                ba_u32,
                rows_u32,
                ba_u32,
                h_u32,
                GemmDtype::Bf16,
                GemmDtype::F32,
                false,
                rows_u32,
                stream,
            )?;
            KernelLaunch::new(ctx.gpu, wide_k)
                .grid([rows_u32, 1, 1])
                .block([ba_u32, 1, 1])
                .arg_ptr(ba)
                .arg_ptr(self.ssm.a_log.weight)
                .arg_ptr(self.ssm.dt_bias.weight)
                .arg_ptr(gates)
                .arg_u32(ba_u32)
                .arg_u32(vpg_u32)
                .arg_u32(ba_u32)
                .launch(stream)?;
            static LOGGED_WIDE: std::sync::Once = std::sync::Once::new();
            LOGGED_WIDE.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_BA_GEMM_ENGAGED selector={DENSE_SELECTOR} backend=cublaslt_f32_wide_glue rows={rows} hidden={h} ba={ba_size}"
                );
            });
            return Ok(true);
        }

        if self.compute_gdn_gates_multi_seq_k.0 == 0 {
            return Ok(false);
        }
        let ba = scratch(ctx.gpu, rows * ba_size * 2)?;
        spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
            input.0,
            h_u32,
            self.ssm.in_proj_ba.weight.0,
            ba.0,
            ba_u32,
            rows_u32,
            ba_u32,
            h_u32,
            GemmDtype::Bf16,
            GemmDtype::Bf16,
            false,
            rows_u32,
            stream,
        )?;
        let beta_out = gates.offset(nv * std::mem::size_of::<f32>());
        ops::compute_gdn_gates_multi_seq(
            ctx.gpu,
            self.compute_gdn_gates_multi_seq_k,
            ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates,
            beta_out,
            rows_u32,
            u32::try_from(nv)?,
            u32::try_from(nk)?,
            vpg_u32,
            ba_u32,
            ba_u32,
            stream,
        )?;
        static LOGGED_BF16: std::sync::Once = std::sync::Once::new();
        LOGGED_BF16.call_once(|| {
            tracing::info!(
                "SSM_PREFILL_BA_GEMM_ENGAGED selector={DENSE_SELECTOR} backend=cublaslt_bf16_gdn_gates rows={rows} hidden={h} ba={ba_size}"
            );
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{dense_ba_gemm_eligible, resolve_dense_ba_gemm_enabled};

    #[test]
    fn dense_ba_gemm_auto_enables_above_64_sms_and_honors_opt_out() {
        assert!(!resolve_dense_ba_gemm_enabled(None, 48));
        assert!(!resolve_dense_ba_gemm_enabled(None, 64));
        assert!(resolve_dense_ba_gemm_enabled(Some("1"), 48));
        assert!(resolve_dense_ba_gemm_enabled(Some("true"), 48));

        for sm in [65u32, 110, 170, 188] {
            assert!(resolve_dense_ba_gemm_enabled(None, sm), "sm={sm}");
            assert!(!resolve_dense_ba_gemm_enabled(Some("0"), sm), "sm={sm}");
            assert!(!resolve_dense_ba_gemm_enabled(Some("false"), sm), "sm={sm}");
        }
    }

    #[test]
    fn dense_ba_gemm_eligibility_supports_qwen38_27b_and_qwen4_hidden_sizes() {
        // Qwen3.8-27B: H=5120, nk=16, nv=48, ba_size=96.
        assert!(dense_ba_gemm_eligible(
            true, 8192, 5120, 96, 16, 48, 0x1000, 0x2000
        ));
        // Qwen4-Exp: H=2560, nk=16, nv=48, ba_size=96.
        assert!(dense_ba_gemm_eligible(
            true, 2048, 2560, 96, 16, 48, 0x1000, 0x2000
        ));
        // Short prompts, disabled policy, or misaligned pointers fall back cleanly.
        assert!(!dense_ba_gemm_eligible(
            true, 31, 5120, 96, 16, 48, 0x1000, 0x2000
        ));
        assert!(!dense_ba_gemm_eligible(
            false, 8192, 5120, 96, 16, 48, 0x1000, 0x2000
        ));
        assert!(!dense_ba_gemm_eligible(
            true, 8192, 5120, 96, 16, 48, 0x1008, 0x2000
        ));
        assert!(!dense_ba_gemm_eligible(
            true, 8192, 5120, 64, 16, 48, 0x1000, 0x2000
        ));
    }
}
