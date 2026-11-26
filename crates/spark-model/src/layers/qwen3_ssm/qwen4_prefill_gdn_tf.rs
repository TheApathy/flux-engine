// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off FP32 token-serial GDN recurrence for the Flash-Next exact
//! prefill, after TensorFold's `chain_kernel` (`gdn_prefill_tf_chain.cu`).
//!
//! `ATLAS_QWEN4_PREFILL_GDN_TF=1` (128 value rows / 256 threads per CTA, one
//! CTA per value head) or `=2` (64 rows / 128 threads, two CTAs per head)
//! replaces only the recurrence kernel of the exact route: it reads the same
//! FP32 conv rows and gate rows, writes the same FP32 GDN rows, and commits
//! the same [HV][K][V] FP32 H state, so conv, gated norm and everything
//! downstream are unchanged. Exact-class (FP32 throughout; only the
//! association of the 128-term dot products differs from regfinal), not
//! bit-exact. Two threads share one value row, which halves the serial
//! dependency chain per token against the one-thread-per-column regfinal
//! kernels.

use spark_runtime::kernel_args::KernelLaunch;

use super::qwen4_prefill_exact_plan::SsmShape;
use super::*;

const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_TF";
const MODULE: &str = "gdn_prefill_tf_chain";
const HEAD_DIM: u32 = 128;

/// Value rows per CTA (the kernel entry point and its launch shape).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TfRows {
    R128,
    R64,
}

impl TfRows {
    const fn rows(self) -> u32 {
        match self {
            Self::R128 => 128,
            Self::R64 => 64,
        }
    }

    const fn symbol(self) -> &'static str {
        match self {
            Self::R128 => "qwen4_gdn_prefill_tf_chain_r128",
            Self::R64 => "qwen4_gdn_prefill_tf_chain_r64",
        }
    }
}

fn parse(value: Option<&std::ffi::OsStr>) -> Result<Option<TfRows>, &'static str> {
    match value.map(|v| v.to_str()) {
        None => Ok(None),
        Some(Some("0")) => Ok(None),
        Some(Some("1")) => Ok(Some(TfRows::R128)),
        Some(Some("2")) => Ok(Some(TfRows::R64)),
        Some(_) => Err("must be absent, 0, 1 (128 rows/CTA) or 2 (64 rows/CTA)"),
    }
}

pub(crate) fn selected() -> Result<Option<TfRows>> {
    static SEL: std::sync::OnceLock<Result<Option<TfRows>, &'static str>> =
        std::sync::OnceLock::new();
    (*SEL.get_or_init(|| parse(std::env::var_os(SELECTOR).as_deref())))
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: {error}"))
}

fn kernel(gpu: &dyn GpuBackend, variant: TfRows) -> Result<KernelHandle> {
    type Slot = std::sync::OnceLock<Result<KernelHandle, String>>;
    static R128: Slot = Slot::new();
    static R64: Slot = Slot::new();
    let slot = match variant {
        TfRows::R128 => &R128,
        TfRows::R64 => &R64,
    };
    slot.get_or_init(|| {
        gpu.kernel(MODULE, variant.symbol())
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| {
        anyhow::anyhow!(
            "{SELECTOR}: {MODULE}::{} unavailable: {e}",
            variant.symbol()
        )
    })
}

impl Qwen3SsmLayer {
    /// Runs the recurrence on the exact route's buffers when selected.
    /// `Ok(false)` means not selected: the caller runs nosnap (or regfinal when
    /// `ATLAS_QWEN4_PREFILL_GDN_LAZYFINAL` is set).
    /// `conv_rows` holds FP32 [Q|K|V] rows of stride QKVZ, `gates` the FP32
    /// [gate(nv)|beta(nv)] rows, and the FP32 output rows (stride QKVZ) go to
    /// `gdn_rows`, exactly as `ops::gdn_decode_f32_sequence` is called there.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_prefill_qwen4_gdn_tf(
        &self,
        state: &SsmLayerState,
        conv_rows: DevicePtr,
        gates: DevicePtr,
        gdn_rows: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(variant) = selected()? else {
            return Ok(false);
        };
        let s = SsmShape::from_heads(
            ctx.config.linear_num_key_heads,
            ctx.config.linear_num_value_heads,
        );
        let key_heads = s.nk as u32;
        let value_heads = s.nv as u32;
        let k = kernel(ctx.gpu, variant)?;
        let split = HEAD_DIM / variant.rows();
        KernelLaunch::new(ctx.gpu, k)
            .grid([value_heads * split, 1, 1])
            .block([2 * variant.rows(), 1, 1])
            .arg_ptr(state.h_state)
            .arg_ptr(conv_rows)
            .arg_ptr(conv_rows.offset(s.key_dim * 4))
            .arg_ptr(conv_rows.offset(s.key_dim * 2 * 4))
            .arg_ptr(gates)
            .arg_ptr(gates.offset(s.nv * 4))
            .arg_ptr(gdn_rows)
            .arg_u32(rows as u32)
            .arg_u32(key_heads)
            .arg_u32(value_heads)
            .arg_u32(s.qkvz as u32)
            .arg_u32(s.qkvz as u32)
            .arg_u32(s.gates as u32)
            .arg_u32(s.qkvz as u32)
            .launch(stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "QWEN4_PREFILL_GDN_TF_ENGAGED selector={SELECTOR} kernel={} rows={rows}",
                variant.symbol()
            );
        });
        Ok(true)
    }
}
