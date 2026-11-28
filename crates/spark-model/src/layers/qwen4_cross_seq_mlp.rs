// SPDX-License-Identifier: AGPL-3.0-only

//! MLP half of a cross-sequence Qwen4 K-row verify layer
//! (`ATLAS_QWEN4_BATCHED_K5`).
//!
//! The single-sequence K5 verifier prepares the MLP hyperconnection for its
//! K rows, runs the exact `_t_lanes` MoE rows and injects the saved scales.
//! The cross-sequence frame does exactly that over every row of every
//! sequence at once: the exact hyperconnection tiers preserve K1 lane
//! ownership per row, the MoE is tiled into exact <=9-row `_t_lanes` tiles
//! (`MoeLayer::forward_qwen4_native_target_rows`), and each row's injection
//! scales live in its own residual tail, so no row depends on which other
//! rows share its launch.

use std::cell::Cell;
use std::sync::Once;

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layer::{ForwardContext, Qwen4CrossSeqFrame};
use crate::layers::deterministic_batch::{
    DeterministicBatchPolicy, deterministic_batch_policy, deterministic_batch_policy_or_disabled,
};
use crate::layers::{FfnComponent, Qwen4HyperConnection};

/// `ATLAS_QWEN4_BATCHED_HC_FUSE_INJECT_GN` (default ON unless `"0"`): fuse
/// `inject_saved_batched` with the immediately following `group_norm` via
/// [`Qwen4HyperConnection::prepare_batched_after_inject`] across intra-layer
/// (SSM/Attn -> MLP) and inter-layer (Layer `i` MLP -> Layer `i+1` Attn/SSM and
/// last layer -> `qwen4_final_mixer`) boundaries. Strictly 0-ULP and 100%
/// CUDA-graph safe.
pub(crate) const BATCHED_HC_FUSE_INJECT_GN_ENV: &str = "ATLAS_QWEN4_BATCHED_HC_FUSE_INJECT_GN";

#[inline]
#[must_use]
pub(crate) fn parse_batched_hc_fuse_inject_gn(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// Resolve `ATLAS_QWEN4_BATCHED_HC_FUSE_INJECT_GN` under `det_batch`:
/// `DeterministicBatchPolicy::Enabled` disables cross-sequence HC inject/GN
/// deferral so every layer preserves canonical unfused `B=1` ordering.
#[inline]
#[must_use]
pub(crate) const fn batched_hc_fuse_inject_gn_enabled_with_policy(
    det_batch: DeterministicBatchPolicy,
    env_enabled: bool,
) -> bool {
    !det_batch.is_enabled() && env_enabled
}

#[inline]
#[must_use]
fn batched_hc_fuse_inject_gn_env_enabled() -> bool {
    let read_live = || {
        parse_batched_hc_fuse_inject_gn(
            std::env::var(BATCHED_HC_FUSE_INJECT_GN_ENV).ok().as_deref(),
        )
    };
    #[cfg(not(test))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

#[inline]
#[must_use]
pub(crate) fn batched_hc_fuse_inject_gn_enabled() -> bool {
    batched_hc_fuse_inject_gn_enabled_with_policy(
        deterministic_batch_policy_or_disabled(),
        batched_hc_fuse_inject_gn_env_enabled(),
    )
}

pub(crate) fn log_batched_hc_fuse_inject_gn_engaged(rows: usize) {
    static ENGAGED: Once = Once::new();
    ENGAGED.call_once(|| {
        tracing::info!(
            selector = BATCHED_HC_FUSE_INJECT_GN_ENV,
            rows,
            "QWEN4_BATCHED_HC_FUSE_INJECT_GN_ENGAGED"
        );
    });
}

/// Deferred cross-sequence MLP `inject_saved_batched` waiting to fuse into the
/// next hyperconnection's `prepare_batched_after_inject` (next layer's
/// `attn_hyper` or `qwen4_final_mixer`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingBatchedHcInject {
    pub hyper: DevicePtr,
    pub core: DevicePtr,
    pub residual: DevicePtr,
    pub rows: usize,
    pub stream: u64,
}

impl PendingBatchedHcInject {
    #[inline]
    #[must_use]
    pub(crate) const fn matches(
        self,
        hyper: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> bool {
        self.hyper.0 == hyper.0
            && self.residual.0 == residual.0
            && self.rows == rows
            && self.stream == stream
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CrossSeqHcFuseState {
    defer_mlp_tail_allowed: bool,
    pending: Option<PendingBatchedHcInject>,
}

thread_local! {
    static CROSS_SEQ_HC_FUSE: Cell<CrossSeqHcFuseState> = const {
        Cell::new(CrossSeqHcFuseState {
            defer_mlp_tail_allowed: false,
            pending: None,
        })
    };
}

/// Scoped inter-layer deferred-inject state for `verify_qwen4_batched.rs`.
///
/// Dropping the guard restores the prior state and discards any unconsumed
/// entry so an early error or aborted CUDA graph capture never leaks stale
/// device pointers into a subsequent eager fallback or verify round.
pub(crate) struct CrossSeqHcFuseScope {
    prev: CrossSeqHcFuseState,
}

impl CrossSeqHcFuseScope {
    #[must_use]
    pub(crate) fn enter() -> Self {
        let prev = CROSS_SEQ_HC_FUSE.with(|cell| {
            cell.replace(CrossSeqHcFuseState {
                defer_mlp_tail_allowed: false,
                pending: None,
            })
        });
        Self { prev }
    }

    /// Arm or disarm trailing MLP `inject_saved_batched` deferral for the next
    /// layer call inside this scope. Must remain `false` for Layer 0 and every
    /// DFlash target-hidden capture tap layer.
    pub(crate) fn set_defer_mlp_tail(&mut self, allow: bool) {
        CROSS_SEQ_HC_FUSE.with(|cell| {
            let mut state = cell.get();
            state.defer_mlp_tail_allowed = allow;
            cell.set(state);
        });
    }
}

impl Drop for CrossSeqHcFuseScope {
    fn drop(&mut self) {
        CROSS_SEQ_HC_FUSE.with(|cell| cell.set(self.prev));
    }
}

#[inline]
pub(crate) fn take_pending_inject() -> Option<PendingBatchedHcInject> {
    CROSS_SEQ_HC_FUSE.with(|cell| {
        let mut state = cell.get();
        let pending = state.pending.take();
        cell.set(state);
        pending
    })
}

#[inline]
fn defer_mlp_tail_allowed() -> bool {
    CROSS_SEQ_HC_FUSE.with(|cell| cell.get().defer_mlp_tail_allowed)
}

fn defer_pending_inject(entry: PendingBatchedHcInject) -> Result<()> {
    CROSS_SEQ_HC_FUSE.with(|cell| {
        let mut state = cell.get();
        if let Some(prior) = state.pending {
            bail!(
                "pending cross-seq HC inject {prior:?} was not consumed before deferring {entry:?}"
            );
        }
        state.pending = Some(entry);
        cell.set(state);
        Ok(())
    })
}

/// `ATLAS_QWEN4_BATCHED_K5_MOE_TILE`: `frame` (default) tiles the whole
/// frame into exact <=9-row `_t_lanes` launches that may span sequences;
/// `seq` runs one launch per sequence, the exact tile C1 runs.
///
/// Both are exact by construction: in `moe_shared_expert_fused_t.cu` every
/// member row of a `_t_lanes_rows` launch owns its accumulator
/// (`acc[RMAX][OPT]`, lines 682/761), walks the same k16 order and reduces
/// through the same per-row tree in reused shared memory (`part`, lines
/// 678/745; the down kernel stages each row separately in `s_act_rows`,
/// line 746), with no atomics; which unit owns a shared expert changes only
/// which block computes a row, never its arithmetic
/// (`t_lanes_rows_members`, line 615). The blend runs one block per token
/// (`moe_shared_expert_fused_batch3.cu:337`) and the router uses the exact
/// multi-row GEMV tiers. `seq` exists as the bisection control for the
/// exactness gate.
pub(crate) const MOE_TILE_ENV: &str = "ATLAS_QWEN4_BATCHED_K5_MOE_TILE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoeTiling {
    Frame,
    Sequence,
}

pub(crate) fn parse_moe_tiling(raw: Option<&str>) -> std::result::Result<MoeTiling, &'static str> {
    match raw {
        None | Some("frame") => Ok(MoeTiling::Frame),
        Some("seq") => Ok(MoeTiling::Sequence),
        Some(_) => Err("ATLAS_QWEN4_BATCHED_K5_MOE_TILE must be frame or seq"),
    }
}

fn read_moe_tiling_live() -> std::result::Result<MoeTiling, String> {
    let raw = match std::env::var(MOE_TILE_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!("{MOE_TILE_ENV} must be valid UTF-8"));
        }
    };
    parse_moe_tiling(raw.as_deref()).map_err(str::to_owned)
}

/// Resolve MoE tiling under `det_batch`: `DeterministicBatchPolicy::Enabled`
/// forces sequence-aligned tiling (`MoeTiling::Sequence`) so no MoE launch
/// spans across distinct sequences.
#[inline]
#[must_use]
pub(crate) const fn moe_tiling_with_policy(
    det_batch: DeterministicBatchPolicy,
    env_tiling: MoeTiling,
) -> MoeTiling {
    if det_batch.is_enabled() {
        MoeTiling::Sequence
    } else {
        env_tiling
    }
}

pub(crate) fn moe_tiling() -> Result<MoeTiling> {
    let det_batch = deterministic_batch_policy()?;
    let env_tiling = {
        #[cfg(not(test))]
        {
            static CACHED: std::sync::OnceLock<std::result::Result<MoeTiling, String>> =
                std::sync::OnceLock::new();
            CACHED
                .get_or_init(read_moe_tiling_live)
                .as_ref()
                .copied()
                .map_err(|msg| anyhow::Error::msg(msg.clone()))?
        }
        #[cfg(test)]
        {
            read_moe_tiling_live().map_err(anyhow::Error::msg)?
        }
    };
    Ok(moe_tiling_with_policy(det_batch, env_tiling))
}

/// `ATLAS_QWEN4_K5_BATCH_HYPER` (default ON unless `"0"`, cached via `OnceLock`
/// outside unit tests).
#[inline]
fn k5_batch_hyper_enabled() -> bool {
    let read_live = || std::env::var("ATLAS_QWEN4_K5_BATCH_HYPER").ok().as_deref() != Some("0");
    #[cfg(not(test))]
    {
        static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GATE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Host-only admission of the MLP half. No device memory is touched.
pub(crate) fn preflight(
    mlp_hyper: Option<&Qwen4HyperConnection>,
    ffn: &FfnComponent,
    frame: Qwen4CrossSeqFrame,
    ctx: &ForwardContext,
) -> Result<()> {
    let _det_batch = deterministic_batch_policy()?;
    let rows = frame.rows();
    let mlp_hyper = mlp_hyper.ok_or_else(|| {
        anyhow::anyhow!("cross-sequence Qwen4 verify requires an MLP hyperconnection")
    })?;
    ensure!(
        mlp_hyper.residual_width() == ctx.config.residual_width(),
        "cross-sequence Qwen4 MLP hyperconnection geometry mismatch"
    );
    // The single-sequence verifier must itself be on the exact rows route
    // this frame reproduces: batched hyper prep and the native exact MoE.
    ensure!(
        k5_batch_hyper_enabled(),
        "cross-sequence Qwen4 verify requires batched hyperconnection rows \
         (unset ATLAS_QWEN4_K5_BATCH_HYPER=0)"
    );
    ensure!(
        crate::layers::qwen4_k5_native_moe_selected(),
        "cross-sequence Qwen4 verify requires ATLAS_QWEN4_K5_NATIVE_MOE=1 so every \
         MoE row uses the exact _t_lanes kernels"
    );
    ensure!(
        crate::layers::qwen4_hybrid_rows(frame.k()),
        "cross-sequence Qwen4 verify K{} is not an admitted exact hybrid width",
        frame.k()
    );
    mlp_hyper.preflight_qwen4_native_batched(rows, true)?;
    ffn.preflight_qwen4_native_exact_rows(rows, ctx)?;
    moe_tiling()?;
    let core_bytes = rows
        .checked_mul(ctx.config.hidden_size)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("cross-sequence MLP core extent overflow"))?;
    let hyper_bytes = rows
        .checked_mul(ctx.config.residual_width())
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("cross-sequence MLP hyper extent overflow"))?;
    let arena = ctx.buffers.sizes();
    ensure!(
        hyper_bytes <= arena.hidden_states
            && hyper_bytes <= arena.residual
            && hyper_bytes <= arena.qkv_output
            && core_bytes <= arena.norm_output
            && core_bytes <= arena.moe_output,
        "cross-sequence Qwen4 MLP rows exceed the serving arenas"
    );
    Ok(())
}

/// Execute the MLP half over every row of the frame.
///
/// When `deferred_mixer_out` is `Some(core)`, fuses the preceding mixer's
/// `inject_saved_batched` with `mlp_hyper`'s `group_norm` via
/// [`Qwen4HyperConnection::prepare_batched_after_inject`] (0-ULP, CUDA-graph
/// safe). When inter-layer deferral is armed inside [`CrossSeqHcFuseScope`] and
/// `ATLAS_QWEN4_BATCHED_HC_FUSE_INJECT_GN != 0`, defers the trailing
/// `mlp_hyper.inject_saved_batched` into the next layer's `attn_hyper` or
/// `qwen4_final_mixer`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    mlp_hyper: &Qwen4HyperConnection,
    ffn: &FfnComponent,
    hidden: DevicePtr,
    residual: DevicePtr,
    deferred_mixer_out: Option<DevicePtr>,
    frame: Qwen4CrossSeqFrame,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let rows = frame.rows();
    let eps = ctx.config.rms_norm_eps as f32;
    let ffn_inputs = if let Some(mixer_out) = deferred_mixer_out {
        log_batched_hc_fuse_inject_gn_engaged(rows);
        mlp_hyper.prepare_batched_after_inject(
            hidden,
            mixer_out,
            residual,
            rows,
            ctx.buffers,
            ctx.gpu,
            eps,
            stream,
        )?
    } else {
        mlp_hyper.prepare_batched(hidden, residual, rows, ctx.buffers, ctx.gpu, eps, stream)?
    };
    spark_runtime::cublaslt::stage_timer::mark("x_hc_in2", stream);
    match moe_tiling()? {
        MoeTiling::Frame => ffn.forward_qwen4_native_target_rows(ffn_inputs, rows, ctx, stream)?,
        MoeTiling::Sequence => {
            ffn.forward_qwen4_seq_aligned_rows(ffn_inputs, frame.seqs(), frame.k(), ctx, stream)?
        }
    }
    spark_runtime::cublaslt::stage_timer::mark("x_moe", stream);
    let moe_out = ctx.buffers.moe_output();
    let det_batch = deterministic_batch_policy()?;
    if batched_hc_fuse_inject_gn_enabled_with_policy(
        det_batch,
        batched_hc_fuse_inject_gn_env_enabled(),
    ) && defer_mlp_tail_allowed()
    {
        defer_pending_inject(PendingBatchedHcInject {
            hyper: hidden,
            core: moe_out,
            residual,
            rows,
            stream,
        })?;
    } else {
        mlp_hyper.inject_saved_batched(hidden, moe_out, residual, rows, ctx.gpu, stream)?;
    }
    spark_runtime::cublaslt::stage_timer::mark("x_inject", stream);
    Ok(())
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::DevicePtr;

    use super::{
        CrossSeqHcFuseScope, MoeTiling, PendingBatchedHcInject, defer_mlp_tail_allowed,
        defer_pending_inject, parse_batched_hc_fuse_inject_gn, parse_moe_tiling,
        take_pending_inject,
    };

    #[test]
    fn moe_tiling_selector_is_strict_and_defaults_to_frame_tiles() {
        assert_eq!(parse_moe_tiling(None), Ok(MoeTiling::Frame));
        assert_eq!(parse_moe_tiling(Some("frame")), Ok(MoeTiling::Frame));
        assert_eq!(parse_moe_tiling(Some("seq")), Ok(MoeTiling::Sequence));
        for hostile in ["", "1", "Seq", "sequence"] {
            assert!(parse_moe_tiling(Some(hostile)).is_err(), "{hostile:?}");
        }
    }

    #[test]
    fn batched_hc_fuse_inject_gn_selector_defaults_on_and_zero_disables() {
        assert!(parse_batched_hc_fuse_inject_gn(None));
        assert!(parse_batched_hc_fuse_inject_gn(Some("")));
        assert!(parse_batched_hc_fuse_inject_gn(Some("1")));
        assert!(parse_batched_hc_fuse_inject_gn(Some("true")));
        assert!(!parse_batched_hc_fuse_inject_gn(Some("0")));
        assert!(!parse_batched_hc_fuse_inject_gn(Some(" 0 ")));
    }

    #[test]
    fn cross_seq_hc_fuse_scope_defers_consumes_rejects_double_defer_and_cleans_on_drop() {
        assert!(!defer_mlp_tail_allowed());
        assert_eq!(take_pending_inject(), None);

        let entry = PendingBatchedHcInject {
            hyper: DevicePtr(0x1000),
            core: DevicePtr(0x2000),
            residual: DevicePtr(0x3000),
            rows: 16,
            stream: 7,
        };
        assert!(entry.matches(DevicePtr(0x1000), DevicePtr(0x3000), 16, 7));
        assert!(!entry.matches(DevicePtr(0x1004), DevicePtr(0x3000), 16, 7));
        assert!(!entry.matches(DevicePtr(0x1000), DevicePtr(0x3004), 16, 7));
        assert!(!entry.matches(DevicePtr(0x1000), DevicePtr(0x3000), 20, 7));
        assert!(!entry.matches(DevicePtr(0x1000), DevicePtr(0x3000), 16, 8));

        {
            let mut scope = CrossSeqHcFuseScope::enter();
            assert!(!defer_mlp_tail_allowed());
            assert_eq!(take_pending_inject(), None);

            scope.set_defer_mlp_tail(true);
            assert!(defer_mlp_tail_allowed());

            defer_pending_inject(entry).expect("first defer succeeds");
            assert!(
                defer_pending_inject(entry).is_err(),
                "second defer before consume must fail closed"
            );
            assert_eq!(take_pending_inject(), Some(entry));
            assert_eq!(take_pending_inject(), None);

            // Re-defer and let Drop clean up (simulating an early error/capture abort).
            defer_pending_inject(entry).expect("defer after consume succeeds");
        }

        assert!(!defer_mlp_tail_allowed());
        assert_eq!(
            take_pending_inject(),
            None,
            "scope drop must discard stale pending inject"
        );
    }

    #[test]
    fn cached_mlp_env_helpers_preserve_defaults_toggles_and_fail_closed_errors() {
        use super::{
            BATCHED_HC_FUSE_INJECT_GN_ENV, MOE_TILE_ENV, batched_hc_fuse_inject_gn_enabled,
            k5_batch_hyper_enabled, moe_tiling,
        };
        use crate::layers::qwen4_verify_lanes::{ScopedEnvVar, test_env_lock};

        let _lock = test_env_lock();

        // `batched_hc_fuse_inject_gn_enabled`: default ON, "0" disables.
        {
            let _g = ScopedEnvVar::remove(BATCHED_HC_FUSE_INJECT_GN_ENV);
            assert!(batched_hc_fuse_inject_gn_enabled());
        }
        {
            let _g = ScopedEnvVar::set(BATCHED_HC_FUSE_INJECT_GN_ENV, "1");
            assert!(batched_hc_fuse_inject_gn_enabled());
        }
        {
            let _g = ScopedEnvVar::set(BATCHED_HC_FUSE_INJECT_GN_ENV, "0");
            assert!(!batched_hc_fuse_inject_gn_enabled());
        }

        // `moe_tiling`: default Frame, "frame" -> Frame, "seq" -> Sequence, invalid -> Err.
        {
            let _g = ScopedEnvVar::remove(MOE_TILE_ENV);
            assert_eq!(moe_tiling().unwrap(), MoeTiling::Frame);
        }
        {
            let _g = ScopedEnvVar::set(MOE_TILE_ENV, "frame");
            assert_eq!(moe_tiling().unwrap(), MoeTiling::Frame);
        }
        {
            let _g = ScopedEnvVar::set(MOE_TILE_ENV, "seq");
            assert_eq!(moe_tiling().unwrap(), MoeTiling::Sequence);
        }
        {
            let _g = ScopedEnvVar::set(MOE_TILE_ENV, "invalid");
            assert!(moe_tiling().is_err());
        }
        {
            let _det = ScopedEnvVar::set(
                crate::layers::deterministic_batch::DETERMINISTIC_BATCH_ENV,
                "1",
            );
            let _hc = ScopedEnvVar::set(BATCHED_HC_FUSE_INJECT_GN_ENV, "1");
            let _tile = ScopedEnvVar::set(MOE_TILE_ENV, "frame");
            assert!(!batched_hc_fuse_inject_gn_enabled());
            assert_eq!(moe_tiling().unwrap(), MoeTiling::Sequence);
        }
        {
            let _det = ScopedEnvVar::set(
                crate::layers::deterministic_batch::DETERMINISTIC_BATCH_ENV,
                "bogus",
            );
            assert!(
                moe_tiling().is_err(),
                "invalid ATLAS_DETERMINISTIC_BATCH must fail closed in moe_tiling"
            );
        }

        // `k5_batch_hyper_enabled`: default ON, "0" disables.
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_K5_BATCH_HYPER");
            assert!(k5_batch_hyper_enabled());
        }
        {
            let _g = ScopedEnvVar::set("ATLAS_QWEN4_K5_BATCH_HYPER", "1");
            assert!(k5_batch_hyper_enabled());
        }
        {
            let _g = ScopedEnvVar::set("ATLAS_QWEN4_K5_BATCH_HYPER", "0");
            assert!(!k5_batch_hyper_enabled());
        }
    }
}
