// SPDX-License-Identifier: AGPL-3.0-only

//! `DenseFfnLayer` struct, weight containers, constructor, workspace allocation, and sparsity/sparse-draft paths.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[cfg(feature = "cuda")]
use super::cublaslt_proj::CublasLtNvfp4Scales;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use super::e2m1_flashinfer::FlashinferFfnPrefill;
#[cfg(feature = "cuda")]
use super::e2m1_flashinfer::try_load_cublaslt_checkpoint_scales;
use super::e2m1_flashinfer::{E2m1CheckpointScales, E2m1Scratch, load_e2m1_checkpoint_scales};
use super::policy::{
    FFN_KGAMMA_WIDE_MAX_ROWS, ffn_tc_active_for_rows, kgamma_m128_active, kgamma_wide_active,
    supports_warp_nvfp4_mma,
};
#[cfg(feature = "cuda")]
use super::policy::{PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD, gpu_sm_count};
use super::sm120_coalesced::warm_coalesced_nvfp4_sm120_kernels;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

pub struct DenseFfnWeights {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

/// BF16 dense MLP weights — alternative to NVFP4 for precision-sensitive
/// models (Gemma-4-31B). Each is `[N, K]` row-major BF16. When installed
/// on a `DenseFfnLayer` via `set_bf16_weights`, the forward paths
/// dispatch to `dense_gemv_bf16` / `dense_gemm_bf16` instead of the
/// w4a16 NVFP4 kernels. Costs ~3.4 GB extra GPU memory on Gemma-4-31B
/// (3 × hidden×intermediate × 2 bytes) vs NVFP4's 0.5 bytes/weight.
pub struct DenseFfnWeightsBf16 {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Activation function for gated FFN (SiLU for Qwen/Llama, GELU for Gemma-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FfnActivation {
    SiLU,
    GeLU,
}

pub struct DenseFfnLayer {
    pub weights: DenseFfnWeights,
    pub(super) activation: FfnActivation,
    pub(super) w4a16_gemv: KernelHandle,
    pub(super) w4a16_gemv_dual: KernelHandle,
    pub(super) w4a16_gemv_silu_input: KernelHandle,
    /// Exact K1-arithmetic-order W4 FFN kernels for dynamic M=2..=32.
    /// Missing selected-tier handles fail closed to independent K1 rows.
    pub(super) exact_ffn_kernels: ops::W4a16ExactFfnKernels,
    /// Optional full-M16 gate/up projections. The diagnostic route is atomic;
    /// any missing handle retains the current split-M8 implementation.
    pub(super) exact_ffn_lowreg_m16: ops::W4a16ExactFfnLowregM16Kernels,
    /// Single-warp-per-output M=1 decode GEMVs — lossless, 8 outputs per
    /// 256-thread block instead of 4, no cross-warp smem round-trip. Loaded
    /// via `try_kernel` so a kernel cache built before these symbols existed
    /// still links; `KernelHandle(0)` on a miss falls back to the 64-thread
    /// base kernels, which are bit-identical.
    pub(super) w4a16_gemv_sw: KernelHandle,
    pub(super) w4a16_gemv_dual_sw: KernelHandle,
    pub(super) w4a16_gemv_silu_input_sw: KernelHandle,
    /// `ATLAS_NO_GEMV_SW != "1"`, cached at construction.
    pub(super) gemv_sw: bool,
    pub(super) w4a16_gemv_dual_batch2: KernelHandle,
    pub(super) w4a16_gemv_dual_batch3: KernelHandle,
    /// Tuned dual-batch3 variant: fuses gate+up into the SAME CTA so the
    /// 3-token activation vector is loaded once per CTA. Gated behind the
    /// `ATLAS_FFN_DUAL_TUNED=1` env var; falls back to the baseline kernel
    /// when off. Loaded via `try_kernel` so older built kernel caches that
    /// pre-date this symbol still link.
    pub(super) w4a16_gemv_dual_batch3_tuned: KernelHandle,
    pub(super) w4a16_gemv_batch2: KernelHandle,
    pub(super) w4a16_gemv_batch3: KernelHandle,
    pub(super) w4a16_gemm: KernelHandle,
    /// cp.async double-buffered byte-exact shadow of `w4a16_gemm` for the
    /// prefill FFN path. Same dequant arithmetic + MMA order; only the load
    /// pipeline differs. Loaded via `try_kernel`; handle 0 causes an eligible
    /// explicit `ATLAS_PREFILL_FFN_PIPE=1` request to fail closed with `bail!`.
    pub(super) w4a16_gemm_pipe: KernelHandle,
    /// Exact up-projection + SiLU(gate) epilogue variant used only by the
    /// opt-in large-M prefill pipe route. It preserves the BF16 up round trip
    /// while avoiding the full up activation buffer and standalone SiLU
    /// launch. Missing symbol fails closed on eligible explicit
    /// `ATLAS_PREFILL_FFN_FUSED_EPILOGUE=1` requests.
    pub(super) w4a16_gemm_pipe_silu_mul: KernelHandle,
    /// Exact large-M gate+up+SiLU fusion. Both projection accumulators retain
    /// their BF16 materialization boundary, but A is loaded once and neither
    /// intermediate activation is written to global memory.
    pub(super) w4a16_gemm_pipe_dual: KernelHandle,
    /// SiLU(gate)*up or GELU(gate)*up depending on activation.
    pub(super) act_mul: KernelHandle,
    /// BF16 dense MLP weights — when `Some`, all forward paths use the
    /// `dense_gemv_bf16` / `dense_gemm_bf16` kernels instead of w4a16
    /// NVFP4. Falls back to the NVFP4 weights when `None`. Set via
    /// `set_bf16_weights`. Used by Gemma-4 dense to avoid the structural
    /// NVFP4 attention drift on greedy code generation (the fib test's
    /// broken-indentation pattern).
    pub(super) bf16_weights: Option<DenseFfnWeightsBf16>,
    pub(super) dense_gemv_bf16_k: KernelHandle,
    pub(super) dense_gemm_bf16_k: KernelHandle,
    /// Transposed (`nvfp4_t` layout) FFN projections — populated when
    /// `ATLAS_FFN_M16_TRANSPOSED` is active (default ON unless `=0`) and the
    /// loader successfully built the transposed copies via
    /// `QuantizedWeight::transpose_for_gemm`. When `Some`, `forward_kgamma`
    /// routes gate/up/down through the `w4a16_gemm_t_m16` (M_TILE=16) kernel
    /// which has near-zero MMA accumulator waste at M=γ+1 (typically 17).
    /// Falls back to the non-transposed `w4a16_gemm` (M_TILE=64) when `None`
    /// OR when the `w4a16_gemm_t_m16` kernel symbol is missing.
    pub(super) gate_proj_t: Option<QuantizedWeight>,
    pub(super) up_proj_t: Option<QuantizedWeight>,
    pub(super) down_proj_t: Option<QuantizedWeight>,
    /// The transposed copies were installed for the M>=128 prefill route only
    /// (`set_prefill_transposed_weights`), so that route does not also need
    /// `w4a16_gemm_t_m16`. `has_transposed_ffn()` still does, which keeps every
    /// decode / verify consumer of the copies off on targets without it.
    pub(super) prefill_t_m128_only: bool,
    /// `w4a16_gemm_t_m16` — small-M (M_TILE=16) transposed-weight GEMM.
    /// Loaded via `try_kernel`; handle is 0 when the symbol is missing
    /// (older PTX caches), in which case `forward_kgamma` always uses the
    /// non-transposed fallback regardless of the transposed weights.
    pub(super) w4a16_gemm_t_m16: KernelHandle,
    /// `w4a16_gemm_t_m16_n64` — small-M (M_TILE=16), small-N (N_TILE=64)
    /// transposed-weight GEMM, tuned for the K=3 MTP verify path on dense
    /// Qwen3.6-27B (M=3 padded to MMA-16). At intermediate=17408 the N=128
    /// parent only fields ~136 CTAs/projection (~2.8 CTAs/SM on GB10's 48 SMs),
    /// limiting occupancy; N=64 doubles the grid to ~272 CTAs/projection
    /// at half the per-CTA work. Loaded via `try_kernel`; handle 0 falls
    /// back to the GEMV path silently.
    pub(super) w4a16_gemm_t_m16_n64: KernelHandle,
    /// `w4a16_gemm_t_m128` — large-M (M_TILE=128, N_TILE=128)
    /// transposed-weight GEMM. Loaded via `try_kernel`; handle is 0 when
    /// the symbol is missing. Used by `forward_prefill` when
    /// `ATLAS_PREFILL_FFN_FAST` is enabled (default ON unless `=0`; explicit
    /// `=1` fails closed when absent) AND transposed weights are installed
    /// AND M >= 128. Mirrors the attention `w4a16_gemm_t_m128_k`
    /// dispatch in `qwen3_attention/prefill_weights.rs`. Designed for
    /// large-M prefill: kernel comment claims ~2x speedup over
    /// `w4a16_gemm_t` at ISL>128. For Qwen3.6-27B prefill at M=3575,
    /// N=17408 (gate/up dual): grid=(136, 28)=3808 CTAs vs the default
    /// (272, 56)=15232 CTAs — 4x fewer CTAs but 4x more work per CTA,
    /// and ~2x less weight DRAM traffic.
    pub(super) w4a16_gemm_t_m128: KernelHandle,
    /// `w4a16_gemm_t_m32_n64` — DFlash K=17 verify specialization:
    /// single B read (one 32-row M-tile) × 272 CTAs (N_TILE=64).
    /// Loaded via `try_kernel`; 0 falls back to m128/m16.
    pub(super) w4a16_gemm_t_m32_n64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu` — FUSED gate_proj + up_proj +
    /// SiLU·mul for the K=γ verify path. Loads the shared [M,K] input
    /// tile once, streams BOTH transposed weights (gate + up), and writes
    /// only the fused silu(gate)*up [M,N] activation in one launch (vs the
    /// baseline's two m32_n64 GEMMs + standalone `moe_silu_mul`). Gated by
    /// `ATLAS_FFN_FUSED_GATEUP`. Loaded via `try_kernel`; handle 0 keeps
    /// the split gate/up path.
    pub(super) w4a16_gemm_t_m32_n64_gateup_silu: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu_pipe` — DEQUANT-IN-REGISTERS fork of
    /// the fused gate+up+SiLU kernel (`ATLAS_DEQUANT_PIPE=1`). Byte-exact
    /// with `w4a16_gemm_t_m32_n64_gateup_silu` (same shape, accumulation
    /// order, and BF16 round-trips) but the NVFP4→FP8 dequant runs in
    /// registers immediately before each MMA instead of via a `smem_B_fp8`
    /// staging array — dropping the 2nd per-K-step `__syncthreads`, shrinking
    /// SMEM ~27% (10.9 KB vs 15.0 KB → higher occupancy), and using
    /// `cp.async.wait_group<1>` so the next tile's load overlaps the current
    /// dequant+MMA. Loaded via `try_kernel`; handle 0 keeps the staged fused
    /// kernel as silent fallback.
    pub(super) w4a16_gemm_t_m32_n64_gateup_silu_pipe: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64` — K_STEP=64 fork of the
    /// `_pipe` register-dequant kernel (`ATLAS_GATEUP_K64=1`). Doubles the
    /// K-tile from 32 to 64 elements: 80 K-loop iterations vs 160, halving
    /// sync count and loop overhead. Each step issues two m16n8k32 MMAs per
    /// accumulator (K[0..31] then K[32..63]) and 2× the cp.async volume
    /// (6 KB vs 3 KB per stage), so the background load overlaps more of the
    /// inline dequant+compute work. Takes priority over ATLAS_DEQUANT_PIPE.
    /// Requires K divisible by 64 (hidden_size 5120 qualifies). Handle 0
    /// silently falls back through _pipe → staged fused kernel.
    pub(super) w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64: KernelHandle,
    /// `w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64` — `N_TILE=32, K_STEP=64`
    /// wide-SM fork (`ATLAS_GATEUP_N32`, auto on `> 128`-SM GPUs). Doubles CTA
    /// count (272 -> 544 at `N=17408`), halves registers (48 vs 96), and is
    /// 0-ULP byte-exact with the N64 fused kernels.
    pub(super) w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_splitk` — split-K variant of the above for
    /// the DFlash K=17 verify `down_proj` ([M=17,N=5120,K=16384]). The
    /// single-slice kernel fields only 80 CTAs (N=5120/64) and is
    /// occupancy-starved on the long K=16384 loop (~91 GB/s vs gate/up's
    /// ~163). Split-K multiplies CTAs by k_splits into an FP32 workspace,
    /// then `reduce_splitk_f32_to_bf16` sums + downcasts. Gated by
    /// `ATLAS_FFN_DOWN_SPLITK` (default 0 on `<= 64` SMs, 4 on `65..=128` SMs,
    /// 8 on `> 128` SMs; 0/1 disables). Loaded via
    /// `try_kernel` — handle 0 keeps the single-slice m32_n64 path.
    pub(super) w4a16_gemm_t_m32_n64_splitk: KernelHandle,
    /// `reduce_splitk_f32_to_bf16` — companion reduce kernel for the
    /// split-K down_proj. Sums the k_splits FP32 partial bands → BF16.
    pub(super) reduce_splitk_k: KernelHandle,
    /// Lazily-allocated FP32 split-K workspace [k_splits, M, N].
    /// `Mutex` because `forward_kgamma` takes `&self`.
    pub(super) splitk_workspace: Mutex<Option<DevicePtr>>,
    /// `w4a16_gemm_t_m128_v2` — 8-warp (blockDim 256) shadow of
    /// `w4a16_gemm_t_m128`. Same 2-stage cp.async pipeline + same SMEM
    /// footprint (~29.8KB → 3 CTAs/SM), but parallelizes chunk 0 and
    /// chunk 1 MMA computation across warps {0-3} and {4-7} instead of
    /// serializing both chunks on 4 warps. Yields 2× more warps/SM (768
    /// vs 384) → more MMA pipeline slots in flight. Originally a
    /// MiniMax-only kernel (kernels/gb10/minimax-m2-229b/nvfp4/
    /// w4a16_gemm_v2.cu) — copied verbatim into the qwen3.6-27b target
    /// so the FFN prefill GEMM can route through it. Gated by
    /// `ATLAS_FFN_M128_V2=1`. Loaded via `try_kernel`; handle 0 keeps
    /// the v1 path as silent fallback.
    pub(super) w4a16_gemm_t_m128_v2: KernelHandle,
    /// `fp8_gemm_t_m128` — large-M FP8×FP8 GEMM kernel for pre-dequanted
    /// FFN weights. Loaded via `try_kernel`. When set together with
    /// `gate_fp8`/`up_fp8`/`down_fp8` (installed by
    /// `predequant_for_prefill`), the FFN prefill bypasses NVFP4 dequant
    /// entirely — saving 1 __syncthreads + the 16-iteration dequant
    /// phase per K-step inside `w4a16_gemm_t_m128`. Mirrors the
    /// attention `predequant_for_prefill` + `fp8_gemm_n128_m128`
    /// pattern (`qwen3_attention/prefill_weights.rs`).
    pub(super) fp8_gemm_t_m128_k: KernelHandle,
    /// Pre-dequanted FP8 [N, K] gate weight. `Some` only when
    /// `ATLAS_FFN_PREDEQUANT_FP8=1` is set at startup and the
    /// `predequant_nvfp4_to_fp8` + `fp8_gemm_t_m128` kernels are present.
    /// Memory cost: N×K bytes per projection (e.g. 17408×5120 = 89 MB
    /// for gate/up, 5120×17408 = 89 MB for down → ~17 GB per Qwen3.6-27B
    /// run, ~270 MB per layer × 64 layers). Roughly DOUBLES the FFN
    /// weight footprint vs NVFP4-only — gate by intent.
    pub(super) gate_fp8: Option<DevicePtr>,
    pub(super) up_fp8: Option<DevicePtr>,
    pub(super) down_fp8: Option<DevicePtr>,
    /// W4A4 NVFP4×NVFP4 native tensor-core GEMM
    /// (`nvfp4_cutlass::nvfp4_nvfp4_gemm_t_m64`). Loaded via `try_kernel`
    /// — handle 0 causes an eligible explicit `ATLAS_E2M1_GEMM=1` request to
    /// fail closed with `bail!`.
    pub(super) nvfp4_gemm_k: KernelHandle,
    /// K-major M=128 W4A4 shadow consuming `transpose_for_gemm` buffers.
    /// Handle 0 makes an eligible explicit `ATLAS_E2M1_KMAJOR=1` request
    /// fail closed; the ordinary row-major W4A4 route is otherwise unchanged.
    pub(super) nvfp4_gemm_kmajor_m128_k: KernelHandle,
    /// Sixteen-warp M=256 K-major long-prefill shadow. An eligible explicit
    /// `ATLAS_E2M1_KMAJOR_M256=1` request fails closed when this is absent;
    /// M128 remains the K-major fallback below the M256 saturation threshold.
    pub(super) nvfp4_gemm_kmajor_m256_k: KernelHandle,
    /// `quantize_nvfp4::nvfp4_global_absmax` — per-tensor absmax scan
    /// used to derive the activation `scale2` for the W4A4 path.
    pub(super) nvfp4_absmax_k: KernelHandle,
    /// `quantize_nvfp4::quantize_bf16_to_nvfp4` — per-row E2M1 quantizer
    /// used to convert BF16 activations to the W4A4 GEMM input layout.
    pub(super) nvfp4_quantize_k: KernelHandle,
    /// Fused SiLU(gate)*up -> NVFP4 down-input quantizer. Handle 0 makes an
    /// eligible explicit `ATLAS_E2M1_SILU_QUANT=1` request fail closed.
    pub(super) nvfp4_silu_quantize_k: KernelHandle,
    /// `quantize_bf16_to_nvfp4_cutlass::quantize_bf16_to_nvfp4_atlas_128x4` —
    /// BF16 → NVFP4 activation quantizer emitting `128x4`-swizzled `UE4M3`
    /// scales for native cuBLASLt (`VEC16_UE4M3`) GEMMs on SM103/SM120.
    pub(super) nvfp4_quantize_128x4_k: KernelHandle,
    /// Single-pass vectorized Blackwell PTX sibling (`quantize_bf16_to_nvfp4_atlas_128x4_ptx`)
    /// using `uint4` loads, `__nv_fp8_e4m3`, `cvt.rn.satfinite.e2m1x2.f32`, and 64-bit stores.
    pub(super) nvfp4_quantize_128x4_ptx_k: KernelHandle,
    /// Fused SwiGLU + vectorized Blackwell PTX NVFP4 quantizer
    /// (`silu_mul_quantize_nvfp4_atlas_128x4`) for cuBLASLt `down_proj` prefill.
    pub(super) nvfp4_silu_quantize_128x4_k: KernelHandle,
    /// Zero-sync device-dynamic `128x4` activation quantizer
    /// (`quantize_bf16_to_nvfp4_atlas_128x4_from_absmax`) consuming `global_max`.
    pub(super) nvfp4_quantize_128x4_from_absmax_k: KernelHandle,
    /// Device-resident combined alpha builder (`nvfp4_dynamic_combined_alpha`)
    /// producing `scale2_a * weight_scale_2` on stream without host sync.
    pub(super) nvfp4_dynamic_combined_alpha_k: KernelHandle,
    /// Fused merged `[gate | up]` SwiGLU + `128x4` NVFP4 quantizer
    /// (`quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4`) for merged `[gate; up]` cuBLASLt.
    pub(super) nvfp4_merged_silu_quantize_128x4_k: KernelHandle,
    /// Vectorized Blackwell PTX merged `[gate | up]` SwiGLU + `128x4` NVFP4 quantizer
    /// (`quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx`), preferred on `sm_count > 64`.
    pub(super) nvfp4_merged_silu_quantize_128x4_ptx_k: KernelHandle,
    /// Fused `residual_add_rms_norm_quantize_nvfp4_128x4` kernel for static-scale
    /// prefill FFN on `sm_count > 64`.
    pub(super) nvfp4_residual_add_rms_norm_quant_128x4_k: KernelHandle,
    /// Lazily-swizzled `128x4` `UE4M3` weight block scales for the cuBLASLt
    /// NVFP4 prefill path (`SM103`/`SM120` auto or `ATLAS_CUBLASLT_NVFP4=1`).
    #[cfg(feature = "cuda")]
    pub(super) cublaslt_nvfp4_scales: Mutex<Option<CublasLtNvfp4Scales>>,
    /// Opportunistically loaded ModelOpt checkpoint `input_scale` scalars used
    /// solely by the cuBLASLt NVFP4 prefill path so SM120/SM103 avoid per-layer
    /// dynamic absmax stream syncs when checkpoint scales are available.
    #[cfg(feature = "cuda")]
    pub(super) cublaslt_checkpoint_scales: Option<E2m1CheckpointScales>,
    /// Scratch arena for the W4A4 fast path. Holds packed activation nibbles,
    /// per-group FP8 scales, and absmax scratch. FlashInfer preparation
    /// allocates its maximum admitted shape at construction; other routes
    /// allocate lazily and resize if M or K grows. `Mutex` because
    /// `forward_prefill` takes `&self`. `Arc` so the layers of one model can
    /// share a single arena (`ATLAS_MEMLEAN_SHARED_SCRATCH`).
    pub(super) e2m1_scratch: Arc<Mutex<Option<E2m1Scratch>>>,
    /// Serializes every full launch group that uses `e2m1_scratch` and binds
    /// it to one CUDA stream. The kernels are asynchronous, so allocation-only
    /// locking is insufficient: a later caller on another stream could
    /// overwrite or free operands still in flight. Reuse on the same stream is
    /// safe because CUDA preserves enqueue order. Shared together with
    /// `e2m1_scratch`, so every layer using one arena is bound to one stream.
    pub(super) e2m1_use_stream: Arc<Mutex<Option<u64>>>,
    /// The arena above is shared with other layers
    /// ([`Self::adopt_shared_e2m1_scratch`]).
    pub(super) e2m1_scratch_shared: bool,
    /// Optional ModelOpt calibration scales read once during layer
    /// construction. `ATLAS_E2M1_STATIC_SCALE=1` uses these with the existing
    /// on-device quantizer, eliminating runtime absmax scans and host syncs.
    /// This is a different activation-quantization contract from dynamic
    /// absmax and remains default-off pending output-quality qualification.
    pub(super) e2m1_checkpoint_scales: Option<E2m1CheckpointScales>,
    /// Exact Atlas-quantization / FlashInfer-GEMM prefill route. Constructed
    /// only by the Qwen3.8 loader after every original checkpoint operand,
    /// scalar, physical 128x4 scale copy, native symbol and zero-workspace
    /// tactic has passed admission. Default-unrouted on every other build.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) flashinfer_ffn_prefill: Option<FlashinferFfnPrefill>,
    /// `ffn_sparsity_measure` — TEAL-style activation-sparsity observer for
    /// the sparsity-drafted self-speculation feasibility gate
    /// (`ATLAS_MEASURE_FFN_SPARSITY=1`). Loaded via `try_kernel`; handle 0
    /// disables the measurement silently (older kernel caches).
    pub(super) ffn_sparsity_measure_k: KernelHandle,
    /// Lazily-allocated device counter buffers for the sparsity measurement.
    /// `Mutex` because `forward` takes `&self`. Allocated on first measured
    /// `forward` call (gpu.alloc is illegal during graph capture, but the
    /// self-spec draft + measurement run EAGER — see the gate docs). None
    /// until the first measured forward.
    sparsity_meas: Mutex<Option<SparsityMeas>>,
    /// W3 (3-bit weight) FFN projections — mixed-precision byte-reduction
    /// lane. `Some` only when `ATLAS_FFN_W3_LAYERS` names this layer AND
    /// the repacked sidecar (`ATLAS_FFN_W3_SIDECAR`, loaded via
    /// `weight_map::w3_sidecar`) contained its tensors; installed by the
    /// loader via `set_w3_weights`. GEMV layout `[N, 3K/8]` — used by the
    /// single-token `forward` SiLU path (dual gate/up + fused SiLU down).
    /// Cuts packed FFN weight bytes 25% vs NVFP4 on a weight-bandwidth-
    /// bound decode. NOT md5-gated (weights differ from W4 by
    /// construction) — quality is protected by the ABBA eval gate; the
    /// default path (gate off / no sidecar) stays byte-identical.
    pub(super) w3_weights: Option<DenseFfnWeights>,
    /// Transposed W3 copies (`[3K/8, N_pad64]`) for the K=γ verify GEMM
    /// path (`w3a16_gemm_t_m32_n64`). Built host-side by the sidecar
    /// loader. `forward_kgamma` routes gate/up/down through the W3 GEMM
    /// when set (superseding the W4 m32/fused/split-K variants on this
    /// layer). Other paths (prefill, K=2/3 batched GEMV) intentionally
    /// stay on the retained W4 weights — they are not weight-bandwidth-
    /// bound the same way and keep their higher-precision copies.
    pub(super) w3_weights_t: Option<DenseFfnWeights>,
    /// `w3a16_gemv_dual` — W3 gate+up dual GEMV. `try_kernel`; handle 0
    /// (older PTX caches) disables the W3 GEMV path silently.
    pub(super) w3a16_gemv_dual_k: KernelHandle,
    /// `w3a16_gemv_silu_input` — W3 fused SiLU-input down GEMV.
    pub(super) w3a16_gemv_silu_input_k: KernelHandle,
    /// `w3a16_gemm_t_m32_n64` — W3 clone of the m32_n64 verify GEMM.
    pub(super) w3a16_gemm_t_m32_n64_k: KernelHandle,
    /// `ffn_build_keep_chunks` — on-device keep-chunk selector for the SPARSE
    /// self-spec DRAFT path (`ATLAS_SELF_SPEC_SPARSE=1`). Handle 0 disables.
    pub(super) ffn_build_keep_chunks_k: KernelHandle,
    /// `w4a16_gemv_sparse_cols` — column-sparse GEMV for the SPARSE draft
    /// path. Handle 0 disables the sparse draft (falls back to dense GEMV).
    pub(super) w4a16_gemv_sparse_cols_k: KernelHandle,
    /// Lazily-allocated per-layer keep_idx / keep_len device scratch for the
    /// sparse draft path. `Mutex` because `forward_draft_sparse` takes `&self`.
    sparse_draft_scratch: Mutex<Option<SparseDraftScratch>>,
}

/// Per-layer device counter buffers for the FFN activation-sparsity
/// measurement. Two sites (gate/up input + down input), each with a
/// `NUM_THRESH`-entry u32 histogram (below-threshold counts, atomically
/// accumulated) and a 2-entry u32 `count` ([0]=rows seen, [1]=elements seen).
struct SparsityMeas {
    /// Histogram for site 0 (gate/up input, K=hidden). `NUM_THRESH` u32s.
    hist_gateup: DevicePtr,
    /// [rows, elements] u32 counter for site 0.
    count_gateup: DevicePtr,
    /// Histogram for site 1 (down input, K=intermediate). `NUM_THRESH` u32s.
    hist_down: DevicePtr,
    /// [rows, elements] u32 counter for site 1.
    count_down: DevicePtr,
    /// Dedicated BF16 scratch [1, intermediate] into which the observer
    /// recomputes `silu(gate)*up` for the DOWN-input site. This is a SEPARATE
    /// buffer from the token-stream's `gate_out`/`up_out` — the fused down
    /// GEMV (`w4a16_gemv_silu_input`) applies SiLU internally and never
    /// materialises a standalone silu'd vector, so the observer computes its
    /// own copy here WITHOUT touching the buffers the fused kernel reads.
    /// Keeps the measurement a pure reader (token stream byte-identical).
    meas_silu: DevicePtr,
    /// Number of measured `forward` calls since process start on this layer.
    /// Drives the periodic D2H dump cadence.
    steps: u64,
}

/// Per-layer device scratch for the SPARSE self-spec draft: `keep_idx`
/// (surviving k8-chunk indices, capacity `K_max/8`) + `keep_len` (1 u32).
struct SparseDraftScratch {
    /// Surviving k8-chunk index list, sized for the largest K this layer
    /// might sparsify (down input K=intermediate). `keep_idx.len == K/8`.
    keep_idx: DevicePtr,
    /// Single u32: number of surviving chunks written by
    /// `ffn_build_keep_chunks`.
    keep_len: DevicePtr,
    /// Capacity in k8 chunks (= K_max/8) the `keep_idx` buffer can hold.
    cap_chunks: usize,
}

impl DenseFfnLayer {
    pub fn new(weights: DenseFfnWeights, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_activation(weights, FfnActivation::SiLU, gpu)
    }

    pub fn new_with_activation(
        weights: DenseFfnWeights,
        activation: FfnActivation,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let act_mul = match activation {
            FfnActivation::SiLU => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            FfnActivation::GeLU => gpu.kernel("gelu", "gelu_mul")?,
        };
        // BF16 path kernels — optional (only loaded if available; gemma4
        // is the only consumer today). `try_kernel` returns
        // `KernelHandle(0)` on miss so we don't break NVFP4-only models
        // that were built without these kernels. Module names per
        // `kernels/gb10/{target}/nvfp4/KERNEL.toml`:
        //   `dense_gemv_bf16 = "gemv"`, `dense_gemm_bf16 = "gemm"`.
        let dense_gemv_bf16_k = crate::layers::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let dense_gemm_bf16_k = crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16");
        let e2m1_checkpoint_scales = load_e2m1_checkpoint_scales(&weights, gpu)?;
        #[cfg(feature = "cuda")]
        let cublaslt_checkpoint_scales =
            try_load_cublaslt_checkpoint_scales(&weights, gpu, e2m1_checkpoint_scales);
        let warp_nvfp4_mma_supported = supports_warp_nvfp4_mma(gpu.compute_capability());
        if gpu_sm_count(gpu) > PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD {
            warm_coalesced_nvfp4_sm120_kernels(gpu);
        }

        Ok(Self {
            weights,
            activation,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            exact_ffn_kernels: ops::W4a16ExactFfnKernels::new(
                crate::layers::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m4"),
                crate::layers::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m8"),
                crate::layers::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m17"),
                crate::layers::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m32"),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_silu_input_exact_m4",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_silu_input_exact_m8",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_silu_input_exact_m17",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_silu_input_exact_m32",
                ),
            )
            .with_materialized_m8(
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_dual_silu_f32_exact_m8",
                ),
                crate::layers::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_f32_input_exact_m8"),
            )
            .with_materialized_m17(
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_dual_silu_f32_exact_m17",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_f32_input_exact_m17",
                ),
            )
            .with_fused_materialized_m17(crate::layers::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_dual_exact_materialize_f32_m17",
            ))
            .with_rt2(
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_dual_exact_materialize_f32_rt2_m17",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_f32_input_exact_rt2_m8",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_f32_input_exact_rt2_m17",
                ),
            ),
            exact_ffn_lowreg_m16: ops::W4a16ExactFfnLowregM16Kernels::new(
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gemv_gate_exact_m16_lowreg",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gemv_up_exact_m16_lowreg",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gate_up_materialize_f32_m16",
                ),
            ),
            w4a16_gemv_sw: crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemv_dual_sw: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_dual_sw",
            ),
            w4a16_gemv_silu_input_sw: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_silu_input_sw",
            ),
            gemv_sw: ops::gemv_sw_enabled(),
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_dual_batch3_tuned: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_dual_batch3_tuned",
            ),
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_pipe: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
            w4a16_gemm_pipe_silu_mul: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_pipe_silu_mul",
            ),
            w4a16_gemm_pipe_dual: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe_dual"),
            act_mul,
            bf16_weights: None,
            dense_gemv_bf16_k,
            dense_gemm_bf16_k,
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
            prefill_t_m128_only: false,
            // Optional small-M (M_TILE=16) transposed-weight GEMM.
            // Missing on older kernel caches — handle 0 disables routing
            // through this kernel regardless of the transposed weights.
            w4a16_gemm_t_m16: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16"),
            // Optional small-M (M_TILE=16), small-N (N_TILE=64) variant for
            // K=3 MTP verify on dense Qwen3.6-27B. Missing on non-qwen3.6-27b
            // kernel caches — handle 0 falls back to GEMV silently.
            w4a16_gemm_t_m16_n64: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16_n64"),
            // Optional large-M (M_TILE=128, N_TILE=128) transposed-weight
            // GEMM for prefill. Handle 0 disables the
            // `ATLAS_PREFILL_FFN_FAST` fast path silently.
            w4a16_gemm_t_m128: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            w4a16_gemm_t_m32_n64: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m32_n64"),
            // Optional fused gate+up+silu kernel. Handle 0 keeps the split
            // gate/up path (ATLAS_FFN_FUSED_GATEUP).
            w4a16_gemm_t_m32_n64_gateup_silu: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu",
            ),
            // Optional dequant-in-registers fork of the fused kernel. Handle
            // 0 keeps the staged fused kernel (ATLAS_DEQUANT_PIPE).
            w4a16_gemm_t_m32_n64_gateup_silu_pipe: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu_pipe",
            ),
            // Optional K_STEP=64 register-dequant fork (ATLAS_GATEUP_K64).
            // Handle 0 falls back through _pipe → staged fused kernel.
            w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64",
            ),
            w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64",
            ),
            // Optional split-K down_proj variant + reduce. Handle 0 keeps
            // the single-slice m32_n64 path (ATLAS_FFN_DOWN_SPLITK).
            w4a16_gemm_t_m32_n64_splitk: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_splitk",
            ),
            reduce_splitk_k: crate::layers::try_kernel(gpu, "w4a16", "reduce_splitk_f32_to_bf16"),
            splitk_workspace: Mutex::new(None),
            // Optional 8-warp shadow of the M=128 kernel. Handle 0
            // disables `ATLAS_FFN_M128_V2` silently.
            w4a16_gemm_t_m128_v2: crate::layers::try_kernel(
                gpu,
                "w4a16_v2",
                "w4a16_gemm_t_m128_v2",
            ),
            // Optional FP8×FP8 M=128 GEMM for the predequant fast path.
            // Handle 0 disables `ATLAS_FFN_PREDEQUANT_FP8` silently.
            fp8_gemm_t_m128_k: crate::layers::try_kernel(gpu, "w4a16", "fp8_gemm_t_m128"),
            gate_fp8: None,
            up_fp8: None,
            down_fp8: None,
            // Optional W4A4 (NVFP4×NVFP4) native tensor-core GEMM. Loaded
            // via `try_kernel` — handle 0 disables the `ATLAS_E2M1_GEMM`
            // fast path silently. On SM103 (`cc == (10, 3)`), warp-level
            // SM120 `kind::mxf4nvf4.block_scale` PTX is a `__trap()` stub
            // (cuBLASLt `VEC16_UE4M3` serves native NVFP4 instead), so these
            // handles stay 0 and can never be dispatched.
            nvfp4_gemm_k: if warp_nvfp4_mma_supported {
                crate::layers::try_kernel(gpu, "nvfp4_cutlass", "nvfp4_nvfp4_gemm_t_m64")
            } else {
                KernelHandle(0)
            },
            nvfp4_gemm_kmajor_m128_k: if warp_nvfp4_mma_supported {
                crate::layers::try_kernel(gpu, "nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m128")
            } else {
                KernelHandle(0)
            },
            nvfp4_gemm_kmajor_m256_k: if warp_nvfp4_mma_supported {
                crate::layers::try_kernel(gpu, "nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m256")
            } else {
                KernelHandle(0)
            },
            nvfp4_absmax_k: crate::layers::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax"),
            nvfp4_quantize_k: crate::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_bf16_to_nvfp4",
            ),
            nvfp4_silu_quantize_k: crate::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_silu_mul_bf16_to_nvfp4",
            ),
            nvfp4_quantize_128x4_k: crate::layers::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "quantize_bf16_to_nvfp4_atlas_128x4",
            ),
            nvfp4_quantize_128x4_ptx_k: crate::layers::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "quantize_bf16_to_nvfp4_atlas_128x4_ptx",
            ),
            nvfp4_silu_quantize_128x4_k: crate::layers::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "silu_mul_quantize_nvfp4_atlas_128x4",
            ),
            nvfp4_quantize_128x4_from_absmax_k: crate::layers::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax",
            ),
            nvfp4_dynamic_combined_alpha_k: crate::layers::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "nvfp4_dynamic_combined_alpha",
            ),
            nvfp4_merged_silu_quantize_128x4_k: crate::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4",
            ),
            nvfp4_merged_silu_quantize_128x4_ptx_k: crate::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx",
            ),
            nvfp4_residual_add_rms_norm_quant_128x4_k: crate::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "residual_add_rms_norm_quantize_nvfp4_128x4",
            ),
            #[cfg(feature = "cuda")]
            cublaslt_nvfp4_scales: Mutex::new(None),
            #[cfg(feature = "cuda")]
            cublaslt_checkpoint_scales,
            e2m1_scratch: Arc::new(Mutex::new(None)),
            e2m1_use_stream: Arc::new(Mutex::new(None)),
            e2m1_scratch_shared: false,
            e2m1_checkpoint_scales,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            flashinfer_ffn_prefill: None,
            // Sparsity-drafted self-speculation kernels (default-off features).
            // The .cu files live in kernels/gb10/common/ and register under
            // their file-stem module names. try_kernel → handle 0 disables the
            // feature silently on caches built before these kernels existed.
            // W3 (3-bit) FFN lane — weights installed later by the loader
            // (set_w3_weights) iff ATLAS_FFN_W3_LAYERS + sidecar match.
            // Kernels live in kernels/gb10/common/w3a16_gemv.cu /
            // w3a16_gemm.cu; try_kernel → handle 0 on caches built before
            // they existed (W3 then stays fully disabled).
            w3_weights: None,
            w3_weights_t: None,
            w3a16_gemv_dual_k: crate::layers::try_kernel(gpu, "w3a16_gemv", "w3a16_gemv_dual"),
            w3a16_gemv_silu_input_k: crate::layers::try_kernel(
                gpu,
                "w3a16_gemv",
                "w3a16_gemv_silu_input",
            ),
            w3a16_gemm_t_m32_n64_k: crate::layers::try_kernel(
                gpu,
                "w3a16_gemm",
                "w3a16_gemm_t_m32_n64",
            ),
            ffn_sparsity_measure_k: crate::layers::try_kernel(
                gpu,
                "ffn_sparsity_measure",
                "ffn_sparsity_measure",
            ),
            sparsity_meas: Mutex::new(None),
            ffn_build_keep_chunks_k: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv_sparse_cols",
                "ffn_build_keep_chunks",
            ),
            w4a16_gemv_sparse_cols_k: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv_sparse_cols",
                "w4a16_gemv_sparse_cols",
            ),
            sparse_draft_scratch: Mutex::new(None),
        })
    }

    /// Install W3 (3-bit) FFN weights for this layer — GEMV-layout copies
    /// (used by the single-token `forward` SiLU path) and transposed
    /// GEMM-layout copies (used by `forward_kgamma`). Called by the loader
    /// when `ATLAS_FFN_W3_LAYERS` names this layer and the sidecar tensors
    /// loaded cleanly (see `weight_map::w3_sidecar`). The original W4
    /// weights are RETAINED for the paths W3 does not cover (prefill,
    /// K=2/3 batched GEMV, GELU models, sparse draft).
    pub fn set_w3_weights(&mut self, gemv: DenseFfnWeights, gemm_t: DenseFfnWeights) {
        self.w3_weights = Some(gemv);
        self.w3_weights_t = Some(gemm_t);
    }

    /// Whether the W3 single-token GEMV path is fully wired: weights
    /// installed + both kernel symbols present + SiLU activation (the W3
    /// GEMV set has no GELU-fused down kernel; GELU models stay on W4).
    pub(super) fn has_w3_gemv(&self) -> bool {
        self.w3_weights.is_some()
            && self.w3a16_gemv_dual_k.0 != 0
            && self.w3a16_gemv_silu_input_k.0 != 0
            && self.activation == FfnActivation::SiLU
    }

    /// Whether the W3 K=γ verify GEMM path is fully wired.
    pub(super) fn has_w3_gemm(&self) -> bool {
        self.w3_weights_t.is_some()
            && self.w3a16_gemm_t_m32_n64_k.0 != 0
            && self.activation == FfnActivation::SiLU
    }

    /// Whether ANY W3 routing is active on this layer (loader log helper).
    pub fn has_w3(&self) -> bool {
        self.has_w3_gemv() || self.has_w3_gemm()
    }

    /// Eagerly allocate the FP32 split-K workspace for the down_proj
    /// (`[k_splits, 32, n]` where n = hidden, M padded to the M_TILE=32
    /// of the split-K kernel). Called at load time (pre-graph-capture)
    /// because `gpu.alloc` is illegal during CUDA graph capture. No-op
    /// when split-K is disabled or the kernel symbols are missing.
    pub fn alloc_splitk_workspace(
        &self,
        gpu: &dyn GpuBackend,
        n: u32,
        shared: Option<&crate::layers::shared_splitk::SharedSplitkWorkspace>,
    ) -> Result<()> {
        // `n` is the largest split-K output dim this layer might use. The
        // down_proj path needs N=hidden; the gate/up path (when
        // `ATLAS_FFN_GATEUP_SPLITK` is on) needs N=intermediate. Callers pass
        // `max(hidden, intermediate)` so ONE FP32 workspace serves both.
        // gate and up run back-to-back on the same stream and each fully
        // reduces into its own output before the next partial phase, so they
        // can safely share this scratch.
        let down_splits = crate::layers::ffn_down_splitk();
        let gateup_splits = crate::layers::ffn_gateup_splitk();
        if (down_splits == 0 && gateup_splits == 0)
            || self.w4a16_gemm_t_m32_n64_splitk.0 == 0
            || self.reduce_splitk_k.0 == 0
        {
            return Ok(());
        }
        let mut slot = self.splitk_workspace.lock().unwrap();
        if slot.is_none() {
            // 8 = max split clamp; 32 = M_TILE of the split-K kernel.
            let bytes = 8usize * 32 * n as usize * 4;
            *slot = Some(match shared {
                Some(shared) => shared.lease(gpu, bytes)?,
                None => gpu.alloc(bytes)?,
            });
        }
        Ok(())
    }

    /// Whether the M_TILE=16 transposed-weight path is wired up.
    /// Used by `forward_kgamma` for dispatch and by the loader log.
    pub fn has_transposed_ffn(&self) -> bool {
        self.gate_proj_t.is_some()
            && self.up_proj_t.is_some()
            && self.down_proj_t.is_some()
            && self.w4a16_gemm_t_m16.0 != 0
    }

    /// Whether the predequanted-FP8 FFN prefill path is wired up.
    /// Returns true when all three projections have FP8 buffers AND the
    /// `fp8_gemm_t_m128` kernel symbol is present. Used by
    /// `forward_prefill` to choose between the NVFP4 + dequant path and
    /// the predequant fast path.
    pub fn has_fp8_ffn(&self) -> bool {
        self.gate_fp8.is_some()
            && self.up_fp8.is_some()
            && self.down_fp8.is_some()
            && self.fp8_gemm_t_m128_k.0 != 0
    }

    /// Pre-dequant NVFP4 FFN weights to FP8 [N, K] for the predequant
    /// fast prefill path. Mirrors
    /// `Qwen3AttentionLayer::predequant_for_prefill` — uses the
    /// NON-transposed weights in `self.weights.gate_proj` etc. (the
    /// `predequant_nvfp4_to_fp8` kernel reads the original [N, K/2]
    /// NVFP4 layout). Allocates 3 × N×K bytes of GPU memory per layer.
    ///
    /// Called by the loader after `DenseFfnLayer::new` when the
    /// `ATLAS_FFN_PREDEQUANT_FP8` env var is set. Silently no-ops
    /// when the kernel symbol is missing.
    ///
    /// `inter` = intermediate_size (gate/up output dim; down input dim).
    /// `hidden` = hidden_size (gate/up input dim; down output dim).
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        hidden: usize,
        inter: usize,
        stream: u64,
    ) -> Result<()> {
        if self.fp8_gemm_t_m128_k.0 == 0 {
            return Ok(()); // FP8 kernel not available — silently skip
        }
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        // gate_proj: [inter, hidden] NVFP4 → [inter, hidden] FP8
        self.gate_fp8 = Some(self.weights.gate_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            inter,
            hidden,
            stream,
        )?);
        // up_proj: same shape as gate
        self.up_fp8 = Some(self.weights.up_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            inter,
            hidden,
            stream,
        )?);
        // down_proj: [hidden, inter]
        self.down_fp8 = Some(self.weights.down_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            hidden,
            inter,
            stream,
        )?);
        Ok(())
    }

    /// Install BF16 dense MLP weights. After this call, the forward paths
    /// dispatch to the BF16 GEMV/GEMM kernels instead of w4a16. The
    /// caller must ensure the BF16 kernels are loaded (see
    /// `dense_gemv_bf16_k` / `dense_gemm_bf16_k` checks). Spec-decode
    /// batched paths (`forward_k2`, `forward_k3`) are NOT supported on
    /// the BF16 path — Gemma-4 dense has no MTP so they're never called.
    pub fn set_bf16_weights(&mut self, gate: DenseWeight, up: DenseWeight, down: DenseWeight) {
        self.bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// Whether this layer can execute batched `forward_kgamma` (quantized W4/W3
    /// weights; BF16 dense MLP weights use per-row GEMV/GEMM).
    pub fn can_forward_kgamma(&self) -> bool {
        self.bf16_weights.is_none()
    }

    /// Whether `forward_kgamma(n)` will take the wide (`32 < n <= 256`)
    /// transposed `w4a16_gemm_t_m128` tensor-core path rather than falling
    /// back to legacy `ops::w4a16_gemm` or breaking default exactness.
    pub fn wide_kgamma_ready(&self, n: u32) -> bool {
        self.bf16_weights.is_none()
            && ffn_tc_active_for_rows(n)
            && n as usize > crate::layers::TC_VERIFY_MAX_ROWS
            && n <= FFN_KGAMMA_WIDE_MAX_ROWS
            && kgamma_wide_active()
            && kgamma_m128_active()
            && self.w4a16_gemm_t_m128.0 != 0
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn()
    }

    /// Whether the FFN activation-sparsity MEASUREMENT harness is wired up:
    /// the env gate is on AND the measure kernel symbol is present.
    pub(super) fn sparsity_measure_active(&self) -> bool {
        crate::layers::measure_ffn_sparsity_enabled() && self.ffn_sparsity_measure_k.0 != 0
    }

    /// Ensure the per-layer sparsity-measurement counter buffers exist and are
    /// zeroed on first allocation. Returns the five device pointers. Allocated
    /// lazily on the first measured `forward` (never during graph capture —
    /// the measured path runs eager).
    pub(super) fn ensure_sparsity_meas(
        &self,
        gpu: &dyn GpuBackend,
        inter: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
        let n_thresh = ops::SPARSITY_NUM_THRESH;
        let mut slot = self.sparsity_meas.lock().unwrap();
        if slot.is_none() {
            let hist_gateup = gpu.alloc(n_thresh * 4)?;
            let count_gateup = gpu.alloc(2 * 4)?;
            let hist_down = gpu.alloc(n_thresh * 4)?;
            let count_down = gpu.alloc(2 * 4)?;
            let meas_silu = gpu.alloc(inter * 2)?; // BF16 [1, intermediate]
            // Zero the accumulators up front (kernel uses atomicAdd).
            gpu.memset(hist_gateup, 0, n_thresh * 4)?;
            gpu.memset(count_gateup, 0, 2 * 4)?;
            gpu.memset(hist_down, 0, n_thresh * 4)?;
            gpu.memset(count_down, 0, 2 * 4)?;
            *slot = Some(SparsityMeas {
                hist_gateup,
                count_gateup,
                hist_down,
                count_down,
                meas_silu,
                steps: 0,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((
            s.hist_gateup,
            s.count_gateup,
            s.hist_down,
            s.count_down,
            s.meas_silu,
        ))
    }

    /// Observer: launch `ffn_sparsity_measure` on `input` at the given site.
    /// PURE READER — never mutates `input` or any token-stream buffer; writes
    /// only into the dedicated `hist`/`count` accumulators. Called from
    /// `forward` at the two FFN sites when the measurement gate is on.
    pub(super) fn measure_sparsity_site(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        hist: DevicePtr,
        count: DevicePtr,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::ffn_sparsity_measure(
            ctx.gpu,
            self.ffn_sparsity_measure_k,
            input,
            hist,
            count,
            k,
            stream,
        )
    }

    /// Periodic D2H dump of the accumulated per-site sparsity histograms,
    /// averaged over all rows measured since process start. Emits a
    /// `tracing::info` line every `measure_ffn_sparsity_dump_every` measured
    /// forwards. Bumps the per-layer step counter each call. No-op when the
    /// dump cadence has not been reached.
    ///
    /// The reported fraction for threshold t at a site is
    /// `hist[t] / elements_seen` — the mean below-threshold activation
    /// fraction, i.e. the UPPER BOUND on the column-skip weight-byte savings
    /// for that projection at that threshold. The go/no-go number is the
    /// down-input (K=intermediate) fraction at the 1% threshold.
    pub(super) fn maybe_dump_sparsity(
        &self,
        ctx: &ForwardContext,
        layer_tag: &str,
        stream: u64,
    ) -> Result<()> {
        let every = crate::layers::measure_ffn_sparsity_dump_every();
        let (hist_gateup, count_gateup, hist_down, count_down, steps) = {
            let mut slot = self.sparsity_meas.lock().unwrap();
            let Some(s) = slot.as_mut() else {
                return Ok(());
            };
            s.steps += 1;
            if !s.steps.is_multiple_of(every) {
                return Ok(());
            }
            (
                s.hist_gateup,
                s.count_gateup,
                s.hist_down,
                s.count_down,
                s.steps,
            )
        };

        // Sync so the accumulators reflect all launched measurements, then
        // read the histograms + counts back to the host.
        ctx.gpu.synchronize(stream)?;
        let n_thresh = ops::SPARSITY_NUM_THRESH;
        let read_hist = |hist: DevicePtr| -> Result<Vec<u32>> {
            let mut bytes = vec![0u8; n_thresh * 4];
            ctx.gpu.copy_d2h(hist, &mut bytes)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let read_count = |count: DevicePtr| -> Result<(u64, u64)> {
            let mut bytes = vec![0u8; 2 * 4];
            ctx.gpu.copy_d2h(count, &mut bytes)?;
            let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
            let elems = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u64;
            Ok((rows, elems))
        };

        let fmt_site = |hist: &[u32], elems: u64| -> String {
            if elems == 0 {
                return "n/a".to_string();
            }
            ops::SPARSITY_TAU
                .iter()
                .zip(hist.iter())
                .map(|(tau, &cnt)| {
                    let frac = cnt as f64 / elems as f64;
                    format!("{:.1}%tau={:.1}%", tau * 100.0, frac * 100.0)
                })
                .collect::<Vec<_>>()
                .join(" ")
        };

        let hg = read_hist(hist_gateup)?;
        let hd = read_hist(hist_down)?;
        let (rows_g, elems_g) = read_count(count_gateup)?;
        let (rows_d, elems_d) = read_count(count_down)?;

        tracing::info!(
            "FFN_SPARSITY[{layer_tag}] steps={steps} \
             gateup_in(K=hidden rows={rows_g}): {} | \
             down_in(K=inter rows={rows_d}): {}",
            fmt_site(&hg, elems_g),
            fmt_site(&hd, elems_d),
        );
        Ok(())
    }

    /// Whether the column-sparse self-spec DRAFT FFN path is wired up: both
    /// kernel symbols present. The env gate (`ATLAS_SELF_SPEC_SPARSE`) is
    /// checked by the caller (`step_self_spec`) so this only reports capability.
    pub fn has_sparse_draft(&self) -> bool {
        self.ffn_build_keep_chunks_k.0 != 0 && self.w4a16_gemv_sparse_cols_k.0 != 0
    }

    /// Ensure the per-layer sparse-draft scratch (`keep_idx` + `keep_len`)
    /// exists with capacity for `k` (the largest K this layer will sparsify —
    /// the down input K=intermediate). Allocated lazily on the first sparse
    /// draft forward (eager path, no graph capture).
    fn ensure_sparse_draft_scratch(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let need_chunks = k / 8;
        let mut slot = self.sparse_draft_scratch.lock().unwrap();
        let realloc = match slot.as_ref() {
            Some(s) => s.cap_chunks < need_chunks,
            None => true,
        };
        if realloc {
            if slot.is_some() {
                gpu.synchronize_device()?;
            }
            if let Some(prev) = slot.take() {
                let _ = gpu.free(prev.keep_idx);
                let _ = gpu.free(prev.keep_len);
            }
            let keep_idx = gpu.alloc(need_chunks * 4)?; // u32 per chunk
            let keep_len = match gpu.alloc(4) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(keep_idx);
                    return Err(err);
                }
            }; // single u32
            *slot = Some(SparseDraftScratch {
                keep_idx,
                keep_len,
                cap_chunks: need_chunks,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((s.keep_idx, s.keep_len))
    }

    /// SPARSE self-spec DRAFT single-token FFN forward.
    ///
    /// Same gate/up GEMV shape as `forward` (gate/up input is the dense
    /// residual stream — low activation sparsity, per the TEAL analysis, so
    /// it stays a dense dual GEMV), then swaps the down_proj GEMV for the
    /// column-sparse path: `ffn_build_keep_chunks` thresholds the silu(gate)*up
    /// activation into a surviving-chunk list, then `w4a16_gemv_sparse_cols`
    /// reads only those weight columns. APPROXIMATE by design — the dense
    /// verify is the lossless oracle, so this only proposes.
    ///
    /// `thresh_frac` is the keep threshold as a fraction of per-row max-abs
    /// (e.g. 0.01 for 1%). Falls back to the exact `forward` dense path when
    /// the sparse kernels are missing (`has_sparse_draft` false), so callers
    /// can always invoke it safely.
    ///
    /// EAGER only (the self-spec draft never captures a CUDA graph): the
    /// `keep_len` scalar is read back D2H before the sparse GEMV launch.
    pub fn forward_draft_sparse(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        thresh_frac: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        // Capability + BF16-weight guard: the sparse kernels operate on the
        // NVFP4 `QuantizedWeight` layout only. Fall back to the exact dense
        // forward when sparse kernels are missing or BF16 weights are active.
        if !self.has_sparse_draft() || self.bf16_weights.is_some() {
            return self.forward(input, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // gate/up stay DENSE (residual-stream input, low sparsity).
        ops::w4a16_decode_gemv_dual(
            ctx.gpu,
            self.w4a16_gemv_dual,
            self.w4a16_gemv_dual_sw,
            self.gemv_sw,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;

        // silu(gate)*up → gate_out (the down-proj input we sparsify).
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            inter,
            stream,
        )?;

        // Threshold the down-input into a surviving k8-chunk list.
        let (keep_idx, keep_len) = self.ensure_sparse_draft_scratch(ctx.gpu, inter as usize)?;
        ops::ffn_build_keep_chunks(
            ctx.gpu,
            self.ffn_build_keep_chunks_k,
            gate_out,
            thresh_frac,
            keep_idx,
            keep_len,
            inter,
            stream,
        )?;

        // Read back keep_len (scalar-by-value kernel arg). Sync is acceptable
        // on the eager draft path; it also bounds the sparse GEMV's loop.
        ctx.gpu.synchronize(stream)?;
        let mut kl_bytes = [0u8; 4];
        ctx.gpu.copy_d2h(keep_len, &mut kl_bytes)?;
        let keep_len_val = u32::from_le_bytes(kl_bytes);

        // Column-sparse down_proj GEMV over the surviving chunks only.
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_sparse_cols(
            ctx.gpu,
            self.w4a16_gemv_sparse_cols_k,
            gate_out,
            &self.weights.down_proj,
            keep_idx,
            keep_len_val,
            output,
            h,
            inter,
            stream,
        )?;
        Ok(output)
    }
}
