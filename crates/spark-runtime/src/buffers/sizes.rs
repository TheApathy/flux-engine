// SPDX-License-Identifier: AGPL-3.0-only

//! Byte sizes for the per-pass GPU buffer arena.

use anyhow::Result;
use atlas_core::{
    config::ModelConfig,
    device::{active_device_is_integrated, active_sm_count, sm121::NUM_SMS},
};

use super::checked_math::{ArenaWidths, align_up, product, sum};

macro_rules! mul {
    ($($value:expr),+ $(,)?) => {
        product(&[$($value),+], stringify!($($value),+))?
    };
}

macro_rules! add {
    ($($value:expr),+ $(,)?) => {
        sum(&[$($value),+], stringify!($($value),+))?
    };
}

pub(super) fn qwen38_flashinfer_merged_output_bytes(
    config: &ModelConfig,
    max_batch_tokens: usize,
    requested: bool,
) -> Result<usize> {
    let exact_qwen38_dense = config.model_type == "qwen3_5"
        && config.num_experts == 0
        && config.hidden_size == 5_120
        && config.intermediate_size == 17_408
        && config.tp_world_size.max(1) == 1;
    if requested && exact_qwen38_dense {
        Ok(mul!(max_batch_tokens.min(8_192), 2, config.intermediate_size, 2).max(256))
    } else {
        Ok(256)
    }
}

/// SM-count threshold above which `ATLAS_FLASH_ATTN_KGAMMA_SPLITK` defaults to
/// enabled when unset (matching `spark_model::layers::flash_attn_kgamma_splitk_enabled`),
/// so `splitk_workspace` always allocates the extended multi-split arena on
/// high-SM discrete GPUs while keeping 48-SM GB10 unchanged.
const AUTO_KGAMMA_SPLITK_SM_THRESHOLD: u32 = 64;

const QWEN4_CROSS_SEQ_DEFAULT_MAX_ROWS: usize = 128;
const QWEN4_CROSS_SEQ_WIDE_MAX_ROWS: usize = 256;
const QWEN4_CROSS_SEQ_WIDE_SM_THRESHOLD: u32 = 64;
const QWEN4_CROSS_SEQ_WIDE_MIN_BATCH: usize = 24;

/// Maximum cross-sequence Qwen4 verify rows provisioned in arena scratch,
/// logits, and SSM arenas.
///
/// Keeps GB10 UMA (`sm_count <= 64`, not discrete) strictly at `128` rows so
/// unified-memory footprints do not grow, while scaling discrete / `> 64`-SM
/// servers with `max_batch_size >= 24` (or an explicit valid
/// `ATLAS_QWEN4_ADAPTIVE_MAX_ROWS` override) up to `256` rows.
fn qwen4_cross_seq_row_cap(
    max_batch_size: usize,
    sm_count: u32,
    is_integrated: Option<bool>,
    env_max_rows: Option<&str>,
) -> usize {
    if let Some(raw) = env_max_rows
        && let Ok(n) = raw.trim().parse::<usize>()
        && (2..=QWEN4_CROSS_SEQ_WIDE_MAX_ROWS).contains(&n)
    {
        return n;
    }
    let wide_gpu = sm_count > QWEN4_CROSS_SEQ_WIDE_SM_THRESHOLD || is_integrated == Some(false);
    if wide_gpu && max_batch_size >= QWEN4_CROSS_SEQ_WIDE_MIN_BATCH {
        QWEN4_CROSS_SEQ_WIDE_MAX_ROWS
    } else {
        QWEN4_CROSS_SEQ_DEFAULT_MAX_ROWS
    }
}

/// Header bytes reserved before the packed verify block table (`pos` + `slot`
/// + `seq_len` arrays, matching `verify_qwen4_batched::frame_metadata_bytes`).
#[inline]
const fn verify_bt_header_bytes(bt_rows: usize) -> usize {
    if bt_rows <= 32 {
        768
    } else if bt_rows <= 128 {
        3072
    } else {
        6144
    }
}

fn resolve_kgamma_splitk_enabled(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        None => sm_count > AUTO_KGAMMA_SPLITK_SM_THRESHOLD,
    }
}

fn splitk_workspace_bytes_for_sm(
    q_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    decode_rows: usize,
    extended: bool,
    sm_count: u32,
) -> Result<usize> {
    let partial_bytes = mul!(add!(head_dim, 2), 4);
    if !extended {
        let slots = sm_count.max(NUM_SMS) as usize;
        return Ok(mul!(slots, partial_bytes));
    }

    const SPLIT_TILE: usize = 512;
    const MAX_SPLITS_CAP: usize = 64;
    let max_splits = max_seq_len.div_ceil(SPLIT_TILE).clamp(2, MAX_SPLITS_CAP);
    Ok(mul!(decode_rows, q_heads, max_splits, partial_bytes))
}

#[cfg(test)]
fn splitk_workspace_bytes(
    q_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    decode_rows: usize,
    extended: bool,
) -> Result<usize> {
    splitk_workspace_bytes_for_sm(
        q_heads,
        head_dim,
        max_seq_len,
        decode_rows,
        extended,
        active_sm_count(),
    )
}

/// Byte sizes of each buffer, derived from ModelConfig.
#[derive(Debug, Clone)]
pub struct BufferSizes {
    pub hidden_states: usize,
    pub residual: usize,
    pub norm_output: usize,
    pub qkv_output: usize,
    pub attn_output: usize,
    pub gate_logits: usize,
    pub moe_output: usize,
    pub logits: usize,
    pub ssm_qkvz: usize,
    pub ssm_ba: usize,
    pub ssm_deinterleaved: usize,
    pub ssm_gates: usize,
    pub ssm_conv_out_f32: usize,
    /// Shared row-major BF16 `[M, 2 * intermediate]` result for the exact
    /// Qwen3.8 merged gate/up FlashInfer prefill route.
    pub ffn_gate_up_bf16: usize,
    pub scratch: usize,
    pub expert_gate_out: usize,
    pub expert_up_out: usize,
    pub expert_down_out: usize,
    /// Compact NVFP4 MoE tile work-list: two u32 words per item.
    pub moe_worklist: usize,
    /// Device-written compact work-list length (one i32).
    pub moe_worklist_total: usize,
    pub splitk_workspace: usize,
    /// Flash-Next prefill gatepre shared-expert gate scalars: one f32 per row,
    /// written by the shared W4A8 quant and read by the routed combine. Zero
    /// (no buffer) for every other model.
    pub moe_gate_scalars: usize,
}

const NVFP4_WORKLIST_N_TILE: usize = 128;
const NVFP4_WORKLIST_GATE_UP_M_TILE: usize = 64;
const NVFP4_WORKLIST_DOWN_M_TILE: usize = 64;

fn nvfp4_worklist_capacity_items(
    total_expanded: usize,
    num_experts: usize,
    n_tiles: usize,
    m_tile: usize,
) -> Result<usize> {
    // One expert may receive every row; spreading rows across experts adds at
    // most one partial M tile per expert. The extra item mirrors the proven
    // FP8 work-list bound and keeps the allocation conservative at boundaries.
    anyhow::ensure!(m_tile > 0, "buffer arena worklist tile must be positive");
    Ok(mul!(
        add!(total_expanded.div_ceil(m_tile), num_experts, 1),
        n_tiles
    ))
}

impl BufferSizes {
    /// Compute all buffer sizes from model config and max batch tokens.
    ///
    /// All sizes in bytes. BF16 = 2 bytes per element.
    /// Logits buffer is capped: only needed for decode (1 token) or
    /// speculative verification (K tokens), never for full prefill.
    ///
    /// `max_seq_len` and `kv_block_size` are needed to size the scratch
    /// buffer for block table metadata during batched decode / verify.
    pub fn from_config(
        config: &ModelConfig,
        max_batch_tokens: usize,
        max_seq_len: usize,
        kv_block_size: usize,
        max_batch_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            kv_block_size > 0,
            "buffer arena KV block size must be positive"
        );
        let widths = ArenaWidths::checked(config)?;
        let decode_meta = super::DecodeMetaLayout::for_max_batch_size(max_batch_size);
        let bf16 = 2;
        let m = max_batch_tokens;
        // Expert-major Qwen4 prefill (ATLAS_QWEN4_PREFILL_MOE_EXPERT_MAJOR=1) runs mixers on
        // ATLAS_QWEN4_PREFILL_MIXER_ROWS-row slices of an m-row super-chunk: only the row
        // state and MoE buffers need m rows; mixer scratch is capped at the slice width.
        let mm = expert_major_mixer_rows().map_or(m, |rows| rows.min(m));
        // Rows the MoE pass still needs in capped mixer scratch (0 unless capped).
        let em = if mm < m { m } else { 0 };
        let h = config.hidden_size;
        // Qwen4-Exp carries hc_count parallel residual streams between
        // blocks. Core attention/GDN/MoE tensors remain hidden_size wide;
        // only the persistent hidden/residual pair uses the expanded width.
        let residual_width = widths.residual;

        // Q projection output: gated models produce [Q, gate] (2× nq*hd),
        // ungated models (VL) produce only [Q] (nq*hd).
        let q_heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let hd = config.head_dim;
        let q_proj_mul = if config.attn_gated { 2 } else { 1 };
        let qkv_dim = mul!(add!(mul!(q_heads, q_proj_mul), mul!(2, kv_heads)), hd);

        let top_k = config.num_experts_per_tok;

        // Scratch layout (two users, take max):
        //
        // A) Prefill chunk metadata (after MoE routing data):
        //   [0 .. moe_scratch): MoE topK routing indices+weights
        //   [moe_scratch .. ): positions(m*4) + slots(m*8) + block_table(max_blocks*4) + seq_len(4)
        //
        // B) Batched decode/verify metadata:
        //   [0 .. 32768): fixed metadata region
        //   [32768 .. 32768+header): decode/verify metadata (positions, slots, seq_lens;
        //     `header` scales with row count: 768 / 3072 / 6144 bytes via `verify_bt_header_bytes`)
        //   [32768+header .. ): block table (padded_n × max_blocks × 4 bytes)
        //
        // MoE scratch: 2 * M * top_k * 4 (indices [M*top_k] u32 + weights [M*top_k] f32)
        let moe_scratch = mul!(2, m, top_k, 4);
        let max_blocks = add!(max_seq_len / kv_block_size, 1);
        // Prefill metadata: mirrors exact layout in prefill_chunk(). MRoPE
        // (Qwen3-VL / Qwen3.6) uploads THREE u32 position streams packed
        // back-to-back (T, H, W); every other model uploads ONE. Sizing the
        // scratch region for 1× with MRoPE active caused `cuMemcpyHtoDAsync_v2
        // status 1` failures on long-context prefills (observed: 16k Qwen3.6
        // failed, 8k passed because the extra 64 KB of write overflow happened
        // to still land inside the over-provisioned `moe_scratch + meta`
        // aggregate).
        let pos_streams = if config.mrope_interleaved { 3 } else { 1 };
        let pos_bytes = mul!(m, 4, pos_streams);
        let slot_offset = align_up(pos_bytes, 8, "prefill slots alignment")?;
        let slot_end = add!(slot_offset, mul!(m, 8));
        let bt_offset = align_up(slot_end, 4, "prefill block table alignment")?;
        let bt_end = add!(bt_offset, mul!(max_blocks, 4));
        let sl_offset = align_up(bt_end, 4, "prefill sequence lengths alignment")?;
        // Qwen4's correctness-first serialized prefill consumes the decode
        // attention path one row at a time, so it needs one cumulative
        // sequence length per prompt row. Other prefill implementations only
        // need the final scalar length.
        let seq_lens_bytes = if config.is_qwen4_exp() { mul!(m, 4) } else { 4 };
        let prefill_meta = add!(sl_offset, seq_lens_bytes);
        // Block table metadata: max(batch_size=8, K=4 verify, K=γ DFlash verify)
        // rows × max_blocks × 4 bytes. DFlash γ-block verify uses up to γ+1=17
        // rows (γ=16 for Qwen3.6-DFlash), so size for the worst case.
        let sm_count = active_sm_count();
        let is_integrated = active_device_is_integrated();
        let qwen4_cross_seq_rows = if config.is_qwen4_exp()
            && max_batch_size > 1
            && std::env::var("ATLAS_QWEN4_BATCHED_K5").ok().as_deref() == Some("1")
        {
            let row_cap = qwen4_cross_seq_row_cap(
                max_batch_size,
                sm_count,
                is_integrated,
                std::env::var("ATLAS_QWEN4_ADAPTIVE_MAX_ROWS")
                    .ok()
                    .as_deref(),
            );
            mul!(max_batch_size, 9).min(row_cap)
        } else {
            0
        };
        let bt_rows = 32usize.max(qwen4_cross_seq_rows); // headroom for K=γ DFlash or 128/256-row Qwen4 cross-seq verify
        let verify_bt_header = verify_bt_header_bytes(bt_rows);
        let verify_bt_meta = add!(verify_bt_header, mul!(bt_rows, max_blocks, 4));
        let bt_meta = add!(
            32768,
            verify_bt_meta.max(decode_meta.checked_meta_bytes(max_blocks)?)
        );
        let scratch_min = 64 * 1024;
        let scratch = scratch_min
            .max(add!(moe_scratch, prefill_meta))
            .max(bt_meta);

        // Batched expert output buffers for MoE (or dense FFN).
        // Sized for max(K=3 verify, prefill chunk) × top_k experts.
        let k_max = m.max(3); // prefill chunk or K=3 verify, whichever larger
        let expert_inter = if config.num_experts > 0 {
            mul!(
                k_max,
                config.num_experts_per_tok,
                config.moe_intermediate_size
            )
        } else {
            mul!(k_max, config.intermediate_size)
        };
        let qwen4_moe_w4a8 = config.is_qwen4_exp()
            && std::env::var("ATLAS_QWEN4_PREFILL_MOE_W4A8")
                .ok()
                .as_deref()
                == Some("1")
            && std::env::var("ATLAS_QWEN4_PREFILL_MOE_COMPACT_CHECK")
                .ok()
                .as_deref()
                != Some("1");
        let qwen4_moe_slot_mxfp8 = qwen4_moe_w4a8
            && std::env::var("ATLAS_QWEN4_PREFILL_MOE_SLOT_MXFP8")
                .ok()
                .as_deref()
                == Some("1")
            && std::env::var_os("ATLAS_QWEN4_ORACLE_DUMP").is_none();
        let expert_gate_out = mul!(expert_inter, bf16);
        let expert_up_out = if qwen4_moe_w4a8 && config.num_experts > 0 {
            // W4A8 route (rows > 128) only uses expert_up_out for FP8 act + scales (k_max * 2640 B).
            mul!(k_max, 2640).max(mul!(
                128,
                config.num_experts_per_tok,
                config.moe_intermediate_size,
                bf16
            ))
        } else {
            mul!(expert_inter, bf16)
        };
        // Routed expert down output: [k_max * top_k, moe_input_size].
        // For LatentMoE (Super 120B), routed experts output in latent space.
        let moe_out_dim = config.moe_input_size();
        let expert_down_out = if qwen4_moe_slot_mxfp8 && config.num_experts > 0 {
            // MXFP8 slot route (rows > 128) stores 1-byte e4m3 slots + e8m0 scales (expanded * 2640 B).
            mul!(k_max, config.num_experts_per_tok, 2640).max(mul!(
                128,
                config.num_experts_per_tok,
                moe_out_dim,
                bf16
            ))
        } else if config.num_experts > 0 {
            mul!(k_max, config.num_experts_per_tok, moe_out_dim, bf16)
        } else {
            mul!(k_max, h, bf16)
        };

        // Persistent compact work-list used by the default-off ordinary-NVFP4
        // prefill route. Gate+up and down use the parent's compilable M64 K64
        // kernels over fused 2*intermediate N and hidden-size N respectively.
        // A single arena allocation is reused sequentially for both builders
        // on the compute stream.
        let moe_worklist = if config.num_experts > 0 {
            let total_expanded = mul!(m, top_k);
            let gate_up_n_tiles =
                mul!(2, config.moe_intermediate_size).div_ceil(NVFP4_WORKLIST_N_TILE);
            let down_n_tiles = h.div_ceil(NVFP4_WORKLIST_N_TILE);
            let gate_up_items = nvfp4_worklist_capacity_items(
                total_expanded,
                config.num_experts,
                gate_up_n_tiles,
                NVFP4_WORKLIST_GATE_UP_M_TILE,
            )?;
            let down_items = nvfp4_worklist_capacity_items(
                total_expanded,
                config.num_experts,
                down_n_tiles,
                NVFP4_WORKLIST_DOWN_M_TILE,
            )?;
            mul!(gate_up_items.max(down_items), 2, std::mem::size_of::<u32>())
        } else {
            256
        };
        let moe_worklist_total = std::mem::size_of::<i32>();
        let moe_gate_scalars = if config.is_qwen4_exp() {
            mul!(m, std::mem::size_of::<f32>())
        } else {
            0
        };

        // The mixed path may park a prefill row immediately after the widest
        // decode batch, hence rows+1.  The 32-row floor preserves existing
        // allocations; C33..C128, 128/256-row cross-seq verify, and opt-in
        // batched flat-DFlash verify (`ATLAS_DFLASH_BATCHED_VERIFY=1`, up to
        // C=16 x gamma=7 = 112 proposal logits rows) grow only this bounded output arena.
        let dflash_batched_verify_rows =
            if std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref() == Some("1") {
                128usize
            } else {
                0usize
            };
        let logits_tokens = m.min(
            32.max(qwen4_cross_seq_rows)
                .max(dflash_batched_verify_rows)
                .max(add!(decode_meta.rows(), 1)),
        );

        // Mamba-2 d_inner may exceed hidden_size; norm_output and attn_output must fit.
        let mamba2_d_inner = widths.mamba_inner;
        let max_dim = h.max(mamba2_d_inner);
        // Gated-delta layers can be wider than full attention. Qwen4-Exp, for
        // example, has 48 value heads x 128 = 6144 elements while its full
        // attention output is only 16 x 256 = 4096. `attn_output` is the BF16
        // GDN output scratch in both decode and prefill, so size it for the
        // widest producer rather than assuming attention is the maximum.
        let linear_value_dim = widths.linear_value;
        let hyper_scratch = if config.hc_count > 1 {
            mul!(mm, add!(config.hc_lowrank, config.hc_count), bf16)
        } else {
            0
        };

        // Split-K decode workspace: stores partials [o[head_dim], m, l] per
        // (seq, q_head, split). Indexed as
        //   workspace[seq_idx, q_head, split] (head_dim + 2) F32 values.
        //
        // Base mode (both split-K selectors off): `sm_count.max(NUM_SMS)` slots
        // (48 on GB10; scales with active SM count on wider GPUs), sufficient when
        // num_splits is derived solely from SM occupancy.
        //
        // Extended mode (`ATLAS_PAGED_DECODE_SPLITK=1` or kgamma split-K, which
        // auto-enables on `sm_count > AUTO_KGAMMA_SPLITK_SM_THRESHOLD` unless
        // explicitly set to 0): sizes the arena so num_splits can scale with
        // seq_len:
        //   max_decode_seqs = decode_meta.rows() (the admitted serving width)
        //   MAX_SPLITS      = max_seq_len.div_ceil(512).clamp(2, 64)
        // At C128 the arena is intentionally four times the legacy C32
        // envelope. A fixed 32-row multiplier would let rows 32..127 write
        // beyond the allocation whenever the wider serving envelope is enabled.
        let splitk_enabled = std::env::var("ATLAS_PAGED_DECODE_SPLITK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        // kgamma split-K reuses the same workspace layout
        // ([num_qtile, num_q_heads, num_splits, (hd+2)] F32) so the
        // extended-arena formula covers both paged-decode and kgamma split-K
        // even when `ATLAS_PAGED_DECODE_SPLITK` is off.
        let kgamma_splitk_enabled = resolve_kgamma_splitk_enabled(
            std::env::var("ATLAS_FLASH_ATTN_KGAMMA_SPLITK")
                .ok()
                .as_deref(),
            sm_count,
        );
        let splitk_workspace = splitk_workspace_bytes_for_sm(
            q_heads,
            hd,
            max_seq_len,
            decode_meta.rows(),
            splitk_enabled || kgamma_splitk_enabled,
            sm_count,
        )?;

        // Residual dtype controlled by config.use_fp32_residual().
        // FP32 prevents BF16 truncation across 48 layers but costs 2x bandwidth.
        let residual_elem = if config.use_fp32_residual() { 4 } else { bf16 };

        // FlashInfer emits merged gate/up as one row-major [M, 34816]
        // result, while Atlas keeps two disjoint [M, 17408] destinations.
        // No existing shared arena is large enough: the closest Qwen3.8 SSM
        // arena is M*16384*4 = M*65536 bytes, 6.25% below M*69632. Allocate
        // one model-wide arena (never one per layer), and cap it at the
        // largest admitted row count because larger/tail shapes cannot route.
        // Only the exact opt-in string allocates it; malformed values fail in
        // the authoritative model selector and receive only the sentinel.
        let flashinfer_ffn_requested =
            std::env::var_os("ATLAS_PREFILL_FFN_FLASHINFER").is_some_and(|value| value == "1");
        let ffn_gate_up_bf16 =
            qwen38_flashinfer_merged_output_bytes(config, m, flashinfer_ffn_requested)?;

        let sizes = Self {
            hidden_states: mul!(m, residual_width, residual_elem),
            residual: mul!(m, residual_width, residual_elem),
            norm_output: mul!(m, max_dim, bf16),
            // Expert-major: the MoE pass stages one residual row + m packed hidden rows here
            // (qwen4_prefill_moe Plan).
            qkv_output: mul!(mm, qkv_dim, bf16).max(if em > 0 {
                mul!(add!(mul!(em, h), residual_width), bf16)
            } else {
                0
            }),
            attn_output: mul!(mm, config.num_attention_heads, config.head_dim, bf16)
                .max(mul!(mm, mamba2_d_inner, bf16))
                .max(mul!(mm, linear_value_dim, bf16))
                // Shared expert & compact MoE write [M, hidden_size] BF16 here even when
                // TP=4 shrinks local attention/SSM head width below hidden_size (1536 < 2560).
                .max(mul!(m, h, bf16))
                // MLA absorbed: attention output is [M, nq, mla_cache_dim=kv_lora+rope]
                .max(if config.kv_lora_rank > 0 {
                    mul!(
                        mm,
                        config.num_attention_heads,
                        add!(config.kv_lora_rank, config.qk_rope_head_dim),
                        bf16
                    )
                } else {
                    0
                }),
            gate_logits: if config.num_experts > 0 {
                mul!(m, config.num_experts, bf16)
            } else {
                256
            },
            moe_output: mul!(m, h, bf16),
            logits: mul!(logits_tokens, config.vocab_size, bf16), // BF16 from LM head kernel
            // SSM buffers are also reused by attention prefill/multi-seq as scratch:
            //   ssm_qkvz: K+V contiguous storage in prefill [M, 2*kv_dim]
            //             Mamba-2 in_proj output [M, in_proj_size]
            //             multi-seq attention Q staging [M, q_heads*q_proj_mul*hd]
            //   ssm_deinterleaved: Q contiguous copy [M, nq*hd]
            //                      Mamba-2 conv1d output [M, d_xBC]
            // Use max across all uses with minimum 256 to avoid 0-byte alloc.
            //
            // The Q-staging term is load-bearing, not defensive padding. The
            // multi-seq attention QKV routes (`ms_qkv_exact`, `ms_qkv_batch3`,
            // `ms_qkv_batched_plain`) all stage Q in `ssm_qkvz()` at the FULL
            // projection width, which is 2*nq*hd on a gated model. On
            // Qwen3.8-27B that is 12288 — numerically equal to this arch's
            // `ssm_qkvz_size()`, so the buffer fit by coincidence, not by
            // construction. Any model whose gated q_proj_dim exceeds its SSM
            // QKVZ width would have overrun the allocation and silently
            // corrupted whatever followed it in the arena.
            ssm_qkvz: (if config.is_qwen4_exp()
                && std::env::var("ATLAS_QWEN4_PREFILL_SSM_EXACT")
                    .ok()
                    .as_deref()
                    == Some("1")
            {
                mul!(
                    64.max(qwen4_cross_seq_rows).max(decode_meta.rows()),
                    widths.ssm_qkvz,
                    bf16
                )
            } else {
                mul!(mm, widths.ssm_qkvz, bf16)
            })
            .max(mul!(mm, widths.mamba_in_proj, bf16))
            .max(mul!(mm, 2, kv_heads, hd, bf16))
            .max(mul!(mm, q_heads, q_proj_mul, hd, bf16)) // gated attention Q staging
            .max(mul!(mm, config.shared_expert_intermediate_size, bf16)) // MoE shared up scratch
            .max(mul!(em, config.shared_expert_intermediate_size, bf16))
            .max(256),
            ssm_ba: mul!(mm, widths.ssm_ba, bf16)
                .max(mul!(mm, config.moe_latent_size, bf16)) // LatentMoE latent buffer
                // Native Qwen4 hyperconnection prepare stores low-rank logits
                // plus one injection scale per stream in this arena.
                .max(hyper_scratch)
                // MLA reuses ssm_ba for two separate buffers:
                //   - q_latent    [M, q_lora_rank]    BF16 — output of wq_a GEMM
                //   - k_rope_buf  [M, qk_rope_head_dim] BF16 — output of wkv_a_rope GEMM
                // Both are written sequentially (q_latent is consumed before
                // k_rope_buf is allocated). Size for the larger of the two.
                .max(if config.kv_lora_rank > 0 {
                    mul!(mm, config.qk_rope_head_dim, bf16).max(mul!(mm, config.q_lora_rank, bf16))
                } else {
                    0
                })
                .max(256),
            ssm_deinterleaved: mul!(mm, widths.ssm_qkvz, bf16)
                .max(mul!(em, config.shared_expert_intermediate_size, bf16)) // MoE shared gate/act
                .max(mul!(mm, widths.mamba_xbc, bf16))
                .max(mul!(mm, q_heads, hd, bf16))
                // MLA absorbed: Q_absorbed buffer is [M, nq, mla_cache_dim=kv_lora+rope]
                .max(if config.kv_lora_rank > 0 {
                    mul!(
                        mm,
                        q_heads,
                        add!(config.kv_lora_rank, config.qk_rope_head_dim),
                        bf16
                    )
                } else {
                    0
                })
                .max(256),
            ssm_gates: mul!(mm, config.linear_num_value_heads, 2, 4).max(256),
            // FP32 conv output for SSM recurrent path precision (4 bytes/element),
            // or 16-bit Q/K/V + GDN output (2 bytes/element) when Qwen4 GDN_FI is active.
            // Also reused by MLA as q_rope contiguous buffer: [M, nq * qk_rope_head_dim] BF16.
            ssm_conv_out_f32: (if config.is_qwen4_exp()
                && matches!(
                    std::env::var("ATLAS_QWEN4_PREFILL_GDN_FI").ok().as_deref(),
                    Some("1" | "2")
                )
                && std::env::var("ATLAS_QWEN4_PREFILL_SSM_CHECK")
                    .ok()
                    .as_deref()
                    != Some("1")
            {
                let conv_dim = if std::env::var("ATLAS_QWEN4_PREFILL_GDN_NORM_FP8")
                    .ok()
                    .as_deref()
                    == Some("3")
                {
                    config.ssm_qkv_size()
                } else {
                    widths.ssm_qkvz
                };
                mul!(mm, conv_dim, 2).max(mul!(
                    64.max(qwen4_cross_seq_rows).max(decode_meta.rows()),
                    widths.ssm_qkvz,
                    4
                ))
            } else {
                mul!(mm, widths.ssm_qkvz, 4)
            })
            .max(if config.kv_lora_rank > 0 {
                mul!(mm, q_heads, config.qk_rope_head_dim, bf16)
            } else {
                0
            })
            .max(256),
            ffn_gate_up_bf16,
            scratch,
            expert_gate_out,
            expert_up_out,
            expert_down_out,
            moe_worklist,
            moe_worklist_total,
            splitk_workspace,
            moe_gate_scalars,
        };
        sizes.total_bytes()?;
        Ok(sizes)
    }

    /// Total bytes across all buffers.
    pub fn total_bytes(&self) -> Result<usize> {
        sum(
            &[
                self.hidden_states,
                self.residual,
                self.norm_output,
                self.qkv_output,
                self.attn_output,
                self.gate_logits,
                self.moe_output,
                self.logits,
                self.ssm_qkvz,
                self.ssm_ba,
                self.ssm_deinterleaved,
                self.ssm_gates,
                self.ssm_conv_out_f32,
                self.ffn_gate_up_bf16,
                self.scratch,
                self.expert_gate_out,
                self.expert_up_out,
                self.expert_down_out,
                self.moe_worklist,
                self.moe_worklist_total,
                self.splitk_workspace,
                self.moe_gate_scalars,
            ],
            "total bytes",
        )
    }
}

/// `Some(rows)` when expert-major Qwen4 prefill caps the mixer scratch at a slice width.
fn expert_major_mixer_rows() -> Option<usize> {
    if std::env::var("ATLAS_QWEN4_PREFILL_MOE_EXPERT_MAJOR")
        .ok()
        .as_deref()
        != Some("1")
    {
        return None;
    }
    Some(
        std::env::var("ATLAS_QWEN4_PREFILL_MIXER_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&r| r >= 2048 && r.is_multiple_of(2048))
            .unwrap_or(16384),
    )
}

#[cfg(test)]
mod worklist_tests {
    use atlas_core::config::ModelConfig;

    use super::{
        BufferSizes, NUM_SMS, nvfp4_worklist_capacity_items, qwen4_cross_seq_row_cap,
        resolve_kgamma_splitk_enabled, splitk_workspace_bytes, splitk_workspace_bytes_for_sm,
        verify_bt_header_bytes,
    };

    #[test]
    fn qwen4_c2_c32_hyper_scratch_is_part_of_the_ssm_ba_arena() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.hidden_size = 2_560;
        config.hc_count = 4;
        config.hc_lowrank = 320;
        for rows in 2..=128 {
            let sizes = BufferSizes::from_config(&config, rows, 2_048, 16, rows).unwrap();
            let required = rows * (config.hc_lowrank + config.hc_count) * 2;
            assert!(
                sizes.ssm_ba >= required,
                "C{rows}: {} < {required}",
                sizes.ssm_ba
            );
        }
    }

    #[test]
    fn qwen4_c128_exact_moe_arenas_cover_single_sweep_bf16_extents() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.hidden_size = 2_560;
        config.num_experts = 512;
        config.num_experts_per_tok = 10;
        config.moe_intermediate_size = 640;
        config.hc_count = 4;
        config.hc_lowrank = 320;
        let sizes = BufferSizes::from_config(&config, 128, 2_048, 16, 128).unwrap();
        let routed_inter = 128 * config.num_experts_per_tok * config.moe_intermediate_size * 2;
        let routed_down = 128 * config.num_experts_per_tok * config.hidden_size * 2;
        assert!(sizes.expert_gate_out >= routed_inter);
        assert!(sizes.expert_up_out >= routed_inter);
        assert!(sizes.expert_down_out >= routed_down);
    }

    #[test]
    fn flash_next_16k_capacity_matches_gate_up_and_down_geometry() {
        let total_expanded = 16_000 * 10;
        let experts = 512;
        let gate_up_items = nvfp4_worklist_capacity_items(total_expanded, experts, 10, 64).unwrap();
        let down_items = nvfp4_worklist_capacity_items(total_expanded, experts, 20, 64).unwrap();

        assert_eq!(gate_up_items * 8, 241_040);
        assert_eq!(down_items * 8, 482_080);
        assert!(down_items > gate_up_items);
    }

    #[test]
    fn aggressive_splitk_workspace_scales_to_c128() {
        let c32 = splitk_workspace_bytes(24, 256, 16_384, 32, true).unwrap();
        let c128 = splitk_workspace_bytes(24, 256, 16_384, 128, true).unwrap();
        assert_eq!(c32, 32 * 24 * 32 * 258 * 4);
        assert_eq!(c128, 128 * 24 * 32 * 258 * 4);
        assert_eq!(c128, c32 * 4);
    }

    #[test]
    fn default_splitk_workspace_remains_one_sm_envelope() {
        let c1 = splitk_workspace_bytes(24, 256, 16_384, 32, false).unwrap();
        let c128 = splitk_workspace_bytes(24, 256, 16_384, 128, false).unwrap();
        assert_eq!(c1, NUM_SMS as usize * 258 * 4);
        assert_eq!(c128, c1);
    }

    #[test]
    fn splitk_workspace_scales_slots_with_active_sm_count_and_floors_at_sm121() {
        let partial_bytes = (256 + 2) * 4;
        assert_eq!(
            splitk_workspace_bytes_for_sm(24, 256, 16_384, 32, false, 0).unwrap(),
            48 * partial_bytes
        );
        assert_eq!(
            splitk_workspace_bytes_for_sm(24, 256, 16_384, 32, false, 48).unwrap(),
            48 * partial_bytes
        );
        for sm in [110u32, 170, 188] {
            assert_eq!(
                splitk_workspace_bytes_for_sm(24, 256, 16_384, 32, false, sm).unwrap(),
                sm as usize * partial_bytes,
                "sm_count={sm}"
            );
        }

        // Auto kgamma split-K mirrors layers::flash_attn_kgamma_splitk_enabled.
        assert!(!resolve_kgamma_splitk_enabled(None, 48));
        assert!(!resolve_kgamma_splitk_enabled(None, 64));
        assert!(resolve_kgamma_splitk_enabled(None, 110));
        assert!(resolve_kgamma_splitk_enabled(None, 170));
        assert!(resolve_kgamma_splitk_enabled(None, 188));
        assert!(!resolve_kgamma_splitk_enabled(Some("0"), 188));
        assert!(resolve_kgamma_splitk_enabled(Some("1"), 48));
        assert!(resolve_kgamma_splitk_enabled(Some("true"), 48));
    }

    #[test]
    fn qwen4_cross_seq_row_cap_preserves_gb10_uma_and_scales_sm120_to_256() {
        // GB10 UMA (48 SMs, integrated or unset) stays strictly at 128 across all batch sizes.
        for batch in [2usize, 16, 24, 32, 48, 64, 128] {
            assert_eq!(qwen4_cross_seq_row_cap(batch, 48, Some(true), None), 128);
            assert_eq!(qwen4_cross_seq_row_cap(batch, 64, Some(true), None), 128);
            assert_eq!(qwen4_cross_seq_row_cap(batch, 48, None, None), 128);
        }
        // Discrete / >64-SM GPU keeps 128 for batch < 24 and lifts to 256 for batch >= 24.
        for batch in [2usize, 8, 16, 23] {
            assert_eq!(qwen4_cross_seq_row_cap(batch, 188, Some(false), None), 128);
        }
        for batch in [24usize, 32, 48, 64, 128] {
            assert_eq!(qwen4_cross_seq_row_cap(batch, 188, Some(false), None), 256);
            assert_eq!(qwen4_cross_seq_row_cap(batch, 170, None, None), 256);
            assert_eq!(qwen4_cross_seq_row_cap(batch, 48, Some(false), None), 256);
        }
        // Explicit env override takes precedence within 2..=256.
        assert_eq!(
            qwen4_cross_seq_row_cap(32, 48, Some(true), Some("256")),
            256
        );
        assert_eq!(
            qwen4_cross_seq_row_cap(64, 188, Some(false), Some("128")),
            128
        );
        assert_eq!(
            qwen4_cross_seq_row_cap(64, 188, Some(false), Some("bad")),
            256
        );

        // Verify header tiers match verify_qwen4_batched::frame_metadata_bytes.
        assert_eq!(verify_bt_header_bytes(1), 768);
        assert_eq!(verify_bt_header_bytes(32), 768);
        assert_eq!(verify_bt_header_bytes(33), 3072);
        assert_eq!(verify_bt_header_bytes(128), 3072);
        assert_eq!(verify_bt_header_bytes(129), 6144);
        assert_eq!(verify_bt_header_bytes(256), 6144);
    }

    #[test]
    fn splitk_and_worklist_helpers_fail_closed_on_overflow() {
        assert!(splitk_workspace_bytes(1, usize::MAX, 1, 1, false).is_err());
        assert!(splitk_workspace_bytes(usize::MAX, 1, 1024, 128, true).is_err());
        assert!(nvfp4_worklist_capacity_items(1, 1, 1, 0).is_err());
        assert!(nvfp4_worklist_capacity_items(usize::MAX, 0, 1, 1).is_err());
        assert!(nvfp4_worklist_capacity_items(0, usize::MAX, 1, 64).is_err());
        assert!(nvfp4_worklist_capacity_items(0, 1, usize::MAX, 64).is_err());
        assert_eq!(
            nvfp4_worklist_capacity_items(usize::MAX - 1, 0, 1, 1).unwrap(),
            usize::MAX
        );
    }
}
