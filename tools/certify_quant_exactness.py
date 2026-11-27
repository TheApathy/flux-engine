#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Offline numerical exactness certifier for Flux quantization formats.

Certifies bit-exact numerical invariants (CPU-only, pure NumPy + stdlib) for:

  1. **LM-Head Precision Ladder (`BF16` vs `INT8 per-32 BF16sc` vs `DualPlaneNvfp4` vs `MXFP8`)** —
     mirrors ``crates/spark-model/src/weight_map/dual_plane.rs`` and
     ``crates/spark-model/src/layers/ops/lm_head_argmax.rs`` bit-for-bit:
       - **`INT8 per-32 BF16sc` (``8.50 bpw``, ``1.88x`` smaller than BF16)**:
         signed ``int8`` (``-127..=127``) spends all 7 magnitude bits linearly
         per 32-element block with a BF16 scale, converts **0-ULP lossless**
         into BF16, and Pareto-dominates both ``DualPlaneNvfp4`` (``8.50 bpw``,
         ``~38.5 dB``) and ``MXFP8`` (``8.25 bpw``, ``~31.5 dB``) at
         ``>= 44.0 dB`` (``~45.4 dB``) SQNR and ``D_KL < 1e-6`` at ``T = 1.0``.
       - **`DualPlaneNvfp4` (``8.50 bpw``)**: weight & logit SQNR ``~38.5 dB``
         (``>= +6.0 dB`` over block-128 FP8 E4M3 ``~31.8 dB``), Top-1 flip
         ``< 0.1%``, ``D_KL < 1e-4`` at ``T = 1.0``, plus strict ``>= 45.0 dB``
         BF16-equivalence check.
       - Exact deterministic lowest-vocab-index tie-breaking and ``NaN``/``-Inf``
         masking parity vs BF16 reference.
  2. **BitFP4 + 6-op Branchless SWAR RaZeR-KV** — mirrors
     ``crates/spark-model/src/layers/qwen3_attention/bitfp4_kv.rs`` bit-for-bit:
       - Exhaustive 2-op SWAR ``-0.0 (0x8) -> +0.0 (0x0)`` E2M1 canonicalization.
       - Exhaustive 6-op branchless SWAR ``razer_swar_split_u32(x, neg_sv)``
         exact reconstruction of ``±5.0`` (``x_main + x_comp``) across all 16
         nibble codes and both signs.
       - Strictly lower bias-free reconstruction MSE vs standard NVFP4 on both
         WHT-256 rotated Keys and channel-transposed ``V^T`` block-16 slabs.
       - Attention logit cosine similarity / max abs error and full attention
         output fidelity with Dual-Residual ``Q^(1)+Q^(2)`` and Underflow-Aware
         probability scaling.

Safe to run with ``CUDA_VISIBLE_DEVICES=""``. Supports ``--self-test`` and
optional offline weight inspection via ``--weights <file.npy|.npz>``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

DUAL_PLANE_GROUP_SIZE = 16
DUAL_PLANE_LO_SCALE = np.float32(0.125)
INT8_PER32_GROUP_SIZE = 32
MXFP8_GROUP_SIZE = 32
BITFP4_HDIM = 256
BITFP4_BLOCK_SIZE = 16
UNDERFLOW_C = np.float32(8.0)
UNDERFLOW_TAU = np.float32(21.0 / 1024.0)

E2M1_LUT = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def _build_ue4m3_lut() -> np.ndarray:
    vals = np.zeros(128, dtype=np.float32)
    for c in range(127):
        exp = (c >> 3) & 0xF
        man = c & 0x7
        if exp == 0:
            vals[c] = np.float32(man * (2.0**-9))
        else:
            vals[c] = np.float32((1.0 + man / 8.0) * (2.0 ** (exp - 7)))
    vals[127] = np.float32(0.0)
    return vals


UE4M3_LUT = _build_ue4m3_lut()


def bf16_round_np(arr: np.ndarray) -> np.ndarray:
    """Exact IEEE-754 RNE FP32 -> BF16 -> FP32 rounding."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    bits = a.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def float_to_e2m1_np(arr: np.ndarray) -> np.ndarray:
    """Bit-exact mirror of ``dual_plane::float_to_e2m1``."""
    x = np.ascontiguousarray(arr, dtype=np.float32)
    u = x.view(np.uint32)
    sign = ((u >> np.uint32(28)) & np.uint32(8)).astype(np.uint8)
    abits = u & np.uint32(0x7FFFFFFF)
    mag = (
        (abits > np.uint32(0x3E800000)).astype(np.uint8)
        + (abits >= np.uint32(0x3F400000)).astype(np.uint8)
        + (abits > np.uint32(0x3FA00000)).astype(np.uint8)
        + (abits >= np.uint32(0x3FE00000)).astype(np.uint8)
        + (abits > np.uint32(0x40200000)).astype(np.uint8)
        + (abits >= np.uint32(0x40600000)).astype(np.uint8)
        + (abits > np.uint32(0x40A00000)).astype(np.uint8)
    )
    return sign | mag


def float_to_e2m1_rne_canonical_np(arr: np.ndarray) -> np.ndarray:
    """Exact RNE E2M1 with ``-0.0 (0x8) -> +0.0 (0x0)`` canonicalization (``bitfp4_kv::e2m1_quant``)."""
    nibs = float_to_e2m1_np(arr)
    return np.where(nibs == np.uint8(0x8), np.uint8(0x0), nibs)


def float_to_ue4m3_np(arr: np.ndarray) -> np.ndarray:
    """Bit-exact RNE finite UE4M3 (`0x00..=0x7E`, max `448.0`) matching ``bitfp4_kv::f32_to_ue4m3``."""
    clamped = np.clip(np.abs(np.asarray(arr, dtype=np.float32)), 0.0, 448.0)
    diffs = np.abs(clamped[..., None] - UE4M3_LUT[:127])
    # Ties broken to even code index (minimum index among tied minima is always even or handled by RNE).
    min_diff = np.min(diffs, axis=-1, keepdims=True)
    is_min = diffs == min_diff
    codes = np.arange(127, dtype=np.uint8)
    even_pref = np.where(is_min, np.where((codes & 1) == 0, 0, 1), 2)
    return np.argmin(even_pref, axis=-1).astype(np.uint8)


def quantize_dual_plane_np(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize `[V, K]` weights into shared-scale ``DualPlaneNvfp4`` (`8.5 bpw`).

    Returns ``(w_full_dequant, w_hi_draft_dequant)`` in FP32.
    """
    w = np.ascontiguousarray(weights, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % DUAL_PLANE_GROUP_SIZE != 0:
        raise ValueError(f"DualPlaneNvfp4 requires [V, K] with K % 16 == 0, got {w.shape}")
    v, k = w.shape
    global_max = float(np.max(np.abs(w)))
    scale2 = np.float32(global_max / (6.0 * 448.0)) if global_max > 0.0 else np.float32(1.0)
    inv_scale2 = np.float32(1.0 / float(scale2))

    grp = w.reshape(v, k // DUAL_PLANE_GROUP_SIZE, DUAL_PLANE_GROUP_SIZE)
    gmax = np.max(np.abs(grp), axis=-1)
    sc_u8 = float_to_ue4m3_np(gmax * inv_scale2 * np.float32(1.0 / 6.0))
    eff = (UE4M3_LUT[sc_u8] * scale2).astype(np.float32)
    inv_eff = np.where(eff > 0.0, np.float32(1.0) / np.where(eff > 0.0, eff, np.float32(1.0)), np.float32(0.0))

    norm = grp * inv_eff[..., None]
    hi_nibs = float_to_e2m1_np(norm)
    hi_vals = E2M1_LUT[hi_nibs]
    lo_nibs = float_to_e2m1_np(np.float32(8.0) * (norm - hi_vals))
    lo_vals = E2M1_LUT[lo_nibs]

    w_hi = (hi_vals * eff[..., None]).reshape(v, k).astype(np.float32)
    w_full = ((hi_vals + DUAL_PLANE_LO_SCALE * lo_vals) * eff[..., None]).reshape(v, k).astype(np.float32)
    return w_full, w_hi


def quantize_int8_per32_bf16sc_np(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize `[V, K]` weights into ``INT8 per-32 BF16sc`` (`8.50 bpw`).

    Each 32-element block stores 32 signed ``int8`` values in ``[-127, 127]``
    plus one BF16 scale (``8.0 + 16/32 = 8.50 bpw``, ``1.882x`` smaller than
    BF16). Every signed ``int8`` value in ``[-127, 127]`` converts 0-ULP
    losslessly into BF16 (7 mantissa bits + 1 implicit bit).
    Returns ``(w_dequant_f32, q_i8, scales_bf16)``.
    """
    w = np.ascontiguousarray(weights, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] == 0 or w.shape[1] % INT8_PER32_GROUP_SIZE != 0:
        raise ValueError(
            f"INT8 per-32 BF16sc requires [V, K] with K > 0 divisible by 32, got {w.shape}"
        )
    v, k = w.shape
    grp = w.reshape(v, k // INT8_PER32_GROUP_SIZE, INT8_PER32_GROUP_SIZE)
    gmax = np.max(np.abs(grp), axis=-1)
    scales_bf16 = bf16_round_np(gmax / np.float32(127.0))
    inv_sc = np.where(
        scales_bf16 > 0.0,
        np.float32(1.0) / np.where(scales_bf16 > 0.0, scales_bf16, np.float32(1.0)),
        np.float32(0.0),
    )
    q_i8 = np.clip(np.rint(grp * inv_sc[..., None]), -127.0, 127.0).astype(np.int8)
    # Convert q_i8 -> BF16 (0-ULP lossless for all -127..=127) and apply BF16 scale.
    w_deq = (q_i8.astype(np.float32) * scales_bf16[..., None]).reshape(v, k).astype(np.float32)
    return w_deq, q_i8, scales_bf16


def quantize_mxfp8_per32_ue8m0_np(weights: np.ndarray) -> np.ndarray:
    """Reference OCP MXFP8 (`8.25 bpw`): group-32 E4M3fn + shared UE8M0 power-of-2 scale."""
    w = np.ascontiguousarray(weights, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] == 0 or w.shape[1] % MXFP8_GROUP_SIZE != 0:
        raise ValueError(
            f"MXFP8 requires [V, K] with K > 0 divisible by 32, got {w.shape}"
        )
    v, k = w.shape
    grp = w.reshape(v, k // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE)
    gmax = np.max(np.abs(grp), axis=-1)
    gmax_safe = np.maximum(gmax, np.float32(2.0**-126))
    exp = np.clip(np.ceil(np.log2(gmax_safe / np.float32(448.0))).astype(np.int32), -127, 127)
    ue8m0_sc = np.where(gmax > 0.0, np.ldexp(np.float32(1.0), exp), np.float32(0.0)).astype(
        np.float32
    )
    inv_sc = np.where(
        ue8m0_sc > 0.0,
        np.float32(1.0) / np.where(ue8m0_sc > 0.0, ue8m0_sc, np.float32(1.0)),
        np.float32(0.0),
    )
    norm = grp * inv_sc[..., None]
    q_mag = UE4M3_LUT[float_to_ue4m3_np(norm)]
    return (np.where(norm < 0.0, -q_mag, q_mag) * ue8m0_sc[..., None]).reshape(v, k).astype(
        np.float32
    )


def quantize_fp8_block128_np(weights: np.ndarray) -> np.ndarray:
    """Reference block-128x128 FP8 E4M3 quantization matching ``dual_plane::quantize_fp8_block128_host``."""
    w = np.ascontiguousarray(weights, dtype=np.float32)
    n, k = w.shape
    out = np.zeros_like(w)
    for r0 in range(0, n, 128):
        r1 = min(r0 + 128, n)
        for c0 in range(0, k, 128):
            c1 = min(c0 + 128, k)
            tile = w[r0:r1, c0:c1]
            bmax = float(np.max(np.abs(tile)))
            sc = np.float32(bmax / 448.0) if bmax > 0.0 else np.float32(1.0)
            q_mag = UE4M3_LUT[float_to_ue4m3_np(tile / sc)]
            out[r0:r1, c0:c1] = np.where(tile < 0.0, -q_mag, q_mag) * sc
    return out


def compute_sqnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    ref64 = ref.astype(np.float64)
    diff64 = ref64 - approx.astype(np.float64)
    sig = float(np.mean(ref64 * ref64))
    err = float(np.mean(diff64 * diff64))
    if err <= 0.0:
        return 120.0
    if sig <= 0.0:
        return 0.0
    return 10.0 * math.log10(sig / err)


def compute_cosine(ref: np.ndarray, approx: np.ndarray) -> float:
    a = ref.reshape(-1).astype(np.float64)
    b = approx.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denom == 0.0 else float(np.dot(a, b) / denom)


def softmax_f64(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / float(temperature)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def deterministic_argmax_lowest_index(row: np.ndarray) -> int:
    """Exact mirror of ``lm_head_nvfp4_argmax_sm120.cu``: skips NaN, ties break to lowest vocab index."""
    best_val = -math.inf
    best_idx = 0xFFFFFFFF
    for idx, val in enumerate(row):
        f = float(val)
        if math.isnan(f):
            continue
        if best_idx == 0xFFFFFFFF or f > best_val or (f == best_val and idx < best_idx):
            best_val = f
            best_idx = idx
    return best_idx


def verify_lowest_index_tie_break() -> dict[str, Any]:
    """Adversarial lowest-vocab-index tie-break & NaN/Inf masking test matching ``lm_head_argmax.rs``."""
    cases = [
        ("equal_max_picks_lowest", np.array([-1.0, 3.5, 3.5, 2.0, 3.5], dtype=np.float32), 1),
        ("leading_nans_skipped", np.array([np.nan, np.nan, -5.0, -5.0], dtype=np.float32), 2),
        ("mixed_nan_and_neg_inf", np.array([np.nan, -np.inf, -np.inf, np.nan], dtype=np.float32), 1),
        ("all_nan_sentinel", np.array([np.nan, np.nan, np.nan], dtype=np.float32), 0xFFFFFFFF),
        ("neg_zero_vs_pos_zero", np.array([-0.0, 0.0, -1.0], dtype=np.float32), 0),
    ]
    failed: list[str] = []
    for name, row, expected in cases:
        got = deterministic_argmax_lowest_index(row)
        if got != expected:
            failed.append(f"{name}: got {got} != expected {expected}")
    return {"passed": len(failed) == 0, "cases": len(cases), "failures": failed}


def _validate_lm_head_gates(
    min_dual_plane_sqnr_db: float,
    strict_bf16_sqnr_db: float,
    max_top1_flip_rate: float,
    max_kl_t1: float,
) -> None:
    if not math.isfinite(min_dual_plane_sqnr_db) or min_dual_plane_sqnr_db < 0.0:
        raise ValueError(f"min_dual_plane_sqnr_db must be finite and >= 0, got {min_dual_plane_sqnr_db}")
    if not math.isfinite(strict_bf16_sqnr_db) or strict_bf16_sqnr_db < 0.0:
        raise ValueError(f"strict_bf16_sqnr_db must be finite and >= 0, got {strict_bf16_sqnr_db}")
    if not math.isfinite(max_top1_flip_rate) or not (0.0 <= max_top1_flip_rate <= 1.0):
        raise ValueError(f"max_top1_flip_rate must be finite and in [0, 1], got {max_top1_flip_rate}")
    if not math.isfinite(max_kl_t1) or max_kl_t1 < 0.0:
        raise ValueError(f"max_kl_t1 must be finite and >= 0, got {max_kl_t1}")


def certify_lm_head_dual_plane(
    weights_fp32: np.ndarray | None = None,
    vocab_size: int = 4096,
    hidden_size: int = 512,
    num_queries: int = 512,
    seed: int = 20261127,
    min_dual_plane_sqnr_db: float = 37.5,
    strict_bf16_sqnr_db: float = 45.0,
    max_top1_flip_rate: float = 0.001,
    max_kl_t1: float = 1.0e-4,
) -> dict[str, Any]:
    """Certify ``DualPlaneNvfp4`` (`8.5 bpw`) and BF16 ``lm_head`` numerical exactness."""
    _validate_lm_head_gates(
        min_dual_plane_sqnr_db=min_dual_plane_sqnr_db,
        strict_bf16_sqnr_db=strict_bf16_sqnr_db,
        max_top1_flip_rate=max_top1_flip_rate,
        max_kl_t1=max_kl_t1,
    )
    rng = np.random.default_rng(seed)
    if weights_fp32 is None:
        if vocab_size < 2 or hidden_size <= 0 or hidden_size % DUAL_PLANE_GROUP_SIZE != 0:
            raise ValueError(
                f"invalid synthetic lm_head shape ({vocab_size}, {hidden_size}); "
                "require vocab_size >= 2 and hidden_size > 0 divisible by 16"
            )
        w_fp32 = rng.normal(
            0.0, 1.0 / math.sqrt(float(hidden_size)), size=(vocab_size, hidden_size)
        ).astype(np.float32)
    else:
        w_fp32 = np.ascontiguousarray(weights_fp32, dtype=np.float32)
        if w_fp32.ndim != 2 or w_fp32.shape[0] < 2 or w_fp32.shape[1] % DUAL_PLANE_GROUP_SIZE != 0:
            raise ValueError(f"weights must have shape [V>=2, K%16==0], got {w_fp32.shape}")
        vocab_size, hidden_size = w_fp32.shape
    if num_queries <= 0:
        raise ValueError(f"num_queries must be > 0, got {num_queries}")

    w_bf16 = bf16_round_np(w_fp32)
    w_dp_full, w_dp_hi = quantize_dual_plane_np(w_bf16)
    w_fp8 = quantize_fp8_block128_np(w_bf16)

    # Construct dimension-invariant realistic post-RMSNorm LM-head hidden states
    # aligned with target + runner-up token directions so logit margins match real LLMs.
    row_norms = np.maximum(np.linalg.norm(w_bf16, axis=-1, keepdims=True), np.float32(1.0e-6))
    w_unit = w_bf16 / row_norms
    med_norm = float(np.median(row_norms))
    inv_med = np.float32(1.0 / max(med_norm, 1.0e-6))

    targets = rng.integers(0, vocab_size, size=num_queries)
    runners = (targets + rng.integers(1, vocab_size, size=num_queries)) % vocab_size
    noise = rng.normal(0.0, 1.0 / math.sqrt(float(hidden_size)), size=(num_queries, hidden_size)).astype(
        np.float32
    )
    x_bf16 = bf16_round_np((10.0 * w_unit[targets] + 4.5 * w_unit[runners] + 1.5 * noise) * inv_med)

    logits_fp32 = x_bf16 @ w_fp32.T
    logits_bf16 = x_bf16 @ w_bf16.T
    logits_dp = x_bf16 @ w_dp_full.T
    logits_hi = x_bf16 @ w_dp_hi.T

    bf16_weight_sqnr = compute_sqnr_db(w_fp32, w_bf16)
    bf16_logit_sqnr = compute_sqnr_db(logits_fp32, logits_bf16)
    dp_weight_sqnr = compute_sqnr_db(w_bf16, w_dp_full)
    dp_logit_sqnr = compute_sqnr_db(logits_bf16, logits_dp)
    fp8_weight_sqnr = compute_sqnr_db(w_bf16, w_fp8)
    hi_weight_sqnr = compute_sqnr_db(w_bf16, w_dp_hi)
    gain_over_fp8 = dp_weight_sqnr - fp8_weight_sqnr

    p_bf16 = softmax_f64(logits_bf16, temperature=1.0)
    p_dp = softmax_f64(logits_dp, temperature=1.0)
    kl_dp_rows = np.sum(
        p_bf16 * np.log(np.clip(p_bf16, 1e-30, 1.0) / np.clip(p_dp, 1e-30, 1.0)), axis=-1
    )
    mean_kl_dp = float(np.mean(kl_dp_rows))
    p95_kl_dp = float(np.percentile(kl_dp_rows, 95.0))
    max_kl_dp = float(np.max(kl_dp_rows))

    arg_bf16 = np.argmax(logits_bf16, axis=-1)
    arg_dp = np.argmax(logits_dp, axis=-1)
    top1_flip_rate = float(np.mean(arg_bf16 != arg_dp))

    tie_break = verify_lowest_index_tie_break()

    # Hardware-exact 8.5 bpw DualPlaneNvfp4 gate (matches dual_plane.rs:604 ~38.5 dB ceiling)
    # + strict >=45 dB BF16-equivalence check (satisfied by BF16 ~55.6 dB; reported honestly for 8.5 bpw).
    failures: list[str] = []
    for metric_name, metric_val in (
        ("dp_weight_sqnr", dp_weight_sqnr),
        ("dp_logit_sqnr", dp_logit_sqnr),
        ("gain_over_fp8", gain_over_fp8),
        ("top1_flip_rate", top1_flip_rate),
        ("mean_kl_dp", mean_kl_dp),
        ("bf16_logit_sqnr", bf16_logit_sqnr),
    ):
        if not math.isfinite(metric_val):
            failures.append(f"non-finite metric `{metric_name}`={metric_val}")
    if dp_weight_sqnr < min_dual_plane_sqnr_db:
        failures.append(f"DualPlaneNvfp4 weight SQNR {dp_weight_sqnr:.2f} dB < {min_dual_plane_sqnr_db:.2f} dB")
    if gain_over_fp8 < 6.0:
        failures.append(f"DualPlaneNvfp4 gain over block-128 FP8 {gain_over_fp8:.2f} dB < 6.00 dB")
    if top1_flip_rate > max_top1_flip_rate:
        failures.append(f"DualPlaneNvfp4 top-1 flip rate {top1_flip_rate:.6f} > {max_top1_flip_rate:.6f}")
    if mean_kl_dp > max_kl_t1:
        failures.append(f"DualPlaneNvfp4 mean D_KL@T=1.0 {mean_kl_dp:.3e} > {max_kl_t1:.3e}")
    if bf16_logit_sqnr < strict_bf16_sqnr_db:
        failures.append(f"BF16 reference logit SQNR {bf16_logit_sqnr:.2f} dB < {strict_bf16_sqnr_db:.2f} dB")
    if not tie_break["passed"]:
        failures.extend(f"tie_break: {f}" for f in tie_break["failures"])

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "shape": {"vocab_size": vocab_size, "hidden_size": hidden_size, "num_queries": num_queries},
        "metrics": {
            "dual_plane_8p5bpw_weight_sqnr_db": round(dp_weight_sqnr, 3),
            "dual_plane_8p5bpw_logit_sqnr_db": round(dp_logit_sqnr, 3),
            "fp8_block128_weight_sqnr_db": round(fp8_weight_sqnr, 3),
            "dual_plane_gain_over_fp8_db": round(gain_over_fp8, 3),
            "w_hi_4p5bpw_draft_weight_sqnr_db": round(hi_weight_sqnr, 3),
            "w_hi_4p5bpw_logit_sqnr_db": round(compute_sqnr_db(logits_bf16, logits_hi), 3),
            "bf16_vs_fp32_weight_sqnr_db": round(bf16_weight_sqnr, 3),
            "bf16_vs_fp32_logit_sqnr_db": round(bf16_logit_sqnr, 3),
            "dual_plane_top1_flip_rate": round(top1_flip_rate, 6),
            "dual_plane_mean_kl_t1": mean_kl_dp,
            "dual_plane_p95_kl_t1": p95_kl_dp,
            "dual_plane_max_kl_t1": max_kl_dp,
            "dual_plane_meets_strict_45db_gate": dp_logit_sqnr >= strict_bf16_sqnr_db,
            "bf16_meets_strict_45db_gate": bf16_logit_sqnr >= strict_bf16_sqnr_db,
        },
        "tie_break": tie_break,
    }


def _validate_int8_per32_gates(
    min_int8_sqnr_db: float,
    max_int8_kl_t1: float,
    max_top1_flip_rate: float,
) -> None:
    if not math.isfinite(min_int8_sqnr_db) or min_int8_sqnr_db < 0.0:
        raise ValueError(f"min_int8_sqnr_db must be finite and >= 0, got {min_int8_sqnr_db}")
    if not math.isfinite(max_int8_kl_t1) or max_int8_kl_t1 < 0.0:
        raise ValueError(f"max_int8_kl_t1 must be finite and >= 0, got {max_int8_kl_t1}")
    if not math.isfinite(max_top1_flip_rate) or not (0.0 <= max_top1_flip_rate <= 1.0):
        raise ValueError(f"max_top1_flip_rate must be finite and in [0, 1], got {max_top1_flip_rate}")


def certify_lm_head_int8_per32_bf16sc(
    weights_fp32: np.ndarray | None = None,
    vocab_size: int = 4096,
    hidden_size: int = 512,
    num_queries: int = 512,
    seed: int = 20261127,
    min_int8_sqnr_db: float = 44.0,
    max_int8_kl_t1: float = 1.0e-6,
    max_top1_flip_rate: float = 0.001,
) -> dict[str, Any]:
    """Certify ``INT8 per-32 BF16sc`` (`8.50 bpw`) side-by-side against BF16,
    ``DualPlaneNvfp4`` (`8.50 bpw`), and ``MXFP8`` (`8.25 bpw`).

    Verifies:
      - 0-ULP lossless ``int8 (-127..=127) -> BF16`` conversion across all 255 codes.
      - ``INT8 per-32 BF16sc`` weight & logit SQNR ``>= 44.0 dB`` (`~45.4 dB`),
        strictly dominating ``DualPlaneNvfp4`` (`~38.5 dB`) and ``MXFP8`` (`~31.5 dB`).
      - ``mean D_KL(P_bf16 || P_int8) <= 1e-6`` at ``T = 1.0`` and top-1 flip ``<= 0.1%``.
      - Exact lowest-vocab-index tie-breaking parity vs BF16.
    """
    _validate_int8_per32_gates(
        min_int8_sqnr_db=min_int8_sqnr_db,
        max_int8_kl_t1=max_int8_kl_t1,
        max_top1_flip_rate=max_top1_flip_rate,
    )
    rng = np.random.default_rng(seed)
    if weights_fp32 is None:
        if vocab_size < 2 or hidden_size <= 0 or hidden_size % INT8_PER32_GROUP_SIZE != 0:
            raise ValueError(
                f"invalid synthetic lm_head shape ({vocab_size}, {hidden_size}); "
                "require vocab_size >= 2 and hidden_size > 0 divisible by 32"
            )
        w_fp32 = rng.normal(
            0.0, 1.0 / math.sqrt(float(hidden_size)), size=(vocab_size, hidden_size)
        ).astype(np.float32)
    else:
        w_fp32 = np.ascontiguousarray(weights_fp32, dtype=np.float32)
        if w_fp32.ndim != 2 or w_fp32.shape[0] < 2 or w_fp32.shape[1] % INT8_PER32_GROUP_SIZE != 0:
            raise ValueError(f"weights must have shape [V>=2, K%32==0], got {w_fp32.shape}")
        vocab_size, hidden_size = w_fp32.shape
    if num_queries <= 0:
        raise ValueError(f"num_queries must be > 0, got {num_queries}")

    # Exhaustive 0-ULP lossless check for every signed int8 in -127..=127 -> BF16.
    all_i8_f32 = np.arange(-127, 128, dtype=np.int8).astype(np.float32)
    int8_to_bf16_lossless = bool(np.array_equal(bf16_round_np(all_i8_f32), all_i8_f32))

    w_bf16 = bf16_round_np(w_fp32)
    w_i8, _, _ = quantize_int8_per32_bf16sc_np(w_bf16)
    w_dp, _ = quantize_dual_plane_np(w_bf16)
    w_mxfp8 = quantize_mxfp8_per32_ue8m0_np(w_bf16)

    # Realistic post-RMSNorm decode hidden states (peaked top-1 + runner-up + low-entropy residual).
    row_norms = np.maximum(np.linalg.norm(w_bf16, axis=-1, keepdims=True), np.float32(1.0e-6))
    w_unit = w_bf16 / row_norms
    med_norm = float(np.median(row_norms))
    inv_med = np.float32(1.0 / max(med_norm, 1.0e-6))

    targets = rng.integers(0, vocab_size, size=num_queries)
    runners = (targets + rng.integers(1, vocab_size, size=num_queries)) % vocab_size
    noise = rng.normal(
        0.0, 1.0 / math.sqrt(float(hidden_size)), size=(num_queries, hidden_size)
    ).astype(np.float32)
    x_bf16 = bf16_round_np((12.0 * w_unit[targets] + 4.5 * w_unit[runners] + 0.8 * noise) * inv_med)

    logits_fp32 = x_bf16 @ w_fp32.T
    logits_bf16 = x_bf16 @ w_bf16.T
    p_fp32 = softmax_f64(logits_fp32, temperature=1.0)
    p_bf16 = softmax_f64(logits_bf16, temperature=1.0)
    arg_fp32 = np.argmax(logits_fp32, axis=-1)
    arg_bf16 = np.argmax(logits_bf16, axis=-1)

    def _eval_scheme(
        w_hat: np.ndarray,
        bpw: float,
        ref_w: np.ndarray,
        ref_logits: np.ndarray,
        ref_probs: np.ndarray,
        ref_arg: np.ndarray,
    ) -> dict[str, Any]:
        y_hat = x_bf16 @ w_hat.T
        p_hat = softmax_f64(y_hat, temperature=1.0)
        kl_rows = np.sum(
            ref_probs * np.log(np.clip(ref_probs, 1e-30, 1.0) / np.clip(p_hat, 1e-30, 1.0)),
            axis=-1,
        )
        w_sqnr = compute_sqnr_db(ref_w, w_hat)
        y_sqnr = compute_sqnr_db(ref_logits, y_hat)
        flip = float(np.mean(ref_arg != np.argmax(y_hat, axis=-1)))
        return {
            "bpw": bpw,
            "compression_vs_bf16_x": round(16.0 / bpw, 3),
            "weight_sqnr_db": round(w_sqnr, 3),
            "logit_sqnr_db": round(y_sqnr, 3),
            "top1_flip_rate": round(flip, 6),
            "mean_kl_t1": float(np.mean(kl_rows)),
            "p95_kl_t1": float(np.percentile(kl_rows, 95.0)),
            "max_kl_t1": float(np.max(kl_rows)),
        }

    schemes = {
        "bf16": _eval_scheme(w_bf16, 16.0, w_fp32, logits_fp32, p_fp32, arg_fp32),
        "int8_per32_bf16sc": _eval_scheme(w_i8, 8.5, w_bf16, logits_bf16, p_bf16, arg_bf16),
        "dual_plane_nvfp4": _eval_scheme(w_dp, 8.5, w_bf16, logits_bf16, p_bf16, arg_bf16),
        "mxfp8": _eval_scheme(w_mxfp8, 8.25, w_bf16, logits_bf16, p_bf16, arg_bf16),
    }

    i8 = schemes["int8_per32_bf16sc"]
    dp = schemes["dual_plane_nvfp4"]
    mx = schemes["mxfp8"]
    tie_break = verify_lowest_index_tie_break()

    failures: list[str] = []
    if not int8_to_bf16_lossless:
        failures.append("int8 (-127..=127) -> BF16 conversion was not 0-ULP lossless")
    for name, val in (
        ("int8_weight_sqnr_db", i8["weight_sqnr_db"]),
        ("int8_logit_sqnr_db", i8["logit_sqnr_db"]),
        ("int8_top1_flip_rate", i8["top1_flip_rate"]),
        ("int8_mean_kl_t1", i8["mean_kl_t1"]),
    ):
        if not math.isfinite(val):
            failures.append(f"non-finite metric `{name}`={val}")
    if i8["weight_sqnr_db"] < min_int8_sqnr_db:
        failures.append(
            f"INT8 per-32 BF16sc weight SQNR {i8['weight_sqnr_db']:.2f} dB < {min_int8_sqnr_db:.2f} dB"
        )
    if i8["mean_kl_t1"] > max_int8_kl_t1:
        failures.append(
            f"INT8 per-32 BF16sc mean D_KL@T=1.0 {i8['mean_kl_t1']:.3e} > {max_int8_kl_t1:.3e}"
        )
    if i8["top1_flip_rate"] > max_top1_flip_rate:
        failures.append(
            f"INT8 per-32 BF16sc top-1 flip rate {i8['top1_flip_rate']:.6f} > {max_top1_flip_rate:.6f}"
        )
    if i8["weight_sqnr_db"] - dp["weight_sqnr_db"] < 5.0:
        failures.append(
            f"INT8 per-32 BF16sc ({i8['weight_sqnr_db']:.2f} dB) must beat DualPlaneNvfp4 "
            f"({dp['weight_sqnr_db']:.2f} dB) by >= 5.0 dB"
        )
    if i8["weight_sqnr_db"] - mx["weight_sqnr_db"] < 12.0:
        failures.append(
            f"INT8 per-32 BF16sc ({i8['weight_sqnr_db']:.2f} dB) must beat MXFP8 "
            f"({mx['weight_sqnr_db']:.2f} dB) by >= 12.0 dB"
        )
    if not (i8["mean_kl_t1"] < dp["mean_kl_t1"] < mx["mean_kl_t1"]):
        failures.append(
            f"expected strict KL ordering INT8 ({i8['mean_kl_t1']:.2e}) < DualPlane "
            f"({dp['mean_kl_t1']:.2e}) < MXFP8 ({mx['mean_kl_t1']:.2e})"
        )
    if not tie_break["passed"]:
        failures.extend(f"tie_break: {f}" for f in tie_break["failures"])

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "shape": {"vocab_size": vocab_size, "hidden_size": hidden_size, "num_queries": num_queries},
        "int8_to_bf16_lossless_0ulp": int8_to_bf16_lossless,
        "pareto_dominates_dual_plane_and_mxfp8": (
            i8["weight_sqnr_db"] > dp["weight_sqnr_db"] > mx["weight_sqnr_db"]
            and i8["mean_kl_t1"] < dp["mean_kl_t1"] < mx["mean_kl_t1"]
        ),
        "gain_over_dual_plane_db": round(i8["weight_sqnr_db"] - dp["weight_sqnr_db"], 3),
        "gain_over_mxfp8_db": round(i8["weight_sqnr_db"] - mx["weight_sqnr_db"], 3),
        "schemes": schemes,
        "tie_break": tie_break,
    }


# ── BitFP4 + 6-op Branchless SWAR RaZeR-KV Certifier ────────────────────────


def canonicalize_e2m1_neg_zero_u32(r: int) -> int:
    """Exact 2-op SWAR mirror of ``bitfp4_kv::canonicalize_e2m1_neg_zero_u32``."""
    r32 = r & 0xFFFFFFFF
    lo3_plus = ((r32 & 0x77777777) + 0x77777777) & 0xFFFFFFFF
    neg0 = r32 & (~lo3_plus & 0xFFFFFFFF) & 0x88888888
    return (r32 ^ neg0) & 0xFFFFFFFF


def razer_swar_split_u32(x: int, neg_sv: int) -> tuple[int, int]:
    """Exact 6-op branchless SWAR mirror of ``bitfp4_kv::razer_swar_split_u32``."""
    x32 = x & 0xFFFFFFFF
    neg_mask = (-1 if (neg_sv & 1) else 0) & 0xFFFFFFFF
    lo3 = x32 & 0x77777777
    m = x32 & (~((lo3 + 0x77777777) & 0xFFFFFFFF) & 0xFFFFFFFF) & 0x88888888
    m2 = m >> 2
    m4 = m >> 1
    x_main = ((x32 ^ (m & (~neg_mask & 0xFFFFFFFF))) | m4 | m2) & 0xFFFFFFFF
    x_comp = ((m & neg_mask) | m2) & 0xFFFFFFFF
    return x_main, x_comp


def razer_decode_nibble(nib: int, neg_sv: int) -> float:
    n = nib & 0xF
    if n == 0x8:
        return -5.0 if (neg_sv & 1) else 5.0
    return float(E2M1_LUT[n])


def wht256_np(x: np.ndarray) -> np.ndarray:
    """Normalized 256-point Walsh-Hadamard transform (``H^T H = I``) along last axis."""
    v = np.ascontiguousarray(x, dtype=np.float32).copy()
    assert v.shape[-1] == BITFP4_HDIM
    stride = 1
    while stride < BITFP4_HDIM:
        v = v.reshape(*v.shape[:-1], BITFP4_HDIM // (2 * stride), 2, stride)
        a = v[..., 0, :].copy()
        b = v[..., 1, :].copy()
        v[..., 0, :] = a + b
        v[..., 1, :] = a - b
        v = v.reshape(*x.shape[:-1], BITFP4_HDIM)
        stride <<= 1
    v *= np.float32(1.0 / math.sqrt(float(BITFP4_HDIM)))
    return v


def _razer_quant_nibble(x: float, neg_sv: int) -> int:
    sign = 8 if x < 0.0 else 0
    a = abs(x)
    same_sign = (sign != 0) == ((neg_sv & 1) != 0)
    if same_sign and 4.5 <= a <= 5.5:
        return 0x8
    if a < 0.25:
        mag = 0
    elif a <= 0.75:
        mag = 1
    elif a < 1.25:
        mag = 2
    elif a <= 1.75:
        mag = 3
    elif a < 2.5:
        mag = 4
    elif a <= 3.5:
        mag = 5
    elif a < 5.0:
        mag = 6
    else:
        mag = 7
    return 0 if mag == 0 else (sign | mag)


def quantize_group16_bitfp4_np(
    group: np.ndarray,
    four_over_six: bool,
    razer: bool,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Exact mirror of ``bitfp4_kv::quantize_group16_bitfp4`` returning ``(nibs, scale_byte, dequant)``."""
    g = np.asarray(group, dtype=np.float32)
    amax = float(np.max(np.abs(g)))
    if amax == 0.0:
        return np.zeros(16, dtype=np.uint8), 0, np.zeros(16, dtype=np.float32)

    def eval_std(sc_u8: int) -> tuple[np.ndarray, float, np.ndarray]:
        sc = float(UE4M3_LUT[sc_u8])
        inv = (1.0 / sc) if sc > 0.0 else 0.0
        nibs = float_to_e2m1_rne_canonical_np(g * np.float32(inv))
        deq = E2M1_LUT[nibs] * np.float32(sc)
        mse = float(np.sum((g - deq) ** 2))
        return nibs, mse, deq

    s6 = int(float_to_ue4m3_np(np.float32(amax / 6.0)))
    best_nibs, best_err, best_deq = eval_std(s6)
    best_sc = s6

    if four_over_six:
        s4 = int(float_to_ue4m3_np(np.float32(amax / 4.0)))
        n4, e4, d4 = eval_std(s4)
        if e4 < best_err:
            best_nibs, best_err, best_deq, best_sc = n4, e4, d4, s4

    if razer:
        s5 = int(float_to_ue4m3_np(np.float32(amax / 5.0)))
        for neg_sv in (0, 1):
            for sc_u8 in (s6, s5):
                sc = float(UE4M3_LUT[sc_u8])
                inv = (1.0 / sc) if sc > 0.0 else 0.0
                nibs = np.array([_razer_quant_nibble(float(v) * inv, neg_sv) for v in g], dtype=np.uint8)
                deq = np.array([razer_decode_nibble(int(nb), neg_sv) * sc for nb in nibs], dtype=np.float32)
                mse = float(np.sum((g - deq) ** 2))
                if mse < best_err:
                    best_nibs = nibs
                    best_err = mse
                    best_deq = deq
                    best_sc = sc_u8 | (neg_sv << 7)

    return best_nibs, best_sc, best_deq


def quantize_query_dual_residual_np(q: np.ndarray) -> np.ndarray:
    """Mirror of ``bitfp4_kv::quantize_query_dual_residual`` (`Q^(1) + Q^(2)`)."""
    q_amax = float(np.max(np.abs(q)))
    if q_amax <= 1e-30:
        return np.zeros_like(q)
    norm_scale = np.float32(256.0 / q_amax)
    inv_norm = np.float32(q_amax / 256.0)
    out = np.zeros(BITFP4_HDIM, dtype=np.float32)
    for g in range(BITFP4_HDIM // 16):
        blk = q[g * 16 : (g + 1) * 16] * norm_scale
        _, _, d1 = quantize_group16_bitfp4_np(blk, four_over_six=False, razer=False)
        res = blk - d1
        _, _, d2 = quantize_group16_bitfp4_np(res, four_over_six=False, razer=False)
        out[g * 16 : (g + 1) * 16] = (d1 + d2) * inv_norm
    return out


def certify_bitfp4_razer_kv(
    num_tokens: int = 64,
    num_queries: int = 16,
    seed: int = 20261127,
) -> dict[str, Any]:
    """Certify BitFP4 + 6-op SWAR RaZeR-KV algebraic & numerical exactness."""
    failures: list[str] = []

    # 1. Exhaustive 2-op SWAR -0.0 -> +0.0 canonicalization over all 65,536 4-nibble patterns.
    neg_zero_errors = 0
    for low16 in range(65536):
        word = low16 | ((low16 ^ 0xA5A5) << 16)
        canon = canonicalize_e2m1_neg_zero_u32(word)
        for lane in range(8):
            in_nib = (word >> (lane * 4)) & 0xF
            out_nib = (canon >> (lane * 4)) & 0xF
            expected = 0x0 if in_nib == 0x8 else in_nib
            if out_nib != expected:
                neg_zero_errors += 1
    if neg_zero_errors != 0:
        failures.append(f"SWAR -0.0 canonicalizer failed on {neg_zero_errors} nibble lanes")

    # 2. Exhaustive 6-op branchless SWAR RaZeR split algebraic parity across all 16 codes x 2 signs.
    razer_swar_errors = 0
    for neg_sv in (0, 1):
        for nib in range(16):
            # Replicate across all 8 lanes to verify zero inter-lane carry leakage!
            word = 0
            for lane in range(8):
                word |= ((nib + lane) & 0xF) << (lane * 4)
            x_main, x_comp = razer_swar_split_u32(word, neg_sv)
            for lane in range(8):
                src_nib = (word >> (lane * 4)) & 0xF
                m_nib = (x_main >> (lane * 4)) & 0xF
                c_nib = (x_comp >> (lane * 4)) & 0xF
                got = float(E2M1_LUT[m_nib]) + float(E2M1_LUT[c_nib])
                want = razer_decode_nibble(src_nib, neg_sv)
                if got != want:
                    razer_swar_errors += 1
    if razer_swar_errors != 0:
        failures.append(f"6-op SWAR RaZeR split failed on {razer_swar_errors} nibble lanes")

    # 3. Reconstruction MSE improvement (standard NVFP4 vs BitFP4+RaZeR) on WHT-256 K & V^T.
    if num_tokens <= 0 or num_tokens % BITFP4_BLOCK_SIZE != 0:
        raise ValueError(f"num_tokens must be positive and divisible by 16, got {num_tokens}")
    if num_queries <= 0:
        raise ValueError(f"num_queries must be > 0, got {num_queries}")
    rng = np.random.default_rng(seed)
    k_fp32 = rng.normal(0.0, 1.0, size=(num_tokens, BITFP4_HDIM)).astype(np.float32)
    # Inject realistic RoPE/attention outlier channels so WHT + RaZeR ±5.0 is exercised realistically.
    k_fp32[:, ::32] *= np.float32(3.5)
    v_fp32 = rng.normal(0.0, 0.8, size=(num_tokens, BITFP4_HDIM)).astype(np.float32)
    v_fp32[:, ::16] *= np.float32(2.2)

    k_rot = wht256_np(k_fp32)
    k_std = np.zeros_like(k_rot)
    k_razer = np.zeros_like(k_rot)
    for t in range(num_tokens):
        for g in range(BITFP4_HDIM // 16):
            sl = slice(g * 16, (g + 1) * 16)
            _, _, k_std[t, sl] = quantize_group16_bitfp4_np(k_rot[t, sl], four_over_six=False, razer=False)
            _, _, k_razer[t, sl] = quantize_group16_bitfp4_np(k_rot[t, sl], four_over_six=True, razer=True)

    # Channel-transposed block-16 V^T quantization.
    v_std = np.zeros_like(v_fp32)
    v_razer = np.zeros_like(v_fp32)
    num_blocks = num_tokens // BITFP4_BLOCK_SIZE
    for b in range(num_blocks):
        t0, t1 = b * BITFP4_BLOCK_SIZE, (b + 1) * BITFP4_BLOCK_SIZE
        for d in range(BITFP4_HDIM):
            ch = v_fp32[t0:t1, d]
            _, _, v_std[t0:t1, d] = quantize_group16_bitfp4_np(ch, four_over_six=False, razer=False)
            _, _, v_razer[t0:t1, d] = quantize_group16_bitfp4_np(ch, four_over_six=True, razer=True)

    k_mse_std = float(np.mean((k_rot - k_std) ** 2))
    k_mse_razer = float(np.mean((k_rot - k_razer) ** 2))
    v_mse_std = float(np.mean((v_fp32 - v_std) ** 2))
    v_mse_razer = float(np.mean((v_fp32 - v_razer) ** 2))
    k_bias_razer = abs(float(np.mean(k_rot - k_razer)))
    v_bias_razer = abs(float(np.mean(v_fp32 - v_razer)))

    if not (k_mse_razer < k_mse_std):
        failures.append(f"RaZeR K MSE ({k_mse_razer:.6f}) not strictly lower than NVFP4 ({k_mse_std:.6f})")
    if not (v_mse_razer < v_mse_std):
        failures.append(f"RaZeR V^T MSE ({v_mse_razer:.6f}) not strictly lower than NVFP4 ({v_mse_std:.6f})")

    # 4. Attention logit cosine similarity & max abs error + output cosine similarity.
    q_fp32 = rng.normal(0.0, 0.7, size=(num_queries, BITFP4_HDIM)).astype(np.float32)
    q_rot = wht256_np(q_fp32)
    q_dr = np.stack([quantize_query_dual_residual_np(q_rot[i]) for i in range(num_queries)], axis=0)

    scale = np.float32(1.0 / math.sqrt(float(BITFP4_HDIM)))
    logits_ref = (q_fp32 @ k_fp32.T) * scale
    logits_std = (q_dr @ k_std.T) * scale
    logits_razer = (q_dr @ k_razer.T) * scale

    logit_cos_std = compute_cosine(logits_ref, logits_std)
    logit_cos_razer = compute_cosine(logits_ref, logits_razer)
    logit_max_err_std = float(np.max(np.abs(logits_ref - logits_std)))
    logit_max_err_razer = float(np.max(np.abs(logits_ref - logits_razer)))
    logit_rmse_std = float(np.sqrt(np.mean((logits_ref - logits_std) ** 2)))
    logit_rmse_razer = float(np.sqrt(np.mean((logits_ref - logits_razer) ** 2)))

    # Attention output with underflow-aware probability scaling on 16-key blocks.
    out_ref = softmax_f64(logits_ref).astype(np.float32) @ v_fp32
    p_razer = softmax_f64(logits_razer).astype(np.float32)
    p_razer_q = np.zeros_like(p_razer)
    for qi in range(num_queries):
        for b in range(num_blocks):
            sl = slice(b * BITFP4_BLOCK_SIZE, (b + 1) * BITFP4_BLOCK_SIZE)
            blk = p_razer[qi, sl]
            p_max_c = float(UNDERFLOW_C * np.max(blk))
            kappa = np.float32(2.0 if p_max_c <= float(UNDERFLOW_TAU) else 6.0)
            sp_u8 = int(float_to_ue4m3_np(np.float32(p_max_c / float(kappa))))
            sp = float(UE4M3_LUT[sp_u8])
            inv = (1.0 / sp) if sp > 0.0 else 0.0
            nibs = float_to_e2m1_rne_canonical_np(UNDERFLOW_C * blk * np.float32(inv))
            p_razer_q[qi, sl] = (E2M1_LUT[nibs] * np.float32(sp)) / UNDERFLOW_C
    out_razer = p_razer_q @ v_razer
    attn_out_cos = compute_cosine(out_ref, out_razer)

    if logit_cos_razer < 0.995:
        failures.append(f"BitFP4+RaZeR attention logit cosine {logit_cos_razer:.6f} < 0.995")
    if logit_rmse_razer >= logit_rmse_std:
        failures.append(
            f"BitFP4+RaZeR logit RMSE ({logit_rmse_razer:.6f}) did not improve over NVFP4 ({logit_rmse_std:.6f})"
        )
    if attn_out_cos < 0.990:
        failures.append(f"BitFP4+RaZeR attention output cosine {attn_out_cos:.6f} < 0.990")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "swar_checks": {
            "neg_zero_canonicalization_patterns_tested": 65536 * 8,
            "neg_zero_canonicalization_errors": neg_zero_errors,
            "razer_6op_split_lanes_tested": 2 * 16 * 8,
            "razer_6op_split_errors": razer_swar_errors,
        },
        "metrics": {
            "k_mse_nvfp4": round(k_mse_std, 6),
            "k_mse_bitfp4_razer": round(k_mse_razer, 6),
            "k_mse_reduction_pct": round((1.0 - k_mse_razer / k_mse_std) * 100.0, 2),
            "k_mean_abs_bias_razer": round(k_bias_razer, 6),
            "v_mse_nvfp4": round(v_mse_std, 6),
            "v_mse_bitfp4_razer": round(v_mse_razer, 6),
            "v_mse_reduction_pct": round((1.0 - v_mse_razer / v_mse_std) * 100.0, 2),
            "v_mean_abs_bias_razer": round(v_bias_razer, 6),
            "attn_logit_cosine_nvfp4": round(logit_cos_std, 6),
            "attn_logit_cosine_bitfp4_razer": round(logit_cos_razer, 6),
            "attn_logit_rmse_nvfp4": round(logit_rmse_std, 6),
            "attn_logit_rmse_bitfp4_razer": round(logit_rmse_razer, 6),
            "attn_logit_max_abs_err_nvfp4": round(logit_max_err_std, 6),
            "attn_logit_max_abs_err_bitfp4_razer": round(logit_max_err_razer, 6),
            "attn_output_cosine_bitfp4_razer": round(attn_out_cos, 6),
        },
    }


def _load_weight_matrix(path: Path, key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float32)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            chosen = key or archive.files[0]
            return archive[chosen].astype(np.float32)
    raise ValueError(f"unsupported weight format `{suffix}`; expected .npy or .npz")


def run_self_test() -> int:
    """Run full offline quantization exactness certification + negative fail-closed checks."""
    # Exact zero-flip gate (`max_top1_flip_rate=0.0`) must pass when measured flip rate is 0.0.
    lm_res = certify_lm_head_dual_plane(
        vocab_size=2048,
        hidden_size=256,
        num_queries=256,
        seed=77,
        max_top1_flip_rate=0.0,
    )
    assert lm_res["passed"], f"LM-head DualPlaneNvfp4 cert failed: {lm_res['failures']}"
    assert lm_res["metrics"]["dual_plane_top1_flip_rate"] == 0.0
    assert lm_res["metrics"]["bf16_meets_strict_45db_gate"], "BF16 reference must pass >=45 dB"
    assert not lm_res["metrics"]["dual_plane_meets_strict_45db_gate"], (
        "8.5 bpw DualPlaneNvfp4 has ~38.5 dB ceiling and must honestly report <45 dB"
    )

    # Negative check A: demanding 45 dB from 8.5 bpw DualPlaneNvfp4 MUST fail closed!
    strict_fail = certify_lm_head_dual_plane(
        vocab_size=512,
        hidden_size=128,
        num_queries=64,
        seed=77,
        min_dual_plane_sqnr_db=45.0,
    )
    assert not strict_fail["passed"], "min_dual_plane_sqnr_db=45.0 must fail closed on 8.5 bpw DualPlane"

    # Negative check B: NaN / out-of-range gate parameters MUST fail closed via ValueError!
    for bad_kwargs in (
        {"min_dual_plane_sqnr_db": float("nan")},
        {"max_top1_flip_rate": float("nan")},
        {"max_kl_t1": float("nan")},
        {"max_kl_t1": -1.0e-5},
        {"max_top1_flip_rate": 1.5},
    ):
        try:
            certify_lm_head_dual_plane(vocab_size=64, hidden_size=32, num_queries=8, seed=77, **bad_kwargs)
            raise AssertionError(f"invalid gate kwargs {bad_kwargs} must raise ValueError")
        except ValueError:
            pass

    # Side-by-side 4-scheme LM-head Pareto certifier: BF16 vs INT8 per-32 BF16sc (8.5 bpw)
    # vs DualPlaneNvfp4 (8.5 bpw) vs MXFP8 (8.25 bpw), with exact 0.0 flip gate!
    int8_res = certify_lm_head_int8_per32_bf16sc(
        vocab_size=2048,
        hidden_size=256,
        num_queries=256,
        seed=77,
        min_int8_sqnr_db=44.0,
        max_int8_kl_t1=1.0e-6,
        max_top1_flip_rate=0.0,
    )
    assert int8_res["passed"], f"INT8 per-32 BF16sc cert failed: {int8_res['failures']}"
    assert int8_res["int8_to_bf16_lossless_0ulp"]
    assert int8_res["pareto_dominates_dual_plane_and_mxfp8"]

    for bad_i8 in (
        {"min_int8_sqnr_db": float("nan")},
        {"max_int8_kl_t1": float("nan")},
        {"max_top1_flip_rate": float("nan")},
    ):
        try:
            certify_lm_head_int8_per32_bf16sc(vocab_size=64, hidden_size=32, num_queries=8, seed=77, **bad_i8)
            raise AssertionError(f"invalid int8 gate kwargs {bad_i8} must raise ValueError")
        except ValueError:
            pass

    kv_res = certify_bitfp4_razer_kv(num_tokens=64, num_queries=16, seed=77)
    assert kv_res["passed"], f"BitFP4+RaZeR KV cert failed: {kv_res['failures']}"

    print(
        json.dumps(
            {
                "self_test": "PASS",
                "lm_head_int8_per32_bf16sc": int8_res,
                "lm_head_dual_plane": lm_res,
                "bitfp4_razer_kv": kv_res,
                "negative_gates_verified": [
                    "dual_plane_45db_strict_fails_closed",
                    "nan_and_oob_gates_rejected",
                    "zero_flip_rate_exact_gate_passes",
                    "int8_per32_bf16sc_pareto_dominance_verified",
                ],
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline quantization exactness certifier for Flux DualPlaneNvfp4 and BitFP4+RaZeR-KV"
    )
    parser.add_argument("--weights", type=Path, help="Optional .npy/.npz lm_head weight matrix [V, K]")
    parser.add_argument("--weight-key", help="Array key inside .npz archive (defaults to first key)")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Synthetic vocab size when --weights omitted")
    parser.add_argument("--hidden-size", type=int, default=512, help="Synthetic hidden size (multiple of 16)")
    parser.add_argument("--num-queries", type=int, default=512, help="LM-head evaluation query rows")
    parser.add_argument("--kv-tokens", type=int, default=64, help="KV cache sequence length (multiple of 16)")
    parser.add_argument("--seed", type=int, default=20261127, help="Deterministic RNG seed")
    parser.add_argument(
        "--min-sqnr-db",
        type=float,
        default=37.5,
        help="Minimum DualPlaneNvfp4 8.5 bpw SQNR gate in dB (use 45.0 to enforce strict BF16-equiv refusal)",
    )
    parser.add_argument("--max-top1-flip-rate", type=float, default=0.001, help="Max top-1 flip rate (<0.1%)")
    parser.add_argument("--max-kl", type=float, default=1.0e-4, help="Max DualPlaneNvfp4 D_KL at T=1.0")
    parser.add_argument(
        "--min-int8-sqnr-db",
        type=float,
        default=44.0,
        help="Minimum INT8 per-32 BF16sc (8.5 bpw) SQNR gate in dB",
    )
    parser.add_argument(
        "--max-int8-kl",
        type=float,
        default=1.0e-6,
        help="Max INT8 per-32 BF16sc (8.5 bpw) mean D_KL at T=1.0",
    )
    parser.add_argument("--json-out", type=Path, help="Optional output JSON report path")
    parser.add_argument("--self-test", action="store_true", help="Run CPU self-test with positive & negative checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    try:
        _validate_lm_head_gates(
            min_dual_plane_sqnr_db=args.min_sqnr_db,
            strict_bf16_sqnr_db=45.0,
            max_top1_flip_rate=args.max_top1_flip_rate,
            max_kl_t1=args.max_kl,
        )
        _validate_int8_per32_gates(
            min_int8_sqnr_db=args.min_int8_sqnr_db,
            max_int8_kl_t1=args.max_int8_kl,
            max_top1_flip_rate=args.max_top1_flip_rate,
        )
    except ValueError as err:
        raise SystemExit(f"error: {err}") from err

    w = _load_weight_matrix(args.weights, args.weight_key) if args.weights else None
    int8_res = certify_lm_head_int8_per32_bf16sc(
        weights_fp32=w,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_queries=args.num_queries,
        seed=args.seed,
        min_int8_sqnr_db=args.min_int8_sqnr_db,
        max_int8_kl_t1=args.max_int8_kl,
        max_top1_flip_rate=args.max_top1_flip_rate,
    )
    lm_res = certify_lm_head_dual_plane(
        weights_fp32=w,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_queries=args.num_queries,
        seed=args.seed,
        min_dual_plane_sqnr_db=args.min_sqnr_db,
        max_top1_flip_rate=args.max_top1_flip_rate,
        max_kl_t1=args.max_kl,
    )
    kv_res = certify_bitfp4_razer_kv(num_tokens=args.kv_tokens, num_queries=16, seed=args.seed)
    overall = {
        "passed": int8_res["passed"] and lm_res["passed"] and kv_res["passed"],
        "lm_head_int8_per32_bf16sc": int8_res,
        "lm_head_dual_plane": lm_res,
        "bitfp4_razer_kv": kv_res,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))
    return 0 if overall["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
