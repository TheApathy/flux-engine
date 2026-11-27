#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Zero-calibration AutoEMatic (+0.4 bpw) AutoFP6 sidecar builder.

Reads upstream BF16 or block-128 FP8 E4M3 FFN weights from safetensors shards
and emits an authenticated `.safetensors` sidecar holding sparse 2-bit signed
mantissa offsets (`delta = d * 2^-3`, `d in {-2, -1, 0, +1}`) on the top-rho
4-element subgroups ranked by exact NMSE reduction (default rho=0.20, max 2
refined subgroups per 16-element NVFP4 group).

Output tensors per requested layer prefix `<lp>` and projection
`<proj> in {gate_proj, up_proj, down_proj}`:
  * `<lp>.mlp.<proj>.ae_refine_codes`  U8  `[N, num_refined_sg]`
  * `<lp>.mlp.<proj>.ae_refine_mask`   U8  `[K/16]` (low 4 bits = SG bitmask)
  * `<lp>.mlp.<proj>.ae_group_prefix`  U32 `[K/16]` (exclusive popcount prefix)
  * `<lp>.mlp.<proj>.ae_meta`          F32 `[2]`    (`[rho_actual, 0.25]`)

Requires only Python stdlib + NumPy (no GPU, no calibration dataset required).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

GROUP_SIZE = 16
SUBGROUP_SIZE = 4
SUBGROUPS_PER_GROUP = GROUP_SIZE // SUBGROUP_SIZE
MAX_REFINED_PER_GROUP = 2
REFINEMENT_SCALE_FACTOR = 0.25

E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_LUT = np.concatenate([E2M1_POS, -E2M1_POS])

# Two's-complement 2-bit signed offset codes:
#   0 (00) ->  0.0,  1 (01) -> +1/8,  2 (10) -> -2/8 (-1/4),  3 (11) -> -1/8
OFFSET_VALUES = np.array([0.0, 0.125, -0.25, -0.125], dtype=np.float32)


def decode_bf16_bytes(raw: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def build_e4m3_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = (b >> 7) & 1
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 15 and man == 7:
            val = 0.0
        elif exp == 0:
            val = float(man) * (2.0**-9)
        else:
            val = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        lut[b] = -val if sign else val
    return lut


E4M3_LUT = build_e4m3_lut()


def float_to_ue4m3_array(values: np.ndarray) -> np.ndarray:
    """Vectorized positive FP32 -> UE4M3 matching dual_plane_nvfp4.cu / quantize_bf16_to_nvfp4.cu."""
    clamped = np.clip(np.abs(values.astype(np.float32)), 0.0, 448.0)
    bits = clamped.view(np.uint32)
    f32_exp = ((bits >> 23) & 0xFF).astype(np.int32) - 127
    f32_man = bits & 0x7FFFFF

    out = np.zeros(clamped.shape, dtype=np.uint8)
    sub = (clamped > 0.0) & (f32_exp >= -9) & (f32_exp < -6)
    if np.any(sub):
        man = np.clip(np.floor(clamped[sub] * 512.0 + 0.5).astype(np.int32), 0, 7)
        out[sub] = man.astype(np.uint8)

    norm = (clamped > 0.0) & (f32_exp >= -6)
    if np.any(norm):
        e = f32_exp[norm] + 7
        m = ((f32_man[norm] + (1 << 19)) >> 20).astype(np.int32)
        carry = m > 7
        m = np.where(carry, 0, m)
        e = np.where(carry, e + 1, e)
        overflow = (e > 15) | ((e == 15) & (m > 6))
        e = np.where(overflow, 15, np.maximum(e, 1))
        m = np.where(overflow, 6, m)
        out[norm] = ((e << 3) | m).astype(np.uint8)
    return out


def quantize_e2m1_array(x_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Branchless round-to-nearest E2M1 matching e2m1_branchless.cu."""
    x = x_norm.astype(np.float32)
    sign = (x < 0.0).astype(np.uint8) << 3
    ax = np.abs(x)
    mag = (
        (ax > 0.25).astype(np.uint8)
        + (ax >= 0.75).astype(np.uint8)
        + (ax > 1.25).astype(np.uint8)
        + (ax >= 1.75).astype(np.uint8)
        + (ax > 2.5).astype(np.uint8)
        + (ax >= 3.5).astype(np.uint8)
        + (ax > 5.0).astype(np.uint8)
    )
    codes = sign | mag
    return codes, E2M1_LUT[codes]


def build_projection_sidecar(
    weight_f32: np.ndarray,
    rho: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Build AutoEMatic AutoFP6 sparse refinement tensors for one `[N, K]` matrix."""
    if weight_f32.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {weight_f32.shape}")
    n, k = weight_f32.shape
    if n == 0 or k == 0 or (k % GROUP_SIZE) != 0:
        raise ValueError(f"Invalid projection shape [{n}, {k}]; K must be positive multiple of 16")
    if not (0.0 < rho <= 0.5):
        raise ValueError(f"Refinement budget rho must lie in (0, 0.5], got {rho}")

    w = weight_f32.astype(np.float32, copy=False)
    num_groups = k // GROUP_SIZE
    global_max = float(np.max(np.abs(w)))
    scale2 = np.float32(global_max / (6.0 * 448.0) if global_max > 0.0 else 1.0)

    w_g = w.reshape(n, num_groups, GROUP_SIZE)
    gmax = np.max(np.abs(w_g), axis=2)
    scale_bytes = float_to_ue4m3_array(gmax / (6.0 * scale2))
    eff_scale = E4M3_LUT[scale_bytes] * scale2
    inv_eff = np.where(eff_scale > 0.0, 1.0 / np.maximum(eff_scale, 1e-30), 0.0).astype(np.float32)

    w_norm = w_g * inv_eff[:, :, None]
    _, q_vals = quantize_e2m1_array(w_norm)
    residual = w_norm - q_vals

    # Assign nearest 2-bit signed offset d * 2^-3 in {0, +0.125, -0.25, -0.125}
    diffs = np.abs(residual[:, :, :, None] - OFFSET_VALUES[None, None, None, :])
    best_codes = np.argmin(diffs, axis=3).astype(np.uint8)
    best_deltas = OFFSET_VALUES[best_codes]

    base_sq_err = ((w_norm - q_vals) * eff_scale[:, :, None]) ** 2
    ref_sq_err = ((w_norm - (q_vals + best_deltas)) * eff_scale[:, :, None]) ** 2

    # Subgroup error reductions summed across all output rows N -> shape [num_groups, 4]
    sg_gain = (base_sq_err - ref_sq_err).reshape(n, num_groups, SUBGROUPS_PER_GROUP, SUBGROUP_SIZE).sum(axis=(0, 3))

    target_refined_sg = max(1, min(num_groups * MAX_REFINED_PER_GROUP, int(round(rho * num_groups * SUBGROUPS_PER_GROUP))))
    # Rank (group, subgroup) pairs while enforcing at most MAX_REFINED_PER_GROUP=2 per group
    order = np.argsort(sg_gain, axis=1)[:, ::-1]  # descending within each group
    candidates: List[Tuple[float, int, int]] = []
    for g in range(num_groups):
        for rank in range(MAX_REFINED_PER_GROUP):
            sg = int(order[g, rank])
            candidates.append((float(sg_gain[g, sg]), g, sg))
    candidates.sort(key=lambda item: item[0], reverse=True)

    mask = np.zeros(num_groups, dtype=np.uint8)
    for _, g, sg in candidates[:target_refined_sg]:
        mask[g] |= np.uint8(1 << sg)

    counts = np.array([int(mask[g]).bit_count() for g in range(num_groups)], dtype=np.uint32)
    group_prefix = np.zeros(num_groups, dtype=np.uint32)
    if num_groups > 1:
        group_prefix[1:] = np.cumsum(counts[:-1], dtype=np.uint32)
    num_refined_sg = int(counts.sum())

    # Pack refined 4-element subgroups into 1 byte each: [N, num_refined_sg]
    codes_sg = best_codes.reshape(n, num_groups, SUBGROUPS_PER_GROUP, SUBGROUP_SIZE)
    refine_codes = np.zeros((n, num_refined_sg), dtype=np.uint8)
    out_col = 0
    for g in range(num_groups):
        m = int(mask[g])
        for sg in range(SUBGROUPS_PER_GROUP):
            if (m & (1 << sg)) != 0:
                c0 = codes_sg[:, g, sg, 0]
                c1 = codes_sg[:, g, sg, 1]
                c2 = codes_sg[:, g, sg, 2]
                c3 = codes_sg[:, g, sg, 3]
                refine_codes[:, out_col] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
                out_col += 1

    rho_actual = num_refined_sg / float(num_groups * SUBGROUPS_PER_GROUP)
    meta = np.array([rho_actual, REFINEMENT_SCALE_FACTOR], dtype=np.float32)

    total_sig = float(np.sum(w.astype(np.float64) ** 2))
    base_mse = float(np.sum(base_sq_err, dtype=np.float64))
    selected_gain = float(sum(sg_gain[g, sg] for g in range(num_groups) for sg in range(4) if (int(mask[g]) & (1 << sg))))
    ref_mse = max(base_mse - selected_gain, 1e-30)
    stats = {
        "base_nmse": base_mse / max(total_sig, 1e-30),
        "refined_nmse": ref_mse / max(total_sig, 1e-30),
        "rho_actual": rho_actual,
        "added_bpw": 2.0 * rho_actual,
        "num_refined_sg": float(num_refined_sg),
    }
    return refine_codes, mask, group_prefix, meta, stats


def write_safetensors(path: Path, tensors: Dict[str, np.ndarray]) -> Tuple[int, str]:
    """Write a canonical safetensors file (sorted keys, contiguous non-overlapping offsets)."""
    dtype_map = {
        np.dtype(np.uint8): "U8",
        np.dtype(np.uint32): "U32",
        np.dtype(np.float32): "F32",
    }
    header: Dict[str, dict] = {}
    payloads: List[bytes] = []
    offset = 0
    for name in sorted(tensors.keys()):
        arr = np.ascontiguousarray(tensors[name])
        if arr.dtype not in dtype_map:
            raise ValueError(f"Unsupported sidecar dtype {arr.dtype} for {name}")
        raw = arr.tobytes()
        end = offset + len(raw)
        header[name] = {
            "dtype": dtype_map[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, end],
        }
        payloads.append(raw)
        offset = end

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    prefix = struct.pack("<Q", len(header_bytes))

    hasher = hashlib.sha256()
    total_size = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for chunk in (prefix, header_bytes, *payloads):
            fh.write(chunk)
            hasher.update(chunk)
            total_size += len(chunk)
    return total_size, hasher.hexdigest()


def parse_layers_spec(spec: str) -> List[int]:
    layers: List[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            layers.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            layers.append(int(item))
    return sorted(set(layers))


def read_safetensors_header(shard_path: Path) -> Tuple[int, Dict[str, dict]]:
    with shard_path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
    return 8 + header_len, header


def load_weight_from_checkpoint(checkpoint_dir: Path, tensor_prefix: str) -> np.ndarray:
    """Load a BF16 or block-128 FP8 E4M3 `[N, K]` tensor from safetensors shards."""
    weight_key = f"{tensor_prefix}.weight"
    scale_key = f"{tensor_prefix}.weight_scale_inv"
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shard_path = checkpoint_dir / weight_map[weight_key]
        scale_shard = checkpoint_dir / weight_map[scale_key] if scale_key in weight_map else None
    else:
        shards = sorted(checkpoint_dir.glob("*.safetensors"))
        shard_path = None
        scale_shard = None
        for candidate in shards:
            _, hdr = read_safetensors_header(candidate)
            if weight_key in hdr:
                shard_path = candidate
            if scale_key in hdr:
                scale_shard = candidate
        if shard_path is None:
            raise FileNotFoundError(f"Tensor {weight_key} not found under {checkpoint_dir}")

    data_start, hdr = read_safetensors_header(shard_path)
    meta = hdr[weight_key]
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    off0, off1 = meta["data_offsets"]
    with shard_path.open("rb") as fh:
        fh.seek(data_start + off0)
        raw = fh.read(off1 - off0)

    if dtype == "BF16":
        return decode_bf16_bytes(raw, shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dtype == "F8_E4M3":
        if scale_shard is None:
            raise ValueError(f"FP8 tensor {weight_key} missing companion {scale_key}")
        s_start, s_hdr = read_safetensors_header(scale_shard)
        s_meta = s_hdr[scale_key]
        s_shape = tuple(s_meta["shape"])
        so0, so1 = s_meta["data_offsets"]
        with scale_shard.open("rb") as fh:
            fh.seek(s_start + so0)
            s_raw = fh.read(so1 - so0)
        scales = (
            decode_bf16_bytes(s_raw, s_shape)
            if s_meta["dtype"] == "BF16"
            else np.frombuffer(s_raw, dtype=np.float32).reshape(s_shape)
        )
        fp8_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        w = E4M3_LUT[fp8_u8]
        n, k = shape
        n_idx = np.arange(n) // 128
        k_idx = np.arange(k) // 128
        return w * scales[n_idx[:, None], k_idx[None, :]]
    raise ValueError(f"Unsupported source dtype {dtype} for {weight_key}")


def format_layers_spec(layers: List[int]) -> str:
    """Format sorted unique layer indices into canonical comma/range spec (e.g. '1,3-5')."""
    if not layers:
        return ""
    ordered = sorted(set(layers))
    ranges: List[str] = []
    start = ordered[0]
    prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = idx
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def discover_checkpoint_ffn_layers(
    checkpoint_dir: Path,
    layer_prefix: Optional[str] = None,
) -> Tuple[str, List[int]]:
    """Discover complete FFN layers ({gate,up,down}_proj) in a HuggingFace checkpoint."""
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        all_keys = list(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].keys())
    else:
        all_keys = []
        for shard in sorted(checkpoint_dir.glob("*.safetensors")):
            _, hdr = read_safetensors_header(shard)
            all_keys.extend(k for k in hdr.keys() if k != "__metadata__")

    pattern = re.compile(r"^(.+)\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
    by_prefix: Dict[str, Dict[int, set]] = {}
    for key in all_keys:
        match = pattern.match(key)
        if not match:
            continue
        pref, idx_s, proj = match.group(1), match.group(2), match.group(3)
        if layer_prefix is not None and pref != layer_prefix:
            continue
        by_prefix.setdefault(pref, {}).setdefault(int(idx_s), set()).add(proj)

    if not by_prefix:
        raise FileNotFoundError(
            f"No FFN projections (*.*.mlp.{{gate,up,down}}_proj.weight) found under {checkpoint_dir}"
        )

    best_prefix = max(
        sorted(by_prefix.keys()),
        key=lambda p: sum(1 for projs in by_prefix[p].values() if len(projs) == 3),
    )
    complete_layers = sorted(
        idx for idx, projs in by_prefix[best_prefix].items() if len(projs) == 3
    )
    if not complete_layers:
        raise ValueError(f"No layer under {checkpoint_dir} has all 3 FFN projections")
    return best_prefix, complete_layers


def score_weight_sensitivity(
    weight_f32: np.ndarray,
    rho: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute kurtosis, block-16 outlier ratio, NVFP4 NMSE, and AutoFP6 sidecar tensors."""
    w = weight_f32.astype(np.float32, copy=False)
    flat = w.reshape(-1).astype(np.float64)
    mean = float(np.mean(flat))
    centered = flat - mean
    var = float(np.mean(centered * centered))
    if var > 1e-24:
        fourth = float(np.mean((centered * centered) ** 2))
        excess_kurtosis = max(0.0, fourth / (var * var) - 3.0)
    else:
        excess_kurtosis = 0.0

    n, k = w.shape
    blocks = w.reshape(n, k // GROUP_SIZE, GROUP_SIZE).astype(np.float64)
    bmax = np.max(np.abs(blocks), axis=2)
    brms = np.sqrt(np.mean(blocks * blocks, axis=2))
    outlier_ratios = np.where(brms > 1e-12, bmax / np.maximum(brms, 1e-12), 1.0)
    block16_outlier_mean = float(np.mean(outlier_ratios))
    block16_outlier_p95 = float(np.percentile(outlier_ratios, 95))

    codes, mask, prefix, meta, stats = build_projection_sidecar(w, rho=rho)
    nmse_gain = max(0.0, stats["base_nmse"] - stats["refined_nmse"])
    sensitivity_score = (
        100.0
        * stats["base_nmse"]
        * (1.0 + 0.25 * math.log1p(excess_kurtosis))
        * (block16_outlier_p95 / 2.0)
    )
    metrics = {
        "excess_kurtosis": excess_kurtosis,
        "block16_outlier_mean": block16_outlier_mean,
        "block16_outlier_p95": block16_outlier_p95,
        "base_nmse": stats["base_nmse"],
        "refined_nmse": stats["refined_nmse"],
        "nmse_gain": nmse_gain,
        "added_bpw": stats["added_bpw"],
        "sensitivity_score": sensitivity_score,
    }
    return codes, mask, prefix, meta, metrics


def scan_checkpoint(
    checkpoint_dir: Path,
    rho: float = 0.20,
    top_k_layers: Optional[int] = None,
    budget_fraction: float = 0.25,
    layer_prefix: Optional[str] = None,
    output_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Scan a checkpoint's FFN layers, rank sensitivity, and optionally emit sidecar+manifest."""
    if not (0.0 < budget_fraction <= 1.0):
        raise ValueError(f"budget_fraction must lie in (0, 1], got {budget_fraction}")

    detected_prefix, discovered_layers = discover_checkpoint_ffn_layers(
        checkpoint_dir, layer_prefix=layer_prefix
    )
    if top_k_layers is None:
        k_select = max(1, min(len(discovered_layers), int(round(len(discovered_layers) * budget_fraction))))
    else:
        if top_k_layers < 1 or top_k_layers > len(discovered_layers):
            raise ValueError(
                f"top_k_layers={top_k_layers} must lie in 1..={len(discovered_layers)}"
            )
        k_select = top_k_layers

    layer_scores: List[Dict[str, object]] = []
    cached_tensors: Dict[int, Dict[str, np.ndarray]] = {}

    for layer in discovered_layers:
        lp = f"{detected_prefix}.{layer}"
        proj_metrics: Dict[str, Dict[str, float]] = {}
        layer_tensors: Dict[str, np.ndarray] = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            full_prefix = f"{lp}.mlp.{proj}"
            weight = load_weight_from_checkpoint(checkpoint_dir, full_prefix)
            codes, mask, prefix, meta, metrics = score_weight_sensitivity(weight, rho=rho)
            proj_metrics[proj] = metrics
            if output_path is not None:
                layer_tensors[f"{full_prefix}.ae_refine_codes"] = codes
                layer_tensors[f"{full_prefix}.ae_refine_mask"] = mask
                layer_tensors[f"{full_prefix}.ae_group_prefix"] = prefix
                layer_tensors[f"{full_prefix}.ae_meta"] = meta
        if output_path is not None:
            cached_tensors[layer] = layer_tensors

        mean_kurtosis = float(np.mean([m["excess_kurtosis"] for m in proj_metrics.values()]))
        mean_outlier_p95 = float(np.mean([m["block16_outlier_p95"] for m in proj_metrics.values()]))
        mean_base_nmse = float(np.mean([m["base_nmse"] for m in proj_metrics.values()]))
        mean_ref_nmse = float(np.mean([m["refined_nmse"] for m in proj_metrics.values()]))
        layer_sensitivity = float(np.mean([m["sensitivity_score"] for m in proj_metrics.values()]))
        layer_scores.append(
            {
                "layer": layer,
                "sensitivity_score": round(layer_sensitivity, 6),
                "excess_kurtosis": round(mean_kurtosis, 4),
                "block16_outlier_p95": round(mean_outlier_p95, 4),
                "base_nmse": round(mean_base_nmse, 6),
                "refined_nmse": round(mean_ref_nmse, 6),
            }
        )

    ranked = sorted(
        layer_scores,
        key=lambda row: (float(row["sensitivity_score"]), -int(row["layer"])),
        reverse=True,
    )
    recommended_layers = sorted(int(row["layer"]) for row in ranked[:k_select])
    layers_spec = format_layers_spec(recommended_layers)

    report: Dict[str, object] = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "layer_prefix": detected_prefix,
        "discovered_layers": discovered_layers,
        "recommended_layers": recommended_layers,
        "recommended_layers_spec": layers_spec,
        "rho": rho,
        "ranked_layers": ranked,
    }

    if output_path is not None:
        selected_tensors: Dict[str, np.ndarray] = {}
        for layer in recommended_layers:
            selected_tensors.update(cached_tensors[layer])
        size_bytes, digest = write_safetensors(output_path, selected_tensors)
        resolved_out = str(output_path.resolve())
        report["sidecar"] = {
            "path": resolved_out,
            "size_bytes": size_bytes,
            "sha256": digest,
            "env": {
                "ATLAS_AUTOEMATIC_LAYERS": layers_spec,
                "ATLAS_AUTOEMATIC_SIDECAR": resolved_out,
                "ATLAS_AUTOEMATIC_SIDECAR_SHA256": digest,
                "ATLAS_AUTOEMATIC_SIDECAR_SIZE": str(size_bytes),
            },
        }
        mpath = manifest_path if manifest_path is not None else output_path.with_suffix(".manifest.json")
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["manifest_path"] = str(mpath.resolve())

    return report


def _encode_bf16_array(arr_f32: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr_f32, dtype=np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def _write_test_safetensors_raw(path: Path, entries: Dict[str, Tuple[str, Tuple[int, ...], bytes]]) -> None:
    header: Dict[str, dict] = {}
    payloads: List[bytes] = []
    offset = 0
    for name in sorted(entries.keys()):
        dtype, shape, raw = entries[name]
        end = offset + len(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, end]}
        payloads.append(raw)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for chunk in payloads:
            fh.write(chunk)


def run_self_test() -> int:
    rng = np.random.default_rng(2026)
    # 1. Isotropic Gaussian: refining 20% subgroups reduces NMSE by ~14-18%.
    w_iso = rng.standard_normal((64, 256)).astype(np.float32)
    codes, mask, prefix, meta, stats_iso = build_projection_sidecar(w_iso, rho=0.20)
    assert mask.shape == (16,)
    assert prefix.shape == (16,)
    assert codes.shape[0] == 64
    assert 0.15 <= stats_iso["rho_actual"] <= 0.25, f"unexpected rho: {stats_iso}"
    assert stats_iso["refined_nmse"] < stats_iso["base_nmse"] * 0.90, (
        f"insufficient isotropic NMSE gain: {stats_iso}"
    )

    # 2. Outlier-channel LLM-like matrix (8 groups carry 3x outlier subgroup):
    #    ranking by exact NMSE gain achieves roughly 2x the isotropic NMSE drop
    #    under the same rho=0.20 / max-2-per-group budget.
    w_het = rng.standard_normal((64, 256)).astype(np.float32)
    for g in range(0, 16, 2):
        w_het[:, g * 16 : g * 16 + 4] *= 3.0
    _, _, _, _, stats_het = build_projection_sidecar(w_het, rho=0.20)
    het_ratio = stats_het["refined_nmse"] / stats_het["base_nmse"]
    iso_ratio = stats_iso["refined_nmse"] / stats_iso["base_nmse"]
    assert het_ratio < 0.75 and het_ratio < iso_ratio - 0.10, (
        f"insufficient heterogeneous NMSE gain: het={het_ratio:.4f}, iso={iso_ratio:.4f}, {stats_het}"
    )

    # 3. Verify safetensors serialization round-trip + SHA-256 determinism.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ae-sidecar-test-") as tmp:
        tmp_path = Path(tmp)
        out_path = tmp_path / "test.safetensors"
        size_b, digest = write_safetensors(
            out_path,
            {
                "model.layers.0.mlp.gate_proj.ae_refine_codes": codes,
                "model.layers.0.mlp.gate_proj.ae_refine_mask": mask,
                "model.layers.0.mlp.gate_proj.ae_group_prefix": prefix,
                "model.layers.0.mlp.gate_proj.ae_meta": meta,
            },
        )
        assert size_b == out_path.stat().st_size
        assert digest == hashlib.sha256(out_path.read_bytes()).hexdigest()

        # 4. Verify --scan-checkpoint end-to-end on a 4-layer sharded BF16+FP8 fixture:
        #    layers 1 and 3 carry heavy-tailed block-16 outlier spikes; layers 0 and 2 are smooth.
        ckpt_dir = tmp_path / "synthetic_ckpt"
        ckpt_dir.mkdir()
        shard0: Dict[str, Tuple[str, Tuple[int, ...], bytes]] = {}
        shard1: Dict[str, Tuple[str, Tuple[int, ...], bytes]] = {}
        weight_map: Dict[str, str] = {}
        shapes = {"gate_proj": (64, 128), "up_proj": (64, 128), "down_proj": (128, 64)}

        for layer in range(4):
            is_outlier = layer in (1, 3)
            for proj, (n_dim, k_dim) in shapes.items():
                mat = rng.standard_normal((n_dim, k_dim)).astype(np.float32) * 0.5
                if is_outlier:
                    # Inject heavy-tailed single-element block-16 spikes (10x magnitude)
                    mat[:, 0::16] *= 10.0
                key = f"model.layers.{layer}.mlp.{proj}.weight"
                if layer < 2:
                    # Shard 0: BF16
                    shard0[key] = ("BF16", (n_dim, k_dim), _encode_bf16_array(mat).tobytes())
                    weight_map[key] = "model-00001-of-00002.safetensors"
                else:
                    # Shard 1: block-128 FP8 E4M3 + BF16 weight_scale_inv
                    amax = float(np.max(np.abs(mat)))
                    bscale = np.array([[amax / 448.0 if amax > 0.0 else 1.0]], dtype=np.float32)
                    scaled = mat / bscale[0, 0]
                    fp8_mag = float_to_ue4m3_array(scaled)
                    fp8_bytes = np.where((scaled < 0.0) & (fp8_mag != 0), fp8_mag | 0x80, fp8_mag).astype(np.uint8)
                    scale_key = f"model.layers.{layer}.mlp.{proj}.weight_scale_inv"
                    shard1[key] = ("F8_E4M3", (n_dim, k_dim), fp8_bytes.tobytes())
                    shard1[scale_key] = ("BF16", (1, 1), _encode_bf16_array(bscale).tobytes())
                    weight_map[key] = "model-00002-of-00002.safetensors"
                    weight_map[scale_key] = "model-00002-of-00002.safetensors"

        _write_test_safetensors_raw(ckpt_dir / "model-00001-of-00002.safetensors", shard0)
        _write_test_safetensors_raw(ckpt_dir / "model-00002-of-00002.safetensors", shard1)
        (ckpt_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
        )

        scan_out = tmp_path / "scanned_sidecar.safetensors"
        scan_manifest = tmp_path / "scanned_manifest.json"
        scan_report = scan_checkpoint(
            ckpt_dir,
            rho=0.20,
            top_k_layers=2,
            output_path=scan_out,
            manifest_path=scan_manifest,
        )
        assert scan_report["discovered_layers"] == [0, 1, 2, 3], scan_report
        assert scan_report["recommended_layers"] == [1, 3], (
            f"expected outlier layers [1, 3], got {scan_report['recommended_layers']}"
        )
        assert scan_report["recommended_layers_spec"] == "1,3"
        sidecar_info = scan_report["sidecar"]
        assert isinstance(sidecar_info, dict)
        assert scan_out.stat().st_size == sidecar_info["size_bytes"]
        assert hashlib.sha256(scan_out.read_bytes()).hexdigest() == sidecar_info["sha256"]
        assert scan_manifest.is_file()
        # Verify emitted sidecar header contains exact 4 tensors x 3 projs x 2 layers = 24 tensors.
        _, emitted_hdr = read_safetensors_header(scan_out)
        assert len(emitted_hdr) == 24, f"expected 24 tensors, got {len(emitted_hdr)}"

    print(
        json.dumps(
            {
                "status": "PASS",
                "iso_base_nmse": round(stats_iso["base_nmse"], 6),
                "iso_refined_nmse": round(stats_iso["refined_nmse"], 6),
                "het_base_nmse": round(stats_het["base_nmse"], 6),
                "het_refined_nmse": round(stats_het["refined_nmse"], 6),
                "added_bpw": round(stats_iso["added_bpw"], 4),
                "num_refined_sg": int(stats_iso["num_refined_sg"]),
                "scan_recommended_layers": scan_report["recommended_layers"],
                "scan_emitted_tensors": len(emitted_hdr),
            }
        )
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build zero-calibration AutoEMatic (+0.4 bpw) AutoFP6 sidecar.")
    parser.add_argument("--checkpoint", type=Path, help="Upstream BF16/FP8 checkpoint directory")
    parser.add_argument(
        "--scan-checkpoint",
        type=Path,
        help="Scan HF safetensors checkpoint, rank per-layer quantization sensitivity, recommend top-K layers, and optionally emit authenticated sidecar + manifest",
    )
    parser.add_argument("--output", type=Path, help="Destination .safetensors sidecar path")
    parser.add_argument("--manifest", type=Path, help="Destination JSON manifest path for --scan-checkpoint")
    parser.add_argument("--layers", type=str, default="0", help="Layer indices/ranges (e.g. '0-63' or '4,8,12')")
    parser.add_argument("--layer-prefix", type=str, default=None, help="HF layer prefix before .<idx> (auto-detected under --scan-checkpoint; defaults to 'model.layers' otherwise)")
    parser.add_argument("--top-k-layers", type=int, default=None, help="Number of top-sensitivity layers to select under --scan-checkpoint")
    parser.add_argument("--budget-fraction", type=float, default=0.25, help="Fraction of layers to select when --top-k-layers is unset (default 0.25)")
    parser.add_argument("--rho", type=float, default=0.20, help="Subgroup refinement budget (default 0.20 -> +0.4 bpw)")
    parser.add_argument("--self-test", action="store_true", help="Run built-in numerical and --scan-checkpoint verification and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        return run_self_test()
    if args.scan_checkpoint is not None:
        report = scan_checkpoint(
            args.scan_checkpoint,
            rho=args.rho,
            top_k_layers=args.top_k_layers,
            budget_fraction=args.budget_fraction,
            layer_prefix=args.layer_prefix,
            output_path=args.output,
            manifest_path=args.manifest,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required unless --self-test or --scan-checkpoint is given")

    layer_prefix = args.layer_prefix if args.layer_prefix is not None else "model.layers"
    layers = parse_layers_spec(args.layers)
    tensors: Dict[str, np.ndarray] = {}
    nmse_ratios: List[float] = []
    bpw_list: List[float] = []

    for layer in layers:
        lp = f"{layer_prefix}.{layer}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            full_prefix = f"{lp}.mlp.{proj}"
            weight = load_weight_from_checkpoint(args.checkpoint, full_prefix)
            codes, mask, prefix, meta, stats = build_projection_sidecar(weight, rho=args.rho)
            tensors[f"{full_prefix}.ae_refine_codes"] = codes
            tensors[f"{full_prefix}.ae_refine_mask"] = mask
            tensors[f"{full_prefix}.ae_group_prefix"] = prefix
            tensors[f"{full_prefix}.ae_meta"] = meta
            nmse_ratios.append(stats["refined_nmse"] / max(stats["base_nmse"], 1e-30))
            bpw_list.append(stats["added_bpw"])

    size_bytes, digest = write_safetensors(args.output, tensors)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "layers": args.layers,
                "size_bytes": size_bytes,
                "sha256": digest,
                "mean_added_bpw": round(float(np.mean(bpw_list)), 4),
                "mean_nmse_ratio": round(float(np.mean(nmse_ratios)), 4),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
