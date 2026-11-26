#!/usr/bin/env python3
"""Stream or materialize Qwen3.8-Flash-Next's 51B-parameter PLE table to NVFP4.

Automatically detects and supports both PLE checkpoint formats:
  1. Mode A — Native NVFP4 Passthrough (e.g. JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid):
     When the 128 PLE shards (`*.ngram_embedding.shard_{0..127}.weight`) are already
     `U8 [rows, 80]` accompanied by `.weight_scale` (`F8_E4M3 [rows, 10]`) and
     `.weight_scale_2` (`F32 [1]` or `[]`), copies the 80 packed E2M1 bytes + 10
     E4M3 scale bytes per row byte-for-byte into 8-KiB page-aligned (91 records/page)
     `ple-offload/ple-ngram-nvfp4-{shard:03d}.bin` sidecar files and writes
     `ple-offload/manifest.json`. By default (`--keep-source-shards`), source
     safetensors shards and `model.safetensors.index.json` are left untouched because
     `flux-server` automatically skips uploading `*.ngram_embedding.*` tensors when
     `ple-offload/manifest.json` is present.
  2. Mode B — BF16 / FP8 Quantize (e.g. Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2,
     nvidia/Qwen3.8-Flash-Next-NVFP4):
     When the 128 PLE shards are `BF16`, `F8_E4M3`, `F16`, or `F32` of shape
     `[rows, 160]`, quantizes each row to group-16 NVFP4 (`U8 [rows, 80]` +
     `F8_E4M3 [rows, 10]` + `F32 [1]` global `scale2`) and writes either 8-KiB
     page-aligned `ple-offload/` sidecars (`--mode offload`) or resident NVFP4
     safetensors (`--mode resident`).

Supports both PyTorch + safetensors (when installed on host) and a zero-PyTorch
bit-exact NumPy + native safetensors reader/writer (used inside the container).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    if os.environ.get("FLUX_REPACK_FORCE_NUMPY", "") == "1":
        raise ImportError("Forced NumPy backend via FLUX_REPACK_FORCE_NUMPY=1")
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    safe_open = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]
    HAS_TORCH = False


PLE_MARKER = ".ple.ple_embedding.ngram_embedding.shard_"
PLE_TABLE_MARKER = ".ple.ple_embedding.ngram_embedding."
GLOBAL_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
NATIVE_GLOBAL_SCALE2_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale_2"
FORMAT_TAG = "qwen38-flash-next-ple-nvfp4-v1"
OFFLOAD_FORMAT_TAG = "qwen38-flash-next-ple-nvfp4-direct-v1"
OFFLOAD_PAGE_BYTES = 8192
OFFLOAD_PACKED_BYTES = 80
OFFLOAD_SCALE_BYTES = 10
OFFLOAD_RECORD_BYTES = OFFLOAD_PACKED_BYTES + OFFLOAD_SCALE_BYTES  # 90
OFFLOAD_RECORDS_PER_PAGE = OFFLOAD_PAGE_BYTES // OFFLOAD_RECORD_BYTES  # 91
LOGICAL_PLE_WIDTH = 160

E2M1_BOUNDS_LIST = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
E2M1_VALUES_LIST = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]
E2M1_BOUNDS_NP = np.array(E2M1_BOUNDS_LIST, dtype=np.float32)
E2M1_VALUES_NP = np.array(E2M1_VALUES_LIST, dtype=np.float32)

if HAS_TORCH:
    E2M1_BOUNDS = torch.tensor(E2M1_BOUNDS_LIST)
    E2M1_VALUES = torch.tensor(E2M1_VALUES_LIST)


def _build_fp8_e4m3fn_lut() -> np.ndarray:
    """Build exact 256-entry IEEE/OFP8 E4M3FN uint8 -> float32 lookup table."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp == 0:
            val = (mant / 8.0) * (2.0 ** -6)
        elif exp == 15 and mant == 7:
            val = np.nan
        else:
            val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        lut[b] = sign * val
    return lut


FP8_E4M3_LUT = _build_fp8_e4m3fn_lut()
_FP8_POS_VALS = FP8_E4M3_LUT[:127]  # 0..126 monotonic positive finite values (0.0 .. 448.0)
_FP8_MIDS = 0.5 * (_FP8_POS_VALS[:-1] + _FP8_POS_VALS[1:])


def f32_to_f8e4m3fn_np(arr: np.ndarray) -> np.ndarray:
    """Bit-exact float32 -> float8_e4m3fn (uint8) matching torch.float8_e4m3fn round-to-even."""
    arr = np.asarray(arr, dtype=np.float32)
    sign = (np.signbit(arr).astype(np.uint8)) << 7
    mag = np.abs(arr)
    idx = np.searchsorted(_FP8_MIDS, mag, side="left").astype(np.uint8)
    clamped = np.minimum(idx, 125)
    is_tie = (idx < 126) & (mag == _FP8_MIDS[clamped])
    odd_low = (idx & 1) == 1
    idx = idx + (is_tie & odd_low).astype(np.uint8)
    idx = np.minimum(idx, 126)
    out = sign | idx
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        out[nan_mask] = 0x7F
    return out


def is_ple_weight(name: str) -> bool:
    return PLE_MARKER in name and name.endswith(".weight")


def is_ple_related_tensor(name: str) -> bool:
    return PLE_TABLE_MARKER in name


def parse_ple_shard_index(name: str) -> int:
    return int(name.rsplit(".shard_", 1)[1].removesuffix(".weight"))


# ─────────────────────────────────────────────────────────────────────────────
# Native Safetensors Reader/Writer (for zero-PyTorch container execution)
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"{path}: truncated safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    return header, 8 + header_len


def read_scalar_f32_from_fd(fd: int, data_base: int, info: dict[str, Any], name: str) -> float:
    dtype = info["dtype"]
    shape = info["shape"]
    if dtype != "F32" or shape not in ([1], []):
        raise TypeError(f"{name}: expected F32 [1] or [], got {dtype} {shape}")
    begin, end = info["data_offsets"]
    if end - begin != 4:
        raise ValueError(f"{name}: expected 4 bytes for F32 scalar, got {end - begin}")
    val = float(struct.unpack("<f", os.pread(fd, 4, data_base + begin))[0])
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name}: scale must be finite and positive, got {val}")
    return val


def decode_tensor_rows_np(
    fd: int, data_base: int, info: dict[str, Any], start_row: int, end_row: int
) -> np.ndarray:
    dtype = info["dtype"]
    shape = info["shape"]
    begin, _ = info["data_offsets"]
    row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    elem_bytes = DTYPE_BYTES[dtype]
    byte_start = data_base + begin + start_row * row_elems * elem_bytes
    byte_end = data_base + begin + end_row * row_elems * elem_bytes
    sub_shape = (end_row - start_row, *shape[1:]) if len(shape) > 1 else (end_row - start_row,)
    buf = os.pread(fd, byte_end - byte_start, byte_start)
    if dtype == "F8_E4M3":
        u8 = np.frombuffer(buf, dtype=np.uint8).reshape(sub_shape)
        return FP8_E4M3_LUT[u8]
    if dtype == "BF16":
        u16 = np.frombuffer(buf, dtype=np.uint16).reshape(sub_shape)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(buf, dtype=np.float16).reshape(sub_shape).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).reshape(sub_shape).copy()
    raise TypeError(f"expected FP8 E4M3 or BF16/FP16/FP32, got {dtype}")


def write_safetensors_raw(
    destination: Path,
    entries: list[tuple[str, str, list[int], bytes | memoryview]],
    metadata: dict[str, str],
) -> None:
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for name, dtype, shape, data in entries:
        nbytes = len(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    if pad:
        header_bytes += b" " * pad
    with destination.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _, _, _, data in entries:
            f.write(data)


# ─────────────────────────────────────────────────────────────────────────────
# Quantization Core (PyTorch + NumPy Bit-Exact Parity)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_chunk_np(x: np.ndarray, scale2_val: float) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or x.shape[1] % 16:
        raise ValueError(f"expected [rows, K] with K divisible by 16, got {tuple(x.shape)}")
    rows, width = x.shape
    scale2_f32 = np.float32(scale2_val)
    grouped = x.astype(np.float32, copy=False).reshape(rows, width // 16, 16)
    block_amax = np.abs(grouped).max(axis=-1)
    block_scale = block_amax / (np.float32(6.0) * scale2_f32)
    block_scale[block_scale == 0] = 1.0
    block_scale_u8 = f32_to_f8e4m3fn_np(block_scale)
    block_scale_f32 = FP8_E4M3_LUT[block_scale_u8]
    scaled = grouped / (block_scale_f32 * scale2_f32)[:, :, None]
    flat = scaled.reshape(rows, width)

    magnitude = np.abs(flat)
    ordinal = np.searchsorted(E2M1_BOUNDS_NP, magnitude, side="left").astype(np.uint8)
    odd_tie = (
        (magnitude == E2M1_BOUNDS_NP[1])
        | (magnitude == E2M1_BOUNDS_NP[3])
        | (magnitude == E2M1_BOUNDS_NP[5])
    )
    code = ((flat < 0).astype(np.uint8) << 3) + ordinal + odd_tie.astype(np.uint8)
    packed = (code[:, 1::2] << 4) | code[:, 0::2]
    return np.ascontiguousarray(packed, dtype=np.uint8), np.ascontiguousarray(block_scale_u8, dtype=np.uint8)


def dequantize_np(packed: np.ndarray, scales_u8: np.ndarray, scale2_val: float) -> np.ndarray:
    code = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    code[:, 0::2] = packed & 0x0F
    code[:, 1::2] = packed >> 4
    values = E2M1_VALUES_NP[code].reshape(packed.shape[0], scales_u8.shape[1], 16)
    scales_f32 = FP8_E4M3_LUT[scales_u8]
    return (values * scales_f32[:, :, None] * np.float32(scale2_val)).reshape(packed.shape[0], -1)


def quantize_chunk(x: Any, scale2: Any) -> tuple[Any, Any]:
    """Exact CPU equivalent of ModelOpt NVFP4QTensor.quantize(block_size=16)."""
    if not HAS_TORCH or isinstance(x, np.ndarray):
        s2 = float(scale2.item() if hasattr(scale2, "item") else scale2)
        return quantize_chunk_np(np.asarray(x), s2)
    if x.ndim != 2 or x.shape[1] % 16:
        raise ValueError(f"expected [rows, K] with K divisible by 16, got {tuple(x.shape)}")
    rows, width = x.shape
    grouped = x.float().reshape(rows, width // 16, 16)
    block_amax = grouped.abs().amax(dim=-1)
    block_scale = block_amax / (6.0 * scale2)
    block_scale[block_scale == 0] = 1.0
    block_scale = block_scale.to(torch.float8_e4m3fn)
    scaled = grouped / (block_scale.float() * scale2).unsqueeze(-1)
    flat = scaled.reshape(rows, width)

    magnitude = flat.abs()
    ordinal = torch.searchsorted(E2M1_BOUNDS, magnitude, out_int32=True).to(torch.uint8)
    # ModelOpt uses round-to-even at the three boundaries whose upper code is odd.
    odd_tie = (magnitude.unsqueeze(-1) == E2M1_BOUNDS[[1, 3, 5]]).any(dim=-1)
    code = ((flat < 0).to(torch.uint8) << 3) + ordinal + odd_tie.to(torch.uint8)
    packed = (code[:, 1::2] << 4) | code[:, 0::2]
    return packed.contiguous(), block_scale.contiguous()


def dequantize(packed: Any, scales: Any, scale2: Any) -> Any:
    if not HAS_TORCH or isinstance(packed, np.ndarray):
        s2 = float(scale2.item() if hasattr(scale2, "item") else scale2)
        return dequantize_np(np.asarray(packed), np.asarray(scales, dtype=np.uint8), s2)
    code = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8)
    code[:, 0::2] = packed & 0x0F
    code[:, 1::2] = packed >> 4
    values = E2M1_VALUES[code.long()].reshape(packed.shape[0], scales.shape[1], 16)
    return (values * scales.float().unsqueeze(-1) * scale2).flatten(1)


def tensor_absmax(tensor: Any, source_scale: float, chunk_rows: int) -> float:
    maximum = 0.0
    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = tensor[start : start + chunk_rows].float()
        maximum = max(maximum, float(chunk.abs().max()) * source_scale)
    return maximum


def quantize_tensor(
    source: Any, source_scale: float, chunk_rows: int
) -> tuple[Any, Any, Any]:
    rows, width = source.shape
    if width % 16:
        raise ValueError(f"PLE width {width} is not divisible by group size 16")
    maximum = tensor_absmax(source, source_scale, chunk_rows)
    scale2 = torch.tensor([maximum / (6.0 * 448.0) if maximum else 1.0], dtype=torch.float32)
    packed = torch.empty((rows, width // 2), dtype=torch.uint8)
    scales = torch.empty((rows, width // 16), dtype=torch.float8_e4m3fn)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        x = source[start:end].float().mul_(source_scale)
        packed[start:end], scales[start:end] = quantize_chunk(x, scale2)
    return packed, scales, scale2


def quantize_tensor_np(
    fd: int,
    data_base: int,
    info: dict[str, Any],
    source_scale: float,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, width = info["shape"]
    if width % 16:
        raise ValueError(f"PLE width {width} is not divisible by group size 16")
    maximum = 0.0
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = decode_tensor_rows_np(fd, data_base, info, start, end)
        maximum = max(maximum, float(np.abs(chunk).max()) * source_scale)
    scale2_val = float(np.float32(maximum / (6.0 * 448.0) if maximum else 1.0))
    scale2_arr = np.array([scale2_val], dtype=np.float32)
    packed = np.empty((rows, width // 2), dtype=np.uint8)
    scales_u8 = np.empty((rows, width // 16), dtype=np.uint8)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = decode_tensor_rows_np(fd, data_base, info, start, end)
        if source_scale != 1.0:
            chunk = chunk * np.float32(source_scale)
        packed[start:end], scales_u8[start:end] = quantize_chunk_np(chunk, scale2_val)
    return packed, scales_u8, scale2_arr


# ─────────────────────────────────────────────────────────────────────────────
# Format Inspection & Scale Discovery (Mode A Native NVFP4 vs Mode B BF16/FP8)
# ─────────────────────────────────────────────────────────────────────────────

def detect_checkpoint_ple_mode(files: list[Path]) -> str:
    """Return 'native_nvfp4' if PLE shards are already U8 [rows, 80], else 'quantize'."""
    for path in files:
        header, _ = read_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__" or not is_ple_weight(name):
                continue
            dtype = info.get("dtype")
            shape = info.get("shape") or []
            if dtype == "U8" and len(shape) == 2 and int(shape[1]) == OFFLOAD_PACKED_BYTES:
                return "native_nvfp4"
            return "quantize"
    raise ValueError("no PLE shard weights (*.ple.ple_embedding.ngram_embedding.shard_*.weight) found")


NVFP4_GLOBAL_SCALE2_SUFFIX = NATIVE_GLOBAL_SCALE2_SUFFIX


def is_ple_nvfp4_auxiliary(name: str) -> bool:
    """Scales that accompany an already-NVFP4 source table (absent from FP8/BF16 sources)."""
    return (PLE_MARKER in name and name.endswith((".weight_scale", ".weight_scale_2"))) or name.endswith(
        NVFP4_GLOBAL_SCALE2_SUFFIX
    )


def has_nvfp4_ple(header: dict[str, Any]) -> bool:
    """Whether a file holds part of an already-NVFP4 PLE table (shards or their scales)."""
    return any(
        name != "__metadata__"
        and ((is_ple_weight(name) and info["dtype"] == "U8") or is_ple_nvfp4_auxiliary(name))
        for name, info in header.items()
    )


def source_is_nvfp4_ple(files: list[Path]) -> bool:
    """Return True if the source checkpoint PLE shards are already in native NVFP4 format."""
    return detect_checkpoint_ple_mode(files) == "native_nvfp4"


def find_native_global_scale2(files: list[Path]) -> float | None:
    """Find table-wide *.ngram_embedding.weight_scale_2 (F32 [1] or []) if present."""
    for path in files:
        header, data_base = read_safetensors_header(path)
        for name, info in header.items():
            if name != "__metadata__" and name.endswith(NATIVE_GLOBAL_SCALE2_SUFFIX):
                with path.open("rb") as f:
                    return read_scalar_f32_from_fd(f.fileno(), data_base, info, name)
    return None


find_nvfp4_global_scale2 = find_native_global_scale2


def read_nvfp4_ple(
    fd: int, data_base: int, header: dict[str, Any], name: str, global_scale2: float | None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Packed E2M1 bytes, FP8 block-scale bytes and global scale of an already-NVFP4 shard."""
    info = header[name]
    rows, packed_width = info["shape"]
    prefix = name.removesuffix(".weight")
    scale_name = f"{prefix}.weight_scale"
    scale_info = header.get(scale_name)
    if scale_info is None:
        raise ValueError(f"{name}: uint8 (NVFP4) PLE shard without {scale_name} in the same file")
    if scale_info["dtype"] not in ("F8_E4M3", "U8") or scale_info["shape"] != [rows, packed_width * 2 // 16]:
        raise TypeError(
            f"{scale_name}: expected F8_E4M3 {[rows, packed_width * 2 // 16]}, "
            f"got {scale_info['dtype']} {scale_info['shape']}"
        )

    def raw(entry: dict[str, Any]) -> bytes:
        begin, end = entry["data_offsets"]
        return os.pread(fd, end - begin, data_base + begin)

    packed = np.frombuffer(raw(info), dtype=np.uint8).reshape(rows, packed_width)
    scales = np.frombuffer(raw(scale_info), dtype=np.uint8).reshape(rows, packed_width * 2 // 16)
    per_shard = header.get(f"{prefix}.weight_scale_2")
    if per_shard is not None:
        scale2 = read_scalar_f32_from_fd(fd, data_base, per_shard, f"{prefix}.weight_scale_2")
    elif global_scale2 is not None:
        scale2 = global_scale2
    else:
        raise ValueError(f"{name}: NVFP4 PLE shard has no per-shard or table-wide weight_scale_2")
    return packed, scales, scale2


def validate_nvfp4_source(files: list[Path]) -> float | None:
    """Validate that a native NVFP4 PLE source has valid weight_scale and weight_scale_2 tensors."""
    global_s2 = find_native_global_scale2(files)
    for path in files:
        header, data_base = read_safetensors_header(path)
        with path.open("rb") as f:
            fd = f.fileno()
            for name, info in header.items():
                if name != "__metadata__" and is_ple_weight(name) and info.get("dtype") == "U8":
                    read_nvfp4_ple(fd, data_base, header, name, global_s2)
    return global_s2


def find_source_scale(files: list[Path]) -> float:
    for path in files:
        header, data_base = read_safetensors_header(path)
        for name, info in header.items():
            if name != "__metadata__" and name.endswith(GLOBAL_SCALE_SUFFIX):
                with path.open("rb") as f:
                    arr = decode_tensor_rows_np(f.fileno(), data_base, info, 0, 1)
                    return float(arr.reshape(-1)[0])
    print(
        "no *.ngram_embedding.weight_scale in source: assuming unscaled (BF16/FP16/FP32) PLE table, source_scale=1.0",
        flush=True,
    )
    return 1.0


def completed_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        header, _ = read_safetensors_header(path)
        return (header.get("__metadata__") or {}).get("atlas_repack") == FORMAT_TAG
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 8-KiB Page-Aligned Sidecar Writers (91 records x 90 bytes + 2 pad bytes)
# ─────────────────────────────────────────────────────────────────────────────

def write_offload_tensor(
    output: Path, packed: Any, scales: Any, *, pages_per_chunk: int = 1024
) -> tuple[int, str]:
    """Write page-indexable records; every row is served by one aligned read.

    91 records fit in an 8 KiB direct-I/O page with only two padding bytes.
    This avoids the 6-byte-per-row waste of a 4 KiB/96-byte layout while
    retaining exactly one O_DIRECT request for any row.
    """
    if HAS_TORCH and not isinstance(packed, np.ndarray):
        packed_np = packed.numpy()
        scales_np = scales.view(torch.uint8).numpy()
    else:
        packed_np = np.asarray(packed, dtype=np.uint8)
        scales_np = np.asarray(scales, dtype=np.uint8)

    if packed_np.ndim != 2 or packed_np.shape[1] != OFFLOAD_PACKED_BYTES:
        raise ValueError(f"packed PLE table must be [rows, {OFFLOAD_PACKED_BYTES}], got {packed_np.shape}")
    if scales_np.ndim != 2 or scales_np.shape[1] != OFFLOAD_SCALE_BYTES:
        raise ValueError(f"PLE group scales must be [rows, {OFFLOAD_SCALE_BYTES}], got {scales_np.shape}")
    if packed_np.shape[0] != scales_np.shape[0]:
        raise ValueError("packed PLE values and group scales have different row counts")

    rows = packed_np.shape[0]
    total_pages = (rows + OFFLOAD_RECORDS_PER_PAGE - 1) // OFFLOAD_RECORDS_PER_PAGE
    temporary = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb", buffering=0) as handle:
        for first_page in range(0, total_pages, pages_per_chunk):
            page_count = min(pages_per_chunk, total_pages - first_page)
            first_row = first_page * OFFLOAD_RECORDS_PER_PAGE
            end_row = min(rows, first_row + page_count * OFFLOAD_RECORDS_PER_PAGE)
            row_count = end_row - first_row

            records = np.zeros((page_count * OFFLOAD_RECORDS_PER_PAGE, OFFLOAD_RECORD_BYTES), dtype=np.uint8)
            records[:row_count, :OFFLOAD_PACKED_BYTES] = packed_np[first_row:end_row]
            records[:row_count, OFFLOAD_PACKED_BYTES:] = scales_np[first_row:end_row]

            page_block = np.zeros((page_count, OFFLOAD_PAGE_BYTES), dtype=np.uint8)
            page_block[:, : OFFLOAD_RECORDS_PER_PAGE * OFFLOAD_RECORD_BYTES] = records.reshape(
                page_count, OFFLOAD_RECORDS_PER_PAGE * OFFLOAD_RECORD_BYTES
            )
            raw = page_block.tobytes()
            handle.write(raw)
            digest.update(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output.stat().st_size, digest.hexdigest()


def write_native_offload_from_fd(
    fd: int,
    data_base: int,
    weight_info: dict[str, Any],
    scale_info: dict[str, Any],
    output: Path,
    *,
    pages_per_chunk: int = 1024,
) -> tuple[int, str]:
    """Stream native U8 [rows, 80] + F8_E4M3 [rows, 10] directly into 8-KiB pages byte-for-byte."""
    w_dtype = weight_info["dtype"]
    s_dtype = scale_info["dtype"]
    w_shape = weight_info["shape"]
    s_shape = scale_info["shape"]
    if w_dtype != "U8":
        raise TypeError(f"native PLE weight must be U8, got {w_dtype}")
    if s_dtype not in ("F8_E4M3", "U8"):
        raise TypeError(f"native PLE weight_scale must be F8_E4M3, got {s_dtype}")
    if len(w_shape) != 2 or int(w_shape[1]) != OFFLOAD_PACKED_BYTES:
        raise ValueError(f"native PLE weight must be [rows, {OFFLOAD_PACKED_BYTES}], got {w_shape}")
    if len(s_shape) != 2 or int(s_shape[1]) != OFFLOAD_SCALE_BYTES:
        raise ValueError(f"native PLE weight_scale must be [rows, {OFFLOAD_SCALE_BYTES}], got {s_shape}")
    rows = int(w_shape[0])
    if int(s_shape[0]) != rows:
        raise ValueError(f"native PLE row count mismatch: weight={rows} vs scale={s_shape[0]}")

    w_begin, w_end = weight_info["data_offsets"]
    s_begin, s_end = scale_info["data_offsets"]
    if (w_end - w_begin) != rows * OFFLOAD_PACKED_BYTES:
        raise ValueError(f"native PLE weight byte length mismatch: {w_end - w_begin}")
    if (s_end - s_begin) != rows * OFFLOAD_SCALE_BYTES:
        raise ValueError(f"native PLE scale byte length mismatch: {s_end - s_begin}")

    total_pages = (rows + OFFLOAD_RECORDS_PER_PAGE - 1) // OFFLOAD_RECORDS_PER_PAGE
    temporary = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb", buffering=0) as handle:
        for first_page in range(0, total_pages, pages_per_chunk):
            page_count = min(pages_per_chunk, total_pages - first_page)
            first_row = first_page * OFFLOAD_RECORDS_PER_PAGE
            end_row = min(rows, first_row + page_count * OFFLOAD_RECORDS_PER_PAGE)
            row_count = end_row - first_row

            w_buf = os.pread(
                fd,
                row_count * OFFLOAD_PACKED_BYTES,
                data_base + w_begin + first_row * OFFLOAD_PACKED_BYTES,
            )
            s_buf = os.pread(
                fd,
                row_count * OFFLOAD_SCALE_BYTES,
                data_base + s_begin + first_row * OFFLOAD_SCALE_BYTES,
            )
            w_np = np.frombuffer(w_buf, dtype=np.uint8).reshape(row_count, OFFLOAD_PACKED_BYTES)
            s_np = np.frombuffer(s_buf, dtype=np.uint8).reshape(row_count, OFFLOAD_SCALE_BYTES)

            records = np.zeros((page_count * OFFLOAD_RECORDS_PER_PAGE, OFFLOAD_RECORD_BYTES), dtype=np.uint8)
            records[:row_count, :OFFLOAD_PACKED_BYTES] = w_np
            records[:row_count, OFFLOAD_PACKED_BYTES:] = s_np

            page_block = np.zeros((page_count, OFFLOAD_PAGE_BYTES), dtype=np.uint8)
            page_block[:, : OFFLOAD_RECORDS_PER_PAGE * OFFLOAD_RECORD_BYTES] = records.reshape(
                page_count, OFFLOAD_RECORDS_PER_PAGE * OFFLOAD_RECORD_BYTES
            )
            raw = page_block.tobytes()
            handle.write(raw)
            digest.update(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output.stat().st_size, digest.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# File Processing (Mode A Passthrough & Mode B Quantize)
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_file(
    source: Path,
    destination: Path,
    source_scale: float,
    chunk_rows: int,
    native_global_scale2: float | None = None,
) -> int:
    header, data_base = read_safetensors_header(source)
    if HAS_TORCH and not has_nvfp4_ple(header):
        tensors: dict[str, Any] = {}
        converted = 0
        with safe_open(source, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if not is_ple_weight(name):
                    tensors[name] = tensor
                    continue
                if tensor.dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32):
                    raise TypeError(f"{name}: expected FP8 E4M3 or BF16/FP16/FP32, got {tensor.dtype}")
                packed, scales, scale2 = quantize_tensor(tensor, source_scale, chunk_rows)
                tensors[name] = packed
                prefix = name.removesuffix(".weight")
                tensors[f"{prefix}.weight_scale"] = scales
                tensors[f"{prefix}.weight_scale_2"] = scale2
                converted += 1
                print(
                    f"  {name}: {tuple(tensor.shape)} -> packed={tuple(packed.shape)} "
                    f"scale2={float(scale2):.9g}",
                    flush=True,
                )
        temporary = destination.with_suffix(destination.suffix + ".partial")
        save_file(tensors, temporary, metadata={"atlas_repack": FORMAT_TAG})
        os.replace(temporary, destination)
        return converted

    converted = 0
    out_entries: list[tuple[str, str, list[int], bytes | memoryview]] = []
    with source.open("rb") as f:
        fd = f.fileno()
        for name, info in header.items():
            if name == "__metadata__" or is_ple_nvfp4_auxiliary(name):
                continue
            if not is_ple_weight(name):
                begin, end = info["data_offsets"]
                out_entries.append((name, info["dtype"], info["shape"], os.pread(fd, end - begin, data_base + begin)))
                continue
            if info["dtype"] == "U8":
                packed, scales_u8, scale2 = read_nvfp4_ple(fd, data_base, header, name, native_global_scale2)
                scale2_arr = np.array([scale2], dtype=np.float32)
            else:
                packed, scales_u8, scale2_arr = quantize_tensor_np(fd, data_base, info, source_scale, chunk_rows)
            prefix = name.removesuffix(".weight")
            out_entries.append((name, "U8", list(packed.shape), packed.tobytes()))
            out_entries.append((f"{prefix}.weight_scale", "F8_E4M3", list(scales_u8.shape), scales_u8.tobytes()))
            out_entries.append((f"{prefix}.weight_scale_2", "F32", [1], scale2_arr.tobytes()))
            converted += 1
            print(
                f"  {name}: {tuple(info['shape'])} -> packed={tuple(packed.shape)} "
                f"scale2={float(scale2_arr[0]):.9g}"
                + (" (NVFP4 pass-through)" if info["dtype"] == "U8" else ""),
                flush=True,
            )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    write_safetensors_raw(temporary, out_entries, metadata={"atlas_repack": FORMAT_TAG})
    os.replace(temporary, destination)
    return converted


repack_nvfp4_file = rewrite_file


def process_file_offload(
    source: Path,
    destination: Path,
    offload_dir: Path,
    source_scale: float,
    chunk_rows: int,
    *,
    native_global_scale2: float | None = None,
    keep_source_shards: bool = False,
    shard_filter: set[int] | None = None,
    pages_per_chunk: int = 1024,
) -> list[dict[str, object]]:
    """Materialize 8-KiB page-aligned NVFP4 PLE sidecars (passthrough or quantized)."""
    header, data_base = read_safetensors_header(source)
    ple_weight_names = [
        name for name in header.keys()
        if name != "__metadata__" and is_ple_weight(name)
        and (shard_filter is None or parse_ple_shard_index(name) in shard_filter)
    ]
    if not ple_weight_names:
        if not keep_source_shards and source != destination and not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        return []

    first_info = header[ple_weight_names[0]]
    is_native = (
        first_info.get("dtype") == "U8"
        and len(first_info.get("shape") or []) == 2
        and int(first_info["shape"][1]) == OFFLOAD_PACKED_BYTES
    )

    entries: list[dict[str, object]] = []
    if is_native:
        with source.open("rb") as f:
            fd = f.fileno()
            for name in ple_weight_names:
                w_info = header[name]
                prefix = name.removesuffix(".weight")
                s_name = f"{prefix}.weight_scale"
                if s_name not in header:
                    raise KeyError(f"{source.name}: missing companion scale tensor {s_name}")
                s_info = header[s_name]
                s2_name = f"{prefix}.weight_scale_2"
                if s2_name in header:
                    scale2_val = read_scalar_f32_from_fd(fd, data_base, header[s2_name], s2_name)
                elif native_global_scale2 is not None:
                    scale2_val = native_global_scale2
                else:
                    raise KeyError(
                        f"{source.name}: missing {s2_name} and no table-wide *{NATIVE_GLOBAL_SCALE2_SUFFIX} found"
                    )
                shard = parse_ple_shard_index(name)
                filename = f"ple-ngram-nvfp4-{shard:03d}.bin"
                size, sha256 = write_native_offload_from_fd(
                    fd,
                    data_base,
                    w_info,
                    s_info,
                    offload_dir / filename,
                    pages_per_chunk=pages_per_chunk,
                )
                rows = int(w_info["shape"][0])
                entries.append(
                    {
                        "tensor": name,
                        "shard": shard,
                        "file": filename,
                        "rows": rows,
                        "width": LOGICAL_PLE_WIDTH,
                        "scale2": float(scale2_val),
                        "bytes": size,
                        "sha256": sha256,
                    }
                )
                print(
                    f"  [passthrough] {name}: offload={filename} rows={rows} bytes={size} scale2={scale2_val:.9g}",
                    flush=True,
                )

        if keep_source_shards:
            if source != destination and not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        else:
            out_entries: list[tuple[str, str, list[int], bytes | memoryview]] = []
            with source.open("rb") as f:
                fd = f.fileno()
                for name, info in header.items():
                    if name == "__metadata__" or is_ple_related_tensor(name):
                        continue
                    begin, end = info["data_offsets"]
                    out_entries.append((name, info["dtype"], info["shape"], os.pread(fd, end - begin, data_base + begin)))
            temporary = destination.with_suffix(destination.suffix + ".partial")
            write_safetensors_raw(temporary, out_entries, metadata={"atlas_repack": OFFLOAD_FORMAT_TAG})
            os.replace(temporary, destination)
        return entries

    # Mode B: BF16 / FP8 / FP16 / FP32 -> Group-16 NVFP4 Quantization
    if HAS_TORCH:
        tensors: dict[str, Any] = {}
        with safe_open(source, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not is_ple_weight(name):
                    if not keep_source_shards and not is_ple_related_tensor(name):
                        tensors[name] = handle.get_tensor(name)
                    continue
                shard = parse_ple_shard_index(name)
                if shard_filter is not None and shard not in shard_filter:
                    continue
                tensor = handle.get_tensor(name)
                if tensor.dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32):
                    raise TypeError(f"{name}: expected FP8 E4M3 or BF16/FP16/FP32, got {tensor.dtype}")
                packed, scales, scale2 = quantize_tensor(tensor, source_scale, chunk_rows)
                filename = f"ple-ngram-nvfp4-{shard:03d}.bin"
                size, sha256 = write_offload_tensor(
                    offload_dir / filename, packed, scales, pages_per_chunk=pages_per_chunk
                )
                entries.append(
                    {
                        "tensor": name,
                        "shard": shard,
                        "file": filename,
                        "rows": int(tensor.shape[0]),
                        "width": int(tensor.shape[1]),
                        "scale2": float(scale2),
                        "bytes": size,
                        "sha256": sha256,
                    }
                )
                print(
                    f"  [quantize] {name}: offload={filename} bytes={size} scale2={float(scale2):.9g}",
                    flush=True,
                )
        if keep_source_shards:
            if source != destination and not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        else:
            temporary = destination.with_suffix(destination.suffix + ".partial")
            save_file(tensors, temporary, metadata={"atlas_repack": OFFLOAD_FORMAT_TAG})
            os.replace(temporary, destination)
        return entries

    out_entries = []
    with source.open("rb") as f:
        fd = f.fileno()
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not is_ple_weight(name):
                if not keep_source_shards and not is_ple_related_tensor(name):
                    begin, end = info["data_offsets"]
                    out_entries.append((name, info["dtype"], info["shape"], os.pread(fd, end - begin, data_base + begin)))
                continue
            shard = parse_ple_shard_index(name)
            if shard_filter is not None and shard not in shard_filter:
                continue
            packed, scales_u8, scale2_arr = quantize_tensor_np(fd, data_base, info, source_scale, chunk_rows)
            filename = f"ple-ngram-nvfp4-{shard:03d}.bin"
            size, sha256 = write_offload_tensor(
                offload_dir / filename, packed, scales_u8, pages_per_chunk=pages_per_chunk
            )
            entries.append(
                {
                    "tensor": name,
                    "shard": shard,
                    "file": filename,
                    "rows": int(info["shape"][0]),
                    "width": int(info["shape"][1]),
                    "scale2": float(scale2_arr[0]),
                    "bytes": size,
                    "sha256": sha256,
                }
            )
            print(
                f"  [quantize] {name}: offload={filename} bytes={size} scale2={float(scale2_arr[0]):.9g}",
                flush=True,
            )
    if keep_source_shards:
        if source != destination and not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
    else:
        temporary = destination.with_suffix(destination.suffix + ".partial")
        write_safetensors_raw(temporary, out_entries, metadata={"atlas_repack": OFFLOAD_FORMAT_TAG})
        os.replace(temporary, destination)
    return entries


def _patch_cyberfrost_chat_template(destination: Path) -> None:
    """Normalize Cyber-Frost chat_template.jinja / tokenizer_config.json for standard reasoning_effort."""
    ct_path = destination / "chat_template.jinja"
    if ct_path.exists():
        try:
            ct = ct_path.read_text(encoding="utf-8")
            new_ct = ct.replace("reasoning_effort|default('xhigh')", "reasoning_effort|default('xhigh', true)")
            new_ct = new_ct.replace("{%- set enable_thinking = true %}\n", "")
            if new_ct != ct:
                ct_path.write_text(new_ct, encoding="utf-8")
        except Exception:
            pass

    tc_path = destination / "tokenizer_config.json"
    if tc_path.exists():
        try:
            tc_raw = tc_path.read_text(encoding="utf-8")
            if "reasoning_effort|default('xhigh')" in tc_raw or "{%- set enable_thinking = true %}" in tc_raw:
                tc_data = json.loads(tc_raw)
                ct_str = tc_data.get("chat_template")
                if isinstance(ct_str, str):
                    new_ct_str = ct_str.replace("reasoning_effort|default('xhigh')", "reasoning_effort|default('xhigh', true)")
                    new_ct_str = new_ct_str.replace("{%- set enable_thinking = true %}\n", "")
                    if new_ct_str != ct_str:
                        tc_data["chat_template"] = new_ct_str
                        tc_path.write_text(json.dumps(tc_data, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass


def copy_metadata(source: Path, destination: Path, *, include_index: bool = False) -> None:
    if source != destination:
        for path in source.iterdir():
            if not path.is_file() or path.suffix == ".safetensors":
                continue
            if path.name == "model.safetensors.index.json" and not include_index:
                continue
            target = destination / path.name
            if not target.exists():
                shutil.copy2(path, target)
    _patch_cyberfrost_chat_template(destination)


def rebuild_index(destination: Path) -> None:
    weight_map: dict[str, str] = {}
    total_size = 0
    for path in sorted(destination.glob("*.safetensors")):
        header, _ = read_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map:
                raise RuntimeError(f"duplicate output tensor {name}")
            weight_map[name] = path.name
            elements = 1
            for dim in info["shape"]:
                elements *= int(dim)
            total_size += elements * DTYPE_BYTES[info["dtype"]]
    payload = {"metadata": {"total_size": total_size, "atlas_repack": FORMAT_TAG}, "weight_map": weight_map}
    index = destination / "model.safetensors.index.json"
    temporary = index.with_suffix(index.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, index)


def self_test() -> None:
    if HAS_TORCH:
        torch.manual_seed(38)
        x = torch.randn(37, 160, dtype=torch.float32) * 0.07
        scale2 = torch.tensor([float(x.abs().max()) / (6.0 * 448.0)])
        packed, scales = quantize_chunk(x, scale2)
        restored = dequantize(packed, scales, scale2)
        if not torch.isfinite(restored).all():
            raise AssertionError("non-finite NVFP4 reconstruction")
        relative_l2 = float(torch.linalg.vector_norm(restored - x) / torch.linalg.vector_norm(x))
        packed_np = packed.numpy()
        scales_u8_np = scales.view(torch.uint8).numpy()
        np_packed, np_scales = quantize_chunk_np(x.numpy(), float(scale2.item()))
        if not np.array_equal(packed_np, np_packed) or not np.array_equal(scales_u8_np, np_scales):
            raise AssertionError("NumPy backend does not match PyTorch bit-for-bit")
        scale2_val = float(scale2.item())
    else:
        rng = np.random.default_rng(38)
        x_np = (rng.standard_normal((37, 160), dtype=np.float32) * 0.07).astype(np.float32)
        scale2_val = float(np.float32(float(np.abs(x_np).max()) / (6.0 * 448.0)))
        packed_np, scales_u8_np = quantize_chunk_np(x_np, scale2_val)
        restored_np = dequantize_np(packed_np, scales_u8_np, scale2_val)
        if not np.isfinite(restored_np).all():
            raise AssertionError("non-finite NVFP4 reconstruction")
        relative_l2 = float(np.linalg.norm(restored_np - x_np) / np.linalg.norm(x_np))
        packed, scales = packed_np, scales_u8_np

    if relative_l2 > 0.16:
        raise AssertionError(f"unexpected NVFP4 error: relative_l2={relative_l2}")
    with tempfile.TemporaryDirectory() as directory:
        tmp_dir = Path(directory)
        path = tmp_dir / "tiny.bin"
        size, sha_direct = write_offload_tensor(path, packed, scales)
        expected_pages = (37 + OFFLOAD_RECORDS_PER_PAGE - 1) // OFFLOAD_RECORDS_PER_PAGE
        if size != expected_pages * OFFLOAD_PAGE_BYTES:
            raise AssertionError(f"offload size {size} != {expected_pages} pages")
        raw = path.read_bytes()
        for row in (0, 36):
            page = row // OFFLOAD_RECORDS_PER_PAGE
            slot = row % OFFLOAD_RECORDS_PER_PAGE
            offset = page * OFFLOAD_PAGE_BYTES + slot * OFFLOAD_RECORD_BYTES
            record = raw[offset : offset + OFFLOAD_RECORD_BYTES]
            if record[:80] != packed_np[row].tobytes():
                raise AssertionError(f"packed offload row {row} mismatch")
            if record[80:] != scales_u8_np[row].tobytes():
                raise AssertionError(f"scale offload row {row} mismatch")

        # Also test Mode A (Native NVFP4 Passthrough) end-to-end from a synthetic safetensors file
        native_st = tmp_dir / "model-00001-of-00001.safetensors"
        t_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        write_safetensors_raw(
            native_st,
            [
                (f"{t_prefix}.shard_0.weight", "U8", [37, 80], packed_np.tobytes()),
                (f"{t_prefix}.shard_0.weight_scale", "F8_E4M3", [37, 10], scales_u8_np.tobytes()),
                (f"{t_prefix}.weight_scale_2", "F32", [1], np.array([scale2_val], dtype=np.float32).tobytes()),
            ],
            metadata={},
        )
        mode_detected = detect_checkpoint_ple_mode([native_st])
        if mode_detected != "native_nvfp4":
            raise AssertionError(f"expected native_nvfp4 mode detection, got {mode_detected}")
        s2_detected = find_native_global_scale2([native_st])
        if s2_detected is None or abs(s2_detected - scale2_val) > 1e-12:
            raise AssertionError(f"native global scale2 mismatch: {s2_detected} vs {scale2_val}")
        offload_sub = tmp_dir / "ple-offload"
        offload_sub.mkdir()
        entries = process_file_offload(
            native_st,
            native_st,
            offload_sub,
            1.0,
            chunk_rows=16,
            native_global_scale2=s2_detected,
            keep_source_shards=True,
        )
        if len(entries) != 1 or entries[0]["sha256"] != sha_direct or entries[0]["width"] != 160:
            raise AssertionError(f"native passthrough mismatch: {entries}")

        # Regression: native NVFP4 with separate destination defaults to offload; explicit
        # --mode resident rebuilds model.safetensors.index.json with per-shard scale2 keys;
        # --mode offload --strip-source-shards --shards 0 rebuilds index without stripped PLE.
        src_ckpt = tmp_dir / "src-native"
        src_ckpt.mkdir()
        src_st = src_ckpt / "model-00001-of-00001.safetensors"
        write_safetensors_raw(
            src_st,
            [
                ("model.embed_tokens.weight", "BF16", [4, 4], np.zeros((4, 4), dtype=np.uint16).tobytes()),
                (f"{t_prefix}.shard_0.weight", "U8", [37, 80], packed_np.tobytes()),
                (f"{t_prefix}.shard_0.weight_scale", "F8_E4M3", [37, 10], scales_u8_np.tobytes()),
                (f"{t_prefix}.weight_scale_2", "F32", [1], np.array([scale2_val], dtype=np.float32).tobytes()),
            ],
            metadata={},
        )
        rebuild_index(src_ckpt)
        dst_res = tmp_dir / "dst-resident"
        orig_argv = sys.argv
        try:
            sys.argv = [orig_argv[0], str(src_ckpt), str(dst_res), "--mode", "resident"]
            main()
            res_idx = json.loads((dst_res / "model.safetensors.index.json").read_text())
            wmap = res_idx["weight_map"]
            if f"{t_prefix}.weight_scale_2" in wmap:
                raise AssertionError("stale table-wide weight_scale_2 remained in resident index")
            if f"{t_prefix}.shard_0.weight_scale_2" not in wmap:
                raise AssertionError("missing per-shard weight_scale_2 in resident index")

            dst_strip = tmp_dir / "dst-strip-shard0"
            sys.argv = [
                orig_argv[0],
                str(src_ckpt),
                str(dst_strip),
                "--mode",
                "offload",
                "--strip-source-shards",
                "--shards",
                "0",
            ]
            main()
            strip_idx = json.loads((dst_strip / "model.safetensors.index.json").read_text())
            strip_wmap = strip_idx["weight_map"]
            if f"{t_prefix}.shard_0.weight" in strip_wmap:
                raise AssertionError("stripped shard_0 PLE weight remained in index")
            if "model.embed_tokens.weight" not in strip_wmap:
                raise AssertionError("non-PLE weight lost from stripped index")

            dst_default = tmp_dir / "dst-native-default"
            sys.argv = [orig_argv[0], str(src_ckpt), str(dst_default)]
            main()
            if not (dst_default / "ple-offload" / "manifest.json").is_file():
                raise AssertionError("native NVFP4 separate-dst did not default to offload mode")
        finally:
            sys.argv = orig_argv

    modelopt = ""
    if HAS_TORCH:
        try:
            from modelopt.torch.quantization.qtensor import NVFP4QTensor

            reference, reference_scales, reference_scale2 = NVFP4QTensor.quantize(
                x.clone(), 16, weights_scaling_factor_2=scale2
            )
            if not torch.equal(packed, reference._quantized_data):
                raise AssertionError("packed E2M1 data differs from ModelOpt")
            if not torch.equal(scales.view(torch.uint8), reference_scales.view(torch.uint8)):
                raise AssertionError("FP8 group scales differ from ModelOpt")
            if not torch.equal(scale2, reference_scale2):
                raise AssertionError("global scale differs from ModelOpt")
            modelopt = "; exact ModelOpt parity"
        except ImportError:
            modelopt = "; ModelOpt not installed (parity check skipped)"
    else:
        modelopt = "; NumPy standalone backend (bit-exact)"
    print(f"self-test passed: relative_l2={relative_l2:.6f}; native NVFP4 passthrough verified{modelopt}")


def parse_shards_arg(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    result: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="Source model checkpoint directory")
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        help="Destination checkpoint directory (optional; defaults to in-place inside SOURCE when omitted)",
    )
    parser.add_argument(
        "--model",
        "--source",
        dest="model_flag",
        type=Path,
        help="Source model checkpoint directory (alias for positional SOURCE)",
    )
    parser.add_argument(
        "--output-dir",
        "--destination",
        dest="output_dir_flag",
        type=Path,
        help="Destination checkpoint directory or ple-offload directory",
    )
    parser.add_argument("--chunk-rows", type=int, default=32768)
    parser.add_argument("--pages-per-chunk", type=int, default=1024)
    parser.add_argument(
        "--mode",
        choices=("resident", "offload"),
        default=None,
        help="offload writes 8-KiB-page NVMe sidecars into ple-offload/ (default for in-place/native NVFP4); resident writes NVFP4 safetensors",
    )
    parser.add_argument(
        "--offload-dir",
        type=Path,
        help="sidecar directory for --mode offload (default: DESTINATION/ple-offload)",
    )
    shard_group = parser.add_mutually_exclusive_group()
    shard_group.add_argument(
        "--keep-source-shards",
        action="store_true",
        default=None,
        help="Keep PLE tensors in safetensors shards and model.safetensors.index.json untouched (default for Native NVFP4 passthrough and in-place offload)",
    )
    shard_group.add_argument(
        "--strip-source-shards",
        action="store_true",
        default=False,
        help="Strip PLE tensors from destination safetensors shards and rebuild model.safetensors.index.json to reclaim disk space",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Optional comma/range list of PLE shard indices to materialize (e.g. '0' or '0-7'; default: all 0..127)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    src_arg = args.source or args.model_flag
    dst_arg = args.destination
    offload_dir_arg = args.offload_dir

    if src_arg is None:
        parser.error("source directory (positional or --model) is required unless --self-test is used")
    source = src_arg.resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")

    if args.output_dir_flag is not None:
        out_resolved = args.output_dir_flag.resolve()
        if dst_arg is None:
            if out_resolved.name == "ple-offload" or out_resolved.parent == source:
                offload_dir_arg = offload_dir_arg or out_resolved
                dst_arg = source
            else:
                dst_arg = out_resolved
        elif offload_dir_arg is None:
            offload_dir_arg = out_resolved

    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    if args.pages_per_chunk <= 0:
        parser.error("--pages-per-chunk must be positive")

    files = sorted(source.glob("*.safetensors"))
    if not files:
        parser.error(f"no safetensors files found in {source}")

    ple_source_mode = detect_checkpoint_ple_mode(files)
    shard_filter = parse_shards_arg(args.shards)

    # Resolve destination and default mode after probing checkpoint PLE format:
    # - In-place (destination omitted or == source) or native NVFP4 defaults to offload.
    # - Quantized BF16/FP8 source with a separate destination defaults to resident.
    if dst_arg is None:
        destination = source
    else:
        destination = dst_arg.resolve()
    mode = args.mode or (
        "offload" if (destination == source or ple_source_mode == "native_nvfp4") else "resident"
    )

    if mode == "resident" and source == destination:
        parser.error("--mode resident requires a separate destination directory")

    # Determine whether to keep or strip source safetensors shards:
    # - Resident mode always rewrites PLE shards (replacing table-wide weight_scale_2
    #   with per-shard .shard_N.weight_scale_2), so the index must be rebuilt.
    # - Offload mode:
    #   - Explicit --strip-source-shards -> keep_source_shards=False
    #   - Explicit --keep-source-shards -> keep_source_shards=True
    #   - In-place, native_nvfp4, or partial --shards -> keep_source_shards=True by default
    #   - Otherwise (Mode B quantize with separate destination) -> strip by default
    if mode == "resident":
        keep_source_shards = False
    elif args.strip_source_shards:
        keep_source_shards = False
    elif args.keep_source_shards is not None:
        keep_source_shards = args.keep_source_shards
    elif source == destination or ple_source_mode == "native_nvfp4" or shard_filter is not None:
        keep_source_shards = True
    else:
        keep_source_shards = False

    destination.mkdir(parents=True, exist_ok=True)
    offload_dir = (offload_dir_arg or (destination / "ple-offload")).resolve()
    if mode == "offload":
        offload_dir.mkdir(parents=True, exist_ok=True)

    copy_metadata(source, destination, include_index=keep_source_shards)

    native_global_scale2: float | None = None
    source_scale = 1.0
    if ple_source_mode == "native_nvfp4":
        native_global_scale2 = find_native_global_scale2(files)
        print(
            f"detected Mode A (Native NVFP4 PLE passthrough): global_scale2={native_global_scale2} "
            f"keep_source_shards={keep_source_shards}",
            flush=True,
        )
    else:
        source_scale = find_source_scale(files)
        print(
            f"detected Mode B (PLE -> NVFP4 quantize): source_scale={source_scale:.9g} "
            f"keep_source_shards={keep_source_shards}",
            flush=True,
        )

    total = 0
    offload_entries: list[dict[str, object]] = []
    for number, path in enumerate(files, 1):
        output = destination / path.name
        header, _ = read_safetensors_header(path)
        contains_ple = any(
            is_ple_weight(name) and (shard_filter is None or parse_ple_shard_index(name) in shard_filter)
            for name in header.keys()
            if name != "__metadata__"
        )
        if not contains_ple:
            if source != destination and not output.exists() and (shard_filter is None or not keep_source_shards):
                try:
                    os.link(path, output)
                except OSError:
                    shutil.copy2(path, output)
            continue
        if mode == "resident" and completed_output(output):
            print(f"[{number}/{len(files)}] resume {path.name}", flush=True)
            continue
        print(f"[{number}/{len(files)}] process {path.name}", flush=True)
        if mode == "resident":
            total += rewrite_file(
                path,
                output,
                source_scale,
                args.chunk_rows,
                native_global_scale2=native_global_scale2,
            )
        else:
            entries = process_file_offload(
                path,
                output,
                offload_dir,
                source_scale,
                args.chunk_rows,
                native_global_scale2=native_global_scale2,
                keep_source_shards=keep_source_shards,
                shard_filter=shard_filter,
                pages_per_chunk=args.pages_per_chunk,
            )
            offload_entries.extend(entries)
            total += len(entries)

    if mode == "resident" or not keep_source_shards:
        rebuild_index(destination)

    if mode == "offload":
        manifest = {
            "format": OFFLOAD_FORMAT_TAG,
            "page_bytes": OFFLOAD_PAGE_BYTES,
            "record_bytes": OFFLOAD_RECORD_BYTES,
            "records_per_page": OFFLOAD_RECORDS_PER_PAGE,
            "packed_bytes": OFFLOAD_PACKED_BYTES,
            "scale_bytes": OFFLOAD_SCALE_BYTES,
            "group_size": 16,
            "entries": sorted(offload_entries, key=lambda item: int(item["shard"])),
        }
        manifest_path = offload_dir / "manifest.json"
        temporary = manifest_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, manifest_path)
    print(f"complete: materialized {total} PLE shard(s) into {offload_dir if mode == 'offload' else destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
