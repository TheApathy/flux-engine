#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Offline trainer and ATLDC001 authenticated sidecar exporter for Flux-DCOCO-v2.

Trains the lightweight contextual pairwise draft corrector matching the exact ABI
and tensor dimensions of ``kernels/gb10/common/dcoco_selector.cu`` and
``crates/spark-model/src/layers/dflash_head/dcoco.rs``:

  - Projection rank ``r = 128``
  - Pair MLP width ``3 * r + 2 = 386 -> 256 -> 1`` (exact erf-GELU)
  - 10 contiguous BF16 tensors with zero-initialized residual output layer
    (``mlp_w2 = 0``, ``mlp_b2 = 0``) so step-0 behavior equals raw unary logits
  - Deterministic, SHA-256 authenticated ``ATLDC001`` binary sidecar + JSON receipt

Supports CPU-only PyTorch (``--backend torch``) and a zero-PyTorch vectorized
pure-NumPy analytic-backprop AdamW fallback (``--backend numpy``). Safe to run
with ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DCOCO_SIDECAR_MAGIC = b"ATLDC001"
DCOCO_SIDECAR_VERSION = 1
DCOCO_SIDECAR_HEADER_PREFIX_BYTES = 40
DCOCO_SIDECAR_HEADER_BYTES = 72

DCOCO_RANK = 128
DCOCO_MLP_IN = 3 * DCOCO_RANK + 2  # 386
DCOCO_MLP_HIDDEN = 256
DCOCO_MAX_K = 16
DCOCO_DEFAULT_LN_EPS = 1.0e-5

DCOCO_SIDECAR_ENV = "ATLAS_DCOCO_SIDECAR"
DCOCO_SIDECAR_SHA256_ENV = "ATLAS_DCOCO_SIDECAR_SHA256"
DCOCO_SIDECAR_SIZE_ENV = "ATLAS_DCOCO_SIDECAR_SIZE"

TENSOR_ORDER = (
    "w_cand",
    "w_ctx",
    "w_fuse",
    "fuse_ln_w",
    "fuse_ln_b",
    "w_pair",
    "mlp_w1",
    "mlp_b1",
    "mlp_w2",
    "mlp_b2",
)


def expected_tensor_shapes(hidden_size: int) -> dict[str, tuple[int, ...]]:
    """Return exact tensor shapes matching ``dcoco_selector.cu``."""
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
    return {
        "w_cand": (DCOCO_RANK, hidden_size),
        "w_ctx": (DCOCO_RANK, hidden_size),
        "w_fuse": (DCOCO_RANK, 2 * DCOCO_RANK),
        "fuse_ln_w": (DCOCO_RANK,),
        "fuse_ln_b": (DCOCO_RANK,),
        "w_pair": (DCOCO_RANK, 2 * hidden_size),
        "mlp_w1": (DCOCO_MLP_HIDDEN, DCOCO_MLP_IN),
        "mlp_b1": (DCOCO_MLP_HIDDEN,),
        "mlp_w2": (DCOCO_MLP_HIDDEN,),
        "mlp_b2": (1,),
    }


def expected_param_count(hidden_size: int) -> int:
    """Return exact parameter count ``512 * d + 132_353``."""
    return 512 * hidden_size + 132_353


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    """Convert FP32 array to IEEE-754 RNE BF16 ``<u2`` matching CUDA ``__float2bfloat16_rn``."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(a)):
        raise ValueError("DCOCO tensor contains NaN or Inf; refusing to pack BF16")
    bits = a.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    rounded = (bits + rounding_bias) >> np.uint32(16)
    bf16 = rounded.astype("<u2")
    if np.any((bf16 & np.uint16(0x7F80)) == np.uint16(0x7F80)):
        raise ValueError("BF16 rounding overflowed to Inf/NaN; refusing to pack sidecar")
    return bf16


def bf16_u16_to_f32(bf16: np.ndarray) -> np.ndarray:
    """Widen little-endian BF16 ``u16`` values back to FP32."""
    u16 = np.ascontiguousarray(bf16, dtype="<u2")
    if np.any((u16 & np.uint16(0x7F80)) == np.uint16(0x7F80)):
        raise ValueError("BF16 payload contains NaN or Inf")
    bits = u16.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def round_bf16_f32_np(arr: np.ndarray) -> np.ndarray:
    """Round FP32 array to BF16 RNE and widen back to FP32."""
    return bf16_u16_to_f32(f32_to_bf16_u16(arr))


def dcoco_gelu_np(x: np.ndarray) -> np.ndarray:
    """Exact A&S 7.1.26 erf-GELU matching CUDA ``dcoco_gelu`` and Rust ``dcoco_gelu_f32``."""
    x32 = np.asarray(x, dtype=np.float32)
    z = x32 * np.float32(1.0 / math.sqrt(2.0))
    az = np.abs(z)
    t = np.float32(1.0) / (np.float32(1.0) + np.float32(0.3275911) * az)
    poly = (
        (
            (
                (np.float32(1.0614054) * t - np.float32(1.453152)) * t
                + np.float32(1.4214137)
            )
            * t
            - np.float32(0.28449674)
        )
        * t
        + np.float32(0.2548296)
    )
    erf_abs = np.float32(1.0) - (poly * t) * np.exp(-(az * az))
    erf_val = np.where(z < 0.0, -erf_abs, erf_abs)
    return np.float32(0.5) * x32 * (np.float32(1.0) + erf_val)


def dcoco_gelu_grad_np(x: np.ndarray) -> np.ndarray:
    """Exact derivative of erf-GELU for pure-NumPy analytic backprop."""
    x32 = np.asarray(x, dtype=np.float32)
    z = x32 * np.float32(1.0 / math.sqrt(2.0))
    az = np.abs(z)
    t = np.float32(1.0) / (np.float32(1.0) + np.float32(0.3275911) * az)
    poly = (
        (
            (
                (np.float32(1.0614054) * t - np.float32(1.453152)) * t
                + np.float32(1.4214137)
            )
            * t
            - np.float32(0.28449674)
        )
        * t
        + np.float32(0.2548296)
    )
    erf_abs = np.float32(1.0) - (poly * t) * np.exp(-(az * az))
    erf_val = np.where(z < 0.0, -erf_abs, erf_abs)
    cdf = np.float32(0.5) * (np.float32(1.0) + erf_val)
    pdf = np.exp(np.float32(-0.5) * x32 * x32) * np.float32(1.0 / math.sqrt(2.0 * math.pi))
    return cdf + x32 * pdf


def _ln_forward(
    x: np.ndarray,
    eps: float,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    mean = np.mean(x, axis=-1, keepdims=True)
    xc = x - mean
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    inv_std = np.float32(1.0) / np.sqrt(var + np.float32(eps))
    x_hat = xc * inv_std
    if weight is not None and bias is not None:
        y = x_hat * weight + bias
    else:
        y = x_hat
    return y.astype(np.float32, copy=False), (x_hat.astype(np.float32, copy=False), inv_std.astype(np.float32, copy=False))


def _ln_backward(
    dy: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray],
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    x_hat, inv_std = cache
    r = np.float32(x_hat.shape[-1])
    if weight is not None:
        reduce_axes = tuple(range(dy.ndim - 1))
        d_weight = np.sum(dy * x_hat, axis=reduce_axes).astype(np.float32)
        d_bias = np.sum(dy, axis=reduce_axes).astype(np.float32)
        d_xhat = dy * weight
    else:
        d_weight = None
        d_bias = None
        d_xhat = dy
    sum_dxhat = np.sum(d_xhat, axis=-1, keepdims=True)
    sum_dxhat_xhat = np.sum(d_xhat * x_hat, axis=-1, keepdims=True)
    dx = (inv_std / r) * (r * d_xhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    return dx.astype(np.float32, copy=False), d_weight, d_bias


@dataclass(frozen=True)
class RolloutBatch:
    """Validated rollout batch for offline DCOCO-v2 contextual corrector training."""

    embed_tokens: np.ndarray  # [V, d] float32
    hidden_states: np.ndarray  # [N, gamma + 1, d] float32
    anchor_ids: np.ndarray  # [N] int64
    anchor_logits: np.ndarray  # [N] float32
    candidates: np.ndarray  # [N, gamma, K] int64
    unary: np.ndarray  # [N, gamma, K] float32
    target_slot_idx: np.ndarray  # [N, gamma] int64 (-1 when absent from shortlist)
    target_dist: np.ndarray  # [N, gamma, K] float32 (sums to 1 on valid positions)
    valid_mask: np.ndarray  # [N, gamma] float32 (1.0 if both prev & curr valid)

    @property
    def num_samples(self) -> int:
        return int(self.hidden_states.shape[0])

    @property
    def gamma(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def top_k(self) -> int:
        return int(self.candidates.shape[2])

    @property
    def hidden_size(self) -> int:
        return int(self.embed_tokens.shape[1])


def _build_rollout_batch(
    embed_tokens: np.ndarray,
    hidden_states: np.ndarray,
    anchor_ids: np.ndarray,
    candidates: np.ndarray,
    unary: np.ndarray,
    target_tokens: np.ndarray | None = None,
    target_logits: np.ndarray | None = None,
    anchor_logits: np.ndarray | None = None,
) -> RolloutBatch:
    embed = np.ascontiguousarray(embed_tokens, dtype=np.float32)
    hidden = np.ascontiguousarray(hidden_states, dtype=np.float32)
    anchors = np.ascontiguousarray(anchor_ids, dtype=np.int64)
    cands = np.ascontiguousarray(candidates, dtype=np.int64)
    un = np.ascontiguousarray(unary, dtype=np.float32)

    if embed.ndim != 2 or embed.shape[0] == 0 or embed.shape[1] == 0:
        raise ValueError(f"embed_tokens must have shape [V, d], got {embed.shape}")
    vocab_size, d = embed.shape
    if hidden.ndim != 3 or hidden.shape[2] != d or hidden.shape[1] < 2:
        raise ValueError(f"hidden_states must have shape [N, gamma+1, {d}], got {hidden.shape}")
    n, slots, _ = hidden.shape
    gamma = slots - 1
    if anchors.shape != (n,):
        raise ValueError(f"anchor_ids must have shape ({n},), got {anchors.shape}")
    if cands.ndim != 3 or cands.shape[0] != n or cands.shape[1] != gamma:
        raise ValueError(f"candidates must have shape ({n}, {gamma}, K), got {cands.shape}")
    top_k = cands.shape[2]
    if not (1 <= top_k <= DCOCO_MAX_K):
        raise ValueError(f"top_k must be in 1..={DCOCO_MAX_K}, got {top_k}")
    if un.shape != (n, gamma, top_k):
        raise ValueError(f"unary must have shape ({n}, {gamma}, {top_k}), got {un.shape}")
    if anchor_logits is None:
        a_logits = np.zeros((n,), dtype=np.float32)
    else:
        a_logits = np.ascontiguousarray(anchor_logits, dtype=np.float32)
        if a_logits.shape != (n,):
            raise ValueError(f"anchor_logits must have shape ({n},), got {a_logits.shape}")

    for name, arr in (
        ("embed_tokens", embed),
        ("hidden_states", hidden),
        ("unary", un),
        ("anchor_logits", a_logits),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"rollout array {name} contains NaN or Inf")
    if np.any(anchors < 0) or np.any(anchors >= vocab_size):
        raise ValueError("anchor_ids contains out-of-range token index")
    if np.any(cands < 0) or np.any(cands >= vocab_size):
        raise ValueError("candidates contains out-of-range token index")

    target_slot_idx = np.full((n, gamma), -1, dtype=np.int64)
    target_dist = np.zeros((n, gamma, top_k), dtype=np.float32)
    pos_valid = np.zeros((n, gamma), dtype=np.bool_)

    if target_logits is not None:
        t_log = np.ascontiguousarray(target_logits, dtype=np.float32)
        if t_log.shape != (n, gamma, top_k):
            raise ValueError(f"target_logits must have shape ({n}, {gamma}, {top_k}), got {t_log.shape}")
        if not np.all(np.isfinite(t_log)):
            raise ValueError("target_logits contains NaN or Inf")
        shifted = t_log - np.max(t_log, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        target_dist = (exps / np.sum(exps, axis=-1, keepdims=True)).astype(np.float32)
        target_slot_idx = np.argmax(target_dist, axis=-1).astype(np.int64)
        pos_valid[:, :] = True
    elif target_tokens is not None:
        t_tok = np.ascontiguousarray(target_tokens, dtype=np.int64)
        if t_tok.shape != (n, gamma):
            raise ValueError(f"target_tokens must have shape ({n}, {gamma}), got {t_tok.shape}")
        matches = cands == t_tok[:, :, None]
        has_match = np.any(matches, axis=-1)
        first_idx = np.argmax(matches, axis=-1).astype(np.int64)
        target_slot_idx = np.where(has_match, first_idx, -1)
        pos_valid = has_match
        rows, cols = np.nonzero(has_match)
        target_dist[rows, cols, first_idx[rows, cols]] = 1.0
    else:
        raise ValueError("rollouts must supply target_tokens or target_logits")

    # A transition at p > 0 requires a valid teacher-forced previous target slot at p - 1.
    valid_mask = np.zeros((n, gamma), dtype=np.float32)
    valid_mask[:, 0] = pos_valid[:, 0].astype(np.float32)
    if gamma > 1:
        valid_mask[:, 1:] = (pos_valid[:, 1:] & pos_valid[:, :-1]).astype(np.float32)
    if float(np.sum(valid_mask)) <= 0.0:
        raise ValueError("rollouts contain zero valid teacher-forced transitions inside top-K")

    return RolloutBatch(
        embed_tokens=embed,
        hidden_states=hidden,
        anchor_ids=anchors,
        anchor_logits=a_logits,
        candidates=cands,
        unary=un,
        target_slot_idx=target_slot_idx,
        target_dist=target_dist,
        valid_mask=valid_mask,
    )


def load_rollouts(path: Path) -> RolloutBatch:
    """Ingest rollouts from ``.npz`` or ``.json``."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            raw: dict[str, Any] = {k: data[k] for k in data.files}
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if not isinstance(loaded, dict):
            raise ValueError("JSON rollouts root must be an object")
        raw = {k: np.asarray(v) for k, v in loaded.items()}
    else:
        raise ValueError(f"unsupported rollout format `{suffix}`; expected .npz or .json")

    candidates = raw.get("drafter_topk_ids", raw.get("candidates"))
    if candidates is None:
        raise ValueError("rollouts missing `drafter_topk_ids` (or `candidates`)")

    if "unary" in raw:
        unary = np.asarray(raw["unary"], dtype=np.float32)
    elif "drafter_topk_logits" in raw:
        logits = np.asarray(raw["drafter_topk_logits"], dtype=np.float32)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        unary = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    elif "drafter_topk_probs" in raw:
        probs = np.clip(np.asarray(raw["drafter_topk_probs"], dtype=np.float32), 1.0e-12, 1.0)
        unary = np.log(probs).astype(np.float32)
    else:
        raise ValueError(
            "rollouts missing `unary`, `drafter_topk_logits`, or `drafter_topk_probs`"
        )

    return _build_rollout_batch(
        embed_tokens=raw["embed_tokens"],
        hidden_states=raw["hidden_states"],
        anchor_ids=raw["anchor_ids"],
        anchor_logits=raw.get("anchor_logits"),
        candidates=candidates,
        unary=unary,
        target_tokens=raw.get("target_tokens"),
        target_logits=raw.get("target_logits"),
    )


def generate_synthetic_rollouts(
    hidden_size: int,
    top_k: int,
    gamma: int = 4,
    num_samples: int = 128,
    vocab_size: int = 64,
    seed: int = 20261127,
) -> RolloutBatch:
    """Generate deterministic contextual transition rollouts where unary top-1 fails.

    In every position ``p``, the drafter's independent unary distribution puts
    highest probability on candidate slot ``0``, while the ground-truth next
    token depends on the *preceding* token parity/class (simulating parallel
    block-diffusion token inconsistency in ``<think>`` chains). A unary-only
    selector achieves < 35% accuracy while the DCOCO-v2 pairwise MLP learns
    the contextual transition to > 95% accuracy.
    """
    if vocab_size < max(top_k + 4, 16):
        raise ValueError(f"vocab_size={vocab_size} too small for top_k={top_k}")
    rng = np.random.default_rng(seed)

    # Structured embeddings: first 2 coordinates cleanly encode token parity class
    # while remaining coordinates are isotropic Gaussian features.
    embed_tokens = rng.normal(0.0, 0.35, size=(vocab_size, hidden_size)).astype(np.float32)
    for tok in range(vocab_size):
        cls = tok % top_k
        embed_tokens[tok, cls % hidden_size] += 2.5
        embed_tokens[tok, (cls + top_k) % hidden_size] -= 1.5

    hidden_states = rng.normal(0.0, 0.25, size=(num_samples, gamma + 1, hidden_size)).astype(
        np.float32
    )
    anchor_ids = rng.integers(0, vocab_size, size=(num_samples,), dtype=np.int64)
    anchor_logits = rng.normal(-0.5, 0.2, size=(num_samples,)).astype(np.float32)

    candidates = np.zeros((num_samples, gamma, top_k), dtype=np.int64)
    unary = np.zeros((num_samples, gamma, top_k), dtype=np.float32)
    target_tokens = np.zeros((num_samples, gamma), dtype=np.int64)

    base_unary = np.linspace(-0.35, -1.85, top_k, dtype=np.float32)
    for n in range(num_samples):
        prev_tok = int(anchor_ids[n])
        for p in range(gamma):
            # One distinct candidate per parity class so the target class is always in top-K.
            base_block = int(rng.integers(0, vocab_size // top_k)) * top_k
            row_cands = np.array([(base_block + k) % vocab_size for k in range(top_k)], dtype=np.int64)
            # True target depends on the previous token's class shifted by (p + 1) mod top_k.
            true_slot = (prev_tok + p + 1) % top_k
            candidates[n, p, :] = row_cands
            # Independent parallel drafter systematically biases slot 0 (wrong whenever true_slot != 0).
            noise = rng.normal(0.0, 0.05, size=(top_k,)).astype(np.float32)
            logits = base_unary + noise
            shifted = logits - np.max(logits)
            unary[n, p, :] = shifted - np.log(np.sum(np.exp(shifted)))
            target_tokens[n, p] = row_cands[true_slot]
            # Encode position step signal lightly in hidden states too.
            hidden_states[n, p + 1, p % hidden_size] += 0.6
            prev_tok = int(row_cands[true_slot])

    return _build_rollout_batch(
        embed_tokens=embed_tokens,
        hidden_states=hidden_states,
        anchor_ids=anchor_ids,
        anchor_logits=anchor_logits,
        candidates=candidates,
        unary=unary,
        target_tokens=target_tokens,
    )


def init_dcoco_weights_np(hidden_size: int, seed: int) -> dict[str, np.ndarray]:
    """Deterministically initialize the 10 DCOCO-v2 tensors with zero residual output."""
    rng = np.random.default_rng(seed)

    def xavier(rows: int, cols: int) -> np.ndarray:
        scale = math.sqrt(2.0 / float(rows + cols))
        return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)

    return {
        "w_cand": xavier(DCOCO_RANK, hidden_size),
        "w_ctx": xavier(DCOCO_RANK, hidden_size),
        "w_fuse": xavier(DCOCO_RANK, 2 * DCOCO_RANK),
        "fuse_ln_w": np.ones((DCOCO_RANK,), dtype=np.float32),
        "fuse_ln_b": np.zeros((DCOCO_RANK,), dtype=np.float32),
        "w_pair": xavier(DCOCO_RANK, 2 * hidden_size),
        "mlp_w1": xavier(DCOCO_MLP_HIDDEN, DCOCO_MLP_IN),
        "mlp_b1": np.zeros((DCOCO_MLP_HIDDEN,), dtype=np.float32),
        # Zero-initialized residual output guarantees step-0 delta == 0.0 identically.
        "mlp_w2": np.zeros((DCOCO_MLP_HIDDEN,), dtype=np.float32),
        "mlp_b2": np.zeros((1,), dtype=np.float32),
    }


def eval_single_sample_full_scores_np(
    embed_tokens: np.ndarray,
    hidden_states: np.ndarray,
    anchor_id: int,
    anchor_logit: float,
    candidates: np.ndarray,
    unary: np.ndarray,
    weights: dict[str, np.ndarray],
    ln_eps: float = DCOCO_DEFAULT_LN_EPS,
    bf16_round: bool = True,
) -> np.ndarray:
    """Exact single-sample evaluator mirroring ``dcoco_proj_fuse_cpu`` + ``dcoco_pair_mlp_factored_cpu``.

    Returns full ``[gamma, top_k, top_k]`` transition score tensor (unused rows
    for ``p == 0, a > 0`` set to ``-inf``).
    """
    gamma, top_k = candidates.shape
    slots = gamma + 1
    q = round_bf16_f32_np if bf16_round else (lambda arr: np.asarray(arr, dtype=np.float32))

    w_cand = q(weights["w_cand"])
    w_ctx = q(weights["w_ctx"])
    w_fuse = q(weights["w_fuse"])
    fuse_ln_w = q(weights["fuse_ln_w"])
    fuse_ln_b = q(weights["fuse_ln_b"])
    w_pair = q(weights["w_pair"])
    mlp_w1 = q(weights["mlp_w1"])
    mlp_b1 = q(weights["mlp_b1"])
    mlp_w2 = q(weights["mlp_w2"])
    mlp_b2 = float(q(weights["mlp_b2"])[0])

    emb_q = q(embed_tokens)
    hid_q = q(hidden_states)

    z_cand = np.zeros((slots, top_k, DCOCO_RANK), dtype=np.float32)
    s_pair = np.zeros((gamma, DCOCO_RANK), dtype=np.float32)

    for slot in range(slots):
        h_cur = hid_q[slot]
        g_vec, _ = _ln_forward(w_ctx @ h_cur, ln_eps)
        slot_k = 1 if slot == 0 else top_k
        for k in range(slot_k):
            cid = int(anchor_id) if slot == 0 else int(candidates[slot - 1, k])
            b_vec, _ = _ln_forward(w_cand @ emb_q[cid], ln_eps)
            bg = np.concatenate([b_vec, g_vec], axis=0)
            u_vec, _ = _ln_forward(w_fuse @ bg, ln_eps, fuse_ln_w, fuse_ln_b)
            z_cand[slot, k] = q(u_vec)
        if slot < gamma:
            h_pair = np.concatenate([h_cur, hid_q[slot + 1]], axis=0)
            s_pair[slot] = q(w_pair @ h_pair)

    out = np.full((gamma, top_k, top_k), -np.inf, dtype=np.float32)
    for p in range(gamma):
        k_prev = 1 if p == 0 else top_k
        sp = q(s_pair[p])
        base = mlp_b1 + mlp_w1[:, 0:DCOCO_RANK] @ sp
        for a in range(k_prev):
            za = q(z_cand[p, a])
            ell_a = float(anchor_logit) if p == 0 else float(unary[p - 1, a])
            term_a = mlp_w1[:, DCOCO_RANK : 2 * DCOCO_RANK] @ za + mlp_w1[:, 384] * np.float32(ell_a)
            for b in range(top_k):
                zb = q(z_cand[p + 1, b])
                ell_b = float(unary[p, b])
                term_b = mlp_w1[:, 2 * DCOCO_RANK : 3 * DCOCO_RANK] @ zb + mlp_w1[:, 385] * np.float32(ell_b)
                pre = base + term_a + term_b
                acc = float(np.dot(mlp_w2, dcoco_gelu_np(pre)))
                out[p, a, b] = np.float32(ell_b + acc + mlp_b2)
    return out


def _numpy_forward_and_grads(
    batch: RolloutBatch,
    params: dict[str, np.ndarray],
    ln_eps: float,
) -> tuple[float, float, dict[str, np.ndarray], np.ndarray]:
    """Vectorized forward pass + exact analytic gradients for teacher-forced DCOCO-v2."""
    n = batch.num_samples
    gamma = batch.gamma
    top_k = batch.top_k

    # 1. Context projections g_s: [N, gamma + 1, 128]
    g_pre = np.einsum("nsd,rd->nsr", batch.hidden_states, params["w_ctx"], optimize=True)
    g, g_cache = _ln_forward(g_pre, ln_eps)

    # 2. Anchor candidate projection (slot 0): [N, 128]
    emb_0 = batch.embed_tokens[batch.anchor_ids]  # [N, d]
    b0_pre = emb_0 @ params["w_cand"].T
    b0, b0_cache = _ln_forward(b0_pre, ln_eps)
    bg0 = np.concatenate([b0, g[:, 0, :]], axis=-1)  # [N, 256]
    u0_pre = bg0 @ params["w_fuse"].T
    z0, u0_cache = _ln_forward(u0_pre, ln_eps, params["fuse_ln_w"], params["fuse_ln_b"])

    # 3. Draft candidate projections (slots 1..gamma): [N, gamma, K, 128]
    emb_c = batch.embed_tokens[batch.candidates]  # [N, gamma, K, d]
    bc_pre = np.einsum("ngkd,rd->ngkr", emb_c, params["w_cand"], optimize=True)
    bc, bc_cache = _ln_forward(bc_pre, ln_eps)
    g_c = np.broadcast_to(g[:, 1:, None, :], (n, gamma, top_k, DCOCO_RANK))
    bgc = np.concatenate([bc, g_c], axis=-1)  # [N, gamma, K, 256]
    uc_pre = np.einsum("ngkj,rj->ngkr", bgc, params["w_fuse"], optimize=True)
    zc, uc_cache = _ln_forward(uc_pre, ln_eps, params["fuse_ln_w"], params["fuse_ln_b"])

    # 4. Pair state s_p: [N, gamma, 128]
    h_pair = np.concatenate(
        [batch.hidden_states[:, :-1, :], batch.hidden_states[:, 1:, :]], axis=-1
    )  # [N, gamma, 2*d]
    s_pair = np.einsum("ngj,rj->ngr", h_pair, params["w_pair"], optimize=True)

    # 5. Gather teacher-forced previous state z_{p, a_prev} and ell_{p}(a_prev)
    z_prev = np.zeros((n, gamma, DCOCO_RANK), dtype=np.float32)
    ell_prev = np.zeros((n, gamma), dtype=np.float32)
    z_prev[:, 0, :] = z0
    ell_prev[:, 0] = batch.anchor_logits
    prev_idx_safe = np.zeros((n, max(gamma - 1, 0)), dtype=np.int64)
    if gamma > 1:
        prev_idx_safe = np.maximum(batch.target_slot_idx[:, :-1], 0)
        n_idx = np.arange(n)[:, None]
        p_idx = np.arange(gamma - 1)[None, :]
        z_prev[:, 1:, :] = zc[n_idx, p_idx, prev_idx_safe, :]
        ell_prev[:, 1:] = batch.unary[n_idx, p_idx, prev_idx_safe]

    # 6. Assemble pair MLP input [N, gamma, K, 386]
    feat = np.empty((n, gamma, top_k, DCOCO_MLP_IN), dtype=np.float32)
    feat[:, :, :, 0:DCOCO_RANK] = s_pair[:, :, None, :]
    feat[:, :, :, DCOCO_RANK : 2 * DCOCO_RANK] = z_prev[:, :, None, :]
    feat[:, :, :, 2 * DCOCO_RANK : 3 * DCOCO_RANK] = zc
    feat[:, :, :, 384] = ell_prev[:, :, None]
    feat[:, :, :, 385] = batch.unary

    pre = np.einsum("ngkj,hj->ngkh", feat, params["mlp_w1"], optimize=True) + params["mlp_b1"]
    act = dcoco_gelu_np(pre)
    delta = np.einsum("ngkh,h->ngk", act, params["mlp_w2"], optimize=True) + params["mlp_b2"][0]
    scores = batch.unary + delta

    # 7. Masked cross-entropy loss & top-1 accuracy
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    probs = np.exp(log_probs)
    per_pos_loss = -np.sum(batch.target_dist * log_probs, axis=-1)
    denom = float(np.sum(batch.valid_mask))
    loss = float(np.sum(per_pos_loss * batch.valid_mask) / denom)

    pred_slot = np.argmax(scores, axis=-1)
    correct = (pred_slot == batch.target_slot_idx).astype(np.float32) * batch.valid_mask
    acc = float(np.sum(correct) / denom)

    # 8. Analytic backward pass
    d_scores = ((probs - batch.target_dist) * batch.valid_mask[:, :, None]) / np.float32(denom)
    d_mlp_b2 = np.array([float(np.sum(d_scores))], dtype=np.float32)
    d_mlp_w2 = np.einsum("ngk,ngkh->h", d_scores, act, optimize=True).astype(np.float32)

    d_act = d_scores[:, :, :, None] * params["mlp_w2"][None, None, None, :]
    d_pre = d_act * dcoco_gelu_grad_np(pre)
    d_mlp_b1 = np.sum(d_pre, axis=(0, 1, 2)).astype(np.float32)
    d_mlp_w1 = np.einsum("ngkh,ngkj->hj", d_pre, feat, optimize=True).astype(np.float32)

    d_feat = np.einsum("ngkh,hj->ngkj", d_pre, params["mlp_w1"], optimize=True)
    d_s_pair = np.sum(d_feat[:, :, :, 0:DCOCO_RANK], axis=2)
    d_w_pair = np.einsum("ngr,ngj->rj", d_s_pair, h_pair, optimize=True).astype(np.float32)

    d_z_prev = np.sum(d_feat[:, :, :, DCOCO_RANK : 2 * DCOCO_RANK], axis=2)
    d_zc = d_feat[:, :, :, 2 * DCOCO_RANK : 3 * DCOCO_RANK].copy()
    if gamma > 1:
        n_idx = np.arange(n)[:, None]
        p_idx = np.arange(gamma - 1)[None, :]
        np.add.at(d_zc, (n_idx, p_idx, prev_idx_safe), d_z_prev[:, 1:, :])
    d_z0 = d_z_prev[:, 0, :]

    d_u0_pre, d_lnw0, d_lnb0 = _ln_backward(d_z0, u0_cache, params["fuse_ln_w"])
    d_uc_pre, d_lnwc, d_lnbc = _ln_backward(d_zc, uc_cache, params["fuse_ln_w"])
    assert d_lnw0 is not None and d_lnwc is not None
    assert d_lnb0 is not None and d_lnbc is not None
    d_fuse_ln_w = (d_lnw0 + d_lnwc).astype(np.float32)
    d_fuse_ln_b = (d_lnb0 + d_lnbc).astype(np.float32)

    d_w_fuse = (
        (d_u0_pre.T @ bg0)
        + np.einsum("ngkr,ngkj->rj", d_uc_pre, bgc, optimize=True)
    ).astype(np.float32)
    d_bg0 = d_u0_pre @ params["w_fuse"]
    d_bgc = np.einsum("ngkr,rj->ngkj", d_uc_pre, params["w_fuse"], optimize=True)

    d_b0 = d_bg0[:, 0:DCOCO_RANK]
    d_g0 = d_bg0[:, DCOCO_RANK : 2 * DCOCO_RANK]
    d_bc = d_bgc[:, :, :, 0:DCOCO_RANK]
    d_gc = np.sum(d_bgc[:, :, :, DCOCO_RANK : 2 * DCOCO_RANK], axis=2)

    d_b0_pre, _, _ = _ln_backward(d_b0, b0_cache)
    d_bc_pre, _, _ = _ln_backward(d_bc, bc_cache)
    d_w_cand = (
        (d_b0_pre.T @ emb_0)
        + np.einsum("ngkr,ngkd->rd", d_bc_pre, emb_c, optimize=True)
    ).astype(np.float32)

    d_g = np.concatenate([d_g0[:, None, :], d_gc], axis=1)
    d_g_pre, _, _ = _ln_backward(d_g, g_cache)
    d_w_ctx = np.einsum("nsr,nsd->rd", d_g_pre, batch.hidden_states, optimize=True).astype(
        np.float32
    )

    grads = {
        "w_cand": d_w_cand,
        "w_ctx": d_w_ctx,
        "w_fuse": d_w_fuse,
        "fuse_ln_w": d_fuse_ln_w,
        "fuse_ln_b": d_fuse_ln_b,
        "w_pair": d_w_pair,
        "mlp_w1": d_mlp_w1,
        "mlp_b1": d_mlp_b1,
        "mlp_w2": d_mlp_w2,
        "mlp_b2": d_mlp_b2,
    }
    return loss, acc, grads, scores


def train_numpy(
    batch: RolloutBatch,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    ln_eps: float = DCOCO_DEFAULT_LN_EPS,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Train DCOCO-v2 corrector with pure-NumPy vectorized AdamW."""
    params = init_dcoco_weights_np(batch.hidden_size, seed)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v_opt = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, adam_eps = np.float32(0.9), np.float32(0.999), np.float32(1.0e-8)

    init_loss, init_acc, _, init_scores = _numpy_forward_and_grads(batch, params, ln_eps)
    init_residual_max = float(np.max(np.abs(init_scores - batch.unary)))

    final_loss, final_acc = init_loss, init_acc
    for step in range(1, steps + 1):
        loss, acc, grads, _ = _numpy_forward_and_grads(batch, params, ln_eps)
        final_loss, final_acc = loss, acc
        b1_corr = np.float32(1.0 - float(beta1) ** step)
        b2_corr = np.float32(1.0 - float(beta2) ** step)
        for name in TENSOR_ORDER:
            g = grads[name]
            m[name] = beta1 * m[name] + (np.float32(1.0) - beta1) * g
            v_opt[name] = beta2 * v_opt[name] + (np.float32(1.0) - beta2) * (g * g)
            m_hat = m[name] / b1_corr
            v_hat = v_opt[name] / b2_corr
            if weight_decay > 0.0 and params[name].ndim >= 2:
                params[name] *= np.float32(1.0 - lr * weight_decay)
            params[name] -= np.float32(lr) * m_hat / (np.sqrt(v_hat) + adam_eps)

    final_loss, final_acc, _, _ = _numpy_forward_and_grads(batch, params, ln_eps)
    metrics = {
        "backend": "numpy",
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "initial_loss": round(init_loss, 6),
        "final_loss": round(final_loss, 6),
        "initial_acc": round(init_acc, 6),
        "final_acc": round(final_acc, 6),
        "initial_zero_residual_max_abs": init_residual_max,
    }
    return params, metrics


def train_torch(
    batch: RolloutBatch,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    ln_eps: float = DCOCO_DEFAULT_LN_EPS,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Train DCOCO-v2 corrector on CPU with PyTorch AdamW."""
    import torch
    import torch.nn.functional as F

    device = torch.device("cpu")
    torch.manual_seed(seed)

    init_np = init_dcoco_weights_np(batch.hidden_size, seed)
    t_params = {
        k: torch.nn.Parameter(torch.from_numpy(init_np[k].copy()).to(device=device, dtype=torch.float32))
        for k in TENSOR_ORDER
    }

    embed = torch.from_numpy(batch.embed_tokens).to(device=device)
    hidden = torch.from_numpy(batch.hidden_states).to(device=device)
    anchor_ids = torch.from_numpy(batch.anchor_ids).to(device=device)
    anchor_logits = torch.from_numpy(batch.anchor_logits).to(device=device)
    candidates = torch.from_numpy(batch.candidates).to(device=device)
    unary = torch.from_numpy(batch.unary).to(device=device)
    target_slot_idx = torch.from_numpy(batch.target_slot_idx).to(device=device)
    target_dist = torch.from_numpy(batch.target_dist).to(device=device)
    valid_mask = torch.from_numpy(batch.valid_mask).to(device=device)

    n, gamma, top_k = batch.num_samples, batch.gamma, batch.top_k
    prev_idx_safe = (
        torch.clamp(target_slot_idx[:, :-1], min=0) if gamma > 1 else torch.empty((n, 0), dtype=torch.long)
    )
    h_pair = torch.cat([hidden[:, :-1, :], hidden[:, 1:, :]], dim=-1)
    emb_0 = embed[anchor_ids]
    emb_c = embed[candidates]
    denom = valid_mask.sum().clamp_min(1.0)

    def forward_scores() -> torch.Tensor:
        g_pre = torch.einsum("nsd,rd->nsr", hidden, t_params["w_ctx"])
        g = F.layer_norm(g_pre, (DCOCO_RANK,), eps=ln_eps)

        b0 = F.layer_norm(F.linear(emb_0, t_params["w_cand"]), (DCOCO_RANK,), eps=ln_eps)
        z0 = F.layer_norm(
            F.linear(torch.cat([b0, g[:, 0, :]], dim=-1), t_params["w_fuse"]),
            (DCOCO_RANK,),
            t_params["fuse_ln_w"],
            t_params["fuse_ln_b"],
            eps=ln_eps,
        )

        bc = F.layer_norm(
            torch.einsum("ngkd,rd->ngkr", emb_c, t_params["w_cand"]),
            (DCOCO_RANK,),
            eps=ln_eps,
        )
        g_c = g[:, 1:, None, :].expand(n, gamma, top_k, DCOCO_RANK)
        zc = F.layer_norm(
            torch.einsum("ngkj,rj->ngkr", torch.cat([bc, g_c], dim=-1), t_params["w_fuse"]),
            (DCOCO_RANK,),
            t_params["fuse_ln_w"],
            t_params["fuse_ln_b"],
            eps=ln_eps,
        )

        s_pair = torch.einsum("ngj,rj->ngr", h_pair, t_params["w_pair"])

        if gamma > 1:
            n_idx = torch.arange(n, device=device)[:, None]
            p_idx = torch.arange(gamma - 1, device=device)[None, :]
            z_prev_tail = zc[n_idx, p_idx, prev_idx_safe, :]
            ell_prev_tail = unary[n_idx, p_idx, prev_idx_safe]
            z_prev = torch.cat([z0[:, None, :], z_prev_tail], dim=1)
            ell_prev = torch.cat([anchor_logits[:, None], ell_prev_tail], dim=1)
        else:
            z_prev = z0[:, None, :]
            ell_prev = anchor_logits[:, None]

        feat = torch.cat(
            [
                s_pair[:, :, None, :].expand(n, gamma, top_k, DCOCO_RANK),
                z_prev[:, :, None, :].expand(n, gamma, top_k, DCOCO_RANK),
                zc,
                ell_prev[:, :, None, None].expand(n, gamma, top_k, 1),
                unary[:, :, :, None],
            ],
            dim=-1,
        )
        pre = F.linear(feat, t_params["mlp_w1"], t_params["mlp_b1"])
        act = F.gelu(pre, approximate="none")
        delta = torch.einsum("ngkh,h->ngk", act, t_params["mlp_w2"]) + t_params["mlp_b2"][0]
        return unary + delta

    decay_params = [t_params[k] for k in TENSOR_ORDER if t_params[k].ndim >= 2]
    no_decay_params = [t_params[k] for k in TENSOR_ORDER if t_params[k].ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    with torch.no_grad():
        init_scores = forward_scores()
        init_residual_max = float((init_scores - unary).abs().max().item())
        init_log_probs = F.log_softmax(init_scores, dim=-1)
        init_loss = float((-(target_dist * init_log_probs).sum(dim=-1) * valid_mask).sum().item() / denom.item())
        init_acc = float(((init_scores.argmax(dim=-1) == target_slot_idx).float() * valid_mask).sum().item() / denom.item())

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        scores = forward_scores()
        log_probs = F.log_softmax(scores, dim=-1)
        loss = (-(target_dist * log_probs).sum(dim=-1) * valid_mask).sum() / denom
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_scores = forward_scores()
        final_log_probs = F.log_softmax(final_scores, dim=-1)
        final_loss = float((-(target_dist * final_log_probs).sum(dim=-1) * valid_mask).sum().item() / denom.item())
        final_acc = float(((final_scores.argmax(dim=-1) == target_slot_idx).float() * valid_mask).sum().item() / denom.item())

    out_params = {k: t_params[k].detach().cpu().numpy().astype(np.float32, copy=True) for k in TENSOR_ORDER}
    metrics = {
        "backend": "torch",
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "initial_loss": round(init_loss, 6),
        "final_loss": round(final_loss, 6),
        "initial_acc": round(init_acc, 6),
        "final_acc": round(final_acc, 6),
        "initial_zero_residual_max_abs": init_residual_max,
    }
    return out_params, metrics


def select_backend(requested: str) -> str:
    """Resolve training backend (``torch`` or ``numpy``)."""
    if requested == "numpy":
        return "numpy"
    if requested == "torch":
        import torch  # noqa: F401
        return "torch"
    if requested == "auto":
        try:
            import torch  # noqa: F401
            return "torch"
        except ImportError:
            return "numpy"
    raise ValueError(f"unsupported backend `{requested}`; expected auto, torch, or numpy")


def serialize_atldc001(
    weights: dict[str, np.ndarray],
    hidden_size: int,
    top_k: int,
    ln_eps: float = DCOCO_DEFAULT_LN_EPS,
) -> tuple[bytes, bytes, int]:
    """Serialize the 10 DCOCO-v2 tensors into the deterministic ATLDC001 wire format.

    Returns ``(full_bytes, header_payload_sha256_bytes, payload_bytes_len)``.
    """
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
    if not (1 <= top_k <= DCOCO_MAX_K):
        raise ValueError(f"top_k must be in 1..={DCOCO_MAX_K}, got {top_k}")
    if not math.isfinite(ln_eps) or ln_eps <= 0.0:
        raise ValueError(f"ln_eps must be finite and positive, got {ln_eps}")

    shapes = expected_tensor_shapes(hidden_size)
    chunks: list[bytes] = []
    total_params = 0
    for name in TENSOR_ORDER:
        if name not in weights:
            raise ValueError(f"missing DCOCO tensor `{name}`")
        arr = np.asarray(weights[name], dtype=np.float32)
        if arr.shape != shapes[name]:
            raise ValueError(f"tensor `{name}` shape {arr.shape} != expected {shapes[name]}")
        bf16 = f32_to_bf16_u16(arr)
        chunks.append(bf16.tobytes())
        total_params += bf16.size

    if total_params != expected_param_count(hidden_size):
        raise ValueError(
            f"total_params={total_params} != expected {expected_param_count(hidden_size)}"
        )
    payload = b"".join(chunks)
    payload_bytes = len(payload)
    ln_eps_bits = struct.unpack("<I", struct.pack("<f", float(ln_eps)))[0]

    header_prefix = struct.pack(
        "<8sIIIIIIQ",
        DCOCO_SIDECAR_MAGIC,
        DCOCO_SIDECAR_VERSION,
        hidden_size,
        top_k,
        DCOCO_RANK,
        DCOCO_MLP_HIDDEN,
        ln_eps_bits,
        payload_bytes,
    )
    assert len(header_prefix) == DCOCO_SIDECAR_HEADER_PREFIX_BYTES
    embedded_digest = hashlib.sha256(header_prefix + payload).digest()
    full_bytes = header_prefix + embedded_digest + payload
    assert len(full_bytes) == DCOCO_SIDECAR_HEADER_BYTES + payload_bytes
    return full_bytes, embedded_digest, payload_bytes


def parse_and_verify_atldc001(raw: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Parse and authenticate an ``ATLDC001`` byte buffer (fail-closed)."""
    if len(raw) < DCOCO_SIDECAR_HEADER_BYTES:
        raise ValueError(f"ATLDC001 truncated: {len(raw)} < {DCOCO_SIDECAR_HEADER_BYTES} bytes")
    (
        magic,
        version,
        hidden_size,
        top_k,
        rank,
        mlp_hidden,
        ln_eps_bits,
        payload_bytes,
    ) = struct.unpack("<8sIIIIIIQ", raw[:DCOCO_SIDECAR_HEADER_PREFIX_BYTES])
    if magic != DCOCO_SIDECAR_MAGIC:
        raise ValueError(f"bad ATLDC001 magic: {magic!r}")
    if version != DCOCO_SIDECAR_VERSION:
        raise ValueError(f"unsupported ATLDC001 version: {version}")
    if hidden_size == 0:
        raise ValueError("ATLDC001 hidden_size must be > 0")
    if not (1 <= top_k <= DCOCO_MAX_K):
        raise ValueError(f"ATLDC001 top_k out of range: {top_k}")
    if rank != DCOCO_RANK:
        raise ValueError(f"ATLDC001 rank={rank} != {DCOCO_RANK}")
    if mlp_hidden != DCOCO_MLP_HIDDEN:
        raise ValueError(f"ATLDC001 mlp_hidden={mlp_hidden} != {DCOCO_MLP_HIDDEN}")
    ln_eps = struct.unpack("<f", struct.pack("<I", ln_eps_bits))[0]
    if not math.isfinite(ln_eps) or ln_eps <= 0.0:
        raise ValueError(f"ATLDC001 ln_eps invalid: {ln_eps}")
    expected_bytes = expected_param_count(hidden_size) * 2
    if payload_bytes != expected_bytes:
        raise ValueError(f"ATLDC001 payload_bytes={payload_bytes} != expected {expected_bytes}")
    if len(raw) != DCOCO_SIDECAR_HEADER_BYTES + payload_bytes:
        raise ValueError(
            f"ATLDC001 file length {len(raw)} != expected {DCOCO_SIDECAR_HEADER_BYTES + payload_bytes}"
        )

    embedded_digest = raw[DCOCO_SIDECAR_HEADER_PREFIX_BYTES:DCOCO_SIDECAR_HEADER_BYTES]
    payload = raw[DCOCO_SIDECAR_HEADER_BYTES:]
    computed = hashlib.sha256(raw[:DCOCO_SIDECAR_HEADER_PREFIX_BYTES] + payload).digest()
    if computed != embedded_digest:
        raise ValueError("ATLDC001 embedded SHA-256 mismatch (header or payload tampered)")

    bf16_all = np.frombuffer(payload, dtype="<u2")
    f32_all = bf16_u16_to_f32(bf16_all)
    shapes = expected_tensor_shapes(hidden_size)
    tensors: dict[str, np.ndarray] = {}
    cursor = 0
    for name in TENSOR_ORDER:
        shape = shapes[name]
        count = 1
        for dim in shape:
            count *= dim
        tensors[name] = f32_all[cursor : cursor + count].reshape(shape).copy()
        cursor += count

    header = {
        "version": version,
        "hidden_size": hidden_size,
        "top_k": top_k,
        "rank": rank,
        "mlp_hidden": mlp_hidden,
        "ln_eps": ln_eps,
        "payload_bytes": payload_bytes,
        "header_payload_sha256": embedded_digest.hex(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }
    return header, tensors


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".atldc_tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def export_sidecar_and_receipt(
    weights: dict[str, np.ndarray],
    hidden_size: int,
    top_k: int,
    out_path: Path,
    receipt_path: Path | None,
    training_metrics: dict[str, Any],
    ln_eps: float = DCOCO_DEFAULT_LN_EPS,
    reference_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``ATLDC001`` binary sidecar atomically and emit its JSON receipt."""
    full_bytes, embedded_digest, payload_bytes = serialize_atldc001(
        weights, hidden_size=hidden_size, top_k=top_k, ln_eps=ln_eps
    )
    _atomic_write_bytes(out_path, full_bytes)
    canonical_out = str(out_path.resolve())
    file_sha256 = hashlib.sha256(full_bytes).hexdigest()
    file_size = len(full_bytes)

    receipt: dict[str, Any] = {
        "format": DCOCO_SIDECAR_MAGIC.decode("ascii"),
        "version": DCOCO_SIDECAR_VERSION,
        "path": canonical_out,
        "sha256": file_sha256,
        "header_payload_sha256": embedded_digest.hex(),
        "size": file_size,
        "hidden_size": hidden_size,
        "top_k": top_k,
        "rank": DCOCO_RANK,
        "mlp_hidden": DCOCO_MLP_HIDDEN,
        "ln_eps": ln_eps,
        "param_count": expected_param_count(hidden_size),
        "payload_bytes": payload_bytes,
        "training": training_metrics,
        "env": {
            DCOCO_SIDECAR_ENV: canonical_out,
            DCOCO_SIDECAR_SHA256_ENV: file_sha256,
            DCOCO_SIDECAR_SIZE_ENV: str(file_size),
        },
    }
    if reference_check is not None:
        receipt["reference_check"] = reference_check

    if receipt_path is not None:
        _atomic_write_bytes(
            receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
    return receipt


def build_reference_check_payload(
    weights: dict[str, np.ndarray],
    hidden_size: int,
    top_k: int,
    ln_eps: float = DCOCO_DEFAULT_LN_EPS,
    seed: int = 9001,
) -> dict[str, Any]:
    """Build a compact deterministic sample + BF16-rounded scores for cross-language verification."""
    gamma = 2
    vocab_size = max(top_k * 2, 12)
    rng = np.random.default_rng(seed)
    embed_tokens = rng.normal(0.0, 0.5, size=(vocab_size, hidden_size)).astype(np.float32)
    hidden_states = rng.normal(0.0, 0.5, size=(gamma + 1, hidden_size)).astype(np.float32)
    anchor_id = 1
    anchor_logit = -0.375
    candidates = np.array(
        [[(p * top_k + k) % vocab_size for k in range(top_k)] for p in range(gamma)],
        dtype=np.int64,
    )
    unary = np.linspace(-0.25, -1.75, gamma * top_k, dtype=np.float32).reshape(gamma, top_k)
    full_scores = eval_single_sample_full_scores_np(
        embed_tokens=embed_tokens,
        hidden_states=hidden_states,
        anchor_id=anchor_id,
        anchor_logit=anchor_logit,
        candidates=candidates,
        unary=unary,
        weights=weights,
        ln_eps=ln_eps,
        bf16_round=True,
    )
    # Flatten finite scores in exact row-major [p, a, b] order (using null/sentinel for -inf unused rows).
    flat_scores: list[float | None] = [
        None if not math.isfinite(float(v)) else float(v) for v in full_scores.reshape(-1)
    ]
    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "gamma": gamma,
        "top_k": top_k,
        "anchor_id": anchor_id,
        "anchor_logit": anchor_logit,
        "embed_tokens": [float(v) for v in embed_tokens.reshape(-1)],
        "hidden_states": [float(v) for v in hidden_states.reshape(-1)],
        "candidates": [int(v) for v in candidates.reshape(-1)],
        "unary": [float(v) for v in unary.reshape(-1)],
        "expected_scores": flat_scores,
    }


def run_self_test() -> int:
    """Run deterministic CPU self-test exercising both NumPy and PyTorch (when installed)."""
    hidden_size = 32
    top_k = 4
    gamma = 4
    batch = generate_synthetic_rollouts(
        hidden_size=hidden_size,
        top_k=top_k,
        gamma=gamma,
        num_samples=96,
        vocab_size=32,
        seed=4242,
    )

    backends_to_test = ["numpy"]
    try:
        import torch  # noqa: F401
        backends_to_test.append("torch")
    except ImportError:
        pass

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="atlas_dcoco_selftest_") as tmpdir:
        tmp = Path(tmpdir)
        for backend in backends_to_test:
            trainer = train_torch if backend == "torch" else train_numpy
            weights_a, metrics_a = trainer(
                batch=batch, steps=35, lr=1.5e-2, weight_decay=1.0e-4, seed=1337
            )
            assert metrics_a["initial_zero_residual_max_abs"] == 0.0, (
                f"{backend}: zero-initialized residual must start at exact 0.0, "
                f"got {metrics_a['initial_zero_residual_max_abs']}"
            )
            assert metrics_a["final_loss"] < metrics_a["initial_loss"] * 0.35, (
                f"{backend}: training loss did not converge strongly: {metrics_a}"
            )
            assert metrics_a["final_acc"] >= 0.90, (
                f"{backend}: contextual accuracy too low: {metrics_a}"
            )

            out_a = tmp / f"dcoco_{backend}_a.atldc"
            receipt_a_path = tmp / f"dcoco_{backend}_a.json"
            receipt_a = export_sidecar_and_receipt(
                weights=weights_a,
                hidden_size=hidden_size,
                top_k=top_k,
                out_path=out_a,
                receipt_path=receipt_a_path,
                training_metrics=metrics_a,
            )

            # Verify determinism across repeated run with identical seed.
            weights_b, _ = trainer(
                batch=batch, steps=35, lr=1.5e-2, weight_decay=1.0e-4, seed=1337
            )
            out_b = tmp / f"dcoco_{backend}_b.atldc"
            receipt_b = export_sidecar_and_receipt(
                weights=weights_b,
                hidden_size=hidden_size,
                top_k=top_k,
                out_path=out_b,
                receipt_path=None,
                training_metrics=metrics_a,
            )
            assert receipt_a["sha256"] == receipt_b["sha256"], (
                f"{backend}: non-deterministic ATLDC001 export"
            )
            assert out_a.read_bytes() == out_b.read_bytes()

            # Verify round-trip parse & tamper rejection.
            raw = out_a.read_bytes()
            header, decoded = parse_and_verify_atldc001(raw)
            assert header["file_sha256"] == receipt_a["sha256"]
            assert header["size"] == receipt_a["size"]
            for name in TENSOR_ORDER:
                assert np.array_equal(decoded[name], round_bf16_f32_np(weights_a[name]))

            tampered_hdr = bytearray(raw)
            tampered_hdr[12] ^= 0x01
            try:
                parse_and_verify_atldc001(bytes(tampered_hdr))
                raise AssertionError(f"{backend}: header tamper was not rejected")
            except ValueError:
                pass

            tampered_body = bytearray(raw)
            tampered_body[-1] ^= 0x01
            try:
                parse_and_verify_atldc001(bytes(tampered_body))
                raise AssertionError(f"{backend}: payload tamper was not rejected")
            except ValueError:
                pass

            # Verify .npz rollout ingestion round-trip once (on numpy pass).
            if backend == "numpy":
                npz_path = tmp / "rollouts.npz"
                np.savez(
                    npz_path,
                    embed_tokens=batch.embed_tokens,
                    hidden_states=batch.hidden_states,
                    anchor_ids=batch.anchor_ids,
                    anchor_logits=batch.anchor_logits,
                    drafter_topk_ids=batch.candidates,
                    drafter_topk_probs=np.exp(batch.unary),
                    target_tokens=np.take_along_axis(
                        batch.candidates, batch.target_slot_idx[:, :, None], axis=-1
                    ).squeeze(-1),
                )
                reloaded = load_rollouts(npz_path)
                assert reloaded.num_samples == batch.num_samples
                assert np.allclose(reloaded.unary, batch.unary, atol=1.0e-5)

            results.append(
                {
                    "backend": backend,
                    "initial_loss": metrics_a["initial_loss"],
                    "final_loss": metrics_a["final_loss"],
                    "initial_acc": metrics_a["initial_acc"],
                    "final_acc": metrics_a["final_acc"],
                    "sha256": receipt_a["sha256"],
                    "size": receipt_a["size"],
                }
            )

    print(json.dumps({"self_test": "PASS", "backends": results}, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline trainer & ATLDC001 sidecar exporter for Flux-DCOCO-v2"
    )
    parser.add_argument("--rollouts", type=Path, help="Input rollout archive (.npz or .json)")
    parser.add_argument("--out", type=Path, help="Output ATLDC001 sidecar file path")
    parser.add_argument("--receipt", type=Path, help="Output JSON verification receipt path")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for synthetic mode")
    parser.add_argument("--top-k", type=int, default=8, help="Shortlist K (1..=16) for synthetic mode")
    parser.add_argument("--gamma", type=int, default=4, help="Draft horizon gamma for synthetic mode")
    parser.add_argument("--samples", type=int, default=128, help="Synthetic rollout count")
    parser.add_argument("--steps", type=int, default=40, help="AdamW training steps")
    parser.add_argument("--lr", type=float, default=1.0e-2, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1.0e-4, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=20261127, help="Deterministic RNG seed")
    parser.add_argument(
        "--backend",
        choices=("auto", "torch", "numpy"),
        default="auto",
        help="Training backend (CPU-only torch or pure-NumPy analytic backprop)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic CPU self-test (both NumPy and PyTorch when installed)",
    )
    parser.add_argument(
        "--emit-test-fixture",
        type=Path,
        help="Train a deterministic small fixture and emit ATLDC001 + reference scores in receipt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    if args.emit_test_fixture is not None:
        out_path = args.emit_test_fixture
        receipt_path = args.receipt or out_path.with_suffix(".json")
        batch = generate_synthetic_rollouts(
            hidden_size=args.hidden_size,
            top_k=args.top_k,
            gamma=args.gamma,
            num_samples=max(args.samples, 64),
            vocab_size=max(args.top_k * 4, 24),
            seed=args.seed,
        )
        backend = select_backend(args.backend)
        trainer = train_torch if backend == "torch" else train_numpy
        weights, metrics = trainer(
            batch=batch,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        ref_check = build_reference_check_payload(
            weights=weights,
            hidden_size=batch.hidden_size,
            top_k=batch.top_k,
            seed=args.seed + 7,
        )
        receipt = export_sidecar_and_receipt(
            weights=weights,
            hidden_size=batch.hidden_size,
            top_k=batch.top_k,
            out_path=out_path,
            receipt_path=receipt_path,
            training_metrics=metrics,
            reference_check=ref_check,
        )
        print(json.dumps({"fixture_emitted": receipt["path"], "sha256": receipt["sha256"], "size": receipt["size"]}))
        return 0

    if args.out is None:
        raise SystemExit("error: --out is required unless --self-test or --emit-test-fixture is set")

    if args.rollouts is not None:
        batch = load_rollouts(args.rollouts)
    else:
        batch = generate_synthetic_rollouts(
            hidden_size=args.hidden_size,
            top_k=args.top_k,
            gamma=args.gamma,
            num_samples=args.samples,
            vocab_size=max(args.top_k * 4, 32),
            seed=args.seed,
        )

    backend = select_backend(args.backend)
    trainer = train_torch if backend == "torch" else train_numpy
    weights, metrics = trainer(
        batch=batch,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    receipt_path = args.receipt or args.out.with_suffix(".receipt.json")
    receipt = export_sidecar_and_receipt(
        weights=weights,
        hidden_size=batch.hidden_size,
        top_k=batch.top_k,
        out_path=args.out,
        receipt_path=receipt_path,
        training_metrics=metrics,
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
