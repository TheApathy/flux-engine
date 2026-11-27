#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""ISA/PTX & SASS binary-level formal contract verifier for GB10 (`sm_121a`).

Inspired by Anthropic + Theorem's compiled-ISA verification methodology: instead
of trusting high-level `.cu` C++ source or preprocessor macros, this verifier
compiles our critical Blackwell kernels with `nvcc` + `ptxas -arch=sm_121a -v`
and mechanically proves instruction-level and register/spill-level invariants
directly over the emitted `sm_121a` PTX and SASS assembly metadata:

1. Zero half-rate plain E4M3 `mma.sync` (`f32.e4m3.e4m3.f32` without
   `kind::mxf8f6f4.block_scale.scale_vec::1X`) in upgraded FP8 Tensor-Core kernels,
   and positive presence of full-rate block-scaled `kind::mxf8f6f4...ue8m0` MMAs.
2. Zero `.local` stack frame (`0 bytes stack frame`) and zero register spills
   (`0 bytes spill stores, 0 bytes spill loads`) across every entry point in
   our critical decode, speculative, MoE compact scan, DSpine, PLE hyperconnection,
   and GDN norm/FP8 kernels.
3. Positive emission of L2 cache-line discard instructions (`discard.global.L2` in
   `sm_121a` PTX and `CCTL.E.RML2` in disassembled `sm_121a` SASS) in Wave-18
   L2-discarding GDN norm/FP8 kernels, plus positive presence of required Wave-18
   entrypoints (e.g. `qwen4_ple_fuse_verify_rows_ragged`).
4. Zero illegal odd 64-bit uniform descriptor registers (`desc[UR<odd>]`) across
   all disassembled `sm_121a` SASS instructions (catches the CUDA 13.0 `ptxas`
   register-allocator bug where `LDGSTS [R+UR], desc[UR]` emits `desc[UR1]` and
   raises `CUDA_ERROR_ILLEGAL_INSTRUCTION` on GB10 hardware).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
_NVCC_ON_PATH = shutil.which("nvcc")
NVCC = Path(_NVCC_ON_PATH) if _NVCC_ON_PATH else Path("/usr/local/cuda/bin/nvcc")
_CUOBJDUMP_ON_PATH = shutil.which("cuobjdump")
CUOBJDUMP = Path(_CUOBJDUMP_ON_PATH) if _CUOBJDUMP_ON_PATH else (NVCC.parent / "cuobjdump")

# (relative kernel path, require_blockscaled_fp8_mma, forbid_plain_fp8_mma,
#  require_zero_spills, require_l2_discard, required_ptx_entries)
CRITICAL_KERNELS: tuple[tuple[str, bool, bool, bool, bool, tuple[str, ...]], ...] = (
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_qsa_prefill_attn_fp8.cu",
        True,
        True,
        True,
        False,
        (
            "qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe",
            "qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe_2cta",
        ),
    ),
    ("kernels/gb10/common/w3a16_gemm.cu", True, True, True, False, ()),
    ("kernels/gb10/common/sampler_topk_candidates.cu", False, False, True, False, ()),
    (
        "kernels/gb10/common/dflash2_selector.cu",
        False,
        False,
        True,
        False,
        ("dflash2_selector_walk_ngram_batched",),
    ),
    (
        "kernels/gb10/common/dcoco_selector.cu",
        False,
        False,
        True,
        False,
        (
            "dcoco_proj_fuse_bf16",
            "dcoco_pair_mlp_bf16",
            "dcoco_walk_and_export_q",
        ),
    ),
    ("kernels/gb10/common/dspine_bridge.cu", False, False, True, False, ()),
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_k32_v28.cu",
        False,
        False,
        True,
        True,
        (
            "qwen4_shared_w4a8_gemm_k32_v28_2stage",
            "qwen4_shared_w4a8_gemm_k32_v28_discard",
            "qwen4_moe_compact_t_gemm_gateup_k32_v28_m16_ts4e",
            "qwen4_moe_compact_t_gemm_k32_v28q_m16_ts4e",
            "qwen4_moe_compact_t_gemm_gateup_k32_v30_2cta",
            "qwen4_moe_compact_t_gemm_k32_v30q_2cta",
            "qwen4_moe_compact_t_gemm_gateup_k32_v30_2cta_ts4e",
            "qwen4_moe_compact_t_gemm_k32_v30q_2cta_ts4e",
            "qwen4_moe_compact_t_gemm_gateup_k32_v31_ws",
            "qwen4_moe_compact_t_gemm_k32_v31_ws",
            "qwen4_moe_compact_t_gemm_k32_v31q_ws",
            "qwen4_moe_compact_t_gemm_gateup_k32_v31_ws_ts4e",
            "qwen4_moe_compact_t_gemm_k32_v31_ws_ts4e",
            "qwen4_moe_compact_t_gemm_k32_v31q_ws_ts4e",
        ),
    ),
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_shared_w4a8_2stage.cu",
        True,
        True,
        True,
        False,
        ("qwen4_shared_w4a8_gateup_k32_v28_2stage",),
    ),
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_attn_prep_fused_fp8.cu",
        False,
        False,
        True,
        False,
        (
            "qwen4_attn_prep_fused_fp8_kv_only",
            "qwen4_attn_prep_fused_fp8_kv_only_k8",
            "qwen4_attn_prep_fused_fp8_q_only",
        ),
    ),
    ("kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_plan_k32_v8.cu", False, False, True, False, ()),
    ("kernels/gb10/common/quantize_bf16_to_nvfp4.cu", False, False, True, False, ()),
    ("kernels/gb10/common/paged_decode_attn_fp8.cu", False, False, True, False, ()),
    ("kernels/gb10/common/paged_decode_attn_nvfp4.cu", False, False, True, False, ()),
    ("kernels/gb10/common/paged_decode_attn_kgamma_nvfp4.cu", False, False, True, False, ()),
    ("kernels/gb10/common/paged_decode_attn_turbo4.cu", False, False, True, False, ()),
    ("kernels/gb10/common/paged_decode_attn_fp8k_turbo4v.cu", False, False, True, False, ()),
    (
        "kernels/gb10/common/qwen4_hyper.cu",
        False,
        False,
        True,
        False,
        (
            "qwen4_ple_fuse_verify_rows_ragged",
            "qwen4_hc_inject_group_norm_v2",
        ),
    ),
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_gdn_norm_fp8.cu",
        False,
        False,
        True,
        True,
        (
            "qwen4_gdn_norm_fp8_bf16_v2",
            "qwen4_gdn_norm_fp8_bf16_v2_strip",
            "qwen4_gdn_norm_fp8_f16_v2",
            "qwen4_gdn_norm_fp8_f16_v2_strip",
        ),
    ),
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_hc_combine_gemm.cu",
        False,
        False,
        True,
        True,
        (
            "qwen4_hc_inject_upmix_ws_gatepre",
            "qwen4_hc_inject_upmix_ws_gatepre_amax",
            "qwen4_hc_inject_upmix_ws_l2h_gatepre",
            "qwen4_hc_inject_upmix_ws_l2h_gatepre_amax",
            "qwen4_hc_combine_moe_upmix_mx_gatepre",
            "qwen4_hc_combine_moe_upmix_mx_gatepre_amax",
            "qwen4_hc_inject_upmix_bm64_w4",
            "qwen4_hc_inject_upmix_bm64_w4_amax",
            "qwen4_hc_inject_upmix_bm64_w4_mxfp8_gate",
            "qwen4_hc_inject_upmix_bm64_w4_mxfp8_gate_amax",
            "qwen4_hc_combine_inject_upmix_bm64_w4_mx_gatepre",
            "qwen4_hc_combine_inject_upmix_bm64_w4_mx_gatepre_amax",
            "qwen4_hc_inject_gemm_ws_gatepre_w4",
            "qwen4_hc_inject_gemm_ws_l2h_gatepre_w4",
            "qwen4_hc_combine_moe_gemm_gatepre_w4",
            "qwen4_hc_combine_moe_gemm_mx_gatepre_w4",
        ),
    ),
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_hc_upmix_roof.cu",
        False,
        False,
        True,
        True,
        (
            "qwen4_hc_discard_l2_ranges",
            "qwen4_hc_upmix_roof_t128_r4_mxfp8_gate",
            "qwen4_hc_upmix_roof_t128_r4_mxfp8_gate_amax",
            "qwen4_hc_upmix_fp8_act_from_bf16",
            "qwen4_hc_upmix_roof_t128_stream_mxfp8_gate",
            "qwen4_hc_upmix_roof_t128_stream_mxfp8_gate_amax",
            "qwen4_hc_upmix_roof_t128_r4_stream_mxfp8_gate",
            "qwen4_hc_upmix_roof_t128_r4_stream_mxfp8_gate_amax",
            "qwen4_hc_upmix_roof_t128_w4_stream_mxfp8_gate",
            "qwen4_hc_upmix_roof_t128_w4_stream_mxfp8_gate_amax",
        ),
    ),
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/w4a16_gemm.cu",
        False,
        False,
        True,
        False,
        (
            "w4a16_gemm_coalesced_m32",
            "w4a16_gemm_coalesced_m64",
            "w4a16_gemm_coalesced_m128",
            "w4a16_gemm_coalesced_v2_n64_m64",
            "w4a16_gemm_coalesced_v2_n32_sk2_m64",
            "w4a16_gemm_coalesced_v2_n64_m80",
            "w4a16_gemm_coalesced_v2_n32_sk2_m80",
            "w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div",
            "w4a16_gemm_coalesced_n32_sk4_m16_dual_silu_div",
            "w4a16_gemm_coalesced_n16_sk4_m16_dual_silu_div_v2",
            "w4a16_shared_gateup_silu_t_n16_sk4_m16",
            "w4a16_shared_gateup_silu_t_n16_sk4_m32",
            "w4a16_shared_gateup_silu_t_n16_sk2_m32",
            "w4a16_shared_gateup_silu_t_n16_sk2_m64",
            "w4a16_shared_down_t_n32_m16",
            "w4a16_shared_down_t_n32_m32",
            "w4a16_shared_down_t_n32_m64",
            "w4a16_shared_down_t_n32_m80",
            "w4a16_shared_down_t_n32_m128",
            "qwen4_hc_upmix_coalesced_m80",
        ),
    ),
    (
        "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_gdn_fi_glue.cu",
        False,
        False,
        True,
        True,
        (
            "qwen4_conv1d_l2norm_f16_prefill_v3_strip",
            "qwen4_conv1d_l2norm_f16_prefill_v4_strip",
            "qwen4_conv1d_l2norm_bf16_prefill_v4_strip",
        ),
    ),
)

PLAIN_FP8_MMA_RE = re.compile(
    r"mma\.sync\.aligned\.(?:kind::f8f6f4\.)?m16n8k32\.row\.col\.f32\.e4m3\.e4m3\.f32\b"
)
BLOCKSCALED_FP8_MMA_RE = re.compile(
    r"mma\.sync\.aligned\.kind::mxf8f6f4\.block_scale\.scale_vec::1X\.m16n8k32\.row\.col\.f32\.e4m3\.e4m3\.f32\.ue8m0\b"
)
DISCARD_L2_PTX_RE = re.compile(
    r"^\s*(?:@!?%p\d+\s+)?discard\.global\.L2\b",
    re.MULTILINE,
)
CCTL_RML2_SASS_RE = re.compile(r"\bCCTL\.E\.RML2\b")
ODD_DESC_UR_SASS_RE = re.compile(r"\bdesc\[UR\d*[13579]\]")
SPILL_LINE_RE = re.compile(
    r"(\d+)\s+bytes\s+stack\s+frame,\s*(\d+)\s+bytes\s+spill\s+stores,\s*(\d+)\s+bytes\s+spill\s+loads"
)


@dataclass(frozen=True)
class KernelIsaProof:
    kernel: str
    entries_checked: int
    blockscaled_fp8_mmas: int
    l2_discard_ptx_ops: int
    cctl_rml2_sass_ops: int
    max_stack_bytes: int
    max_spill_stores: int
    max_spill_loads: int
    status: str


def verify_all_ptx_isa() -> list[KernelIsaProof]:
    if not NVCC.is_file():
        raise RuntimeError(f"nvcc not found at {NVCC}")

    results: list[KernelIsaProof] = []
    with tempfile.TemporaryDirectory(prefix="atlas-ptx-isa-") as tmp:
        tmpdir = Path(tmp)
        for (
            rel,
            req_bs_mma,
            forbid_plain_mma,
            req_zero_spills,
            req_l2_discard,
            req_entries,
        ) in CRITICAL_KERNELS:
            src = ROOT / rel
            ptx_out = tmpdir / (src.stem + ".ptx")
            cubin_out = tmpdir / (src.stem + ".cubin")

            # 1. Emit sm_121a PTX
            ptx_cmd = [
                str(NVCC),
                "-O3",
                "--fmad=false",
                "-arch=compute_121a",
                "-ptx",
                f"-I{src.parent}",
                f"-I{ROOT / 'kernels/gb10/common'}",
                str(src),
                "-o",
                str(ptx_out),
            ]
            proc_ptx = subprocess.run(ptx_cmd, capture_output=True, text=True, check=False)
            if proc_ptx.returncode != 0:
                raise AssertionError(f"PTX compile failed for {rel}:\n{proc_ptx.stderr}")
            ptx_text = ptx_out.read_text()

            plain_hits = len(PLAIN_FP8_MMA_RE.findall(ptx_text))
            bs_hits = len(BLOCKSCALED_FP8_MMA_RE.findall(ptx_text))
            l2_discard_ptx_hits = len(DISCARD_L2_PTX_RE.findall(ptx_text))

            if forbid_plain_mma and plain_hits != 0:
                raise AssertionError(f"{rel}: found {plain_hits} forbidden half-rate plain FP8 MMA instructions in PTX!")
            if req_bs_mma and bs_hits == 0:
                raise AssertionError(f"{rel}: expected >0 block-scaled kind::mxf8f6f4 FP8 MMA instructions in PTX, found 0!")
            if req_l2_discard and l2_discard_ptx_hits == 0:
                raise AssertionError(f"{rel}: expected >0 discard.global.L2 instructions in PTX, found 0!")
            for entry_name in req_entries:
                if not re.search(rf"\.visible\s+\.entry\s+{re.escape(entry_name)}\b", ptx_text):
                    raise AssertionError(f"{rel}: required entrypoint `{entry_name}` missing from compiled sm_121a PTX!")

            # 2. Assemble sm_121a SASS cubin with ptxas -v to inspect hardware stack & spill stats
            cubin_cmd = [
                str(NVCC),
                "-O3",
                "--fmad=false",
                "-gencode",
                "arch=compute_121a,code=sm_121a",
                "--ptxas-options=-v",
                "-c",
                f"-I{src.parent}",
                f"-I{ROOT / 'kernels/gb10/common'}",
                str(src),
                "-o",
                str(cubin_out),
            ]
            proc_cubin = subprocess.run(cubin_cmd, capture_output=True, text=True, check=False)
            if proc_cubin.returncode != 0:
                raise AssertionError(f"SASS cubin compile failed for {rel}:\n{proc_cubin.stderr}")

            spill_matches = SPILL_LINE_RE.findall(proc_cubin.stderr)
            if not spill_matches:
                raise AssertionError(f"{rel}: ptxas -v emitted no entry function stats:\n{proc_cubin.stderr}")

            max_stack = max(int(m[0]) for m in spill_matches)
            max_stores = max(int(m[1]) for m in spill_matches)
            max_loads = max(int(m[2]) for m in spill_matches)
            if req_zero_spills and (max_stack != 0 or max_stores != 0 or max_loads != 0):
                raise AssertionError(
                    f"{rel}: non-zero stack/spills on sm_121a SASS: stack={max_stack}, "
                    f"spill_stores={max_stores}, spill_loads={max_loads}\n{proc_cubin.stderr}"
                )

            if not CUOBJDUMP.is_file():
                raise RuntimeError(f"cuobjdump not found at {CUOBJDUMP}")
            proc_sass = subprocess.run(
                [str(CUOBJDUMP), "--dump-sass", str(cubin_out)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc_sass.returncode != 0:
                raise AssertionError(f"cuobjdump --dump-sass failed for {rel}:\n{proc_sass.stderr}")
            odd_desc_hits = ODD_DESC_UR_SASS_RE.findall(proc_sass.stdout)
            if odd_desc_hits:
                raise AssertionError(
                    f"{rel}: found {len(odd_desc_hits)} illegal odd desc[UR] SASS instructions "
                    f"(sm_121a ptxas LDGSTS/desc bug, sample={odd_desc_hits[:4]})!"
                )
            cctl_rml2_sass_hits = len(CCTL_RML2_SASS_RE.findall(proc_sass.stdout))
            if req_l2_discard and cctl_rml2_sass_hits == 0:
                raise AssertionError(f"{rel}: expected >0 CCTL.E.RML2 instructions in sm_121a SASS, found 0!")

            results.append(
                KernelIsaProof(
                    kernel=rel,
                    entries_checked=len(spill_matches),
                    blockscaled_fp8_mmas=bs_hits,
                    l2_discard_ptx_ops=l2_discard_ptx_hits,
                    cctl_rml2_sass_ops=cctl_rml2_sass_hits,
                    max_stack_bytes=max_stack,
                    max_spill_stores=max_stores,
                    max_spill_loads=max_loads,
                    status="PROVED (sm_121a PTX+SASS)",
                )
            )
    return results


def main() -> int:
    import json

    proofs = verify_all_ptx_isa()
    print(
        json.dumps(
            {
                "kernels_verified": len(proofs),
                "entries_verified": sum(p.entries_checked for p in proofs),
                "proofs": [asdict(p) for p in proofs],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
