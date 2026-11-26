"""Generate delta_rule_sm120_nf8.py: FlashInfer 0.6.14's SM120 chunked GDN
prefill kernel with the gated RMS norm + static-scale e4m3 quantization of
Flash-Next's out-projection activation fused into the O store warp.

The math warp groups, loads and pipelines are untouched. Only the STORE_O warp
changes: instead of TMA-storing each 16-bit O tile (D x BLK_Q) it reads the
tile from shared memory, and for every token row computes
    v = bf16(o * rsqrt(mean(o^2) + eps) * w / (1 + exp(-z)))
    out8 = e4m3(v * 448 / (sqrt(D) * max|w|))
(the arithmetic of kernels/.../qwen4_gdn_norm_fp8.cu with a null amax), and
writes 8 e4m3 bytes per lane. z rows are prefetched into L2 one block ahead.
The LDST warp group's register budget rises from 24 to LDST_REGS.

Usage: python make_variant.py <flashinfer delta_rule_sm120.py> <out.py>
"""

import re
import sys

src_path, out_path = sys.argv[1], sys.argv[2]
s = open(src_path).read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} x {old[:70]!r}, found {n}")
    s = s.replace(old, new)


# Absolute imports (the variant lives outside the flashinfer package).
P = "flashinfer.gdn_kernels.delta_rule_dsl"
sub("from ...utils import", "from flashinfer.utils import")
for mod in ("alpha", "collective_store_tma", "custom_compile_cache", "collective_inverse_hmma", "helpers", "schedule"):
    sub(f"from .{mod} import", f"from {P}.{mod} import")
sub("import cuda.bindings.driver as cuda\n", "import cuda.bindings.driver as cuda\n" + "import os\nfrom fi_gdn_nf8_store import CollectiveStoreNormFp8\n\n"
    "LDST_REGS = int(os.environ.get(\"NF8_LDST_REGS\", \"40\"))\n")

s = s.replace("_FullyFusedDeltaRuleSm120", "_FullyFusedDeltaRuleSm120NormFp8")
sub("        load_registers = 40 - 2 * reg_alloc_granularity\n", "        load_registers = LDST_REGS\n")
sub('            "alpha_beta_stage",\n        )', '            "alpha_beta_stage",\n            "norm_fp8_ldst_regs",\n        )')
sub("        self.alpha_beta_stage = 2\n", "        self.alpha_beta_stage = 2\n        self.norm_fp8_ldst_regs = LDST_REGS\n")

# __call__ and kernel signatures.
sub(
    "        g_o: cute.Tensor,\n        g_alpha: cute.Tensor,",
    "        g_o: cute.Tensor,\n        g_z: cute.Tensor,\n        g_normw: cute.Tensor,\n"
    "        g_out8: cute.Tensor,\n        g_scale_out: cute.Tensor,\n        g_alpha: cute.Tensor,",
)
sub(
    "        cu_seqlens: cute.Tensor,\n        scale: cutlass.Float32,\n        num_q_heads: cutlass.Int32,\n"
    "        num_k_heads: cutlass.Int32,\n        num_v_heads: cutlass.Int32,\n        num_sab_heads: cutlass.Int32,\n"
    "        num_seqs: cutlass.Int32,\n        total_checkpoints: cutlass.Int32,\n"
    "        checkpoint_every_n_tokens: cutlass.Int32,\n        grid_x: cutlass.Int32,",
    "        cu_seqlens: cute.Tensor,\n        scale: cutlass.Float32,\n        norm_eps: cutlass.Float32,\n"
    "        num_q_heads: cutlass.Int32,\n"
    "        num_k_heads: cutlass.Int32,\n        num_v_heads: cutlass.Int32,\n        num_sab_heads: cutlass.Int32,\n"
    "        num_seqs: cutlass.Int32,\n        total_checkpoints: cutlass.Int32,\n"
    "        checkpoint_every_n_tokens: cutlass.Int32,\n        grid_x: cutlass.Int32,",
)
sub(
    "            tma_atom_o,\n            tma_tensor_o,\n            g_state,",
    "            tma_atom_o,\n            tma_tensor_o,\n            g_z,\n            g_normw,\n            g_out8,\n"
    "            g_scale_out,\n            g_state,",
)
sub(
    "            cu_seqlens,\n            scale,\n            num_q_heads,",
    "            cu_seqlens,\n            scale,\n            norm_eps,\n            num_q_heads,",
)
sub(
    "        tma_atom_o: cute.CopyAtom,\n        tma_tensor_o: cute.Tensor,\n        g_state: cute.Tensor,",
    "        tma_atom_o: cute.CopyAtom,\n        tma_tensor_o: cute.Tensor,\n        g_z: cute.Tensor,\n"
    "        g_normw: cute.Tensor,\n        g_out8: cute.Tensor,\n        g_scale_out: cute.Tensor,\n"
    "        g_state: cute.Tensor,",
)
sub(
    "        cu_seqlens: cute.Tensor,\n        scale: cutlass.Float32,\n        num_q_heads: cutlass.Int32,\n"
    "        num_k_heads: cutlass.Int32,\n        num_v_heads: cutlass.Int32,\n        num_sab_heads: cutlass.Int32,\n"
    "        num_seqs: cutlass.Int32,\n        total_checkpoints: cutlass.Int32,\n"
    "        checkpoint_every_n_tokens: cutlass.Int32,\n    ):",
    "        cu_seqlens: cute.Tensor,\n        scale: cutlass.Float32,\n        norm_eps: cutlass.Float32,\n"
    "        num_q_heads: cutlass.Int32,\n"
    "        num_k_heads: cutlass.Int32,\n        num_v_heads: cutlass.Int32,\n        num_sab_heads: cutlass.Int32,\n"
    "        num_seqs: cutlass.Int32,\n        total_checkpoints: cutlass.Int32,\n"
    "        checkpoint_every_n_tokens: cutlass.Int32,\n    ):",
)

# The store warp.
sub(
    """                CollectiveStoreTma(self.BLK_Q, self.D).run(
                    sO,
                    tma_atom_o,
                    tma_tensor_o,
                    g_tensormaps,
                    o_pipeline,
                    num_blocks,
                    work_desc,
                    num_seqs,
                    self.o_stage,
                    num_q_heads,
                    num_v_heads,
                )""",
    """                CollectiveStoreNormFp8(self.BLK_Q, self.D).run(
                    sO,
                    g_z,
                    g_normw,
                    g_out8,
                    g_scale_out,
                    norm_eps,
                    o_pipeline,
                    num_blocks,
                    work_desc,
                    self.o_stage,
                    num_q_heads,
                    num_v_heads,
                )""",
)

# NF8_NORM_WARPS=2: the LOAD_ALPHA warp also loads beta, and the LOAD_BETA warp
# becomes a second norm warp (each takes half of every O tile's rows).
NORM_WARPS = int(__import__("os").environ.get("NF8_NORM_WARPS", "1"))
if NORM_WARPS == 2:
    sub(
        "        o_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)\n",
        "        o_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 64)\n",
    )
    sub(
        """                    num_q_heads,
                    num_v_heads,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:
                self.run_load_beta_role(
                    sBeta,
                    g_beta,
                    beta_pipeline,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_ALPHA:
                self.run_load_alpha_role(
                    sAlpha,
                    g_alpha,
                    alpha_pipeline,
                    scale,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                )""",
        """                    num_q_heads,
                    num_v_heads,
                    cutlass.Int32(0),
                    2,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:
                CollectiveStoreNormFp8(self.BLK_Q, self.D).run(
                    sO,
                    g_z,
                    g_normw,
                    g_out8,
                    g_scale_out,
                    norm_eps,
                    o_pipeline,
                    num_blocks,
                    work_desc,
                    self.o_stage,
                    num_q_heads,
                    num_v_heads,
                    cutlass.Int32(1),
                    2,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_ALPHA:
                self.run_load_beta_role(
                    sBeta,
                    g_beta,
                    beta_pipeline,
                    num_blocks,
                    work_desc.tok_offset,
                    tok_end,
                    work_desc.o_head_idx(num_q_heads, num_v_heads),
                    num_sab_heads,
                    sAlpha,
                    g_alpha,
                    alpha_pipeline,
                    scale,
                )""",
    )
    # run_load_beta_role(..., sAlpha, g_alpha, alpha_pipeline, scale): per block,
    # beta then alpha from one warp.
    sub(
        """        num_sab_heads: cutlass.Int32,
    ):
        beta_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )""",
        """        num_sab_heads: cutlass.Int32,
        sAlpha: cute.Tensor,
        g_alpha: cute.Tensor,
        alpha_pipeline,
        scale: cutlass.Float32,
    ):
        beta_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        alpha_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )""",
    )
    sub(
        """                cute.arch.fence_view_async_shared()
                beta_pipeline.producer_commit(beta_producer_state)
                beta_producer_state.advance()
""",
        """                cute.arch.fence_view_async_shared()
                beta_pipeline.producer_commit(beta_producer_state)
                beta_producer_state.advance()
            if cutlass.const_expr(self.needs_alpha):
                alpha_pipeline.producer_acquire(alpha_producer_state)
                cute.arch.fence_view_async_shared()
                self.load_alpha(
                    sAlpha,
                    g_alpha,
                    tok_start + blk * cutlass.Int32(self.BLK_Q),
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    alpha_producer_state.index,
                )
                AlphaProcessor().run(
                    sAlpha[None, None, alpha_producer_state.index], scale
                )
                cute.arch.fence_view_async_shared()
                alpha_pipeline.producer_commit(alpha_producer_state)
                alpha_producer_state.advance()
""",
    )
    sub('            "norm_fp8_ldst_regs",\n', '            "norm_fp8_ldst_regs",\n            "norm_fp8_warps",\n')
    sub("        self.norm_fp8_ldst_regs = LDST_REGS\n", "        self.norm_fp8_ldst_regs = LDST_REGS\n        self.norm_fp8_warps = 2\n")
else:
    sub(
        """                    num_q_heads,
                    num_v_heads,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:""",
        """                    num_q_heads,
                    num_v_heads,
                    cutlass.Int32(0),
                    1,
                )
            elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:""",
    )

# Public API.
sub("def delta_rule_prefill_dsl(\n", "def delta_rule_prefill_dsl_nf8(\n")
sub(
    "    checkpoint_every_n_tokens: int = 0,\n):\n",
    "    checkpoint_every_n_tokens: int = 0,\n    z: torch.Tensor = None,  # (T, num_o_heads*D) bf16 view, any row stride\n"
    "    normw: torch.Tensor = None,  # (D,) bf16\n    out8: torch.Tensor = None,  # (T, num_o_heads*D) uint8, contiguous\n"
    "    scale_out: torch.Tensor = None,  # (1,) fp32\n    norm_eps: float = 1e-6,\n):\n",
)
sub(
    "    cu_cute = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()\n",
    "    cu_cute = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()\n"
    "    z_cute = from_dlpack(z, assumed_align=16).mark_layout_dynamic(leading_dim=1)\n"
    "    normw_cute = from_dlpack(normw, assumed_align=16).mark_layout_dynamic()\n"
    "    out8_cute = from_dlpack(out8.view(torch.int32), assumed_align=16).mark_layout_dynamic(leading_dim=1)\n"
    "    scale_out_cute = from_dlpack(scale_out, assumed_align=4).mark_layout_dynamic()\n",
)
sub(
    "        o_cute,\n        alpha_cute,",
    "        o_cute,\n        z_cute,\n        normw_cute,\n        out8_cute,\n        scale_out_cute,\n        alpha_cute,",
)
sub(
    "        cutlass.Float32(scale),\n        cutlass.Int32(num_q_heads),",
    "        cutlass.Float32(scale),\n        cutlass.Float32(norm_eps),\n        cutlass.Int32(num_q_heads),",
)
open(out_path, "w").write(s)
print("wrote", out_path)
