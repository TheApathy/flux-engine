"""STORE_O warp for the norm-fp8 FlashInfer GDN variant (see make_variant.py).

One warp per CTA (one v-head). Each BLK_Q x D O tile in shared memory is
consumed two token rows at a time: half-warp h (lanes 16h..16h+15) owns row
2p+h and lane l of it owns columns 8(l%16)..+7 of the head:
    v = bf16(o * rsqrt(sum(o^2) / D + eps) * w / (1 + exp(-z)))
    out8 = e4m3(v * 448 / (sqrt(D) * max|w|))    (static bound, never clips)
then two 32-bit words of e4m3 per lane. The arithmetic is that of
kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_gdn_norm_fp8.cu with a null amax;
the sum of squares is reduced in a different order (per-lane 8, then a
16-lane butterfly), so a byte can differ in its last bit.

z (16 B per lane and row) is loaded through ld.global.nc one row pair ahead
(two static register slots) and the next block's z rows are prefetched into
L2 when a block starts, so the warp is not bound by one DRAM latency per row.

v3: fast-math sigmoid/rsqrt (ex2/rcp/rsqrt.approx, as the CUDA reference under
--use_fast_math), 16-byte vector smem reads of each lane's 8 O values, and row
addresses by stride arithmetic.
"""

import math

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

PREFETCH_NEXT_BLOCK = True


@dsl_user_op
def _addr(tensor: cute.Tensor, coord, *, loc=None, ip=None) -> Int64:
    p = tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    return Int64(llvm.ptrtoint(T.i64(), p.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _ld_nc_v4(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_v2(addr: Int64, a: Int32, b: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prefetch_l2(addr: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bf16_lo(u: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), (Uint32(u) << Uint32(16)).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _bf16_hi(u: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), (Uint32(u) & Uint32(0xFFFF0000)).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def bf16_round(x: Float32, *, loc=None, ip=None) -> Float32:
    bits = llvm.inline_asm(
        T.i32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 t, zz; cvt.rn.bf16.f32 t, $1; mov.b16 zz, 0; mov.b32 $0, {zz, t}; }",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(llvm.bitcast(T.f32(), bits, loc=loc, ip=ip))


@dsl_user_op
def e4m3x4(x0: Float32, x1: Float32, x2: Float32, x3: Float32, *, loc=None, ip=None) -> Int32:
    # cvt.e4m3x2 d, a, b puts cvt(a) in the high byte; mov.b32 {lo, hi}.
    bits = llvm.inline_asm(
        T.i32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (x0, x1, x2, x3)],
        "{ .reg .b16 l, h; cvt.rn.satfinite.e4m3x2.f32 l, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 h, $4, $3; mov.b32 $0, {l, h}; }",
        "=r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(bits)


LOG2E = 1.4426950408889634


class CollectiveStoreNormFp8:
    def __init__(self, blk_q: int, d: int):
        self.BLK_Q = blk_q
        self.D = d

    @cute.jit
    def load_z(self, zbuf: cute.Tensor, slot: cutlass.Constexpr, addr: Int64):
        z = _ld_nc_v4(addr)
        for i in cutlass.range_constexpr(4):
            zbuf[slot, i] = z[i]

    @cute.jit
    def row(
        self,
        sOv: cute.Tensor,
        lane16: Int32,
        r: Int32,
        valid,
        zbuf: cute.Tensor,
        slot: cutlass.Constexpr,
        w: cute.Tensor,
        inv_d: Float32,
        norm_eps: Float32,
        qs: Float32,
        out_addr: Int64,
    ):
        src = sOv[(None, None), (lane16, r)]
        oh = cute.make_rmem_tensor_like(src)
        cute.autovec_copy(src, oh)
        o = cute.make_rmem_tensor(8, Float32)
        ss = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            o[i] = Float32(oh[i])
            ss = ss + o[i] * o[i]
        for k in cutlass.range_constexpr(4):
            ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=8 >> k)
        rms = cute.math.rsqrt(ss * inv_d + norm_eps, approx=True)
        one = Float32(1.0)
        nl2e = Float32(-LOG2E)
        q = cute.make_rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(4):
            u = zbuf[slot, i]
            for h in cutlass.range_constexpr(2):
                zz = _bf16_lo(u) if h == 0 else _bf16_hi(u)
                j = 2 * i + h
                # fast-math sigmoid: (o*rms*w) * rcp(1 + exp(-z)), as __fdividef/__expf.
                sg = cute.math.rcp(one + cute.math.exp2(zz * nl2e, fastmath=True), approx=True)
                q[j] = bf16_round(o[j] * rms * w[j] * sg) * qs
        if valid:
            _st_v2(out_addr, e4m3x4(q[0], q[1], q[2], q[3]), e4m3x4(q[4], q[5], q[6], q[7]))

    @cute.jit
    def run(
        self,
        sO: cute.Tensor,  # (D, BLK_Q, o_stage) 16-bit, swizzled
        g_z: cute.Tensor,  # (T, HV*D) bf16, row stride QKVZ
        g_normw: cute.Tensor,  # (D,) bf16
        g_out8: cute.Tensor,  # (T, HV*D/4) int32 words of 4 e4m3
        g_scale_out: cute.Tensor,  # (1,) fp32
        norm_eps: Float32,
        o_pipeline,
        num_blocks: Int32,
        work_desc,
        o_stage: int,
        num_q_heads: Int32,
        num_v_heads: Int32,
        warp_slot: Int32,
        nwarps: cutlass.Constexpr = 1,
    ):
        # nwarps warps share each tile: warp s owns row pairs [s*ppw, (s+1)*ppw).
        ppw = self.BLK_Q // 2 // nwarps
        lane = cute.arch.lane_idx()
        half = lane // Int32(16)
        lane16 = lane % Int32(16)
        c0 = lane16 * Int32(8)
        o_head = work_desc.o_head_idx(num_q_heads, num_v_heads)
        col = o_head * Int32(self.D) + c0
        word_col = col // Int32(4)
        w = cute.make_rmem_tensor(8, Float32)
        wmax = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            w[i] = Float32(g_normw[c0 + Int32(i)])
            wmax = cute.math.max(wmax, cute.math.absf(w[i]))
        for k in cutlass.range_constexpr(5):
            wmax = cute.math.max(wmax, cute.arch.shuffle_sync_bfly(wmax, offset=1 << k))
        amax = cute.math.max(Float32(math.sqrt(self.D)) * wmax, Float32(1e-30))
        qs = Float32(448.0) / amax
        bx, _, _ = cute.arch.block_idx()
        if bx == Int32(0) and lane == Int32(0) and warp_slot == Int32(0):
            g_scale_out[0] = amax / Float32(448.0)
        inv_d = Float32(1.0 / self.D)
        seq_len = work_desc.seq_len
        tok0 = work_desc.tok_offset
        last = seq_len - Int32(1)
        # Row addresses by stride arithmetic (no layout math in the loop).
        z_base = _addr(g_z, (tok0, col))
        z_row = _addr(g_z, (tok0 + Int32(1), col)) - z_base
        zh_base = _addr(g_z, (tok0, o_head * Int32(self.D)))
        o_base = _addr(g_out8, (tok0, word_col))
        o_row = _addr(g_out8, (tok0 + Int32(1), word_col)) - o_base
        zbuf = cute.make_rmem_tensor((2, 4), Uint32)

        o_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, o_stage)
        for blk in cutlass.range(num_blocks, unroll=1):
            row0 = blk * Int32(self.BLK_Q)
            if cutlass.const_expr(PREFETCH_NEXT_BLOCK):
                nxt = row0 + Int32(self.BLK_Q)
                if nxt < seq_len:
                    for j in cutlass.range_constexpr(2 // nwarps):
                        t = cute.math.min(nxt + lane + Int32(32) * (warp_slot + Int32(nwarps * j)), last)
                        za = zh_base + Int64(t) * z_row
                        _prefetch_l2(za)
                        _prefetch_l2(za + Int64(128))
            rows = seq_len - row0
            if rows > Int32(self.BLK_Q):
                rows = Int32(self.BLK_Q)
            rmax = rows - Int32(1)
            npairs = (rows + Int32(1)) // Int32(2)
            ps = warp_slot * Int32(ppw)
            pe = cute.math.min(npairs, ps + Int32(ppw))
            rend = cute.math.min(rows, pe * Int32(2))  # rows of this warp end here
            zb = z_base + Int64(row0) * z_row
            ob = o_base + Int64(row0) * o_row
            # z for the first pair while waiting for the tile.
            self.load_z(zbuf, 0, zb + Int64(cute.math.min(ps * Int32(2) + half, rmax)) * z_row)
            o_pipeline.consumer_wait(o_consumer_state)
            sOv = cute.zipped_divide(sO[None, None, o_consumer_state.index], (8, 1))
            for p in cutlass.range(ps, pe, 2, unroll=1):
                # pair p (slot 0) is loaded; fetch pair p+1 into slot 1.
                r0 = p * Int32(2) + half
                r1 = r0 + Int32(2)
                c1 = cute.math.min(r1, rmax)
                self.load_z(zbuf, 1, zb + Int64(c1) * z_row)
                c0r = cute.math.min(r0, rmax)
                self.row(sOv, lane16, c0r, r0 < rend, zbuf, 0, w, inv_d, norm_eps, qs, ob + Int64(c0r) * o_row)
                self.load_z(zbuf, 0, zb + Int64(cute.math.min(r1 + Int32(2), rmax)) * z_row)
                self.row(sOv, lane16, c1, r1 < rend, zbuf, 1, w, inv_d, norm_eps, qs, ob + Int64(c1) * o_row)
            o_pipeline.consumer_release(o_consumer_state)
            o_consumer_state.advance()
