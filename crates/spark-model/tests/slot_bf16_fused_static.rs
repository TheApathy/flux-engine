// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contract for ATLAS_QWEN4_PREFILL_MOE_SLOT_BF16_FUSED: the quad-lane bf16 -> MXFP8 x
//! quantiser (`qwen4_mxfp8_quant_rows_v16q4`) must encode exactly what `qwen4_mxfp8_quant_rows_v16`
//! encodes, and with the selector unset every W4A8 set must load the entry it loaded before the
//! selector existed. With the selector set, the V16 and V28 sets (the V28 module includes the V16
//! source) load the twin; every other set is refused. Standalone (std only): `rustc --test --edition 2024 slot_bf16_fused_static.rs`.

#[path = "../src/layers/moe/qwen4_xq_quant.rs"]
#[allow(dead_code)]
mod qwen4_xq_quant;

use qwen4_xq_quant::{XQ_QUANT_KERNEL_V16Q4, xq_quant_kernel};

const KERNEL: &str =
    include_str!("../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_k32_v16.cu");
const COMPACT: &str = include_str!("../src/layers/moe/qwen4_prefill_compact.rs");
const OPS: &str = include_str!("../src/layers/moe/qwen4_compact_ops.rs");
const SHARED: &str = include_str!("../src/layers/moe/qwen4_shared_v7.rs");
/// Every W4A8 set `load_wide` can select, with the module source its quantiser must come from.
const W4A8_SETS: [(&str, &str); 5] = [
    (
        "v10",
        include_str!(
            "../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_k32_v10.cu"
        ),
    ),
    (
        "v12",
        include_str!(
            "../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_k32_v12.cu"
        ),
    ),
    (
        "v14",
        include_str!(
            "../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_k32_v14.cu"
        ),
    ),
    ("v16", KERNEL),
    ("v28", KERNEL_V28),
];
const KERNEL_V28: &str =
    include_str!("../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_k32_v28.cu");
const KERNEL_V28_SHARED: &str = include_str!(
    "../../../kernels/gb10/qwen3.8-flash-next/nvfp4/detail/moe_compact/v28_shared.cuh"
);
const KERNEL_V28_Q4: &str =
    include_str!("../../../kernels/gb10/qwen3.8-flash-next/nvfp4/detail/moe_compact/v28_q4.cuh");

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let end_index = source[start_index..]
        .find(end)
        .map(|offset| start_index + offset)
        .unwrap_or_else(|| panic!("missing section end: {end}"));
    &source[start_index..end_index]
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn v16_body() -> &'static str {
    section(
        KERNEL,
        "extern \"C\" __global__ void qwen4_mxfp8_quant_rows_v16(",
        "extern \"C\" __global__ void qwen4_mxfp8_quant_rows_v16q4(",
    )
}

fn q4_body() -> &'static str {
    section(
        KERNEL,
        "extern \"C\" __global__ void qwen4_mxfp8_quant_rows_v16q4(",
        "// 64-byte smem rows",
    )
}

fn block_rule() -> &'static str {
    section(
        KERNEL,
        "__device__ __forceinline__ void v10_mxfp8_block(",
        "template <int BID>",
    )
}

/// CUDA `fmaxf` semantics: the non-NaN operand when one is NaN.
fn fmaxf(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Deterministic value stream covering zeros, subnormals, ordinary magnitudes, huge values and NaNs.
fn block_values(seed: u64, block: usize) -> [f32; 32] {
    let mut state = seed ^ (block as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut v = [0.0f32; 32];
    for slot in &mut v {
        let r = next();
        *slot = match r % 23 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits((r >> 32) as u32 & 0x007f_ffff), // positive subnormal
            3 => -f32::from_bits((r >> 32) as u32 & 0x007f_ffff),
            4 => f32::from_bits(0x7f00_0000 | ((r >> 32) as u32 & 0x000f_ffff)), // huge finite
            5 if block % 97 == 3 => f32::NAN,
            _ => {
                // bf16-representable value of either sign, like the real input rows.
                let bits = ((r >> 16) as u32 & 0xffff) << 16;
                let x = f32::from_bits(bits);
                if x.is_finite() { x } else { 1.0 }
            }
        };
    }
    v
}

fn serial_amax(v: &[f32; 32]) -> f32 {
    let mut amax = 0.0f32;
    for x in v {
        amax = fmaxf(amax, x.abs());
    }
    amax
}

/// The v16q4 reduction: four lanes fold their own eight magnitudes from 0, then xor-shuffle 1 and 2.
fn quad_amax(v: &[f32; 32]) -> f32 {
    let mut lane = [0.0f32; 4];
    for (part, acc) in lane.iter_mut().enumerate() {
        for i in 0..8 {
            *acc = fmaxf(*acc, v[part * 8 + i].abs());
        }
    }
    let mut step1 = [0.0f32; 4];
    for part in 0..4 {
        step1[part] = fmaxf(lane[part], lane[part ^ 1]);
    }
    let mut step2 = [0.0f32; 4];
    for part in 0..4 {
        step2[part] = fmaxf(step1[part], step1[part ^ 2]);
    }
    assert!(
        step2.iter().all(|x| x.to_bits() == step2[0].to_bits()),
        "quad lanes disagree on the block absmax"
    );
    step2[0]
}

#[test]
fn quad_absmax_tree_is_bit_identical_to_the_serial_chain() {
    for block in 0..40_000usize {
        let v = block_values(0x5107_0b16, block);
        let serial = serial_amax(&v);
        let quad = quad_amax(&v);
        assert_eq!(
            serial.to_bits(),
            quad.to_bits(),
            "block {block}: serial {serial:e} quad {quad:e}"
        );
        // The zero path (payload 0, scale byte 0) is decided by the same predicate in both kernels.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let (serial_zero, quad_zero) = (!(serial > 0.0), !(quad > 0.0));
        assert_eq!(serial_zero, quad_zero);
    }
}

#[test]
fn quad_lane_mapping_covers_every_block_byte_exactly_once() {
    // Pin q4_body()'s actual load/unpack/quant/store index expressions directly:
    // each of the 4 lanes (`part = lane & 3u`) loads 8 BF16 inputs at `src + b * 32 + part * 8`,
    // packs `v[4 * i + j]` into byte `j` of `qq[i]` (`i in 0..2, j in 0..4`), and stores the
    // 8 quantized bytes contiguously at `q + b * 32 + part * 8`.
    let q4 = compact(q4_body());
    for needle in [
        "part=lane&3u",
        "src+b*32+part*8",
        "for(inti=0;i<4;++i){v[2*i]=__uint_as_float(u[i]<<16);v[2*i+1]=__uint_as_float(u[i]&0xffff0000u);}",
        "for(inti=0;i<2;++i){unsignedwq=0;#pragmaunrollfor(intj=0;j<4;++j)wq|=static_cast<unsigned>(__nv_cvt_float_to_fp8(v[4*i+j]*inv,__NV_SATFINITE,__NV_E4M3))<<(8*j);qq[i]=wq;}",
        "*reinterpret_cast<uint2*>(q+b*32+part*8)=make_uint2(qq[0],qq[1]);",
    ] {
        assert!(
            q4.contains(needle),
            "q4_body() index contract drifted: {needle}"
        );
    }

    let mut seen = [0u8; 32];
    let mut expected_block_index = 0usize;
    for part in 0..4usize {
        for i in 0..2usize {
            for j in 0..4usize {
                let dst_byte = part * 8 + i * 4 + j;
                assert_eq!(dst_byte, expected_block_index);
                seen[dst_byte] += 1;
                expected_block_index += 1;
            }
        }
    }
    assert_eq!(expected_block_index, 32);
    assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
}

#[test]
fn quad_twin_entry_mapping_and_stores_are_pinned() {
    let q4 = compact(q4_body());
    for needle in [
        "lane=threadIdx.x&31u,quad=lane>>2,part=lane&3u",
        "base=warp*8;base<blocks;base+=warp_stride*8",
        "b=base+quad",
        "src+b*32+part*8",
        "__shfl_xor_sync(0xffffffffu,amax,1)",
        "__shfl_xor_sync(0xffffffffu,amax,2)",
        "*reinterpret_cast<uint2*>(q+b*32+part*8)=make_uint2(qq[0],qq[1])",
        "if(part==0)s[b]=static_cast<unsignedchar>(ss)",
        "if(amax>0.0f)",
    ] {
        assert!(q4.contains(needle), "v16q4 lost {needle}");
    }
    // Memory operations are predicated on `valid`; the shuffles are not (warp-uniform loop).
    assert!(q4.contains("if(valid){*reinterpret_cast<uint2*>"));
    assert!(!q4.contains("if(valid){amax"));
}

#[test]
fn quad_twin_keeps_the_v10_scale_and_rounding_rule() {
    let rule = compact(block_rule());
    let q4 = compact(q4_body());
    let v16 = compact(v16_body());
    for needle in [
        "frexpf(__fmul_rn(amax,1.0f/448.0f),&e)",
        "intk=(m==0.5f)?e-1:e",
        "k=max(-127,min(127,k))",
        "constfloatinv=__int_as_float((127-k)<<23)",
        "__nv_cvt_float_to_fp8(v[4*i+j]*inv,__NV_SATFINITE,__NV_E4M3))<<(8*j)",
        "static_cast<unsigned>(k+127)",
    ] {
        assert!(rule.contains(needle), "v10_mxfp8_block lost {needle}");
        assert!(
            q4.contains(needle),
            "v16q4 departs from the v10 rule at {needle}"
        );
    }
    // The bf16 -> f32 widening is the same bit shift in both kernels.
    for needle in [
        "__uint_as_float(u[i]<<16)",
        "__uint_as_float(u[i]&0xffff0000u)",
    ] {
        assert!(v16.contains(needle) && q4.contains(needle), "{needle}");
    }
    assert!(v16.contains("v10_mxfp8_block(v,qq,ss)"));
}

/// Selector unset = the pre-selector loader: every W4A8 set gets its own `qwen4_mxfp8_quant_rows_<tag>`,
/// and that entry exists in the module `load_wide` loads it from. (The first cut returned the v16
/// entry for every set, which the v10/v12/v14 modules do not have; this test is what certifies the
/// off branch now.)
#[test]
fn selector_off_loads_each_sets_own_entry_which_its_module_defines() {
    let compact_src = compact(COMPACT);
    for (tag, module_source) in W4A8_SETS {
        let entry = xq_quant_kernel(false, tag).unwrap_or_else(|e| panic!("{tag}: {e}"));
        assert_eq!(entry, format!("qwen4_mxfp8_quant_rows_{tag}"), "{tag}");
        assert!(
            !entry.ends_with("v16q4"),
            "{tag}: off branch named the twin"
        );
        assert!(
            module_source.contains(&format!("__global__ void {entry}(")),
            "{tag}: module lacks {entry}"
        );
        // The set is one `load_wide` can select from this tree.
        assert!(
            compact_src.contains(&format!("(\"qwen4_moe_compact_t_k32_{tag}\",\"{tag}\",")),
            "{tag}: not a load_wide set"
        );
    }
    // The tag list is open (a later set brings its own entry): no closed match here.
    assert_eq!(
        xq_quant_kernel(false, "v99").unwrap(),
        "qwen4_mxfp8_quant_rows_v99"
    );
}

/// Selector set = the twin, and only on the V16 and V28 sets; every other set fails closed at load.
#[test]
fn selector_on_names_the_twin_only_for_the_v16_and_v28_sets() {
    assert_eq!(xq_quant_kernel(true, "v16").unwrap(), XQ_QUANT_KERNEL_V16Q4);
    assert_eq!(xq_quant_kernel(true, "v28").unwrap(), XQ_QUANT_KERNEL_V16Q4);
    assert_eq!(XQ_QUANT_KERNEL_V16Q4, "qwen4_mxfp8_quant_rows_v16q4");
    assert!(KERNEL.contains("__global__ void qwen4_mxfp8_quant_rows_v16q4("));
    for tag in ["v10", "v12", "v14", "v99", ""] {
        let refused = xq_quant_kernel(true, tag).unwrap_err();
        assert!(refused.contains("W4A8_WIDE=2 or 7"), "{tag:?}: {refused}");
    }
}

/// The V28 module carries the twin through its `#include` of the V16 source (no copy that could
/// drift), and its own `qwen4_mxfp8_quant_rows_v28` is V16's quantiser token for token: so the
/// twin's bit-equality with v16 (micro 90) is bit-equality with v28, and WIDE=7 may load it.
#[test]
fn v28_module_carries_the_twin_and_quantises_like_v16() {
    assert!(KERNEL_V28.contains("#include \"qwen4_moe_compact_t_k32_v16.cu\""));
    assert!(
        !KERNEL_V28.contains("qwen4_mxfp8_quant_rows_v16q4"),
        "V28 must not redefine the twin (it comes from the include)"
    );
    let v28 = section(
        KERNEL_V28,
        "extern \"C\" __global__ void qwen4_mxfp8_quant_rows_v28(",
        "\n}\n",
    );
    let v16 = section(
        KERNEL,
        "extern \"C\" __global__ void qwen4_mxfp8_quant_rows_v16(",
        "\n}\n",
    );
    assert_eq!(
        compact(v28).replace("_v28(", "_v16("),
        compact(v16),
        "qwen4_mxfp8_quant_rows_v28 departs from qwen4_mxfp8_quant_rows_v16"
    );
    // load_wide's V28 arm loads its entries from the V28 module by the "v28" tag.
    assert!(compact(COMPACT).contains("(\"qwen4_moe_compact_t_k32_v28\",\"v28\","));
}

#[test]
fn selector_is_default_off_v16_gated_and_receipted() {
    let compact_src = compact(COMPACT);
    assert!(compact_src.contains(
        "constSLOT_BF16_FUSED_SELECTOR:&str=\"ATLAS_QWEN4_PREFILL_MOE_SLOT_BF16_FUSED\""
    ));
    assert!(compact_src.contains("usesuper::qwen4_xq_quant::xq_quant_kernel;"));
    assert!(compact_src.contains("letxq_fused=slot_bf16_fused_selected()?;letxq_quant_kernel=matchxq_quant_kernel(xq_fused,v){Ok(name)=>name,"));
    assert!(compact_src.contains("MOE_PREFILL_SLOT_BF16_FUSED_REFUSED"));
    assert!(compact_src.contains("w4a8_quant:gpu.kernel(w4a8_module,&xq_quant_kernel)?,xq_fused,"));
    // Admission and post-load validation both pin the selector.
    assert!(compact_src.contains("ifslot_bf16_fused_selected()?{ensure!(selected()?&&transposed"));
    assert!(compact_src.contains(
        "&&w4a8_selected()?&&matches!(w4a8_wide_level()?,2|7),\"{SLOT_BF16_FUSED_SELECTOR}requires"
    ));
    assert!(compact_src.contains(".is_none_or(|wide|wide.xq_fused==xq_fused)"));

    let ops = compact(OPS);
    assert!(
        ops.contains("MOE_PREFILL_SLOT_BF16_FUSED_ENGAGEDselector={}kernel={}set={}rows={rows}")
    );
    assert!(ops.contains("qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4,wide.w4a8_tag"));
    assert!(ops.contains("ifwide.xq_fused&&!SLOT_BF16_FUSED_RECEIPT_LOGGED.swap(true"));
    assert!(ops.contains("qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4"));
    // Both quant sites go through the one helper; no direct launch of the quant handle remains.
    // Both callers pass the arena (fix/test-triage: gatepre gate scalars live there).
    assert_eq!(
        OPS.matches("launch_xq_quant(ctx.gpu, ctx.buffers, wide,")
            .count(),
        1
    );
    assert_eq!(
        SHARED
            .matches("launch_xq_quant(ctx.gpu, ctx.buffers, &wide,")
            .count(),
        1
    );
    assert_eq!(
        OPS.matches("KernelLaunch::new(gpu, wide.w4a8_quant)")
            .count(),
        1
    );
    assert!(!SHARED.contains("wide.w4a8_quant"));
}

#[test]
fn v28_k11b2_l2_residency_and_512t_pretranscode_contract() {
    assert!(KERNEL_V28.contains("#include \"detail/moe_compact/v28_shared.cuh\""));
    assert!(KERNEL_V28.contains("#include \"detail/moe_compact/v28_q4.cuh\""));

    let pre = section(
        KERNEL_V28_SHARED,
        "void v28_pretranscode_slice(",
        "template <bool GATEUP, bool DISCARD = false, typename Shape = Qwen4FlashNextShape>\n__device__ __forceinline__ void v28_shared_impl(",
    );
    let pre_c = compact(pre);
    assert!(pre_c.contains("v28_cp16_evict_first(&sB[mtx][r][cc*16]"));
    assert!(pre_c.contains("v28_cp16_evict_last(&sS[m2][r2][c2*16]"));
    assert!(pre_c.contains("constunsignedt_col=tid&255u,t_h=tid>>8;"));
    assert!(!pre_c.contains("if(tid<V14_BC)"));

    let shared = section(
        KERNEL_V28_SHARED,
        "void v28_shared_impl(",
        "extern \"C\" __global__ __launch_bounds__(512, 2) void qwen4_shared_pretranscode_v28(",
    );
    let shared_c = compact(shared);
    assert!(shared_c.contains("constboola_keep=GATEUP||(nt+1u<n_tiles);"));
    assert!(shared_c.contains("constboolbt_reused=(local+static_cast<int>(V10_TILE_M)<M_expert);"));
    assert!(shared_c.contains("v28_cp16_evict_last_pred("));
    assert!(shared_c.contains("v28_cp16_evict_first_pred("));
    assert!(
        shared_c.contains("v28_st8_evict_last(oq+orow*N+cta_n+col8,make_uint2(qq0[0],qq0[1]))")
    );
    assert!(shared_c.contains("if(task+gridDim.x<task_count)__syncthreads();"));

    let v28q = section(
        KERNEL_V28_Q4,
        "void v28q_impl(",
        "extern \"C\" __global__ __launch_bounds__(512, 1) void qwen4_moe_compact_t_gemm_gateup_k32_v28q4(",
    );
    let v28q_c = compact(v28q);
    assert!(v28q_c.contains("constboola_keep=GATEUP||(nt+1u<n_tiles);"));
    assert!(v28q_c.contains("constboolb_reused=(local+static_cast<int>(V10_TILE_M)<M_expert);"));
    assert!(v28q_c.contains("if(b_reused){v28_cp16_evict_last(&sB[ks%V14_DB][mtx][r][cc*16]"));
    assert!(v28q_c.contains("v28_cp16_evict_first(&sB[ks%V14_DB][mtx][r][cc*16]"));
    assert!(v28q_c.contains("v28_st8_evict_last(oq+orow*N+cta_n+col8,make_uint2(qq0[0],qq0[1]))"));
    assert!(v28q_c.contains("v28_st8_evict_first(oq+orow*N+cta_n+col8,make_uint2(qq0[0],qq0[1]))"));
    assert!(v28q_c.contains("constexprunsignedPF_DIST=GATEUP?10u:5u;"));
    assert!(v28q_c.contains("if(valid_rows<=64){if(tid>=V14_BC){constunsignedt_col=tid-V14_BC;transcode_unit(sl,tb,t_col,0);transcode_unit(sl,tb,t_col,1);}}else{transcode_unit(sl,tb,tid&255u,tid>>8);}"));
    assert!(v28q_c.contains("if(task+gridDim.x<task_count)__syncthreads();"));

    let ops_c = compact(OPS);
    assert!(ops_c.contains("letuse_v28q_nopf=wide.v28q_level==2&&rows<=4096;"));
}
