// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contract for ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED: the E4M3 QSA prefill core's `_amax`
//! entries fold |bf16| of every output value they store into the Q1 slot with
//! `qwen4_fp8_act_amax`'s rule, and the o_proj's tensor-scale FP8 projection then quantises from
//! that slot instead of re-reading [rows][24*256] for its absmax. The max is order-free, so the
//! slot bits, the scale and the e4m3 rows are identical; this test pins the rule on both sides and
//! the hand-off wiring, and checks the order-freedom on a model. The real-kernel check is micro 110.
//! Standalone (std only): `rustc --test --edition 2024 oproj_amax_static.rs`.

const CORE: &str =
    include_str!("../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_qsa_prefill_attn_fp8.cu");
const PROJ: &str = include_str!("../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_fp8_proj.cu");
const RUN: &str = include_str!("../src/layers/qwen4_qsa/prefill_sparse_run.rs");
const FP8: &str = include_str!("../src/layers/qwen4_qsa/prefill_fp8.rs");
const FAST: &str = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_fast_attn.rs");
const Q1: &str = include_str!("../src/layers/qwen4_fp8_act_amax.rs");
const FAST_PROJ: &str = include_str!("../src/layers/qwen4_fast_proj.rs");
const FWD: &str = include_str!("../src/model/trait_impl/prefill_b/forward_layers.rs");

fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn section<'a>(src: &'a str, start: &str, end: &str) -> &'a str {
    let a = src.find(start).unwrap_or_else(|| panic!("missing {start}"));
    let b = src[a..].find(end).map(|o| a + o).unwrap_or(src.len());
    &src[a..b]
}

#[test]
fn consumer_absmax_rule_is_the_one_the_producer_copies() {
    let amax = compact(section(
        PROJ,
        "extern \"C\" __global__ void qwen4_fp8_act_amax(",
        "// One block per weight row",
    ));
    for needle in [
        "floatm=0.0f;",
        "constfloat2f=__bfloat1622float2(p[j]);m=fmaxf(m,fmaxf(fabsf(f.x),fabsf(f.y)));",
        "m=block_max(m,sm);if(threadIdx.x==0)atomicMax(amax_bits,__float_as_uint(m));",
    ] {
        assert!(
            amax.contains(needle),
            "qwen4_fp8_act_amax changed: {needle}"
        );
    }
    // The quantising pass reads the slot the same way whichever kernel filled it.
    let rows = compact(section(
        PROJ,
        "extern \"C\" __global__ void qwen4_fp8_act_rows_tensor(",
        "\u{0}",
    ));
    assert!(rows.contains("constfloatamax=fmaxf(__uint_as_float(amax_bits[0]),1e-30f);"));
}

#[test]
fn producer_folds_every_stored_value_with_the_same_rule() {
    let core = compact(CORE);
    assert!(core.contains(
        "__device____forceinline__floatamax_bf16x2(floatamax,unsignedintw){returnfmaxf(amax,fmaxf(fabsf(__uint_as_float(w<<16)),fabsf(__uint_as_float(w&0xffff0000u))));}"
    ));
    // PV8 path: both packed words of each 4-value store; BF16 path: the one packed word.
    assert!(core.contains(
        "constunsignedintw01=pack_bf16(x0,x1);constunsignedintw23=pack_bf16(x2,x3);*reinterpret_cast<uint2*>(dst+d0)=make_uint2(w01,w23);ifconstexpr(AMAX==1)amax=amax_bf16x2(amax_bf16x2(amax,w01),w23);"
    ));
    assert!(core.contains(
        "constunsignedintw01=pack_bf16(x0,x1);*reinterpret_cast<unsignedint*>(dst+d0)=w01;ifconstexpr(AMAX==1)amax=amax_bf16x2(amax,w01);"
    ));
    // Each MMA row is stored at most once (live rows) before the CTA-wide amax reduction.
    assert!(core.contains("floatamax=0.0f;if(ra.live){store_row<PV8,AMAX>("));
    assert!(core.contains("if(rb.live){store_row<PV8,AMAX>("));
    // One atomic per CTA (warp fold, CTA fold behind one barrier), skipped when the slot already holds it.
    assert!(core.contains("ifconstexpr(AMAX!=0){//EverythreadoftheCTAreachesthispoint"));
    assert!(core.contains(
        "__shared__floats_amax[kWarps];for(intd=16;d;d>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,d));if(lane==0)s_amax[warp]=amax;__syncthreads();if(tid==0){floatm=s_amax[0];"
    ));
    assert!(core.contains("for(intw=1;w<kWarps;++w)m=fmaxf(m,s_amax[w]);"));
    assert!(core.contains(
        "constfloatseen=__uint_as_float(*reinterpret_cast<volatileunsignedint*>(amax_bits));if(m>0.0f&&m>seen)atomicMax(amax_bits,__float_as_uint(m));}}}"
    ));
    // The only early exit before the epilogue is CTA-uniform; the tile skip is warp-uniform.
    assert!(core.contains("if(r0>=rows)return;"));
    assert!(core.contains("if((__reduce_or_sync(0xffffffffu,need)&warp_rows)==0)continue;"));
    let ship_core = compact(section(
        CORE,
        "template <bool PV8, int ACC, int STAGES, int AMAX = 0>",
        "#define QSA_PREFILL_ATTN_FP8_ENTRY(",
    ));
    assert_eq!(
        ship_core.matches("return;").count(),
        1,
        "a new early return would skip the warp shuffle"
    );
    assert_eq!(
        core.matches("return;").count(),
        core.matches("if(r0>=rows)return;").count()
            + core.matches("if(tile_it==0xffffu)return;").count(),
        "every return; is CTA-uniform r0>=rows or a lambda early-out"
    );
}

#[test]
fn shipping_entries_are_untouched_and_the_twins_add_one_argument() {
    let core = compact(CORE);
    assert!(core.contains("QSA_PREFILL_ATTN_FP8_ENTRY(qwen4_qsa_prefill_attn_fp8_qk,false,0,2)"));
    assert!(core.contains("QSA_PREFILL_ATTN_FP8_ENTRY(qwen4_qsa_prefill_attn_fp8_qkpv,true,2,2)"));
    assert!(
        core.contains(
            "QSA_PREFILL_ATTN_FP8_AMAX_ENTRY(qwen4_qsa_prefill_attn_fp8_qk_amax,false,0,2)"
        )
    );
    assert!(core.contains(
        "QSA_PREFILL_ATTN_FP8_AMAX_ENTRY(qwen4_qsa_prefill_attn_fp8_qkpv_amax,true,2,2)"
    ));
    let plain = compact(section(
        CORE,
        "#define QSA_PREFILL_ATTN_FP8_ENTRY(",
        "// AMAX mode of the shipped _amax entries",
    ));
    assert!(plain.contains("qsa_prefill_attn_fp8<PV8,ACC,STAGES>("));
    assert!(!plain.contains("amax"));
    let twin = compact(section(
        CORE,
        "#define QSA_PREFILL_ATTN_FP8_AMAX_ENTRY(",
        "QSA_PREFILL_ATTN_FP8_ENTRY(qwen4",
    ));
    assert!(twin.contains("floatout_scale_h1,\\unsignedint*__restrict__amax_bits){"));
    assert!(twin.contains("qsa_prefill_attn_fp8<PV8,ACC,STAGES,QSA_OPROJ_AMAX_MODE>("));
    assert!(core.contains("#ifndefQSA_OPROJ_AMAX_MODE#defineQSA_OPROJ_AMAX_MODE3#endif"));
    assert!(twin.contains("out_scale_h0,out_scale_h1,amax_bits);"));
    assert!(core.contains("template<boolPV8,intACC,intSTAGES,intAMAX=0>"));
    assert!(core.contains("unsignedint*__restrict__amax_bits=nullptr){"));
}

#[test]
fn hand_off_is_selected_scoped_and_matches_the_o_proj_shape() {
    let run = compact(RUN);
    assert!(run.contains("letamax_slot=ifact_amax::oproj_selected()?&&act_amax::take_armed(){letslot=act_amax::slot(gpu)?;gpu.memset_async(slot,0,4,stream)?;Some(slot)}else{None};"));
    assert!(run.contains("(None,Some(_),Mode::Qk)=>{(prefill_fp8::amax_kernels(gpu)?.attn_qk,mode.smem()asu32)}(None,Some(_),Mode::QkPv)=>(prefill_fp8::amax_kernels(gpu)?.attn_qkpv,mode.smem()asu32,),"));
    assert!(run.contains(".arg_f32(out_scale(1));matchamax_slot{Some(slot)=>{launch.arg_ptr(slot).launch(stream)?;act_amax::note_ready(act_amax::Ready{act:c.out.0,rows:c.rows,cols:QUERY_HEADS*HEAD_DIM,stream,producer:act_amax::Producer::QsaCore,});}None=>launch.launch(stream)?,}"));
    let fp8 = compact(FP8);
    assert!(fp8.contains("load(\"qwen4_qsa_prefill_attn_fp8_qk_amax\")?"));
    assert!(fp8.contains("load(\"qwen4_qsa_prefill_attn_fp8_qkpv_amax\")?"));
    let fast = compact(FAST);
    assert!(fast.contains("letoproj_amax_on=crate::layers::qwen4_fp8_act_amax::oproj_active()?&&qwen4_fast_proj::fp8_selection()?.attn&&qwen4_fast_proj::fp8_tensor_scales();letoproj_amax_arm=crate::layers::qwen4_fp8_act_amax::Arm::enter_selected(oproj_amax_on);"));
    // The test-only trigger narrows the selector once per call site; the core only follows the arm.
    let q1t = compact(Q1);
    assert!(
        q1t.contains("Ok(oproj_selected()?&&oproj_trigger().is_none_or(std::path::Path::exists))")
    );
    assert!(fast.contains("drop(oproj_amax_arm);mark(\"proj_o\",stream);"));
    assert!(fast.contains("OPROJ_AMAX_FUSED_REFUSEDselector={}reason=core_not_e4m3_qsa"));
    // The o_proj is the next tensor-scale consumer: proj_gemm_ex over raw_outputs [num_tokens][attn_width].
    assert!(fast.contains("qwen4_fast_proj::proj_gemm_ex(ctx.gpu,qwen4_fast_proj::fp8_selection()?.attn,raw_outputs,&self.attn.o_proj,packed_inputs,num_tokens,ctx.config.hidden_size,attn_width,"));
    let fp = compact(FAST_PROJ);
    assert!(fp.contains("if!crate::layers::qwen4_fp8_act_amax::take_matching(act,m,k,stream){returnfp8_act_tensor(gpu,tk,act,m,k,ws,a8,stream);}"));
    let q1 = compact(Q1);
    assert!(
        q1.contains("pub(crate)constOPROJ_SELECTOR:&str=\"ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED\";")
    );
    assert!(q1.contains("OPROJ_AMAX_FUSED_ENGAGEDselector={OPROJ_SELECTOR}producer=qsa_e4m3_core"));
    // Q1's own receipt is unchanged (gates grep it).
    assert!(q1.contains(
        "FP8_ACT_AMAX_FUSED_ENGAGEDselector={SELECTOR}producer=hc_upmixconsumer=fp8_tensor"
    ));
}

/// FTZ fabsf (the target's fast-math flags): subnormal magnitudes count as zero on both sides.
fn mag(bits16: u16) -> f32 {
    let x = f32::from_bits((bits16 as u32) << 16).abs();
    if x.is_subnormal() { 0.0 } else { x }
}
fn fmaxf(a: f32, b: f32) -> f32 {
    a.max(b) // NaN-ignoring, like CUDA fmaxf
}

#[test]
fn producer_partition_and_consumer_grid_give_the_same_slot_bits() {
    let mut s = 0x0b0d_cafe_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for trial in 0..20 {
        let n = 6144 * (8 + trial);
        let v: Vec<u16> = (0..n)
            .map(|i| match next() % 50 {
                0 => 0x7fc0,                       // NaN
                1 => 0x8000,                       // -0
                2 => 0x0001 | (i as u16 & 0x8000), // subnormal
                _ => (next() >> 20) as u16 & 0x7f7f | ((next() as u16) & 0x8000),
            })
            .collect();
        // consumer: 256 x 256 threads grid-stride over 8-element vectors, block_max, atomicMax
        let mut slot_c = 0u32;
        let threads = 256 * 256;
        for tid in 0..threads {
            let mut m = 0.0f32;
            let mut i = tid;
            while i < n / 8 {
                for j in 0..8 {
                    m = fmaxf(m, mag(v[i * 8 + j]));
                }
                i += threads;
            }
            slot_c = slot_c.max(m.to_bits());
        }
        // producer: arbitrary partition (warps of a tile plan), per-warp fold, atomicMax
        let mut slot_p = 0u32;
        let mut start = 0usize;
        while start < n {
            let len = (1 + next() as usize % 3000).min(n - start);
            let mut m = 0.0f32;
            for &x in v[start..start + len].iter().rev() {
                m = fmaxf(m, mag(x));
            }
            if m > 0.0 {
                slot_p = slot_p.max(m.to_bits());
            }
            start += len;
        }
        assert_eq!(slot_c, slot_p, "trial {trial}");
    }
}

#[test]
fn modes_two_and_three_fold_the_same_values() {
    let core = compact(CORE);
    assert!(core.contains(
        "ifconstexpr(AMAX==2)amax=fmaxf(fmaxf(fmaxf(fmaxf(amax,fabsf(x0)),fabsf(x1)),fabsf(x2)),fabsf(x3));"
    ));
    assert!(core.contains("ifconstexpr(AMAX==2)amax=fmaxf(fmaxf(amax,fabsf(x0)),fabsf(x1));"));
    assert!(core.contains("ifconstexpr(AMAX==2)amax=__bfloat162float(__float2bfloat16(amax));"));
    assert!(core.contains("ld.global.cg.v2.u32{%0,%1},[%2];"));
    assert!(core.contains(
        "if(ra.live)amax=reread_row<PV8>(out,r0+ra.bit,kvh*kGroup+ma_row%kGroup,o_stride,t,amax);"
    ));
    assert!(core.contains(
        "if(rb.live)amax=reread_row<PV8>(out,r0+rb.bit,kvh*kGroup+mb_row%kGroup,o_stride,t,amax);"
    ));
    // reread_row walks exactly store_row's addresses.
    assert!(core.contains("constunsignedintd0=c*16+t*4;"));
    assert!(core.contains("\"l\"(src+c*16+t*4)"));
    assert!(core.contains("constunsignedintd0=i*8+t*2;"));
    assert!(core.contains("\"l\"(src+i*8+t*2)"));
}

#[test]
fn mode_three_rereads_are_ordered_after_the_stores() {
    // K10 C: both re-read loads carry a "memory" clobber, so the compiler can neither sink
    // store_row's plain stores below them nor hoist them above the stores (a stale word would be
    // the previous layer's raw_outputs). Micro 170 checks the same order in the engine-flag PTX.
    let core = compact(CORE);
    assert!(core.contains(
        "asmvolatile(\"ld.global.cg.v2.u32{%0,%1},[%2];\":\"=r\"(w0),\"=r\"(w1):\"l\"(src+c*16+t*4):\"memory\");"
    ));
    assert!(core.contains(
        "asmvolatile(\"ld.global.cg.u32%0,[%1];\":\"=r\"(w0):\"l\"(src+i*8+t*2):\"memory\");"
    ));
    assert_eq!(
        core.matches("asmvolatile(\"ld.global").count(),
        2,
        "every global load in asm is one of the two clobbered re-reads"
    );
    let stores = core
        .find("if(rb.live){store_row<PV8,AMAX>(")
        .expect("second store_row");
    let rereads = core
        .find("ifconstexpr(AMAX==3){if(ra.live)amax=reread_row<PV8>(")
        .expect("mode-3 re-reads");
    assert!(stores < rereads, "the re-reads follow both store_row calls");
}

#[test]
fn every_prefill_chunk_reports_its_o_proj_hand_offs() {
    let q1 = compact(Q1);
    assert!(q1.contains(
        "OPROJ_AMAX_FUSED_CHUNKselector={OPROJ_SELECTOR}start={start}rows={rows}hits={}refused={}missed={}"
    ));
    assert!(q1.contains(
        "ifready.producer==Producer::QsaCore{tally(|t|ifhit{t.hits+=1}else{t.missed+=1});}"
    ));
    assert!(q1.contains("ifmatches!(oproj_selected(),Ok(true)){"));
    // The MISS line names the producer's selector, not always Q1's.
    assert!(q1.contains("FP8_ACT_AMAX_FUSED_MISSselector={}ready={ready:?}"));
    assert!(q1.contains("ready.producer.selector()"));
    let fast = compact(FAST);
    assert!(fast.contains(
        "crate::layers::qwen4_fp8_act_amax::note_oproj_refused();staticOPROJ_REFUSED:std::sync::Once"
    ));
    let fwd = compact(FWD);
    let begin = fwd
        .find("crate::layers::qwen4_fp8_act_amax::oproj_chunk_begin();")
        .expect("chunk begin");
    let layers = fwd
        .find("for(i,layer)inself.layers.iter().enumerate(){")
        .expect("layer loop");
    let receipt = fwd
        .find("crate::layers::qwen4_fp8_act_amax::oproj_chunk_receipt(effective_seq_len_start,proc_count);")
        .expect("chunk receipt");
    assert!(begin < layers && layers < receipt);
}

/// bf16 round-to-nearest-even of an f32 (the cvt the kernel's pack uses), as f32.
fn rn_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}
fn ftz_abs(x: f32) -> f32 {
    let a = x.abs();
    if a.is_subnormal() { 0.0 } else { a }
}

#[test]
fn mode_two_rounding_identity_max_of_rounded_is_rounded_max() {
    assert!(
        rn_bf16(3.3962e38).is_infinite(),
        "3.3962e38 (0x7f7f_806f) must round to +inf in bf16"
    );
    let mut s = 0x2b0d_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // On slices containing at least one normal or infinite value, Mode 2
    // (fmaxf over FTZ |x|, rounded once to bf16 at the end) matches Mode 1
    // (fmaxf over |widened bf16|).
    for trial in 0..2000 {
        let n = 1 + (next() % 300) as usize;
        let xs: Vec<f32> = (0..n)
            .map(|_| match next() % 40 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => -f32::INFINITY,
                3 => f32::from_bits((next() as u32) & 0x807f_ffff), // subnormal of either sign
                4 => 3.3962e38,                                     // rounds to inf in bf16
                _ => f32::from_bits(next() as u32 & 0xbfff_ffff | 0x2000_0000),
            })
            .collect();
        // mode 1 / act_amax: fmaxf over |widened bf16| (FTZ abs), from 0
        let mut m1 = 0.0f32;
        for &x in &xs {
            m1 = m1.max(ftz_abs(rn_bf16(x)));
        }
        // mode 2: fmaxf over |x| (FTZ abs) in f32, one rounding at the end
        let mut m2 = 0.0f32;
        for &x in &xs {
            m2 = m2.max(ftz_abs(x));
        }
        let m2 = rn_bf16(m2);
        assert_eq!(m1.to_bits(), m2.to_bits(), "trial {trial}");
    }
    // Subnormal-only boundary where Mode 2 diverges from Mode 1 (and why Mode 3 ships):
    // f32 subnormals in [0x007f_8000..=0x007f_ffff] round up to MIN_POSITIVE in bf16,
    // whereas FTZ `fmaxf` flushes them to 0.0 before rounding.
    let sub = f32::from_bits(0x007f_8000);
    assert_eq!(ftz_abs(rn_bf16(sub)).to_bits(), f32::MIN_POSITIVE.to_bits());
    assert_eq!(rn_bf16(ftz_abs(sub)).to_bits(), 0.0f32.to_bits());
}
