// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contract for ATLAS_QWEN4_PREFILL_GDN_CONV_V2: `qwen4_conv1d_l2norm_{f16,bf16}_prefill_v2`
//! must write the bytes `..._prefill_parallel` writes. Both kernels are modelled here (v1: one thread
//! per channel and token, block L2 tree; v2: 8 channels per lane, 32-token strips, register window,
//! the mapped tree) with the same per-element arithmetic, on chunks that start from a live conv
//! state, end in a partial strip and cross strip boundaries. The real-kernel byte check is micro 111.
//! Standalone (std only): `rustc --test --edition 2024 gdn_conv_v2_static.rs`.

#[path = "../src/layers/qwen3_ssm/qwen4_gdn_conv_v2.rs"]
#[allow(dead_code)]
mod qwen4_gdn_conv_v2;

use qwen4_gdn_conv_v2 as v2;

const GLUE: &str =
    include_str!("../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_gdn_fi_glue.cu");
const FI: &str = include_str!("../src/layers/qwen3_ssm/qwen4_gdn_fi.rs");
const MOD: &str = include_str!("../src/layers/qwen3_ssm/mod.rs");

const DIM: usize = 1024; // 4 CTAs of 256 channels: 2 Q|K, 2 V (same structure as 10240 / 4096)
const QK: usize = 512;
const EPS: f32 = 1e-6;

/// Per-element conv + SiLU, identical in both kernels (bias = none -> 0.0f start).
fn conv_silu(taps: [f32; 4], w: [f32; 4]) -> f32 {
    let mut acc = 0.0f32;
    for k in 0..4 {
        acc += taps[k] * w[k];
    }
    let sigmoid = 1.0f32 / (1.0f32 + (-acc).exp());
    acc * sigmoid
}
fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

struct Case {
    x: Vec<f32>,     // [rows][DIM] (bf16 values)
    state: Vec<f32>, // [DIM][4]
    w: Vec<[f32; 4]>,
    rows: usize,
}

fn tap(c: &Case, t: isize, ch: usize) -> f32 {
    if t >= 0 {
        c.x[t as usize * DIM + ch]
    } else {
        c.state[ch * 4 + (t + 4) as usize]
    }
}

/// v1: thread (token, ch); per 256-channel block, two heads of 4 warps; shfl_down tree; ((S0+S1)+S2)+S3.
fn v1(c: &Case) -> Vec<u32> {
    let mut out = vec![0u32; c.rows * DIM];
    for t in 0..c.rows {
        let mut silu = vec![0.0f32; DIM];
        for ch in 0..DIM {
            let taps = [0, 1, 2, 3].map(|k| tap(c, t as isize - 3 + k as isize, ch));
            silu[ch] = conv_silu(taps, c.w[ch]);
        }
        for head in 0..QK / 128 {
            let mut s = [0.0f32; 4];
            for wq in 0..4 {
                let mut v: Vec<f32> = (0..32)
                    .map(|l| {
                        let x = silu[head * 128 + wq * 32 + l];
                        x * x
                    })
                    .collect();
                for off in [16usize, 8, 4, 2, 1] {
                    let p = v.clone();
                    for l in 0..32 {
                        let src = if l + off < 32 { l + off } else { l };
                        v[l] = p[l] + p[src];
                    }
                }
                s[wq] = v[0];
            }
            let total = s[0] + s[1] + s[2] + s[3];
            let r = rsqrt(total + EPS);
            for i in 0..128 {
                silu[head * 128 + i] *= r;
            }
        }
        for ch in 0..DIM {
            out[t * DIM + ch] = silu[ch].to_bits();
        }
    }
    out
}

/// v2: CTA (blockIdx.x, blockIdx.y), warp = strip of TOKENS tokens, lane = 8 channels.
fn v2_model(c: &Case) -> Vec<u32> {
    let mut out = vec![0xdead_beefu32; c.rows * DIM];
    let [gx, gy, _] = v2::grid(DIM as u32, c.rows as u32);
    for bx in 0..gx as usize {
        let l2 = bx * 256 < QK;
        for by in 0..gy as usize {
            for warp in 0..v2::WARPS as usize {
                let t_begin = (by * v2::WARPS as usize + warp) * v2::TOKENS as usize;
                if t_begin >= c.rows {
                    continue;
                }
                let t_end = (t_begin + v2::TOKENS as usize).min(c.rows);
                // per lane: window win[k][c], k = 0..2
                let mut win = [[[0.0f32; 8]; 3]; 32];
                for lane in 0..32 {
                    let ch0 = bx * 256 + lane * 8;
                    for k in 0..3 {
                        let idx = t_begin as isize - 3 + k as isize;
                        for cc in 0..8 {
                            win[lane][k][cc] = tap(c, idx, ch0 + cc);
                        }
                    }
                }
                for t in t_begin..t_end {
                    let mut silu = [[0.0f32; 8]; 32];
                    let mut cur = [[0.0f32; 8]; 32];
                    for lane in 0..32 {
                        let ch0 = bx * 256 + lane * 8;
                        for cc in 0..8 {
                            cur[lane][cc] = c.x[t * DIM + ch0 + cc];
                            let taps = [
                                win[lane][0][cc],
                                win[lane][1][cc],
                                win[lane][2][cc],
                                cur[lane][cc],
                            ];
                            silu[lane][cc] = conv_silu(taps, c.w[ch0 + cc]);
                        }
                    }
                    if l2 {
                        let mut sq = [[0.0f32; 8]; 32];
                        for lane in 0..32 {
                            for cc in 0..8 {
                                sq[lane][cc] = silu[lane][cc] * silu[lane][cc];
                            }
                        }
                        for off in [2usize, 1] {
                            let p = sq;
                            for lane in 0..32 {
                                let src = if lane + off < 32 { lane + off } else { lane };
                                for cc in 0..8 {
                                    sq[lane][cc] = p[lane][cc] + p[src][cc];
                                }
                            }
                        }
                        for lane in 0..32 {
                            for cc in 0..4 {
                                sq[lane][cc] += sq[lane][cc + 4];
                            }
                            for cc in 0..2 {
                                sq[lane][cc] += sq[lane][cc + 2];
                            }
                            sq[lane][0] += sq[lane][1];
                        }
                        for lane in 0..32 {
                            let h = lane & 16;
                            let total = sq[h][0] + sq[h | 4][0] + sq[h | 8][0] + sq[h | 12][0];
                            let r = rsqrt(total + EPS);
                            for cc in 0..8 {
                                silu[lane][cc] *= r;
                            }
                        }
                    }
                    for lane in 0..32 {
                        let ch0 = bx * 256 + lane * 8;
                        for cc in 0..8 {
                            out[t * DIM + ch0 + cc] = silu[lane][cc].to_bits();
                        }
                        for cc in 0..8 {
                            win[lane][0][cc] = win[lane][1][cc];
                            win[lane][1][cc] = win[lane][2][cc];
                            win[lane][2][cc] = cur[lane][cc];
                        }
                    }
                }
            }
        }
    }
    out
}

fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn case(rows: usize, seed: u64, live_state: bool) -> Case {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let x = (0..rows * DIM).map(|_| bf16(next() * 6.0)).collect();
    let state = (0..DIM * 4)
        .map(|_| if live_state { next() * 4.0 } else { 0.0 })
        .collect();
    let w = (0..DIM)
        .map(|_| [bf16(next()), bf16(next()), bf16(next()), bf16(next())])
        .collect();
    Case { x, state, w, rows }
}

#[test]
fn v2_writes_v1_bytes_at_chunk_start_strip_edges_and_partial_strips() {
    for (rows, live) in [
        (256usize, true),
        (300, true),
        (33, false),
        (1, true),
        (520, true),
        (257, false),
    ] {
        let c = case(rows, 0x111 + rows as u64, live);
        let a = v1(&c);
        let b = v2_model(&c);
        let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "rows {rows}: {diff} elements differ");
        assert!(
            !b.contains(&0xdead_beef),
            "rows {rows}: an element was not written"
        );
    }
}

#[test]
fn rule_selector_grid_and_refusals() {
    assert_eq!(v2::parse(None), Ok(false));
    assert_eq!(v2::parse(Some("1")), Ok(true));
    assert!(v2::parse(Some("2")).is_err());
    // Flash-Next: CONV_DIM 10240, Q|K 4096, QKVZ stride 16384.
    let ok = [0x1000u64, 0x2000, 0x3000];
    assert_eq!(
        v2::refusal(8192, 10240, 4096, 128, 4, 16384, 10240, ok),
        None
    );
    assert_eq!(v2::grid(10240, 8192), [40, 32, 1]);
    assert_eq!(v2::grid(10240, 16384), [40, 64, 1]);
    assert_eq!(v2::grid(10240, 65), [40, 1, 1]);
    // TP=2 Flash-Next shard: CONV_DIM 5120, Q|K 2048, QKVZ stride 8192.
    assert_eq!(v2::refusal(8192, 5120, 2048, 128, 4, 8192, 5120, ok), None);
    assert_eq!(v2::grid(5120, 8192), [20, 32, 1]);
    assert_eq!(
        v2::refusal(0, 10240, 4096, 128, 4, 16384, 10240, ok),
        Some("no_rows")
    );
    assert_eq!(
        v2::refusal(8, 10200, 4096, 128, 4, 16384, 10240, ok),
        Some("channels_not_256_blocks")
    );
    assert_eq!(
        v2::refusal(8, 10240, 4000, 128, 4, 16384, 10240, ok),
        Some("channels_not_256_blocks")
    );
    assert_eq!(
        v2::refusal(8, 10240, 4096, 64, 4, 16384, 10240, ok),
        Some("head_dim_not_128")
    );
    assert_eq!(
        v2::refusal(8, 10240, 4096, 128, 3, 16384, 10240, ok),
        Some("d_conv_not_4")
    );
    assert_eq!(
        v2::refusal(8, 10240, 4096, 128, 4, 16380, 10240, ok),
        Some("row_stride_not_16b")
    );
    assert_eq!(
        v2::refusal(
            8,
            10240,
            4096,
            128,
            4,
            16384,
            10240,
            [0x1000, 0x2008, 0x3000]
        ),
        Some("pointer_not_16b_aligned")
    );
}

fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn v1_still_has_the_modelled_arithmetic() {
    let g = compact(GLUE);
    for needle in [
        "floatacc=(bias!=nullptr)?bias[ch]:0.0f;for(unsignedintk=0;k<d_conv;k++){constintidx=(int)token-(int)(d_conv-1)+(int)k;constfloats=(idx>=0)?(float)new_input[(unsignedlonglong)idx*input_stride+ch]:conv_state[(unsignedlonglong)ch*d_conv+(k+token+1)];acc+=s*(float)w[k];}",
        "floatsigmoid_acc=qwen4_fast_rcp_rn_1p_exp(acc);silu=acc*sigmoid_acc;",
        "floatsq=valid?silu*silu:0.0f;",
        "for(intoffset=16;offset>=1;offset>>=1)sq+=__shfl_down_sync(0xFFFFFFFF,sq,offset);",
        "floattotal=warp_sums[base_warp]+warp_sums[base_warp+1]+warp_sums[base_warp+2]+warp_sums[base_warp+3];warp_sums[base_warp]=rsqrtf(total+l2_eps);",
        "if(valid)silu*=warp_sums[base_warp];",
    ] {
        assert!(g.contains(needle), "v1 changed: {needle}");
    }
}

#[test]
fn v2_kernel_implements_the_model() {
    let g = compact(GLUE);
    for needle in [
        "constunsignedintch0=blockIdx.x*256u+lane*8u;",
        "constbooll2=blockIdx.x*256u<qk_channels;",
        "constunsignedintt_begin=(blockIdx.y*8u+warp)*TOKENS;",
        "win[k][c]=conv_state[(unsignedlonglong)(ch0+c)*4u+(unsignedint)(idx+4)];",
        "bias_c[c]=(bias!=nullptr)?bias[ch0+c]:0.0f;",
        "floatacc=bias_c[c];acc+=win[0][c]*wt[c][0];acc+=win[1][c]*wt[c][1];acc+=win[2][c]*wt[c][2];acc+=cur[c]*wt[c][3];floatsigmoid_acc=qwen4_fast_rcp_rn_1p_exp(acc);silu[c]=acc*sigmoid_acc;",
        "sq[c]=silu[c]*silu[c];",
        "sq[c]+=__shfl_down_sync(0xFFFFFFFF,sq[c],2);",
        "sq[c]+=__shfl_down_sync(0xFFFFFFFF,sq[c],1);",
        "for(intc=0;c<4;++c)sq[c]+=sq[c+4];",
        "for(intc=0;c<2;++c)sq[c]+=sq[c+2];sq[0]+=sq[1];",
        "floattotal=s0+s1+s2+s3;constfloatinv=rsqrtf(total+l2_eps);",
        "win[0][c]=win[1][c];win[1][c]=win[2][c];win[2][c]=cur[c];",
        "#defineQWEN4_FI_CONV_V2_TOKENS32u",
        "QWEN4_FI_CONV_V2_ENTRY(qwen4_conv1d_l2norm_f16_prefill_v2,__half)",
        "QWEN4_FI_CONV_V2_ENTRY(qwen4_conv1d_l2norm_bf16_prefill_v2,__nv_bfloat16)",
        "return(unsignedint)__half_as_ushort(__float2half_rn(a))|",
        "return(unsignedint)__bfloat16_as_ushort(__float2bfloat16(a))|",
    ] {
        assert!(g.contains(needle), "v2 lost: {needle}");
    }
    // v1's stores use the same conversions.
    assert!(g.contains("qwen4_fi_store(__half*p,floatv){*p=__float2half_rn(v);}"));
    assert!(g.contains("qwen4_fi_store(__nv_bfloat16*p,floatv){*p=__float2bfloat16(v);}"));
    assert_eq!(v2::TOKENS, 32);
}

#[test]
fn host_dispatch_is_default_off_and_receipted() {
    let fi = compact(FI);
    assert!(fi.contains("letconv_launch=matchconv_v2_kernel(ctx.gpu,dtype)?{"));
    assert!(
        fi.contains("ifSEL.get_or_init(||v2::parse(std::env::var(v2::SELECTOR).ok().as_deref()))")
            || fi
                .contains(".get_or_init(||v2::parse(std::env::var(v2::SELECTOR).ok().as_deref()))")
    );
    assert!(fi.contains("if!on{returnOk(None);}"));
    assert!(fi.contains("QWEN4_PREFILL_GDN_CONV_V2_ENGAGEDselector={}"));
    assert!(fi.contains("QWEN4_PREFILL_GDN_CONV_V2_REFUSEDselector={}reason={reason}"));
    assert!(fi.contains("(&F16,\"qwen4_conv1d_l2norm_f16_prefill_v2\")"));
    assert!(fi.contains("(&BF16,\"qwen4_conv1d_l2norm_bf16_prefill_v2\")"));
    assert!(fi.contains("lets=super::qwen4_prefill_exact_plan::SsmShape::from_heads("));
    assert!(fi.contains(".grid(v2::grid(s.conv_dimasu32,rows_u32))"));
    // v1 keeps its geometry on both fallbacks.
    assert_eq!(
        FI.matches(".grid([div_ceil(s.conv_dim as u32, 256), rows_u32, 1])")
            .count(),
        2
    );
    // Same argument list for both kernels; the commit still follows.
    assert!(fi.contains("conv_launch.arg_ptr(state.conv_state).arg_ptr(deinterleaved).arg_ptr(self.ssm.conv1d.weight).arg_ptr(DevicePtr::NULL).arg_ptr(qkv)"));
    assert!(MOD.contains("mod qwen4_gdn_conv_v2;"));
    assert_eq!(v2::SELECTOR, "ATLAS_QWEN4_PREFILL_GDN_CONV_V2");
}
