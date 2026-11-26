// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contract for ATLAS_QWEN4_PREFILL_MOE_TOPK_WARP: the warp-per-token router top-k twin
//! `moe_topk_softmax_batched_w32` must return the indices and weights `moe_topk_softmax_batched`
//! returns, bit for bit. Both kernels are modelled here step by step (thread scan, shuffle
//! tournament and trees with simultaneous lane updates, FTZ comparisons) and run on tie-heavy,
//! NaN, signed-zero, subnormal and all-equal rows. The real-kernel bit check is micro 109.
//! Standalone (std only): `rustc --test --edition 2024 moe_topk_warp_static.rs`.

#[path = "../src/layers/moe/qwen4_topk_warp_rule.rs"]
#[allow(dead_code)]
mod qwen4_topk_warp_rule;

use qwen4_topk_warp_rule as rule;

const KERNEL: &str = include_str!("../../../kernels/gb10/common/moe_topk.cu");
const FORWARD: &str = include_str!("../src/layers/moe/forward_prefill.rs");
const GLUE: &str = include_str!("../src/layers/moe/qwen4_topk_warp.rs");
const MOD: &str = include_str!("../src/layers/moe/mod.rs");

const NEG: f32 = -1e30;

/// The target builds with --use_fast_math (ftz): f32 comparisons see subnormals as zero.
fn ftz(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}
fn gt(a: f32, b: f32) -> bool {
    ftz(a) > ftz(b)
}
fn eq(a: f32, b: f32) -> bool {
    ftz(a) == ftz(b)
}
/// Stand-in for `__expf` (both models call the same function in the same places).
fn ex(x: f32) -> f32 {
    ftz(ftz(x).exp())
}
fn add(a: f32, b: f32) -> f32 {
    ftz(ftz(a) + ftz(b))
}
fn div(a: f32, b: f32) -> f32 {
    ftz(ftz(a) / ftz(b))
}

#[derive(Debug, PartialEq)]
struct Out {
    idx: Vec<u32>,
    weights: Vec<u32>,
}

/// `moe_topk_softmax_batched` for one token (256 threads, 8 warps, 512 experts).
fn block_kernel(logits: &[f32; 512], top_k: usize, normalize: bool) -> Out {
    let mut s = *logits;
    let mut top_vals = vec![0.0f32; top_k];
    let mut top_idxs = vec![0u32; top_k];
    for t in 0..top_k {
        let mut lv = [NEG; 256];
        let mut li = [0u32; 256];
        for tid in 0..256 {
            for i in [tid, tid + 256] {
                if gt(s[i], lv[tid]) {
                    lv[tid] = s[i];
                    li[tid] = i as u32;
                }
            }
        }
        let mut wv = [0.0f32; 8];
        let mut wi = [0u32; 8];
        for w in 0..8 {
            let mut v: Vec<f32> = lv[w * 32..w * 32 + 32].to_vec();
            let mut ix: Vec<u32> = li[w * 32..w * 32 + 32].to_vec();
            for off in [16usize, 8, 4, 2, 1] {
                let (pv, pi) = (v.clone(), ix.clone());
                for l in 0..32 {
                    let src = if l + off < 32 { l + off } else { l };
                    if gt(pv[src], pv[l]) {
                        v[l] = pv[src];
                        ix[l] = pi[src];
                    }
                }
            }
            wv[w] = v[0];
            wi[w] = ix[0];
        }
        let (mut bv, mut bi) = (wv[0], wi[0]);
        for w in 1..8 {
            if gt(wv[w], bv) {
                bv = wv[w];
                bi = wi[w];
            }
        }
        top_vals[t] = bv;
        top_idxs[t] = bi;
        s[bi as usize] = NEG;
    }
    let gmax = top_vals[0];
    let mut warp_sum = [0.0f32; 8];
    for w in 0..8 {
        let mut v = [0.0f32; 32];
        for l in 0..32 {
            let tid = w * 32 + l;
            let mut ls = 0.0f32;
            ls = add(ls, ex(s[tid] - gmax));
            ls = add(ls, ex(s[tid + 256] - gmax));
            v[l] = ls;
        }
        for off in [16usize, 8, 4, 2, 1] {
            let p = v;
            for l in 0..32 {
                let src = if l + off < 32 { l + off } else { l };
                v[l] = add(p[l], p[src]);
            }
        }
        warp_sum[w] = v[0];
    }
    let mut total = 0.0f32;
    for w in warp_sum {
        total = add(total, w);
    }
    for &tv in &top_vals {
        total = add(total, ex(tv - gmax));
    }
    finish(top_vals, top_idxs, total, gmax, normalize)
}

fn finish(
    mut top_vals: Vec<f32>,
    top_idxs: Vec<u32>,
    exp_sum: f32,
    gmax: f32,
    normalize: bool,
) -> Out {
    let k = top_vals.len();
    let mut weights = vec![0.0f32; k];
    for t in 0..k {
        let sm = div(ex(top_vals[t] - gmax), exp_sum);
        if normalize {
            top_vals[t] = sm;
        } else {
            weights[t] = sm;
        }
    }
    if normalize {
        let mut sum = 0.0f32;
        for t in 0..k {
            sum = add(sum, top_vals[t]);
        }
        for t in 0..k {
            weights[t] = div(top_vals[t], sum);
        }
    }
    Out {
        idx: top_idxs,
        weights: weights.iter().map(|w| w.to_bits()).collect(),
    }
}

/// `moe_topk_softmax_batched_w32` for one token: one warp, lane L holds x[8j + c] = e(256j + 8L + c).
fn warp_kernel(logits: &[f32; 512], top_k: usize, normalize: bool) -> Out {
    let mut x = [[0.0f32; 16]; 32];
    for lane in 0..32 {
        for k in 0..16 {
            let (j, c) = (k >> 3, k & 7);
            x[lane][k] = logits[256 * j + 8 * lane + c];
        }
    }
    let mut top_vals = vec![0.0f32; top_k];
    let mut top_idxs = vec![0u32; top_k];
    for t in 0..top_k {
        let mut bv = [NEG; 32];
        let mut br = [u32::MAX; 32];
        for lane in 0..32u32 {
            let rank_hi = (lane >> 2) << 6;
            let l_base = (lane & 3) << 3;
            for k in 0..16u32 {
                let v = x[lane as usize][k as usize];
                let r = rank_hi | (rule::bitrev5(l_base + (k & 7)) << 1) | (k >> 3);
                let (b, rr) = (bv[lane as usize], br[lane as usize]);
                if gt(v, NEG) && (gt(v, b) || (eq(v, b) && r < rr)) {
                    bv[lane as usize] = v;
                    br[lane as usize] = r;
                }
            }
        }
        for d in [16usize, 8, 4, 2, 1] {
            let (pv, pr) = (bv, br);
            for l in 0..32 {
                let (ov, or) = (pv[l ^ d], pr[l ^ d]);
                if gt(ov, pv[l]) || (eq(ov, pv[l]) && or < pr[l]) {
                    bv[l] = ov;
                    br[l] = or;
                }
            }
        }
        for l in 1..32 {
            assert_eq!(
                (bv[l].to_bits(), br[l]),
                (bv[0].to_bits(), br[0]),
                "butterfly lanes disagree"
            );
        }
        let (val, idx, owner, ksel) = if br[0] == u32::MAX {
            (NEG, 0u32, 0u32, 0u32)
        } else {
            let r = br[0];
            let (w, l, j) = (r >> 6, rule::bitrev5((r >> 1) & 31), r & 1);
            (
                bv[0],
                (j << 8) | (w << 5) | l,
                (w << 2) | (l >> 3),
                (j << 3) | (l & 7),
            )
        };
        x[owner as usize][ksel as usize] = NEG;
        top_vals[t] = val;
        top_idxs[t] = idx;
    }
    let gmax = top_vals[0];
    let mut ls = [[0.0f32; 8]; 32];
    for lane in 0..32 {
        for c in 0..8 {
            let mut s = 0.0f32;
            s = add(s, ex(x[lane][c] - gmax));
            s = add(s, ex(x[lane][8 + c] - gmax));
            ls[lane][c] = s;
        }
    }
    for off in [2usize, 1] {
        let p = ls;
        for lane in 0..32 {
            let src = if lane + off < 32 { lane + off } else { lane };
            for c in 0..8 {
                ls[lane][c] = add(p[lane][c], p[src][c]);
            }
        }
    }
    for lane in 0..32 {
        for c in 0..4 {
            ls[lane][c] = add(ls[lane][c], ls[lane][c + 4]);
        }
        for c in 0..2 {
            ls[lane][c] = add(ls[lane][c], ls[lane][c + 2]);
        }
        ls[lane][0] = add(ls[lane][0], ls[lane][1]);
    }
    let mut total = 0.0f32;
    for w in 0..8 {
        total = add(total, ls[w * 4][0]);
    }
    for &tv in &top_vals {
        total = add(total, ex(tv - gmax));
    }
    finish(top_vals, top_idxs, total, gmax, normalize)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn bf16(x: f32) -> f32 {
    // round to nearest even on the top 16 bits, like the router's bf16 store
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

fn row(kind: u32, rng: &mut Rng) -> [f32; 512] {
    let mut v = [0.0f32; 512];
    for x in v.iter_mut() {
        *x = match kind {
            // realistic router logits (bf16 spacing 0.031 in [4, 8): many exact ties)
            0 => bf16((rng.unit() - 0.5) * 12.0),
            // few distinct levels: ties everywhere, across warps, lanes and halves
            1 => [-4.109375f32, -4.140625, 1.0, 2.5, 2.5, -0.0][(rng.next() % 6) as usize],
            // all equal
            2 => 0.75,
            // signed zeros, subnormals (FTZ-equal to zero) and tiny normals
            3 => [
                0.0f32,
                -0.0,
                f32::from_bits(1),
                -f32::from_bits(5),
                f32::MIN_POSITIVE,
                1e-30,
            ][(rng.next() % 6) as usize],
            // NaN-heavy: fewer candidates than top_k in some rows
            4 => {
                if rng.next().is_multiple_of(97) {
                    bf16(rng.unit())
                } else {
                    f32::NAN
                }
            }
            // -inf and huge negatives (never candidates) mixed with a few values
            5 => match rng.next() % 4 {
                0 => f32::NEG_INFINITY,
                1 => -3e30,
                2 => bf16(rng.unit() * 3.0),
                _ => -1e38,
            },
            _ => bf16(rng.unit()),
        };
    }
    v
}

#[test]
fn warp_twin_equals_block_kernel_bit_for_bit() {
    let mut rng = Rng(0x7091_0bad_cafe_f00d);
    let mut cases = 0;
    for kind in 0..6 {
        for _ in 0..60 {
            let r = row(kind, &mut rng);
            for (top_k, normalize) in [
                (10usize, true),
                (10, false),
                (1, true),
                (16, true),
                (8, false),
            ] {
                let a = block_kernel(&r, top_k, normalize);
                let b = warp_kernel(&r, top_k, normalize);
                assert_eq!(a, b, "kind {kind} top_k {top_k} normalize {normalize}");
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 6 * 60 * 5);
}

#[test]
fn tie_order_is_not_lowest_index_and_the_model_knows_it() {
    // Equal maxima at experts 50 (thread 50: warp 1, lane 18) and 300 (thread 44: warp 1,
    // lane 12, j = 1): the block kernel keeps 300 (lane 12 = 01100 -> bitrev 00110 = 6 beats lane
    // 18 = 10010 -> 01001 = 9), not the lower index.
    let mut r = [0.0f32; 512];
    r[50] = 3.0;
    r[300] = 3.0;
    let a = block_kernel(&r, 1, true);
    assert_eq!(a.idx, vec![300]);
    assert!(rule::rank(300) < rule::rank(50));
    assert_eq!(warp_kernel(&r, 1, true), a);
}

#[test]
fn rank_is_a_bijection_and_owner_matches_the_lane_layout() {
    let mut seen = [false; 512];
    for e in 0..512u32 {
        let r = rule::rank(e);
        assert!(r < 512 && !seen[r as usize]);
        seen[r as usize] = true;
        assert_eq!(rule::expert_of_rank(r), e);
        let (lane, slot) = rule::owner_of(e);
        let (j, c) = (slot >> 3, slot & 7);
        assert_eq!(256 * j + 8 * lane + c, e);
        // the kernel's rank expression for that lane/slot
        let k = slot;
        let kr = ((lane >> 2) << 6) | (rule::bitrev5(((lane & 3) << 3) + (k & 7)) << 1) | (k >> 3);
        assert_eq!(kr, r, "expert {e}");
    }
}

#[test]
fn selector_and_admission() {
    assert_eq!(rule::parse(None), Ok(false));
    assert_eq!(rule::parse(Some("0")), Ok(false));
    assert_eq!(rule::parse(Some("1")), Ok(true));
    assert!(rule::parse(Some("on")).is_err());
    assert_eq!(rule::refusal(512, 10, 8192, 0x1000), None);
    assert_eq!(
        rule::refusal(256, 10, 8192, 0x1000),
        Some("num_experts_not_512")
    );
    assert_eq!(
        rule::refusal(512, 0, 8192, 0x1000),
        Some("top_k_not_1_to_16")
    );
    assert_eq!(
        rule::refusal(512, 17, 8192, 0x1000),
        Some("top_k_not_1_to_16")
    );
    assert_eq!(rule::refusal(512, 10, 0, 0x1000), Some("no_tokens"));
    assert_eq!(
        rule::refusal(512, 10, 8192, 0x1008),
        Some("logits_not_16b_aligned")
    );
    assert_eq!(rule::grid(8192), 1024);
    assert_eq!(rule::grid(8193), 1025);
    assert_eq!(rule::grid(1), 1);
}

fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn section<'a>(src: &'a str, start: &str, end: &str) -> &'a str {
    let a = src.find(start).unwrap_or_else(|| panic!("missing {start}"));
    let b = src[a..].find(end).map(|o| a + o).unwrap_or(src.len());
    &src[a..b]
}

#[test]
fn reference_kernel_still_has_the_modelled_rules() {
    let block = compact(section(
        KERNEL,
        "extern \"C\" __global__ void moe_topk_softmax_batched(",
        "#define MOE_TOPK_W_EXPERTS",
    ));
    for needle in [
        "for(unsignedinti=tid;i<actual_n;i+=BLOCK_SIZE){floatv=s_vals[i];if(v>local_max){",
        "floatlocal_max=-1e30f;unsignedintlocal_idx=0;",
        "floatother_val=__shfl_down_sync(0xFFFFFFFF,local_max,offset);",
        "if(other_val>local_max){",
        "for(unsignedintw=1;w<num_warps;w++){if(s_warp_val[w]>best_val){",
        "s_vals[best_idx]=-1e30f;",
        "local_sum+=__expf(s_vals[i]-global_max);",
        "local_sum+=__shfl_down_sync(0xFFFFFFFF,local_sum,offset);",
        "total+=s_warp_val[w];",
        "total+=__expf(s_top_vals[t]-global_max);",
        "floatsoftmax_weight=__expf(s_top_vals[t]-global_max)/exp_sum;",
        "topk_sum+=s_top_vals[t];",
        "my_weights[t]=s_top_vals[t]/topk_sum;",
    ] {
        assert!(
            block.contains(needle),
            "moe_topk_softmax_batched changed: {needle}"
        );
    }
}

#[test]
fn twin_kernel_implements_the_model() {
    let twin = compact(section(
        KERNEL,
        "extern \"C\" __global__ void __launch_bounds__(256) moe_topk_softmax_batched_w32(",
        "\u{0}",
    ));
    for needle in [
        "constunsignedinttoken=blockIdx.x*8u+warp_id;if(token>=num_tokens)return;",
        "constuint4w0=row[lane];constuint4w1=row[32u+lane];",
        "x[2*i]=__uint_as_float(u[i]<<16);x[2*i+1]=__uint_as_float(u[i]&0xffff0000u);",
        "constunsignedintrank_hi=(lane>>2)<<6;constunsignedintl_base=(lane&3u)<<3;",
        "constunsignedintr=rank_hi|(moe_topk_w_bitrev5(l_base+(k&7))<<1)|(unsignedint)(k>>3);",
        "if(v>-1e30f&&(v>bv||(v==bv&&r<br))){",
        "if(ov>bv||(ov==bv&&orr<br)){",
        "idx=(j<<8)|(w<<5)|l;owner=(w<<2)|(l>>3);ksel=(j<<3)|(l&7u);",
        "if((unsignedint)k==ksel)x[k]=-1e30f;",
        "floatlocal_sum=0.0f;local_sum+=__expf(x[c]-global_max);local_sum+=__expf(x[8+c]-global_max);",
        "ls[c]+=__shfl_down_sync(0xffffffffu,ls[c],2);",
        "ls[c]+=__shfl_down_sync(0xffffffffu,ls[c],1);",
        "for(intc=0;c<4;++c)ls[c]+=ls[c+4];",
        "for(intc=0;c<2;++c)ls[c]+=ls[c+2];ls[0]+=ls[1];",
        "warp_sums[w]=__shfl_sync(0xffffffffu,ls[0],w*4);",
        "total+=warp_sums[w];",
        "total+=__expf(s_top_vals[warp_id][t]-global_max);",
        "floatsoftmax_weight=__expf(s_top_vals[warp_id][t]-global_max)/exp_sum;",
        "my_weights[t]=s_top_vals[warp_id][t]/topk_sum;",
    ] {
        assert!(twin.contains(needle), "twin lost: {needle}");
    }
    assert!(compact(KERNEL).contains("returnx>>27;") || KERNEL.contains("__brev(x) >> 27"));
}

#[test]
fn dispatch_is_default_off_softmax_only_and_receipted() {
    let fwd = compact(FORWARD);
    // The sigmoid/bias router keeps its kernel; the softmax branch tries the twin first.
    assert!(
        fwd.contains("ifletSome(bias)=self.correction_bias_dev{ops::moe_topk_sigmoid_batched(")
    );
    assert!(fwd.contains("}elseif!super::qwen4_topk_warp::try_launch(ctx.gpu,gate_logits,indices_dev,weights_dev,num_experts,top_k,ctx.config.norm_topk_prob,n,stream,)?{"));
    assert_eq!(FORWARD.matches("ops::moe_topk_softmax_batched(").count(), 1);
    let glue = compact(GLUE);
    assert!(glue.contains("if!selected()?{returnOk(false);}"));
    assert!(
        glue.contains(
            "ifletSome(reason)=rule::refusal(num_experts,top_k,num_tokens,gate_logits.0)"
        )
    );
    assert!(glue.contains("MOE_TOPK_WARP_ENGAGEDselector={}kernel={}"));
    assert!(glue.contains("MOE_TOPK_WARP_REFUSEDselector={}reason={reason}"));
    assert!(glue.contains("gpu.kernel(\"moe_topk\",rule::KERNEL)"));
    assert!(MOD.contains("mod qwen4_topk_warp;") && MOD.contains("mod qwen4_topk_warp_rule;"));
    assert_eq!(rule::SELECTOR, "ATLAS_QWEN4_PREFILL_MOE_TOPK_WARP");
    assert_eq!(rule::KERNEL, "moe_topk_softmax_batched_w32");
}
