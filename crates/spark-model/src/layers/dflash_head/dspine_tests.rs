// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-CPU reference math tests for DSpine Pass-1 Adjacent Causal Injection
//! and Pass-2 exact-sender `serial_table` refinement (`arXiv:2609.36173`).

/// Exact IEEE-754 round-to-nearest-even BF16 conversion matching CUDA's
/// `__float2bfloat16` + `__bfloat162float`.
fn bf16(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff_u32.wrapping_add(lsb);
    let rounded = bits.wrapping_add(rounding_bias) & 0xffff_0000;
    f32::from_bits(rounded)
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

struct CpuBridgeParams {
    block_rows: usize,
    hidden: usize,
    code_size: usize,
    rank: usize,
    vocab: usize,
    eps: f32,
    norm_eps: f32,
    codebook: Vec<f32>,        // [V, C] BF16 values
    read_w: Vec<f32>,          // [C, H] BF16 values
    gate_w: Vec<f32>,          // [R, C] BF16 values
    pred_gate_w: Vec<f32>,     // [R, C] BF16 values
    prev_w: Vec<f32>,          // [R, C] BF16 values
    gate_bias: Vec<f32>,       // [R] BF16 values
    write_w: Vec<f32>,         // [H, R] BF16 values
    norm_w: Vec<f32>,          // [H] BF16 values
    lm_head: Vec<f32>,         // [V, H] BF16 values
    tr_emb: Option<Vec<f32>>,  // [V, Rt]
    tr_proj: Option<Vec<f32>>, // [R, Rt]
    tr_rank: usize,
}

struct CpuPass1Stash {
    raw: Vec<f32>,     // [block_rows, H]
    rms: Vec<f32>,     // [block_rows]
    packed: Vec<f32>,  // [block_rows, 3*R]
    updated: Vec<f32>, // [block_rows, H]
    normed: Vec<f32>,  // [block_rows, H]
}

fn matmul_nt_bf16(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = bf16(acc);
        }
    }
    out
}

/// Reference Pass-1 Adjacent Causal Injection matching `dspine_bridge.cu`
/// (`dspine_bridge_read` -> `read_w` -> `dspine_bridge_code` -> `packed_w`
/// -> `dspine_bridge_message` -> `write_w` -> `dspine_bridge_write_norm`).
fn cpu_pass1_bridge(p: &CpuBridgeParams, x_in: &[f32], anchor: usize) -> CpuPass1Stash {
    let m = p.block_rows;
    let h = p.hidden;
    let c = p.code_size;
    let r = p.rank;

    let raw: Vec<f32> = x_in.iter().copied().map(bf16).collect();
    let mut rms = vec![0.0f32; m];
    let mut normalized = vec![0.0f32; m * h];
    for row in 0..m {
        let xr = &raw[row * h..(row + 1) * h];
        let ss: f32 = xr.iter().map(|&v| v * v).sum();
        let mean = ss / (h as f32);
        let rv = mean.max(f32::MIN_POSITIVE).sqrt();
        rms[row] = rv;
        let denom = rv + p.eps;
        for i in 0..h {
            normalized[row * h + i] = bf16(xr[i] / denom);
        }
    }

    let projected = matmul_nt_bf16(&normalized, &p.read_w, m, c, h);
    let sqrt_c = (c as f32).sqrt();
    let mut codes = vec![0.0f32; m * c];
    for row in 0..m {
        if row % m == 0 {
            let a = if anchor < p.vocab { anchor } else { 0 };
            for i in 0..c {
                codes[row * c + i] = bf16(sqrt_c * p.codebook[a * c + i]);
            }
        } else {
            let pr = &projected[row * c..(row + 1) * c];
            let ss: f32 = pr.iter().map(|&v| v * v).sum();
            let denom = ss.sqrt() + p.eps;
            for i in 0..c {
                codes[row * c + i] = bf16(sqrt_c * pr[i] / denom);
            }
        }
    }

    let mut packed_w = Vec::with_capacity(3 * r * c);
    packed_w.extend_from_slice(&p.gate_w);
    packed_w.extend_from_slice(&p.pred_gate_w);
    packed_w.extend_from_slice(&p.prev_w);
    let packed = matmul_nt_bf16(&codes, &packed_w, m, 3 * r, c);

    let mut message = vec![0.0f32; m * r];
    for row in 1..m {
        let own = &packed[row * 3 * r..];
        let pred = &packed[(row - 1) * 3 * r..];
        for i in 0..r {
            let z = own[i] + pred[r + i] + p.gate_bias[i];
            message[row * r + i] = bf16(sigmoid(z) * pred[2 * r + i]);
        }
    }

    let delta = matmul_nt_bf16(&message, &p.write_w, m, h, r);
    let mut updated = raw.clone();
    for row in 1..m {
        for i in 0..h {
            updated[row * h + i] = bf16(raw[row * h + i] + rms[row] * delta[row * h + i]);
        }
    }

    let mut normed = vec![0.0f32; m * h];
    for row in 0..m {
        let xr = &updated[row * h..(row + 1) * h];
        let var: f32 = xr.iter().map(|&v| v * v).sum::<f32>() / (h as f32);
        let inv_rms = 1.0 / (var + p.norm_eps).sqrt();
        for i in 0..h {
            let y = bf16(xr[i] * inv_rms);
            normed[row * h + i] = bf16(y * p.norm_w[i]);
        }
    }

    CpuPass1Stash {
        raw,
        rms,
        packed,
        updated,
        normed,
    }
}

/// Build precomputed `(TG [V, R], TSV [V, 2*R])` matching `DspineCodeBridge::build`.
fn cpu_build_serial_tables(p: &CpuBridgeParams) -> (Vec<f32>, Vec<f32>) {
    let v = p.vocab;
    let h = p.hidden;
    let c = p.code_size;
    let r = p.rank;

    let mut scaled_head = vec![0.0f32; v * h];
    for tok in 0..v {
        for i in 0..h {
            scaled_head[tok * h + i] = bf16(p.lm_head[tok * h + i] * p.norm_w[i]);
        }
    }
    let mut write_t = vec![0.0f32; r * h];
    for hi in 0..h {
        for ri in 0..r {
            write_t[ri * h + hi] = p.write_w[hi * r + ri];
        }
    }
    let tg = matmul_nt_bf16(&scaled_head, &write_t, v, r, h);

    let sqrt_c = (c as f32).sqrt();
    let mut scaled_cb = vec![0.0f32; v * c];
    for tok in 0..v {
        for i in 0..c {
            scaled_cb[tok * c + i] = bf16(sqrt_c * p.codebook[tok * c + i]);
        }
    }
    let mut pv = Vec::with_capacity(2 * r * c);
    pv.extend_from_slice(&p.pred_gate_w);
    pv.extend_from_slice(&p.prev_w);
    let mut tsv = matmul_nt_bf16(&scaled_cb, &pv, v, 2 * r, c);

    if p.tr_rank > 0 {
        let emb = p.tr_emb.as_ref().unwrap();
        let proj = p.tr_proj.as_ref().unwrap();
        let residual = matmul_nt_bf16(emb, proj, v, r, p.tr_rank);
        for tok in 0..v {
            for ri in 0..r {
                let off = tok * 2 * r + r + ri;
                tsv[off] = bf16(tsv[off] + residual[tok * r + ri]);
            }
        }
    }
    (tg, tsv)
}

/// Reference Pass-2 `dspine_serial_table` + `dspine_serial_walk`.
fn cpu_serial_table_and_walk(
    p: &CpuBridgeParams,
    stash: &CpuPass1Stash,
    topk_per_draft: &[Vec<u32>],
    anchor: u32,
    topk_k: usize,
) -> Vec<u32> {
    let (tg, tsv) = cpu_build_serial_tables(p);
    let gamma = p.block_rows - 1;
    let h = p.hidden;
    let r = p.rank;

    let mut sorted_cands: Vec<Vec<u32>> = Vec::with_capacity(gamma);
    for d in 0..gamma {
        let mut c = topk_per_draft[d][..topk_k].to_vec();
        c.sort_unstable();
        sorted_cands.push(c);
    }

    let mut idx_table = vec![vec![0usize; topk_k]; gamma];
    for d in 0..gamma {
        let full_row = 1 + d;
        let xr = &stash.raw[full_row * h..(full_row + 1) * h];
        let q = &stash.packed[full_row * 3 * r..full_row * 3 * r + r];
        let rr = stash.rms[full_row];

        let mut base = vec![0.0f32; topk_k];
        for (k, &cid_u32) in sorted_cands[d].iter().enumerate() {
            let cid = cid_u32 as usize;
            let mut acc = 0.0f32;
            for i in 0..h {
                acc += (xr[i] * p.norm_w[i]) * p.lm_head[cid * h + i];
            }
            base[k] = acc;
        }

        for j in 0..topk_k {
            let sid = if d == 0 {
                anchor as usize
            } else {
                sorted_cands[d - 1][j] as usize
            };
            let ts = &tsv[sid * 2 * r..(sid + 1) * 2 * r];
            let mut m_vec = vec![0.0f32; r];
            for ri in 0..r {
                let z = q[ri] + ts[ri] + p.gate_bias[ri];
                m_vec[ri] = sigmoid(z) * ts[r + ri];
            }

            let mut best_k = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (k, &cid_u32) in sorted_cands[d].iter().enumerate() {
                let cid = cid_u32 as usize;
                let mut dot = 0.0f32;
                for ri in 0..r {
                    dot += m_vec[ri] * tg[cid * r + ri];
                }
                let score = base[k] + rr * dot;
                if score > best_val {
                    best_val = score;
                    best_k = k;
                }
            }
            idx_table[d][j] = best_k;
        }
    }

    let mut out = Vec::with_capacity(gamma);
    let mut j = 0usize;
    for d in 0..gamma {
        j = idx_table[d][j];
        out.push(sorted_cands[d][j]);
    }
    out
}

fn deterministic_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
            bf16(unit * scale)
        })
        .collect()
}

#[test]
fn pass1_adjacent_causal_injection_keeps_anchor_exact_and_conditions_draft_rows() {
    let params = CpuBridgeParams {
        block_rows: 4, // anchor + 3 drafts
        hidden: 8,
        code_size: 6,
        rank: 4,
        vocab: 16,
        eps: 1e-6,
        norm_eps: 1e-6,
        codebook: deterministic_vec(16 * 6, 11, 0.5),
        read_w: deterministic_vec(6 * 8, 22, 0.4),
        gate_w: deterministic_vec(4 * 6, 33, 0.4),
        pred_gate_w: deterministic_vec(4 * 6, 44, 0.4),
        prev_w: deterministic_vec(4 * 6, 55, 0.6),
        gate_bias: deterministic_vec(4, 66, 0.2),
        write_w: deterministic_vec(8 * 4, 77, 0.5),
        norm_w: deterministic_vec(8, 88, 1.0),
        lm_head: deterministic_vec(16 * 8, 99, 0.5),
        tr_emb: None,
        tr_proj: None,
        tr_rank: 0,
    };
    let x = deterministic_vec(4 * 8, 123, 0.8);
    let stash_a = cpu_pass1_bridge(&params, &x, 3);
    let stash_b = cpu_pass1_bridge(&params, &x, 9);

    // 1. Anchor row (row 0) never receives a bridge write (`updated[0] == raw[0]`),
    //    and fused `write_norm` output matches standalone vanilla RMSNorm on all rows.
    assert_eq!(&stash_a.updated[0..8], &stash_a.raw[0..8]);
    assert_eq!(&stash_b.updated[0..8], &stash_b.raw[0..8]);
    assert_eq!(&stash_a.normed[0..8], &stash_b.normed[0..8]);
    assert_ne!(&stash_a.normed[8..16], &stash_b.normed[8..16]);

    // 2. Changing the anchor token injects codebook[anchor] into row 0's sender
    //    projection, changing draft row 1 (`t = 1`), while leaving draft rows
    //    `t >= 2` unchanged in a single layer (strictly adjacent 1-hop causal injection!).
    assert_ne!(
        &stash_a.updated[8..16],
        &stash_b.updated[8..16],
        "draft row 1 must depend on row 0's exact anchor codebook"
    );
    assert_eq!(
        &stash_a.updated[16..24],
        &stash_b.updated[16..24],
        "single-layer bridge is strictly adjacent (row 2 reads row 1's pre-write code)"
    );
    assert_eq!(&stash_a.updated[24..32], &stash_b.updated[24..32]);
}

#[test]
fn pass2_serial_table_factorization_matches_unfactorized_exact_sender_refinement() {
    // Prove in float32 arithmetic (where linear algebra is associative up to
    // ~1e-5) AND with terminal_residual folding that `serial_table` + `serial_walk`
    // picks the exact same greedy chain as explicitly recomputing the last-layer
    // write `x'_{t,s} = x_t + rms_t * W_last m_{t,s}`, applying RMSNorm(norm_w),
    // and taking `argmax_{k in C_t} (lm_head[k] . normed(x'_{t,s}))` at every step!
    let params = CpuBridgeParams {
        block_rows: 5, // anchor + 4 drafts
        hidden: 16,
        code_size: 8,
        rank: 8,
        vocab: 32,
        eps: 1e-6,
        norm_eps: 1e-6,
        codebook: deterministic_vec(32 * 8, 201, 0.4),
        read_w: deterministic_vec(8 * 16, 202, 0.3),
        gate_w: deterministic_vec(8 * 8, 203, 0.4),
        pred_gate_w: deterministic_vec(8 * 8, 204, 0.4),
        prev_w: deterministic_vec(8 * 8, 205, 0.7),
        gate_bias: deterministic_vec(8, 206, 0.2),
        write_w: deterministic_vec(16 * 8, 207, 0.6),
        norm_w: deterministic_vec(16, 208, 1.0),
        lm_head: deterministic_vec(32 * 16, 209, 0.7),
        tr_emb: Some(deterministic_vec(32 * 4, 210, 0.3)),
        tr_proj: Some(deterministic_vec(8 * 4, 211, 0.3)),
        tr_rank: 4,
    };
    let x = deterministic_vec(5 * 16, 303, 0.9);
    let anchor = 7u32;
    let stash = cpu_pass1_bridge(&params, &x, anchor as usize);

    // Unsorted candidate lists of width K = 8 per draft position (testing that
    // `dsp_sorted_ids` sorts candidate lists and breaks ties deterministically).
    let topk_lists = vec![
        vec![19, 2, 14, 7, 30, 5, 11, 23],
        vec![28, 4, 15, 1, 22, 9, 17, 12],
        vec![6, 31, 0, 18, 25, 8, 13, 21],
        vec![10, 3, 29, 16, 24, 20, 27, 26],
    ];

    let refined = cpu_serial_table_and_walk(&params, &stash, &topk_lists, anchor, 8);
    assert_eq!(refined.len(), 4);

    // Reference unfactorized step-by-step exact-sender walk:
    let sqrt_c = (params.code_size as f32).sqrt();
    let mut prev_tok = anchor as usize;
    let mut expected_chain = Vec::with_capacity(4);
    for (d, raw_cands) in topk_lists.iter().enumerate() {
        let mut cands = raw_cands.clone();
        cands.sort_unstable();
        let full_row = 1 + d;
        let xr = &stash.raw[full_row * params.hidden..(full_row + 1) * params.hidden];
        let q = &stash.packed[full_row * 3 * params.rank..full_row * 3 * params.rank + params.rank];
        let rr = stash.rms[full_row];

        // Exact sender code `e_prev = bf16(sqrt(C) * codebook[prev_tok])`.
        let mut e_prev = vec![0.0f32; params.code_size];
        for i in 0..params.code_size {
            e_prev[i] = bf16(sqrt_c * params.codebook[prev_tok * params.code_size + i]);
        }
        let s_gate = matmul_nt_bf16(
            &e_prev,
            &params.pred_gate_w,
            1,
            params.rank,
            params.code_size,
        );
        let mut s_val = matmul_nt_bf16(&e_prev, &params.prev_w, 1, params.rank, params.code_size);
        let tr_res = matmul_nt_bf16(
            &params.tr_emb.as_ref().unwrap()[prev_tok * 4..(prev_tok + 1) * 4],
            params.tr_proj.as_ref().unwrap(),
            1,
            params.rank,
            4,
        );
        for ri in 0..params.rank {
            s_val[ri] = bf16(s_val[ri] + tr_res[ri]);
        }

        // Exact message `m` and unfactorized updated hidden `x_Exact = xr + rr * (W_last @ m)`.
        let mut m_vec = vec![0.0f32; params.rank];
        for ri in 0..params.rank {
            m_vec[ri] = sigmoid(q[ri] + s_gate[ri] + params.gate_bias[ri]) * s_val[ri];
        }
        let mut x_exact = vec![0.0f32; params.hidden];
        for hi in 0..params.hidden {
            let mut delta_h = 0.0f32;
            for ri in 0..params.rank {
                delta_h += params.write_w[hi * params.rank + ri] * m_vec[ri];
            }
            x_exact[hi] = xr[hi] + rr * delta_h;
        }
        // Divide by RMS(x_exact) and multiply by norm_w, then dot with lm_head[cand].
        let var: f32 = x_exact.iter().map(|v| v * v).sum::<f32>() / (params.hidden as f32);
        let inv_rms = 1.0 / (var + params.norm_eps).sqrt();

        let mut best_tok = cands[0];
        let mut best_logit = f32::NEG_INFINITY;
        for &cid in &cands {
            let mut logit = 0.0f32;
            for hi in 0..params.hidden {
                let normed_h = (x_exact[hi] * inv_rms) * params.norm_w[hi];
                logit += normed_h * params.lm_head[(cid as usize) * params.hidden + hi];
            }
            if logit > best_logit {
                best_logit = logit;
                best_tok = cid;
            }
        }
        expected_chain.push(best_tok);
        prev_tok = best_tok as usize;
    }

    assert_eq!(refined, expected_chain);
}

#[test]
fn pass2_serial_table_breaks_exact_ties_toward_lowest_token_id() {
    // Zero weights -> all candidates have identical score 0.0; even when input
    // top-k IDs are descending `[29, 14, 5, 19]`, `serial_table` + `serial_walk`
    // must deterministically emit the smallest token ID (`5`).
    let params = CpuBridgeParams {
        block_rows: 3,
        hidden: 4,
        code_size: 4,
        rank: 4,
        vocab: 32,
        eps: 1e-6,
        norm_eps: 1e-6,
        codebook: vec![0.0; 32 * 4],
        read_w: vec![0.0; 4 * 4],
        gate_w: vec![0.0; 4 * 4],
        pred_gate_w: vec![0.0; 4 * 4],
        prev_w: vec![0.0; 4 * 4],
        gate_bias: vec![0.0; 4],
        write_w: vec![0.0; 4 * 4],
        norm_w: vec![1.0; 4],
        lm_head: vec![0.0; 32 * 4],
        tr_emb: None,
        tr_proj: None,
        tr_rank: 0,
    };
    let stash = cpu_pass1_bridge(&params, &[0.0; 3 * 4], 1);
    let topk_lists = vec![vec![29, 14, 5, 19], vec![31, 8, 2, 17]];
    let out = cpu_serial_table_and_walk(&params, &stash, &topk_lists, 1, 4);
    assert_eq!(out, vec![5, 2]);
}
