// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only oracle for the official Qwen3.8 DFlash2 BF16 chronology.
//!
//! This deliberately does not bless the current CUDA algebra. It models
//! z-lab/dflash commit 07ebd93, `model.py:478-489,524-547`, and demonstrates
//! inputs where the current reassociated conv and direct triple-product
//! selector choose different results. Runtime correction requires a separate
//! CUDA tranche and native parity gate.

// Moved under `dflash2_lexer/`: a .rs file directly in `tests/` is compiled
// as its OWN test binary as well as being included here, and at a crate root
// `use super::` / `pub(super)` do not resolve — there is nothing above it.
// Cargo does not auto-compile files in a tests/ SUBDIRECTORY, so the module
// is reachable from here and nowhere else.
#[path = "dflash2_lexer/dflash2_bf16_reference_lexer.rs"]
mod python_receipt;

use python_receipt::python_canonical;

const GAMMA: usize = 7;
const RANK: usize = 256;
const TOP_K: usize = 16;
const VOCAB: usize = 248_320;

const CONV_CUDA: &str = include_str!("../../../kernels/gb10/common/dflash2_conv.cu");
const SELECTOR_CUDA: &str = include_str!("../../../kernels/gb10/common/dflash2_selector.cu");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OfficialIdentity {
    gamma: usize,
    rank: usize,
    top_k: usize,
    vocab: usize,
}

impl OfficialIdentity {
    fn validate(self) -> Result<OfficialExtents, &'static str> {
        if self
            != (Self {
                gamma: GAMMA,
                rank: RANK,
                top_k: TOP_K,
                vocab: VOCAB,
            })
        {
            return Err("not the official gamma7/rank256/top16/vocab248320 profile");
        }
        Ok(OfficialExtents {
            candidate_items: self
                .gamma
                .checked_mul(self.top_k)
                .ok_or("candidate extent overflow")?,
            codebook_items: self
                .vocab
                .checked_mul(self.rank)
                .ok_or("codebook extent overflow")?,
        })
    }

    fn validate_candidates(self, anchor: u32, candidates: &[u32]) -> Result<(), &'static str> {
        let extents = self.validate()?;
        if candidates.len() != extents.candidate_items {
            return Err("candidate extent does not equal gamma*top_k");
        }
        if usize::try_from(anchor).map_err(|_| "anchor conversion overflow")? >= self.vocab
            || candidates.iter().any(|&token| token as usize >= self.vocab)
        {
            return Err("selector token is outside official vocab");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OfficialExtents {
    candidate_items: usize,
    codebook_items: usize,
}

/// IEEE f32 to BF16, round-to-nearest-even. NaNs stay NaNs rather than
/// rounding into infinity.
fn bf16_rne(value: f32) -> f32 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    let rounded = if exponent == 0x7f80_0000 && mantissa != 0 {
        (bits >> 16) | 0x0040
    } else {
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16
    };
    f32::from_bits((rounded & 0xffff) << 16)
}

/// Official `_grouped_dynamic_convolve` chronology for one channel across
/// its two causal taps. Inputs represent values already stored as BF16.
fn official_conv_channel(base: [f32; 2], dynamic: [f32; 2], values: [f32; 2]) -> f32 {
    let mut output = bf16_rne(0.0);
    for offset in 0..2 {
        let base = bf16_rne(base[offset]);
        let dynamic = bf16_rne(dynamic[offset]);
        let value = bf16_rne(values[offset]);
        let base_product = bf16_rne(base * value);
        output = bf16_rne(output + base_product);
        output = bf16_rne(output + dynamic * value);
    }
    output
}

/// Diagnostic model of current CUDA: `(base + dynamic) * value` accumulates
/// in FP32 and rounds once. Kept separate so a delta is a failure signal, not
/// an accepted tolerance.
fn current_cuda_conv_channel(base: [f32; 2], dynamic: [f32; 2], values: [f32; 2]) -> f32 {
    let mut accumulator = 0.0_f32;
    for offset in 0..2 {
        accumulator +=
            (bf16_rne(base[offset]) + bf16_rne(dynamic[offset])) * bf16_rne(values[offset]);
    }
    bf16_rne(accumulator)
}

/// The official BF16 `einsum` returns a BF16 tensor. Accumulation is modeled
/// in FP32, then the observable tensor result is rounded once to BF16.
fn official_bf16_einsum(left: &[f32; RANK], right: &[f32; RANK]) -> f32 {
    let mut accumulator = 0.0_f32;
    for rank in 0..RANK {
        accumulator += bf16_rne(left[rank]) * bf16_rne(right[rank]);
    }
    bf16_rne(accumulator)
}

/// Both inputs and the result of the official unary-plus-einsum expression
/// remain BF16 because the benchmark loads target and drafter as BF16.
fn official_bf16_add(left: f32, right: f32) -> f32 {
    bf16_rne(bf16_rne(left) + bf16_rne(right))
}

/// Official selector materializes `predecessor * hidden` as BF16 before the
/// BF16-result einsum and BF16 unary addition.
fn official_selector_scores(
    unary: &[f32; TOP_K],
    predecessor: &[f32; RANK],
    hidden: &[f32; RANK],
    successors: &[[f32; RANK]; TOP_K],
) -> [f32; TOP_K] {
    let mut gate = [0.0_f32; RANK];
    for rank in 0..RANK {
        gate[rank] = bf16_rne(bf16_rne(predecessor[rank]) * bf16_rne(hidden[rank]));
    }
    let mut scores = [0.0_f32; TOP_K];
    for candidate in 0..TOP_K {
        let contribution = official_bf16_einsum(&gate, &successors[candidate]);
        scores[candidate] = official_bf16_add(unary[candidate], contribution);
    }
    scores
}

/// Diagnostic model of current CUDA: no BF16 gate exists; each thread sums
/// the direct FP32 triple product.
fn current_cuda_selector_scores(
    unary: &[f32; TOP_K],
    predecessor: &[f32; RANK],
    hidden: &[f32; RANK],
    successors: &[[f32; RANK]; TOP_K],
) -> [f32; TOP_K] {
    let mut scores = [0.0_f32; TOP_K];
    for candidate in 0..TOP_K {
        let mut contribution = 0.0_f32;
        for rank in 0..RANK {
            contribution += bf16_rne(predecessor[rank])
                * bf16_rne(hidden[rank])
                * bf16_rne(successors[candidate][rank]);
        }
        scores[candidate] = bf16_rne(unary[candidate]) + contribution;
    }
    scores
}

/// Candidate order is an input from `topk(sorted=False)`. Within that row,
/// greedy selection is first-index-on-tie. Nonfinite scores reject because
/// PyTorch/CUDA NaN ordering has not yet been proved equivalent.
fn greedy_candidate_index(scores: &[f32]) -> Result<usize, &'static str> {
    if scores.len() != TOP_K {
        return Err("selector row must contain exactly top16 scores");
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("selector scores must be finite");
    }
    let mut best = 0;
    for candidate in 1..TOP_K {
        if scores[candidate] > scores[best] {
            best = candidate;
        }
    }
    Ok(best)
}

#[test]
#[allow(clippy::excessive_precision)]
fn bf16_rne_uses_ties_to_even_and_preserves_nan() {
    assert_eq!(bf16_rne(1.003_906_25).to_bits(), 1.0_f32.to_bits());
    assert_eq!(bf16_rne(1.011_718_75).to_bits(), 1.015_625_f32.to_bits());
    assert!(bf16_rne(f32::from_bits(0x7f80_0001)).is_nan());
}

#[test]
fn upstream_python_normalizer_ignores_only_layout_whitespace() {
    let canonical = "scores = (unary + torch.einsum(\"br,bkr->bk\", left, right))";
    let indented = "scores=(\n unary + torch . einsum ( \"br,bkr->bk\" , left , right )\n)";
    assert_eq!(python_canonical(canonical), python_canonical(indented));
    let expected = python_canonical(canonical);
    for mutant in [
        canonical.replace("unary", "binary"),
        canonical.replace("left, right", "right, left"),
    ] {
        assert_ne!(expected, python_canonical(&mutant));
    }
}

#[test]
fn conv_reassociation_is_a_discriminating_failure() {
    let base = [-8.0, -0.0625];
    let dynamic = [-0.03125, -1.0];
    let values = [-1.0, 1.007_812_5];
    let official = official_conv_channel(base, dynamic, values);
    let current = current_cuda_conv_channel(base, dynamic, values);
    assert_eq!(official.to_bits(), 6.9375_f32.to_bits());
    assert_eq!(current.to_bits(), 6.96875_f32.to_bits());
    assert_ne!(official.to_bits(), current.to_bits());
}

#[test]
#[allow(clippy::excessive_precision)]
fn selector_bf16_gate_materialization_can_flip_the_path() {
    let mut unary = [-100.0_f32; TOP_K];
    unary[0] = 0.0;
    unary[1] = 0.000_030_517_578_125;
    let mut predecessor = [0.0_f32; RANK];
    let mut hidden = [0.0_f32; RANK];
    let mut successors = [[0.0_f32; RANK]; TOP_K];
    predecessor[0] = 1.007_812_5;
    hidden[0] = 1.007_812_5;
    successors[0][0] = 1.0;
    predecessor[1] = 1.0;
    hidden[1] = 1.015_625;
    successors[0][1] = -1.0;

    let official = official_selector_scores(&unary, &predecessor, &hidden, &successors);
    let current = current_cuda_selector_scores(&unary, &predecessor, &hidden, &successors);
    assert_eq!(bf16_rne(unary[1]).to_bits(), unary[1].to_bits());
    assert_eq!(official[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(official[1].to_bits(), unary[1].to_bits());
    assert_eq!(current[0].to_bits(), 0.000_061_035_156_25_f32.to_bits());
    assert!(current[0] > unary[1] && official[0] < unary[1]);
    assert_eq!(greedy_candidate_index(&official), Ok(1));
    assert_eq!(greedy_candidate_index(&current), Ok(0));
}

#[test]
fn selector_pins_bf16_einsum_and_output_add_rounding() {
    let mut gate = [0.0_f32; RANK];
    let mut successor = [0.0_f32; RANK];
    gate[0] = 1.015_625;
    successor[0] = 1.007_812_5;
    assert_eq!(
        official_bf16_einsum(&gate, &successor).to_bits(),
        1.023_437_5_f32.to_bits()
    );
    assert_eq!(
        official_bf16_add(1.0, 0.003_906_25).to_bits(),
        1.0_f32.to_bits()
    );
}

#[test]
fn cutoff_tie_and_nan_policy_is_explicitly_fail_closed() {
    let mut scores = [0.0_f32; TOP_K];
    scores[0] = 4.0;
    scores[15] = 4.0;
    assert_eq!(greedy_candidate_index(&scores), Ok(0));
    assert!(greedy_candidate_index(&scores[..15]).is_err());
    let mut seventeen = scores.to_vec();
    seventeen.push(100.0);
    assert!(greedy_candidate_index(&seventeen).is_err());
    scores[7] = f32::NAN;
    assert!(greedy_candidate_index(&scores).is_err());
    scores[7] = f32::INFINITY;
    assert!(greedy_candidate_index(&scores).is_err());
}

#[test]
fn official_identity_and_extents_reject_old_microtest_geometry() {
    let official = OfficialIdentity {
        gamma: GAMMA,
        rank: RANK,
        top_k: TOP_K,
        vocab: VOCAB,
    };
    assert_eq!(
        official.validate(),
        Ok(OfficialExtents {
            candidate_items: 112,
            codebook_items: 63_569_920,
        })
    );
    let candidates = vec![VOCAB as u32 - 1; 112];
    assert!(
        official
            .validate_candidates(VOCAB as u32 - 1, &candidates)
            .is_ok()
    );
    for hostile in [
        OfficialIdentity {
            gamma: 8,
            ..official
        },
        OfficialIdentity {
            rank: 64,
            ..official
        },
        OfficialIdentity {
            top_k: 15,
            ..official
        },
        OfficialIdentity {
            vocab: 1000,
            ..official
        },
    ] {
        assert!(hostile.validate().is_err());
    }
    assert!(official.validate_candidates(0, &candidates[..111]).is_err());
    assert!(
        official
            .validate_candidates(VOCAB as u32, &candidates)
            .is_err()
    );
    let mut bad_candidate = candidates;
    bad_candidate[111] = VOCAB as u32;
    assert!(official.validate_candidates(0, &bad_candidate).is_err());
}

#[test]
fn cuda_source_pins_current_reassociated_conv_and_triple_product_selector() {
    assert!(CONV_CUDA.contains("(__bfloat162float(base[c]) + __bfloat162float(dr[g]))"));
    assert!(SELECTOR_CUDA.contains("__bfloat162float(pr[r]) * __bfloat162float(hr[r])"));
    assert!(!CONV_CUDA.contains("base_product"));
    assert!(!SELECTOR_CUDA.contains("__nv_bfloat16 gate"));
}
