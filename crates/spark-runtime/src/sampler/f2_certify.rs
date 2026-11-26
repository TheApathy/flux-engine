// SPDX-License-Identifier: AGPL-3.0-only
//
// Certified F2 confidence decision for the device top-k route (SPEED_PLAN
// W1a2). F2 (the plain path's confidence early stop) asks, on the edited
// full-vocabulary row x':
//
//     max = fold(f32::max), S_h = sequential f32 sum of exp(x'_i - max),
//     confident = S_h > 0 && 1.0f32 / S_h >= 0.95f32
//
// The device cannot reproduce that summation order, so it computes S_d with
// its own order and the host only accepts the device's answer when it is
// PROVABLY the same:
//
// * every term t_i = exp(x'_i - max) lies in [0, 1] and the max term is 1,
//   so S >= 1;
// * host summation: each add P_k = fl(P_{k-1} + t_k) errs by at most u*P_k
//   (u = 2^-24), so |S_h - sum t_hat| <= u * sum_k P_k = u * sum_i t_i (n - i)
//   up to a (1 + gamma) factor. That weighted sum W is order-free, so the
//   device computes it;
// * terms: the f32 argument x - max errs by u|a| and expf by <= 2 ulp on
//   either side, so |t_hat - t_dev| <= t (2|a| + 6) u; the device sums
//   E = sum t (2|a| + 6);
// * device accumulation: per-lane f32 sums of L = ceil(n / 1024) terms, merged
//   in f64: <= L u S_d;
// * fl(1/S) is monotone in S, so the decision is a threshold on S, bracketed
//   with a 1e-6 relative guard for the f32 division and the f32 0.95.
//
// With D = 1.05 u (1.02 W + E + L S_d) + 1e-9, the device's answer is accepted
// only when [S_d - D, S_d + D] lies wholly on one side of the bracket. Anything
// else, a NaN / +inf in the edited row, or an all -inf row, is uncertain: the
// scheduler copies the full row and runs the host test verbatim (receipt
// reason f2_host_full). Either way the decision equals the plain path's.

/// One host edit that precedes F2 on the plain path, in application order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreF2Edit {
    /// `logit -= value` (F1 reflection suppression: 10.0).
    Sub(f32),
    /// `logit = -inf` (min-thinking floor on `</think>`).
    NegInf,
}

impl PreF2Edit {
    #[inline]
    pub fn apply(self, x: &mut f32) {
        match self {
            PreF2Edit::Sub(v) => *x -= v,
            PreF2Edit::NegInf => *x = f32::NEG_INFINITY,
        }
    }

    /// Device encoding: (kind, value) with kind 0 = subtract, 1 = set -inf.
    pub fn encode(self) -> (u32, f32) {
        match self {
            PreF2Edit::Sub(v) => (0, v),
            PreF2Edit::NegInf => (1, 0.0),
        }
    }
}

/// Most edits the device accepts per row.
pub const F2_EDITS_MAX: usize = 64;

/// What the device reports for one edited row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumExpStats {
    /// `fold(f32::max)` over the edited row (exact: max is order-free).
    pub max: f32,
    /// S_d: sum of exp(x - max), per-lane f32 merged in f64.
    pub sum: f64,
    /// W: sum of t_i * (n - i) (0-based i: the number of host partial sums
    /// that contain term i), the host summation-error weight.
    pub weighted: f64,
    /// E: sum of t_i * (2 |x_i - max| + 6), the term-error weight.
    pub term_err: f64,
    /// Row length n.
    pub n: u32,
    /// The edited row holds a NaN or +inf.
    pub nonfinite: bool,
}

/// Device threads per row (lane count of the per-lane f32 sums).
pub const F2_LANES: usize = 1024;

/// Absolute bound D on |S_host - S_device| (see the module comment).
pub fn absolute_error_bound(s: &SumExpStats) -> f64 {
    let u = 2f64.powi(-24);
    let lane_terms = (s.n as usize).div_ceil(F2_LANES) as f64;
    1.05 * u * (1.02 * s.weighted + s.term_err + lane_terms * s.sum) + 1e-9
}

/// Some(decision) when the host's F2 test is provably `decision`, else None.
pub fn certify(s: &SumExpStats) -> Option<bool> {
    let finite =
        s.max.is_finite() && s.sum.is_finite() && s.weighted.is_finite() && s.term_err.is_finite();
    if s.nonfinite || !finite || s.sum < 1.0 {
        return None;
    }
    let d = absolute_error_bound(s);
    let threshold = 1.0 / f64::from(0.95f32);
    let (lo, hi) = (threshold * (1.0 - 1e-6), threshold * (1.0 + 1e-6));
    if s.sum + d <= lo {
        Some(true)
    } else if s.sum - d >= hi {
        Some(false)
    } else {
        None
    }
}

/// The plain path's F2 test, verbatim (decode_logits_seq.rs).
pub fn host_f2_confident(edited: &[f32]) -> bool {
    let max_logit = edited.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = edited.iter().map(|&l| (l - max_logit).exp()).sum();
    sum_exp > 0.0 && 1.0 / sum_exp >= 0.95
}

/// CPU specification of the device kernel `edited_sum_exp_bf16`: apply
/// `edits` in order, then the max, the per-lane f32 sums merged in f64, and
/// the two error weights in f64.
pub fn edited_sum_exp_reference(row_bf16: &[u8], edits: &[(u32, PreF2Edit)]) -> SumExpStats {
    let mut x: Vec<f32> = row_bf16
        .chunks_exact(2)
        .map(|b| f32::from_bits(((b[0] as u32) | ((b[1] as u32) << 8)) << 16))
        .collect();
    for &(id, e) in edits {
        if let Some(v) = x.get_mut(id as usize) {
            e.apply(v);
        }
    }
    let n = x.len();
    let nonfinite = x.iter().any(|v| v.is_nan() || *v == f32::INFINITY);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut lanes = vec![0f32; F2_LANES];
    let (mut weighted, mut term_err) = (0f64, 0f64);
    for (i, &v) in x.iter().enumerate() {
        let a = v - max;
        let t = a.exp();
        lanes[i % F2_LANES] += t;
        if t > 0.0 {
            weighted += f64::from(t) * (n - i) as f64;
            term_err += f64::from(t) * (2.0 * f64::from(a.abs()) + 6.0);
        }
    }
    let sum = lanes.iter().map(|&s| f64::from(s)).sum();
    SumExpStats {
        max,
        sum,
        weighted,
        term_err,
        n: n as u32,
        nonfinite,
    }
}

#[cfg(test)]
#[path = "f2_certify_tests.rs"]
mod tests;
