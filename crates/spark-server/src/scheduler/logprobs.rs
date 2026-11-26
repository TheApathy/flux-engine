// SPDX-License-Identifier: AGPL-3.0-only

//! Logprobs extraction helpers.

use super::*;

/// Extract top-K logprobs from an FP32 logits slice for one token position.
///
/// Computes log-softmax over the logits, extracts the logprob of the sampled
/// token, and returns the top-K alternatives sorted descending by logprob.
pub fn extract_logprobs_from_f32(
    f32_logits: &[f32],
    sampled_token: u32,
    k: usize,
) -> crate::api::TokenLogprobs {
    if f32_logits.is_empty() {
        return crate::api::TokenLogprobs {
            token_id: sampled_token,
            logprob: f32::NEG_INFINITY,
            top: Vec::new(),
        };
    }
    // Log-softmax: logprob = logit - log(sum(exp(logits)))
    let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = max_logit
        + f32_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum::<f32>()
            .ln();
    let sampled_logprob = if (sampled_token as usize) < f32_logits.len() {
        f32_logits[sampled_token as usize] - log_sum_exp
    } else {
        f32::NEG_INFINITY
    };
    if k == 0 {
        return crate::api::TokenLogprobs {
            token_id: sampled_token,
            logprob: sampled_logprob,
            top: Vec::new(),
        };
    }
    // Find top-K by partial sort.
    let mut indexed: Vec<(u32, f32)> = f32_logits
        .iter()
        .enumerate()
        .map(|(j, &l)| (j as u32, l - log_sum_exp))
        .collect();
    let nth = k.min(indexed.len() - 1);
    indexed.select_nth_unstable_by(nth, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<(u32, f32)> = indexed[..k.min(indexed.len())].to_vec();
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    crate::api::TokenLogprobs {
        token_id: sampled_token,
        logprob: sampled_logprob,
        top,
    }
}

fn decode_logits_row(slice: &[u8], fp32: bool) -> Vec<f32> {
    if fp32 {
        slice
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        slice
            .chunks_exact(2)
            .map(|c| bf16_to_f32(c[0], c[1]))
            .collect()
    }
}

/// Extract logprobs for K positions from the logits buffer on GPU (BF16 or FP32).
///
/// Copies `[K, vocab_size]` logits D2H (dispatching by `decode_logits_fp32` /
/// `logits_ptr_is_fp32`), converts to FP32, and extracts top-K logprobs per
/// position. Returns empty Vec on copy failure.
pub fn extract_verify_logprobs(
    model: &dyn Model,
    tokens: &[u32],
    k_logprobs: u8,
) -> Vec<crate::api::TokenLogprobs> {
    let k = tokens.len();
    let vocab = model.vocab_size();
    let logits = model.logits_buffer_ptr();
    let fp32 = model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits);
    let elem_bytes = if fp32 { 4 } else { 2 };
    let mut buf = vec![0u8; k * vocab * elem_bytes];
    if model.copy_logits_to_host(logits, &mut buf).is_err() {
        return Vec::new();
    }
    tokens
        .iter()
        .enumerate()
        .map(|(i, &tok)| {
            let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
            let f32_logits = decode_logits_row(slice, fp32);
            extract_logprobs_from_f32(&f32_logits, tok, k_logprobs as usize)
        })
        .collect()
}

/// Extract logprobs for a single token from the logits buffer on GPU (BF16 or FP32).
///
/// Copies `[1, vocab_size]` logits D2H (dispatching by `decode_logits_fp32` /
/// `logits_ptr_is_fp32`), converts to FP32, and extracts top-K logprobs.
/// Returns None on copy failure.
pub fn extract_single_logprobs(
    model: &dyn Model,
    logits: DevicePtr,
    sampled_token: u32,
    k_logprobs: u8,
) -> Option<crate::api::TokenLogprobs> {
    let vocab = model.vocab_size();
    let fp32 = model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits);
    let elem_bytes = if fp32 { 4 } else { 2 };
    let mut buf = vec![0u8; vocab * elem_bytes];
    if model.copy_logits_to_host(logits, &mut buf).is_err() {
        return None;
    }
    let f32_logits = decode_logits_row(&buf, fp32);
    Some(extract_logprobs_from_f32(
        &f32_logits,
        sampled_token,
        k_logprobs as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_logprobs_handles_empty_and_zero_k_without_panicking() {
        let empty = extract_logprobs_from_f32(&[], 0, 5);
        assert_eq!(empty.token_id, 0);
        assert_eq!(empty.logprob, f32::NEG_INFINITY);
        assert!(empty.top.is_empty());

        let zero_k = extract_logprobs_from_f32(&[1.0, 2.0, 3.0], 2, 0);
        assert_eq!(zero_k.token_id, 2);
        assert!(zero_k.logprob.is_finite());
        assert!(zero_k.top.is_empty());
    }

    #[test]
    fn decode_logits_row_handles_fp32_and_bf16() {
        let raw_fp32: Vec<u8> = [1.25f32, -3.5f32, 0.0f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(decode_logits_row(&raw_fp32, true), vec![1.25, -3.5, 0.0]);

        let raw_bf16: Vec<u8> = [1.5f32, -2.0f32]
            .into_iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(decode_logits_row(&raw_bf16, false), vec![1.5, -2.0]);
    }
}
