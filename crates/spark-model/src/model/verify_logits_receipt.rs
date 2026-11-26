// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_VERIFY_LOGITS_RECEIPT=1`: qualification-only receipt of every
//! Qwen4 K-row verify frame's logits, emitted identically by the
//! single-sequence verifier and the cross-sequence batched frame
//! (`ATLAS_QWEN4_BATCHED_K5`). Each line keys one sequence's frame by a hash
//! of its committed history plus the verify input tokens, and carries an
//! FNV-1a hash of its K BF16 logits rows and its argmax row, so a harness can
//! require bit-identical logits between a batched arm and a C1 arm without
//! relying on token IDs alone. Off by default; when on, each frame pays one
//! synchronous K x vocab logits read per sequence.

use anyhow::Result;

pub(crate) const RECEIPT_ENV: &str = "ATLAS_QWEN4_VERIFY_LOGITS_RECEIPT";

pub(crate) fn parse_selector(raw: Option<&str>) -> std::result::Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_QWEN4_VERIFY_LOGITS_RECEIPT must be exact value 0 or 1"),
    }
}

pub(crate) fn selected() -> Result<bool> {
    let raw = std::env::var(RECEIPT_ENV).ok();
    parse_selector(raw.as_deref()).map_err(anyhow::Error::msg)
}

/// FNV-1a 64 over raw bytes.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn fnv_u32s(values: &[u32]) -> u64 {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fnv1a64(&bytes)
}

fn csv(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// The receipt line for one sequence's verify frame. `history` is the
/// sequence's committed tokens before the frame; `logits` is its K logits
/// rows exactly as the LM head wrote them.
pub(crate) fn line(
    route: &str,
    history: &[u32],
    inputs: &[u32],
    argmax: &[u32],
    logits: &[u8],
) -> String {
    format!(
        "QWEN4_VERIFY_LOGITS_RECEIPT route={route} prefix_len={} prefix_fnv={:016x} k={} inputs={} argmax={} logits_fnv={:016x}",
        history.len(),
        fnv_u32s(history),
        inputs.len(),
        csv(inputs),
        csv(argmax),
        fnv1a64(logits),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_is_strict_and_default_off() {
        assert_eq!(parse_selector(None), Ok(false));
        assert_eq!(parse_selector(Some("1")), Ok(true));
        assert!(parse_selector(Some("true")).is_err());
    }

    #[test]
    fn fnv_matches_the_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn line_keys_by_history_and_inputs_and_hashes_logits() {
        let a = line("c1", &[1, 2, 3], &[4, 5], &[5, 6], &[0, 1, 2, 3]);
        let b = line("batched", &[1, 2, 3], &[4, 5], &[5, 6], &[0, 1, 2, 3]);
        let key = |l: &str| {
            l.split(" logits_fnv=")
                .next()
                .unwrap()
                .replace("route=batched", "route=c1")
        };
        assert_eq!(key(&a), key(&b));
        assert_eq!(a.rsplit('=').next(), b.rsplit('=').next());
        let c = line("c1", &[1, 2, 3], &[4, 5], &[5, 6], &[0, 1, 2, 4]);
        assert_ne!(a.rsplit('=').next(), c.rsplit('=').next());
        assert!(a.contains("prefix_len=3") && a.contains("inputs=4,5") && a.contains("k=2"));
    }
}
