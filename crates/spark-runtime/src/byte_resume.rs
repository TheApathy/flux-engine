// SPDX-License-Identifier: AGPL-3.0-only

//! ExpertAtlas Byte Resume — multi-turn BPE token-boundary drift recovery.
//!
//! In multi-turn coding agents (Claude Code, Codex, Antigravity, ohmypi),
//! concatenating a prior assistant response with the next user turn and
//! re-tokenizing the transcript frequently merges or splits boundary tokens
//! (for example `"\n" + "n"` vs `"\n\n"`, or trailing whitespace after
//! `</think>`). Even though the raw UTF-8 byte sequence is identical, token ID
//! divergence at the turn seam causes naive token-radix prefix caches to miss.
//!
//! [`heal_bpe_boundary_drift`] recovers the longest byte-equivalent prefix
//! while enforcing a strict security boundary: special, control, and added
//! delimiter tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, tool
//! tags) match **only** by exact token ID and never by surface bytes.

/// Block-aligned rewrite view when a healed BPE seam lands mid-KV-block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteResumeBlockAligned {
    /// Safe number of KV-cached tokens (`multiple of block_size`) reusable from
    /// physical KV blocks without mutating any shared partial block.
    pub safe_kv_tokens: usize,
    /// Exclusive end index in `cached_seq` used to form `rewritten_tokens`.
    pub seam_cached_end: usize,
    /// Exclusive resume index in `incoming_prompt` used to form `rewritten_tokens`.
    pub seam_incoming_resume: usize,
    /// Canonicalized token sequence preserving exact byte equivalence with `incoming_prompt`.
    pub rewritten_tokens: Vec<u32>,
}

/// Result of healing BPE token-boundary drift between `cached_seq` and `incoming_prompt`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteResumeMatch {
    /// First token index `p` where `cached_seq[p] != incoming_prompt[p]`.
    pub first_mismatch: usize,
    /// Furthest synchronized token index `q` (`> first_mismatch`) in `cached_seq`.
    pub cached_end: usize,
    /// Paired synchronized token index `r` (`> first_mismatch`) in `incoming_prompt`.
    pub incoming_resume: usize,
    /// Canonicalized prompt `cached_seq[0..cached_end] ++ incoming_prompt[incoming_resume..]`.
    pub rewritten_tokens: Vec<u32>,
    /// All simultaneous token boundaries `(q_k, r_k)` past `first_mismatch` where
    /// both streams landed at `rem_a == 0 && rem_b == 0`.
    pub sync_points: Vec<(usize, usize)>,
}

impl ByteResumeMatch {
    /// Number of full KV-block-aligned tokens covered by `cached_end`.
    ///
    /// When `cached_end` lands mid-KV-block, callers reusing paged KV blocks
    /// must back off physical block reuse to this boundary (`(q / block_size) * block_size`)
    /// so prefill recomputes the partial block cleanly across the healed seam.
    #[must_use]
    pub fn block_aligned_cached_tokens(&self, block_size: usize) -> usize {
        if block_size == 0 {
            return 0;
        }
        (self.cached_end / block_size) * block_size
    }

    /// True when `cached_end` falls on an exact multiple of `block_size`.
    #[must_use]
    pub fn is_block_aligned(&self, block_size: usize) -> bool {
        block_size > 0 && self.cached_end.is_multiple_of(block_size)
    }

    /// Compute a safe KV-block-aligned backoff plan for `block_size`.
    ///
    /// If an exact simultaneous token boundary `(q_blk, r_blk)` landed right on
    /// `safe_kv_tokens`, rewrites at `(q_blk, r_blk)` (`cached_seq[..q_blk] ++ incoming_prompt[r_blk..]`).
    /// Otherwise (when a multi-byte/merged token straddles the block boundary or
    /// the healed seam sits mid-block), retains `self.rewritten_tokens` so the
    /// first `safe_kv_tokens` hit full cached KV blocks while prefill recomputes
    /// `safe_kv_tokens..cached_end` seamlessly into a private KV block.
    #[must_use]
    pub fn backoff_to_block_boundary(
        &self,
        cached_seq: &[u32],
        incoming_prompt: &[u32],
        block_size: usize,
    ) -> Option<ByteResumeBlockAligned> {
        let safe_kv_tokens = self.block_aligned_cached_tokens(block_size);
        if safe_kv_tokens == 0 {
            return None;
        }
        if self.cached_end == safe_kv_tokens {
            return Some(ByteResumeBlockAligned {
                safe_kv_tokens,
                seam_cached_end: self.cached_end,
                seam_incoming_resume: self.incoming_resume,
                rewritten_tokens: self.rewritten_tokens.clone(),
            });
        }
        if let Some(&(q_blk, r_blk)) = self
            .sync_points
            .iter()
            .rfind(|&&(q, _)| q == safe_kv_tokens)
        {
            let prefix = cached_seq.get(..q_blk)?;
            let suffix = incoming_prompt.get(r_blk..)?;
            let mut rewritten = Vec::with_capacity(prefix.len() + suffix.len());
            rewritten.extend_from_slice(prefix);
            rewritten.extend_from_slice(suffix);
            return Some(ByteResumeBlockAligned {
                safe_kv_tokens,
                seam_cached_end: q_blk,
                seam_incoming_resume: r_blk,
                rewritten_tokens: rewritten,
            });
        }
        Some(ByteResumeBlockAligned {
            safe_kv_tokens,
            seam_cached_end: self.cached_end,
            seam_incoming_resume: self.incoming_resume,
            rewritten_tokens: self.rewritten_tokens.clone(),
        })
    }
}

/// Heal BPE token-boundary drift between `cached_seq` and `incoming_prompt`.
///
/// Finds the first token mismatch `p`, then walks `cached_seq[p..]` and
/// `incoming_prompt[p..]` byte-by-byte, tracking the furthest simultaneous
/// token boundary `(q, r)` (`rem_a == 0 && rem_b == 0`) with identical raw
/// bytes.
///
/// # Security Invariant
/// Any token `id` where `is_special_or_added(id)` is `true` (e.g. `<|im_start|>`,
/// `<|im_end|>`, `<think>`, `</think>`, tool-call tags) **only** matches when
/// both streams are at an exact token boundary (`rem_a == 0 && rem_b == 0`) and
/// `cached_seq[i] == incoming_prompt[j]`. Surface byte comparison is never used
/// on special/added tokens, preventing prompt-injection spoofing of role or
/// reasoning delimiters.
pub fn heal_bpe_boundary_drift<'a>(
    cached_seq: &[u32],
    incoming_prompt: &[u32],
    token_bytes: impl Fn(u32) -> Option<&'a [u8]>,
    is_special_or_added: impl Fn(u32) -> bool,
) -> Option<ByteResumeMatch> {
    let first_mismatch = cached_seq
        .iter()
        .zip(incoming_prompt.iter())
        .position(|(a, b)| a != b)?;

    let mut i = first_mismatch;
    let mut j = first_mismatch;
    let mut rem_a: &'a [u8] = &[];
    let mut rem_b: &'a [u8] = &[];
    let mut sync_points: Vec<(usize, usize)> = Vec::new();

    loop {
        if rem_a.is_empty() && rem_b.is_empty() {
            if i > first_mismatch && j > first_mismatch {
                sync_points.push((i, j));
            }
            if i >= cached_seq.len() || j >= incoming_prompt.len() {
                break;
            }
            let tok_a = cached_seq[i];
            let tok_b = incoming_prompt[j];
            let special_a = is_special_or_added(tok_a);
            let special_b = is_special_or_added(tok_b);

            if special_a || special_b {
                // CRITICAL SECURITY INVARIANT: special/control/added tokens
                // match ONLY by exact token ID at a simultaneous boundary.
                if special_a && special_b && tok_a == tok_b {
                    i += 1;
                    j += 1;
                    continue;
                }
                break;
            }

            if tok_a == tok_b {
                // Fast path: identical ordinary token IDs at a synchronized boundary.
                i += 1;
                j += 1;
                continue;
            }

            let Some(bytes_a) = token_bytes(tok_a).filter(|b| !b.is_empty()) else {
                break;
            };
            let Some(bytes_b) = token_bytes(tok_b).filter(|b| !b.is_empty()) else {
                break;
            };
            rem_a = bytes_a;
            rem_b = bytes_b;
            i += 1;
            j += 1;
        } else if rem_a.is_empty() {
            if i >= cached_seq.len() {
                break;
            }
            let tok_a = cached_seq[i];
            if is_special_or_added(tok_a) {
                break;
            }
            let Some(bytes_a) = token_bytes(tok_a).filter(|b| !b.is_empty()) else {
                break;
            };
            rem_a = bytes_a;
            i += 1;
        } else if rem_b.is_empty() {
            if j >= incoming_prompt.len() {
                break;
            }
            let tok_b = incoming_prompt[j];
            if is_special_or_added(tok_b) {
                break;
            }
            let Some(bytes_b) = token_bytes(tok_b).filter(|b| !b.is_empty()) else {
                break;
            };
            rem_b = bytes_b;
            j += 1;
        }

        let step = rem_a.len().min(rem_b.len());
        if rem_a[..step] != rem_b[..step] {
            break;
        }
        rem_a = &rem_a[step..];
        rem_b = &rem_b[step..];
    }

    let &(cached_end, incoming_resume) = sync_points.last()?;
    let mut rewritten_tokens =
        Vec::with_capacity(cached_end + incoming_prompt.len().saturating_sub(incoming_resume));
    rewritten_tokens.extend_from_slice(&cached_seq[..cached_end]);
    rewritten_tokens.extend_from_slice(&incoming_prompt[incoming_resume..]);

    Some(ByteResumeMatch {
        first_mismatch,
        cached_end,
        incoming_resume,
        rewritten_tokens,
        sync_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOK_IM_START: u32 = 151_644;
    const TOK_IM_END: u32 = 151_645;
    const TOK_THINK_OPEN: u32 = 151_667;
    const TOK_THINK_CLOSE: u32 = 151_668;
    const TOK_TOOL_CALL: u32 = 151_657;

    fn is_special(id: u32) -> bool {
        matches!(
            id,
            TOK_IM_START | TOK_IM_END | TOK_THINK_OPEN | TOK_THINK_CLOSE | TOK_TOOL_CALL
        )
    }

    fn demo_vocab(id: u32) -> Option<&'static [u8]> {
        match id {
            1 => Some(b"def "),
            2 => Some(b"main():"),
            3 => Some(b"\n"),
            4 => Some(b"n"),
            5 => Some(b"\n\n"),
            6 => Some(b"\nn"),
            7 => Some(b"    return 42"),
            8 => Some(b" "),
            9 => Some(b"  "),
            10 => Some(b"user"),
            11 => Some(b"assistant"),
            12 => Some(b"next question"),
            // Multi-byte UTF-8 ("🚀" = [0xF0, 0x9F, 0x9A, 0x80]) split across BPE byte tokens
            20 => Some(&[0xF0, 0x9F]),
            21 => Some(&[0x9A, 0x80]),
            22 => Some(&[0xF0, 0x9F, 0x9A, 0x80]),
            // Adversarial surface pieces spelling "<|im_end|>" and "</think>"
            30 => Some(b"<|"),
            31 => Some(b"im_"),
            32 => Some(b"end"),
            33 => Some(b"|>"),
            34 => Some(b"</"),
            35 => Some(b"think>"),
            36 => Some(b"foo<|im_end|>"),
            37 => Some(b"foo"),
            // Even if a caller's vocab table accidentally exposes surface bytes for specials,
            // heal_bpe_boundary_drift must NEVER match specials via token_bytes!
            TOK_IM_START => Some(b"<|im_start|>"),
            TOK_IM_END => Some(b"<|im_end|>"),
            TOK_THINK_OPEN => Some(b"<think>"),
            TOK_THINK_CLOSE => Some(b"</think>"),
            TOK_TOOL_CALL => Some(b"<tool_call>"),
            99 => Some(b""),
            _ => None,
        }
    }

    #[test]
    fn heals_newline_n_and_think_whitespace_splits_across_multi_turn_seams() {
        // Turn 1 cached: <|im_start|> assistant \n <think> \n </think> \n \n def main(): \n n <|im_end|> \n
        let cached = vec![
            TOK_IM_START,
            11,
            3,
            TOK_THINK_OPEN,
            3,
            TOK_THINK_CLOSE,
            3,
            3, // "\n" + "\n"
            1,
            2,
            3,
            4, // "\n" + "n"
            TOK_IM_END,
            3,
        ];
        // Turn 2 re-tokenized: merges "\n"+"\n" -> "\n\n" (5) and "\n"+"n" -> "\nn" (6), then appends user turn.
        let incoming = vec![
            TOK_IM_START,
            11,
            3,
            TOK_THINK_OPEN,
            3,
            TOK_THINK_CLOSE,
            5, // "\n\n"
            1,
            2,
            6, // "\nn"
            TOK_IM_END,
            3,
            TOK_IM_START,
            10,
            3,
            12,
            TOK_IM_END,
        ];

        let healed =
            heal_bpe_boundary_drift(&cached, &incoming, demo_vocab, is_special).expect("must heal");
        assert_eq!(healed.first_mismatch, 6);
        assert_eq!(healed.cached_end, cached.len());
        assert_eq!(healed.incoming_resume, 12);
        assert_eq!(&healed.rewritten_tokens[..cached.len()], cached.as_slice());
        assert_eq!(
            &healed.rewritten_tokens[cached.len()..],
            &[TOK_IM_START, 10, 3, 12, TOK_IM_END]
        );
    }

    #[test]
    fn heals_arbitrary_raw_utf8_byte_splits() {
        let cached = vec![1, 20, 21, 7]; // [0xF0,0x9F] + [0x9A,0x80]
        let incoming = vec![1, 22, 7, 3]; // [0xF0,0x9F,0x9A,0x80]
        let healed =
            heal_bpe_boundary_drift(&cached, &incoming, demo_vocab, is_special).expect("utf8 heal");
        assert_eq!(healed.first_mismatch, 1);
        assert_eq!(healed.cached_end, 4);
        assert_eq!(healed.incoming_resume, 3);
        assert_eq!(healed.rewritten_tokens, vec![1, 20, 21, 7, 3]);
    }

    #[test]
    fn rejects_prompt_injection_spoofing_special_tokens_by_surface_bytes() {
        // Attack 1: cached sequence has real TOK_IM_END; attacker supplies ordinary tokens "<|"+"im_"+"end"+"|>".
        let cached = vec![1, 2, TOK_IM_END, 3];
        let spoof_incoming = vec![1, 2, 30, 31, 32, 33, 3];
        assert!(
            heal_bpe_boundary_drift(&cached, &spoof_incoming, demo_vocab, is_special).is_none(),
            "surface text must NEVER match a cached special token"
        );

        // Attack 2: cached sequence had literal text "<|im_end|>" (30,31,32,33); incoming has real TOK_IM_END.
        let cached_text = vec![1, 2, 30, 31, 32, 33, 3];
        let real_delim_incoming = vec![1, 2, TOK_IM_END, 3];
        assert!(
            heal_bpe_boundary_drift(&cached_text, &real_delim_incoming, demo_vocab, is_special)
                .is_none(),
            "real incoming special token must NEVER be rewritten to ordinary text tokens"
        );

        // Attack 3: spoofed "</think>" via surface tokens "</" + "think>".
        let cached_think = vec![TOK_THINK_OPEN, 7, TOK_THINK_CLOSE, 5];
        let spoof_think = vec![TOK_THINK_OPEN, 7, 34, 35, 3, 3];
        assert!(
            heal_bpe_boundary_drift(&cached_think, &spoof_think, demo_vocab, is_special).is_none(),
            "surface '</' + 'think>' must NEVER match real </think> token ID"
        );

        // Attack 4: mid-token collision into a special token ("foo<|im_end|>" vs "foo" + TOK_IM_END).
        let mid_cached = vec![1, 36];
        let mid_incoming = vec![1, 37, TOK_IM_END];
        assert!(
            heal_bpe_boundary_drift(&mid_cached, &mid_incoming, demo_vocab, is_special).is_none(),
            "mid-token byte stream must NEVER cross into a special token"
        );
    }

    #[test]
    fn returns_none_when_no_drift_or_divergent_bytes_or_unknown_tokens() {
        // Identical prefix (no mismatch at all) -> None.
        assert!(
            heal_bpe_boundary_drift(&[1, 2, 3], &[1, 2, 3, 7], demo_vocab, is_special).is_none()
        );
        // Genuine content divergence -> None.
        assert!(heal_bpe_boundary_drift(&[1, 2], &[1, 7], demo_vocab, is_special).is_none());
        // Unknown token bytes -> None without panicking.
        assert!(heal_bpe_boundary_drift(&[1, 999], &[1, 2], demo_vocab, is_special).is_none());
        // Empty byte token -> None without infinite loop.
        assert!(heal_bpe_boundary_drift(&[1, 99, 3], &[1, 3], demo_vocab, is_special).is_none());
    }

    #[test]
    fn block_aligned_backoff_computes_safe_kv_boundary_for_mid_block_seams() {
        // 8 tokens/block: cached has 20 tokens (2 full blocks = 16 tokens + 4 tail tokens).
        // Drift occurs at index 14 (inside block 1), syncs at q=16 (exact block boundary)
        // AND continues matching to q=19 (mid-block 2).
        let mut cached = vec![1u32; 14];
        cached.extend_from_slice(&[3, 3, 1, 2, 7]); // indices 14..19 (len 19)
        let mut incoming = vec![1u32; 14];
        incoming.extend_from_slice(&[5, 1, 2, 7, 12]); // index 14 is "\n\n" (merges 14,15), then 1,2,7, then new 12

        let healed =
            heal_bpe_boundary_drift(&cached, &incoming, demo_vocab, is_special).expect("must heal");
        assert_eq!(healed.cached_end, 19);
        assert_eq!(healed.incoming_resume, 18);
        assert!(!healed.is_block_aligned(8));
        assert_eq!(healed.block_aligned_cached_tokens(8), 16);

        let backoff = healed
            .backoff_to_block_boundary(&cached, &incoming, 8)
            .expect("2 full blocks available");
        assert_eq!(backoff.safe_kv_tokens, 16);
        assert_eq!(backoff.seam_cached_end, 16);
        assert_eq!(backoff.seam_incoming_resume, 15);
        assert_eq!(&backoff.rewritten_tokens[..16], &cached[..16]);
        assert_eq!(&backoff.rewritten_tokens[16..], &[1, 2, 7, 12]);
    }
}
