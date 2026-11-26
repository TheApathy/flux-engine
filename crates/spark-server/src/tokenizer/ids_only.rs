// SPDX-License-Identifier: AGPL-3.0-only

//! IDs-only tokenizer encoding.
//!
//! `Tokenizer::encode(text, false)` builds the full `Encoding`: token IDs plus
//! a `String` and char-offset span per output token. Server callers only read
//! `encoding.get_ids()`, so `Tokenizer::encode_fast(text, false)` computes the
//! exact same token IDs while skipping the per-token UTF-8 string and offset
//! allocations (~605 MiB peak host RSS saved on multi-megabyte prompts).

use tokenizers::{Encoding, Tokenizer};

/// Default on: server encode paths only consume `Encoding::get_ids()`.
pub(super) const DEFAULT_IDS_ONLY: bool = true;

pub(super) fn encode(
    tokenizer: &Tokenizer,
    text: &str,
    ids_only: bool,
) -> tokenizers::Result<Encoding> {
    if ids_only {
        tokenizer.encode_fast(text, false)
    } else {
        tokenizer.encode(text, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::AddedToken;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    fn byte_level_tokenizer() -> Tokenizer {
        let vocab: tokenizers::models::bpe::Vocab = ByteLevel::alphabet()
            .into_iter()
            .enumerate()
            .map(|(id, c)| (c.to_string(), id as u32))
            .collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer
            .add_special_tokens([
                AddedToken::from("<|im_start|>", true),
                AddedToken::from("<|im_end|>", true),
            ])
            .unwrap();
        tokenizer
            .add_tokens([AddedToken::from("<think>", false)])
            .unwrap();
        tokenizer
    }

    #[test]
    fn ids_only_matches_full_encode_across_ascii_unicode_and_specials() {
        let tok = byte_level_tokenizer();
        let samples = [
            "",
            " ",
            "\n\n\t",
            "hello world 12345 !@#$%^&*()",
            "café — €100 — naïve résumé åäö",
            "日本語と한국어とالعربية🙂🚀🎉",
            "<|im_start|>user\nhello <think>\ncafé € 🙂<|im_end|>\n<|im_start|>assistant\n",
            &"long-prompt-segment-with-unicode-€-🙂-<think>-".repeat(512),
        ];

        for text in samples {
            let fast = encode(&tok, text, true).unwrap();
            let full = encode(&tok, text, false).unwrap();
            assert_eq!(fast.get_ids(), full.get_ids(), "mismatch on {text:?}");
            if !text.is_empty() {
                assert!(
                    fast.get_tokens().iter().all(String::is_empty),
                    "encode_fast must omit non-empty token strings"
                );
                assert!(
                    fast.get_offsets().iter().all(|&off| off == (0, 0)),
                    "encode_fast must omit char offsets"
                );
                assert!(
                    full.get_tokens().iter().any(|s| !s.is_empty()),
                    "full encode populates token strings"
                );
            }
        }
    }
}
