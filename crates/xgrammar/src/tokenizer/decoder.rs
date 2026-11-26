// SPDX-License-Identifier: AGPL-3.0-only
//
// Token decoders — port of `class TokenDecoder` in
// `cpp/tokenizer_info.cc`.
//
// Turns a raw (encoded) vocabulary string into the actual token byte
// sequence, according to the tokenizer's `VocabType`:
//
//   * Raw          — the string is the literal token bytes.
//   * ByteFallback — `<0xHH>` tokens decode to a single byte; the
//                    `▁` (U+2581) marker decodes back to a space.
//   * ByteLevel    — the GPT-2 bytes-to-unicode transform is inverted.
//
// The C++ returns `std::string` (raw bytes). We return `Vec<u8>` since
// a decoded token need not be valid UTF-8.

use crate::tokenizer::vocab_type::VocabType;

/// Decode one raw vocabulary token to its byte sequence for `vocab_type`.
pub fn decode_token(token: &str, vocab_type: VocabType) -> Vec<u8> {
    match vocab_type {
        VocabType::Raw => token.as_bytes().to_vec(),
        VocabType::ByteFallback => space_replacer_decoder(&byte_fallback_decoder(token)),
        VocabType::ByteLevel => byte_level_decoder(token),
    }
}

/// Parse a single uppercase ASCII hexadecimal nibble (`0`–`9` or `A`–`F`).
#[inline]
fn decode_upper_hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// ByteFallback decoder: transform a token like `<0x1B>` into the single
/// raw byte `0x1B`. Any other token (including malformed `<0x..>` tokens
/// whose payload is not two uppercase hex digits) is returned unchanged.
fn byte_fallback_decoder(token: &str) -> Vec<u8> {
    let bytes = token.as_bytes();
    if bytes.len() == 6
        && &bytes[0..3] == b"<0x"
        && bytes[5] == b'>'
        && let (Some(hi), Some(lo)) = (
            decode_upper_hex_nibble(bytes[3]),
            decode_upper_hex_nibble(bytes[4]),
        )
    {
        return vec![(hi << 4) | lo];
    }
    token.as_bytes().to_vec()
}

/// SpaceReplacer decoder: transform the UTF-8 of `▁` (U+2581, "lower one
/// eighth block", bytes `E2 96 81`) back into an ASCII space.
///
/// Faithful to C++ `SpaceReplacerDecoder`, which scans bytes directly.
fn space_replacer_decoder(token: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(token.len());
    let mut i = 0;
    while i < token.len() {
        // C++ guard is `i + 2 < size`, i.e. all three bytes addressable.
        if i + 2 < token.len() && token[i] == 0xE2 && token[i + 1] == 0x96 && token[i + 2] == 0x81 {
            result.push(b' ');
            i += 3;
        } else {
            result.push(token[i]);
            i += 1;
        }
    }
    result
}

/// The inverse of the GPT-2 bytes-to-unicode map.
///
/// `char_to_byte_map()[codepoint]` is the original byte, or `-1` when
/// no byte maps to that codepoint. The table has 324 entries, matching
/// `std::array<int, 324>` in the C++ source verbatim.
pub fn char_to_byte_map() -> &'static [i16; 324] {
    &CHAR_TO_BYTE_MAP
}

/// The GPT-2 bytes-to-unicode forward map: byte -> unicode codepoint.
///
/// This is the table HuggingFace uses to *encode* the vocabulary; it is
/// the exact inverse of `CHAR_TO_BYTE_MAP`. Derived (not duplicated)
/// from the inverse table to keep a single source of truth.
pub fn byte_to_char_map() -> &'static [u32; 256] {
    &BYTE_TO_CHAR_MAP
}

#[rustfmt::skip]
static CHAR_TO_BYTE_MAP: [i16; 324] = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
    92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, -1,
    174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
    246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
    147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 173,
];

/// Forward map derived once from `CHAR_TO_BYTE_MAP` (SSOT: the inverse
/// table is authoritative; this is `byte -> codepoint` such that
/// `CHAR_TO_BYTE_MAP[byte_to_char(byte)] == byte`).
static BYTE_TO_CHAR_MAP: [u32; 256] = build_byte_to_char_map();

const fn build_byte_to_char_map() -> [u32; 256] {
    let mut out = [0u32; 256];
    let mut cp = 0usize;
    while cp < CHAR_TO_BYTE_MAP.len() {
        let b = CHAR_TO_BYTE_MAP[cp];
        if b >= 0 {
            out[b as usize] = cp as u32;
        }
        cp += 1;
    }
    out
}

/// ByteLevel decoder: invert the GPT-2 bytes-to-unicode transform.
///
/// Faithful to C++ `ByteLevelDecoder`. If the token contains a codepoint
/// with no inverse mapping (including NUL `'\0'`), the original token
/// bytes are returned unchanged.
fn byte_level_decoder(token: &str) -> Vec<u8> {
    let map = char_to_byte_map();
    let mut decoded = Vec::with_capacity(token.len());
    for ch in token.chars() {
        let cp = ch as usize;
        if cp >= map.len() || map[cp] == -1 {
            return token.as_bytes().to_vec();
        }
        decoded.push(map[cp] as u8);
    }
    decoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_is_literal() {
        assert_eq!(decode_token("hello", VocabType::Raw), b"hello");
        assert_eq!(decode_token("", VocabType::Raw), b"");
    }

    #[test]
    fn byte_fallback_hex_token() {
        // <0x1B> -> ESC byte 0x1B
        assert_eq!(decode_token("<0x1B>", VocabType::ByteFallback), vec![0x1B]);
        // <0x00> -> NUL
        assert_eq!(decode_token("<0x00>", VocabType::ByteFallback), vec![0x00]);
        // <0xFF> -> 0xFF
        assert_eq!(decode_token("<0xFF>", VocabType::ByteFallback), vec![0xFF]);
    }

    #[test]
    fn byte_fallback_non_hex_passthrough() {
        // Not the <0xHH> shape -> unchanged.
        assert_eq!(decode_token("er", VocabType::ByteFallback), b"er");
        assert_eq!(decode_token("<0x1>", VocabType::ByteFallback), b"<0x1>");
        // Malformed 6-byte `<0x..>` tokens with bytes below 'A', above 'F', or
        // lowercase hex must not panic/underflow and must pass through unchanged.
        for malformed in ["<0x  >", "<0x-->", "<0x??>", "<0xab>", "<0xG0>", "<0x0/>"] {
            assert_eq!(
                decode_token(malformed, VocabType::ByteFallback),
                malformed.as_bytes(),
                "malformed token {malformed:?} should pass through unchanged"
            );
        }
    }

    #[test]
    fn byte_fallback_space_marker() {
        // U+2581 "▁" -> space
        let token = "\u{2581}hello";
        assert_eq!(decode_token(token, VocabType::ByteFallback), b" hello");
    }

    #[test]
    fn byte_fallback_double_space_marker() {
        let token = "\u{2581}\u{2581}";
        assert_eq!(decode_token(token, VocabType::ByteFallback), b"  ");
    }

    #[test]
    fn byte_level_round_trip_ascii() {
        // For printable ASCII, byte_to_char is the identity, so the
        // encoded string equals the literal — decode returns the bytes.
        let map = byte_to_char_map();
        // 'A' (0x41) maps to itself.
        assert_eq!(map[0x41], 0x41);
        let mut enc = String::new();
        for &b in b"automotive" {
            enc.push(char::from_u32(map[b as usize]).unwrap());
        }
        assert_eq!(decode_token(&enc, VocabType::ByteLevel), b"automotive");
    }

    #[test]
    fn byte_level_space_is_g_with_dot() {
        // GPT-2 maps byte 0x20 (space) to U+0120 'Ġ'.
        let map = byte_to_char_map();
        assert_eq!(map[0x20], 0x0120);
        // " automotive" -> "Ġautomotive"
        let mut enc = String::from("\u{0120}");
        for &b in b"automotive" {
            enc.push(char::from_u32(map[b as usize]).unwrap());
        }
        assert_eq!(decode_token(&enc, VocabType::ByteLevel), b" automotive");
    }

    #[test]
    fn byte_level_high_bytes() {
        // The UTF-8 of '我' is E6 88 91 — encode each byte via the map.
        let map = byte_to_char_map();
        let mut enc = String::new();
        for &b in "我".as_bytes() {
            enc.push(char::from_u32(map[b as usize]).unwrap());
        }
        assert_eq!(decode_token(&enc, VocabType::ByteLevel), "我".as_bytes());
    }

    #[test]
    fn byte_level_unmapped_codepoint_passthrough() {
        // U+4E2D '中' has no inverse mapping -> token returned as-is.
        assert_eq!(decode_token("中", VocabType::ByteLevel), "中".as_bytes());
    }

    #[test]
    fn inverse_table_is_consistent() {
        // Every mapped entry of the inverse table must round-trip.
        let inv = char_to_byte_map();
        let fwd = byte_to_char_map();
        for byte in 0u32..256 {
            let cp = fwd[byte as usize];
            assert_eq!(inv[cp as usize], byte as i16);
        }
        // All 256 bytes have a forward mapping (the map is a bijection
        // onto 256 distinct codepoints).
        let mut seen = std::collections::HashSet::new();
        for byte in 0u32..256 {
            assert!(seen.insert(fwd[byte as usize]));
        }
    }

    #[test]
    fn byte_level_nul_passthrough() {
        // NUL (U+0000) maps to -1 in the inverse table and must NOT truncate
        // the token or collapse "\0" into an empty (special) token.
        assert_eq!(char_to_byte_map()[0], -1);
        assert_eq!(decode_token("\0", VocabType::ByteLevel), b"\0");
        assert_eq!(decode_token("a\0b", VocabType::ByteLevel), b"a\0b");
    }
}
