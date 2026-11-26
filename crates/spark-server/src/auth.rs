// SPDX-License-Identifier: AGPL-3.0-only

//! Enterprise bearer & `x-api-key` authentication with scoped token prefixes.
//!
//! Atlas serves OpenAI, Anthropic, and Ollama-compatible endpoints; mainstream
//! client SDKs send either `Authorization: Bearer <key>` or Anthropic's
//! `x-api-key: <key>`. Both headers are accepted transparently.
//!
//! Tokens are loaded once at startup from any of:
//!   - `--auth-tokens-file <PATH>`: one token per line, blank lines and
//!     `#` comments ignored.
//!   - `--auth-token <TOKEN>` / `--api-key <TOKEN>`: single inline token.
//!   - `FLUX_API_KEY` (or `ATLAS_API_KEY`) environment variable.
//!
//! Supplying any token source automatically activates authentication even when
//! `--require-auth` is omitted, while `--require-auth` with zero tokens still
//! fails closed at startup.
//!
//! ### Scoped Token Prefixes
//! Configured tokens may carry an explicit scope prefix:
//!   - `read:<secret>`      → grants `read` (GET telemetry, model/Ollama discovery, stored reads)
//!   - `inference:<secret>` → grants `inference` + `read` (generation, tokenization, conversations)
//!   - `admin:<secret>`     → grants `admin` + `inference` + `read` (mutating admin/settings/swap)
//!   - `<secret>`           → unprefixed tokens grant full `admin` + `inference` + `read` scope
//!                            for 100% backwards compatibility.
//!
//! Callers may present either `<secret>` or the full `<scope>:<secret>` string;
//! both authenticate with the exact configured scope and never escalate privileges.
//! Validation uses branchless constant-time byte comparison (`ct_eq`) across all
//! loaded entries so timing does not leak prefix matches or scope boundaries.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use axum::http::{HeaderMap, Method, header};

const SCOPE_READ: u8 = 0b001;
const SCOPE_INFERENCE: u8 = 0b010;
const SCOPE_ADMIN: u8 = 0b100;
const SCOPE_ALL: u8 = SCOPE_READ | SCOPE_INFERENCE | SCOPE_ADMIN;

/// Access level required by an HTTP route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenScope {
    /// Read-only access: GET telemetry, model/Ollama discovery, stored item reads.
    Read,
    /// Generation, tokenization, and stateful session writes (implies `Read`).
    Inference,
    /// Mutating server-wide settings, swap, or admin actions (implies `Inference` + `Read`).
    Admin,
}

impl TokenScope {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Inference => "inference",
            Self::Admin => "admin",
        }
    }

    const fn bit(self) -> u8 {
        match self {
            Self::Read => SCOPE_READ,
            Self::Inference => SCOPE_INFERENCE,
            Self::Admin => SCOPE_ADMIN,
        }
    }
}

/// Outcome of constant-time token + scope verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthOutcome {
    /// Token matched and holds the required [`TokenScope`].
    Allowed,
    /// Token did not match any configured secret.
    InvalidToken,
    /// Token matched a configured secret, but its scope does not cover `required`.
    Forbidden { required: TokenScope },
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TokenEntry {
    secret: Vec<u8>,
    prefixed: Option<Vec<u8>>,
    scope_mask: u8,
}

impl std::fmt::Debug for TokenEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenEntry")
            .field("secret", &"[REDACTED]")
            .field("has_prefix", &self.prefixed.is_some())
            .field("scope_mask", &self.scope_mask)
            .finish()
    }
}

impl TokenEntry {
    fn parse(raw_line: &str) -> Result<Self> {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("auth token must not be empty"));
        }
        let scoped = trimmed
            .strip_prefix("read:")
            .map(|rest| ("read", rest.trim(), SCOPE_READ))
            .or_else(|| {
                trimmed
                    .strip_prefix("inference:")
                    .map(|rest| ("inference", rest.trim(), SCOPE_READ | SCOPE_INFERENCE))
            })
            .or_else(|| {
                trimmed
                    .strip_prefix("admin:")
                    .map(|rest| ("admin", rest.trim(), SCOPE_ALL))
            });

        if let Some((prefix, secret, scope_mask)) = scoped {
            if secret.is_empty() {
                return Err(anyhow!(
                    "scoped auth token '{prefix}:' must contain a non-empty secret"
                ));
            }
            let canonical_prefixed = format!("{prefix}:{secret}").into_bytes();
            Ok(Self {
                secret: secret.as_bytes().to_vec(),
                prefixed: Some(canonical_prefixed),
                scope_mask,
            })
        } else {
            Ok(Self {
                secret: trimmed.as_bytes().to_vec(),
                prefixed: None,
                scope_mask: SCOPE_ALL,
            })
        }
    }
}

/// Loaded token validator. Constructed once at startup; shared via `Arc<AuthConfig>`.
pub struct AuthConfig {
    entries: Vec<TokenEntry>,
}

impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthConfig")
            .field("token_count", &self.entries.len())
            .field("entries", &"[REDACTED]")
            .finish()
    }
}

impl AuthConfig {
    /// Load tokens from a file. One token per line; blank lines and lines
    /// starting with `#` are ignored. Trailing whitespace is trimmed from
    /// each token. Returns an error if the file is empty after parsing or
    /// contains a malformed scoped prefix (`inference:` with empty secret).
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading auth tokens file {}", path.display()))?;
        let mut entries = Vec::new();
        for (idx, line) in raw.lines().enumerate() {
            let s = line.trim();
            if s.is_empty() || s.starts_with('#') {
                continue;
            }
            let entry = TokenEntry::parse(s)
                .with_context(|| format!("parsing auth token at {}:{}", path.display(), idx + 1))?;
            entries.push(entry);
        }
        entries.sort();
        entries.dedup();
        if entries.is_empty() {
            return Err(anyhow!(
                "auth tokens file {} contains no usable tokens \
                 (lines must be non-empty and not start with `#`)",
                path.display()
            ));
        }
        Ok(Self { entries })
    }

    /// Build from a single inline token (`--auth-token`, `--api-key`, or `FLUX_API_KEY`).
    pub fn from_inline(token: &str) -> Result<Self> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("--auth-token must not be empty"));
        }
        let entry = TokenEntry::parse(trimmed)?;
        Ok(Self {
            entries: vec![entry],
        })
    }

    /// Number of distinct token entries loaded. For startup logging only —
    /// never log the tokens themselves.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.entries.len()
    }

    /// Authorize a presented token against `required` scope in constant time
    /// across all loaded token entries.
    #[must_use]
    pub fn authorize(&self, presented: &[u8], required: TokenScope) -> AuthOutcome {
        let mut any_match = 0u8;
        let mut granted_mask = 0u8;
        for entry in &self.entries {
            let match_secret = ct_eq(presented, &entry.secret);
            let match_prefixed = entry
                .prefixed
                .as_deref()
                .map_or(0u8, |prefixed| ct_eq(presented, prefixed));
            let matched = match_secret | match_prefixed;
            // Branchless mask expansion: 0u8 -> 0x00, 1u8 -> 0xFF.
            let mask_selector = 0u8.wrapping_sub(matched);
            any_match |= matched;
            granted_mask |= entry.scope_mask & mask_selector;
        }
        if any_match == 0 {
            AuthOutcome::InvalidToken
        } else if (granted_mask & required.bit()) != 0 {
            AuthOutcome::Allowed
        } else {
            AuthOutcome::Forbidden { required }
        }
    }

    /// Validate whether `presented` matches any loaded token (regardless of
    /// route scope; every valid token holds at least [`TokenScope::Read`]).
    #[must_use]
    pub fn validate(&self, presented: &[u8]) -> bool {
        self.authorize(presented, TokenScope::Read) == AuthOutcome::Allowed
    }
}

/// Extract a presented API token from `Authorization: Bearer <token>`
/// (case-insensitive `Bearer` scheme) or Anthropic's `x-api-key: <token>`.
#[must_use]
pub fn extract_presented_token(headers: &HeaderMap) -> Option<&str> {
    if let Some(auth_val) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let trimmed = auth_val.trim();
        if trimmed.len() > 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
            let tok = trimmed[7..].trim();
            if !tok.is_empty() {
                return Some(tok);
            }
        }
    }
    if let Some(api_key) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        let tok = api_key.trim();
        if !tok.is_empty() {
            return Some(tok);
        }
    }
    None
}

/// Determine the [`TokenScope`] required for `(method, path)`, or `None` for
/// open health/metrics/UI/CORS-preflight routes.
#[must_use]
pub fn required_scope(method: &Method, path: &str) -> Option<TokenScope> {
    if *method == Method::OPTIONS {
        return None;
    }
    if matches!(
        path,
        "/" | "/flux/modes" | "/recipes" | "/health" | "/ready" | "/health/live" | "/metrics"
    ) {
        return None;
    }
    if matches!(path, "/api/tags" | "/api/show") {
        return Some(TokenScope::Read);
    }
    if path == "/tokenize" || path == "/detokenize" {
        return Some(TokenScope::Inference);
    }
    if path.starts_with("/v1/") {
        let is_read_method = matches!(*method, Method::GET | Method::HEAD);
        if path.starts_with("/v1/admin")
            || path == "/v1/swap"
            || path.starts_with("/v1/swap/")
            || path == "/v1/models/swap"
            || path == "/v1/debug/prompt"
            || (path.starts_with("/v1/disagg/") && !is_read_method)
            || (path.starts_with("/v1/flux/") && !is_read_method)
        {
            return Some(TokenScope::Admin);
        }
        if is_read_method {
            return Some(TokenScope::Read);
        }
        return Some(TokenScope::Inference);
    }
    None
}

/// Constant-time secret equality using a per-process random-keyed HMAC-SHA256
/// (`ring::hmac::verify`, eliminating secret-length timing leakage and avoiding
/// deprecated `ring::constant_time`) combined with a non-short-circuiting
/// length check. Returns `1` iff `a == b`, else `0`.
fn ct_eq(a: &[u8], b: &[u8]) -> u8 {
    static KEY: std::sync::OnceLock<ring::hmac::Key> = std::sync::OnceLock::new();
    let key = KEY.get_or_init(|| {
        let rng = ring::rand::SystemRandom::new();
        ring::hmac::Key::generate(ring::hmac::HMAC_SHA256, &rng)
            .unwrap_or_else(|_| ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"flux-auth-ct-eq"))
    });
    let tag_a = ring::hmac::sign(key, a);
    let mac_eq = u8::from(ring::hmac::verify(key, b, tag_a.as_ref()).is_ok());
    let mut len_diff = 0u8;
    for (x, y) in a.len().to_be_bytes().into_iter().zip(b.len().to_be_bytes()) {
        len_diff |= std::hint::black_box(x ^ y);
    }
    mac_eq & u8::from(std::hint::black_box(len_diff) == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn ct_eq_matches_only_when_equal() {
        assert_eq!(ct_eq(b"hello", b"hello"), 1);
        assert_eq!(ct_eq(b"hello", b"world"), 0);
        assert_eq!(ct_eq(b"hello", b"hellz"), 0);
        assert_eq!(ct_eq(b"hello", b"helloo"), 0);
        assert_eq!(ct_eq(b"", b""), 1);
        assert_eq!(ct_eq(b"a", b""), 0);
    }

    #[test]
    fn validates_single_inline_token_with_full_backwards_compat_scope() {
        let cfg = AuthConfig::from_inline("sk-test-token").unwrap();
        assert!(cfg.validate(b"sk-test-token"));
        assert!(!cfg.validate(b"sk-test-toke"));
        assert!(!cfg.validate(b"sk-test-tokenx"));
        assert!(!cfg.validate(b""));
        assert_eq!(cfg.token_count(), 1);
        assert_eq!(
            cfg.authorize(b"sk-test-token", TokenScope::Read),
            AuthOutcome::Allowed
        );
        assert_eq!(
            cfg.authorize(b"sk-test-token", TokenScope::Inference),
            AuthOutcome::Allowed
        );
        assert_eq!(
            cfg.authorize(b"sk-test-token", TokenScope::Admin),
            AuthOutcome::Allowed
        );
    }

    #[test]
    fn rejects_empty_inline_and_empty_scoped_secret() {
        assert!(AuthConfig::from_inline("").is_err());
        assert!(AuthConfig::from_inline("   ").is_err());
        assert!(AuthConfig::from_inline("read:").is_err());
        assert!(AuthConfig::from_inline("inference:   ").is_err());
        assert!(AuthConfig::from_inline("admin:").is_err());
    }

    #[test]
    fn loads_file_with_comments_and_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.txt");
        std::fs::write(
            &path,
            "# project A\n\
             alpha-token\n\
             \n\
             # project B\n\
             beta-token\n\
             alpha-token\n",
        )
        .unwrap();
        let cfg = AuthConfig::from_file(&path).unwrap();
        assert_eq!(cfg.token_count(), 2);
        assert!(cfg.validate(b"alpha-token"));
        assert!(cfg.validate(b"beta-token"));
        assert!(!cfg.validate(b"# project A"));
        assert!(!cfg.validate(b"gamma-token"));
    }

    #[test]
    fn empty_file_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, "# only a comment\n\n   \n").unwrap();
        assert!(AuthConfig::from_file(&path).is_err());
    }

    #[test]
    fn scoped_tokens_enforce_read_inference_and_admin_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scoped.txt");
        std::fs::write(
            &path,
            "read:viewer-secret\n\
             inference:runner-secret\n\
             admin:ops-secret\n",
        )
        .unwrap();
        let cfg = AuthConfig::from_file(&path).unwrap();
        assert_eq!(cfg.token_count(), 3);

        // Read token: allowed on Read, forbidden on Inference & Admin.
        for tok in [
            b"viewer-secret".as_slice(),
            b"read:viewer-secret".as_slice(),
        ] {
            assert_eq!(cfg.authorize(tok, TokenScope::Read), AuthOutcome::Allowed);
            assert_eq!(
                cfg.authorize(tok, TokenScope::Inference),
                AuthOutcome::Forbidden {
                    required: TokenScope::Inference
                }
            );
            assert_eq!(
                cfg.authorize(tok, TokenScope::Admin),
                AuthOutcome::Forbidden {
                    required: TokenScope::Admin
                }
            );
        }
        // Forging `admin:viewer-secret` must NOT match.
        assert_eq!(
            cfg.authorize(b"admin:viewer-secret", TokenScope::Admin),
            AuthOutcome::InvalidToken
        );

        // Inference token: allowed on Read + Inference, forbidden on Admin.
        for tok in [
            b"runner-secret".as_slice(),
            b"inference:runner-secret".as_slice(),
        ] {
            assert_eq!(cfg.authorize(tok, TokenScope::Read), AuthOutcome::Allowed);
            assert_eq!(
                cfg.authorize(tok, TokenScope::Inference),
                AuthOutcome::Allowed
            );
            assert_eq!(
                cfg.authorize(tok, TokenScope::Admin),
                AuthOutcome::Forbidden {
                    required: TokenScope::Admin
                }
            );
        }

        // Admin token: allowed everywhere.
        for tok in [b"ops-secret".as_slice(), b"admin:ops-secret".as_slice()] {
            assert_eq!(cfg.authorize(tok, TokenScope::Read), AuthOutcome::Allowed);
            assert_eq!(
                cfg.authorize(tok, TokenScope::Inference),
                AuthOutcome::Allowed
            );
            assert_eq!(cfg.authorize(tok, TokenScope::Admin), AuthOutcome::Allowed);
        }
    }

    #[test]
    fn extracts_bearer_and_anthropic_x_api_key_headers() {
        let mut h = HeaderMap::new();
        assert_eq!(extract_presented_token(&h), None);

        h.insert("x-api-key", HeaderValue::from_static("anth-secret"));
        assert_eq!(extract_presented_token(&h), Some("anth-secret"));

        // Bearer takes precedence when both are present, case-insensitively.
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("bearer bearer-secret"),
        );
        assert_eq!(extract_presented_token(&h), Some("bearer-secret"));
    }

    #[test]
    fn required_scope_maps_telemetry_inference_and_admin_routes() {
        assert_eq!(required_scope(&Method::GET, "/health"), None);
        assert_eq!(required_scope(&Method::GET, "/ready"), None);
        assert_eq!(required_scope(&Method::GET, "/health/live"), None);
        assert_eq!(required_scope(&Method::GET, "/metrics"), None);
        assert_eq!(
            required_scope(&Method::OPTIONS, "/v1/chat/completions"),
            None
        );

        assert_eq!(
            required_scope(&Method::GET, "/v1/models"),
            Some(TokenScope::Read)
        );
        assert_eq!(
            required_scope(&Method::GET, "/v1/memory"),
            Some(TokenScope::Read)
        );
        assert_eq!(
            required_scope(&Method::GET, "/v1/experts/usage"),
            Some(TokenScope::Read)
        );
        assert_eq!(
            required_scope(&Method::GET, "/api/tags"),
            Some(TokenScope::Read)
        );
        assert_eq!(
            required_scope(&Method::POST, "/api/show"),
            Some(TokenScope::Read)
        );
        assert_eq!(
            required_scope(&Method::GET, "/v1/flux/settings"),
            Some(TokenScope::Read)
        );

        assert_eq!(
            required_scope(&Method::POST, "/v1/chat/completions"),
            Some(TokenScope::Inference)
        );
        assert_eq!(
            required_scope(&Method::POST, "/v1/messages"),
            Some(TokenScope::Inference)
        );
        assert_eq!(
            required_scope(&Method::POST, "/v1/decisions"),
            Some(TokenScope::Inference)
        );
        assert_eq!(
            required_scope(&Method::POST, "/tokenize"),
            Some(TokenScope::Inference)
        );

        assert_eq!(
            required_scope(&Method::PATCH, "/v1/flux/settings"),
            Some(TokenScope::Admin)
        );
        assert_eq!(
            required_scope(&Method::POST, "/v1/models/swap"),
            Some(TokenScope::Admin)
        );
        assert_eq!(
            required_scope(&Method::POST, "/v1/admin/reload"),
            Some(TokenScope::Admin)
        );
        assert_eq!(
            required_scope(&Method::POST, "/v1/disagg/decode"),
            Some(TokenScope::Admin)
        );
        assert_eq!(
            required_scope(&Method::POST, "/v1/debug/prompt"),
            Some(TokenScope::Admin)
        );
    }

    #[test]
    fn debug_redacts_secrets() {
        let cfg = AuthConfig::from_inline("admin:super-secret-value-12345").unwrap();
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("[REDACTED]"), "{dbg}");
        assert!(!dbg.contains("super-secret-value-12345"), "{dbg}");
        assert!(
            !dbg.contains("115, 117"),
            "must not dump secret bytes: {dbg}"
        );
    }
}
