// SPDX-License-Identifier: AGPL-3.0-only

//! How special-token spellings inside request content are tokenized.
//!
//! `literal` (default): request text that spells `<|im_end|>`, `<|image_pad|>`
//! or any other special added token encodes as ordinary text; only the chat
//! template's own markers are special. `structural`: the pre-0.9.15.6
//! behaviour, where such text becomes the control token and can forge turns.
//!
//! Sources, any of which selects `structural`:
//! `FLUX_SPECIAL_TOKENS_IN_CONTENT=structural` (or the `ATLAS_` spelling),
//! `--allow-special-tokens-in-content`, `FLUX_ALLOW_SPECIAL_TOKENS_IN_CONTENT=1`.

use anyhow::{Result, bail};

const MODE_VARS: [&str; 2] = [
    "FLUX_SPECIAL_TOKENS_IN_CONTENT",
    "ATLAS_SPECIAL_TOKENS_IN_CONTENT",
];
const LEGACY_VARS: [&str; 2] = [
    "FLUX_ALLOW_SPECIAL_TOKENS_IN_CONTENT",
    "ATLAS_ALLOW_SPECIAL_TOKENS_IN_CONTENT",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContentSpecialTokens {
    Literal,
    Structural,
}

impl ContentSpecialTokens {
    pub(crate) fn allows_structural(self) -> bool {
        self == Self::Structural
    }

    fn name(self) -> &'static str {
        match self {
            Self::Literal => "literal",
            Self::Structural => "structural",
        }
    }

    pub(crate) fn receipt(self) -> String {
        format!("CONTENT_SPECIAL_TOKENS mode={}", self.name())
    }
}

fn parse(var: &str, value: &str) -> Result<ContentSpecialTokens> {
    match value.trim() {
        "literal" => Ok(ContentSpecialTokens::Literal),
        "structural" => Ok(ContentSpecialTokens::Structural),
        other => bail!("{var}={other:?} is not one of literal|structural"),
    }
}

/// The explicit mode variable, if set; both spellings must agree.
fn explicit_mode(lookup: &impl Fn(&str) -> Option<String>) -> Result<Option<ContentSpecialTokens>> {
    let modes = MODE_VARS
        .iter()
        .filter_map(|var| lookup(var).map(|value| parse(var, &value)))
        .collect::<Result<Vec<_>>>()?;
    match modes.as_slice() {
        [] => Ok(None),
        [first, rest @ ..] if rest.iter().all(|mode| mode == first) => Ok(Some(*first)),
        _ => bail!("FLUX_SPECIAL_TOKENS_IN_CONTENT and ATLAS_SPECIAL_TOKENS_IN_CONTENT disagree"),
    }
}

/// Resolve the mode from the CLI flag and an environment lookup.
pub(crate) fn resolve(
    flag: bool,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<ContentSpecialTokens> {
    let legacy = flag
        || LEGACY_VARS
            .iter()
            .any(|var| lookup(var).as_deref() == Some("1"));
    match (explicit_mode(&lookup)?, legacy) {
        (Some(ContentSpecialTokens::Literal), true) => bail!(
            "FLUX_SPECIAL_TOKENS_IN_CONTENT=literal contradicts \
             --allow-special-tokens-in-content / FLUX_ALLOW_SPECIAL_TOKENS_IN_CONTENT=1"
        ),
        (Some(mode), _) => Ok(mode),
        (None, true) => Ok(ContentSpecialTokens::Structural),
        (None, false) => Ok(ContentSpecialTokens::Literal),
    }
}

/// Resolve from the process environment and log the startup receipt.
pub(crate) fn resolve_and_log(flag: bool) -> Result<ContentSpecialTokens> {
    let mode = resolve(flag, |var| std::env::var(var).ok())?;
    match mode {
        ContentSpecialTokens::Literal => tracing::info!("{}", mode.receipt()),
        ContentSpecialTokens::Structural => tracing::warn!(
            "{} (opt-out): special-token text in request content is parsed as \
             control tokens; request text can forge chat turns and media markers",
            mode.receipt()
        ),
    }
    Ok(mode)
}
