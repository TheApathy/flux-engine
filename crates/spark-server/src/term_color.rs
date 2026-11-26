// SPDX-License-Identifier: AGPL-3.0-only

//! Colored Rust terminal output for plain (`--no-tui`) server startup & status cards.
//!
//! Uses the same Flux brand palette as `tui::theme` (`#BE9DF8` purple,
//! `#49C3DB` cyan, `#12B981` green, `#E5C07B` amber) in 24-bit Truecolor when
//! `COLORTERM` advertises `truecolor`/`24bit`, with clean ANSI-256 fallback so
//! both `docker run` and `docker logs` render crisp status colors.
//! Strictly honors `NO_COLOR` (non-empty disables all ANSI escapes per
//! <https://no-color.org>), `CLICOLOR=0`, and `TERM=dumb`.

use std::fmt::Write as _;

use crate::main_modules::model_host::ModelHost;

/// Resolved ANSI color capability for plain-mode Rust terminal output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColorMode {
    /// No ANSI escapes (honors `NO_COLOR`, `CLICOLOR=0`, `TERM=dumb`).
    None,
    /// Pinned 256-color palette matching `tui::theme`.
    Ansi256,
    /// 24-bit Truecolor (`\x1b[38;2;R;G;Bm`).
    TrueColor,
}

/// Pure env resolver for [`ColorMode`], matching `NO_COLOR` precedence.
pub(crate) fn color_mode_of(
    no_color: Option<&str>,
    clicolor: Option<&str>,
    clicolor_force: Option<&str>,
    term: Option<&str>,
    colorterm: Option<&str>,
) -> ColorMode {
    if no_color.is_some_and(|v| !v.is_empty()) {
        return ColorMode::None;
    }
    let truecolor = colorterm.is_some_and(|v| v.contains("truecolor") || v.contains("24bit"));
    if clicolor_force.is_some_and(|v| !v.is_empty() && v != "0") {
        return if truecolor {
            ColorMode::TrueColor
        } else {
            ColorMode::Ansi256
        };
    }
    if clicolor == Some("0") || term == Some("dumb") {
        return ColorMode::None;
    }
    if truecolor {
        ColorMode::TrueColor
    } else {
        ColorMode::Ansi256
    }
}

/// Resolve current process color mode from the environment.
pub(crate) fn current_color_mode() -> ColorMode {
    color_mode_of(
        std::env::var("NO_COLOR").ok().as_deref(),
        std::env::var("CLICOLOR").ok().as_deref(),
        std::env::var("CLICOLOR_FORCE").ok().as_deref(),
        std::env::var("TERM").ok().as_deref(),
        std::env::var("COLORTERM").ok().as_deref(),
    )
}

#[derive(Clone, Copy)]
struct Palette {
    purple: &'static str,
    cyan: &'static str,
    green: &'static str,
    amber: &'static str,
    dim: &'static str,
    bold: &'static str,
    reset: &'static str,
}

impl Palette {
    const fn for_mode(mode: ColorMode) -> Self {
        match mode {
            ColorMode::None => Self {
                purple: "",
                cyan: "",
                green: "",
                amber: "",
                dim: "",
                bold: "",
                reset: "",
            },
            ColorMode::Ansi256 => Self {
                purple: "\x1b[1;38;5;141m",
                cyan: "\x1b[1;38;5;80m",
                green: "\x1b[1;38;5;36m",
                amber: "\x1b[1;38;5;179m",
                dim: "\x1b[38;5;244m",
                bold: "\x1b[1m",
                reset: "\x1b[0m",
            },
            ColorMode::TrueColor => Self {
                purple: "\x1b[1;38;2;190;157;248m",
                cyan: "\x1b[1;38;2;73;195;219m",
                green: "\x1b[1;38;2;18;185;129m",
                amber: "\x1b[1;38;2;229;192;123m",
                dim: "\x1b[38;2;147;151;160m",
                bold: "\x1b[1m",
                reset: "\x1b[0m",
            },
        }
    }
}

/// Snapshot rendered in the Rust plain-mode `READY` terminal card.
pub(crate) struct ReadyBannerInfo {
    pub version: String,
    pub model_name: String,
    pub bind: String,
    pub port: u16,
    pub concurrency: Option<u64>,
    pub max_seq_len: Option<u64>,
    pub speculation: Option<String>,
}

/// Format a crisp, color-coded plain-mode Rust `READY` card.
pub(crate) fn render_ready_banner(info: &ReadyBannerInfo, mode: ColorMode) -> String {
    let p = Palette::for_mode(mode);
    let ver = if info.version.starts_with('v') {
        info.version.clone()
    } else {
        format!("v{}", info.version)
    };
    let display_host = if info.bind == "0.0.0.0" || info.bind == "::" {
        "localhost"
    } else {
        info.bind.as_str()
    };

    let mut serving_parts = Vec::new();
    if let Some(c) = info.concurrency {
        serving_parts.push(format!("C={c}"));
    }
    if let Some(ctx) = info.max_seq_len {
        serving_parts.push(format!("ctx={ctx}"));
    }
    if let Some(spec) = info.speculation.as_deref() {
        serving_parts.push(format!("spec={spec}"));
    }
    let serving_line = if serving_parts.is_empty() {
        "auto".to_string()
    } else {
        serving_parts.join(" · ")
    };

    let mut out = String::with_capacity(512);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{dim}╭──────────────────────────────────────────────────────────────────────────────╮{r}",
        dim = p.dim,
        r = p.reset,
    );
    let _ = writeln!(
        out,
        "{dim}│{r}  {purp}⚡ FLUX ENGINE{r} {green}{ver}{r}  {green}● READY{r}  {dim}({bind}:{port}){r}",
        dim = p.dim,
        purp = p.purple,
        green = p.green,
        r = p.reset,
        bind = info.bind,
        port = info.port,
    );
    let _ = writeln!(
        out,
        "{dim}│{r}  {dim}Model      :{r} {bold}{model}{r}  {cyan}[{serving}]{r}",
        dim = p.dim,
        bold = p.bold,
        cyan = p.cyan,
        r = p.reset,
        model = info.model_name,
        serving = serving_line,
    );
    let _ = writeln!(
        out,
        "{dim}│{r}  {dim}Web Studio :{r} {cyan}http://{host}:{port}/{r}",
        dim = p.dim,
        cyan = p.cyan,
        r = p.reset,
        host = display_host,
        port = info.port,
    );
    let _ = writeln!(
        out,
        "{dim}│{r}  {dim}OpenAI API :{r} {cyan}http://{host}:{port}/v1/chat/completions{r}  {dim}·  Health:{r} {green}/health{r}",
        dim = p.dim,
        cyan = p.cyan,
        green = p.green,
        r = p.reset,
        host = display_host,
        port = info.port,
    );
    let _ = writeln!(
        out,
        "{dim}│{r}  {amber}Tip:{r} {dim}pass -it to docker run for the interactive 24-bit Truecolor Rust TUI{r}",
        dim = p.dim,
        amber = p.amber,
        r = p.reset,
    );
    let _ = writeln!(
        out,
        "{dim}╰──────────────────────────────────────────────────────────────────────────────╯{r}",
        dim = p.dim,
        r = p.reset,
    );
    out
}

/// Print the colored Rust `READY` banner once the HTTP listener binds in plain mode.
///
/// Strictly skipped while the Ratatui TUI owns the alternate screen so raw stdout
/// writes can never corrupt the TUI viewport.
pub(crate) fn print_ready_banner(host: &ModelHost, bind: &str, port: u16) {
    if crate::tui::is_active() {
        return;
    }
    let current = host.current();
    let model_name = current
        .as_ref()
        .map(|s| s.model_name.clone())
        .or_else(|| std::env::var("FLUX_MODEL_ID").ok())
        .unwrap_or_else(|| "no model loaded".to_string());
    let serving = current.as_ref().map(|s| &s.serving);
    let info = ReadyBannerInfo {
        version: crate::api::flux_ui::engine_version(),
        model_name,
        bind: bind.to_string(),
        port,
        concurrency: serving.and_then(|s| s.get("concurrency").and_then(serde_json::Value::as_u64)),
        max_seq_len: serving.and_then(|s| s.get("max_seq_len").and_then(serde_json::Value::as_u64)),
        speculation: serving.and_then(|s| {
            s.get("speculation")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
        }),
    };
    let rendered = render_ready_banner(&info, current_color_mode());
    print!("{rendered}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_color_always_disables_ansi_even_with_truecolor_or_force() {
        assert_eq!(
            color_mode_of(
                Some("1"),
                None,
                Some("1"),
                Some("xterm-256color"),
                Some("truecolor")
            ),
            ColorMode::None
        );
        assert_eq!(
            color_mode_of(Some("0"), None, None, None, Some("truecolor")),
            ColorMode::None
        );
        assert_eq!(
            color_mode_of(Some(""), None, None, None, Some("truecolor")),
            ColorMode::TrueColor
        );
    }

    #[test]
    fn clicolor_zero_and_dumb_term_disable_unless_forced() {
        assert_eq!(
            color_mode_of(
                None,
                Some("0"),
                None,
                Some("xterm-256color"),
                Some("truecolor")
            ),
            ColorMode::None
        );
        assert_eq!(
            color_mode_of(None, None, None, Some("dumb"), Some("truecolor")),
            ColorMode::None
        );
        assert_eq!(
            color_mode_of(None, Some("0"), Some("1"), Some("dumb"), Some("24bit")),
            ColorMode::TrueColor
        );
    }

    #[test]
    fn truecolor_and_ansi256_fallback_resolution() {
        assert_eq!(
            color_mode_of(None, None, None, Some("xterm-256color"), Some("truecolor")),
            ColorMode::TrueColor
        );
        assert_eq!(
            color_mode_of(None, None, None, Some("xterm-256color"), None),
            ColorMode::Ansi256
        );
    }

    #[test]
    fn ready_banner_renders_clean_ansi_and_plain_modes() {
        let info = ReadyBannerInfo {
            version: "1.0.0".to_string(),
            model_name: "Qwen3.8-Flash-Next-NVFP4".to_string(),
            bind: "0.0.0.0".to_string(),
            port: 8899,
            concurrency: Some(16),
            max_seq_len: Some(262_144),
            speculation: Some("dflash".to_string()),
        };
        let plain = render_ready_banner(&info, ColorMode::None);
        assert!(!plain.contains('\x1b'));
        assert!(plain.contains("FLUX ENGINE v1.0.0"));
        assert!(plain.contains("Qwen3.8-Flash-Next-NVFP4"));
        assert!(plain.contains("C=16 · ctx=262144 · spec=dflash"));
        assert!(plain.contains("http://localhost:8899/"));

        let ansi = render_ready_banner(&info, ColorMode::Ansi256);
        assert!(ansi.contains("\x1b[1;38;5;141m"));
        assert!(ansi.contains("\x1b[1;38;5;36m"));

        let tc = render_ready_banner(&info, ColorMode::TrueColor);
        assert!(tc.contains("\x1b[1;38;2;190;157;248m"));
        assert!(tc.contains("\x1b[1;38;2;73;195;219m"));
    }
}
