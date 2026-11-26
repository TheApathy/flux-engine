// SPDX-License-Identifier: AGPL-3.0-only

//! The Benchmarks pane: is the box fit to measure on, and what did the
//! harnesses record?
//!
//! Deliberately read-only: measurements are taken by the standalone `bench/`
//! harnesses (with their prewarm assertions and GPU lock discipline) rather
//! than from inside the TUI, so this pane reports readiness and recent harness
//! results directly from disk.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::panel;
use crate::tui::app::{App, BenchSub};
use crate::tui::data::bench;
use crate::tui::data::metrics_poll::{effective_weights_gbps, model_weights_gb};
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    if area.height < 3 {
        return;
    }
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(2)])
        .split(area);
    draw_subtab_strip(f, app.bench_sub, rows[0]);
    match app.bench_sub {
        BenchSub::Box => draw_box(f, app, rows[1]),
        BenchSub::Runs => draw_runs(f, app, rows[1]),
    }
}

fn draw_subtab_strip(f: &mut Frame, sub: BenchSub, area: Rect) {
    let active_style = theme::brand_cyan().add_modifier(Modifier::BOLD);
    let inactive_style = theme::dim();
    let (box_glyph, box_style, runs_glyph, runs_style) = match sub {
        BenchSub::Box => ("▸", active_style, "◦", inactive_style),
        BenchSub::Runs => ("◦", inactive_style, "▸", active_style),
    };
    let line = Line::from(vec![
        Span::styled(format!(" {box_glyph} Box readiness "), box_style),
        Span::styled("  ", theme::dim()),
        Span::styled(format!(" {runs_glyph} Recorded runs "), runs_style),
        Span::styled("   ←/→ switch sub-view", theme::dim()),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn draw_box(f: &mut Frame, app: &App, area: Rect) {
    let snap = &app.bench;
    let lock_path = &snap.lock_path;
    let st = &snap.box_state;
    let (verdict, blocked) = if snap.sampled {
        let v = st.verdict();
        let b = v.starts_with("DO NOT");
        (v, b)
    } else {
        ("sampling box readiness…".to_string(), false)
    };
    let mem_gb = st.mem_available_mib as f64 / 1024.0;
    let active_reqs = app.stats.requests_active;
    let active_seqs = app.stats.sched.map_or(0, |s| s.active_seqs);

    let mut lines = vec![
        Line::from(Span::styled(
            format!("  {verdict}"),
            Style::default()
                .fg(if blocked {
                    theme::ERROR.color()
                } else {
                    theme::GREEN.color()
                })
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  GPU lock      ", theme::text2()),
            match st.lock_holder {
                Some(p) => Span::styled(format!("✗ held by pid {p}"), theme::warn()),
                None => Span::styled("✓ free", theme::brand_green()),
            },
            Span::styled(format!("  ({})", lock_path.display()), theme::dim()),
        ]),
        Line::from(vec![
            Span::styled("  queued        ", theme::text2()),
            if st.waiters.is_empty() {
                Span::styled("nobody", theme::text())
            } else {
                Span::styled(
                    st.waiters
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    theme::warn(),
                )
            },
        ]),
        Line::from(vec![
            Span::styled("  UMA headroom  ", theme::text2()),
            Span::styled(
                format!("{mem_gb:.1} GB available"),
                if st.mem_available_mib < 40 * 1024 {
                    theme::warn()
                } else {
                    theme::brand_green()
                },
            ),
            Span::styled("  (40.0 GB floor)", theme::dim()),
        ]),
        Line::from(vec![
            Span::styled("  builders      ", theme::text2()),
            Span::styled(
                format!("{}", st.builders),
                if st.builders > 0 {
                    theme::warn()
                } else {
                    theme::text()
                },
            ),
            Span::styled("  (cargo / rustc / nvcc / ptxas)", theme::dim()),
        ]),
        Line::from(vec![
            Span::styled("  this server   ", theme::text2()),
            Span::styled(
                format!("{active_reqs} active req · {active_seqs} active seq"),
                if active_reqs > 0 || active_seqs > 0 {
                    theme::brand_cyan()
                } else {
                    theme::dim()
                },
            ),
        ]),
        Line::from(""),
        // The three things that have actually invalidated measurements here,
        // named so the pane teaches the hazard rather than only reporting it.
        Line::from(Span::styled(
            "  A run taken while another job holds the lock is not a slower",
            theme::dim(),
        )),
        Line::from(Span::styled(
            "  number — it is a wrong one. `nvidia-smi` shows a clear GPU",
            theme::dim(),
        )),
        Line::from(Span::styled(
            "  between arms while the lock is correctly held: the lock is the",
            theme::dim(),
        )),
        Line::from(Span::styled(
            "  authority, the GPU is a lagging indicator.",
            theme::dim(),
        )),
    ];
    if st.lock_holder.is_some() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  A dead holder pid is NOT a stale lock: it survives through a",
            theme::dim(),
        )));
        lines.push(Line::from(Span::styled(
            "  descriptor an alive child inherited. Do not clear it.",
            theme::dim(),
        )));
    }
    f.render_widget(
        Paragraph::new(lines).block(panel("BOX READINESS ─".into(), !blocked)),
        area,
    );
}

fn is_control_arm(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    if n.contains("ctrl") || n.contains("control") {
        return true;
    }
    n.split(['-', '_']).any(|seg| {
        seg == "c"
            || seg
                .strip_prefix('c')
                .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
    })
}

pub(crate) fn format_delta_pct(median: f64, ctrl_mean: Option<f64>) -> String {
    let Some(base) = ctrl_mean.filter(|&m| m > 0.0) else {
        return "      —".to_string();
    };
    let pct = (median - base) / base * 100.0;
    format!("{pct:+6.1}%")
}

pub(crate) fn render_arm_rows(
    arms: &[bench::ArmSummary],
    weights_gb: Option<f64>,
    max_rows: usize,
) -> Vec<Line<'static>> {
    let controls: Vec<bench::ArmSummary> = arms
        .iter()
        .filter(|a| is_control_arm(&a.name))
        .cloned()
        .collect();
    let spread = bench::control_spread_pct(&controls);
    let ctrl_mean = (!controls.is_empty())
        .then(|| controls.iter().map(|c| c.median_tok_s).sum::<f64>() / (controls.len() as f64));
    let max_median = arms.iter().map(|a| a.median_tok_s).fold(1.0_f64, f64::max);

    let mut lines = Vec::new();
    match spread {
        Some(pct) => lines.push(Line::from(Span::styled(
            format!(
                "  control spread {pct:.2}% over {} arms — THE FLOOR: deltas within ±{pct:.2}% are noise",
                controls.len()
            ),
            Style::default()
                .fg(theme::WARN.color())
                .add_modifier(Modifier::BOLD),
        ))),
        None => lines.push(Line::from(Span::styled(
            "  fewer than two control arms — add ≥2 control runs (*-ctrl*) to establish noise floor",
            theme::dim(),
        ))),
    }
    lines.push(Line::from(""));

    let bw_header = if weights_gb.is_some() {
        "  wt GB/s"
    } else {
        ""
    };
    lines.push(Line::from(Span::styled(
        format!(
            "  {:<22} {:>6} {:>11} {:>8} {:>9} {:>9}  {:<12}{bw_header}",
            "arm", "trials", "median tok/s", "Δ ctrl", "min", "max", "relative"
        ),
        theme::dim(),
    )));

    let bar_w = 12usize;
    let noise_floor = spread.unwrap_or(1.5);
    for a in arms.iter().take(max_rows) {
        let delta_str = format_delta_pct(a.median_tok_s, ctrl_mean);
        let delta_style = match ctrl_mean.filter(|&m| m > 0.0) {
            Some(base) => {
                let pct = (a.median_tok_s - base) / base * 100.0;
                if pct > noise_floor {
                    theme::brand_green()
                } else if pct < -noise_floor {
                    theme::warn()
                } else {
                    theme::dim()
                }
            }
            None => theme::dim(),
        };
        let frac = (a.median_tok_s / max_median).clamp(0.0, 1.0);
        let filled = ((frac * bar_w as f64).round() as usize).min(bar_w);
        let bar = "█".repeat(filled) + &"░".repeat(bar_w - filled);
        let name_trunc: String = if a.name.chars().count() > 22 {
            a.name.chars().take(21).collect::<String>() + "…"
        } else {
            a.name.clone()
        };

        let mut spans = vec![
            Span::styled(format!("  {name_trunc:<22}"), theme::text()),
            Span::styled(format!(" {:>6}", a.trials), theme::text2()),
            Span::styled(
                format!(" {:>11.1}", a.median_tok_s),
                theme::text().add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" {delta_str:>8}"), delta_style),
            Span::styled(
                format!(" {:>9.1} {:>9.1}  ", a.min_tok_s, a.max_tok_s),
                theme::text2(),
            ),
            Span::styled(bar, theme::brand_cyan()),
        ];
        if let Some(gbps) = effective_weights_gbps(a.median_tok_s, weights_gb) {
            spans.push(Span::styled(format!("  {gbps:>7.0}"), theme::text2()));
        }
        lines.push(Line::from(spans));
    }
    lines
}

fn draw_runs(f: &mut Frame, app: &App, area: Rect) {
    let snap = &app.bench;
    if !snap.sampled {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  Sampling recorded runs…",
                theme::dim(),
            )))
            .block(panel("RECORDED RUNS ─".into(), false)),
            area,
        );
        return;
    }
    if snap.runs_dirs.is_empty() {
        f.render_widget(
            Paragraph::new(vec![
                Line::from("  No run directories configured."),
                Line::from(""),
                Line::from(Span::styled(
                    "  Set ATLAS_BENCH_RUNS to a colon-separated list of the",
                    theme::dim(),
                )),
                Line::from(Span::styled(
                    "  harnesses' runs/ directories, e.g.",
                    theme::dim(),
                )),
                Line::from(Span::styled(
                    "    ATLAS_BENCH_RUNS=/path/to/bench/runs",
                    theme::dim(),
                )),
            ])
            .block(panel("RECORDED RUNS ─".into(), false)),
            area,
        );
        return;
    }

    let arms = &snap.arms;
    let weights = model_weights_gb(
        app.progress.disk_gb,
        app.stats.gpu_known,
        app.stats.atlas_used_gb,
    );
    let max_rows = area.height.saturating_sub(6) as usize;
    let lines = render_arm_rows(arms, weights, max_rows);
    f.render_widget(
        Paragraph::new(lines).block(panel(
            format!("RECORDED RUNS ─ {} arms ─", arms.len()),
            true,
        )),
        area,
    );
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::tui::app::Section;
    use crate::tui::data::bench::resolve_gpu_lock_path;
    use crate::tui::render::harness::{has, screen};

    #[test]
    fn resolve_gpu_lock_prefers_explicit_env_then_ancestor_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let lock = dir.path().join(".gb10.lock");
        std::fs::write(&lock, "").expect("create lock");
        let nested = dir.path().join("crates/spark-server");
        std::fs::create_dir_all(&nested).expect("mkdir");

        assert_eq!(
            resolve_gpu_lock_path(Some("/custom/gpu.lock"), None, Some(&nested)),
            PathBuf::from("/custom/gpu.lock")
        );
        assert_eq!(resolve_gpu_lock_path(None, None, Some(&nested)), lock);
    }

    #[test]
    fn render_arm_rows_computes_control_delta_and_weight_bandwidth() {
        let arms = vec![
            bench::ArmSummary {
                name: "dflash-k8-cand".into(),
                trials: 5,
                median_tok_s: 115.0,
                min_tok_s: 112.0,
                max_tok_s: 117.0,
            },
            bench::ArmSummary {
                name: "base-ctrl-1".into(),
                trials: 5,
                median_tok_s: 100.0,
                min_tok_s: 99.0,
                max_tok_s: 101.0,
            },
            bench::ArmSummary {
                name: "base-ctrl-2".into(),
                trials: 5,
                median_tok_s: 100.0,
                min_tok_s: 99.5,
                max_tok_s: 100.5,
            },
        ];
        let lines = render_arm_rows(&arms, Some(20.0), 10);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("control spread 0.00% over 2 arms"), "{text}");
        assert!(text.contains("+15.0%"), "{text}");
        assert!(
            text.contains("2300"),
            "115 tok/s * 20 GB = 2300 GB/s:\n{text}"
        );
    }

    #[test]
    fn benchmarks_tab_draws_cleanly_for_both_subtabs_and_tiny_frames() {
        let mut a = crate::tui::render::tests::app();
        a.section = Section::Benchmarks;
        a.bench_sub = BenchSub::Box;
        let box_rows = screen(&a, 140, 36);
        assert!(has(&box_rows, "BOX READINESS"), "{box_rows:#?}");
        assert!(has(&box_rows, "UMA headroom"), "{box_rows:#?}");

        a.bench_sub = BenchSub::Runs;
        let run_rows = screen(&a, 140, 36);
        assert!(has(&run_rows, "RECORDED RUNS"), "{run_rows:#?}");

        for (w, h) in [(20u16, 4u16), (40, 10), (160, 3)] {
            assert_eq!(screen(&a, w, h).len(), h as usize);
        }
    }
}
