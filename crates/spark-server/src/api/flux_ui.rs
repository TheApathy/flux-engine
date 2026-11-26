// SPDX-License-Identifier: AGPL-3.0-only

//! Self-guiding Flux Engine Recipe Studio & Mission Control UI (`GET /`, `GET /flux/modes`).

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{Html, IntoResponse, Json, Response};
use std::sync::Arc;

use crate::main_modules::model_host::ModelHost;

pub(crate) fn resolved_engine_version_from(env_val: Option<&str>) -> String {
    env_val
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| crate::cli::ATLAS_VERSION.to_string())
}

pub(crate) fn engine_version() -> String {
    resolved_engine_version_from(std::env::var("FLUX_ENGINE_VERSION").ok().as_deref())
}

/// Structured self-guiding recipe modes catalog served at `/flux/modes`, `/recipes`, and `/v1/flux/modes`.
pub async fn modes_json(State(host): State<Arc<ModelHost>>) -> Response {
    let active_mode = std::env::var("FLUX_MODE").unwrap_or_else(|_| "auto".to_string());
    let engine_version = engine_version();
    let current = host.current();
    let ready = current
        .as_ref()
        .is_some_and(|s| s.model_ready.load(std::sync::atomic::Ordering::Relaxed));
    let model_name = current
        .as_ref()
        .map(|s| s.model_name.clone())
        .or_else(|| std::env::var("FLUX_MODEL_ID").ok())
        .unwrap_or_else(|| "Qwen3.8-Flash-Next-NVFP4".to_string());
    let snap = crate::scheduler::snapshot::read();

    let is_qwen4 = host.args().as_ref().map(|a| a.qwen4_qsa).unwrap_or(false)
        || model_name.to_lowercase().contains("qwen4");
    let agent_title = if is_qwen4 {
        "Agentic Coding Harness Mode (Tool Parser + 16K Budget)"
    } else {
        "Agentic Coding Harness Mode (Prefix Cache + 16 SSM Slots)"
    };
    let agent_speculative = if is_qwen4 {
        "native-mtp (K=5) + Tool-Call Parser (16K budget)"
    } else {
        "native-mtp (K=5) + Radix Prefix Cache + 16 SSM State Slots"
    };
    let agent_loop_desc = if is_qwen4 {
        "Multi-turn agentic loops with tool calls and 16K tool budget"
    } else {
        "Multi-turn agentic loops with tool calls and warm KV + GDN/SSM state reuse"
    };

    // Only resolved load-time facts; absent (null) until a model has loaded.
    let serving = current.as_ref().map(|s| s.serving.clone());
    let body = serde_json::json!({
        "engine": "flux-engine",
        // What this server actually serves, for routers and launchers (flux-router reads
        // `serving.concurrency` to cap each backend). Absent while no serve args exist.
        "serving": serving,
        "version": engine_version,
        "status": if ready { "ready" } else { "loading" },
        "active_model": model_name,
        "active_mode": active_mode,
        "scheduler": snap.map(|s| serde_json::json!({
            "active_seqs": s.active_seqs,
            "prefilling_seqs": s.prefilling_seqs,
            "delivered_tps": s.delivered_tps,
            "steps_total": s.steps_total,
            "kv_blocks_free": s.kv_blocks_free,
            "kv_blocks_total": s.kv_blocks_total,
        })),
        "modes": [
            {
                "id": "max-prefill",
                "aliases": ["auto", "nextgen", "v1.0", "w16384"],
                "title": "Next-Gen Max-Prefill & 0-ULP Speculative (Default)",
                "recipe_file": "flux-nextgen-max-prefill-decode.env",
                "prefill_chunk": 16384,
                "max_seq_len": 262144,
                "speculative": "auto DFlash2 / native-mtp (0-ULP SHA-256 greedy exact)",
                "measured_gb10": {
                    "prefill_16k_tps": 6380.8,
                    "prefill_16k_peak_tps": 6469.9,
                    "prefill_128k_tps": 5905.6,
                    "decode_c1_greedy_tps": 83.5,
                    "decode_c16_prose_tps": 298.8,
                    "cold_boot_s": 33.1
                },
                "best_for": [
                    "Zero-flag OOB serving across 8K..262K prompts with auto DFlash2 / K=5 MTP",
                    "RAG pipelines, tool calling (92/92 hardmode), and C=1..16 batched serving"
                ],
                "launch_env": "FLUX_MODE=auto",
                "launch_flag": "--mode auto"
            },
            {
                "id": "interactive",
                "aliases": ["w4096", "v9-w4096"],
                "title": "Interactive Low-TTFT Serving (W=4096)",
                "recipe_file": "flux-v9-w4096-interactive.env",
                "prefill_chunk": 4096,
                "max_seq_len": 135168,
                "speculative": "native-mtp (K=5, 4 drafts)",
                "measured_gb10": {
                    "prefill_8k_tps": 5452.6,
                    "prefill_16k_tps": 5336.8,
                    "prefill_128k_tps": 5071.2,
                    "uma_free_floor_mib": 9871
                },
                "best_for": [
                    "Interactive chat UIs requiring smooth 4K-chunk prefill pacing",
                    "Workloads needing >9.8 GiB free unified memory headroom on 128 GB DGX Spark"
                ],
                "launch_env": "FLUX_MODE=interactive",
                "launch_flag": "--mode interactive"
            },
            {
                "id": "agent",
                "aliases": ["coding", "claude-code", "codex", "antigravity", "ohmypi"],
                "title": agent_title,
                "recipe_file": "flux-v9-w4096-interactive.env",
                "prefill_chunk": 4096,
                "max_seq_len": 135168,
                "speculative": agent_speculative,
                "measured_gb10": {
                    "warm_turn_ttft_ms": 45.0,
                    "decode_code_tps": 63.6,
                    "tool_max_tokens": 16384
                },
                "best_for": [
                    "Claude Code, OpenAI Codex CLI, Antigravity, ohmypi, Cline, OpenCode",
                    agent_loop_desc
                ],
                "launch_env": "FLUX_MODE=agent",
                "launch_flag": "--mode agent"
            },
            {
                "id": "decode",
                "aliases": ["k5", "k5-lossless-decode"],
                "title": "Dedicated Lossless K=5 MTP Decode Velocity",
                "recipe_file": "flux-k5-lossless-decode.env",
                "prefill_chunk": 8192,
                "max_seq_len": 135168,
                "speculative": "native-mtp (K=5, OB=64 MoE gate_up + 3/4-gram hybrid fallback)",
                "measured_gb10": {
                    "decode_rep0_tps": 92.1,
                    "decode_prose8k_tps": 68.8,
                    "decode_chat0_tps": 66.2,
                    "decode_code0_tps": 63.6,
                    "decode_legal32k_tps": 60.0,
                    "decode_doc64k_tps": 55.3,
                    "decode_mix128k_tps": 50.1
                },
                "best_for": [
                    "Long generation outputs, reasoning chains, and code generation",
                    "Sustaining 50–92 tok/s lossless decode from 0 to 128K context"
                ],
                "launch_env": "FLUX_MODE=decode",
                "launch_flag": "--mode decode"
            },
            {
                "id": "long-context",
                "aliases": ["136k", "256k", "512k", "1m"],
                "title": "Long-Context & 262K–1M Document Synthesis",
                "recipe_file": "flux-estonia-136k.env",
                "prefill_chunk": 16384,
                "max_seq_len": 262144,
                "speculative": "native-mtp (K=5, full-context 32-bit RoPE & QSA sparse index)",
                "measured_gb10": {
                    "needle_2k_to_262k_accuracy": "100% (5/5 depths PASS)",
                    "prefill_64k_tps": 5998.4,
                    "prefill_128k_tps": 5898.6
                },
                "best_for": [
                    "136K–262K native needle-in-a-haystack retrieval & repository analysis",
                    "512K–1M extended YaRN context (--mode 256k | --mode 512k | --mode 1m)"
                ],
                "launch_env": "FLUX_MODE=256k",
                "launch_flag": "--mode 256k"
            },
            {
                "id": "exact",
                "aliases": ["reference", "strict"],
                "title": "Bit-Exact Reference Baseline (Non-Speculative)",
                "recipe_file": "flux-exact-reference.env",
                "prefill_chunk": 16384,
                "max_seq_len": 135168,
                "speculative": "disabled (pure target greedy/sampling reference)",
                "measured_gb10": {
                    "determinism": "100% bit-identical raw logits across repeats"
                },
                "best_for": [
                    "0-ULP reference auditing and Plain vs K=5 parity verification"
                ],
                "launch_env": "FLUX_MODE=exact",
                "launch_flag": "--mode exact"
            }
        ]
    });
    Json(body).into_response()
}

/// Interactive dark-mode HTML Mission Control & Self-Guiding Recipe Studio (`GET /`).
pub async fn index_html() -> impl IntoResponse {
    (
        [(CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(MISSION_CONTROL_HTML),
    )
}

const MISSION_CONTROL_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apathy Flux Engine — Blackwell SM121a Mission Control & Recipe Studio</title>
<style>
  :root {
    --bg: #070a10;
    --bg-card: #0d1117;
    --bg-panel: #161b22;
    --bg-inset: #090d13;
    --border: #30363d;
    --border-subtle: #21262d;
    --text: #f0f6fc;
    --muted: #8b949e;
    --green: #10b981;
    --green-bright: #34d399;
    --cyan: #38bdf8;
    --purple: #a855f7;
    --purple-bright: #c084fc;
    --amber: #f59e0b;
    --gold: #fbbf24;
    --nvidia: #76b900;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: radial-gradient(circle at 18% 0%, rgba(56,189,248,0.08), transparent 42%),
                radial-gradient(circle at 85% 8%, rgba(16,185,129,0.07), transparent 38%),
                var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 24px 28px 40px;
    max-width: 1360px;
    margin: 0 auto;
  }
  .mono { font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }

  /* Hero Banner with Official Documentation Hexagonal Tensor Core + Gold Lightning Emblem */
  .hero-banner {
    position: relative;
    background: linear-gradient(135deg, #090d14 0%, #0d1117 55%, #131a24 100%);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 22px;
    overflow: hidden;
    box-shadow: 0 18px 44px rgba(0,0,0,0.45);
  }
  .hero-banner::before {
    content: "";
    position: absolute;
    top: 0; left: 16px; right: 16px; height: 3.5px;
    background: linear-gradient(90deg, #76B900 0%, #10B981 45%, #38BDF8 100%);
    border-radius: 4px;
  }
  .hero-grid-bg {
    position: absolute;
    inset: 0;
    background-image: linear-gradient(to right, rgba(48,54,61,0.22) 1px, transparent 1px),
                      linear-gradient(to bottom, rgba(48,54,61,0.22) 1px, transparent 1px);
    background-size: 24px 24px;
    pointer-events: none;
  }
  .hero-inner {
    position: relative;
    display: grid;
    grid-template-columns: 164px 1fr;
    gap: 24px;
    align-items: center;
  }
  @media (max-width: 860px) {
    .hero-inner { grid-template-columns: 1fr; }
  }
  .emblem-box {
    width: 164px;
    height: 184px;
    background: var(--bg-panel);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .hero-right { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .hero-top-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 12px;
  }
  .title-row { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
  .brand-title {
    font-size: 28px;
    font-weight: 900;
    letter-spacing: 1.2px;
    color: var(--text);
  }
  .chip {
    font-size: 11px;
    font-weight: 700;
    padding: 4px 11px;
    border-radius: 999px;
    letter-spacing: 0.4px;
  }
  .chip-green { background: #122d1f; border: 1.3px solid var(--green); color: var(--green-bright); }
  .chip-cyan  { background: #111d2e; border: 1.3px solid var(--cyan); color: var(--cyan); }
  .chip-gold  { background: #271c0c; border: 1.3px solid var(--amber); color: var(--gold); }
  .subtitle-cyan {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--cyan);
    margin-top: 2px;
  }
  .subtitle-muted {
    font-size: 13px;
    color: var(--muted);
  }
  .pill {
    padding: 7px 15px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12.5px;
    border: 1.5px solid var(--border);
    display: inline-flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
  }
  .pill.ready { background: rgba(16,185,129,0.14); color: var(--green-bright); border-color: var(--green); box-shadow: 0 0 18px rgba(16,185,129,0.18); }
  .pill.loading { background: rgba(56,189,248,0.14); color: var(--cyan); border-color: var(--cyan); }
  .dot-live { width: 8px; height: 8px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }

  .hero-divider { height: 1.5px; background: var(--border-subtle); width: 100%; }

  /* 4 Signature Telemetry Cards matching assets/logo.svg */
  .telemetry-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(180px, 1fr));
    gap: 12px;
  }
  @media (max-width: 1024px) {
    .telemetry-strip { grid-template-columns: repeat(2, 1fr); }
  }
  .t-card {
    background: var(--bg-panel);
    border-radius: 10px;
    padding: 10px 14px;
    border: 1.4px solid var(--border);
  }
  .t-card.c-green  { border-color: var(--green); }
  .t-card.c-cyan   { border-color: var(--cyan); }
  .t-card.c-purple { border-color: var(--purple); }
  .t-card.c-amber  { border-color: var(--amber); }
  .t-head { display: flex; align-items: center; gap: 7px; font-size: 10.5px; font-weight: 700; color: var(--muted); letter-spacing: 0.4px; }
  .t-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .t-val { font-size: 18px; font-weight: 800; margin: 3px 0 1px; }
  .t-sub { font-size: 10.5px; color: var(--muted); }

  /* Section Header */
  .section-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 22px 0 12px;
    flex-wrap: wrap;
    gap: 8px;
  }
  .section-title {
    font-size: 12.5px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: var(--cyan);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-hint { font-size: 12px; color: var(--muted); }

  /* Self-Guiding Recipe Modes Grid */
  .grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(300px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
  }
  @media (max-width: 1060px) { .grid { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 680px)  { .grid { grid-template-columns: 1fr; } }

  .card {
    background: linear-gradient(180deg, #131922 0%, #0d1117 100%);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    transition: all 0.16s ease;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
  }
  .card:hover {
    border-color: var(--cyan);
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
  }
  .card.selected {
    border-color: var(--cyan);
    background: linear-gradient(180deg, #132030 0%, #0d131d 100%);
    box-shadow: 0 0 0 1px var(--cyan) inset, 0 10px 28px rgba(56,189,248,0.12);
  }
  .card-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; }
  .mode-id { font-size: 15px; font-weight: 800; color: var(--text); display: flex; align-items: center; gap: 7px; }
  .badge {
    font-size: 10.5px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    background: #161b22;
    border: 1px solid var(--border);
    color: var(--cyan);
  }
  .badge.active {
    background: rgba(16,185,129,0.18);
    border-color: var(--green);
    color: var(--green-bright);
  }
  .card-title { font-weight: 700; font-size: 13px; color: #e2e8f0; margin-bottom: 6px; }
  .desc { color: var(--muted); font-size: 12px; margin-bottom: 12px; line-height: 1.45; }
  .meta-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: auto; padding-top: 8px; border-top: 1px solid var(--border-subtle); }
  .m-pill {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 6px;
    background: var(--bg-inset);
    border: 1px solid var(--border-subtle);
    color: var(--cyan);
  }
  .m-pill.gold { color: var(--gold); }

  /* Bottom Two-Column Studio Workspace */
  .workspace-grid {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 16px;
  }
  @media (max-width: 980px) { .workspace-grid { grid-template-columns: 1fr; } }

  .panel-box {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .panel-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }
  .panel-title { font-size: 14px; font-weight: 800; color: var(--text); display: flex; align-items: center; gap: 8px; }
  .cmd-box {
    background: var(--bg-inset);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 13px 15px;
    font-size: 12px;
    color: var(--cyan);
    overflow-x: auto;
    white-space: pre-wrap;
    line-height: 1.55;
  }
  .btn {
    background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
    color: #06090f;
    border: none;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 800;
    font-size: 12px;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.92; }
  .chat-row { display: flex; gap: 8px; }
  input[type=text] {
    flex: 1;
    background: var(--bg-inset);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 9px 12px;
    border-radius: 7px;
    font-size: 12.5px;
  }
  input[type=text]:focus { outline: none; border-color: var(--cyan); }
</style>
</head>
<body>

  <!-- Official Documentation Logo & Telemetry Banner -->
  <div class="hero-banner">
    <div class="hero-grid-bg"></div>
    <div class="hero-inner">
      <!-- Official Hexagonal Tensor Core & Gold Flux Vector Lightning Bolt Emblem (from assets/logo.svg) -->
      <div class="emblem-box">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 200" width="154" height="172">
          <defs>
            <linearGradient id="coreGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#F59E0B" stop-opacity="0.24"/>
              <stop offset="100%" stop-color="#38BDF8" stop-opacity="0.10"/>
            </linearGradient>
            <linearGradient id="goldBoltGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#FEF08A"/>
              <stop offset="45%" stop-color="#FBBF24"/>
              <stop offset="100%" stop-color="#D97706"/>
            </linearGradient>
          </defs>
          <polygon points="90,18 146,50 146,112 90,144 34,112 34,50" fill="url(#coreGrad)" stroke="#38BDF8" stroke-width="2.3" stroke-linejoin="round"/>
          <polygon points="90,34 132,58 132,104 90,128 48,104 48,58" fill="none" stroke="#FBBF24" stroke-width="2.0" stroke-dasharray="5 3"/>
          <line x1="34" y1="50" x2="90" y2="81" stroke="#F59E0B" stroke-width="1.8" stroke-dasharray="4 3" opacity="0.85"/>
          <line x1="146" y1="50" x2="90" y2="81" stroke="#F59E0B" stroke-width="1.8" stroke-dasharray="4 3" opacity="0.85"/>
          <line x1="90" y1="81" x2="90" y2="144" stroke="#FBBF24" stroke-width="1.8" stroke-dasharray="4 3" opacity="0.85"/>
          <polygon points="90,30 95,38 85,38" fill="#FBBF24"/>
          <polygon points="135,106 127,107 131,99" fill="#FBBF24"/>
          <polygon points="45,106 49,99 53,107" fill="#FBBF24"/>
          <path d="M 72 52 H 112 L 90 77 H 110 L 70 112 L 81 82 H 65 Z" fill="url(#goldBoltGrad)" stroke="#FEF08A" stroke-width="1.3" stroke-linejoin="round"/>
          <rect x="20" y="156" width="140" height="26" rx="13" fill="#0d1117" stroke="#F59E0B" stroke-width="1.4"/>
          <text x="90" y="173" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="700" fill="#FBBF24" text-anchor="middle">SM121a · GB10</text>
        </svg>
      </div>

      <!-- Right Column: Wordmark, Badges, Status Pill & 4 Telemetry Cards -->
      <div class="hero-right">
        <div class="hero-top-row">
          <div>
            <div class="title-row">
              <span class="brand-title">APATHY FLUX ENGINE</span>
              <span id="ver-chip" class="chip chip-green mono">v1.0.3 · UNIFIED</span>
              <span class="chip chip-cyan mono">0 REGISTER SPILLS</span>
              <a class="chip chip-gold mono" href="https://github.com/TheApathy/flux-engine" target="_blank" rel="noopener noreferrer" style="text-decoration:none" title="AGPL-3.0-only — Corresponding Source">AGPL-3.0 · Source ↗</a>
            </div>
            <div class="subtitle-cyan mono">PURE RUST + SM121a CUDA/PTX INFERENCE RUNTIME · SELF-GUIDING RECIPE STUDIO</div>
            <div class="subtitle-muted">Hardware-specific NVFP4 / MXFP8 / W4A8 kernels at the NVIDIA Grace Blackwell 273 GB/s LPDDR5x ceiling</div>
          </div>
          <div id="status-pill" class="pill loading mono"><span class="dot-live"></span><span>CONNECTING…</span></div>
        </div>

        <div class="hero-divider"></div>

        <div class="telemetry-strip mono">
          <div class="t-card c-green">
            <div class="t-head"><span class="t-dot" style="background:#10B981"></span>16K PREFILL PEAK (W16384)</div>
            <div class="t-val" style="color:#34D399">6,380.8 tok/s</div>
            <div class="t-sub">6,469.9 peak · 128K 5.9K</div>
          </div>
          <div class="t-card c-cyan">
            <div class="t-head"><span class="t-dot" style="background:#38BDF8"></span>0-ULP SPECULATIVE DECODE</div>
            <div class="t-val" style="color:#38BDF8">83.5 tok/s</div>
            <div class="t-sub">0-ULP SHA-256 exact · auto DFlash2 / MTP</div>
          </div>
          <div class="t-card c-purple">
            <div class="t-head"><span class="t-dot" style="background:#A855F7"></span>BATCHED CONCURRENCY (C=16)</div>
            <div class="t-val" style="color:#C084FC">298.8 tok/s</div>
            <div class="t-sub">Prose T=1.0 · 413.2 HashMap · C=1..16</div>
          </div>
          <div class="t-card c-amber">
            <div class="t-head"><span class="t-dot" style="background:#F59E0B"></span>AOT COLD-START BOOT</div>
            <div class="t-val" style="color:#FBBF24">33.1s Ready</div>
            <div class="t-sub">AOT CUBIN + GPU transpose · 262K 5/5</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Self-Guiding Recipe Mode Selector -->
  <div class="section-bar">
    <div class="section-title mono">⚡ 1. Self-Guiding Recipe Modes — Select Any Workload Profile</div>
    <div class="section-hint mono">API Catalog: <span style="color:var(--cyan)">GET /flux/modes</span> · CLI Advisor: <span style="color:var(--gold)">flux wizard</span></div>
  </div>
  <div id="modes-grid" class="grid"></div>

  <!-- Bottom Two-Column Studio -->
  <div class="workspace-grid">
    <div class="panel-box">
      <div class="panel-head">
        <div class="panel-title"><span>⚡ Selected Mode Launch & Tuning Recipe</span></div>
        <button id="copy-btn" class="btn mono" onclick="copyCmd()">Copy Recipe</button>
      </div>
      <div id="selected-cmd" class="cmd-box mono">Loading recipe catalog…</div>
    </div>

    <div class="panel-box">
      <div class="panel-head">
        <div class="panel-title"><span>💬 Live Interactive Probe (<span class="mono" style="color:var(--cyan)">/v1/chat/completions</span>)</span></div>
        <button class="btn mono" onclick="sendChat()">Run Probe</button>
      </div>
      <div class="chat-row">
        <input id="prompt" type="text" value="Briefly introduce Apathy Flux Engine and its self-guiding recipe modes." />
      </div>
      <div id="chat-out" class="cmd-box mono" style="color:var(--text);min-height:118px;">Ready when server status is SERVING.</div>
    </div>
  </div>

<script>
let modesData = null;
let selectedId = 'max-prefill';
function setStatusPill(pill, className, label) {
  pill.className = className;
  const dot = document.createElement('span');
  dot.className = 'dot-live';
  const text = document.createElement('span');
  text.textContent = label;
  pill.replaceChildren(dot, text);
}
async function refresh() {
  try {
    const r = await fetch('/flux/modes');
    modesData = await r.json();
    if (modesData.version) {
      document.getElementById('ver-chip').textContent = `v${modesData.version} · UNIFIED`;
    }
    const pill = document.getElementById('status-pill');
    if (modesData.status === 'ready') {
      const tps = modesData.scheduler && modesData.scheduler.delivered_tps > 0
        ? ` · ${modesData.scheduler.delivered_tps.toFixed(1)} tok/s`
        : '';
      setStatusPill(
        pill,
        'pill ready mono',
        `SERVING · ${modesData.active_model} (${modesData.active_mode})${tps}`
      );
    } else {
      setStatusPill(
        pill,
        'pill loading mono',
        `LOADING WEIGHTS · ${modesData.active_model} (${modesData.active_mode})`
      );
    }
    renderModes();
  } catch (e) {}
}
function renderModes() {
  if (!modesData) return;
  const grid = document.getElementById('modes-grid');
  const cards = modesData.modes.map(m => {
    const isAct = m.id === modesData.active_mode || (modesData.active_mode === 'auto' && m.id === 'max-prefill');
    const isSel = m.id === selectedId;
    const card = document.createElement('div');
    card.className = `card ${isSel ? 'selected' : ''}`;
    card.addEventListener('click', () => selectMode(m.id));

    const upper = document.createElement('div');
    const cardTop = document.createElement('div');
    cardTop.className = 'card-top';

    const modeId = document.createElement('span');
    modeId.className = 'mode-id mono';
    const bolt = document.createElement('span');
    bolt.style.color = 'var(--gold)';
    bolt.textContent = '⚡';
    modeId.append(bolt, document.createTextNode(String(m.id ?? '')));

    const badge = document.createElement('span');
    badge.className = `badge mono ${isAct ? 'active' : ''}`;
    badge.textContent = isAct ? '★ ACTIVE MODE' : String(m.recipe_file ?? '');
    cardTop.append(modeId, badge);

    const title = document.createElement('div');
    title.className = 'card-title';
    title.textContent = String(m.title ?? '');

    const desc = document.createElement('div');
    desc.className = 'desc';
    desc.textContent = (m.best_for || []).join(' • ');
    upper.append(cardTop, title, desc);

    const meta = document.createElement('div');
    meta.className = 'meta-pills mono';
    const chunkPill = document.createElement('span');
    chunkPill.className = 'm-pill';
    chunkPill.textContent = `Chunk W=${m.prefill_chunk}`;
    const seqPill = document.createElement('span');
    seqPill.className = 'm-pill';
    seqPill.textContent = `MaxSeq ${Number(m.max_seq_len).toLocaleString()}`;
    const specPill = document.createElement('span');
    specPill.className = 'm-pill gold';
    specPill.textContent = String(m.speculative ?? '').split('(')[0].trim();
    meta.append(chunkPill, seqPill, specPill);

    card.append(upper, meta);
    return card;
  });
  grid.replaceChildren(...cards);
  const sel = modesData.modes.find(m => m.id === selectedId) || modesData.modes[0];
  const imgTag = `v${modesData.version || '1.0.3'}`;
  if (sel) {
    document.getElementById('selected-cmd').textContent =
`# 1. Zero-Flag Container Launch (${sel.id} mode)
docker run --rm -it --gpus all --ipc=host --network=host \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  -v /path/to/models:/models:ro \\
  -e ${sel.launch_env} \\
  ghcr.io/theapathy/flux-engine:${imgTag}

# 2. Export & Customize .env Recipe (${sel.recipe_file})
docker run --rm ghcr.io/theapathy/flux-engine:${imgTag} export-recipe ${sel.id} > ${sel.recipe_file}`;
  }
}
function selectMode(id) { selectedId = id; renderModes(); }
function copyCmd() {
  navigator.clipboard.writeText(document.getElementById('selected-cmd').textContent);
  const b = document.getElementById('copy-btn');
  b.textContent = '✓ Copied!';
  setTimeout(() => { b.textContent = 'Copy Recipe'; }, 1400);
}
async function sendChat() {
  const out = document.getElementById('chat-out');
  out.textContent = 'Generating response…';
  const t0 = performance.now();
  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: modesData ? modesData.active_model : 'default',
        messages: [{role: 'user', content: document.getElementById('prompt').value}],
        max_tokens: 160,
        temperature: 0,
        chat_template_kwargs: {enable_thinking: false}
      })
    });
    const d = await r.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    const txt = d.choices && d.choices[0] ? d.choices[0].message.content : JSON.stringify(d);
    const u = d.usage ? `\n\n[prompt=${d.usage.prompt_tokens} tok · completion=${d.usage.completion_tokens} tok · wall=${dt}s]` : '';
    out.textContent = txt + u;
  } catch (e) { out.textContent = 'Error: ' + e; }
}
refresh();
setInterval(refresh, 2500);
</script>
</body>
</html>
"##;

/// Scheduler facts as resolved by serve_load, after every clamp and proposer check.
pub(crate) struct ResolvedServing {
    /// Rows the scheduler actually batches per step (EP forces 1).
    pub max_batch_size: usize,
    /// `--max-num-seqs`: the ingress channel bound, not execution capacity.
    pub ingress_queue: usize,
    pub max_seq_len: usize,
    /// `(use_internal_spec, use_mtp)` exactly as the scheduler resolves them
    /// (`scheduler::speculation_routes`): kill switch, model-internal drafter (DSpark),
    /// and an installed external proposer.
    pub routes: (bool, bool),
    pub dflash: bool,
    pub self_speculative: bool,
    pub ngram: bool,
    /// The scheduler's draft count (DFlash: the drafter-resolved gamma).
    pub num_drafts: usize,
    pub world_size: usize,
}

fn resolved_speculation(r: &ResolvedServing, env: &dyn Fn(&str) -> Option<String>) -> &'static str {
    let on =
        |flux_k: &str, atlas_k: &str| env(flux_k).or_else(|| env(atlas_k)).as_deref() == Some("1");
    let (internal, mtp) = r.routes;
    if internal {
        "internal"
    } else if mtp && r.dflash {
        "dflash"
    } else if mtp {
        if on(
            "FLUX_QWEN4_BATCHED_ADAPTIVE_K",
            "ATLAS_QWEN4_BATCHED_ADAPTIVE_K",
        ) {
            "adaptive-k"
        } else if on("FLUX_QWEN4_BATCHED_K5", "ATLAS_QWEN4_BATCHED_K5") {
            "batched-k5"
        } else {
            "native-mtp"
        }
    } else if r.self_speculative {
        "self-speculative"
    } else if r.ngram {
        "ngram"
    } else {
        "target-only"
    }
}

/// The `/flux/modes` `serving` block.
///
/// `concurrency` is the scheduler's resolved batch width: how many sequences it will hold and
/// step at once, and the only figure a router may use as a per-backend cap. It is scheduler
/// CAPACITY, not a promise of native batched rows: some routes still step a frame row by row
/// (a Qwen4 frame serializes once any row's next position reaches the 2048-token QSA boundary,
/// unless `ATLAS_QWEN4_NATIVE_BATCH_QSA=1`; profiling; `ATLAS_QWEN4_NATIVE_BATCH=0`; per-request
/// self/ngram fallback). `ingress_queue` sizes the request channel and is not execution capacity.
pub(crate) fn resolved_serving_block(r: ResolvedServing) -> serde_json::Value {
    resolved_serving_block_env(&r, &|k| std::env::var(k).ok())
}

fn resolved_serving_block_env(
    r: &ResolvedServing,
    env: &dyn Fn(&str) -> Option<String>,
) -> serde_json::Value {
    let speculation = resolved_speculation(r, env);
    serde_json::json!({
        "concurrency": r.max_batch_size,
        "ingress_queue": r.ingress_queue,
        "max_seq_len": r.max_seq_len,
        "speculation": speculation,
        // internal drafters (DSpark) choose a variable width inside Model::spec_verify, so no
        // scheduler-side draft count applies; report null rather than an unrelated CLI value
        "num_drafts": match speculation {
            "target-only" => serde_json::json!(0),
            "internal" => serde_json::Value::Null,
            _ => serde_json::json!(r.num_drafts),
        },
        "world_size": r.world_size,
        "concurrency_kind": "scheduler-capacity",
        "source": "resolved-at-load",
    })
}

#[cfg(test)]
mod serving_tests {
    use super::*;

    fn base() -> ResolvedServing {
        ResolvedServing {
            max_batch_size: 32,
            ingress_queue: 32,
            max_seq_len: 65_536,
            routes: (false, false),
            dflash: false,
            self_speculative: false,
            ngram: false,
            num_drafts: 4,
            world_size: 1,
        }
    }

    fn none(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn engine_version_prefers_release_env_over_crate_version() {
        assert_eq!(resolved_engine_version_from(Some("1.0.0")), "1.0.0");
        assert_eq!(resolved_engine_version_from(Some(" 1.0.0 ")), "1.0.0");
        assert_eq!(
            resolved_engine_version_from(Some("   ")),
            crate::cli::ATLAS_VERSION
        );
        assert_eq!(
            resolved_engine_version_from(None),
            crate::cli::ATLAS_VERSION
        );
    }

    #[test]
    fn target_only_reports_the_resolved_width() {
        let b = resolved_serving_block_env(&base(), &none);
        assert_eq!(
            (
                b["speculation"].as_str(),
                b["concurrency"].as_u64(),
                b["num_drafts"].as_u64()
            ),
            (Some("target-only"), Some(32), Some(0))
        );
        assert_eq!(b["source"], "resolved-at-load");
    }

    /// EP forces batch 1 in serve_load; the block must report 1, not the CLI arg.
    #[test]
    fn ep_forced_width_is_what_is_published() {
        let b = resolved_serving_block_env(
            &ResolvedServing {
                max_batch_size: 1,
                ingress_queue: 32,
                world_size: 2,
                ..base()
            },
            &none,
        );
        assert_eq!(
            (b["concurrency"].as_u64(), b["ingress_queue"].as_u64()),
            (Some(1), Some(32))
        );
    }

    /// A requested proposer that did not install does not count as speculation.
    #[test]
    fn speculation_requires_an_installed_proposer() {
        let b = resolved_serving_block_env(
            &ResolvedServing {
                dflash: true,
                routes: (false, false),
                ..base()
            },
            &none,
        );
        assert_eq!(b["speculation"], "target-only");
        let d = resolved_serving_block_env(
            &ResolvedServing {
                dflash: true,
                routes: (false, true),
                num_drafts: 7,
                ..base()
            },
            &none,
        );
        assert_eq!(
            (d["speculation"].as_str(), d["num_drafts"].as_u64()),
            (Some("dflash"), Some(7))
        );
    }

    /// A DSpark model speculates with its own drafter (has_internal_spec).
    #[test]
    fn internal_drafter_is_reported() {
        let ds = ResolvedServing {
            routes: crate::scheduler::speculation_routes(false, true, false, false),
            ..base()
        };
        let b = resolved_serving_block_env(&ds, &none);
        assert_eq!(b["speculation"], "internal");
        assert_eq!(b["concurrency_kind"], "scheduler-capacity");
        assert!(
            b["num_drafts"].is_null(),
            "internal width is variable, not the CLI num_drafts"
        );
    }

    #[test]
    fn qwen4_modes_and_kill_switch() {
        let spec = ResolvedServing {
            routes: (false, true),
            ..base()
        };
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| (*v).to_string())
            }
        };
        assert_eq!(
            resolved_serving_block_env(&spec, &none)["speculation"],
            "native-mtp"
        );
        assert_eq!(
            resolved_serving_block_env(&spec, &env(&[("ATLAS_QWEN4_BATCHED_K5", "1")]))["speculation"],
            "batched-k5"
        );
        assert_eq!(
            resolved_serving_block_env(&spec, &env(&[("FLUX_QWEN4_BATCHED_K5", "1")]))["speculation"],
            "batched-k5"
        );
        assert_eq!(
            resolved_serving_block_env(
                &spec,
                &env(&[
                    ("ATLAS_QWEN4_BATCHED_K5", "1"),
                    ("ATLAS_QWEN4_BATCHED_ADAPTIVE_K", "1")
                ])
            )["speculation"],
            "adaptive-k"
        );
        assert_eq!(
            resolved_serving_block_env(&spec, &env(&[("FLUX_QWEN4_BATCHED_ADAPTIVE_K", "1")]))["speculation"],
            "adaptive-k"
        );
        // the kill switch arrives already applied in the scheduler's routes
        let killed = ResolvedServing {
            routes: crate::scheduler::speculation_routes(true, true, true, true),
            ..base()
        };
        assert_eq!(
            resolved_serving_block_env(&killed, &none)["speculation"],
            "target-only"
        );
        let ss = ResolvedServing {
            self_speculative: true,
            ..base()
        };
        assert_eq!(
            resolved_serving_block_env(&ss, &none)["speculation"],
            "self-speculative"
        );
    }
}
