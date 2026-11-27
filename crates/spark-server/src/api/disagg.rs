// SPDX-License-Identifier: AGPL-3.0-only

//! Disaggregated Prefill / Decode (`1P + 1D` / `1P + 2D`) HTTP handoff and
//! zero-overhead scheduler state registration.

use std::collections::HashMap;
use std::io::Read as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use atlas_core::config::DisaggRole;
use axum::body::Bytes;
use axum::extract::Query;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use spark_model::traits::Model;
use tokio::sync::oneshot;

use super::compact::openai_error_response;
use super::inference_impl::try_admit_scheduler;
use super::inference_types::{InferenceRequest, InferenceResponse};
use crate::main_modules::model_host::CurrentModel;
use spark_model::traits::SequenceState;

fn err_resp(status: StatusCode, msg: impl Into<String>) -> Response {
    openai_error_response(status, msg.into())
}

const WIRE_MAGIC: [u8; 4] = *b"FDP1";
const WIRE_VERSION: u32 = 1;
const DISAGG_SESSION_PREFIX: u64 = 0xD15A_6600_0000_0000;
const MAX_WIRE_PROMPT_TOKENS: usize = 2_097_152;
const MAX_WIRE_STATE_BYTES: usize = 32 * 1024 * 1024 * 1024;
const DISAGG_PEER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DISAGG_PEER_DEFAULT_GLOBAL_TIMEOUT: Duration = Duration::from_secs(120);
/// Upper bound on an HTTP response body fetched from a disaggregated prefill peer.
const MAX_DISAGG_RESPONSE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct DisaggRuntimeConfig {
    pub role: DisaggRole,
    pub peers: Vec<String>,
    pub pp_rank: usize,
    pub pp_world_size: usize,
}

impl Default for DisaggRuntimeConfig {
    fn default() -> Self {
        Self {
            role: DisaggRole::Unified,
            peers: Vec::new(),
            pp_rank: 0,
            pp_world_size: 1,
        }
    }
}

static RUNTIME_CFG: OnceLock<RwLock<DisaggRuntimeConfig>> = OnceLock::new();
static PEER_RR: AtomicUsize = AtomicUsize::new(0);
static SESSION_SEQ: AtomicU64 = AtomicU64::new(1);
static ACTIVE_HANDOFFS: AtomicUsize = AtomicUsize::new(0);

type ExportSender = oneshot::Sender<Result<DisaggStatePacket>>;

#[derive(Default)]
struct HandoffRegistry {
    exports: HashMap<u64, ExportSender>,
    imports: HashMap<u64, DisaggStatePacket>,
}

static REGISTRY: OnceLock<Mutex<HandoffRegistry>> = OnceLock::new();

fn runtime_lock() -> &'static RwLock<DisaggRuntimeConfig> {
    RUNTIME_CFG.get_or_init(|| RwLock::new(DisaggRuntimeConfig::default()))
}

fn registry_lock() -> &'static Mutex<HandoffRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(HandoffRegistry::default()))
}

pub fn parse_peer_urls(raw: Option<&str>) -> Vec<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
        .collect()
}

pub fn publish_runtime_config(
    role: DisaggRole,
    disagg_peer: Option<&str>,
    pp_rank: usize,
    pp_world_size: usize,
) {
    let mut cfg = runtime_lock().write();
    *cfg = DisaggRuntimeConfig {
        role,
        peers: parse_peer_urls(disagg_peer),
        pp_rank,
        pp_world_size,
    };
}

pub fn current_runtime_config() -> DisaggRuntimeConfig {
    runtime_lock().read().clone()
}

fn select_peer_from_allowlist(
    peers: &[String],
    explicit: Option<&str>,
) -> std::result::Result<String, (StatusCode, String)> {
    if let Some(raw) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        let normalized = raw.trim_end_matches('/');
        if let Some(matched) = peers
            .iter()
            .find(|p| p.trim().trim_end_matches('/') == normalized)
        {
            return Ok(matched.trim().trim_end_matches('/').to_string());
        }
        return Err((
            StatusCode::FORBIDDEN,
            "prefill_peer is not in the configured --disagg-peer allowlist".to_string(),
        ));
    }
    if peers.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "no prefill peer configured: pass --disagg-peer <http://prefill-host:port>".to_string(),
        ));
    }
    let idx = PEER_RR.fetch_add(1, Ordering::Relaxed) % peers.len();
    Ok(peers[idx].clone())
}

fn resolve_prefill_peer(
    explicit: Option<&str>,
) -> std::result::Result<String, (StatusCode, String)> {
    let cfg = runtime_lock().read();
    select_peer_from_allowlist(&cfg.peers, explicit)
}

pub fn select_next_peer(explicit: Option<&str>) -> Option<String> {
    resolve_prefill_peer(explicit).ok()
}

/// Binary state packet transferred from a `--role prefill` producer (`1P`) to a
/// `--role decode` consumer (`1D` / `2D`).
#[derive(Debug, Clone, PartialEq)]
pub struct DisaggStatePacket {
    pub prompt_tokens: Vec<u32>,
    pub first_token: u32,
    pub num_blocks: usize,
    pub seq_len: usize,
    pub cached_prompt_tokens: u32,
    pub ttft_ms: f64,
    pub state_bytes: Vec<u8>,
}

impl DisaggStatePacket {
    pub fn encode_binary(&self) -> Vec<u8> {
        let header_bytes = 4 + 4 + 4 + 8 + 8 + 4 + 8 + 8 + 8;
        let tokens_bytes = self.prompt_tokens.len().saturating_mul(4);
        let mut out = Vec::with_capacity(header_bytes + tokens_bytes + self.state_bytes.len());
        out.extend_from_slice(&WIRE_MAGIC);
        out.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.first_token.to_le_bytes());
        out.extend_from_slice(&(self.num_blocks as u64).to_le_bytes());
        out.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        out.extend_from_slice(&self.cached_prompt_tokens.to_le_bytes());
        out.extend_from_slice(&self.ttft_ms.to_bits().to_le_bytes());
        out.extend_from_slice(&(self.prompt_tokens.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.state_bytes.len() as u64).to_le_bytes());
        for &tok in &self.prompt_tokens {
            out.extend_from_slice(&tok.to_le_bytes());
        }
        out.extend_from_slice(&self.state_bytes);
        out
    }

    pub fn decode_binary(raw: &[u8]) -> Result<Self> {
        const HEADER_LEN: usize = 56;
        anyhow::ensure!(
            raw.len() >= HEADER_LEN,
            "disagg packet truncated: {} < {HEADER_LEN} bytes",
            raw.len()
        );
        anyhow::ensure!(
            raw[0..4] == WIRE_MAGIC,
            "disagg packet magic mismatch: expected FDP1"
        );
        let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        anyhow::ensure!(
            version == WIRE_VERSION,
            "unsupported disagg packet version {version} (expected {WIRE_VERSION})"
        );
        let first_token = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let num_blocks = usize::try_from(u64::from_le_bytes(raw[12..20].try_into().unwrap()))
            .context("disagg num_blocks overflow")?;
        let seq_len = usize::try_from(u64::from_le_bytes(raw[20..28].try_into().unwrap()))
            .context("disagg seq_len overflow")?;
        let cached_prompt_tokens = u32::from_le_bytes(raw[28..32].try_into().unwrap());
        let ttft_ms = f64::from_bits(u64::from_le_bytes(raw[32..40].try_into().unwrap()));
        let prompt_len = usize::try_from(u64::from_le_bytes(raw[40..48].try_into().unwrap()))
            .context("disagg prompt_len overflow")?;
        let state_len = usize::try_from(u64::from_le_bytes(raw[48..56].try_into().unwrap()))
            .context("disagg state_len overflow")?;
        anyhow::ensure!(
            prompt_len > 0 && prompt_len <= MAX_WIRE_PROMPT_TOKENS,
            "disagg prompt_len {prompt_len} out of bounds (1..={MAX_WIRE_PROMPT_TOKENS})"
        );
        anyhow::ensure!(
            state_len <= MAX_WIRE_STATE_BYTES,
            "disagg state_len {state_len} exceeds ceiling"
        );
        let tok_bytes = prompt_len
            .checked_mul(4)
            .context("disagg prompt token byte overflow")?;
        let expected_total = HEADER_LEN
            .checked_add(tok_bytes)
            .and_then(|n| n.checked_add(state_len))
            .context("disagg packet size overflow")?;
        anyhow::ensure!(
            raw.len() == expected_total,
            "disagg packet size mismatch: got {}, expected {expected_total}",
            raw.len()
        );
        let mut prompt_tokens = Vec::with_capacity(prompt_len);
        let tok_slice = &raw[HEADER_LEN..HEADER_LEN + tok_bytes];
        for chunk in tok_slice.chunks_exact(4) {
            prompt_tokens.push(u32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let state_bytes = raw[HEADER_LEN + tok_bytes..].to_vec();
        Ok(Self {
            prompt_tokens,
            first_token,
            num_blocks,
            seq_len,
            cached_prompt_tokens,
            ttft_ms,
            state_bytes,
        })
    }
}

fn next_disagg_session_hash() -> u64 {
    let seq = SESSION_SEQ.fetch_add(1, Ordering::Relaxed) & 0x0000_FFFF_FFFF_FFFF;
    DISAGG_SESSION_PREFIX | seq
}

pub struct DisaggPrefillGuard {
    session_hash: u64,
}

impl DisaggPrefillGuard {
    pub fn register() -> (Self, oneshot::Receiver<Result<DisaggStatePacket>>) {
        let session_hash = next_disagg_session_hash();
        let (tx, rx) = oneshot::channel();
        ACTIVE_HANDOFFS.fetch_add(1, Ordering::AcqRel);
        registry_lock().lock().exports.insert(session_hash, tx);
        (Self { session_hash }, rx)
    }

    pub const fn session_hash(&self) -> u64 {
        self.session_hash
    }
}

impl Drop for DisaggPrefillGuard {
    fn drop(&mut self) {
        if registry_lock()
            .lock()
            .exports
            .remove(&self.session_hash)
            .is_some()
        {
            ACTIVE_HANDOFFS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

pub struct DisaggDecodeGuard {
    session_hash: u64,
}

impl DisaggDecodeGuard {
    pub fn register(packet: DisaggStatePacket) -> Self {
        let session_hash = next_disagg_session_hash();
        ACTIVE_HANDOFFS.fetch_add(1, Ordering::AcqRel);
        registry_lock().lock().imports.insert(session_hash, packet);
        Self { session_hash }
    }

    pub const fn session_hash(&self) -> u64 {
        self.session_hash
    }
}

impl Drop for DisaggDecodeGuard {
    fn drop(&mut self) {
        if registry_lock()
            .lock()
            .imports
            .remove(&self.session_hash)
            .is_some()
        {
            ACTIVE_HANDOFFS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

#[inline]
pub fn take_decode_import(session_hash: u64) -> Option<DisaggStatePacket> {
    if ACTIVE_HANDOFFS.load(Ordering::Acquire) == 0 {
        return None;
    }
    let pkt = registry_lock().lock().imports.remove(&session_hash)?;
    ACTIVE_HANDOFFS.fetch_sub(1, Ordering::AcqRel);
    Some(pkt)
}

#[inline]
pub fn has_pending_prefill_export(session_hash: u64) -> bool {
    if ACTIVE_HANDOFFS.load(Ordering::Acquire) == 0 {
        return false;
    }
    registry_lock().lock().exports.contains_key(&session_hash)
}

#[inline]
pub fn maybe_fulfill_prefill_export(
    model: &dyn Model,
    session_hash: u64,
    seq: &SequenceState,
    first_token: u32,
    cached_prompt_tokens: u32,
    request_start: std::time::Instant,
) {
    if ACTIVE_HANDOFFS.load(Ordering::Acquire) == 0 {
        return;
    }
    let Some(export_tx) = registry_lock().lock().exports.remove(&session_hash) else {
        return;
    };
    ACTIVE_HANDOFFS.fetch_sub(1, Ordering::AcqRel);
    let mut state_bytes = Vec::new();
    let result = model
        .save_sequence_state(seq, &mut state_bytes)
        .map(|()| DisaggStatePacket {
            prompt_tokens: seq.tokens.clone(),
            first_token,
            num_blocks: seq.block_table.len(),
            seq_len: seq.seq_len,
            cached_prompt_tokens,
            ttft_ms: request_start.elapsed().as_secs_f64() * 1000.0,
            state_bytes,
        });
    let _ = export_tx.send(result);
}

#[derive(Debug, Serialize)]
pub struct DisaggStatusResponse {
    pub role: &'static str,
    pub peers: Vec<String>,
    pub pp_rank: usize,
    pub pp_world_size: usize,
    pub model_name: String,
    pub max_seq_len: usize,
    pub ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisaggPrefillRequest {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub prompt_tokens: Option<Vec<u32>>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
pub struct DisaggDecodeQuery {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DisaggGenerateRequest {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub prompt_tokens: Option<Vec<u32>>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub prefill_peer: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DisaggGenerateResponse {
    pub text: String,
    pub output_tokens: Vec<u32>,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_prompt_tokens: u32,
    pub prefill_ttft_ms: f64,
    pub decode_time_ms: f64,
    pub prefill_peer: Option<String>,
}

fn request_deadline(request_timeout_secs: u32) -> Option<Instant> {
    (request_timeout_secs > 0)
        .then(|| Instant::now() + Duration::from_secs(u64::from(request_timeout_secs)))
}

fn validate_sampling_params(temperature: f32, top_p: f32) -> std::result::Result<(), String> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(format!(
            "invalid temperature {temperature}: must be finite and >= 0.0"
        ));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(format!(
            "invalid top_p {top_p}: must be finite and in (0.0, 1.0]"
        ));
    }
    Ok(())
}

fn validate_packet_for_decode(
    packet: &DisaggStatePacket,
    max_seq_len: usize,
) -> std::result::Result<(), String> {
    if packet.prompt_tokens.is_empty() {
        return Err("disagg packet prompt_tokens must not be empty".to_string());
    }
    if packet.prompt_tokens.len() >= max_seq_len {
        return Err(format!(
            "disagg packet prompt_tokens length ({}) reaches or exceeds max_seq_len ({max_seq_len})",
            packet.prompt_tokens.len(),
        ));
    }
    Ok(())
}

#[derive(Debug)]
enum PeerBodyReadError {
    TooLarge { max_bytes: usize },
    Io(std::io::Error),
}

fn read_bounded_body(
    reader: impl std::io::Read,
    max_bytes: usize,
) -> std::result::Result<Vec<u8>, PeerBodyReadError> {
    let limit = u64::try_from(max_bytes)
        .unwrap_or(u64::MAX - 1)
        .saturating_add(1);
    let mut raw = Vec::new();
    reader
        .take(limit)
        .read_to_end(&mut raw)
        .map_err(PeerBodyReadError::Io)?;
    if raw.len() > max_bytes {
        return Err(PeerBodyReadError::TooLarge { max_bytes });
    }
    Ok(raw)
}

#[allow(clippy::too_many_arguments)]
fn build_blocking_request(
    prompt_tokens: Vec<u32>,
    session_hash: u64,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    top_n_sigma: f32,
    seed: Option<u64>,
    timeout_at: Option<Instant>,
    response_tx: oneshot::Sender<Result<InferenceResponse>>,
) -> InferenceRequest {
    InferenceRequest::Blocking {
        prompt_tokens,
        session_hash,
        image_pixels: Vec::new(),
        max_tokens,
        min_tokens: 0,
        ignore_eos: false,
        temperature: super::canonical_greedy_temperature(temperature, top_k),
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias: Vec::new(),
        stop_tokens: Vec::new(),
        enable_thinking: false,
        thinking_budget: None,
        require_tool_call: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed,
        top_logprobs: None,
        timeout_at,
        decode_timeout_at: None,
        intervention_ledger: None,
        think_nudge: None,
        response_tx,
    }
}

fn resolve_prompt_tokens(
    state: &crate::AppState,
    prompt: Option<&str>,
    prompt_tokens: Option<Vec<u32>>,
) -> std::result::Result<Vec<u32>, String> {
    let tokens = match (prompt_tokens, prompt) {
        (Some(toks), _) => toks,
        (None, Some(text)) => state
            .tokenizer
            .encode(text)
            .map_err(|e| format!("tokenization failed: {e}"))?,
        (None, None) => {
            return Err("either `prompt` or `prompt_tokens` must be provided".to_string());
        }
    };
    if tokens.is_empty() {
        return Err("prompt resolved to 0 tokens".to_string());
    }
    if tokens.len() >= state.max_seq_len {
        return Err(format!(
            "prompt length ({}) reaches or exceeds max_seq_len ({})",
            tokens.len(),
            state.max_seq_len
        ));
    }
    Ok(tokens)
}

/// `GET /v1/disagg/status` — inspect node role (`unified` | `prefill` | `decode`),
/// configured prefill peers, and pipeline-parallel stage coordinates.
pub async fn disagg_status(CurrentModel(state): CurrentModel) -> impl IntoResponse {
    let cfg = current_runtime_config();
    Json(DisaggStatusResponse {
        role: cfg.role.as_str(),
        peers: cfg.peers,
        pp_rank: cfg.pp_rank,
        pp_world_size: cfg.pp_world_size,
        model_name: state.model_name.clone(),
        max_seq_len: state.max_seq_len,
        ready: state.model_ready.load(Ordering::Relaxed),
    })
}

/// `POST /v1/disagg/prefill` — execute prefill on a `1P` producer node and
/// return the serialized KV + SSM + Qwen4 PLE/QSA state packet (`FDP1`).
pub async fn disagg_prefill(
    CurrentModel(state): CurrentModel,
    Json(req): Json<DisaggPrefillRequest>,
) -> Response {
    let cfg = current_runtime_config();
    if cfg.role == DisaggRole::Decode {
        return err_resp(
            StatusCode::BAD_REQUEST,
            "node is configured with --role decode and cannot serve /v1/disagg/prefill",
        );
    }
    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let top_p = req.top_p.unwrap_or(state.default_top_p);
    if let Err(msg) = validate_sampling_params(temperature, top_p) {
        return err_resp(StatusCode::BAD_REQUEST, msg);
    }
    let prompt_tokens =
        match resolve_prompt_tokens(&state, req.prompt.as_deref(), req.prompt_tokens) {
            Ok(t) => t,
            Err(msg) => return err_resp(StatusCode::BAD_REQUEST, msg),
        };
    let (guard, export_rx) = DisaggPrefillGuard::register();
    let (resp_tx, resp_rx) = oneshot::channel();
    let infer_req = build_blocking_request(
        prompt_tokens,
        guard.session_hash(),
        1,
        temperature,
        req.top_k.unwrap_or(state.default_top_k),
        top_p,
        state.default_min_p,
        state.default_top_n_sigma,
        req.seed,
        request_deadline(state.request_timeout),
        resp_tx,
    );
    if let Err((status, msg)) = try_admit_scheduler(&state.request_tx, infer_req) {
        return err_resp(status, msg);
    }
    if let Ok(Err(e)) = resp_rx.await {
        tracing::error!("disaggregated prefill failed: {e:#}");
        return err_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            "disaggregated prefill failed",
        );
    }
    match export_rx.await {
        Ok(Ok(pkt)) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            pkt.encode_binary(),
        )
            .into_response(),
        Ok(Err(e)) => {
            tracing::error!("disaggregated state serialization failed: {e:#}");
            err_resp(
                StatusCode::INTERNAL_SERVER_ERROR,
                "disaggregated state serialization failed",
            )
        }
        Err(_) => err_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            "disaggregated prefill export channel closed without state packet",
        ),
    }
}

async fn run_local_imported_decode(
    state: &crate::AppState,
    packet: DisaggStatePacket,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: Option<u64>,
    prefill_peer: Option<String>,
) -> Response {
    let prompt_len = packet.prompt_tokens.len();
    let prefill_ttft_ms = packet.ttft_ms;
    let prompt_tokens = packet.prompt_tokens.clone();
    let max_tokens = max_tokens
        .max(1)
        .min(state.max_seq_len.saturating_sub(prompt_len).max(1));
    let guard = DisaggDecodeGuard::register(packet);
    let (resp_tx, resp_rx) = oneshot::channel();
    let infer_req = build_blocking_request(
        prompt_tokens,
        guard.session_hash(),
        max_tokens,
        temperature,
        top_k,
        top_p,
        state.default_min_p,
        state.default_top_n_sigma,
        seed,
        request_deadline(state.request_timeout),
        resp_tx,
    );
    if let Err((status, msg)) = try_admit_scheduler(&state.request_tx, infer_req) {
        return err_resp(status, msg);
    }
    match resp_rx.await {
        Ok(Ok(done)) => {
            let text = state
                .tokenizer
                .decode(&done.output_tokens)
                .unwrap_or_default();
            let completion_tokens = done.output_tokens.len();
            Json(DisaggGenerateResponse {
                text,
                output_tokens: done.output_tokens,
                finish_reason: done.finish_reason,
                prompt_tokens: prompt_len,
                completion_tokens,
                cached_prompt_tokens: done.cached_prompt_tokens,
                prefill_ttft_ms,
                decode_time_ms: done.decode_time_ms,
                prefill_peer,
            })
            .into_response()
        }
        Ok(Err(e)) => {
            tracing::error!("disaggregated decode failed: {e:#}");
            err_resp(
                StatusCode::INTERNAL_SERVER_ERROR,
                "disaggregated decode failed",
            )
        }
        Err(_) => err_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            "disaggregated decode cancelled before completion",
        ),
    }
}

/// `POST /v1/disagg/decode` — import a binary `FDP1` prefill state packet and
/// decode locally without re-running prefill.
pub async fn disagg_decode(
    CurrentModel(state): CurrentModel,
    Query(query): Query<DisaggDecodeQuery>,
    body: Bytes,
) -> Response {
    let cfg = current_runtime_config();
    if cfg.role == DisaggRole::Prefill {
        return err_resp(
            StatusCode::BAD_REQUEST,
            "node is configured with --role prefill and cannot serve /v1/disagg/decode",
        );
    }
    let packet = match DisaggStatePacket::decode_binary(&body) {
        Ok(pkt) => pkt,
        Err(e) => {
            return err_resp(
                StatusCode::BAD_REQUEST,
                format!("invalid disagg packet: {e:#}"),
            );
        }
    };
    if let Err(msg) = validate_packet_for_decode(&packet, state.max_seq_len) {
        return err_resp(StatusCode::BAD_REQUEST, msg);
    }
    let temperature = query.temperature.unwrap_or(state.default_temperature);
    let top_p = query.top_p.unwrap_or(state.default_top_p);
    if let Err(msg) = validate_sampling_params(temperature, top_p) {
        return err_resp(StatusCode::BAD_REQUEST, msg);
    }
    run_local_imported_decode(
        &state,
        packet,
        query.max_tokens.or(state.default_max_tokens).unwrap_or(256),
        temperature,
        query.top_k.unwrap_or(state.default_top_k),
        top_p,
        query.seed,
        None,
    )
    .await
}

/// `POST /v1/disagg/generate` — end-to-end `1P + 1D` / `1P + 2D` generation:
/// dispatches prefill to `--disagg-peer` (`1P`) and decodes locally (`1D`/`2D`).
pub async fn disagg_generate(
    CurrentModel(state): CurrentModel,
    Json(req): Json<DisaggGenerateRequest>,
) -> Response {
    let cfg = current_runtime_config();
    if cfg.role == DisaggRole::Prefill {
        return err_resp(
            StatusCode::BAD_REQUEST,
            "node is configured with --role prefill; submit /v1/disagg/generate to a --role decode or unified node",
        );
    }
    let peer_base = match resolve_prefill_peer(req.prefill_peer.as_deref()) {
        Ok(peer) => peer,
        Err((status, msg)) => return err_resp(status, msg),
    };
    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let top_p = req.top_p.unwrap_or(state.default_top_p);
    if let Err(msg) = validate_sampling_params(temperature, top_p) {
        return err_resp(StatusCode::BAD_REQUEST, msg);
    }
    let prompt_tokens =
        match resolve_prompt_tokens(&state, req.prompt.as_deref(), req.prompt_tokens) {
            Ok(t) => t,
            Err(msg) => return err_resp(StatusCode::BAD_REQUEST, msg),
        };
    let prefill_payload = DisaggPrefillRequest {
        prompt: None,
        prompt_tokens: Some(prompt_tokens),
        temperature: Some(temperature),
        top_k: req.top_k,
        top_p: Some(top_p),
        seed: req.seed,
    };
    let global_timeout = if state.request_timeout > 0 {
        Duration::from_secs(u64::from(state.request_timeout))
    } else {
        DISAGG_PEER_DEFAULT_GLOBAL_TIMEOUT
    };
    let connect_timeout = DISAGG_PEER_CONNECT_TIMEOUT.min(global_timeout);
    let target_url = format!("{peer_base}/v1/disagg/prefill");
    let fetch_res = tokio::task::spawn_blocking(
        move || -> std::result::Result<Vec<u8>, (StatusCode, String)> {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .timeout_connect(Some(connect_timeout))
                .timeout_global(Some(global_timeout))
                .build()
                .into();
            let mut resp = agent
                .post(&target_url)
                .send_json(&prefill_payload)
                .map_err(|e| {
                    tracing::error!(url = %target_url, error = %e, "prefill peer POST failed");
                    (
                        StatusCode::BAD_GATEWAY,
                        "prefill peer request failed".to_string(),
                    )
                })?;
            read_bounded_body(resp.body_mut().as_reader(), MAX_DISAGG_RESPONSE_BYTES).map_err(
                |err| match err {
                    PeerBodyReadError::TooLarge { max_bytes } => (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        format!(
                            "prefill peer response exceeded maximum allowed size ({max_bytes} bytes)"
                        ),
                    ),
                    PeerBodyReadError::Io(e) => {
                        tracing::error!(error = %e, "reading disagg prefill response body failed");
                        (
                            StatusCode::BAD_GATEWAY,
                            "reading prefill peer response failed".to_string(),
                        )
                    }
                },
            )
        },
    )
    .await;

    let raw_bytes = match fetch_res {
        Ok(Ok(b)) => b,
        Ok(Err((status, msg))) => return err_resp(status, msg),
        Err(e) => {
            tracing::error!(error = %e, "prefill peer task join failed");
            return err_resp(
                StatusCode::INTERNAL_SERVER_ERROR,
                "prefill peer dispatch failed",
            );
        }
    };
    let packet = match DisaggStatePacket::decode_binary(&raw_bytes) {
        Ok(pkt) => pkt,
        Err(e) => {
            tracing::error!("prefill peer returned invalid FDP1 packet: {e:#}");
            return err_resp(
                StatusCode::BAD_GATEWAY,
                "prefill peer returned an invalid state packet",
            );
        }
    };
    if let Err(msg) = validate_packet_for_decode(&packet, state.max_seq_len) {
        return err_resp(StatusCode::BAD_GATEWAY, msg);
    }
    run_local_imported_decode(
        &state,
        packet,
        req.max_tokens.or(state.default_max_tokens).unwrap_or(256),
        temperature,
        req.top_k.unwrap_or(state.default_top_k),
        top_p,
        req.seed,
        Some(peer_base),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_packet_roundtrip_preserves_all_fields() {
        let pkt = DisaggStatePacket {
            prompt_tokens: vec![151644, 872, 198, 9906, 151645],
            first_token: 358,
            num_blocks: 4,
            seq_len: 5,
            cached_prompt_tokens: 16,
            ttft_ms: 42.5,
            state_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03],
        };
        let encoded = pkt.encode_binary();
        let decoded = DisaggStatePacket::decode_binary(&encoded).expect("valid FDP1 packet");
        assert_eq!(decoded, pkt);
    }

    #[test]
    fn binary_packet_rejects_truncation_bad_magic_and_empty_prompt() {
        assert!(DisaggStatePacket::decode_binary(&[0u8; 12]).is_err());

        let mut bad_magic = DisaggStatePacket {
            prompt_tokens: vec![1, 2],
            first_token: 3,
            num_blocks: 1,
            seq_len: 2,
            cached_prompt_tokens: 0,
            ttft_ms: 1.0,
            state_bytes: vec![7, 8],
        }
        .encode_binary();
        bad_magic[0..4].copy_from_slice(b"NOPE");
        assert!(DisaggStatePacket::decode_binary(&bad_magic).is_err());

        let empty_prompt = DisaggStatePacket {
            prompt_tokens: Vec::new(),
            first_token: 3,
            num_blocks: 1,
            seq_len: 0,
            cached_prompt_tokens: 0,
            ttft_ms: 1.0,
            state_bytes: Vec::new(),
        }
        .encode_binary();
        assert!(DisaggStatePacket::decode_binary(&empty_prompt).is_err());

        let mut length_mismatch = DisaggStatePacket {
            prompt_tokens: vec![10],
            first_token: 11,
            num_blocks: 1,
            seq_len: 1,
            cached_prompt_tokens: 0,
            ttft_ms: 2.0,
            state_bytes: vec![1, 2, 3, 4],
        }
        .encode_binary();
        length_mismatch.pop();
        assert!(DisaggStatePacket::decode_binary(&length_mismatch).is_err());
    }

    #[test]
    fn peer_url_parsing_and_guard_cleanup_leave_zero_dangling_state() {
        let peers = parse_peer_urls(Some(" http://10.0.0.1:8000/ , http://10.0.0.2:8000 ,, "));
        assert_eq!(
            peers,
            vec![
                "http://10.0.0.1:8000".to_string(),
                "http://10.0.0.2:8000".to_string()
            ]
        );

        let baseline = ACTIVE_HANDOFFS.load(Ordering::Acquire);
        {
            let (_prefill_guard, _rx) = DisaggPrefillGuard::register();
            assert_eq!(ACTIVE_HANDOFFS.load(Ordering::Acquire), baseline + 1);
        }
        assert_eq!(ACTIVE_HANDOFFS.load(Ordering::Acquire), baseline);

        let pkt = DisaggStatePacket {
            prompt_tokens: vec![42],
            first_token: 99,
            num_blocks: 1,
            seq_len: 1,
            cached_prompt_tokens: 0,
            ttft_ms: 5.0,
            state_bytes: vec![0xAA],
        };
        let decode_guard = DisaggDecodeGuard::register(pkt.clone());
        let taken = take_decode_import(decode_guard.session_hash()).expect("registered import");
        assert_eq!(taken, pkt);
        assert!(take_decode_import(decode_guard.session_hash()).is_none());
        drop(decode_guard);
        assert_eq!(ACTIVE_HANDOFFS.load(Ordering::Acquire), baseline);

        {
            let (prefill_guard, _rx) = DisaggPrefillGuard::register();
            let sh = prefill_guard.session_hash();
            assert!(
                has_pending_prefill_export(sh),
                "pending export must be visible even when first token is <think>"
            );
            drop(prefill_guard);
            assert!(!has_pending_prefill_export(sh));
        }
    }

    #[test]
    fn select_peer_enforces_allowlist_and_blocks_ssrf() {
        // Empty allowlist must reject any client-supplied prefill_peer (SSRF)
        // and return 400 when omitted.
        let (status, _) =
            select_peer_from_allowlist(&[], Some("http://169.254.169.254/latest/meta-data"))
                .expect_err("client-supplied peer must be rejected when allowlist is empty");
        assert_eq!(status, StatusCode::FORBIDDEN);

        let (status, _) = select_peer_from_allowlist(&[], None)
            .expect_err("missing allowlist must fail with 400");
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Non-empty allowlist: exact match (after trailing `/` normalization) succeeds;
        // arbitrary hosts, path suffixes, and userinfo tricks are rejected.
        let allowlist = parse_peer_urls(Some("http://10.0.0.1:8000/,http://10.0.0.2:8000"));
        assert_eq!(
            select_peer_from_allowlist(&allowlist, Some("http://10.0.0.1:8000/")).unwrap(),
            "http://10.0.0.1:8000"
        );
        assert_eq!(
            select_peer_from_allowlist(&allowlist, Some("  http://10.0.0.2:8000///  ")).unwrap(),
            "http://10.0.0.2:8000"
        );
        for bad in [
            "http://169.254.169.254",
            "http://127.0.0.1:9000",
            "http://10.0.0.1:8000/extra",
            concat!("http://10.0.0.1:8000", "@evil.internal"),
            "http://10.0.0.3:8000",
        ] {
            let (status, _) = select_peer_from_allowlist(&allowlist, Some(bad))
                .expect_err("non-allowlisted peer must be rejected");
            assert_eq!(status, StatusCode::FORBIDDEN, "peer {bad}");
        }

        // Omitting explicit peer selects from the allowlist.
        let picked = select_peer_from_allowlist(&allowlist, None).expect("rr peer");
        assert!(allowlist.contains(&picked));
    }

    #[test]
    fn bounded_peer_body_read_enforces_exact_byte_cap() {
        let exact = vec![0x5Au8; 16];
        let got = read_bounded_body(exact.as_slice(), 16).expect("exact limit must succeed");
        assert_eq!(got, exact);

        let over = vec![0x5Au8; 17];
        match read_bounded_body(over.as_slice(), 16) {
            Err(PeerBodyReadError::TooLarge { max_bytes }) => assert_eq!(max_bytes, 16),
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn decode_validation_checks_prompt_bounds_sampling_and_timeout() {
        let empty = DisaggStatePacket {
            prompt_tokens: Vec::new(),
            first_token: 1,
            num_blocks: 1,
            seq_len: 0,
            cached_prompt_tokens: 0,
            ttft_ms: 1.0,
            state_bytes: Vec::new(),
        };
        assert!(validate_packet_for_decode(&empty, 128).is_err());

        let too_long = DisaggStatePacket {
            prompt_tokens: vec![1; 128],
            ..empty.clone()
        };
        assert!(validate_packet_for_decode(&too_long, 128).is_err());

        let ok_pkt = DisaggStatePacket {
            prompt_tokens: vec![1; 127],
            ..empty
        };
        assert!(validate_packet_for_decode(&ok_pkt, 128).is_ok());

        assert!(validate_sampling_params(0.0, 1.0).is_ok());
        assert!(validate_sampling_params(0.7, 0.95).is_ok());
        assert!(validate_sampling_params(-0.1, 0.9).is_err());
        assert!(validate_sampling_params(f32::NAN, 0.9).is_err());
        assert!(validate_sampling_params(f32::INFINITY, 0.9).is_err());
        assert!(validate_sampling_params(0.7, 0.0).is_err());
        assert!(validate_sampling_params(0.7, -0.5).is_err());
        assert!(validate_sampling_params(0.7, 1.01).is_err());
        assert!(validate_sampling_params(0.7, f32::NAN).is_err());

        assert!(request_deadline(0).is_none());
        assert!(request_deadline(30).is_some());

        let source = include_str!("disagg.rs");
        let prod = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("production source precedes tests");
        assert!(!prod.contains("request_tx.send("));
    }

    #[test]
    fn build_blocking_request_collapses_top_k_one_to_canonical_greedy_temperature() {
        let (tx_greedy, _rx_greedy) = oneshot::channel();
        let greedy = build_blocking_request(
            vec![1, 2],
            42,
            16,
            0.7,
            1,
            0.8,
            0.0,
            0.0,
            Some(7),
            None,
            tx_greedy,
        );
        let InferenceRequest::Blocking {
            temperature, top_k, ..
        } = greedy
        else {
            panic!("expected blocking request");
        };
        assert_eq!(temperature, 0.0);
        assert_eq!(top_k, 1);

        let (tx_sampled, _rx_sampled) = oneshot::channel();
        let sampled = build_blocking_request(
            vec![1, 2],
            42,
            16,
            0.7,
            20,
            0.8,
            0.0,
            0.0,
            Some(7),
            None,
            tx_sampled,
        );
        let InferenceRequest::Blocking {
            temperature, top_k, ..
        } = sampled
        else {
            panic!("expected blocking request");
        };
        assert_eq!(temperature, 0.7);
        assert_eq!(top_k, 20);
    }
}
