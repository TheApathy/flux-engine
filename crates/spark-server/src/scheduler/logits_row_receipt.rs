// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded, observation-only row/output mapping for a fresh exclusive C1 run.
//! Session hashes are not unique request IDs. External one-request provenance
//! remains mandatory. A missing or truncated pair is incomplete evidence.

use super::{ActiveSeq, DevicePtr, Model};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

const OUTPUT_LIMIT: usize = 24;
const EVENT_LIMIT: usize = 128;
static EVENTS: AtomicUsize = AtomicUsize::new(0);

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_LOGITS_ROW_RECEIPTS").ok().as_deref() == Some("1"))
        && EVENTS.load(Ordering::Relaxed) < EVENT_LIMIT
}

fn emit_event(kind: &str, value: serde_json::Value) {
    if EVENTS
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            (n < EVENT_LIMIT).then_some(n + 1)
        })
        .is_ok()
    {
        tracing::info!("{kind} {value}");
    }
}

fn digest(bytes: &[u8]) -> String {
    let hash = ring::digest::digest(&ring::digest::SHA256, bytes);
    hash.as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn history_digest(tokens: &[u32]) -> String {
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    for token in tokens {
        context.update(&token.to_le_bytes());
    }
    context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Debug)]
pub(super) struct Location {
    pub source: &'static str,
    pub key: String,
    pub logical_row: Option<usize>,
    pub physical_row: Option<usize>,
    pub frame_rows: Option<usize>,
    pub pre_verify_len: Option<usize>,
    pub input: Option<u32>,
    pub candidate: Option<u32>,
}

#[derive(Debug)]
pub(super) struct Snapshot {
    session: u64,
    location: Location,
    before: usize,
    history: String,
    selected: u32,
}

fn snapshot(session: u64, history: &[u32], selected: u32, location: Location) -> Snapshot {
    Snapshot {
        session,
        location,
        before: history.len(),
        history: history_digest(history),
        selected,
    }
}

pub(super) fn capture_host(
    session: u64,
    history: &[u32],
    selected: u32,
    row: &[u8],
    fp32: bool,
    vocab: usize,
    location: Location,
) -> Option<Snapshot> {
    if !enabled() || history.len() >= OUTPUT_LIMIT {
        return None;
    }
    if vocab.checked_mul(if fp32 { 4 } else { 2 }) != Some(row.len()) {
        return None;
    }
    let saved = snapshot(session, history, selected, location);
    emit_event(
        "LOGITS_RAW_V1",
        serde_json::json!({
            "schema": 1, "session": format!("{session:016x}"),
            "source": saved.location.source, "observation_key": saved.location.key,
            "output_before": saved.before, "history_before_sha256": saved.history,
            "selected_token": selected, "logits_sha256": digest(row),
            "logits_stage": "target_device_pre_policy",
            "logits_dtype": if fp32 { "fp32" } else { "bf16" }, "vocab": vocab,
            "logical_row_index": saved.location.logical_row,
            "physical_row_index": saved.location.physical_row,
            "frame_rows": saved.location.frame_rows,
            "pre_verify_len": saved.location.pre_verify_len,
            "input_token": saved.location.input, "candidate_draft": saved.location.candidate,
        }),
    );
    Some(saved)
}

pub(super) fn capture_device(
    model: &dyn Model,
    ptr: DevicePtr,
    session: u64,
    history: &[u32],
    selected: u32,
    location: Location,
) -> Option<Snapshot> {
    if !enabled() || history.len() >= OUTPUT_LIMIT {
        return None;
    }
    let fp32 = model.logits_ptr_is_fp32(ptr);
    let bytes = model.vocab_size().checked_mul(if fp32 { 4 } else { 2 })?;
    let mut row = vec![0; bytes];
    // Diagnostic failure must not change the selected token or request state.
    if model.copy_logits_to_host(ptr, &mut row).is_err() {
        return None;
    }
    capture_host(
        session,
        history,
        selected,
        &row,
        fp32,
        model.vocab_size(),
        location,
    )
}

pub(super) fn output_map(saved: &Snapshot, output: &[u32]) {
    emit_event("LOGITS_OUTPUT_MAP_V1", output_map_value(saved, output));
}

fn output_map_value(saved: &Snapshot, output: &[u32]) -> serde_json::Value {
    let prefix_preserved = output
        .get(..saved.before)
        .is_some_and(|prefix| history_digest(prefix) == saved.history);
    let emitted = if prefix_preserved {
        &output[saved.before..]
    } else {
        &[]
    };
    serde_json::json!({
        "schema": 1, "session": format!("{:016x}", saved.session),
        "source": saved.location.source, "observation_key": saved.location.key,
        "output_before": saved.before, "output_after": output.len(),
        "emitted_ids": emitted, "history_before_sha256": saved.history,
        "history_after_sha256": history_digest(output),
        "prefix_preserved": prefix_preserved, "selected_token": saved.selected,
    })
}

fn prefill_location() -> Location {
    Location {
        source: "prefill",
        key: "prefill:0".to_owned(),
        logical_row: None,
        physical_row: None,
        frame_rows: None,
        pre_verify_len: None,
        input: None,
        candidate: None,
    }
}

pub(super) fn prefill_raw(model: &dyn Model, ptr: DevicePtr, session: u64, selected: u32) {
    if enabled() {
        let _ = capture_device(model, ptr, session, &[], selected, prefill_location());
    }
}

pub(super) fn prefill_output(a: &ActiveSeq) {
    if enabled() {
        output_map(
            &snapshot(a.session_hash, &[], a.last_token, prefill_location()),
            &a.output_tokens,
        );
    }
}

pub(super) fn single_location(source: &'static str, a: &ActiveSeq, row: usize) -> Location {
    Location {
        source,
        key: format!("{source}:{}", a.output_tokens.len()),
        logical_row: Some(0),
        physical_row: Some(row),
        frame_rows: Some(1),
        pre_verify_len: None,
        input: Some(a.last_token),
        candidate: None,
    }
}

#[derive(Debug)]
pub(super) struct VerifyFrame {
    inputs: Vec<u32>,
    pre_verify_len: usize,
    rows: Vec<u8>,
    fp32: bool,
    vocab: usize,
}

pub(super) fn verify_frame(
    model: &dyn Model,
    a: &ActiveSeq,
    inputs: &[u32],
    pre_verify_len: usize,
) -> Option<VerifyFrame> {
    if !enabled() || a.output_tokens.len() >= OUTPUT_LIMIT || inputs.is_empty() {
        return None;
    }
    // Only the generic flat caller supplies the actual complete input frame.
    let ptr = model.logits_buffer_ptr();
    let fp32 = model.logits_ptr_is_fp32(ptr);
    let vocab = model.vocab_size();
    let bytes = vocab
        .checked_mul(if fp32 { 4 } else { 2 })?
        .checked_mul(inputs.len())?;
    let mut rows = vec![0; bytes];
    if model.copy_logits_to_host(ptr, &mut rows).is_err() {
        return None;
    }
    Some(VerifyFrame {
        inputs: inputs.to_vec(),
        pre_verify_len,
        rows,
        fp32,
        vocab,
    })
}

impl VerifyFrame {
    pub(super) fn row(&self, a: &ActiveSeq, row: usize, selected: u32) -> Option<Snapshot> {
        if !enabled() || a.output_tokens.len() >= OUTPUT_LIMIT {
            return None;
        }
        let input = *self.inputs.get(row)?;
        let bytes = self.vocab.checked_mul(if self.fp32 { 4 } else { 2 })?;
        let offset = row.checked_mul(bytes)?;
        capture_host(
            a.session_hash,
            &a.output_tokens,
            selected,
            self.rows.get(offset..offset.checked_add(bytes)?)?,
            self.fp32,
            self.vocab,
            Location {
                source: "flat_verify",
                key: format!(
                    "flat_verify:{}:{row}:{}",
                    self.pre_verify_len,
                    a.output_tokens.len()
                ),
                logical_row: Some(row),
                physical_row: Some(row),
                frame_rows: Some(self.inputs.len()),
                pre_verify_len: Some(self.pre_verify_len),
                input: Some(input),
                candidate: self.inputs.get(row + 1).copied(),
            },
        )
    }
}

std::thread_local! {
    static ACTIVE_VERIFY_FRAME: std::cell::RefCell<Option<VerifyFrame>> = const { std::cell::RefCell::new(None) };
}

pub(super) struct VerifyFrameGuard(bool);

impl Drop for VerifyFrameGuard {
    fn drop(&mut self) {
        if self.0 {
            ACTIVE_VERIFY_FRAME.with(|slot| {
                *slot.borrow_mut() = None;
            });
        }
    }
}

pub(super) fn enter_verify_frame(
    model: &dyn Model,
    a: &ActiveSeq,
    inputs: &[u32],
    pre_verify_len: usize,
) -> VerifyFrameGuard {
    let Some(frame) = verify_frame(model, a, inputs, pre_verify_len) else {
        return VerifyFrameGuard(false);
    };
    ACTIVE_VERIFY_FRAME.with(|slot| {
        *slot.borrow_mut() = Some(frame);
    });
    VerifyFrameGuard(true)
}

pub(super) fn current_verify_row(a: &ActiveSeq, row: usize, selected: u32) -> Option<Snapshot> {
    if !enabled() {
        return None;
    }
    ACTIVE_VERIFY_FRAME.with(|slot| slot.borrow().as_ref()?.row(a, row, selected))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_history_sha_is_standard_sha256() {
        assert_eq!(
            history_digest(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn history_uses_little_endian_u32_not_text() {
        assert_eq!(
            history_digest(&[1, 0x12345678]),
            digest(&[1, 0, 0, 0, 0x78, 0x56, 0x34, 0x12])
        );
        assert_ne!(history_digest(&[1, 23]), history_digest(&[12, 3]));
    }

    #[test]
    fn selected_prefill_is_not_automatically_output_zero() {
        let selected = snapshot(1, &[], 7, prefill_location());
        let value = output_map_value(&selected, &[]);
        assert_eq!(value["output_after"], 0);
        assert_eq!(value["emitted_ids"], serde_json::json!([]));
    }

    #[test]
    fn output_mapping_records_actual_appended_tokens() {
        let selected = snapshot(1, &[5], 7, prefill_location());
        let value = output_map_value(&selected, &[5, 8, 9]);
        assert_eq!(value["selected_token"], 7);
        assert_eq!(value["emitted_ids"], serde_json::json!([8, 9]));
        assert_eq!(value["output_before"], 1);
        assert_eq!(value["output_after"], 3);
    }

    #[test]
    fn rollback_or_prefix_rewrite_cannot_look_like_append() {
        let selected = snapshot(1, &[5, 6], 7, prefill_location());
        for output in [&[5][..], &[5, 9, 7][..]] {
            let value = output_map_value(&selected, output);
            assert_eq!(value["prefix_preserved"], false);
            assert_eq!(value["emitted_ids"], serde_json::json!([]));
        }
    }
}
