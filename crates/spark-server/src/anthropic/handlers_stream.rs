// SPDX-License-Identifier: AGPL-3.0-only

use axum::response::sse::Event;
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;

use super::translator::*;
/// Translate an OpenAI SSE response into Anthropic's structured event
/// stream, **per-chunk**: each OpenAI `data: {…}` line that arrives
/// produces zero or more Anthropic events emitted to the client
/// before the next OpenAI chunk is processed.
///
/// Implementation: spawn a tokio task that consumes the inner body's
/// `Bytes` stream, splits on `\n\n` event boundaries, parses the
/// `data:` payload as JSON, and feeds it through
/// `AnthropicTranslator::process_openai_chunk`. The translator's
/// emitted events are sent down an `mpsc` channel that's wrapped as
/// a `ReceiverStream` and handed to axum's `Sse::new` — the same
/// pattern api.rs uses for its own SSE response.
pub(super) async fn wrap_chat_sse_for_anthropic(
    chat_resp: Response,
    req_model: String,
) -> Response {
    let (parts, body) = chat_resp.into_parts();
    if !parts.status.is_success() {
        // Forward error envelope verbatim, status preserved.
        return Response::from_parts(parts, body);
    }

    // Match the OpenAI chat_stream sizing — see chat_stream/mod.rs for rationale.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(1024);

    tokio::spawn(async move {
        let mut translator = AnthropicTranslator::new(req_model);
        // Buffer raw bytes until complete `\n\n` SSE frame boundaries so
        // multi-byte UTF-8 codepoints split across HTTP chunks are never
        // corrupted into U+FFFD replacement characters.
        let mut buf: Vec<u8> = Vec::new();
        let mut data_stream = body.into_data_stream();
        let mut pending: Vec<Event> = Vec::new();

        // Inner helper: drain `pending` into the channel. Returns
        // `false` when the receiver has hung up — caller should
        // abort.
        async fn flush(
            tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
            pending: &mut Vec<Event>,
        ) -> bool {
            for ev in pending.drain(..) {
                if tx.send(Ok(ev)).await.is_err() {
                    return false;
                }
            }
            true
        }

        loop {
            let chunk_res = tokio::select! {
                biased;
                _ = tx.closed() => return,
                next = data_stream.next() => next,
            };
            let Some(chunk_res) = chunk_res else {
                break;
            };
            let chunk = match chunk_res {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("anthropic stream: inner body error: {e}");
                    break;
                }
            };
            buf.extend_from_slice(&chunk);

            // Process any complete `data: {…}\n\n` records.
            while let Some(raw_record) = drain_next_sse_frame(&mut buf) {
                feed_sse_record(&raw_record, &mut translator, &mut pending);
                if !flush(&tx, &mut pending).await {
                    return;
                }
            }
        }

        if tx.is_closed() {
            return;
        }

        // Final flush: drain any tail bytes (no trailing `\n\n`) and
        // ensure message_stop fires even if upstream omitted
        // finish_reason.
        if !buf.is_empty() {
            let tail = String::from_utf8_lossy(&buf);
            feed_sse_record(&tail, &mut translator, &mut pending);
        }
        translator.finalize(&mut pending);
        if !flush(&tx, &mut pending).await {
            tracing::warn!("anthropic stream: final flush failed (receiver dropped)");
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(crate::api::chat_stream::sse_keep_alive())
        .into_response()
}

fn drain_next_sse_frame(buf: &mut Vec<u8>) -> Option<String> {
    let boundary = buf.windows(2).position(|w| w == b"\n\n")?;
    let raw: Vec<u8> = buf.drain(..boundary + 2).collect();
    Some(String::from_utf8_lossy(&raw).into_owned())
}

fn feed_sse_record(
    raw_record: &str,
    translator: &mut AnthropicTranslator,
    pending: &mut Vec<Event>,
) {
    for line in raw_record.lines() {
        let payload = match line.strip_prefix("data:") {
            Some(s) => s.trim(),
            None => continue,
        };
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(payload) {
            translator.process_openai_chunk(&val, pending);
        }
    }
}

#[cfg(test)]
mod utf8_frame_split_tests {
    use super::drain_next_sse_frame;

    #[test]
    fn byte_buffered_sse_framing_preserves_multibyte_utf8_split_across_chunks() {
        let frame = "data: {\"choices\":[{\"delta\":{\"content\":\"café 🚀 你好\"}}]}\n\n";
        let bytes = frame.as_bytes();
        // Split right inside the 4-byte UTF-8 encoding of 🚀.
        let rocket_pos = frame.find('🚀').unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&bytes[..rocket_pos + 2]);
        assert!(drain_next_sse_frame(&mut buf).is_none());
        buf.extend_from_slice(&bytes[rocket_pos + 2..]);
        let decoded = drain_next_sse_frame(&mut buf).expect("complete SSE frame");
        assert_eq!(decoded, frame);
        assert!(!decoded.contains('\u{FFFD}'));
    }

    #[tokio::test]
    async fn prefill_phase_disconnect_drops_upstream_stream() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;

        use axum::response::sse::Event;
        use axum::response::{IntoResponse, Sse};
        use tokio::sync::Notify;

        struct PendingDropStream {
            dropped: Arc<AtomicBool>,
            notify: Arc<Notify>,
        }

        impl Drop for PendingDropStream {
            fn drop(&mut self) {
                self.dropped.store(true, Ordering::Release);
                self.notify.notify_one();
            }
        }

        impl futures::Stream for PendingDropStream {
            type Item = Result<Event, std::convert::Infallible>;

            fn poll_next(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Self::Item>> {
                std::task::Poll::Pending
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(Notify::new());
        let upstream = Sse::new(PendingDropStream {
            dropped: dropped.clone(),
            notify: notify.clone(),
        })
        .into_response();

        let resp = super::wrap_chat_sse_for_anthropic(upstream, "test-model".to_string()).await;
        assert!(!dropped.load(Ordering::Acquire));

        // Drop the Anthropic SSE response body to simulate a client disconnect during prefill.
        drop(resp);

        tokio::time::timeout(Duration::from_secs(2), notify.notified())
            .await
            .expect("upstream stream must drop promptly when client disconnects during prefill");
        assert!(dropped.load(Ordering::Acquire));
    }
}
