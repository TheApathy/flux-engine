// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Error(msg)` arm of the streaming `flat_map` closure.

use axum::response::sse::Event;
use std::sync::atomic::{AtomicBool, Ordering};

use super::ctx::StreamCtx;

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

fn encoded_error(message: String) -> String {
    serde_json::json!({
        "error": {"message": message, "type": "server_error", "code": 500}
    })
    .to_string()
}

fn cancel_scheduler(cancel_flag: &AtomicBool) {
    cancel_flag.store(true, Ordering::Release);
}

pub(super) fn handle_error(ctx: &StreamCtx, cancel_flag: &AtomicBool, msg: String) -> SseVec {
    // An error frame is terminal for the HTTP response, but the scheduler may
    // still own a sender and an in-flight sequence.  Publish cancellation
    // before accounting/refund so the sequence is released even if the client
    // keeps the SSE body open long enough to receive the encoded error.
    cancel_scheduler(cancel_flag);
    if !ctx
        .active_guard
        .swap(true, std::sync::atomic::Ordering::SeqCst)
    {
        crate::metrics::REQUESTS_ACTIVE.dec();
    }
    // Abandoned stream — refund the full reservation.
    if let Some(ref rctx) = ctx.req_ctx {
        ctx.state
            .rate_limiter
            .refund_tokens(&rctx.identity, rctx.reserved_tokens);
    }
    let events: SseVec = vec![Ok(Event::default().data(encoded_error(msg)))];
    events
}

#[cfg(test)]
mod tests {
    use super::{cancel_scheduler, encoded_error};
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn token_id_mismatch_is_an_encoded_server_error() {
        let message = "completion token-ID cardinality 1 != usage 2";
        let value: serde_json::Value =
            serde_json::from_str(&encoded_error(message.into())).unwrap();
        assert_eq!(value["error"]["message"], message);
        assert_eq!(value["error"]["type"], "server_error");
        assert_eq!(value["error"]["code"], 500);
    }

    #[test]
    fn terminal_error_publication_uses_release_cancellation() {
        let cancel = AtomicBool::new(false);

        cancel_scheduler(&cancel);

        assert!(cancel.load(Ordering::Acquire));
    }
}
