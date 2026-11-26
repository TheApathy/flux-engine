// SPDX-License-Identifier: AGPL-3.0-only

//! The scheduler owns and joins its admission receiver on every exit path.

use super::{Arc, Condvar, InferenceRequest, Mutex, PendingQueue, push_pending_bounded};

const STOPPED: &str = "Scheduler stopped before this queued request could run";

fn reject(request: InferenceRequest) {
    // A rejected request stops waiting (`--max-queue-depth`).
    super::queue_gate::QUEUE_GATE.leave(1);
    match request {
        InferenceRequest::Blocking { response_tx, .. } => {
            let _ = response_tx.send(Err(anyhow::anyhow!(STOPPED)));
        }
        InferenceRequest::Streaming { token_tx, .. } => {
            // A full sink is dropped immediately; no receiver can keep the
            // scheduler's unwind parked behind a streaming client.
            let _ = token_tx.try_send(crate::api::StreamEvent::Error(STOPPED.to_string()));
        }
    }
}

struct ClosePending(Arc<(Mutex<PendingQueue>, Condvar)>);

impl Drop for ClosePending {
    fn drop(&mut self) {
        self.0.0.lock().closed = true;
        self.0.1.notify_all();
    }
}

pub(in crate::scheduler) struct RequestReceiver {
    pending: Arc<(Mutex<PendingQueue>, Condvar)>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl RequestReceiver {
    pub(super) fn spawn(
        mut rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
        pending: Arc<(Mutex<PendingQueue>, Condvar)>,
        cap: usize,
    ) -> Self {
        let (stop, mut stopped) = tokio::sync::oneshot::channel();
        let receiver_pending = Arc::clone(&pending);
        let thread = std::thread::spawn(move || {
            let _closed = ClosePending(Arc::clone(&receiver_pending));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("request receiver runtime");
            runtime.block_on(async {
                loop {
                    let request = tokio::select! {
                        biased;
                        _ = &mut stopped => break,
                        request = rx.recv() => request,
                    };
                    let Some(request) = request else { break };
                    if let Some(request) = push_pending_bounded(&receiver_pending, request, cap) {
                        reject(request);
                        break;
                    }
                }
            });
            // Closing precedes draining: sends racing scheduler exit either
            // land in this drain or fail as unavailable at HTTP admission.
            rx.close();
            while let Ok(request) = rx.try_recv() {
                reject(request);
            }
        });
        Self {
            pending,
            stop: Some(stop),
            thread: Some(thread),
        }
    }
}

impl Drop for RequestReceiver {
    fn drop(&mut self) {
        let requests = {
            let mut queue = self.pending.0.lock();
            queue.closed = true;
            std::mem::take(&mut queue.requests)
        };
        self.pending.1.notify_all();
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        for request in requests {
            reject(request);
        }
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            atlas_core::fault::global()
                .latch("scheduler admission receiver panicked; restart required");
        }
        // The receiver no longer exists, so admission is closed and no hidden
        // request sink can outlive this guard. Normal exit and unwind agree.
    }
}
