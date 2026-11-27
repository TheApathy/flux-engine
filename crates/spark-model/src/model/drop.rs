// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

use super::types::TransformerModel;

fn destroy_graph_cache<K>(
    gpu: &dyn GpuBackend,
    cache_name: &str,
    cache: &mut HashMap<K, GraphHandle>,
) {
    for graph in cache.drain().map(|(_, graph)| graph) {
        if graph.0 != 0
            && let Err(error) = gpu.destroy_graph(graph)
        {
            tracing::error!(
                "TransformerModel::drop: destroy_graph({cache_name}, {}) failed: {error:#}",
                graph.0
            );
            // Automatic field teardown would free pinned graph sources while
            // the failed graph may still retain them. Fail closed first.
            std::process::abort();
        }
    }
}

impl TransformerModel {
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(crate) fn initialize_qwen4_ple_prefill(&mut self, max_tokens: usize) -> anyhow::Result<()> {
        let Self { qwen4_ple, gpu, .. } = self;
        if let Some(ple) = qwen4_ple.as_mut() {
            ple.initialize_prefill(gpu.as_ref(), max_tokens)?;
            ple.initialize_verify_scratch(gpu.as_ref())?;
        }
        Ok(())
    }
}

impl Drop for TransformerModel {
    fn drop(&mut self) {
        // Every pinned-host source must remain allocated and immutable until
        // its eager async copy completes, and captured H2D source addresses
        // must remain allocated until their graph is destroyed. Drop runs
        // before automatic field teardown, so both the GPU backend and all
        // layer-owned Qwen pointer tables are still alive here.
        let default_stream = self.gpu.default_stream();
        if let Err(error) = self.gpu.synchronize(default_stream) {
            tracing::error!(
                "TransformerModel::drop: default-stream synchronization failed: {error:#}"
            );
            std::process::abort();
        }
        if self.secondary_stream != default_stream
            && let Err(error) = self.gpu.synchronize(self.secondary_stream)
        {
            tracing::error!(
                "TransformerModel::drop: secondary-stream synchronization failed: {error:#}"
            );
            std::process::abort();
        }

        let extra_slots: Vec<crate::traits::SequenceState> = self
            .worker_extra_slots
            .get_mut()
            .drain()
            .map(|(_, s)| s)
            .collect();
        for mut extra_seq in extra_slots {
            let _ = crate::traits::Model::free_sequence(self, &mut extra_seq);
        }

        // Piecewise graphs contain Qwen H2D nodes whose source pointer is
        // owned by `layers`; destroy them first, before any layer field drops.
        destroy_graph_cache(
            self.gpu.as_ref(),
            "piecewise_decode_graphs",
            self.piecewise_decode_graphs.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "decode_graph",
            self.decode_graph.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "batch_decode_graphs",
            self.batch_decode_graphs.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "verify2_graph",
            self.verify2_graph.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "verify3_graph",
            self.verify3_graph.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "verify4_graph",
            self.verify4_graph.get_mut(),
        );
        destroy_graph_cache(
            self.gpu.as_ref(),
            "verify_kgamma_graph",
            self.verify_kgamma_graph.get_mut(),
        );
        self.verify_kgamma_lru.get_mut().last_used.clear();

        if let Err(error) = self.vision_embeddings.get_mut().release(self.gpu.as_ref()) {
            tracing::error!("TransformerModel::drop: vision aggregate release failed: {error:#}");
            std::process::abort();
        }

        for (layer_index, layer) in self.layers.iter().enumerate() {
            if let Err(error) = layer.release_qwen4_cross_seq_resources(self.gpu.as_ref()) {
                tracing::error!(
                    "TransformerModel::drop: layer {layer_index} cross-sequence verify \
                     resource release failed: {error:#}"
                );
            }
        }

        if let Some(ple) = &self.qwen4_ple
            && let Err(error) = ple.destroy_owned_resources(self.gpu.as_ref())
        {
            tracing::error!("TransformerModel::drop: destroy PLE resources failed: {error:#}");
        }

        // Destroy any proposer-owned Stage-2 CUDA graphs and rejection scratch
        // before freeing the model-owned shared rejection buffers they reference.
        if let Some(proposer) = &self.proposer {
            proposer.release_rejection_resources(self.gpu.as_ref());
        }
        if let Some(proposer_alt) = &self.proposer_alt {
            proposer_alt.release_rejection_resources(self.gpu.as_ref());
        }
        if !self.target_rejection_verify_buf.is_null() {
            if let Err(error) = self.gpu.free(self.target_rejection_verify_buf) {
                tracing::error!(
                    "TransformerModel::drop: free(target_rejection_verify_buf) failed: {error:#}"
                );
            }
            self.target_rejection_verify_buf = DevicePtr::NULL;
        }
        if !self.target_rejection_q_buf.is_null() {
            if let Err(error) = self.gpu.free(self.target_rejection_q_buf) {
                tracing::error!(
                    "TransformerModel::drop: free(target_rejection_q_buf) failed: {error:#}"
                );
            }
            self.target_rejection_q_buf = DevicePtr::NULL;
        }

        if self.secondary_event != 0
            && let Err(error) = self.gpu.destroy_event(self.secondary_event)
        {
            tracing::error!(
                "TransformerModel::drop: destroy_event({}) failed: {error:#}",
                self.secondary_event
            );
        }

        if self.dflash_capture_done_event != 0
            && let Err(error) = self.gpu.destroy_event(self.dflash_capture_done_event)
        {
            tracing::error!(
                "TransformerModel::drop: destroy_event({}) (dflash_capture_done) failed: {error:#}",
                self.dflash_capture_done_event
            );
        }

        if let Some(&event) = self.slot_zero_event.get()
            && let Err(error) = self.gpu.destroy_event(event)
        {
            tracing::error!(
                "TransformerModel::drop: destroy_event({event}) (slot zero) failed: {error:#}"
            );
        }

        if let Some(&Some((ein, eout))) = self.k9x_lastk_events.get() {
            for (label, event) in [("in", ein), ("out", eout)] {
                if event != 0
                    && let Err(error) = self.gpu.destroy_event(event)
                {
                    tracing::error!(
                        "TransformerModel::drop: destroy_event({event}) (k9x_lastk_{label}) failed: {error:#}"
                    );
                }
            }
        }

        // Release per-instance tracking entries so future `TransformerModel`
        // instances cannot alias stale slot/cache state after drop.
        self.clear_model_slot_tracking();
        self.clear_ssm_norm_cached_slot();
        self.clear_serial_batch_suppress();

        // Everything above is graphs, vision buffers and events — the small
        // set of GPU allocations this Drop impl already knew how to name.
        // Everything else this model holds (weights, KV cache, `BufferArena`,
        // the SSM pools, and the long tail of ad-hoc persistent `DevicePtr`
        // fields on this struct) is a raw, `Copy` pointer with no destructor
        // of its own, and DELIBERATELY stays unfreed here — see the doc on
        // `GpuBackend::free_all_allocations`.
        //
        // `Drop` is the wrong place to call it: `TransformerModel` drops on
        // whatever thread happens to hold the last reference, which is NOT
        // always the quiescent moment BUG #29 requires. A model that fails
        // to load partway through construction drops on the swap thread
        // while nothing else is concurrently allocating — fine — but a
        // model that drops for any less-analysed reason (a panic unwind,
        // a future EP-worker teardown path) could race concurrent GPU
        // traffic, which is the exact interleaving BUG #29 corrupts.
        // `model_swap::swap` calls `free_all_allocations` explicitly, on
        // the specific backend handle it kept for this purpose, at the one
        // point its own design already reasoned about: right after the
        // scheduler thread — the sole other owner of this model — has been
        // joined, and before the next model starts allocating. See there.
    }
}
