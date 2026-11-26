// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use crate::layers::dense_ffn::{DenseFfnWeights, SharedE2m1Scratch};
use crate::layers::shared_splitk::SharedSplitkWorkspace;
use crate::layers::{DenseFfnLayer, FfnComponent};
use crate::memlean::DenseLoadPlan;
use crate::weight_map::w3_sidecar::{W3SidecarRequest, W3SidecarSession};
use crate::weight_map::{Nvfp4Variant, load_dense_ffn};

pub(super) struct DenseFfnLoadContext<'a> {
    store: &'a WeightStore,
    config: &'a ModelConfig,
    gpu: &'a dyn GpuBackend,
    variant: Nvfp4Variant,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
    flashinfer_requested: bool,
    lean: DenseLoadPlan,
    /// `ATLAS_MEMLEAN_SHARED_SCRATCH`: the activation arena every layer adopts.
    shared_scratch: Option<SharedE2m1Scratch>,
    /// `ATLAS_MEMLEAN_SHARED_SCRATCH`: the model's one split-K workspace.
    shared_splitk: Option<&'a SharedSplitkWorkspace>,
}

impl<'a> DenseFfnLoadContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        store: &'a WeightStore,
        config: &'a ModelConfig,
        gpu: &'a dyn GpuBackend,
        variant: Nvfp4Variant,
        absmax_k: KernelHandle,
        quantize_k: KernelHandle,
        stream: u64,
        flashinfer_requested: bool,
        lean: DenseLoadPlan,
        shared_splitk: Option<&'a SharedSplitkWorkspace>,
    ) -> Self {
        Self {
            store,
            config,
            gpu,
            variant,
            absmax_k,
            quantize_k,
            stream,
            flashinfer_requested,
            lean,
            shared_scratch: lean.shared_scratch.then(SharedE2m1Scratch::default),
            shared_splitk,
        }
    }

    /// Build one layer's dense FFN. Also returns the checkpoint bytes the
    /// MEMLEAN gate/up alias released for it (0 when that selector is off).
    pub(super) fn load(
        &self,
        layer: usize,
        layer_prefix: &str,
        w3_session: &mut Option<W3SidecarSession>,
    ) -> Result<(FfnComponent, usize)> {
        ensure!(
            !self.lean.ffn_gate_up_alias || self.flashinfer_requested,
            "{}=1 requires ATLAS_PREFILL_FFN_FLASHINFER=1 (the merged operand it aliases)",
            crate::memlean::FFN_GATE_UP_ALIAS
        );
        let hidden = self.config.hidden_size;
        let intermediate = self.config.intermediate_size;
        // Allocation-receipt passes, summed over layers and logged with the
        // target-layers ledger phase.
        let weights_pass = spark_runtime::alloc_receipt::pass("ffn/weights");
        let weights = load_dense_ffn(
            self.store,
            layer_prefix,
            self.gpu,
            self.variant,
            self.absmax_k,
            self.quantize_k,
            self.stream,
            self.config,
        )?;
        let mut ffn_layer = DenseFfnLayer::new(weights, self.gpu)?;
        drop(weights_pass);
        if let Some(shared) = &self.shared_scratch {
            ffn_layer.adopt_shared_e2m1_scratch(shared)?;
            if layer == 0 {
                tracing::info!(
                    "MEMLEAN_SHARED_SCRATCH selector={}: one W4A4/FlashInfer activation arena for all {} dense FFN layers",
                    crate::memlean::SHARED_SCRATCH,
                    self.config.num_hidden_layers
                );
            }
        }

        let mut released_gate_up = 0usize;
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.flashinfer_requested {
            let _flashinfer_pass = spark_runtime::alloc_receipt::pass("ffn/flashinfer");
            ffn_layer.prepare_flashinfer_prefill(self.gpu, layer, hidden, intermediate)?;
            if self.lean.ffn_gate_up_alias {
                released_gate_up = self.alias_gate_up(&mut ffn_layer, layer_prefix)?;
            }
            if layer + 1 == self.config.num_hidden_layers {
                tracing::info!(
                    layers = self.config.num_hidden_layers,
                    hidden,
                    intermediate,
                    "FlashInfer SM121 FFN prefill construction complete"
                );
            }
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        ensure!(
            !self.flashinfer_requested,
            "ATLAS_PREFILL_FFN_FLASHINFER=1 requires a Linux CUDA build"
        );

        // Retain the incumbent optional W4 transforms. They remain fallback
        // operands for prefill and every route not covered by W3.
        if crate::layers::ffn_m16_transposed_enabled() {
            let _transposes_pass = spark_runtime::alloc_receipt::pass("ffn/transposes");
            let gate_t = ffn_layer.weights.gate_proj.transpose_for_gemm_cached(
                self.gpu,
                intermediate,
                hidden,
                &format!("L{layer}.mlp.gate_proj.t"),
            )?;
            let up_t = ffn_layer.weights.up_proj.transpose_for_gemm_cached(
                self.gpu,
                intermediate,
                hidden,
                &format!("L{layer}.mlp.up_proj.t"),
            )?;
            let down_t = ffn_layer.weights.down_proj.transpose_for_gemm_cached(
                self.gpu,
                hidden,
                intermediate,
                &format!("L{layer}.mlp.down_proj.t"),
            )?;
            ffn_layer.set_transposed_weights(gate_t, up_t, down_t);
            ffn_layer.alloc_splitk_workspace(
                self.gpu,
                hidden.max(intermediate) as u32,
                self.shared_splitk,
            )?;
            if layer == 0 {
                tracing::info!(
                    "Dense FFN M_TILE=16 transposed-weight path enabled \
                     (default on; disable with ATLAS_FFN_M16_TRANSPOSED=0): \
                     transposed gate/up/down per layer for w4a16_gemm_n128_m16"
                );
            }
        }

        if crate::layers::prefill_ffn_fp8_enabled() {
            let _fp8_pass = spark_runtime::alloc_receipt::pass("ffn/fp8-predequant");
            ffn_layer.predequant_for_prefill(self.gpu, hidden, intermediate, self.stream)?;
            if layer == 0 {
                tracing::info!(
                    "Dense FFN FP8 predequant prefill path enabled \
                     (ATLAS_FFN_PREDEQUANT_FP8=1): \
                     pre-dequanted gate/up/down per layer for fp8_gemm_t_m128"
                );
            }
        }

        let _w3_pass = w3_session
            .is_some()
            .then(|| spark_runtime::alloc_receipt::pass("ffn/w3"));
        if let Some(session) = w3_session.as_mut()
            && let Some(w3) = session.upload_layer(layer, self.gpu)?
        {
            let gemv = DenseFfnWeights {
                gate_proj: w3.gate,
                up_proj: w3.up,
                down_proj: w3.down,
            };
            let gemm_t = DenseFfnWeights {
                gate_proj: w3.gate_t,
                up_proj: w3.up_t,
                down_proj: w3.down_t,
            };
            ffn_layer.set_w3_weights(gemv, gemm_t);
            session.mark_installed(layer)?;
            tracing::info!(layer, "W3 FFN sidecar weights installed");
        }
        Ok((FfnComponent::Dense(ffn_layer), released_gate_up))
    }
}

impl DenseFfnLoadContext<'_> {
    /// `ATLAS_MEMLEAN_FFN_GATE_UP_ALIAS=1`: re-point gate/up at the merged
    /// FlashInfer operand (same bytes) and release the checkpoint packed
    /// weights from the store. The construction stream is drained first, the
    /// same quiescent-window pattern as the loader's other releases (BUG #29).
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    fn alias_gate_up(&self, ffn_layer: &mut DenseFfnLayer, layer_prefix: &str) -> Result<usize> {
        use spark_runtime::weights::WeightDtype;

        self.gpu.synchronize(self.stream)?;
        let (gate, up) = ffn_layer.alias_gate_up_to_flashinfer_merged()?;
        let mut released = 0usize;
        for (name, ptr) in [("gate_proj", gate), ("up_proj", up)] {
            let key = format!("{layer_prefix}.mlp.{name}.weight");
            let bytes =
                self.store
                    .release_consumed_alias(&key, ptr, WeightDtype::UInt8, self.gpu)?;
            ensure!(
                bytes != 0,
                "{key} is not the store allocation the merged gate/up operand replaced"
            );
            released += bytes;
        }
        Ok(released)
    }
}

pub(super) fn prepare_w3_session(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<W3SidecarSession>> {
    let Some(request) = W3SidecarRequest::from_env(config.num_hidden_layers)? else {
        return Ok(None);
    };
    gpu.kernel("w3a16_gemm", "w3a16_gemm_t_m32_n64")?;
    let layer_prefixes = (0..config.num_hidden_layers)
        .map(|layer| config.layer_prefix(layer))
        .collect::<Vec<_>>();
    W3SidecarSession::prepare(
        request,
        &layer_prefixes,
        config.hidden_size,
        config.intermediate_size,
    )
    .map(Some)
}

pub(super) fn finish_w3_session(session: Option<W3SidecarSession>) -> Result<()> {
    let Some(session) = session else {
        return Ok(());
    };
    let receipt = session.finish()?;
    ensure!(
        receipt.requested_layers == receipt.validated_layers
            && receipt.requested_layers == receipt.uploaded_layers
            && receipt.requested_layers == receipt.installed_layers,
        "W3 sidecar layer census differs at publication boundary"
    );
    ensure!(
        receipt.requested_count == receipt.validated_count
            && receipt.requested_count == receipt.uploaded_count
            && receipt.requested_count == receipt.installed_count
            && receipt.requested_count == receipt.requested_layers.len(),
        "W3 sidecar count census differs at publication boundary"
    );
    tracing::info!(
        layers = receipt.installed_count,
        size = receipt.size,
        device = receipt.device,
        inode = receipt.inode,
        "W3 FFN sidecar installation census complete"
    );
    Ok(())
}
