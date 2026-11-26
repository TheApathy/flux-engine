// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn test_empty_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(EmptyLayerState);
    assert!(state.as_any().downcast_ref::<EmptyLayerState>().is_some());
    assert!(state.as_any().downcast_ref::<SsmLayerState>().is_none());
}

#[test]
fn test_ssm_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        wy17_kv_retain: None,
        wy17_gate_retain: None,
        h_is_f16: false,
    });
    let ssm = state.as_any().downcast_ref::<SsmLayerState>().unwrap();
    assert_eq!(ssm.h_state.0, 0x1000);
    assert_eq!(ssm.conv_state.0, 0x2000);
}

#[test]
fn test_ssm_layer_state_mut() {
    let mut state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        wy17_kv_retain: None,
        wy17_gate_retain: None,
        h_is_f16: false,
    });
    let ssm = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
    ssm.h_state = DevicePtr(0x3000);
    assert_eq!(ssm.h_state.0, 0x3000);
}

#[test]
fn test_forward_context_lifetime() {
    use spark_runtime::gpu::mock::MockGpuBackend;

    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let buffers = BufferArena::new(&config, 1, 4096, 16, 1, &gpu).unwrap();

    let ctx = ForwardContext {
        buffers: &buffers,
        gpu: &gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        ddtree_parent_ids_dev: None,
        tree_aware_attn: None,
        ssm_multi_seq_ptr_table_override: None,
        self_spec_sparse_draft: None,
        ffn_defer: None,
    };

    assert_eq!(ctx.config.hidden_size, 2048);
    assert_eq!(ctx.config.num_hidden_layers, 48);
}

#[test]
fn debug_sync_allowed_requires_debug_or_trace() {
    assert!(!debug_sync_allowed(false, false, false));
    assert!(debug_sync_allowed(true, false, false));
    assert!(debug_sync_allowed(false, true, false));
    assert!(debug_sync_allowed(true, true, false));
}

#[test]
fn debug_sync_allowed_refuses_during_graph_capture() {
    // RED case this fix closes: debug/trace logging being on must never
    // permit a blocking host sync while a CUDA graph capture is open —
    // that is exactly what wedged the stream in the Qwen4 PLE suffix-graph
    // bug (decode_qwen4_ple_graph.rs's layers-1..N region).
    assert!(!debug_sync_allowed(true, false, true));
    assert!(!debug_sync_allowed(false, true, true));
    assert!(!debug_sync_allowed(true, true, true));
}

#[test]
fn debug_sync_allowed_false_outside_capture_with_nothing_requested() {
    assert!(!debug_sync_allowed(false, false, true));
}

#[test]
fn decode_debug_dumps_enabled_reads_the_explicit_opt_in_once() {
    // decode_debug_dumps_enabled() caches via OnceLock, so this only
    // asserts it returns a stable bool and does not panic — the env var's
    // actual value in the test process is whatever the harness set (or
    // unset), and re-reading it mid-process after the first call is by
    // design not supported (see the doc comment: ambient, mutable config
    // is exactly the failure mode this replaces).
    let first = debug_sync_allowed(decode_debug_dumps_enabled(), false, false);
    let second = debug_sync_allowed(decode_debug_dumps_enabled(), false, false);
    assert_eq!(first, second);
}
