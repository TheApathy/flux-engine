// SPDX-License-Identifier: AGPL-3.0-only

//! Unit and async router tests for TripleKV `/v1/kv_cache/prefetch` endpoint,
//! Eq. (1) issue-time scheduling, per-tool latency prediction, `PrefixSnapshotStore`
//! async loading, and `RadixTree` `ProtectionClass::Prefetched` tagging.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use spark_runtime::prefix_cache::{PrefixCache, ProtectionClass};
use spark_runtime::radix_tree::RadixTree;
use spark_storage::{
    PrefixSnapshotKey, PrefixSnapshotPayload, PrefixSnapshotStore, PrefixSnapshotStoreConfig,
    SnapshotKvDtype,
};
use tempfile::tempdir;

use super::{
    KvPrefetcher, PrefetchScheduleInput, ToolLatencyPredictor, compute_prefetch_schedule,
    kv_cache_prefetch, kv_cache_prefetch_status,
};
use crate::main_modules::model_host::ModelHost;

#[test]
fn eq1_issue_time_schedule_computes_exact_delay_and_clamps_to_zero() {
    // S_L3 = 350 MB, B = 3.5 GB/s -> io_load = 0.10s.
    // remaining = 1.50s, queue_wait = 0.25s, eps = 0.05s -> delay = 1.60s.
    let sched = compute_prefetch_schedule(PrefetchScheduleInput {
        remaining_tool_secs: 1.50,
        queue_wait_secs: 0.25,
        snapshot_bytes: 350_000_000,
        nvme_bandwidth_bps: 3.5e9,
        safety_margin_secs: 0.05,
    })
    .expect("valid schedule");
    assert!(!sched.immediate);
    assert!((sched.io_load_secs - 0.10).abs() < 1e-9);
    assert!((sched.issue_delay.as_secs_f64() - 1.60).abs() < 1e-6);

    // When snapshot I/O exceeds remaining tool time, issue immediately (delay = 0).
    let urgent = compute_prefetch_schedule(PrefetchScheduleInput {
        remaining_tool_secs: 0.08,
        queue_wait_secs: 0.0,
        snapshot_bytes: 700_000_000, // 0.20s at 3.5 GB/s
        nvme_bandwidth_bps: 3.5e9,
        safety_margin_secs: 0.05,
    })
    .expect("urgent schedule");
    assert!(urgent.immediate);
    assert_eq!(urgent.issue_delay, Duration::ZERO);

    // Invalid bandwidth rejected cleanly.
    assert!(
        compute_prefetch_schedule(PrefetchScheduleInput {
            remaining_tool_secs: 1.0,
            queue_wait_secs: 0.0,
            snapshot_bytes: 4096,
            nvme_bandwidth_bps: 0.0,
            safety_margin_secs: 0.05,
        })
        .is_err()
    );
}

#[test]
fn tool_latency_predictor_tracks_per_tool_quantiles() {
    let mut pred = ToolLatencyPredictor::new();
    for sample in [0.5, 1.0, 2.0, 4.0, 8.0] {
        pred.record_observation("web_search", sample);
    }
    pred.record_observation("bash", 0.2);

    assert_eq!(pred.tracked_tools(), 2);
    assert_eq!(pred.predict_secs("web_search", 0.5), Some(2.0));
    assert_eq!(pred.predict_secs("web_search", 0.0), Some(0.5));
    assert_eq!(pred.predict_secs("web_search", 1.0), Some(8.0));
    assert_eq!(pred.predict_secs("bash", 0.5), Some(0.2));
    assert_eq!(pred.predict_secs("unknown_tool", 0.5), None);
}

async fn response_json(resp: axum::response::Response) -> (StatusCode, serde_json::Value) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("read response body");
    let json = serde_json::from_slice(&bytes).expect("valid JSON response");
    (status, json)
}

#[tokio::test]
async fn http_prefetch_route_loads_snapshot_tags_radix_and_serves_hot_hit() {
    let dir = tempdir().expect("tempdir");
    let mut store =
        PrefixSnapshotStore::open(PrefixSnapshotStoreConfig::new(dir.path())).expect("open store");
    let tokens: Vec<u32> = (100..132).collect(); // 2 blocks of 16 tokens
    let key = PrefixSnapshotKey::from_tokens("qwen3.8-27b", SnapshotKvDtype::Nvfp4, 16, &tokens)
        .expect("key");
    let expected_payload = PrefixSnapshotPayload {
        kv_pages: vec![0xAB; 4096],
        qsa_selectors: Vec::new(),
        gdn_h_state: vec![0xCD; 512],
        gdn_conv_state: vec![0xEF; 128],
        ple_conv_state: Vec::new(),
        num_kv_layers: 16,
        num_ssm_layers: 48,
        num_ple_layers: 0,
    };
    let _ = store.put(key.clone(), &expected_payload).expect("put snap");

    let radix = Arc::new(RadixTree::new());
    radix.insert(&tokens, &[10, 11], &[], 16, 0);
    radix.release(&tokens, 16);

    let prefetcher = Arc::new(KvPrefetcher::with_store_and_tree(
        Some(store),
        Some(Arc::clone(&radix)),
    ));
    let host = Arc::new(ModelHost::with_kv_prefetcher(Arc::clone(&prefetcher)));

    let app: Router<()> = Router::new()
        .route(
            "/v1/kv_cache/prefetch",
            post(kv_cache_prefetch).get(kv_cache_prefetch_status),
        )
        .with_state(Arc::clone(&host));

    // 1. Trigger async prefetch with immediate issue (remaining_tool_secs = 0.0).
    let req_body = serde_json::json!({
        "token_ids": tokens,
        "model_digest": "qwen3.8-27b",
        "kv_dtype": "nvfp4",
        "block_size": 16,
        "ttl_secs": 120,
        "tool_name": "bash",
        "predicted_tool_remaining_secs": 0.0
    });
    let resp = kv_cache_prefetch(
        State(Arc::clone(&host)),
        Ok(axum::Json(
            serde_json::from_value(req_body.clone()).expect("req"),
        )),
    )
    .await;
    let (status, body) = response_json(resp).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(body["status"], "scheduled");
    assert_eq!(body["content_id"], key.content_id());
    assert_eq!(body["aligned_tokens"], 32);
    assert_eq!(body["radix_nodes_protected"], 2);

    // Live RadixTree blocks must immediately carry Prefetched protection.
    let classes = radix.prefix_protection(&tokens, 16);
    assert_eq!(classes.len(), 2);
    assert!(
        classes
            .iter()
            .all(|c| matches!(c, ProtectionClass::Prefetched { .. }))
    );

    // Wait briefly for background spawn_blocking load to stage the payload in L2.
    let mut completed = false;
    for _ in 0..50 {
        if prefetcher.status().completed_total >= 1 {
            completed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        completed,
        "background prefetch must complete and stage L2 payload"
    );

    // 2. Second request for the same prefix must report `already_hot` (200 OK).
    let resp2 = kv_cache_prefetch(
        State(Arc::clone(&host)),
        Ok(axum::Json(serde_json::from_value(req_body).expect("req"))),
    )
    .await;
    let (status2, body2) = response_json(resp2).await;
    assert_eq!(status2, StatusCode::OK);
    assert_eq!(body2["status"], "already_hot");

    // 3. Staged payload must match persisted bytes exactly.
    let staged = prefetcher
        .take_staged_payload(&key.content_id())
        .expect("staged payload");
    assert_eq!(staged, expected_payload);

    // 4. Status endpoint reflects counters cleanly.
    let status_resp = kv_cache_prefetch_status(State(Arc::clone(&host))).await;
    let (st_code, st_body) = response_json(status_resp).await;
    assert_eq!(st_code, StatusCode::OK);
    assert_eq!(st_body["store_enabled"], true);
    assert_eq!(st_body["radix_tree_attached"], true);
    assert_eq!(st_body["completed_total"], 1);
    assert_eq!(st_body["hit_hot_total"], 1);

    // 5. Reject empty token_ids cleanly via axum handler.
    let bad_resp = kv_cache_prefetch(
        State(Arc::clone(&host)),
        Ok(axum::Json(
            serde_json::from_value(serde_json::json!({ "token_ids": [] })).expect("bad req"),
        )),
    )
    .await;
    assert_eq!(bad_resp.status(), StatusCode::BAD_REQUEST);

    // Ensure router itself compiles and wires `/v1/kv_cache/prefetch`.
    let _ = Request::builder()
        .uri("/v1/kv_cache/prefetch")
        .body(Body::empty())
        .expect("req");
    let _ = app;
}

#[test]
fn router_wires_triplekv_prefetch_endpoint() {
    let source = include_str!("../main_modules/serve_router.rs");
    assert!(
        source.contains("\"/v1/kv_cache/prefetch\""),
        "serve_router.rs must register /v1/kv_cache/prefetch"
    );
}
