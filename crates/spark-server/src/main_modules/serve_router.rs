// SPDX-License-Identifier: AGPL-3.0-only

//! HTTP router construction, middleware stack, listener bind, and graceful
//! shutdown for `flux-server`.

use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use axum::routing::{get, post};

use crate::anthropic;
use crate::api;
use crate::main_modules::middleware::{
    fault_middleware, openai_observability_middleware, rate_limit_middleware,
    require_auth_middleware,
};

const SHUTDOWN_READINESS_DELAY_ENV: &str = "FLUX_SHUTDOWN_READINESS_DELAY_SECS";
const MAX_SHUTDOWN_READINESS_DELAY_SECS: u64 = 60;

fn parse_shutdown_readiness_delay(raw: Option<&str>) -> Result<std::time::Duration> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(std::time::Duration::ZERO);
    };
    let secs: u64 = raw.parse().map_err(|_| {
        anyhow::anyhow!(
            "{SHUTDOWN_READINESS_DELAY_ENV}={raw:?}: expected an integer in 0..={MAX_SHUTDOWN_READINESS_DELAY_SECS}"
        )
    })?;
    anyhow::ensure!(
        secs <= MAX_SHUTDOWN_READINESS_DELAY_SECS,
        "{SHUTDOWN_READINESS_DELAY_ENV}={secs}: must be in 0..={MAX_SHUTDOWN_READINESS_DELAY_SECS}"
    );
    Ok(std::time::Duration::from_secs(secs))
}

fn shutdown_readiness_delay_from_env() -> Result<std::time::Duration> {
    parse_shutdown_readiness_delay(std::env::var(SHUTDOWN_READINESS_DELAY_ENV).ok().as_deref())
}

async fn wait_for_delayed_shutdown(
    shutdown: impl std::future::Future<Output = ()>,
    readiness_delay: std::time::Duration,
) {
    shutdown.await;
    if !readiness_delay.is_zero() {
        tracing::info!(
            "Shutdown requested — failing /health and /ready for {}s before closing listener",
            readiness_delay.as_secs()
        );
        tokio::time::sleep(readiness_delay).await;
    }
}

const CORS_ALLOW_ORIGINS_ENV: &str = "FLUX_CORS_ALLOW_ORIGINS";
const LEGACY_CORS_ALLOW_ORIGINS_ENV: &str = "ATLAS_CORS_ALLOW_ORIGINS";

/// Returns `true` for loopback browser origins (`http(s)://localhost[:port]`,
/// `http(s)://127.0.0.1[:port]`, `http(s)://[::1][:port]`).
fn is_loopback_origin(origin: &str) -> bool {
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    let host = if let Some(after_bracket) = rest.strip_prefix('[') {
        let Some((ip, suffix)) = after_bracket.split_once(']') else {
            return false;
        };
        if !suffix.is_empty()
            && !suffix
                .strip_prefix(':')
                .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        {
            return false;
        }
        ip
    } else if let Some((h, port)) = rest.split_once(':') {
        if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        h
    } else {
        rest
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Build the CORS layer.
///
/// By default, only loopback origins are permitted so arbitrary websites open
/// in a local user's browser cannot send cross-origin requests to a keyless
/// localhost server. Set `FLUX_CORS_ALLOW_ORIGINS` (comma-separated origins,
/// or `*` for wildcard without credentials) to explicitly widen access.
fn build_cors_layer() -> tower_http::cors::CorsLayer {
    let configured = std::env::var(CORS_ALLOW_ORIGINS_ENV)
        .or_else(|_| std::env::var(LEGACY_CORS_ALLOW_ORIGINS_ENV))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let allow_origin = match configured.as_deref() {
        Some("*") => tower_http::cors::AllowOrigin::any(),
        Some(list) => {
            let allowed: Arc<Vec<String>> = Arc::new(
                list.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            );
            tower_http::cors::AllowOrigin::predicate(move |origin, _parts| {
                origin
                    .to_str()
                    .ok()
                    .is_some_and(|o| allowed.iter().any(|a| a == o))
            })
        }
        None => tower_http::cors::AllowOrigin::predicate(|origin, _parts| {
            origin.to_str().ok().is_some_and(is_loopback_origin)
        }),
    };

    tower_http::cors::CorsLayer::new()
        .allow_origin(allow_origin)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::PATCH,
            axum::http::Method::DELETE,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers(tower_http::cors::Any)
        // Browser clients may read the request-lifecycle warning header.
        .expose_headers([axum::http::HeaderName::from_static(
            crate::api::chat::CONTRACT_WARNING_HEADER,
        )])
}

/// Build the router once and serve on it for the process lifetime.
///
/// Stated on the `ModelHost`, NOT on an `Arc<AppState>`. The distinction is the
/// whole reason a swap can work: with a state baked in here, `host.publish` of
/// a new model would change nothing a request can see, and every handler would
/// keep serving the `AppState` captured at boot — whose scheduler the swap has
/// joined and whose weights it has freed. Each handler resolves the current
/// model per request through `CurrentModel` instead, and a request that has
/// already resolved one keeps it and finishes against the model it started on.
pub(crate) async fn build_and_serve(
    host: Arc<crate::main_modules::model_host::ModelHost>,
    bind: &str,
    port: u16,
) -> Result<()> {
    let readiness_delay = shutdown_readiness_delay_from_env()?;
    // The socket is bound for the process lifetime and a swap cannot move it,
    // so record where it lands: a recipe naming a different port would
    // otherwise serve on the old one with nothing saying so.
    host.set_bound(bind.to_string(), port);
    let cors = build_cors_layer();

    // Catch unexpected panics in HTTP handlers and convert them into HTTP 500
    // responses without hanging the client connection or leaking backtraces.
    let catch_panic = tower_http::catch_panic::CatchPanicLayer::new();

    let app = Router::new()
        .route("/", get(api::flux_ui::index_html))
        .route("/flux/modes", get(api::flux_ui::modes_json))
        .route("/recipes", get(api::flux_ui::modes_json))
        .route("/v1/flux/modes", get(api::flux_ui::modes_json))
        // Live request defaults. Only under /v1/, so `require_auth_middleware`
        // gates both methods whenever API keys are configured.
        .route(
            "/v1/flux/settings",
            get(crate::runtime_settings::handlers::get_settings)
                .patch(crate::runtime_settings::handlers::patch_settings),
        )
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/chat/completions/{id}", get(api::get_stored_completion))
        .route("/v1/completions", post(api::completions))
        .route("/v1/responses", post(api::responses_endpoint))
        .route(
            "/v1/responses/{id}",
            get(api::get_stored_response).delete(api::delete_stored_response),
        )
        .route(
            "/v1/responses/{id}/input_items",
            get(api::list_response_input_items),
        )
        .route("/v1/responses/{id}/cancel", post(api::cancel_response))
        .route("/v1/conversations", post(api::create_conversation))
        .route(
            "/v1/conversations/{id}",
            get(api::get_conversation)
                .post(api::update_conversation)
                .delete(api::delete_conversation),
        )
        .route(
            "/v1/conversations/{id}/items",
            post(api::add_conversation_items).get(api::list_conversation_items),
        )
        .route(
            "/v1/conversations/{id}/items/{item_id}",
            get(api::get_conversation_item).delete(api::delete_conversation_item),
        )
        .route("/api/tags", get(api::ollama_tags))
        .route("/api/show", post(api::ollama_show))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(api::count_tokens))
        .route("/v1/apply-template", post(api::apply_template))
        .route("/v1/decisions", post(api::decisions_endpoint))
        .route("/v1/memory", get(api::memory_telemetry))
        .route("/v1/experts/usage", get(api::experts_usage))
        .route("/v1/disagg/status", get(api::disagg_status))
        .route("/v1/disagg/prefill", post(api::disagg_prefill))
        .route("/v1/disagg/decode", post(api::disagg_decode))
        .route("/v1/disagg/generate", post(api::disagg_generate))
        .route("/v1/models", get(api::list_models))
        .route("/v1/models/{*model_id}", get(api::get_model))
        .route("/v1/embeddings", post(api::embeddings_stub))
        // 501 stubs: return an OpenAI-shaped error body so auto-probe
        // clients (Helicone, LangChain, Vercel AI SDK) fall back instead
        // of hanging on a silent 404.
        .route(
            "/v1/batches",
            post(api::batches_stub).get(api::batch_list_stub),
        )
        .route(
            "/v1/batches/{id}",
            get(api::batch_get_stub).delete(api::batch_get_stub),
        )
        .route("/v1/batches/{id}/cancel", post(api::batch_get_stub))
        .route("/v1/files", post(api::files_stub).get(api::files_stub))
        .route(
            "/v1/files/{id}",
            get(api::files_stub).delete(api::files_stub),
        )
        .route("/v1/files/{id}/content", get(api::files_stub))
        .route("/v1/audio/transcriptions", post(api::audio_stub))
        .route("/v1/audio/translations", post(api::audio_stub))
        .route("/v1/audio/speech", post(api::audio_stub))
        .route("/v1/images/generations", post(api::images_stub))
        .route("/v1/images/edits", post(api::images_stub))
        .route("/v1/images/variations", post(api::images_stub))
        .route("/v1/moderations", post(api::moderations_stub))
        .route("/v1/debug/prompt", post(api::dsv41::debug_prompt))
        .route("/tokenize", post(api::tokenize))
        .route("/detokenize", post(api::detokenize))
        .route("/health", get(api::health))
        .route("/ready", get(api::health))
        .route("/health/live", get(api::health_live))
        .route("/metrics", get(api::metrics_handler))
        // Body size limit. Default 128 MiB covers 1M-token prompts and
        // multi-image payloads; override via `FLUX_MAX_BODY_BYTES` or
        // `ATLAS_MAX_BODY_BYTES` (in bytes). Lowering it protects against
        // DoS attempts that send oversized payloads to burn CPU on JSON
        // parsing + tokenization before the model even sees them.
        .layer(axum::extract::DefaultBodyLimit::max(
            std::env::var("FLUX_MAX_BODY_BYTES")
                .or_else(|_| std::env::var("ATLAS_MAX_BODY_BYTES"))
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(128 * 1024 * 1024),
        ))
        .layer(axum::middleware::from_fn(fault_middleware))
        .layer(axum::middleware::from_fn(
            crate::main_modules::mem_watch::mem_floor_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            host.clone(),
            rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            host.clone(),
            require_auth_middleware,
        ))
        .layer(axum::middleware::from_fn(openai_observability_middleware))
        .layer(axum::middleware::from_fn(
            crate::main_modules::byte_count::byte_count_middleware,
        ))
        .layer(cors)
        .layer(catch_panic)
        // `host.clone()`, not a move: a layer that captured the only handle
        // would keep it alive for the router's lifetime, and the same mistake
        // made against `Arc<AppState>` is what wedges a swap — a state bound
        // into a layer never reaches a strong count of 1, so the drain window
        // expires, the scheduler never learns to stop, and the join never
        // returns. The host is process-scoped and is meant to be held; the
        // model it hands out is not.
        .with_state(host.clone());

    // Readiness is asserted by `serve_load::load_model`, at the moment the
    // model and its scheduler are actually up. It used to be set here, which
    // was right exactly once: the router is built once at boot and a swap
    // never rebuilds it, so a flag set here would stay true across a swap that
    // FAILED and false for one that succeeded.

    let addr = format!("{bind}:{port}");
    if bind == "0.0.0.0" {
        tracing::warn!(
            "Flux is listening on {addr} — reachable from any host on the network. \
             If this machine is on a shared LAN or has a public IP, pass \
             --bind 127.0.0.1 (or set --require-auth behind a TLS reverse proxy \
             and firewall) before accepting traffic."
        );
    } else if bind == "127.0.0.1" || bind == "localhost" || bind == "::1" {
        // Default loopback bind protects keyless servers from LAN exposure;
        // log an explicit hint so container operators running `--network host`
        // know how to opt into LAN binding intentionally.
        tracing::info!(
            "API reachable only from this machine (loopback). To expose on the \
             LAN pass --bind 0.0.0.0; combine with --require-auth and \
             --auth-tokens-file behind TLS for non-trusted networks."
        );
    }
    tracing::info!("Listening on {addr}");
    spark_runtime::progress::phase(11, "listening");
    crate::load_debug::note_bind();
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    spark_runtime::progress::ready(port);
    // Listener up: the load succeeded, so the load-phase capture ends here.
    crate::load_debug::finish_load();
    crate::term_color::print_ready_banner(&host, bind, port);
    // Disable Nagle's algorithm (TCP_NODELAY) on every accepted connection.
    // SSE token streaming sends one small `data:` frame per committed token
    // (or burst of K speculative tokens); with Nagle enabled, delayed ACKs
    // (~40 ms) stall consecutive frames and artificially inflate client-side
    // inter-token latency. TCP_NODELAY flushes each SSE frame immediately
    // without changing the byte stream.
    use axum::serve::ListenerExt;
    let listener = listener.tap_io(|tcp_stream| {
        if let Err(err) = tcp_stream.set_nodelay(true) {
            tracing::warn!("failed to set TCP_NODELAY on incoming connection: {err:#}");
        }
    });
    // `into_make_service_with_connect_info` exposes the socket peer addr
    // to extractors — needed by `rate_limit_middleware` when the caller
    // didn't send X-Forwarded-For.
    crate::tui::shutdown::disarm_startup_escape();
    let (listener_close_tx, listener_close_rx) = tokio::sync::watch::channel(false);
    let shutdown_delay_task = tokio::spawn(async move {
        wait_for_delayed_shutdown(crate::tui::shutdown::wait(), readiness_delay).await;
        let _ = listener_close_tx.send(true);
    });
    let wait_listener_close = || {
        let mut rx = listener_close_rx.clone();
        async move {
            let _ = rx.wait_for(|&closed| closed).await;
        }
    };
    let serving = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(wait_listener_close());
    // Axum's graceful future can otherwise wait forever for a request body
    // or socket write. Returning on this deadline lets serve() retire the
    // scheduler; remaining AppState holders then hit its bounded fail-stop.
    let drain_result = super::http_drain::bounded(
        async { serving.await.map_err(anyhow::Error::from) },
        wait_listener_close(),
        super::http_drain::HTTP_DRAIN_GRACE,
    )
    .await;
    shutdown_delay_task.abort();
    drain_result?;

    crate::tui::init::flush_tee();
    tracing::info!("Shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{parse_shutdown_readiness_delay, wait_for_delayed_shutdown};

    #[test]
    fn shutdown_readiness_delay_validates_bounds() {
        assert_eq!(
            parse_shutdown_readiness_delay(None).unwrap(),
            Duration::ZERO
        );
        assert_eq!(
            parse_shutdown_readiness_delay(Some("")).unwrap(),
            Duration::ZERO
        );
        assert_eq!(
            parse_shutdown_readiness_delay(Some("0")).unwrap(),
            Duration::ZERO
        );
        assert_eq!(
            parse_shutdown_readiness_delay(Some(" 15 ")).unwrap(),
            Duration::from_secs(15)
        );
        assert_eq!(
            parse_shutdown_readiness_delay(Some("60")).unwrap(),
            Duration::from_secs(60)
        );
        for invalid in ["61", "-1", "5s", "abc"] {
            assert!(
                parse_shutdown_readiness_delay(Some(invalid)).is_err(),
                "{invalid} must be rejected"
            );
        }
    }

    #[tokio::test]
    async fn delayed_shutdown_waits_for_readiness_propagation_before_closing_listener() {
        let started = Instant::now();
        wait_for_delayed_shutdown(std::future::ready(()), Duration::ZERO).await;
        assert!(started.elapsed() < Duration::from_millis(50));

        let started = Instant::now();
        wait_for_delayed_shutdown(std::future::ready(()), Duration::from_millis(40)).await;
        assert!(started.elapsed() >= Duration::from_millis(30));
    }

    #[test]
    fn loopback_cors_predicate_allows_only_local_origins() {
        for ok in [
            "http://localhost",
            "http://localhost:3000",
            "https://localhost:8443",
            "http://127.0.0.1:8000",
            "http://[::1]:8000",
            "https://[::1]",
        ] {
            assert!(super::is_loopback_origin(ok), "expected loopback: {ok}");
        }
        for bad in [
            "https://evil.example.com",
            "http://localhost.evil.com",
            "http://127.0.0.1.evil.com",
            "http://localhost:abc",
            "null",
            "file://",
        ] {
            assert!(!super::is_loopback_origin(bad), "expected reject: {bad}");
        }
    }

    #[test]
    fn ready_route_is_wired_to_readiness_probe() {
        let source = include_str!("serve_router.rs");
        assert!(source.contains(".route(\"/ready\", get(api::health))"));
    }
}
