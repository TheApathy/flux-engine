// SPDX-License-Identifier: AGPL-3.0-only

//! `GET` / `PATCH /v1/flux/settings`.
//!
//! Auth: `/v1/*` is already behind `require_auth_middleware` when the server
//! has API keys, so both methods need the key then. With no key configured,
//! GET (read-only, no secrets) is open like `/v1/flux/modes`, and PATCH is
//! accepted only with the local admin token and straight from this machine
//! (see [`WriteGate`]). The peer is
//! the TCP socket address, never a forwarded header. Under Docker port
//! publishing, host requests arrive from the bridge gateway, not loopback:
//! set an API key, or run `docker exec <container> flux set ...`. Behind a
//! local proxy or tunnel the loopback rule is no boundary at all; an API key
//! is required there.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde_json::Value;

use super::{RuntimeSettings, SettingError};
use crate::main_modules::model_host::{CurrentModel, ModelHost};

/// The largest PATCH body accepted (the full key set is a few KB).
const MAX_PATCH_BYTES: usize = 64 * 1024;

pub const LOOPBACK_ONLY_MESSAGE: &str = "changing settings without an API key is only allowed \
from this machine (loopback): set an API key (--require-auth) or use `docker exec <container> \
flux set key=value`";

pub async fn get_settings(CurrentModel(state): CurrentModel) -> Response {
    Json(state.runtime.describe()).into_response()
}

pub async fn patch_settings(
    State(host): State<Arc<ModelHost>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    CurrentModel(state): CurrentModel,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    // With keys configured the middleware has already checked this one.
    let gate = WriteGate {
        api_key_configured: host.auth().is_some(),
        peer,
        headers: &headers,
        bound: host.bound().map(|(addr, _)| addr),
        admin_token: super::local_admin::current(),
    };
    if let Err(resp) = gate.authorize() {
        return resp;
    }
    // Off the async workers: a configured settings file is fsynced.
    match tokio::task::spawn_blocking(move || patch(&state.runtime, &body, peer)).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!("PATCH /v1/flux/settings task failed: {e}");
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "settings_not_saved",
                None,
                "the settings change failed; see the server log",
            )
        }
    }
}

/// Who may write. With an API key the key is the boundary. Without one a
/// write must carry the local admin token (`X-Flux-Admin-Token`, see
/// [`super::local_admin`]): nothing reaching the server over a network,
/// proxy or tunnel has it. On top of that it must come straight from this
/// machine:
/// - the TCP peer is loopback;
/// - no proxy header (`Forwarded`, `X-Forwarded-For`, `X-Real-IP`): behind
///   a local proxy or tunnel (nginx, cloudflared, tailscale serve, ssh -L)
///   every peer is loopback, so loopback proves nothing there;
/// - no foreign `Origin` and a local `Host` (localhost, 127.0.0.1, [::1] or
///   the bound address): a DNS-rebinding page in a local browser sends its
///   own host name in both.
pub(super) struct WriteGate<'a> {
    pub api_key_configured: bool,
    pub peer: SocketAddr,
    pub headers: &'a HeaderMap,
    /// The `--bind` address.
    pub bound: Option<String>,
    /// This process's admin token (`None`: it could not be written, so
    /// keyless writes are refused).
    pub admin_token: Option<&'a str>,
}

/// Headers a proxy, CDN or tunnel adds. Any of them on a keyless write means
/// the loopback peer is not the real client.
const PROXY_HEADERS: [&str; 7] = [
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-real-ip",
    "via",
    "cf-connecting-ip",
    "true-client-ip",
];

impl WriteGate<'_> {
    #[allow(clippy::result_large_err)]
    pub(super) fn authorize(&self) -> Result<(), Response> {
        if self.api_key_configured {
            return Ok(());
        }
        let refusal = if !is_loopback(self.peer.ip()) {
            Some(("loopback_only", LOOPBACK_ONLY_MESSAGE))
        } else if PROXY_HEADERS.iter().any(|h| self.headers.contains_key(*h)) {
            Some(("proxied_request", PROXIED_MESSAGE))
        } else if self
            .headers
            .get("origin")
            // A non-UTF-8 Origin is foreign.
            .is_some_and(|origin| !origin.to_str().is_ok_and(origin_is_local))
        {
            Some(("cross_origin", FOREIGN_HOST_MESSAGE))
        } else if !self
            .header("host")
            .is_some_and(|host| host_is_local(host, self.bound.as_deref()))
        {
            Some(("host_not_allowed", FOREIGN_HOST_MESSAGE))
        } else {
            match (
                self.admin_token,
                self.headers.get(super::local_admin::HEADER),
            ) {
                (None, _) => Some(("admin_token_unavailable", ADMIN_TOKEN_MESSAGE)),
                (Some(_), None) => Some(("admin_token_required", ADMIN_TOKEN_MESSAGE)),
                (Some(expected), Some(presented)) => {
                    (!super::local_admin::matches(expected, presented.as_bytes()))
                        .then_some(("admin_token_invalid", ADMIN_TOKEN_MESSAGE))
                }
            }
        };
        let Some((code, message)) = refusal else {
            return Ok(());
        };
        tracing::warn!(
            client = %self.peer,
            code,
            "Refused PATCH /v1/flux/settings without an API key"
        );
        Err(error(StatusCode::FORBIDDEN, code, None, message))
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }
}

pub const ADMIN_TOKEN_MESSAGE: &str = "changing settings without an API key needs this server's \
local admin token, which only `flux set` inside the container can read: use `docker exec \
<container> flux set key=value`, or set an API key (--require-auth)";

pub const PROXIED_MESSAGE: &str = "this request came through a proxy or tunnel (it carries a \
forwarding header such as Forwarded, X-Forwarded-For or Via), so a loopback peer does not prove \
it is local: set an API key (--require-auth) to change settings through a proxy";

pub const FOREIGN_HOST_MESSAGE: &str = "changing settings without an API key needs a local Host \
(localhost, 127.0.0.1, [::1] or the bound address) and no foreign Origin: set an API key \
(--require-auth) or use `docker exec <container> flux set key=value`";

fn is_loopback(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    }
}

/// The host part of a `Host` / authority (`[::1]:8899` -> `[::1]`),
/// lowercased.
fn host_part(authority: &str) -> String {
    let a = authority.trim();
    let host = if a.starts_with('[') {
        a.find(']').map_or(a, |end| &a[..=end])
    } else {
        a.rsplit_once(':').map_or(a, |(h, _)| h)
    };
    host.to_ascii_lowercase()
}

fn is_local_name(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

fn host_is_local(host: &str, bound: Option<&str>) -> bool {
    let host = host_part(host);
    if is_local_name(&host) {
        return true;
    }
    // The bound address, unless it is a wildcard (which names no host).
    bound
        .map(|b| {
            let b = b.trim().to_ascii_lowercase();
            if b.contains(':') && !b.starts_with('[') {
                format!("[{b}]")
            } else {
                b
            }
        })
        .is_some_and(|b| !matches!(b.as_str(), "0.0.0.0" | "[::]") && b == host)
}

fn origin_is_local(origin: &str) -> bool {
    let Some((scheme, authority)) = origin.trim().split_once("://") else {
        return false; // includes the opaque "null" origin
    };
    matches!(scheme.to_ascii_lowercase().as_str(), "http" | "https")
        && is_local_name(&host_part(authority))
}

/// Apply a PATCH body and answer with the changes plus the full settings.
pub(super) fn patch(settings: &RuntimeSettings, body: &[u8], peer: SocketAddr) -> Response {
    if body.len() > MAX_PATCH_BYTES {
        return error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_value",
            None,
            "settings body is too large",
        );
    }
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_value",
                None,
                &format!("invalid JSON: {e}"),
            );
        }
    };
    match settings.apply_patch(&parsed) {
        Ok(changes) => {
            for c in &changes {
                tracing::info!(
                    key = %c.key,
                    old = %c.old,
                    new = %c.new,
                    client = %peer,
                    "Runtime setting changed (applies to new requests)"
                );
            }
            let mut out = settings.describe();
            out["changed"] = serde_json::to_value(&changes).unwrap_or(Value::Null);
            Json(out).into_response()
        }
        Err(e) => {
            tracing::warn!(client = %peer, error = e.message(), "Refused PATCH /v1/flux/settings");
            setting_error(&e)
        }
    }
}

pub(super) fn setting_error(e: &SettingError) -> Response {
    match e {
        SettingError::Invalid { key, message } => {
            let code = if message.starts_with("unknown setting") {
                "unknown_setting"
            } else {
                "invalid_value"
            };
            error(StatusCode::BAD_REQUEST, code, Some(key), message)
        }
        SettingError::RestartRequired { key, message } => {
            error(StatusCode::CONFLICT, "restart_required", Some(key), message)
        }
        SettingError::Internal { message } => {
            // The detail (a file path and OS error) goes to the log only.
            tracing::error!("runtime settings: {message}");
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "settings_not_saved",
                None,
                "the change could not be saved, so it was not applied (see the server log)",
            )
        }
    }
}

fn error(status: StatusCode, code: &str, param: Option<&str>, message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param.filter(|p| !p.is_empty()),
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}
