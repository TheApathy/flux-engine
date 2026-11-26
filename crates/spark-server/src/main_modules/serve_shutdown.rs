// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};

fn is_http_drain_timeout(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        let msg = cause.to_string();
        msg.starts_with("HTTP graceful shutdown exceeded ")
            && msg.contains("proceeding to bounded scheduler retirement")
    })
}

pub(super) fn resolve_serve_result(
    serve_result: Result<()>,
    scheduler_join: std::result::Result<Result<()>, tokio::task::JoinError>,
) -> Result<()> {
    match serve_result {
        Err(primary) => match scheduler_join {
            Ok(Ok(())) if is_http_drain_timeout(&primary) => {
                tracing::warn!(
                    "HTTP graceful drain timed out ({primary:#}), but scheduler retirement completed cleanly; exiting 0"
                );
                crate::tui::init::flush_tee();
                Ok(())
            }
            Ok(Ok(())) => Err(primary),
            Ok(Err(error)) => {
                tracing::error!("scheduler shutdown also failed after server error: {error:#}");
                Err(primary)
            }
            Err(error) => {
                tracing::error!("scheduler join task also failed after server error: {error:#}");
                Err(primary)
            }
        },
        Ok(()) => {
            scheduler_join.context("scheduler join task failed")??;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_serve_result;

    #[test]
    fn server_error_remains_primary_when_scheduler_also_fails() {
        let error = resolve_serve_result(
            Err(anyhow::anyhow!("primary server failure")),
            Ok(Err(anyhow::anyhow!("secondary scheduler failure"))),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(error, "primary server failure");
    }

    #[test]
    fn scheduler_error_propagates_after_clean_server_shutdown() {
        let error = resolve_serve_result(
            Ok(()),
            Ok(Err(anyhow::anyhow!("scheduler shutdown failure"))),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(error, "scheduler shutdown failure");
    }

    #[test]
    fn http_drain_timeout_with_clean_scheduler_retirement_exits_ok() {
        let timeout = anyhow::anyhow!(
            "HTTP graceful shutdown exceeded 30s; proceeding to bounded scheduler retirement"
        );
        resolve_serve_result(Err(timeout), Ok(Ok(())))
            .expect("clean scheduler retirement after HTTP drain timeout must exit 0");
    }

    #[test]
    fn non_drain_server_error_still_fails_when_scheduler_retires_cleanly() {
        let error =
            resolve_serve_result(Err(anyhow::anyhow!("Address already in use")), Ok(Ok(())))
                .unwrap_err()
                .to_string();
        assert_eq!(error, "Address already in use");
    }

    #[test]
    fn http_drain_timeout_still_errors_when_scheduler_retirement_fails() {
        let timeout = anyhow::anyhow!(
            "HTTP graceful shutdown exceeded 30s; proceeding to bounded scheduler retirement"
        );
        let error = resolve_serve_result(Err(timeout), Ok(Err(anyhow::anyhow!("scheduler panic"))))
            .unwrap_err()
            .to_string();
        assert!(error.contains("HTTP graceful shutdown exceeded"));
    }

    #[test]
    fn scheduler_spawn_is_after_all_fallible_startup_sites() {
        // Startup moved into `serve_load::load_model` (shared by first boot and
        // every swap); auth and the signal listeners are process-scoped and
        // are built in `serve` before any load, so they cannot follow a spawn.
        let source = include_str!("serve_load.rs");
        let spawn = source
            .find("let scheduler_handle = std::thread::spawn")
            .unwrap();
        let fallible = "serve_phases::resolve_tool_call_parser";
        assert!(source.find(fallible).unwrap() < spawn, "{fallible}");
        let after = &source[spawn..source.find("Ok(Some(Prepared").unwrap()];
        assert!(
            !after.contains(")?") && !after.contains("?;"),
            "a fallible `?` follows the scheduler spawn"
        );
    }
}
