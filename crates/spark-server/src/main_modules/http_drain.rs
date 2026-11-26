// SPDX-License-Identifier: AGPL-3.0-only

//! Bound HTTP graceful shutdown before waiting for scheduler ownership.

use std::future::Future;
use std::time::Duration;

use anyhow::Result;

pub(super) const HTTP_DRAIN_GRACE: Duration = Duration::from_secs(30);

pub(super) async fn bounded(
    serving: impl Future<Output = Result<()>>,
    shutdown: impl Future<Output = ()>,
    grace: Duration,
) -> Result<()> {
    tokio::pin!(serving, shutdown);
    tokio::select! {
        result = &mut serving => result,
        () = &mut shutdown => {
            match tokio::time::timeout(grace, &mut serving).await {
                Ok(result) => result,
                Err(_) => anyhow::bail!(
                    "HTTP graceful shutdown exceeded {}s; proceeding to bounded scheduler retirement",
                    grace.as_secs_f64(),
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stalled_http_body_cannot_prevent_scheduler_retirement() {
        let began = std::time::Instant::now();
        let result = bounded(
            std::future::pending(),
            std::future::ready(()),
            Duration::from_millis(10),
        )
        .await;
        // This is the same ordering as serve(): any HTTP result, including a
        // deadline, reaches retirement before its primary error is returned.
        let retirement_reached = result.is_err();
        assert!(retirement_reached);
        assert!(began.elapsed() < Duration::from_secs(1));
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("scheduler retirement")
        );
    }

    #[tokio::test]
    async fn ordinary_server_error_remains_primary() {
        let error = bounded(
            std::future::ready(Err(anyhow::anyhow!("listener failed"))),
            std::future::pending(),
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "listener failed");
    }

    #[tokio::test]
    async fn graceful_completion_is_accepted_before_the_deadline() {
        bounded(
            async {
                tokio::task::yield_now().await;
                Ok(())
            },
            std::future::ready(()),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
    }
}
