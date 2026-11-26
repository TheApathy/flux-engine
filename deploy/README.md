<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Production Deployment & Operations Templates (`deploy/`)

Hardened reference manifests for running **Flux Engine** in production:

| File | Purpose |
|---|---|
| [`flux-engine.service`](flux-engine.service) | Hardened `systemd` service unit (non-root `flux` user, loopback `127.0.0.1:8888`, read-only `/srv/models`, graceful `SIGTERM` drain). |
| [`k8s-deployment.yaml`](k8s-deployment.yaml) | Kubernetes `Deployment` + `ClusterIP` `Service` with `/health` startup/readiness/liveness probes, `/metrics` Prometheus annotations, non-root `securityContext`, and GPU resource limits. |
| [`grafana-dashboard.json`](grafana-dashboard.json) | Importable Grafana dashboard for `/metrics` (active/queued requests, prefill & decode throughput, KV pool utilization, TTFT p50/p95/p99). |

> **Security tip:** Before deploying container images to production, verify the
> immutable GHCR digest with `./scripts/verify_release.sh --image ghcr.io/theapathy/flux-engine@sha256:<digest>`
> (see [`../SIGNING.md`](../SIGNING.md)) and always set `--api-key` / `FLUX_API_KEY`
> when binding beyond `127.0.0.1`.
