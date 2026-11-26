<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Container &amp; Multi-Node Docker Definitions (<code>docker/</code>)</b></p>

# Container & Multi-Node Docker Definitions (`docker/`)

Dockerfiles and multi-Spark deployment configurations for building and running **Flux Engine** on `aarch64` NVIDIA DGX Spark (`GB10` / `sm_121a`) and `x86_64` discrete GPUs.

---

## Contents

| Path | Purpose |
|---|---|
| [`docker-guide.md`](docker-guide.md) | Complete guide to building per-model and universal container images, volume mounts (`/models`), and RoCEv2 multi-node networking. |
| [`gb10/`](gb10/) | NVIDIA DGX Spark (`GB10` `sm_121a`, CUDA 13.0 `aarch64`) container build contexts. |
| [`sm120/`](sm120/) | Discrete NVIDIA Blackwell SM120 (`sm_120a`, CUDA 13.0 `linux/amd64`) container build context and guide. |
| [`../repro-kit/`](../repro-kit/README.md) | Turnkey OCI Reproducibility Kit (`ghcr.io/theapathy/flux-engine:gb10`) with `flux-bench` and `docker-compose.spark.yml`. |
