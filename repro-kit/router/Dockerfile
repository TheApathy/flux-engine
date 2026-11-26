# syntax=docker/dockerfile:1
FROM haproxy:2.8

ARG FLUX_ENGINE_VERSION="1.0.2"
ARG FLUX_SOURCE_REVISION="unknown"
ARG FLUX_SOURCE_TREE="unknown"

LABEL org.opencontainers.image.title="Flux Router (Multi-Spark HAProxy + Lua Front Door)" \
      org.opencontainers.image.description="Least-outstanding capacity-aware HAProxy 2.8 + Lua 5.3 front door for multi-Spark Flux Engine deployments." \
      org.opencontainers.image.version="${FLUX_ENGINE_VERSION}" \
      org.opencontainers.image.ref.name="v1.0.2" \
      org.opencontainers.image.revision="${FLUX_SOURCE_REVISION}" \
      org.opencontainers.image.vendor="TheApathy" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/TheApathy/flux-engine" \
      com.theapathy.flux.version="${FLUX_ENGINE_VERSION}" \
      com.theapathy.flux.source-tree="${FLUX_SOURCE_TREE}" \
      com.theapathy.flux.component="flux-router" \
      com.theapathy.flux.privacy="privacy-first-airgapped"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/flux/router /run/flux-router \
    && chown -R haproxy:haproxy /run/flux-router \
    && chmod 0700 /run/flux-router

COPY --chmod=0755 flux-router /opt/flux/router/flux-router
COPY --chmod=0644 flux_state.lua /opt/flux/router/flux_state.lua
COPY --chmod=0644 README.md /opt/flux/router/README.md
RUN ln -sf /opt/flux/router/flux-router /usr/local/bin/flux-router

ENV FLUX_ENGINE_VERSION="${FLUX_ENGINE_VERSION}" \
    PYTHONUNBUFFERED="1"

EXPOSE 8888
USER haproxy
ENTRYPOINT ["/usr/local/bin/flux-router"]
