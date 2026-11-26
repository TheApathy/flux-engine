# SPDX-License-Identifier: AGPL-3.0-only
FROM gcr.io/oss-fuzz-base/base-builder-rust@sha256:8c2ccb74bb4b63c10c47e8f91c6633181fa00ca4cc4a4751dd390443e6cc76aa
COPY . $SRC/flux-engine
WORKDIR $SRC/flux-engine
COPY .clusterfuzzlite/build.sh $SRC/
