#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# One-command release & container verifier for Flux Engine (`TheApathy/flux-engine`).
#
# Verifies release archive bundles across up to 5 independent layers:
#   (a) SHA-256 archive & SBOM checksums (`sha256sum -c SHA256SUMS.txt`)
#   (b) Git SSH signed release tag (`git verify-tag` via `.github/allowed_signers`)
#   (c) SLSA Build Level 3 provenance (`slsa-verifier verify-artifact`)
#   (d) Sigstore Cosign keyless Rekor bundle (`cosign verify-blob`)
#   (e) GitHub Artifact Attestations (`gh attestation verify`)
#
# And optionally verifies signed GHCR container images (`--image <oci-ref>`):
#   (f) Sigstore Cosign keyless OCI signature (`cosign verify`)
#   (g) GitHub Native Build Provenance attestation (`gh attestation verify oci://<oci-ref>`)

set -euo pipefail

REPO_SLUG="TheApathy/flux-engine"
OIDC_ISSUER="https://token.actions.githubusercontent.com"
CERT_IDENTITY_RE="^https://github\\.com/TheApathy/flux-engine/\\.github/workflows/release\\.yml@refs/tags/"
CONTAINER_CERT_IDENTITY_RE="^https://github\\.com/TheApathy/flux-engine/\\.github/workflows/container-release\\.yml@refs/tags/"

TAG=""
DIR="."
DIR_EXPLICIT=0
ALLOWED_SIGNERS=""
IMAGE_REF=""
REQUIRE_ALL=0

usage() {
  cat <<'EOF'
Usage: verify_release.sh [OPTIONS]

Verify a Flux Engine release bundle (checksums, signed git tag, SLSA Build
Level 3 provenance, Sigstore Cosign keyless bundle, and GitHub Attestations)
and/or a signed GHCR container image (`--image`).

Options:
  --tag <TAG>                 Release tag (e.g. v1.0.3). Auto-detected from
                              flux-engine-v*-src.tar.gz in --dir if omitted.
  --dir <PATH>                Directory containing downloaded release assets
                              (default: current directory).
  --allowed-signers <PATH>    Path to git allowed_signers file
                              (default: <git-root>/.github/allowed_signers).
  --image <OCI-REF>           Verify a signed GHCR container image (e.g.
                              ghcr.io/theapathy/flux-engine@sha256:<digest>).
                              When passed without --tag or --dir, runs in
                              container-only verification mode.
  --require-all               Fail closed if any optional verification tool
                              (slsa-verifier, cosign, gh) is missing or if
                              not running inside a git checkout.
  -h, --help                  Show this help message.

Examples:
  # Verify all downloaded release assets in ./release-v1.0.3
  ./scripts/verify_release.sh --tag v1.0.3 --dir ./release-v1.0.3

  # Strict mode requiring every verifier CLI (slsa-verifier, cosign, gh, git)
  ./scripts/verify_release.sh --tag v1.0.3 --dir ./release-v1.0.3 --require-all

  # Verify a signed GHCR container image by immutable digest
  ./scripts/verify_release.sh --image ghcr.io/theapathy/flux-engine@sha256:<digest>

  # Verify both a release bundle and its signed GHCR container image
  ./scripts/verify_release.sh --tag v1.0.3 --dir ./release-v1.0.3 \
    --image ghcr.io/theapathy/flux-engine@sha256:<digest> --require-all
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      [[ $# -ge 2 ]] || { echo "ERROR: --tag requires a value" >&2; exit 2; }
      TAG="$2"
      shift 2
      ;;
    --dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --dir requires a value" >&2; exit 2; }
      DIR="$2"
      DIR_EXPLICIT=1
      shift 2
      ;;
    --allowed-signers)
      [[ $# -ge 2 ]] || { echo "ERROR: --allowed-signers requires a value" >&2; exit 2; }
      ALLOWED_SIGNERS="$2"
      shift 2
      ;;
    --image)
      [[ $# -ge 2 ]] || { echo "ERROR: --image requires a value" >&2; exit 2; }
      IMAGE_REF="$2"
      shift 2
      ;;
    --require-all)
      REQUIRE_ALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Normalize optional oci:// prefix on --image so cosign receives the bare OCI
# reference while gh attestation verify receives oci://<ref>.
if [[ -n "$IMAGE_REF" ]]; then
  IMAGE_REF="${IMAGE_REF#oci://}"
fi

VERIFY_RELEASE=1
VERIFY_IMAGE=0
if [[ -n "$IMAGE_REF" ]]; then
  VERIFY_IMAGE=1
  if [[ -z "$TAG" && $DIR_EXPLICIT -eq 0 ]]; then
    VERIFY_RELEASE=0
  fi
fi

if [[ $VERIFY_RELEASE -eq 1 ]]; then
  if [[ ! -d "$DIR" ]]; then
    echo "ERROR: Release asset directory does not exist: $DIR" >&2
    exit 1
  fi
  DIR="$(cd "$DIR" && pwd)"

  if [[ -z "$TAG" ]]; then
    shopt -s nullglob
    candidates=("$DIR"/flux-engine-v*-src.tar.gz)
    shopt -u nullglob
    if [[ ${#candidates[@]} -eq 1 ]]; then
      base="$(basename "${candidates[0]}")"
      TAG="${base#flux-engine-}"
      TAG="${TAG%-src.tar.gz}"
    else
      echo "ERROR: Could not auto-detect release tag in $DIR; pass --tag <vX.Y.Z> or --image <oci-ref>." >&2
      exit 2
    fi
  fi
fi

# Resolve default allowed_signers path when inside a git checkout.
GIT_ROOT=""
if [[ $VERIFY_RELEASE -eq 1 ]]; then
  if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
    GIT_ROOT="$(git rev-parse --show-toplevel)"
  fi
  if [[ -z "$ALLOWED_SIGNERS" ]]; then
    if [[ -n "$GIT_ROOT" && -f "$GIT_ROOT/.github/allowed_signers" ]]; then
      ALLOWED_SIGNERS="$GIT_ROOT/.github/allowed_signers"
    elif [[ -f "$DIR/.github/allowed_signers" ]]; then
      ALLOWED_SIGNERS="$DIR/.github/allowed_signers"
    fi
  fi
fi

echo "=== Flux Engine Release Verifier ==="
echo "Repository:      $REPO_SLUG"
if [[ $VERIFY_RELEASE -eq 1 ]]; then
  echo "Release tag:     $TAG"
  echo "Asset directory: $DIR"
fi
if [[ $VERIFY_IMAGE -eq 1 ]]; then
  echo "Container image: $IMAGE_REF"
fi
echo "Strict mode:     $([[ $REQUIRE_ALL -eq 1 ]] && echo 'on (--require-all)' || echo 'off')"
echo ""
echo "Tool availability:"
if [[ $VERIFY_RELEASE -eq 1 ]]; then
  TOOLS=(sha256sum git slsa-verifier cosign gh)
else
  TOOLS=(cosign gh)
fi
for tool in "${TOOLS[@]}"; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf "  [present] %s (%s)\n" "$tool" "$(command -v "$tool")"
  else
    printf "  [missing] %s\n" "$tool"
  fi
done
echo ""

PASS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

record_pass() {
  echo "[PASS] $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

record_skip() {
  if [[ $REQUIRE_ALL -eq 1 ]]; then
    echo "[FAIL] $1 (required by --require-all)" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  else
    echo "[SKIP] $1"
    SKIP_COUNT=$((SKIP_COUNT + 1))
  fi
}

record_fail() {
  echo "[FAIL] $1" >&2
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

if [[ $VERIFY_RELEASE -eq 1 ]]; then
  # (a) SHA-256 checksums
  CHECKSUMS_FILE="$DIR/SHA256SUMS.txt"
  if [[ ! -f "$CHECKSUMS_FILE" ]]; then
    record_fail "(a) SHA256SUMS.txt not found in $DIR"
  else
    if (cd "$DIR" && sha256sum -c SHA256SUMS.txt); then
      record_pass "(a) sha256sum -c SHA256SUMS.txt"
    else
      record_fail "(a) sha256sum -c SHA256SUMS.txt failed"
    fi
  fi

  # (b) Git SSH signed tag verification
  if [[ -z "$GIT_ROOT" ]]; then
    record_skip "(b) Git signed-tag check skipped (not inside a git checkout)"
  elif [[ -z "$ALLOWED_SIGNERS" || ! -f "$ALLOWED_SIGNERS" ]]; then
    record_skip "(b) Git signed-tag check skipped (.github/allowed_signers not found)"
  elif ! grep -Eq '^[[:space:]]*[^#[:space:]]' "$ALLOWED_SIGNERS"; then
    record_skip "(b) Git signed-tag check skipped (no active signer keys registered in $ALLOWED_SIGNERS yet)"
  elif ! git -C "$GIT_ROOT" rev-parse -q --verify "refs/tags/$TAG" >/dev/null 2>&1; then
    record_skip "(b) Git signed-tag check skipped (tag '$TAG' not fetched locally; run: git fetch --tags)"
  else
    if git -C "$GIT_ROOT" -c "gpg.ssh.allowedSignersFile=$ALLOWED_SIGNERS" verify-tag "$TAG"; then
      record_pass "(b) git verify-tag $TAG (against $ALLOWED_SIGNERS)"
    else
      record_fail "(b) git verify-tag $TAG failed"
    fi
  fi

  # Build list of release subject files to verify with SLSA & GitHub Attestations.
  SUBJECTS=(
    "$DIR/flux-engine-${TAG}-src.tar.gz"
    "$DIR/flux-engine-${TAG}-licenses.tar.gz"
    "$DIR/flux-engine-${TAG}.spdx.json"
    "$CHECKSUMS_FILE"
  )

  # (c) SLSA Build Level 3 provenance
  PROV_FILE="$DIR/flux-engine-${TAG}.intoto.jsonl"
  if ! command -v slsa-verifier >/dev/null 2>&1; then
    record_skip "(c) SLSA L3 provenance check skipped (install slsa-verifier: https://github.com/slsa-framework/slsa-verifier)"
  elif [[ ! -f "$PROV_FILE" ]]; then
    record_fail "(c) SLSA L3 provenance file missing: $PROV_FILE"
  else
    slsa_ok=1
    for artifact in "${SUBJECTS[@]}"; do
      if [[ ! -f "$artifact" ]]; then
        echo "  missing artifact for SLSA verification: $artifact" >&2
        slsa_ok=0
        continue
      fi
      if ! slsa-verifier verify-artifact "$artifact" \
        --provenance-path "$PROV_FILE" \
        --source-uri "github.com/${REPO_SLUG}" \
        --source-tag "$TAG"; then
        slsa_ok=0
      fi
    done
    if [[ $slsa_ok -eq 1 ]]; then
      record_pass "(c) slsa-verifier verified SLSA L3 provenance for all subjects"
    else
      record_fail "(c) slsa-verifier provenance verification failed"
    fi
  fi

  # (d) Sigstore Cosign keyless Rekor bundle
  COSIGN_BUNDLE="$DIR/SHA256SUMS.txt.sigstore.json"
  if ! command -v cosign >/dev/null 2>&1; then
    record_skip "(d) Cosign keyless check skipped (install cosign: https://docs.sigstore.dev/cosign/system_config/installation/)"
  elif [[ ! -f "$COSIGN_BUNDLE" ]]; then
    record_fail "(d) Cosign bundle missing: $COSIGN_BUNDLE"
  elif [[ ! -f "$CHECKSUMS_FILE" ]]; then
    record_fail "(d) SHA256SUMS.txt missing for Cosign verification"
  else
    if cosign verify-blob \
      --bundle "$COSIGN_BUNDLE" \
      --certificate-identity-regexp "$CERT_IDENTITY_RE" \
      --certificate-oidc-issuer "$OIDC_ISSUER" \
      "$CHECKSUMS_FILE"; then
      record_pass "(d) cosign verify-blob verified SHA256SUMS.txt keyless Sigstore bundle"
    else
      record_fail "(d) cosign verify-blob failed"
    fi
  fi

  # (e) GitHub Artifact Attestations
  if ! command -v gh >/dev/null 2>&1; then
    record_skip "(e) GitHub Attestation check skipped (install GitHub CLI >= 2.49.0: https://cli.github.com/)"
  elif ! gh attestation --help >/dev/null 2>&1; then
    record_skip "(e) GitHub Attestation check skipped (installed gh CLI lacks 'attestation' subcommand; upgrade to gh >= 2.49.0)"
  else
    gh_ok=1
    for artifact in "${SUBJECTS[@]}"; do
      if [[ ! -f "$artifact" ]]; then
        echo "  missing artifact for gh attestation verify: $artifact" >&2
        gh_ok=0
        continue
      fi
      if ! gh attestation verify "$artifact" --repo "$REPO_SLUG"; then
        gh_ok=0
      fi
    done
    if [[ $gh_ok -eq 1 ]]; then
      record_pass "(e) gh attestation verify passed for all subjects"
    else
      record_fail "(e) gh attestation verify failed"
    fi
  fi
fi

if [[ $VERIFY_IMAGE -eq 1 ]]; then
  if [[ "$IMAGE_REF" != *"@sha256:"* ]]; then
    echo "[NOTE] Container reference '$IMAGE_REF' is not digest-pinned; prefer '@sha256:<digest>' for immutable verification."
  fi

  # (f) Sigstore Cosign keyless OCI container signature
  if ! command -v cosign >/dev/null 2>&1; then
    record_skip "(f) Cosign OCI image check skipped (install cosign: https://docs.sigstore.dev/cosign/system_config/installation/)"
  else
    if cosign verify \
      --certificate-identity-regexp "$CONTAINER_CERT_IDENTITY_RE" \
      --certificate-oidc-issuer "$OIDC_ISSUER" \
      "$IMAGE_REF"; then
      record_pass "(f) cosign verify verified keyless OCI signature for $IMAGE_REF"
    else
      record_fail "(f) cosign verify failed for $IMAGE_REF"
    fi
  fi

  # (g) GitHub Native Build Provenance attestation for OCI container image
  if ! command -v gh >/dev/null 2>&1; then
    record_skip "(g) GitHub OCI attestation check skipped (install GitHub CLI >= 2.49.0: https://cli.github.com/)"
  elif ! gh attestation --help >/dev/null 2>&1; then
    record_skip "(g) GitHub OCI attestation check skipped (installed gh CLI lacks 'attestation' subcommand; upgrade to gh >= 2.49.0)"
  else
    if gh attestation verify "oci://${IMAGE_REF}" --repo "$REPO_SLUG"; then
      record_pass "(g) gh attestation verify passed for oci://${IMAGE_REF}"
    else
      record_fail "(g) gh attestation verify failed for oci://${IMAGE_REF}"
    fi
  fi
fi

echo ""
echo "=== Summary: pass=$PASS_COUNT skip=$SKIP_COUNT fail=$FAIL_COUNT ==="
if [[ $FAIL_COUNT -gt 0 ]]; then
  exit 1
fi
exit 0
