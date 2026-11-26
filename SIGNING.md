<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Release Signing, Supply-Chain Provenance & Verification (`SIGNING.md`)

**Flux Engine** (`TheApathy/flux-engine`) uses layered cryptographic signing and
build provenance so users, downstream packagers, and security auditors can
verify both **who authored a release** (maintainer Git SSH signatures) and
**how the release artifacts were produced** (isolated CI build provenance,
keyless Sigstore transparency logging, and deterministic SBOMs).

---

## 1. Supply-Chain & Origin-Attribution Model

Every non-dry-run source release produced by `.github/workflows/release.yml`
and container release produced by `.github/workflows/container-release.yml`
ships layered cryptographic verification:

| Layer | Artifact / Anchor | What It Proves |
|---|---|---|
| **1. Signed Git Commits & Tags** | Git SSH signature + [`.github/allowed_signers`](.github/allowed_signers) | Cryptographic maintainer authorship bound to `149847121+TheApathy@users.noreply.github.com`. Unsigned or untrusted tags fail closed in the release preflight gate. |
| **2. SHA-256 Digest Manifest** | `SHA256SUMS.txt` | Bit-exact integrity of the release source archive, license bundle, and SPDX SBOM. |
| **3. SLSA Build Level 3 Provenance** | `flux-engine-<tag>.intoto.jsonl` | Generated in an isolated reusable workflow (`slsa-framework/slsa-github-generator`) proving the release archives were built unmodified from the exact signed tag on `github.com/TheApathy/flux-engine`. |
| **4. Sigstore Cosign Keyless Signatures** | `SHA256SUMS.txt.sigstore.json` + OCI image signatures on `ghcr.io/theapathy/flux-engine` | GitHub OIDC keyless signatures over release checksums and GHCR container digests, recorded in the public Rekor transparency log. |
| **5. GitHub Artifact Attestations** | GitHub Attestations API (`actions/attest-build-provenance` & `actions/attest-sbom`) | Native GitHub-hosted build provenance and SBOM attestations for release archives and GHCR container images, verifiable via `gh attestation verify`. |
| **6. Deterministic SPDX 2.3 SBOM** | `flux-engine-<tag>.spdx.json` (+ Buildx OCI SBOM) | Complete Rust dependency inventory generated deterministically from `Cargo.toml` and `Cargo.lock` (`scripts/generate_cargo_spdx_sbom.py`). |

---

## 2. Verifying a Checkout or Release (Users & Auditors)

### 2.1 Active Release Signing Keys & Local Commit/Tag Verification

Verify commits and release tags against [`.github/allowed_signers`](.github/allowed_signers)
(the machine-readable source of truth for active and historical signing keys):

| Role | Principal | Key Type | Public Key Fingerprint (`ssh-keygen -lf`) |
|---|---|---|---|
| **Primary** | `149847121+TheApathy@users.noreply.github.com` | `ssh-ed25519` | `SHA256:rQNuiiNhQcVwJoB4oyAr0MKx3cOWzSekkUXyRRTLCEY` |
| **Backup (Offline)** | `149847121+TheApathy@users.noreply.github.com` | `ssh-ed25519` | `SHA256:BYCRavG2b6vIi8S3xCRlfHVQMHJyQDjC5zDcKHMuAqw` |

```bash
# Verify HEAD commit signature:
git -c gpg.ssh.allowedSignersFile=.github/allowed_signers verify-commit HEAD

# Verify a release tag (e.g. v1.0.3):
git -c gpg.ssh.allowedSignersFile=.github/allowed_signers verify-tag v1.0.3
```

### 2.2 One-Command Release Verification (`scripts/verify_release.sh`)

Download the release assets (`flux-engine-<tag>-src.tar.gz`,
`flux-engine-<tag>-licenses.tar.gz`, `flux-engine-<tag>.spdx.json`,
`flux-engine-<tag>.intoto.jsonl`, `SHA256SUMS.txt`, and
`SHA256SUMS.txt.sigstore.json`) into a directory and run:

```bash
./scripts/verify_release.sh --tag v1.0.3 --dir /path/to/release-assets

# Strict mode (fail if any optional verifier CLI is not installed):
./scripts/verify_release.sh --tag v1.0.3 --dir /path/to/release-assets --require-all
```

### 2.3 Manual Verification Commands

Each verification step can also be run directly without the helper script:

```bash
TAG="v1.0.3"

# 1. Verify SHA-256 digests:
sha256sum -c SHA256SUMS.txt

# 2. Verify SLSA Build Level 3 provenance (requires slsa-verifier):
slsa-verifier verify-artifact \
  "flux-engine-${TAG}-src.tar.gz" \
  "flux-engine-${TAG}-licenses.tar.gz" \
  "flux-engine-${TAG}.spdx.json" \
  SHA256SUMS.txt \
  --provenance-path "flux-engine-${TAG}.intoto.jsonl" \
  --source-uri github.com/TheApathy/flux-engine \
  --source-tag "${TAG}"

# 3. Verify Sigstore Cosign keyless bundle against Rekor (requires cosign):
cosign verify-blob \
  --bundle SHA256SUMS.txt.sigstore.json \
  --certificate-identity-regexp '^https://github\.com/TheApathy/flux-engine/\.github/workflows/release\.yml@refs/tags/' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  SHA256SUMS.txt

# 4. Verify GitHub build & SBOM attestations (requires GitHub CLI `gh`):
gh attestation verify "flux-engine-${TAG}-src.tar.gz" --repo TheApathy/flux-engine
gh attestation verify "flux-engine-${TAG}-src.tar.gz" --predicate-type https://spdx.dev/Document --repo TheApathy/flux-engine
```

### 2.4 Verifying Signed GHCR Container Images (`ghcr.io/theapathy/flux-engine`)

Official container images built by `.github/workflows/container-release.yml`
(`sm120` `linux/amd64` and `gb10` `linux/arm64`) are signed keyless with
Sigstore Cosign and attested with GitHub Build Provenance + Buildx SBOMs.
Always verify by immutable digest (`@sha256:<digest>`):

```bash
IMAGE="ghcr.io/theapathy/flux-engine@sha256:<digest>"

# One-command verifier (runs both Cosign and GitHub Attestation checks):
./scripts/verify_release.sh --image "${IMAGE}"

# Or run each check directly:
cosign verify \
  --certificate-identity-regexp '^https://github\.com/TheApathy/flux-engine/\.github/workflows/container-release\.yml@refs/tags/' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  "${IMAGE}"

gh attestation verify "oci://${IMAGE}" --repo TheApathy/flux-engine
```

---

## 3. Maintainer Key Setup & Repository Governance

> [!CAUTION]
> **Privacy & Pseudonymity Hygiene:** Always use the pseudonymous GitHub noreply
> address `149847121+TheApathy@users.noreply.github.com` as the SSH key comment
> (`-C`) and Git `user.email`. **Never** embed personal email addresses, legal
> names, or workstation hostnames in signing keys, commit headers, or
> `.github/allowed_signers`.

### 3.1 Generate a Dedicated Signing Key

Choose **Option A** if you have a FIDO2 hardware security key, or **Option B**
for a hardened local software key:

#### Option A (Recommended — Hardware FIDO2 Security Key, e.g. YubiKey 5)

The private key never leaves the hardware token and requires physical touch +
PIN verification on every signature:

```bash
ssh-keygen -t ed25519-sk \
  -O resident \
  -O verify-required \
  -C "149847121+TheApathy@users.noreply.github.com" \
  -f ~/.ssh/flux_signing_ed25519_sk
chmod 600 ~/.ssh/flux_signing_ed25519_sk
```

> **Backup recommendation:** Provision a second FIDO2 hardware key stored
> offline and register both public keys in GitHub and `.github/allowed_signers`
> so key loss never breaks release continuity.

#### Option B (Hardened Software Key — No Hardware Token Required)

Uses 100 KDF rounds (`-a 100`) to protect the private key with a strong
passphrase:

```bash
ssh-keygen -t ed25519 \
  -a 100 \
  -C "149847121+TheApathy@users.noreply.github.com" \
  -f ~/.ssh/flux_signing_ed25519
chmod 600 ~/.ssh/flux_signing_ed25519
```

Store an encrypted offline backup of `~/.ssh/flux_signing_ed25519` (never commit
private keys to any repository or cloud sync folder).

### 3.2 Configure Git for Automatic SSH Signing

Point `user.signingkey` at the `.pub` file for Option A (`flux_signing_ed25519_sk.pub`) or Option B (`flux_signing_ed25519.pub`):

```bash
git config --local user.name "TheApathy"
git config --local user.email "149847121+TheApathy@users.noreply.github.com"
git config --local gpg.format ssh
git config --local user.signingkey ~/.ssh/flux_signing_ed25519_sk.pub   # or ~/.ssh/flux_signing_ed25519.pub
git config --local commit.gpgsign true
git config --local tag.gpgsign true
git config --local gpg.ssh.allowedSignersFile .github/allowed_signers
```

### 3.3 Register the Public Key in GitHub & `.github/allowed_signers`

1. **GitHub account:** Open **GitHub Settings → SSH and GPG keys → New SSH key**,
   select **Key type: Signing Key** (not Authentication Key), and paste the
   contents of your `.pub` file.
2. **Repository trust anchor:** Add one active line to
   [`.github/allowed_signers`](.github/allowed_signers) using the `git`
   namespace:
   ```text
   149847121+TheApathy@users.noreply.github.com namespaces="git" sk-ssh-ed25519@openssh.com AAAA...
   ```
   *(Or `ssh-ed25519 AAAA...` for Option B.)*
3. **Verify locally before tagging:**
   ```bash
   git commit --allow-empty -S -m "chore: verify SSH signing setup"
   git -c gpg.ssh.allowedSignersFile=.github/allowed_signers verify-commit HEAD
   ```

### 3.4 Recommended GitHub Repository Rulesets

To enforce origin integrity server-side on `TheApathy/flux-engine`:

1. **Branch Ruleset (`main`):**
   - Enable **Require signed commits**.
   - Enable **Block force pushes** and **Restrict deletions**.
   - Enable **Require status checks to pass** (`ci.yml` clippy/test/typos + `security.yml`).
2. **Tag Ruleset (`v*`):**
   - Enable **Require signed commits/tags** so only signed maintainer tags can
     trigger `.github/workflows/release.yml`.
   - Enable **Restrict updates and deletions** so published release tags are
     immutable.

---

## 4. Key Rotation & Compromise Response

- **Planned rotation:** Add the new public key to GitHub Signing Keys and
  `.github/allowed_signers`, sign the rotation commit with the **outgoing** key
  (establishing a verifiable chain of custody), then update `user.signingkey`.
  Keep historical keys in `.github/allowed_signers` with `valid-before="YYYYMMDD"`
  so older release tags remain verifiable.
- **Suspected key compromise:** Immediately revoke the key in GitHub Settings,
  annotate the entry in `.github/allowed_signers` with `valid-before` prior to
  the compromise window, and report via [`.github/SECURITY.md`](.github/SECURITY.md).
