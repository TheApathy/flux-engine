# Flux Engine GitHub Workflows & Automation (`.github/`)

This directory defines Flux Engine's CI/CD pipelines, security gates, dependency automation, and hardened Claude Code workflows. Every third-party GitHub Action is pinned to an immutable 40-character commit SHA and runs with least-privilege `permissions`.

## Workflow Inventory

| File | Trigger | Purpose |
|---|---|---|
| [`workflows/ci.yml`](../.github/workflows/ci.yml) | `pull_request`, `push` (`main`) | `cargo fmt --check`, `cargo clippy --workspace --tests -- -D warnings`, SPDX license headers (`skywalking-eyes`), `typos`, CPU-only `cargo test --workspace --locked` (with synthesized `libcuda`/`libcublasLt`/`libcudart`/`libnccl` stubs), macOS Metal build/test (`macos-14`), and the `hygiene` job (`verify_no_leaks.py`, offline relative Markdown link verification via `scripts/check_markdown_links.py`, `gitleaks`, `actionlint`, shebang-aware `bash -n` + `py_compile`, Dockerfile AGPL compliance, and offline `pytest` suites). |
| [`workflows/kernel-compile.yml`](../.github/workflows/kernel-compile.yml) | `pull_request` & `push` (`main`) touching `kernels/**`, `crates/atlas-kernels/**`, or `verify_ptx_isa_contracts.py` + `workflow_dispatch` | GPU-less CUDA 13.0 (`nvidia/cuda:13.0.0-devel-ubuntu24.04`) `nvcc` PTX/SASS compilation and zero-spill ISA verification (`repro-kit/formal/verify_ptx_isa_contracts.py` on `sm_121a` plus flagship `sm_120a` zero-spill compile gate). |
| [`workflows/security.yml`](../.github/workflows/security.yml) | `Cargo.toml`/`Cargo.lock`/`deny.toml` changes + weekly Monday cron | `cargo-deny` (`advisories`, `licenses`, `sources`, `bans`). |
| [`workflows/codeql.yml`](../.github/workflows/codeql.yml) | `push` (`main`), `pull_request` (`main`), weekly Monday cron, `workflow_dispatch` | GitHub CodeQL SAST analysis (`actions` & `python` with `build-mode: none`), uploading security findings to GitHub Code Scanning. |
| [`workflows/scorecard.yml`](../.github/workflows/scorecard.yml) | `push` (`main`), weekly Monday cron, `branch_protection_rule`, `workflow_dispatch` | OpenSSF Scorecard supply-chain security posture analysis (`ossf/scorecard-action`), publishing results to the Scorecard API/badge and uploading SARIF alerts to GitHub Code Scanning. |
| [`workflows/docs.yml`](../.github/workflows/docs.yml) | `crates/**` / `Cargo.*` changes | Builds `cargo doc --workspace --no-deps` and uploads the `rustdoc-api` artifact. |
| [`workflows/file-size-cap.yml`](../.github/workflows/file-size-cap.yml) | `pull_request`, `push` (`main`) | Non-blocking advisory check flagging newly added/modified Rust files exceeding 500 LoC. |
| [`workflows/release.yml`](../.github/workflows/release.yml) | `vX.Y.Z*` tags + `workflow_dispatch` (default `dry_run: true`) | 4-job supply-chain pipeline (`preflight` → `build-artifacts` → isolated `slsa-provenance` → `publish-release`): enforces default-branch ancestry, `Cargo.toml` version parity, SSH signed-tag verification against [`.github/allowed_signers`](../.github/allowed_signers), `verify_no_leaks.py`, `typos`, SPDX headers, and AGPL-3.0 files; packages deterministic `git archive` CCS + license tarballs + SPDX 2.3 SBOM (`scripts/generate_cargo_spdx_sbom.py`) + `SHA256SUMS.txt`; signs `SHA256SUMS.txt` keyless via Sigstore Cosign (`SHA256SUMS.txt.sigstore.json`); generates GitHub Artifact Attestations (`actions/attest-build-provenance` + `actions/attest-sbom`) and isolated SLSA Build Level 3 provenance (`slsa-framework/slsa-github-generator`, `flux-engine-<tag>.intoto.jsonl`); publishes via `gh release create --verify-tag`. |
| [`workflows/container-release.yml`](../.github/workflows/container-release.yml) | `vX.Y.Z*` tags + `workflow_dispatch` (`target`: `both`/`sm120`/`gb10`, default `dry_run: true`) | Signed GHCR container publish (`ghcr.io/theapathy/flux-engine`): runs the same fail-closed `.github/allowed_signers` + default-branch + `Cargo.toml` preflight, builds `docker/sm120/Dockerfile` (`linux/amd64` on `ubuntu-latest`) and `docker/gb10/Dockerfile` (`linux/arm64` on `ubuntu-24.04-arm`) with BuildKit SBOM + `provenance: mode=max`, keyless-signs each published digest via Sigstore Cosign (OIDC + Rekor), pushes native GitHub Build Provenance attestations (`actions/attest-build-provenance`, `push-to-registry: true`), and scans published images with Trivy (`CRITICAL,HIGH`, `ignore-unfixed: true`) uploading SARIF to GitHub Security. |
| [`allowed_signers`](../.github/allowed_signers) | Consumed by `release.yml`, `container-release.yml` & `scripts/verify_release.sh` | Git SSH `allowed_signers` registry of trusted maintainer signing keys (`149847121+TheApathy@users.noreply.github.com`). Non-dry-run tag releases fail closed until an active key is registered. |
| [`ISSUE_TEMPLATE/hardware_report.yml`](../.github/ISSUE_TEMPLATE/hardware_report.yml) | New GitHub Issues | Structured Hardware Bring-Up / Benchmark Report form capturing GPU/VRAM, OS/arch, `nvidia-smi` + `nvcc --version`, Flux version/digest, model/quantization, exact launch command, measured results/logs, and privacy redaction confirmation. |
| [`workflows/claude-code-review.yml`](../.github/workflows/claude-code-review.yml) | `pull_request` (same-repo, non-draft, OWNER/MEMBER only) | Automated, strictly read-only Flux Engine code review using `anthropics/claude-code-action`. |
| [`workflows/claude.yml`](../.github/workflows/claude.yml) | `issue_comment`, `pull_request_review_comment`, `pull_request_review`, `issues` (`opened`) — OWNER/MEMBER `@claude` only | Interactive, strictly read-only `@claude` assistant for maintainers. |
| [`dependabot.yml`](../.github/dependabot.yml) | Weekly Monday 06:00 UTC | Grouped dependency update PRs for `cargo`, `github-actions` (preserving 40-char SHA pins), and `docker`. |

---

## Claude Code Threat Model & Anti-Prompt-Injection Controls

Public repositories face real prompt-injection, fork-PR secret-exfiltration, and quota-abuse threats when connecting an LLM to GitHub Actions. Both [`claude-code-review.yml`](../.github/workflows/claude-code-review.yml) and [`claude.yml`](../.github/workflows/claude.yml) enforce defense-in-depth across 10 independent layers:

1. **Zero `pull_request_target` Triggers:** Neither workflow uses `pull_request_target`, eliminating base-secret execution against untrusted fork checkouts ("pwn requests").
2. **Fork-PR Rejection (Double-Gated):** PR events enforce `github.event.pull_request.head.repo.full_name == github.repository` at the workflow expression layer AND `allow_fork_prs: "false"` inside `anthropics/claude-code-action` (which also queries the GitHub API on `issue_comment` events before checking out code).
3. **Maintainer-Only Human Trust Gate:** Every trigger requires `github.event.sender.type == 'User'` AND `author_association == 'OWNER' || author_association == 'MEMBER'`. `allowed_bots: ""` and `allowed_non_write_users: ""` are explicitly set, and `exclude_comments_by_actor: "*[bot]"` strips bot-authored comments from context. `issues` triggers only on `opened` (not `assigned`/`labeled`) so triaging an external contributor's issue cannot automatically feed an untrusted body into Claude.
4. **Strict Read-Only Repository Permissions:** Top-level `permissions: {}`; job grants only `contents: read`, `pull-requests: write`, `issues: write`, `id-token: write`. No `contents: write` (cannot push commits or alter workflows) and no `actions: read` (cannot ingest PR-controlled CI job logs).
5. **Hard Tool Allowlist (`--permission-mode default`):** Explicitly opts out of Claude Code's `auto` permission mode so `--allowedTools` acts as a hard boundary — any unlisted tool is denied by default.
6. **Explicit Anti-Exfiltration Denylist (`--disallowedTools`):** Blocks file edits (`Write`, `Edit`, `NotebookEdit`), network tools (`WebFetch`, `WebSearch`, `curl`, `wget`, `nc`), interpreter/shell escapes (`python`, `python3`, `node`, `bun`, `perl`, `ruby`, `sh`, `bash`), environment inspection (`env`, `printenv`, `export`, `set`, `cat /proc/*`), credential/admin CLIs (`gh secret`, `gh auth`, `gh api`, `gh workflow`, `gh pr merge`, `gh pr review --approve`), git mutations (`git push`, `git commit`, `git add`, `git remote`, `git config`), and GitHub MCP CI-log/file-ops tools.
7. **Subprocess Secret Scrubbing & Checkout Isolation:** `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB: "1"` strips Anthropic, cloud, and GitHub secrets from subprocess environments; `actions/checkout` sets `persist-credentials: false`; secrets are scoped exclusively to the pinned `claude-code-action` step and never enter `run:` shell commands.
8. **Zero Expression Injection:** Untrusted `${{ github.event.* }}` text fields (PR titles, bodies, comments, commit messages) are never interpolated into `run:` shell scripts or prompts.
9. **Sanitized Public Logs:** `show_full_output: "false"` and `display_report: "false"` prevent tool execution outputs from appearing in public GitHub Actions logs or Step Summaries.
10. **Non-Overridable Anti-Prompt-Injection Instructions + Turn Cap:** System/review prompts explicitly classify all PR/issue content, diffs, comments, and filenames as **untrusted data**, forbid revealing prompts/secrets/env/runner details, and bound every run with `--max-turns 15` and `timeout-minutes: 15`.

---

## Required Secrets & Repository Settings

### Secrets (Settings → Secrets and variables → Actions)

Configure **one** of the following repository secrets to activate `claude.yml` and `claude-code-review.yml` (if neither is set, both workflows skip gracefully with an informational notice instead of failing PR checks):

- **`ANTHROPIC_API_KEY`** *(recommended)*: Direct Anthropic API key scoped to a dedicated workspace with a monthly spend cap.
- **`CLAUDE_CODE_OAUTH_TOKEN`** *(alternative)*: Claude Code OAuth token from `claude setup-token`.

No additional secrets are required for `ci.yml`, `kernel-compile.yml`, `security.yml`, `codeql.yml`, `scorecard.yml`, `docs.yml`, `file-size-cap.yml`, `release.yml`, or `container-release.yml` (all use the built-in short-lived `GITHUB_TOKEN`).

### Recommended GitHub Repository Settings & Rulesets

- **Branch protection / Ruleset on `main`:**
  - Require pull requests, linear or reviewed merges, and passing `CI` (`fmt`, `clippy`, `license-headers`, `typos`, `test`, `hygiene`) checks before merge.
  - Enable **Require signed commits** on `main` (see [`../SIGNING.md`](../SIGNING.md) for YubiKey FIDO2 `ed25519-sk` and hardened `ed25519` setup).
- **Tag Ruleset on `v*`:**
  - Restrict creation/update/deletion of `v*` release tags to repository maintainers, and enable **Require signed commits/tags** so unsigned tags are rejected both at push time and by `release.yml`'s fail-closed `.github/allowed_signers` gate.
- **Fork pull request approval:** Under *Settings → Actions → General → Fork pull request workflows from outside collaborators*, select **Require approval for all outside collaborators**.
- **Never set `ACTIONS_STEP_DEBUG=true`** as a permanent repository secret on a public repository — `claude-code-action` automatically enables full debug logs when step debug mode is active.

---

## Release Supply-Chain Verification

Every tagged release published by [`workflows/release.yml`](../.github/workflows/release.yml) ships six artifacts:
1. `flux-engine-<tag>-src.tar.gz` — deterministic AGPL-3.0-only Complete Corresponding Source tarball
2. `flux-engine-<tag>-licenses.tar.gz` — `LICENSE`, `CREDITS.md`, `THIRD_PARTY_NOTICES.md`, `SOURCE_AND_LICENSES.md`, and `LICENSES/`
3. `flux-engine-<tag>.spdx.json` — deterministic SPDX 2.3 SBOM generated from `Cargo.toml` + `Cargo.lock` (attested via `actions/attest-sbom`)
4. `SHA256SUMS.txt` — SHA-256 manifest of the archives and SBOM
5. `SHA256SUMS.txt.sigstore.json` — Sigstore Cosign keyless OIDC/Rekor bundle for `SHA256SUMS.txt`
6. `flux-engine-<tag>.intoto.jsonl` — isolated SLSA Build Level 3 provenance statement

Verify all release archive layers (and optionally signed GHCR container images from [`workflows/container-release.yml`](../.github/workflows/container-release.yml)) in one command with [`scripts/verify_release.sh`](../scripts/verify_release.sh) (see [`../SIGNING.md`](../SIGNING.md) for manual commands):

```bash
# Verify downloaded GitHub Release archive bundle:
./scripts/verify_release.sh --tag v1.0.3 --dir ./release-v1.0.3 --require-all

# Verify a signed GHCR container image by immutable digest:
./scripts/verify_release.sh --image ghcr.io/theapathy/flux-engine@sha256:<digest> --require-all
```

---

## Local Pre-Push Validation

Run the same checks executed by `.github/workflows/ci.yml` locally:

```bash
# Privacy / PII leak check
python3 -B scripts/verify_no_leaks.py --root .

# Secret scan (no-git working tree; .github/gitleaks.toml ignores local target*/ dirs)
gitleaks detect --config .github/gitleaks.toml --source . --no-git --redact --exit-code 1

# Spellcheck
typos

# Offline Python & shell test suites
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  repro-kit/tests repro/tests \
  tests/test_indexed_loader_admission_source.py \
  tests/test_generate_cargo_spdx_sbom.py \
  tools/recipe_bench/generate_test.py
bash tools/recipe_bench/trap_test.sh

# GitHub Actions workflow linting (requires actionlint)
actionlint
```
