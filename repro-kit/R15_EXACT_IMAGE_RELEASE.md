# R15 exact-image release wrapper

`r15_exact_image_release.py` is the outer fail-closed release wrapper for the
Qwen3.8 Flash-Next R15 image. It launches one immutable image through the
audited lifecycle-v3 adapter, runs one source-pinned qualification program,
checks the complete suite receipt, proves cleanup, and only then writes the
outer release receipt.

The tracked suite entrypoint is `r15-container-suite.py`. It is deliberately a
single-file, standard-library-only program so executing it from a sealed memfd
does not lose sibling imports or resources. It consumes a separately hashed
suite plan from another sealed inherited descriptor. Every gate driver named
by that plan must also be one standalone executable inside the proven source
root; the suite seals its bytes before executing it without a shell.

The wrapper does not turn offline tests, a dry run, or direct-binary evidence
into a container claim. The suite program must execute and seal all required
runtime gates against the container started by the wrapper.

## Prepare the immutable image

Build from the clean integrated R15 commit. Supply every label from the sealed
source, build, and recipe receipts.

```bash
docker build \
  --build-arg FLUX_ENGINE_VERSION=0.9.13-r15 \
  --build-arg FLUX_SOURCE_REVISION=<40-hex-commit> \
  --build-arg FLUX_SOURCE_TREE=<40-hex-tree> \
  --build-arg FLUX_BINARY_SHA256=<64-hex-binary-sha256> \
  --build-arg FLUX_RECIPE_SHA256=<64-hex-recipe-sha256> \
  --build-arg FLUX_INTERACTIVE_RECIPE_SHA256=<64-hex-interactive-recipe-sha256> \
  --build-arg FLUX_HIGH_PREFILL_RECIPE_SHA256=<64-hex-high-prefill-recipe-sha256> \
  -f repro-kit/Dockerfile \
  -t flux-engine:r15-candidate \
  repro-kit
```

Record the immutable image ID, repo digest, and canonical SHA-256 of the exact
ordered `RootFS.Layers` array returned by `docker image inspect`. A mutable tag
is not accepted as launch identity.

Both performance recipe hashes are mandatory image-build inputs. Derive them
from the proven source files `recipes/flux-v9-w4096-interactive.env` and
`recipes/flux-v9-max-prefill.env` within the build context. The image validator
opens and hashes their actual copied bytes through fixed canonical paths;
the performance labels do not replace the singular startup recipe binding.
The outer image inspector checks all four fixed profile path/hash labels,
cross-binds the manifest to proven source recipes and independently hashes both
actual image files before and after the suite. This still does not qualify
either runtime profile. The live broker needs explicit selected-profile proofs,
and canonical high-prefill needs a reviewed 16384-token scheduler contract
rather than the current universal 4096 measurement contract.

## Prepare the launch manifest

Copy
[`examples/r15-exact-image-launch.example.json`](examples/r15-exact-image-launch.example.json)
outside the source tree. Replace every zero hash and placeholder path with the
sealed value. Create `runtime_dir` and its diagnostics directory before the
run. The suite executable must already exist, be executable, and match
`suite.program_sha256`.

The launch schema is `flux-r15-exact-image-launch/2`; legacy `/1` rejects even
for dry-run. `image.performance_recipes` must contain exactly `interactive`
and `high-prefill`, each with only its fixed canonical in-container `path`
and a nonzero `sha256`. These hashes must agree with the existing sealed plan
profile contract, proven source inventory and reopened canonical source bytes.
The example's two profile hashes describe the checked-in recipe bytes, not
measurements or runtime qualification. Re-derive them for the selected source.
The singular startup binding and startup broker `/1` are unchanged.

The wrapper measures both actual performance files through unmounted standoff
containers before and after the suite. The final receipt retains them under
`image_artifacts.performance_recipes`. Labels do not substitute for extracted
byte hashes. Extracted symlinks, hardlinks, FIFOs, oversized/nonregular files,
identity changes and hash mismatches fail closed; FIFO opening is nonblocking.

Copy
[`examples/r15-container-suite-plan.example.json`](examples/r15-container-suite-plan.example.json)
and replace every zero identity, program path, and hash. The checked-in plan is
schema documentation, not a runnable qualification plan: its gate-driver paths
are intentionally unresolved until the real reviewed drivers are supplied.
Hash the completed plan into `suite.plan_sha256`. The outer wrapper seals the
exact plan bytes and substitutes `@PLAN_FD@`; the suite never reopens a mutable
plan pathname. The suite receipt binds that exact plan hash and a private hash
of the opened runtime-directory identity, so a plan pointed at a different
results tree cannot be substituted behind matching relative artifact names.
The plan itself must embed `source_bundle_manifest_sha256`, byte-equal to the
launcher's proven value; a plan authored against a different source closure
(or an all-zero binding) fails closed before suite replay.

The plan source root must be the exact proven clean source root, with matching
directory identity and source, checkpoint, and recipe provenance. The suite
must be the tracked `repro-kit/r15-container-suite.py` entrypoint. It and every
gate driver must be executable `engine-source` entries in the verified source
inventory, with matching size and SHA-256. Merely placing an ignored or
untracked executable below the root is insufficient. The wrapper and suite
copy verified executable bytes into sealed anonymous files and execute those
file descriptors, so pathname replacement cannot change the bytes that run.

The only permitted bind destinations are read-only `/models` and writable
`/results`, each exactly once. `/models` must be the exact no-symlink checkpoint
directory. `/results` must be the exact dedicated `runtime_dir`; broad roots
such as `/`, `/results`, or `/tmp` are rejected. The image root filesystem must
be read-only. Environment injection, device overrides, and mounts that can
cover the binary or recipe are rejected. The wrapper starts the image with the
owned `/usr/bin/sleep infinity` idle entrypoint; the qualification suite owns
every fresh plain and K5 server process it tests.

The wrapper passes these seven identities to the suite as exact argument
tokens:

- full container ID
- immutable image ID
- immutable repo digest
- loopback base URL
- run ID
- sealed suite-plan file descriptor
- exclusive inherited suite receipt file descriptor

The suite must write schema `flux-r15-container-suite-receipt/2` to that file
descriptor. It must mark every required gate `PASS`, bind the same source,
image, binary, recipe, checkpoint, container, and run identities, and include a
complete uncapped raw-log census with zero CUDA fatal, OOM, illegal-memory,
NVRM, or Xid matches. Each gate must name a distinct receipt with its real
SHA-256. Receipts expose only normalized runtime-relative artifact locators,
never private host paths. The wrapper opens every locator through held
directory descriptors with no-follow traversal, validates every referenced
receipt, and scans the raw log itself. It reopens and rehashes all suite, gate,
and raw-log artifacts after exact container cleanup and before the final seal.
Self-attested or all-zero hashes, symlink traversal, mutable replacement paths,
and bounded tails cannot satisfy the release gate.

Gate evidence is not a generic nonempty object. The outer wrapper enforces an
exact semantic contract for every gate:

- startup proves fresh plain and native-K5 processes, ready endpoints,
  checkpoint/model identity, read-only rootfs, and the executable hash
- tool API proves blocking and streaming case counts, exact calls/arguments,
  valid finishes, spacing, and no blank-line or all-tools runaway
- MMLU-Pro and GSM8K bind the pinned datasets, full item counts, accuracy
  floors, and per-item plain/K5 answer, correctness, and token identity
- long-context MTP proves the full 262,144-token envelope, K5 engagement,
  partial acceptance, QSA verification, plain/K5 output identity, and every
  required 2,017-through-262,080 prompt rung including all 2,043-2,050 rows
- native 262K proves BF16 KV, exact retrieval targets, boundary admission,
  overflow rejection, and terminal QSA chunks
- concurrency proves C2/C4/C8 in both plain and native-K5 modes, every-frame
  logits/state/order, cancellation, final highest-slot reuse, and clean drain
- profile evidence covers interactive (`4,096`-chunked) and high-prefill
  (`16,384`-chunked) at every 8K through 262K context with plain/K5 decode and
  a five-percent host band
- memory, cancellation, and complete-log gates have exact zero-error and
  identity contracts

Every gate therefore ships a validator artifact of schema
`flux-r15-gate-validator/1` — strict JSON `{schema, gate, status: "PASS",
release_qualified: true, container_id, run_id, raw}` whose identity fields match
the launching container and manifest run ID. The wrapper recomputes every
census in near-constant memory — fresh-process attempts, tool calls and
finish reasons, per-item accuracy, long-context rungs, 262K boundary handling,
all 18 concurrency arms, and memory sample counts — from `raw` and requires
exact equality with the evidence. Derived float evidence is validated with
bitwise float equality: decode tok/s must be computed exactly as
`tokens / wall_seconds` and the host delta as
`100.0 * (host_wall - profile_wall) / profile_wall`. A validator recipe using any
other rounding order fails the release.

Profile qualification additionally requires the strict suite-plan `/2`
performance policy, canonical host reference `/3`, explicit source-clock
decode and request-wall TTFT-proxy operands, and an independently recensused
72-request response journal. All plain/K5/prefill host bands must pass.
See [Profile Evidence Contract](R16_PROFILE_EVIDENCE_CONTRACT.md). The current
single-recipe image/broker still holds real dual-recipe profile qualification.
Offline arithmetic/orchestration fixtures are not measured host baselines.

The proven checkpoint and the writable runtime tree must be disjoint release
trees: neither may live beneath the other, and a verified run requires distinct
directory identities, so runtime artifacts can never land inside the
publication source. Mountinfo backing roots are resolved so bind aliases of
the same or nested tree also fail. Nested or ambiguous mount mappings are
unsupported and fail closed. Before docker is touched, the wrapper walks every launch
path below `runtime_dir` without following links — it creates the diagnostics
directory and lockfile parents through descriptor chains, refuses a symlinked
cidfile/lockfile target, and refuses a pre-existing cidfile. The wrapper owns
diagnostics capture: `.inspect.json` and `.server.log` artifacts are created
`O_EXCL|O_NOFOLLOW` through the verified chain and re-verified after sealing.
The cidfile, lock, and diagnostics parent descriptors stay open through launch
and cleanup. Renaming a parent and replacing its pathname with a symlink does
not redirect a later write outside the verified tree.

The suite runs `raw-log-safety` last. That driver receives the complete log as
an inherited descriptor, must be silent, and must report the exact final byte
count and hash. Gate receipts and output first land in writable anonymous
files, then become write-sealed before they are copied into the reserved
artifact paths. The suite then re-censuses the complete raw log itself with a
case-insensitive labeled scan — the `Xid`, `NVRM`, `CUDA fatal`,
`CUDA out of memory`, and `illegal memory access` patterns match every casing
variant in `per_pattern_match_counts` over byte boundaries, with no cap and no
skipped middle. Both independent scanners use a read-only mapping of the whole
file, so markers split across an I/O boundary and arbitrarily long separators
remain visible. Any nonzero match fails the run. The gate evidence retains the
wrapper's exact eight-field raw-log contract. Per-pattern counts belong in its
validator raw artifact and in the suite's independent `raw_log_safety` census,
not as an extra gate-evidence field.

Each gate and the outer suite run under Linux child-subreaper containment.
Kernel reparenting retains detached and double-forked descendants even across
`setsid`, and pidfds bind cleanup signals to the observed process identities.
Any descendant remaining after its driver exits is killed and reaped within a
bounded drain, and the gate fails. Supervisors must start single-threaded with
no pre-existing direct children. Missing pidfd or subreaper support fails closed.

## Live startup process proof

Standoff image hashes do not prove what a server actually executes. Startup
therefore requires an independent live proof for plain followed by native K5,
using distinct PID/start-time identities. The plan supplies
`in_container_binary_path` under `/opt/flux/bin/` and
`in_container_recipe_path` under `/opt/flux/repro-kit/recipes/`, matching the
launch manifest. The suite opens `/proc/<host-pid>/root`, traverses every
in-container path component without following symlinks, and compares the actual
`/proc/<host-pid>/exe` inode and bytes to the release binary. Both paths must
map to read-only mounts. Namespace, exact container cgroup, immutable image,
process start time, and liveness are checked while the server is alive.

The recipe supports only literal `KEY=value` or `export KEY=value` assignments,
with shell quoting and comments. Expansion, commands, duplicate keys, and an
empty recipe are rejected. The live process environment must match every
recipe assignment, except speculation-mode keys may be overridden for plain.
Both processes must use `--no-tui` to prohibit live model swaps and explicitly
select `--model-from-path /models`. The display
name in `/v1/models` or `--model-name`/`--served-model-name` is not checkpoint
authority. The supervisor compares the actual `/proc/PID/root/models` directory
identity to the read-only Docker bind source, which the outer wrapper binds to
the proven checkpoint inventory. Symlinks, duplicate selectors, other model
paths, missing or writable binds, and nested mounts inside `/models` fail.
The split and `--option=value` forms are accepted for value selectors.

Plain must use the serving binary's actual `ATLAS_DISABLE_SPECULATION=1` kill
switch. The startup `FLUX_DISABLE_SPECULATION` alias is accepted only when the
corresponding `ATLAS_` key is absent, matching the binary's exact precedence.
`FLUX_SPEC_MODE=none` and the nonexistent `--no-speculative` flag do not prove
target-only scheduling. Native K5 must use
`--speculative` and exactly one `--num-drafts 4`, without the actual kill switch.
Both modes reject foreign MTP sidecars (`--mtp-from-path`), external checkpoint
selectors (`--draft-model`, `--dflash-donor-model`), and auxiliary proposer
switches (`--dflash`, `--self-speculative`, `--ngram-speculative`).
Qualification also requires an ungated mode (omit `--mtp-gate`) with no REST
store or self-context draft override, including their `FLUX_` aliases. These
selectors can replace native proposals or choose serial execution without
changing the process's `--speculative` argument.
Startup readiness probes must use the engine's `/health` endpoint.

The startup driver alone receives `@PROCESS_PROOF_FD@` in its argv, normally
through `--process-proof-fd @PROCESS_PROOF_FD@`. The suite replaces the token
with an inherited AF_UNIX `SOCK_SEQPACKET` descriptor. After each server is
ready, the driver sends one UTF-8 JSON packet of at most 16 KiB:

```json
{"schema":"flux-r15-live-process-request/1","mode":"none","host_pid":12345}
```

The second packet uses `"mode":"native-mtp-k5"` and a different live process.
The driver must keep each server alive and block for its ACK before stopping
it. The ACK is the suite-owned `flux-r15-live-process-proof/1` record with
`status: "PASS"`, mode, container/run identity, PID/start ticks, actual binary
and recipe hashes, and verified environment/namespace/cgroup/read-only flags.
Missing, extra, duplicated, or out-of-order proofs fail the gate. The suite
seals both ACK records as `process_bindings`, which the outer wrapper checks
again. Drivers cannot substitute self-attested process hashes. The same proof
also binds the release URL to a TCP LISTEN socket inode owned by the actual
server PID in the container's network namespace. The CLI-selected port (default
8888) must match the host-network URL or its unique Docker published-port
mapping. Another process serving the URL cannot qualify this process.

Host networking supports an IPv4 loopback or wildcard listener in the host's
actual network namespace. Published-port networking supports a container IPv4
wildcard listener (`--bind 0.0.0.0`) mapped to `127.0.0.1` or `0.0.0.0` on the
release URL port. Missing or ambiguous mappings, duplicate reachable listeners,
and ambiguous IPv6 wildcard listeners fail closed. Listener identity, selected
arguments, environment, model-directory identity and process liveness are
rechecked before the ACK. The ACK schema is unchanged.

Live proof requires local Docker and sufficient host `/proc` access, including
root-owned server processes when the image runs as root. A suitably privileged
qualification supervisor may be required. Remote Docker, hidden exact
container IDs in cgroups, inaccessible process namespaces, descriptors, TCP
tables or environments, ambiguous mounts, and IPv6-only or indirect proxy
endpoints are unsupported. These cases fail closed rather than falling
back to driver-supplied claims. Offline mocked-process tests exercise this
contract but do not qualify an image.

## Review without launching Docker

```bash
python3 repro-kit/r15_exact_image_release.py \
  --manifest /results/r15-exact-image-launch.json \
  --receipt /results/r15-exact-image-release-receipt.json \
  --dry-run
```

The output says `release_qualified: false`. It is a manifest-plan check only.

## Run the exact-image gate

Acquire the documented build, server, container, and GPU resource claims, then
run:

```bash
python3 repro-kit/r15_exact_image_release.py \
  --manifest /results/r15-exact-image-launch.json \
  --receipt /results/r15-exact-image-release-receipt.json
```

The wrapper refuses pre-existing suite outputs or an outer receipt. It
revalidates the launch manifest and the complete source-to-binary closure before
image inspection, before the suite, after the suite, and before the final seal.
It preserves the runtime, checkpoint, and source-root directory identities
across those phases. It inspects the immutable image both before and after
qualification, and inspects the running container before and after the suite
for the exact idle entrypoint, read-only root filesystem, and two expected bind
sources and modes. It also copies the selected server binary and recipe out of
the immutable image and hashes their actual bytes before and after the suite.
Copies never read the running rootfs: the wrapper creates an unmounted standoff
container from the exact image ID with `docker create --entrypoint /bin/false`,
copies the artifacts with `docker cp` (no `--follow-link`, so container-side
symlinks cannot redirect the read), then removes the standoff container. Copied
bytes are hashed against the provenance chain on both probes. Any identity
drift, artifact drift, missing gate, malformed receipt, unsafe cleanup, or
nonzero raw log match leaves the release unqualified.

The outer runner and tracked orchestrator are offline-qualified infrastructure.
They are not themselves a runtime PASS. A release still requires implementing
and independently reviewing every real gate driver named in the suite plan,
then running the plan against the exact immutable image with claimed resources.

Do not update a GHCR version or stable tag until the outer receipt exists and an
independent review confirms its hashes and referenced gate artifacts.
