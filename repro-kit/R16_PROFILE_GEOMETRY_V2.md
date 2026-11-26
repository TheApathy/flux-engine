# Profile Geometry /2 And Dual-Recipe Broker /2

Source only. `r16_profile_geometry.py` is pure validation, tested by
`tests/test_r16_profile_geometry_v2.py` on synthetic argv and journals with
the real recipe hashes. It is not a launch, measurement or release receipt.

`R16_PROFILE_EVIDENCE_CONTRACT.md` (geometry /1, plan /2) is unchanged and
still describes plan /2. It assumes one
universal 4096 chunk, so it cannot describe canonical high-prefill
(WIDE=16384). `/1` stays readable, and `chunk_tokens()` returns 4096 for it,
but `require_dual_profile()` refuses `/1`. Only `/2` can admit both profiles.

## Reviewed Geometry

| profile | recipe | FLUX_MODE | WIDE = chunk = --max-prefill-tokens | block | capacity |
| --- | --- | --- | --- | --- | --- |
| interactive | `flux-v9-w4096-interactive.env` | interactive-w4096 | 4096 | 16 | 262144 |
| high-prefill | `flux-v9-max-prefill.env` | max-prefill | 16384 | 16 | 262144 |

Each `/2` profile binds:
- its recipe sha256 (the two must differ);
- `max_batch_tokens`, which is `null` (meaning `ATLAS_MAX_BATCH_TOKENS` must be
  absent) or an exact bound value;
- the FULL effective argv per mode, base plus overlay.

That argv must pin each of `--max-prefill-tokens`, `--max-seq-len`,
`--block-size`, `--scheduling-policy` and `--tbt-deadline-ms` exactly once,
and must include `--qwen4-qsa` (capacity is above 2048). For one mode, the two
profiles' argv may differ only in the `--max-prefill-tokens` value. Base
flags, policy and deadline are therefore identical, and an omitted policy
fails closed rather than defaulting to fifo.

## Repair Of The 1053a6e6 False Passes

- **Mode.** `argv_mode()` derives the mode from the argv itself.
  `native-mtp-k5` is exactly `--speculative` with `--num-drafts 4`, and
  `none` has no speculation selector; anything else is refused. Every
  geometry argv must select its declared mode. The live broker also requires
  the request mode, the argv mode and the effective kill switch to agree
  (`ATLAS_DISABLE_SPECULATION`, or its `FLUX_` alias, is `1` exactly for
  Plain).
- **Duplicates.** Any option repeated in any spelling (`--x v`, `--x=v`, or
  trailing with no value) is refused, and `--x=v` spellings are refused
  outright.
- **Recipes.** `REVIEWED_RECIPE_SHA256` binds each profile to its own recipe
  bytes, so a consistent swap is refused. `verify_recipe_bytes()` checks the
  sha, `FLUX_MODE`, WIDE equal to the profile chunk, and no arena selector.
- **Integers.** JSON integers must be exact `int` in range (bool and float
  are refused), and `role_geometry` refuses `8192.0`. Text integers go
  through `strict_decimal()`: no sign, float, exponent, whitespace, empty or
  leading zero. `max_batch_tokens` is `null` (key absent; an empty value
  still counts as present and is refused) or an exact int from chunk to
  2^32, and the live value must equal it.

## Release Authority And Mandatory Caller Boundary (fb83 review)

A consistently resealed geometry and broker pair proves only internal
consistency. The helper therefore pins its own authority.

- **Geometry pin.** `REVIEWED_GEOMETRY_SHA256` is the sha of
  `reviewed_geometry_v2()`, the one trusted document. Any other geometry is
  refused, even when it is structurally valid.
- **Executable.** `REVIEWED_EXECUTABLE` is `/opt/flux/bin/flux-server-real`,
  sha256 `3460c766…27d3`, 92082232 bytes, the e82 ELF.
- **Base argv.** `reviewed_effective_argv(profile, mode)` is exact:
  - the executable, then `serve`, `--no-tui` and `--model-from-path /models`;
  - `--model-name` and `--port` placeholders, `--max-batch-size 1`,
    `--max-num-seqs 1`, `--kv-cache-dtype bf16` and prefix caching off;
  - the profile geometry, then `--qwen4-qsa`, fifo and deadline 100;
  - for K5 only, `--speculative --num-drafts 4`.
- **Arena.** `arena_bounds()` mirrors the engine's resolved minimum and
  maximum (kv_cache.rs:131, :223-230, :108-121, :236-246 at e82;
  decode_meta.rs:14-26). The minimum is max(chunk + max_batch_size,
  num_drafts + 2, padded_decode_rows(max_batch_size)), so chunk alone is
  refused and chunk + 1 is admitted at C1.

**The helper is NOT standalone engine admission.** `admit_live_recipe`
requires the caller to observe through /proc and pass in:
- the opened executable's path, sha256 and size;
- the `/models` identity;
- the recipe path and sha;
- the exact argv and environment;
- the REQUIRED keyword `trusted_model_identity`, either
  `{"config_sha256"}` or `{"dev", "ino"}`. It has no default; omitting it
  raises TypeError, and None or an empty value raises GeometryError.

Callers must pin the trusted geometry sha, the exact executable, the model,
the base argv and the KV dtype, and must not substitute their own.

## Version Pairing

| geometry | plan | contract | host reference | metrics | proof |
| --- | --- | --- | --- | --- | --- |
| /1 | suite-plan/2 | performance-profile-contract/1 | host-reference/3 | profile-metrics/3 | live-process-proof/1 |
| /2 | suite-plan/3 | performance-profile-contract/2 | host-reference/4 | profile-metrics/4 | r16-live-process-proof/2 |

`check_version_pairing()` accepts one complete row and refuses every cross
pair. A `/4` writer therefore cannot emit under plan `/2`. Host reference `/4`
keeps the `/3` decode and prefill clocks and host identity unchanged, plus
the binary hash, and adds `geometry_sha256`.

## Rows And Census

`bind_row()` gives every v2 host and measured row its geometry binding:
- profile, mode, context and kind;
- recipe sha and effective-argv sha;
- chunk, derived chunk count and last-chunk rows (262080 is 64x4096 with a
  4032-row tail, or 16x16384 with a 16320-row tail; 8192 under high-prefill
  is one sub-WIDE chunk);
- capacity headroom.

`require_row()` re-derives the whole binding for the required profile, so a
4096 row never satisfies a 16384 requirement or the reverse, and a relabelled
profile, chunk or recipe is refused.

`recensus()` keeps the raw 72-record census: 2 profiles x 2 modes x 6
contexts x 3 kinds. The kinds are `exact` (blocking exact token IDs), `ttft`
(one-token request wall) and `stream` (streaming usage). Every canonical
record must carry its own binding, the census must be exactly the full
product, and duplicates, gaps and relabels are refused. "Matched" means the
host reference covers these same 72 roles row for row.

## Broker /2

`broker_contract()` binds exactly the two reviewed recipes (in-container
path, sha, WIDE, arena bound and per-mode argv sha) and the geometry sha.
`admit_live_recipe()` takes request `flux-r16-live-process-request/2`
(profile, mode, host_pid) and what broker `/1` already reads through /proc
(recipe path and sha, argv, environment). It refuses:
- a single-recipe `/1` broker;
- a broker whose recipe set differs from the reviewed pair, whether narrowed
  or with a third recipe added;
- a third recipe sha;
- the other profile's recipe;
- an argv other than the bound one;
- a WIDE selector that differs;
- an unbound arena.

## Integration (plan /3)

The sealed tools cannot import an unpinned helper, so the suite (shared with
the gate via `args.suite`) and the independent wrapper each implement the
same `/2` rules. Plan `/2` keeps every `/1` behaviour and HOLD unchanged.
Plan `/3` pairs with profile contract `/2`, host reference `/4`, metrics `/4`,
live request and proof `/2`, and every cross pair is refused.

- Suite `load_plan` and `validate_performance_profile_contract`:
  - accept plan `/2` with contract `/1` (unchanged) or plan `/3` with
    contract `/2`;
  - contract `/2` binds per profile the recipe path and sha, the in-container
    recipe, the reviewed chunk, an 11-token overlay and the sha of the full
    launch argv per mode;
  - the overlay sets chunk, context 262144, block 16, policy and deadline
    each once, plus `--qwen4-qsa`;
  - the full launch argv is the fixed base plus the overlay and mode suffix,
    with model and port as placeholders;
  - the two profiles' overlays may differ only in chunk.
- Suite `verify_live_server`:
  - request `/1` is unchanged;
  - request `/2` names a profile and requires the plan to be that profile's
    recipe and chunk;
  - it then requires the live WIDE selectors and `--max-prefill-tokens` to
    equal the chunk, and `ATLAS_MAX_BATCH_TOKENS` to be absent;
  - it returns proof `/2` with the profile.
- Gate (old → new):
  - `CHUNK_TOKENS` and `HOST_REFERENCE_SCHEMA` (:30, :29) → per-profile
    `profile_chunk()`, with host reference `/3` or `/4` chosen by plan;
  - the host census (:338) → `profile_chunk_tokens`;
  - the overlay chunk (:442) → `profile_overlay_v2(overlay, chunk)`;
  - `verify_profile_contract` replaces the dual-recipe HOLD, under `/3`
    only, with source-recipe, host-argv, full-argv and image-label bindings;
  - `inspect_container` opens and hashes both labelled performance recipes;
  - each profile launches its own recipe through `profile_server_plan` and
    `live_request` `/2`;
  - metrics (:1289, :1292, :1317) → `/4` with `profile_chunk_tokens`;
  - every `/3` journal record and raw row carries `chunk_tokens` and
    `recipe_sha256`.
- Suite `:618` overlay chunk → `/2` validator above (`/1` keeps 4096).
- Wrapper (old → new):
  - plan schema (:1437) → plan `/2` or `/3`;
  - `_profile_contract` (:2534, called with its plan at :1371, :1524 and
    :3054) → pairing plus the independent `/2` validator;
  - `_profile_overlay` (:2513) → `/1` only, with `_profile_overlay_v2` for
    `/2`;
  - `_profile_runtime_recipe_binding`: the `/1` HOLD is unchanged, and `/2`
    requires both image performance recipe paths and shas;
  - host census (:3084) → per-profile chunks, with host `/4` binding the
    full-argv sha;
  - `_profile_processes`: the observed argv must still equal the code-built
    argv, and under `/2` its template sha must equal the sealed binding;
  - proofs `/2` carry the profile and its recipe sha;
  - the journal re-census keeps 72 records, with `/2` records binding chunk
    and recipe;
  - raw and evidence rows and the chunk census are dispatched.

These files' source-inventory pins change and are re-pinned downstream.

## Packet v4: Sealed Dual Recipes (on 989c13903)

`repro-kit/recipes/flux-profile-interactive-w4096.json` and
`flux-profile-high-prefill-w16384.json` are the two sealed profile recipes.
Each is `canonical_json(profile_recipe_document(profile))`, and a test
requires the checked-in bytes to equal that derivation. Each binds:
- its environment recipe (path, sha, `FLUX_MODE`, WIDE) and the geometry sha;
- the full effective argv per mode, both as the contract /2
  `launch_argv_sha256` and as the /proc cmdline sha;
- one scheduling policy, `fifo` with deadline 100, for both profiles and
  modes, and `--qwen4-qsa`, since capacity 262144 is above 2048;
- the host recipe engine selectors of `recipes/flux-serve*.argv`:
  `--kernel-target qwen3.8-flash-next`, `--gpu-memory-utilization 0.85`,
  `--ssm-cache-slots 0`, `--oom-guard-mb 4096`, `--request-timeout 300`,
  `--disable-simhash-watchdog`, `--mtp-vocab 0` and
  `--mtp-quantization bf16`. Without them the container ran CLI defaults
  (0.90, 16, 100000) that the host never ran (review finding 2);
- the environment: image ENV, recipe and the kill switch, with
  `ATLAS_MAX_BATCH_TOKENS`, `FLUX_MAX_BATCH_TOKENS` and `RUST_LOG` absent;
- the live proof lines and the per-role geometry with the K5 tail proof.

The suite's `profile_overlay_v2` and the wrapper's `_profile_overlay_v2`
now require the overlay to equal the sealed overlay exactly. A consistent
`slai` for both profiles, or the old 11-item overlay, is refused. The host
argv may differ from the container template only in executable path, model
path, model alias, bind (127.0.0.1) and port (`normalize_host_argv`).

**Live startup proof (finding 6).** `startup_census(log, arms)` requires
exactly one `Prefill config:` line and one `KV cache (post-construction):`
line per launched arm, in launch order and with ANSI stripped:
- `ssm_prefill_chunk=8192` (preflight.rs:283-287);
- `args.max_prefill_tokens` and `prefill_budget` equal to the chunk
  (kv_cache.rs:132-134);
- `max_batch_tokens` equal to the chunk + 1 in both modes
  (kv_cache.rs:223-232, with the override absent);
- a pool of 16386 blocks x 16 (build.rs:1020-1024).

**K5 tail (finding 12).** Admission is inclusive (context_budget.rs:10-12),
so 262080 + 64 is admitted with zero `max_seq_len` headroom. The engine's
addressable rows are still 16385 x 16 = 262160: the block stride
(impl_a1.rs:574), the capped pool minus its dummy block (build.rs:1020-1024,
impl_a1.rs:582) and the MTP cache (qwen4_mtp.rs:294). The final K5 frames
need fewer than 262080 + 64 + 6 = 262150 rows, which leaves 10. This proof
holds only when the live pool census shows the full 16386-block cap; a
budget-limited pool of 16384 or 16385 blocks still starts and is refused by
the census. QSA decode-side buffers were not audited for rows at or beyond
262144.

`r16_profile_host_producer.py` measures the host with the gate's own
`measure_arm` and `build_row`, and round-trips host reference /4 through the
gate's plan /3 loader. It is not wired to a release and measures nothing by
itself.

## Still Held

- The wrapper does not yet run `startup_census` over the suite-owned gate
  stderr artifact. The engine log reaches that artifact through the
  supervisor's stdout=2, but no wrapper call checks it yet.
- `REVIEWED_EXECUTABLE` still pins the e82 ELF. A 989c build must re-pin it,
  and then the manifest and host references must be regenerated
  (finding 10).
- No measured host reference is supplied.
