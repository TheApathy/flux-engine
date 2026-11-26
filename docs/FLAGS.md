# Environment Variables & CLI Flags Reference (`FLAGS.md`)

Complete reference for configuring **Flux Engine** via the universal container appliance (`ghcr.io/theapathy/flux-engine:gb10`), the host environment bridge ([`scripts/flux_env_bridge.sh`](../scripts/flux_env_bridge.sh)), and the [`bench/qwen38-flash-next-gb10/`](../bench/qwen38-flash-next-gb10/) / [`bench/qwen38-gb10/`](../bench/qwen38-gb10/) launchers.

---

## 1. Universal Container & Wrapper Environment Variables (`FLUX_*`)

When launching `ghcr.io/theapathy/flux-engine:gb10` (or invoking `flux serve` / `flux inspect`), these high-level variables control model auto-detection, operating mode, networking, memory budgets, and speculative decoding:

| Variable | Default | Description |
|---|---|---|
| `FLUX_MODEL` | *(auto-discovered under `/models` or positional arg)* | Path to local model checkpoint directory (e.g. `/models/Qwen3.8-Flash-Next-NVFP4-Offload` or `/models/Qwen3.8-27B-NVFP4`). |
| `FLUX_MODE` | `auto` | Operating profile: `auto`, `max-prefill` (`v9` / `w16384`), `interactive` (`interactive-w4096` / `w4096`), `fast-prefill` (`fast`), `decode` (`lossless-decode` / `k5-decode`), `long-context` (`long-context-136k` / `136k`), `exact` (`exact-reference`). |
| `FLUX_PORT` | `8899` | HTTP listening port for the OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, `/health`). |
| `FLUX_HOST` | `127.0.0.1` (host) / `0.0.0.0` (container) | Bind address passed to `--bind` (`--host`). Use `--api-key` when exposing beyond loopback. |
| `FLUX_MAX_SEQ_LEN` | Mode-dependent (`40960` / `65536` / `135168` / `143360`) | Maximum context length in tokens (`--max-seq-len`). |
| `FLUX_KV_CACHE_DTYPE` | `auto` (`bf16` / `fp8` / `turboquant`) | KV cache precision (`bf16` required for lossless Qwen3.8-Flash-Next and Qwen3.8-27B recipes). |
| `FLUX_GPU_MEMORY_UTILIZATION` | `0.55` (GB10 UMA) / `0.90` (discrete) | Safe GPU/UMA memory fraction (`--gpu-memory-utilization`, keep `<= 0.55` on 128 GB GB10 UMA to protect OS page cache and CUDA runtime headroom). |
| `FLUX_DRAFT_MODEL` | *(auto-detected if companion drafter present)* | Path to companion speculative drafter directory (`Inferact-MTP` or `DFlash`). Automatically reads `block_size` from drafter `config.json`. |
| `FLUX_MTP_SOURCE` | `auto` | Qwen3.8-Flash-Next MTP routed-expert source when both the checkpoint and an `--mtp-from-path` / `FLUX_DRAFT_MODEL` sidecar provide one: `auto` (checkpoint bank wins; the sidecar fills in only for checkpoints without `mtp.layers.0.mlp.experts.*`, such as the Offload), `checkpoint` (never load a sidecar), `sidecar` (replace the checkpoint bank with the sidecar's). Only routed experts come from the sidecar. The server logs one `QWEN4_MTP_SOURCE source=...` line. |
| `FLUX_GAMMA` | `auto` (`4` for MTP `K=5`, `block_size - 1` for DFlash) | Number of speculative draft tokens per step (`gamma`). For DFlash, automatically sets `ATLAS_DFLASH_DRAFT_CAP=gamma` and `ATLAS_DDTREE_MAX_NODES=gamma+1`. |
| `FLUX_DISABLE_SPECULATION` | `0` *(unset)* | Set to `1` to disable all speculative decoding (equivalent to `--no-speculative` / `ATLAS_DISABLE_SPECULATION=1`; use `FLUX_DRAFT_MODEL=none` to disable external DFlash/MTP sidecars). |
| `FLUX_THINKING` | unset (the `MODEL.toml` `thinking_default` applies; `false` for Qwen3.8-Flash-Next since `9.15.3.1`) | Default `<think>` reasoning behavior. `on` passes `--default-chat-template-kwargs '{"enable_thinking":true}'`, and requests can still turn it off. `off` also passes `--disable-thinking`, which ignores a per-request `enable_thinking: true`. To keep per-request opt-in, leave it unset rather than `off`. |
| `FLUX_THINKING_BUDGET` | MODEL profile (`262144` in `MODEL.toml` for Qwen3.8-Flash-Next & Qwen3.8-27B) | Optional launcher override for the server thinking-budget ceiling (`--max-thinking-budget`). A request may select a lower budget; when unset, the selected `MODEL.toml` profile remains authoritative. |
| `FLUX_THINK_NUDGE` | `off` | Thinking nudge (Qwen3.8-Flash-Next and Qwen3.8-27B only; ignored with a warning elsewhere). `on` writes one first-person "commit to an approach" sentence into the open think block when thinking goes in circles (25% of the last 1024 thinking tokens are self-repeated 8-grams, after 3072 tokens; at most twice per turn). It never closes thinking and never stops generation. Fenced code is not read. Responses report `usage.completion_tokens_details.think_nudges` (omitted when 0). Also read as `ATLAS_THINK_NUDGE`. |
| `FLUX_THINK_NUDGE_RATE` / `_MIN_TOKENS` / `_MAX` | `0.25` / `3072` / `2` | Nudge trigger rate in (0, 1], minimum thinking tokens, and nudges per turn (0-2). Parsed strictly while the nudge is on (a bad value fails startup); ignored while it is off. |
| `FLUX_SETTINGS_FILE` | `off` | Save live setting changes (`flux set`, `PATCH /v1/flux/settings`) and reapply them when the same model starts again. `on` = `/results/flux-runtime-settings.json`, or give a path. Written atomically, mode `0600`; a file for another model is ignored; an invalid file fails startup. Also read as `ATLAS_SETTINGS_FILE`. |
| `FLUX_CPU_AFFINITY` | `5-9,15-19` (on GB10 20-core ARM64) | CPU core pinning mask passed to `taskset -c` (pins runtime and PLE DMA rings to the 10 Cortex-X925 performance cores). |
| `FLUX_CONCURRENCY` | `1` | Max concurrent sequences (`--max-num-seqs` / `--max-batch-size`). Qwen3.8-Flash-Next batches `2..32` natively by default (`ATLAS_QWEN4_NATIVE_BATCH` unset); `33..128` requires `ATLAS_QWEN4_NATIVE_BATCH=0` and decodes C1 once per row. Startup memory admission still applies. |
| `FLUX_KERNEL_TARGET` | `auto` | Explicit `--kernel-target` override matching a directory under `kernels/gb10/` (e.g. `qwen3.8-flash-next`, `qwen3.8-27b`, `qwen3.6-27b`, `deepseek-v4.1`, `glm5.3-flash`, `gemma-4-26b-a4b`, `gemma-4-31b`). |
| `FLUX_TOPOLOGY` | `auto` | Hardware topology (`auto`, `1x-spark`, `2x-spark`, `4x-spark`, `generic-gpu`). |
| `FLUX_WEIGHT_CACHE_DIR` | `/var/cache/flux-weight-cache` | Directory for persistent post-transform weight cache files (`ATLAS_WEIGHT_CACHE_DIR`). |

---

## 2. Server CLI Arguments (`spark` / `flux-server`)

| CLI Flag | Flagship Value (`Qwen3.8-Flash-Next`) | Flagship Value (`Qwen3.8-27B`) | Purpose |
|---|---|---|---|
| `--model-from-path` | `/path/to/models/Qwen3.8-Flash-Next-NVFP4-Offload` | `/path/to/models/Qwen3.8-27B-NVFP4` | Target model checkpoint directory. |
| `--served-model-name` | `qwen38-native-mtp-frontier-v1` | `Qwen3.8-27B` | OpenAI `/v1/models` and `/v1/chat/completions` model identifier. |
| `--kernel-target` | `qwen3.8-flash-next` | `qwen3.8-27b` | Selects the embedded `sm_121a` PTX dispatch table. |
| `--speculative` | Enabled (`--num-drafts 4`) | Enabled (`--dflash --dflash-gamma 15`) | Activates speculative decoding (`K=5` Inferact MTP or `K=16` DFlash). |
| `--draft-model` | `/path/to/models/Qwen3.8-Flash-Next-Inferact-MTP` | `/path/to/models/Qwen3.8-27B-DFlash2` | Companion speculative drafter checkpoint directory. |
| `--mtp-vocab` | `96000` | — | Vocabulary slice evaluated by the Inferact MTP head (`96,000` top tokens). |
| `--max-prefill-tokens` | `16384` (`w16384`) or `4096` (`w4096`) | `4096` | Maximum tokens processed per prefill chunk. Must match `FLUX_QWEN4_PREFILL_WIDE`. |
| `--max-seq-len` | `135168` (`128K`) or `143360` (`136K+`) | `40960` | Maximum KV cache sequence capacity allocated at server startup. |
| `--kv-cache-dtype` | `bf16` | `bf16` | BF16 KV cache (required for lossless speculative decode equivalence). |
| `--gpu-memory-utilization` | `0.55` | `0.55` | Safe GB10 UMA allocation cap. |
| `--enable-prefix-caching` | `false` (benchmarks) / `true` (chat) | `false` | Disables prefix caching during TTFT sweeps so every prompt computes 100% of tokens. |
| `--ssm-cache-slots` | `0` | `0` | Disables GDN recurrent state prefix caching for cold-cache measurement. |
| `--default-chat-template-kwargs` | `'{"enable_thinking":false}'` | `'{"enable_thinking":false}'` | Default Jinja template kwargs when omitted by the client request. When neither this flag nor `FLUX_THINKING` is given, the `MODEL.toml` `thinking_default` applies (`false` for Qwen3.8-Flash-Next since `9.15.3.1`). |
| `--max-thinking-budget` | unset (MODEL profile: `262144` in `MODEL.toml` for Qwen3.8-Flash-Next) | unset (MODEL profile: `262144`) | Explicit server ceiling override. Request budgets are capped by this value; leaving it unset preserves the model-specific ceiling. |

**Qwen3.8-Flash-Next sampling presets (since `9.15.3.1`):** the thinking-off presets (`non_thinking`, `tools`) use `temperature 0.0` and `presence_penalty 0.0` (`9.15.3`: `0.7` + `1.5`). Since `9.15.6` the thinking presets (`thinking_text`, `thinking_coding`, and tool turns that turn thinking on, via `thinking_sampling_in_tools`) use the checkpoint's `generation_config.json` again: `temperature 1.0`, `top_p 0.95`, `top_k 20` (`9.15.3.1` made them greedy, and greedy long reasoning loops). Thinking is off by default, so this applies only to requests that turn thinking on. Explicit request fields override the preset.

---

### 2.1 Benchmark intervention ledger

Quality and tool benchmarks can add `"report_interventions": true` to a
`/v1/chat/completions` request. The terminal blocking response or terminal SSE
choice then contains an `interventions` object with coalesced `{code, action,
count}` events. Emitted wire JSON `code` strings for cross-turn guards cover
prompt/context hygiene (`"F7"` for stale-observation masking and vacuous-system
removal, `"F23"`/`"F30"` for historical tool-argument repair and CWD enrichment,
`"F28"`/`"F31"` for loop suppression and auto-compaction/overflow truncation),
as well as output and tool-call guards (`"F32"` / `"F39"` / `"F49"` for
streaming role/thinking/orphan-markup sanitization, `conditional F50` for
native rejection feedback, `F35/F52` for schema coercion, path normalization,
and serial-tool truncation, and `"F29"` for blank reasoning suppression and
answer-prefix trimming). The pre-dispatch prompt path also
reports tool-error recovery hints, malformed historical tool-argument
replacement, normalized CWD enrichment, vacuous-system removal,
stale-observation masking, loop suppression/hints, task-pin anchors, historical
XML-leak stripping, auto-compaction, context-overflow truncation, and
template-forced thinking. Streaming tool guards and thinking/content watchdogs
remain covered as well. Published blocking tool-output mutations are also
recorded when schema backfill/type coercion changes arguments, path
normalization rewrites an argument, fuzzy validation changes a tool name, the
JSON-object prefix repair changes content, orphan tool markup is stripped,
native rejection feedback is appended, or a serial-tool policy truncates extra
calls. Invalid parsed calls and DeepSeek streaming serial-tool truncation are
covered by the same policy events. Blank reasoning suppression, answer-prefix
trimming, refusal-field publication, and streaming removal of role/thinking or
orphan/partial tool markup are recorded as well. When the configured
`qwen3_coder` parser salvages, repairs, drops, or accepts a foreign tool-call
wire form, `tool_parser_recovery/recovered_noncanonical_call` is recorded;
exact native Qwen XML remains unflagged, including when a serving stop consumes
only the final outer `</tool_call>`. A canonical Qwen call does not hide a
foreign Mistral, bare-function, or other salvageable sibling: mixed-format
output is still recorded. Ordinary cross-chunk buffering does not count as a
rewrite. The ledger is deliberately report-only:
`policy="report_only"`; requesting it never disables or weakens a guard.
Omitting the request field preserves the existing wire format and serving
behavior.

**`finish_reason` for guard stops (since `9.15.6`):** `"length"` means only
that `max_tokens` or the context ran out. When a server guard ends generation
(the content-loop, fuzzy-repeat, inter-tool-prose or repeated-`</think>`
watchdog, or the always-on catastrophic content-loop guard), blocking and
streaming chat responses report `finish_reason: "stop"` and add the
non-standard choice field `flux_stop_reason`: `"catastrophic_guard"` for the
catastrophic guard, `"watchdog"` for the others. The field is omitted on every
other finish, so the wire format is unchanged for ordinary requests. Before
`9.15.6`, scheduler-side guard stops were reported as `"length"`.

Benchmark harnesses must treat this contract as fail-closed: require a terminal
ledger with the expected schema and an empty `events` array. A missing terminal
ledger, an unknown schema/policy, or any recorded intervention invalidates that
sample rather than silently crediting post-processed output.
The built-in GSM8K and MMLU-Pro LIL profiles and the bounded tool-eval runaway
microgate always opt in and enforce `schema="atlas.interventions.v1"`,
`policy="report_only"`, and `events=[]` on every qualifying response.

The ledger reports interventions that fired; it is not an attestation of the
server's startup flags. A qualification harness must separately pin and record
the launch profile (including any operator-level watchdog disable flags).

An empty ledger means no adaptive guard, recovery rewrite, compaction, or
watchdog covered above changed the request/response. It does **not** mean that
the rendered model prompt is byte-identical to the OpenAI JSON. Deterministic
protocol serialization is intentionally excluded: rendering the client's
declared tool schemas/parser instructions, historical assistant thinking
wrappers required by the checkpoint template, Jinja role/tool-call rendering,
and image-marker/pad expansion. Request/profile sampling and thinking defaults
are also excluded because they are launch/model policy rather than a
conversation rewrite; benchmark manifests must pin them. These exclusions are
required to turn an explicit structured request into the checkpoint's native
input and are not activated by a quality/failure verdict. Stop-sequence
termination and exact structured parsing of the configured canonical tool
envelope are likewise protocol publication rather than repair: they preserve
the authored semantic result instead of inventing or correcting it. Alternate,
malformed, cross-format, and heuristic parser paths are not excluded and must
appear in the ledger. The tool microgate separately proves exact stream/block
text, tool bytes, token IDs, and nonced server-produced rendered-prompt digests.

---

## 3. Advanced Kernel & Runtime Selectors (`FLUX_*` / `ATLAS_*`)

Sourcing [`scripts/flux_env_bridge.sh`](../scripts/flux_env_bridge.sh) translates any `FLUX_<SUFFIX>` variable directly to `ATLAS_<SUFFIX>`. Every selector below is active in [`recipes/flux-v9-max-prefill.env`](../recipes/flux-v9-max-prefill.env):

### 3.1 Routed & Shared MoE (`512` Routed Top-10 + `1` Shared Expert)

| Selector (`FLUX_*` or `ATLAS_*`) | Value | Effect |
|---|:---:|---|
| `QWEN4_PREFILL_WIDE` | `16384` / `4096` | Prefill chunk token width (`16,384` for `max-prefill`, `4,096` for `interactive-w4096`). |
| `QWEN4_PREFILL_MOE_W4A8` | `1` | Enables `sm_121a` W4A8 (`NVFP4` weights x `MXFP8` activations) routed MoE GEMM. |
| `QWEN4_PREFILL_MOE_W4A8_WIDE` | `7` | Selects the `v28` compact transposed W4A8 MoE grouped GEMM (`gate_up`). |
| `QWEN4_PREFILL_MOE_V28Q` | `2` | Zero-`sE` register-quad shuffle MXFP8 epilogue + 16-warp B-weight transcoding (`0` spills). |
| `QWEN4_PREFILL_MOE_COMPACT_K32` | `1` | Enables `qwen4_moe_compact_t_gemm_k32_v17q` (`2 CTAs/SM`, `122` regs, `0` spills) for `down_proj`. |
| `QWEN4_PREFILL_MOE_PLAN_ORDER` | `inter` | Inter-expert planner ordering (`v8`) to maximize L2 weight tile reuse across rows. |
| `QWEN4_PREFILL_MOE_FUSED_SORT_PLAN` | `1` | Fuses expert token sorting and tile offset planning into a single launch. |
| `QWEN4_PREFILL_MOE_TOPK_WARP` | `1` | Warp-cooperative top-10 router softmax and expert selection. |
| `QWEN4_PREFILL_ROUTER_GEMM_PIPE` | `1` | Chunk-aware router GEMM dispatch (`fp8_gemm_ex` at `w4096`, `fp8_gemm_n128` at `w16384`). |
| `QWEN4_PREFILL_SHARED_W4A8_PRETRANSCODE` | `1` | Pre-transcodes shared expert weights in-place inside idle `ssm_deinterleaved` scratch (`+243.4 MiB` saved). |
| `QWEN4_PREFILL_MOE_SLOT_MXFP8` | `1` | Fused `v16q4_gate` MXFP8 slot activation quantization. |

### 3.2 4-Stream Multi-Stream Hyper-Connections (`mHC`)

| Selector (`FLUX_*` or `ATLAS_*`) | Value | Effect |
|---|:---:|---|
| `QWEN4_PREFILL_MHC_FUSED` | `inject,cross,combine,nostage,upmix,injgemm,postpre` | Enables all 7 fused mHC prefill stages across `M=4,096..16,384` (`PIN_M=4096` heuristic admission). |
| `QWEN4_PREFILL_MHC_ROOF` | `t128` | Dispatches `qwen4_hc_upmix_roof_t128_amax` with split `qwen4_hc_prep_up` for 100% L2 residency of `s.up`. |
| `QWEN4_PREFILL_MHC_GATEPRE` | `1` | Dispatches warp-per-row period-6 unrolled `qwen4_hc_combine_moe_gemm_mx_gatepre` (`0` D2D permutation copies). |
| `QWEN4_PREFILL_HC_FUSED_PREP` | `1` | Fuses mHC pre-layer stream preparation and RMSNorm. |

### 3.3 Query-Sparse Attention (`QSA` — 12 Layers)

| Selector (`FLUX_*` or `ATLAS_*`) | Value | Effect |
|---|:---:|---|
| `QWEN4_PREFILL_QSA_FP8` | `qkpv` | Runs both `Q·K` and `P·V` sparse attention stages in FP8 `E4M3` Tensor Core MMA. |
| `QWEN4_PREFILL_QSA_PIPE` | `1` | Dispatches `cp.async` double-buffered `qwen4_qsa_prefill_attn_fp8_qkpv_amax_pipe` with 16-token half-tile skipping. |
| `QWEN4_QSA_SCORE_V2` | `1` | Uses `ldmatrix.sync.aligned.m8n8.x4.shared.b16` block scoring at `5 CTAs/SM` (`96` regs, `0` spills). |
| `QWEN4_PREFILL_QSA_FUSED_TOPK` | `1` | Bimodal top-`k` block selection (`topk_v2` for `<= 8,192` blocks, `topk_fused_c3_v2` for `> 8,192` blocks). |
| `QWEN4_PREFILL_QSA_PLAN_FUSED` | `1` | Fuses QSA sparse block plan construction and descending sort. |
| `QWEN4_PREFILL_OPROJ_AMAX_FUSED` | `1` | Folds `o_proj` input FP8 absmax reduction into the QSA attention epilogue. |

### 3.4 Gated DeltaNet (`GDN` — 36 Layers) & Fast Projections

| Selector (`FLUX_*` or `ATLAS_*`) | Value | Effect |
|---|:---:|---|
| `QWEN4_PREFILL_GDN_FI` | `2` | Dispatches FlashInfer `sm_121a` CuTe-DSL WY chunkwise recurrence (`atlas_gdn_fi_sm121a_prefill_f16`). |
| `QWEN4_PREFILL_GDN_CONV_V3` | `1` | Dispatches shared-memory filter-staged `qwen4_conv1d_l2norm_f16_prefill_v3_commit` (`TOKENS=64`). |
| `QWEN4_PREFILL_GDN_NORM_FP8` | `2` (recommended) / `1` (`exact`) | `2`: state-deterministic static-scale FP8 norm (`LOSSLESS 10/10`); `1`: exact two-pass reference. (`3` is quarantined due to a multi-chunk store-warp ordering race under `K=5` speculative parity.) |
| `QWEN4_PREFILL_GDN_NORM_FP8_V2` | `1` | Vectorized `v2` epilogue for `GDN_NORM_FP8=2`. |
| `QWEN4_PREFILL_FP8_ACT_AMAX_FUSED` | `1` | Dispatches 64-row tile fused `qwen4_fp8_act_and_ba_gates_mma` (`72` regs, `0` spills). |
| `QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE` | `1` | Caches 96 FP8 `E4M3` fast-projection weight matrices in unified memory. |
| `QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB` | `7` (`128K`) / `10` (`136K+`) | Stops FP8 projection caching if free unified memory (`MemAvailable`) approaches the reserve floor. |

### 3.5 4-Ring NVFP4 Per-Layer Embeddings (`PLE`)

| Selector (`FLUX_*` or `ATLAS_*`) | Value | Effect |
|---|:---:|---|
| `PLE_FAST_PLAN` | `1` | Enables pre-deinterleaved `PLE_CHUNK_CACHE` and `PleRowsPlan::Pipelined` (`0` `munmap` TLB shootdowns). |
| `PLE_SECTOR_READ` | `1` | Uses sector-aligned `O_DIRECT` NVMe reads for the 128 NVFP4 PLE shards. |
| `PLE_READ_RINGS` | `4` | Parallelizes NVMe PLE DMA across 4 asynchronous rings. |
| `PLE_READ_AFFINITY` | `big` | Pins PLE reader threads to the 10 Grace `Cortex-X925` performance cores (`cpus=5-9,15-19`). |
| `QWEN4_PLE_PREFILL_FUSED` | `1` | Fuses PLE dequantization and residual stream injection into a single kernel. |

### 3.5a Memory Selectors (opt-in, default off)

These free memory without changing what any kernel reads. Both are opt-in until a GPU identity run qualifies them.

| Selector | Value | Effect |
|---|:---:|---|
| `ATLAS_QWEN4_DENSE_DEDUP` | `1` | After each target layer is built, releases the checkpoint sources of its runtime NVFP4 copies: the attention and GDN MXFP8 originals and the attention BF16 dequants (requires `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1`), the hyper-connection mixer BF16 weights, the BF16 MoE router and the shared-expert MXFP8 originals. About 5.2 GiB on Flash-Next (3.69 GiB attention/GDN, 1.19 GiB mixers, 0.12 GiB routers, 0.23 GiB shared experts). The `mtp.*` layer is not touched. Refuses to start beside `ATLAS_QWEN4_DENSE_MXFP8=1` or `ATLAS_QWEN4_ROUTER_BF16=1`, which read those sources. Logs `QWEN4_DENSE_DEDUP_RELEASED` per layer and family. |
| `ATLAS_QWEN4_TEXT_ONLY` | `1` | Text-only Flash-Next: the vision tower (`model.visual.*`, 856 MiB BF16) is not loaded and image requests are refused (`unsupported_multimodal_model`). Text requests are served exactly as before. Logs `QWEN4_TEXT_ONLY_ENGAGED`. |

### 3.6 Speculative Decoding Selectors (`K=5` Inferact MTP & `K=16` DFlash)

| Selector (`FLUX_*` or `ATLAS_*`) | Value | Target Model | Effect |
|---|:---:|---|---|
| `QWEN4_MTP_CHAIN` | `1` | `Qwen3.8-Flash-Next` | Chains all 4 Inferact MTP draft steps on device without host synchronization. |
| `QWEN4_K5_HYBRID` / `K5_NATIVE_MOE` | `1` | `Qwen3.8-Flash-Next` | Enables native 5-row batched verification across GDN, QSA, mHC, and MoE layers. |
| `QWEN4_MTP_CONF` / `MTP_CONF_EARLY` | `0.6` / `1` | `Qwen3.8-Flash-Next` | Confidence-gated early draft truncation when draft entropy spikes. Set in `flux-v9-max-prefill.env` (`auto`, `max-prefill`, `256k`, `512k`, `1m`) since `9.15.3.1`, and in `flux-k5-lossless-decode.env` and `flux-estonia-136k.env`. `0` / `0` in `flux-v9-w4096-interactive.env` and bounded-transpose concurrency / native-262K BF16 recipes. The launcher forces `0` / `0` when `FLUX_CONCURRENCY > 1`. Engagement is logged once as `MTP_DRAFT_CONF_ENGAGED ... threshold=Some(0.6) early=true`. |
| `QWEN4_QSA_VERIFY_ROWS` | `1` (eligible only when frame endpoint `<=2,048`) | `Qwen3.8-Flash-Next` | Enables the specialized multi-row 0-ULP Flash-Decoding candidate (`partial_rows_v2` + `reduce_rows`) inside its source-enforced dense-endpoint boundary. Longer frames stay on native K=5 and use row-serial QSA; setting this flag cannot override that safety boundary. |
| `SSM_GDN_LAZY` | `1` | `Qwen3.8-27B` & `Flash-Next` | Skips intermediate GDN DRAM snapshot writes during verify (`+5.16 tok/s` on 27B, `0-ULP` bit-exact). |
| `SSM_GDN_SEQ_PERSISTENT` | `1` | `Qwen3.8-27B` | Keeps GDN recurrent state persistent across verify tiles (`+1.70 tok/s`). |
| `ATTN_QKV_FUSED` | `1` | `Qwen3.8-27B` | Fuses Q, K, V projections during DFlash verify (`+0.79 tok/s`). |
| `DFLASH_DRAFT_SPLITK` | `8` | `Qwen3.8-27B` | Split-K reduction in the DFlash drafter projections (`+1.1 tok/s`). |

### 3.7 Concurrency & Scheduling Selectors

No shipped recipe sets these, so the defaults below apply.

| Selector (`FLUX_*` or `ATLAS_*`) | Default | Effect |
|---|:---:|---|
| `QWEN4_NATIVE_BATCH` | unset (native) | Qwen3.8-Flash-Next decode route at `--max-batch-size` 2-32. Unset: one native batched forward per step; a frame that cannot batch runs C1 once per row and is counted by reason in `atlas_qwen4_decode_frames_total` (`qsa_required`, `moe_exact_rows_unavailable`, ...). A width the native kernels cannot run in this process (for example no exact LM-head tier, `ATLAS_TARGET_LMHEAD_VOCAB`, or a MoE layout without the exact row route) serializes with that reason, logged once at startup as `QWEN4_BATCH_PLAN widths=…`. `1`: native or refuse to start, including for any such width. `0`: always serial, and the only way to run `--max-batch-size` 33-128. |
| `QWEN4_NATIVE_BATCH_QSA` | unset (off) | Not GPU-qualified. `1`: requests that can reach 2,048 tokens (QSA-required) join the native frame as per-row QSA rows. Each QSA row runs the C1 QSA index and attention core inside the batched frame; everything else stays batched. Needs native batching and the QSA side index: with `QWEN4_NATIVE_BATCH=0`, a serial plan (for example without `ATLAS_MOE_T_LANES=1`), or without `--qwen4-qsa`, startup is refused. Values other than `0`/`1` are a startup error. |
| `PREFILL_INTERLEAVE` | unset (off) | Opt-in. `1`: while rows are decoding, at most one prefill chunk runs per scheduler tick, so a burst of new prompts stalls running streams for one chunk at a time instead of all at once. Unset or `0`: the legacy tick. Any other value logs an error and keeps the legacy tick. |
| `PREFILL_INTERLEAVE_DECODE_STEPS` | `1` | Decode steps between prefill chunks while rows are decoding (`1..=64`). Higher values favor running streams over time to first token for the burst. |
| `PREFILL_INTERLEAVE_TOKEN_BUDGET` | `0` (= `--max-prefill-tokens`) | Per-tick prefill token budget while rows are decoding, a multiple of 16. On the Qwen3.8-Flash-Next fast-prefill family the chunk stays `--max-prefill-tokens`, because a smaller chunk would drop the request off the fast path; MLA models and non-chunked serving ignore it. |
| `QWEN4_GRAPH_CATALOG` | `off` | CUDA-graph catalog for Qwen4 decode. `off`: today's lazy per-sequence capture. `prepare`: capture every declared decode graph before the server reports READY and never capture afterwards; a frame outside the catalog runs eagerly with a logged reason. `strict`: the same, but such a frame fails with `GRAPH_CATALOG_REFUSED reason=<r> key=<k>` (qualification runs). Any other value is a startup error. The policy ships first; it changes decode only once its decode wiring lands. |
| `QWEN4_GRAPH_CATALOG_QSA` | `0` | `1`: also declare C1 graphs for QSA-required rows (indexed-dense and sparse), only when QSA graph-safe mode is on. Native frames that hold QSA rows stay eager. Values other than `0`/`1` are a startup error. |
| `QWEN4_GRAPH_CATALOG_WIDTHS` | every width `2..=min(B,8)`, then `16`, `32` | Native frame widths to capture, comma-separated and strictly increasing, each in `2..=min(--max-batch-size, 32)`. |
| `QWEN4_GRAPH_CATALOG_MAX_MIB` | `512` | Memory cap for the prepared catalog. Over the cap (or below the `ATLAS_MEMAVAILABLE_FLOOR_MIB` floor), `strict` fails the load and `prepare` falls back to `off` with a log line. |
