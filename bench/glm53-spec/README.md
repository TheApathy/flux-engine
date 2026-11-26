# GLM-5.3 DFlash2 speculative decode: verify-cost fixes and their gates

Result (window w3, 2026-09-23, EXL3 2.05 bpw, greedy, 160 tokens, 3 prompts x 2 trials,
fresh server per arm, target arm first and last):

| arm | short | long | prose | output |
|---|---|---|---|---|
| target-only (T, T2) | 14.09 / 14.51 | 12.23 / 12.55 | 14.13 / 14.54 | reference |
| DFlash2 gamma 3, no fixes (w1 S0) | 7.86 | 9.79 | 8.16 | identical |
| DFlash2 gamma 3, all fixes (S4) | 16.26 | 16.72 | 16.70 | identical |
| DFlash2 gamma 2, all fixes (S4g2) | 16.56 | 16.23 | 17.44 | identical |

All numbers run under `ATLAS_GLM53_UNVALIDATED_BRINGUP=1` (GLM kernel admission is closed).

## The fixes (all default off)

- `ATLAS_GLM53_EXL3_ROWBATCH=1`: the exact-verify EXL3 projection ran the pinned M=1 GEMM once
  per row inside one cooperative launch (weights re-read and re-decoded per row). It now runs
  every row in one pass of the same inner kernel. Bit-identical per row by construction.
- `ATLAS_GLM53_VERIFY_DSA_PRECOMPUTE=1`: DSA row-local inputs (q/kv/indexer projections, norms,
  key absorption) computed once for all verify rows before the causal per-row loop.
- `ATLAS_GLM53_VERIFY_DSA_BATCH_OUTPUT=1`: value absorption and output projection deferred out
  of the per-row loop and run once over all rows.
- `ATLAS_GLM53_PREFIX_COMMIT=1`: a partial acceptance commits
  per-row KDA/DSA snapshots instead of replaying the accepted prefix through the target. The
  original re-staged every KDA layer's conv shift register from the verify scratch, which all
  34 layers share, i.e. from the LAST layer's inputs. Each layer's conv inputs are now saved.

## Gates

- `harness/rowbatch_equiv.cu`: rowbatch vs rowexact vs M=1, 4 tile shapes x rows 1..8, memcmp;
  negative control perturbs one input row and requires only that row to change.
- `score.py <run> <ref-arm>`: every trial's text must equal the reference arm's; a perturbed
  prompt must differ (the gate can fail); phase receipts give the per-step decomposition and
  per-position acceptance.
- `statehash_compare.py`: `ATLAS_GLM53_STATE_HASH=<file>` hashes the committed conv and
  recurrent state of KDA layers 0/17/33 after every commit; spec must equal the target walk at
  every shared position. Control: `ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE=1` re-enables
  the original bug and must fail (it does, from the first prefix commit, on layers 0 and 17 but
  not 33 -- the last layer's own inputs).

## Flux P2 + P3 runbook (flux-engine, after v0.9.15.5)

Both scripts build every server's env and argv from the shipping built-in recipes, so the run tests
what ships:
- `glm-5.3-flash-exl3-local.yaml` for T arms;
- `glm-5.3-flash-exl3-dflash2-local.yaml` for S arms.

Each arm starts a fresh server with only that env. Variables from the caller's shell pass through only
via `--pass-env`. Both scripts fail closed: `status.txt` line 1 is PASS or FAIL with reasons, and line 2
binds the server sha256.

1. **P2, output identity.** About 1.5 h: 7 arms × 5 prompts × 2 trials plus warm-up, on one Spark,
   alone.
   ```
   python3 ts_identity.py --server /opt/flux/bin/flux-server-glm53 \
     --model /models/GLM-5.3-Flash-exl3-2.05bpw --draft-model /models/GLM-5.3-Flash-DFlash2-bf582e4 \
     --out /results/glm53/p2
   ```
   The default arms are T1, S1, T2, S2, T3, S3, T4. The last T and the last S also run the perturbed
   control. It passes only if:
   - every S trial equals T1 byte-for-byte;
   - the T arms agree with each other;
   - each control differs from its short output.
2. **P3, state identity and negative controls.** About 2 h: arms Th, S3h, S2h, Cleg and Cext at
   `--max-seq-len 4096`, with long2k.
   ```
   python3 p3_state.py --server ... --model ... --draft-model ... --out /results/glm53/p3
   ```
   It passes only if:
   - S3h and S2h match Th's per-layer state (`statehash_compare.py`) and its output;
   - both controls fail the state gate and change some output.
3. **Kernel equivalence.** On the same Spark, run `harness/run.sh /results/glm53/harness`. It compiles
   `rowbatch_equiv.cu` from repo-relative paths against `ATLAS_EXLLAMAV3_SOURCE` (pinned e648f1a1),
   **executes** it, and writes `receipt.json`.
   - The receipt records the harness and kernel source sha256, the speculation source hash, the exllamav3
     revision, the nvcc and binary hashes, the build and run exit codes, the result line and the GPU.
   - `COMPILE_ONLY=1` builds without a GPU, but its receipt says `executed=false` and cannot admit.
4. **Evidence.**
   ```
   python3 make_evidence.py --flux /results/glm53/p2 /results/glm53/p3 /results/glm53/harness/receipt.json evidence.json
   ```
   - The builder is strict (`flux_evidence.py`). It recomputes everything from raw receipts:
     - every response file's hash, finish reason and token count;
     - the exact P2 and P3 arm sets, and the status bindings against the arm binaries;
     - the state results, from the raw `.hash` files;
     - controls with complete responses that actually differ;
     - one binary across all arms, with every S arm's logged `GLM53_SPEC_SOURCES` equal to this tree's
       manifest hash;
     - a harness receipt that was built and executed on a GB10 with exit 0, 0 failing cases, and source
       hashes equal to this tree. A log containing the PASS text is not accepted.
   - On any failure it exits 2 and writes nothing.
   - Copy the values into `GLM53_SPECULATIVE_EVIDENCE` in `speculative_admission.rs`. The field-for-field
     test enforces the match.
   - Rebuild. `validate()` also requires the record's source hash to equal the manifest compiled into the
     binary (`spec_sources!`, 32 files on the speculation path). Embedding the evidence doesn't change that
     hash, because `speculative_admission.rs` is not in the manifest.
   - Then rerun P2 with `--admitted` (no bring-up escape) on the rebuilt binary. That is the admission
     receipt.

The checked-in record is a pending flux record. It fails `validate()`, so `install_dflash2` refuses
speculation without `ATLAS_GLM53_UNVALIDATED_BRINGUP=1`. The apathy w4/w5 record
(`evidence.apathy-w4w5.json`) is history. Its `engine` is not `flux-engine`, so it cannot admit flux.

Offline tests (no GPU; `tests/mock_flux_server.py` stands in for the server):
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider bench/glm53-spec/tests/test_ts_identity.py`
runs 48 tests, including adversarial cases: foreign-source binary, tampered raw response, foreign
status binding, missing arm, missing hash file, and empty control.
