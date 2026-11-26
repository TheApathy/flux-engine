# recipe_bench

Measures a built-in recipe (`crates/spark-server/src/recipe/builtin/*.yaml`)
using the exact environment variables and CLI flags that the runtime generates,
avoiding hand-transcription errors (missing env vars, wrong port, or flag
formatting mismatches).

## Files

- `generate.py`: reads a recipe YAML, emits `env` (`KEY=VAL` lines) or `argv`
  (one token per line) mirroring `crates/spark-server/src/recipe/schema.rs`.
  `generate_test.py` covers env emission, flag renames, overrides, and
  presence-only boolean flags (`qwen4-qsa`, `dflash`, `no-tui`).
- `run.sh`: end-to-end benchmark harness that launches the server via
  `generate.py`, verifies the live server's `/proc/<pid>/environ` matches the
  generated `ATLAS_*`/`FLUX_*` variables, drops the first prefill/decode rep,
  and writes one `result.json` per model. Cleanup is registered via
  `trap cleanup EXIT INT TERM` (deliberately omitting `ERR` so routine nonzero
  checks like `grep -q` never kill a healthy server) and kills the server's
  entire process group on exit. Shared-host lockfile serialization
  (`RECIPE_BENCH_LOCK_FILE`), queue logging (`RECIPE_BENCH_QUEUE_FILE`), GPU
  idle waiting (`RECIPE_BENCH_WAIT_GPU=1`), and memory floors
  (`RECIPE_BENCH_MIN_MEM_GB`) are all opt-in via environment variables.
- `trap_test.sh`: CPU-only regression test verifying `run.sh`'s `trap` line and
  nonzero-exit survivability.
- `prompts/coding.json`, `prompts/chat.json`: the two decode probes (256
  tokens each, temperature 0).

## Usage

```bash
tools/recipe_bench/run.sh <lane> <label> <recipe.yaml> <model_dir> <model_name> <bin> \
  <prefill_req_2048.json> <prefill_req_8192.json> [port=8897]
```

The two prefill request files are pre-tokenized `/v1/completions` JSON bodies
(`prompt_token_ids`, `max_tokens: 1`) generated with the target model's
tokenizer.

Output: `tools/recipe_bench/results/<label>/result.json`: prefill tok/s at
2048 and 8192 (warm median of reps 2-4), decode tok/s + output hash on the
coding and chat prompts, and `env_proof_match` (whether `/proc/<pid>/environ`
on the live server matched what `generate.py` produced).

## Tests

```bash
cd tools/recipe_bench && python3 -m pytest generate_test.py -v
bash tools/recipe_bench/trap_test.sh
```

Both tests are CPU-only (no GPU, no server).
