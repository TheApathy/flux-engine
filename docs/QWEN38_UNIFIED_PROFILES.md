# Qwen3.8 unified launch profiles

These built-in profiles are deliberately separate contracts. Their receipts
prove configuration identity only. They do not prove model quality, speculative
engagement, throughput, stability, or endurance.

## Flash-Next deterministic prefill

Recipe id: `qwen3.8/qwen3.8-flash-next-prefill4k-champion`.

The recipe carries the complete 58-key effective environment captured by the
deterministic five-context qualification run. Its canonical environment receipt
is `900beae6f1d5f00a1a5f8864b073f1a8dc0457abb08fe0819f3b0bc9fc6624ab`.
The resident FP8 projection cache is explicitly on. This is the memory-marginal
high-prefill contract and must not be merged with the native-MTP settings.

## Flash-Next memory-safe native MTP

Recipe id: `qwen3.8/qwen3.8-flash-next-native-mtp-safe-local`.

The resident FP8 projection cache is explicitly off, its reserve variable is
forbidden, prefix caching is off, and native-MTP uses its full vocabulary. A
conflicting inherited setting causes launch refusal before model teardown.
The 84-key `ATLAS_*` subset has raw-file SHA-256
`83dc882520091ceed1a69416ff0aa6bf232e34d3b97df3966dbffc2a9a577a2f` and canonical receipt
`281a35c55b3f792d973b9bc1e0c4ca1dd1c8bbfe87fcbd3552eaf7b5f6306755`.
This identifies the candidate configuration. It is not a runtime qualification.

> **Note (`v1.0.x` production default):** For unified high-throughput prefill (`w16384` or `w4096`) combined with lossless `K=5` MTP or auto-attached `DFlash2` speculative decoding on a single 128 GB GB10 node, use `FLUX_MODE=auto` (which selects [`recipes/flux-nextgen-max-prefill-decode.env`](../recipes/flux-nextgen-max-prefill-decode.env) and `MODEL.toml` defaults) or the standalone profiles in [`recipes/`](../recipes/) (`flux-v9-max-prefill.env` / `flux-v9-w4096-interactive.env`).

## Dense-Qwen DFlash V14 and V15

The two cards have distinct recipe ids and compile-time build tracks:

- `qwen3.8/qwen3.8-27b-dflash2-v14-safe-local` requires
  `dense-qwen-dflash-v14`.
- `qwen3.8/qwen3.8-27b-dflash2-v15-safe-local` requires
  `dense-qwen-dflash-v15`.

The runtime refuses either profile from a binary sealed for the other track.
That prevents a V14 qualification result from being attributed to pending V15
source, even though their launch environments currently match.

## Sealed dual-target build

`scripts/build-qwen38-dual-profiled.sh` requires the exact source commit, tree,
and DFlash generation. It compiles only:

```text
gb10/qwen3.8-27b/nvfp4
gb10/qwen3.8-flash-next/nvfp4
```

For a committed V15 build:

```bash
scripts/build-qwen38-dual-profiled.sh \
  --source /absolute/clean/worktree \
  --commit V15_40_HEX_COMMIT \
  --tree V15_40_HEX_TREE \
  --dflash-generation v15
```

The output directory is create-new-only and contains a mode-0555, single-link
binary, an immutable build log, and `build-receipt.json` binding source, target
census, profile file hashes, canonical environment receipts, binary SHA-256, and
GNU build id.
