# `llm-inference-bench` (`v0.7.9`) 0..128K Context & Decode Sweep (`v9-w4096` / `interactive`)

- **Model:** `qwen38-native-mtp-frontier-v1` (`<MODELS_ROOT>/Qwen3.8-Flash-Next-NVFP4-Offload` + `<MODELS_ROOT>/Qwen3.8-Flash-Next-Inferact-MTP`)
- **Operating Recipe:** [`recipes/flux-v9-w4096-interactive.env`](../../../recipes/flux-v9-w4096-interactive.env) (`FLUX_MODE=interactive-w4096`, `WIDE=4096`, `K=5` Inferact MTP, `11,907 – 12,695 MiB` free UMA floor)

| Context (`tokens`) | LIL Prefill (`tok/s`) | LIL `C=1` Decode (`tok/s`) | Server `Done:` Decode (`tok/s`) | MTP Accept Len | Output Len Avg | Re-Prefill Windows |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | — | `42.8` | `43.0` | `2.34` | `1,519` | `0/3` |
| `8,192` | `4,641` | `42.7` | `42.7` | `2.26` | `1,440` | `0/3` |
| `16,384` | `4,676` | `43.5` | `43.1` | `2.41` | `1,424` | `0/3` |
| `32,768` | `4,663` | `43.2` | `43.1` | `2.31` | `1,447` | `0/3` |
| `65,536` | `4,634` | `42.1` | `41.6` | `2.33` | `1,392` | `0/3` |
| `131,072` | `4,501` | `40.7` | `41.9` | `2.16` | `1,354` | `0/3` |

- **Coding Peak (`C=1`, 5 runs):** **`56.3 tok/s` median / `57.4 tok/s` max** (`accept_len = 3.02`).
- **Zero re-prefill windows (`0/18`):** Every 30 s sustained-decode window completed without an mid-window re-prefill, so LIL client `tok/s` and server `Done:` `tok/s` agree within `0.4 tok/s` across `0..64K`.
