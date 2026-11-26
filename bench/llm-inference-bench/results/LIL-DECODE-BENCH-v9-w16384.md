# `llm-inference-bench` (`v0.7.9`) 0..128K Context & Decode Sweep (`v9` / `w16384`)

- **Model:** `qwen38-native-mtp-frontier-v1` (`<MODELS_ROOT>/Qwen3.8-Flash-Next-NVFP4-Offload` + `<MODELS_ROOT>/Qwen3.8-Flash-Next-Inferact-MTP`)
- **Operating Recipe:** [`recipes/flux-v9-max-prefill.env`](../../../recipes/flux-v9-max-prefill.env) (`FLUX_MODE=max-prefill`, `WIDE=16384`, `K=5` Inferact MTP)
- **Raw JSON Receipt:** [`lil-decode-bench-v9-w16384.json`](lil-decode-bench-v9-w16384.json)

| Context (`tokens`) | LIL Prefill (`tok/s`) | LIL `C=1` Decode (`tok/s`) | Server `Done:` Decode (`tok/s`) | MTP Accept Len | Output Len Avg | Re-Prefill Windows |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | — | `42.2` | `42.2` | `2.15` | `1,485` | `0/3` |
| `8,192` | `5,342` | `42.9` | `42.6` | `2.22` | `1,423` | `1/3` |
| `16,384` | `5,754` | `40.9` | `40.8` | `1.95` | `1,372` | `0/3` |
| `32,768` | `5,722` | `39.6` | `39.2` | `1.97` | `1,291` | `1/3` |
| `65,536` | `5,627` | `39.0` | `40.3` | `1.95` | `1,313` | `1/3` |
| `131,072` | `5,508` | `39.0` | `40.1` | `1.92` | `1,307` | `0/3` |

- **Coding Peak (`C=1`, 5 runs):** **`52.1 tok/s` median / `58.7 tok/s` max** (`accept_len = 3.15`).
- **Re-prefill windows note:** `3` of `18` `C=1` decode windows had a request finish inside LIL's 30 s sustained-decode window (hitting natural EOS), so the next request's full-context prefill ran inside that 30 s window. LIL's client window `tok/s` for those cells includes that re-prefill time, whereas the server's `Done:` decode `tok/s` isolates pure token generation.
