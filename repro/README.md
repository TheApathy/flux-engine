<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Sanitized Endpoint Reproduction Suites (<code>repro/</code>)</b></p>

# Sanitized Endpoint Reproduction Suites (`repro/`)

Self-contained, zero-dependency Python 3 endpoint verifiers, SHA-256 manifest contracts, and certified JSON evidence receipts for validating a running `flux-server` instance over HTTP (`/v1/chat/completions`).

---

## Directory Index

| Suite / Directory | Target Model & Gate | Certified Floor / Contract | Primary Files |
|---|---|---|---|
| [`qwen38-flash-next/`](qwen38-flash-next/) | `Qwen3.8-Flash-Next-NVFP4` (`v9` & `v9-w4096`) on GB10 | `16K` prefill `>= 5,900 tok/s` (`6,082.2` peak; `128K >= 5,500 tok/s`), `K=5` decode `>= 51.0 tok/s` (`56.5` server med, `LOSSLESS 10/10`), `136.5K` Estonia `5/5 PASS` | [`run.py`](qwen38-flash-next/run.py) · [`expected.json`](qwen38-flash-next/expected.json) · [`manifest.json`](qwen38-flash-next/manifest.json) · [`evidence/`](qwen38-flash-next/evidence/) |
| [`qwen38-coding/`](qwen38-coding/) | `Qwen3.8-27B-NVFP4` + `DFlash` coding & MinHeap probe | Single-stream decode `>= 40.0 tok/s` (`40.9466 tok/s` reference; `63.9 tok/s` on `v2/BF16-KV`) with deterministic SHA-256 token verification | [`run.py`](qwen38-coding/run.py) · [`expected.json`](qwen38-coding/expected.json) · [`manifest.json`](qwen38-coding/manifest.json) |
| [`tests/`](tests/) | CPU-only offline unit tests (`pytest` / `unittest`) | Validates schema, hash comparison, and failure-mode gates in `< 2 s` without a GPU | [`test_qwen38_flash_next.py`](tests/test_qwen38_flash_next.py) · [`test_qwen38_coding.py`](tests/test_qwen38_coding.py) |

## Quick Usage

```bash
# Run offline unit tests (no GPU required)
python3 -m unittest discover -s repro/tests -p "test_*.py" -v

# Verify a live Qwen3.8-Flash-Next server on port 8899
python3 repro/qwen38-flash-next/run.py --endpoint http://127.0.0.1:8899/v1/chat/completions
```
