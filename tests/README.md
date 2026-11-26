<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Test Suites &amp; Fixtures (<code>tests/</code>)</b></p>

# Integration & Smoke Tests (`tests/`)

End-to-end API compatibility, KV-cache dtype smoke tests, session isolation checks, and standalone CUDA kernel parity tests.

> Crate-level unit and integration tests live alongside each workspace crate in `crates/*/tests/` and run via `cargo test --workspace`. For performance and quality benchmarks (`llm-inference-bench`, Estonia 136K, and Weschera MinHeap), see [`bench/`](../bench/).

| File / Directory | Purpose |
|---|---|
| [`test_openai_compat_v2.py`](test_openai_compat_v2.py) | End-to-end OpenAI API (`/v1/chat/completions`, `/v1/responses`, streaming SSE, tool calls, rate limiting) smoke test. |
| [`test_session_isolation.py`](test_session_isolation.py) | Verifies zero Gated DeltaNet (GDN) recurrent state or KV-cache cross-talk under concurrent multi-session requests. |
| [`test_kv_dtype_smoke.py`](test_kv_dtype_smoke.py) | Smoke-tests every `KvCacheDtype` (`bf16`, `fp8`, `nvfp4`, `turbo3`, `turbo4`, `turbo8`) to prevent dispatch fallthrough regressions. |
| [`test_indexed_loader_admission_source.py`](test_indexed_loader_admission_source.py) | Offline source-contract and checked-arithmetic unit tests for sharded weight loader admission accounting. |
| [`cuda/`](cuda/) | Standalone CUDA kernel parity micro-tests (`dspark_asymmetric_attention_parity.cu`). |
| [`fixtures/`](fixtures/) | Fixed prompt fixtures (`128`, `512`, `1024`, `4096` tokens) and vision test image (`mona_lisa.jpeg`). |
