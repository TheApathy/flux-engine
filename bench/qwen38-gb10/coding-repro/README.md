# Qwen3.8-27B Coding Reproduction Suite

The canonical sanitized Qwen3.8-27B coding reproduction suite and its offline tests live at:

- Verifier, expected contract, manifest, and evidence: [`repro/qwen38-coding/`](../../../repro/qwen38-coding/) (`repro/qwen38-coding/run.py`)
- Offline unit tests: [`repro/tests/test_qwen38_coding.py`](../../../repro/tests/test_qwen38_coding.py)

The former duplicate copies in this directory were removed to eliminate path-depth drift (`Path(__file__).resolve().parents[2]` resolved to `bench/` instead of the repository root).
