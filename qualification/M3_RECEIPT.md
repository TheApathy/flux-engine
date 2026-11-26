# Milestone 3 Qualification Receipt: Accuracy Qualification (Partial Fast-Track Scope)

**Milestone**: Milestone 3 — Accuracy Qualification (R1)  
**Verdict**: **PARTIAL** (Plain GSM8K Complete; Plain MMLU-Pro 340/1,000; Step C Native-MTP K=5 Parity Arm Skipped)  
**Timestamp**: 2026-10-25T20:36:00Z  
**Hardware**: NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory)  
**Model**: `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` with native MTP head  
**Binary Location**: `target/release/flux-server`  
**Binary SHA-256**: `a3b49aa3e1e3eb41b20bc28c4193026522865ee083d2d3e0cc1135d333da8b5e`  
**HEAD Commit**: `b47b9fe6b0c5826df769277bb306f5b41e2372c7`  

---

## 1. Executive Summary & Verification Context

Milestone 3 records the partial fast-track accuracy evaluation for the `flux-engine` `v0.9.14` release container under an exclusive GPU lock.

Key empirical findings:
1. **Plain GSM8K Complete**: Evaluated all 1,319 GSM8K test items on the release binary (`flux-engine:r17-candidate`). 1,319 / 1,319 succeeded (`100% OK`), 1,283 correct, achieving **97.27% accuracy**.
2. **Zero Interventions**: Strictly 0 non-empty intervention events across all 1,319 GSM8K and 340 MMLU-Pro items (`atlas.interventions.v1` ledger 100% compliant and empty).
3. **Memory Floor Strictly Maintained**: Across 43,756 continuous 250ms samples, the lowest host available memory observed was **11,430.0 MiB** (well above the required $\ge 6,656\text{ MiB}$ floor, with > 4.7 GiB safety margin).
4. **Partial Scope Note (2026-10-25T20:34:36Z)**: Plain MMLU-Pro was concluded early at 340 / 1,000 items and the Step C Native-MTP K=5 parity arm was skipped to release the exclusive GPU lock for subsequent GPU gating. Accordingly, this receipt covers Plain GSM8K only and does **not** certify full MMLU-Pro or K=5 parity.

---

## 2. Summary of Accuracy Verdict

| Benchmark | Arm | Items Evaluated | Status | Correct | Score / Accuracy | Interventions | Notes |
|---|---|---:|---|---:|---|---:|---|
| **GSM8K** | Plain | 1,319 / 1,319 | COMPLETE | 1,283 | **97.27% (0.9727)** | 0 | 100% execution success (1,319/1,319 OK) |
| **MMLU-PRO** | Plain | 340 / 1,000 | PARTIAL | — | — | 0 | Stopped early at 340 items |
| **GSM8K / MMLU-PRO** | Native-MTP K=5 | 0 | SKIPPED | — | — | — | Not executed in this fast-track run |

---

## 3. Zero-Intervention Ledger Attestation (`atlas.interventions.v1`)

- Total requests audited: 1,659 (1,319 GSM8K + 340 MMLU-Pro)
- Schema compliance: 100% `atlas.interventions.v1`
- Policy compliance: 100% `report_only`
- Events compliance: 0 non-empty events

---

## 4. Memory Floor & Box Safety Compliance

- Pre-flight `MemAvailable`: >= 100 GiB before launch
- Lowest measured host memory: `11,430.0 MiB` (> 4.7 GiB margin above 6,656 MiB floor)
- Watchdog kills: 0
- Samples recorded: 43,756 samples (250ms sampling interval)
- Exclusive hardware lock: acquired during execution and cleanly released on completion

---

## 5. Artifact Provenance & Checksums

| Artifact | File Path | Exact SHA-256 Digest |
|---|---|---|
| Release Binary | `target/release/flux-server` *(historical build digest)* | `a3b49aa3e1e3eb41b20bc28c4193026522865ee083d2d3e0cc1135d333da8b5e` |
| Plain Recipe | `qualification/accuracy/recipe-plain.env` | `10c47705b5db6a69be612229ffe90ff35d9588c588b80b610da90f65c675aa95` *(historical pre-sanitization: `c5c2c7225ef1714a29d21d1419ad06355516bfd43600ca9057624b56b649e3ec`)* |
| Native K5 Recipe | `qualification/accuracy/recipe-k5.env` | `f595a0cdcfbd1822b9a358cef34c5e2ceddf83f9efc7c7a15901cf3d34fd08cb` *(historical pre-sanitization: `c6bc49bf832788cfe2f153e28f39ff88b3cdf4d53a19f7c88ef74ad811201de4`)* |
| Plain GSM8K Report | `qualification/accuracy/reports/plain-gsm8k.json` | `f9381bb9ec99500c9e925fea6aa78edaad8a8db6ae22accb6554503da83a78dd` *(historical pre-sanitization: `551aa5c3dcf90be675dff377430c3a212827c0405915a43caa4c88fef5294b85`)* |
| Milestone 3 Receipt JSON | `qualification/M3_RECEIPT.json` | *(Self-referenced)* |
