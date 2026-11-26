<p align="left"><a href="../../README.md"><b>← Flux Engine</b></a> · <a href="../README.md"><b>Documentation (<code>docs/</code>)</b></a> · <b>Model &amp; Hardware Compatibility Matrices (<code>docs/compatibility/</code>)</b></p>

# Model & Hardware Compatibility Matrices (`docs/compatibility/`)

Machine-readable compatibility matrices covering all 14 supported model families and 5 hardware compute targets.

---

## Matrix Files

| File | Format | Contents |
|---|---|---|
| [`model-matrix.csv`](model-matrix.csv) & [`model-matrix.yaml`](model-matrix.yaml) | CSV / YAML | Supported model architectures, parameter counts, quantization formats (`NVFP4`, `FP8`, `EXL3`), speculative drafters (`Inferact MTP`, `DFlash`), and `--kernel-target` identifiers. |
| [`hardware-matrix.yaml`](hardware-matrix.yaml) | YAML | Hardware capability matrix across `NVIDIA GB10 (sm_121a)`, `NVIDIA RTX PRO 6000 / 5090 (sm_120a)`, `NVIDIA B200 (sm_100a)`, `NVIDIA H100 (sm_90a)`, and `Apple Metal`. |
