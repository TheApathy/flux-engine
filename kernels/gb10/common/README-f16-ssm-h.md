# Status of the `*_f16.cu` SSM H-State Files

All six `.cu` files listed below are compiled automatically into every GB10 PTX
target by `crates/atlas-kernels/build_hw.rs`. Runtime dispatch is controlled by
the `ATLAS_SSM_H_FP16` environment variable (there is no `--ssm-h-dtype` CLI
flag):

- **Wired in production (`ATLAS_SSM_H_FP16`):** `gdn_f16_state.cuh`,
  `ssm_h_dtype.cu`, `gated_delta_rule_wy_f16.cu`,
  `gated_delta_rule_wy3_f16.cu`, and `gated_delta_rule_wy4_f16.cu` are loaded
  and dispatched by `spark-model` (`qwen3_next` / `qwen3_ssm`).
- **Unwired prototypes:** `gated_delta_rule_wy2_resident_f16.cu` and
  `gated_delta_rule_wy3_resident_f16.cu` are not referenced by Rust dispatch.

## Architectural Constraints

1. **Incomplete kernel coverage.** These files cover only the stage-2 MTP-verify
   twins and dtype converters, not the decode-path fused-norm FP16 variants.
2. **Incompatible with DFlash/DDTree FP32 kernels.** The DFlash/DDTree GDN
   h-state readers and writers (`gated_delta_rule_tree`,
   `gated_delta_rule_tree_wy`, and the `wy17` family) are FP32-only. Reading an
   FP16 state pool with an FP32 kernel does not fault—it silently corrupts state.
   Any future FP16 h-state integration must hard-gate off whenever DFlash/DDTree
   is active.
3. **Bandwidth impact.** The GDN h-state is 144 MiB per full pass against
   ~24.6 GB of weights per step, so halving state width saves <=0.6% of traffic
   at K=1 and <=1.5% at K=4.
