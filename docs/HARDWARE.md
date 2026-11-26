# Adding a new hardware target or model family

Flux Engine's compute stack is structured around **(hardware, model, quant) tuples**.
Each tuple is a self-contained body of work: kernels are written, tuned, and
tested per-tuple. This document explains how to extend the matrix.

## Directory layout

```
kernels/
└── <hardware>/                    e.g. gb10
    ├── HARDWARE.toml              arch, sm, fp32-residual flag
    ├── <quant>/                   shared kernels for this hw + quant
    │   └── *.cu                   e.g. nvfp4/dense_gemm.cu
    └── <model>/                   per-model overrides
        ├── MODEL.toml             model_type list, sampling presets, behavior
        └── <quant>/               per-(model, quant) overrides
            └── *.cu               e.g. qwen3.6-35b-a3b/nvfp4/inferspark_prefill_h128.cu
```

The build script (`crates/atlas-kernels/build.rs`) walks this tree and
compiles every `.cu` to PTX. Model-specific files override shared
files when a name collision occurs.

## Adding a new model family

If your model is similar to an existing one (e.g., adding Qwen3.7 to the
Qwen3.5/3.6 family), the mechanical recipe is:

1. **Create the kernel target dir**:
   ```
   kernels/gb10/qwen3.7-XXB/
   ├── MODEL.toml                  copy from a similar model's MODEL.toml
   └── nvfp4/                      or fp8/ etc. depending on the quant
       └── (per-target overrides — leave empty if shared kernels suffice)
   ```

2. **Write `MODEL.toml`**:
   ```toml
   [model]
   name = "qwen3.7-XXB"
   hf_id = "Qwen/Qwen3.7-XXB"
   params = "XXB"
   active_params = "XXB"
   architecture = "Hybrid Attention + GDN + Dense FFN"

   [[model_types]]
   model_type = "qwen3_5"          # what the HF config.json says
   hidden_size = NNNN              # exact hidden dim — wins over wildcards

   [behavior]
   default_num_drafts = 1
   max_thinking_budget = 512
   thinking_in_tools = false

   [sampling.thinking_text]
   temperature = 0.6
   top_p = 0.95
   top_k = 20

   # ... (other sampling presets — see existing MODEL.toml files)
   ```

3. **Wire to a `WeightLoader`** in `crates/spark-model/src/factory.rs`:
   most Qwen3-family models share `Qwen35WeightLoader` for MoE and
   `Qwen35DenseWeightLoader` for dense FFN. Pick the right one based on
   whether the model has experts.

4. **Add regression coverage** under `repro-kit/tests/` and `tests/` (with and
   without speculative decoding, and `EP=2` if applicable).

5. **Build with the wildcard target**:
   ```
   ATLAS_TARGET_MODEL='*' cargo build --release -p spark-server
   ```
   The new target compiles into the binary. Runtime selects it via
   `model_type` + `hidden_size` matching.

If your model is genuinely new (different attention pattern, novel SSM
variant, etc.), you'll also need to:
6. Write a per-architecture `TransformerLayer` impl in
   `crates/spark-model/src/layers/` (mirror the structure of
   `qwen3_attention/` or `qwen3_ssm/`).
7. Add a new `WeightLoader` in `crates/spark-model/src/weight_loader/`
   if the safetensors key naming differs from existing families.

## Adding a new hardware target

Flux Engine currently ships CUDA kernel targets for **GB10 (Grace Blackwell, `sm_121a`)**,
**discrete Blackwell (`sm_120a` / RTX 50-series & RTX PRO 5000/6000)**, and **Phase-A Blackwell Ultra (`sm_103a` / GB300; see [`SM120_SM103_BRINGUP.md`](SM120_SM103_BRINGUP.md))**,
plus an experimental **Apple Metal** backend ([`METAL_BACKEND.md`](METAL_BACKEND.md)). Adding another
CUDA target (for example `sm_100` for Blackwell B200, or `sm_90` for Hopper H100/H200) requires:

1. **`kernels/<new-hw>/HARDWARE.toml`** (matching the schema in `kernels/gb10/HARDWARE.toml` and `kernels/rtx6000/HARDWARE.toml`):
   ```toml
   [hardware]
   name = "rtx6000"
   vendor = "nvidia"
   arch = "sm_120a"
   extends = "gb10"
   compute_capability = [12, 0]
   memory_bandwidth_gbps = 1792
   memory_type = "gddr7"
   memory_gb = 96
   ```

2. **Per-quant kernel directory**: `kernels/<new-hw>/nvfp4/`,
   `kernels/<new-hw>/fp8/`. Most kernels can be ported from `gb10/` but
   need recompilation with the new SM target. Tile shapes, SMEM budget,
   and tensor-core MMA instructions may need tuning.

3. **`atlas-kernels/build.rs`**: usually no changes needed: the build
   script auto-discovers new `kernels/<hw>/` directories.

4. **`spark-runtime/src/cuda_backend.rs`**: if the hardware has different
   capabilities (e.g., GDS supported, no NVLink, different RDMA NIC),
   wire the relevant flags here.

5. **NCCL env**: launchers like `scripts/start-ep2.sh` default
   `NCCL_SOCKET_IFNAME` to `enp1s0f0np0` (GB10's RDMA NIC), overridable via
   environment variable for your host's interconnect.

6. **CI**: GitHub Actions runs on `ubuntu-latest` with `ATLAS_SKIP_BUILD=1`
   so no GPU is needed. The new target compiles via the wildcard build
   on a host with the right SM.

## Adding a new quantization scheme

Flux Engine supports NVFP4 (E2M1 + FP8 scales), FP8 block-scaled, BF16 raw.
To add a new scheme (e.g., MX4, INT4):

1. **`crates/atlas-core/src/config.rs`**: extend the quant detection
   logic to recognize the new format from `quantization_config` in
   `config.json`.
2. **`crates/spark-model/src/weight_map/`**: add a loader function
   that produces the right `QuantizedWeight` variant.
3. **Per-model kernels**: write `*.cu` for the new quant under
   `kernels/gb10/<model>/<new-quant>/`. The build script auto-picks them.
4. **Dispatch**: `crates/spark-model/src/layers/<layer>/` per-quant
   branches in the forward path.

## Testing a new target

Once compiled:

```bash
# Smoke test (container listens on 8899 by default; mount local checkpoints at /models)
docker run --rm --gpus all --ipc=host -p 127.0.0.1:8899:8899 \
  -v "$MODELS_ROOT:/models:ro" \
  ghcr.io/theapathy/flux-engine:v1.0.3 \
  serve --model /models/<new-model-dir> --max-seq-len 4096

curl http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'

# OpenAI compatibility & session isolation tests
python3 tests/test_openai_compat_v2.py --url http://127.0.0.1:8899
python3 tests/test_session_isolation.py --url http://127.0.0.1:8899

# CPU-only contract & unit suite
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test --workspace --lib
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest repro-kit/tests/
```

## Reference implementations

When in doubt, copy from a model with similar arch:

| New model is... | Look at |
|---|---|
| Hybrid SSM + attention MoE | `qwen3.5-35b-a3b/`, `qwen3.6-35b-a3b/` |
| Hybrid SSM + attention dense | `qwen3.5-27b/`, `qwen3.6-27b/` |
| Pure Mamba2 + MoE | `nemotron-3-nano-30b-a3b/` |
| Pure attention + MoE | `mistral-small-4/`, `minimax-m2-229b/` |
| Pure attention dense | `gemma-4-31b/` |
| Vision-language | `qwen3-vl-30b-a3b/` |
