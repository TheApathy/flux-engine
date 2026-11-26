// SPDX-License-Identifier: AGPL-3.0-only
//
// Tensor-core prefill attention at HDIM=512, for Gemma-4-31B's full-attention
// layers (32 q-heads, 4 kv-heads, head_dim 512).
//
// This is `kernels/gb10/common/inferspark_prefill.cu` (the BR=32 entry)
// instantiated at a second shape, not a copy. Only HDIM and BC differ; the
// default-shape PTX of the common file is byte-identical with and without the
// BC parameterisation this relies on.
//
// SHAPE. BC=16: at HDIM=512 a 32-row K/V tile needs ~132 KB of shared memory,
// BC=16 fits in 84,992 B. BR stays 32 (warp mapping). The host launches it
// with BR=32 (`ops::prefill_attention_512_tc`) and the same 13-arg ABI as
// `inferspark_prefill`.
//
// Upstream (atlas #721) measured this instantiation against the scalar
// `inferspark_prefill_512` at S=1024, 4 q-heads, 2 kv-heads, causal: cosine
// 0.999998, max abs 0.0039, 22.02 ms -> 0.34 ms.
//
// Opt-in: resolved only under ATLAS_GEMMA_PREFILL_512_TC=1.
#define HDIM 512
#define BC 16
#define ATLAS_SKIP_PREFILL_64 1
#define inferspark_prefill inferspark_prefill_512tc
#include "../../common/inferspark_prefill.cu"
