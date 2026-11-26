// SPDX-License-Identifier: AGPL-3.0-only
// Gemma-4-31B W4A16 GEMM: shared NVFP4 body with SM120+ block-scaled FP8 MMA and 5-CTA launch bounds.
#define ATLAS_W4A16_USE_BLOCKSCALE_MMA 1
#define ATLAS_W4A16_GEMM_T_LAUNCH_BOUNDS __launch_bounds__(128, 5)
#include "../../common/shared/w4a16_gemm_nvfp4_v1.cuh"
