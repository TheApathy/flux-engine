// SPDX-License-Identifier: AGPL-3.0-only
//
// Manifold HyperConnection (mHC) cluster-fused combine / inject / upmix GEMM
// kernels for Flash-Next.
//
// Modularized manifest: compile-time model shape traits live in
// `qwen4_geometry.cuh`, and kernel families live under `detail/hc_combine/`.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "qwen4_geometry.cuh"

#include "detail/hc_combine/hc_common.cuh"
#include "detail/hc_combine/hc_cluster_bm64.cuh"
#include "detail/hc_combine/hc_gatepre.cuh"
#include "detail/hc_combine/hc_2cta_bm32.cuh"
#include "detail/hc_combine/hc_cluster_fuse.cuh"
#include "detail/hc_combine/hc_w4_strip.cuh"
#include "detail/hc_combine/hc_bm64_upmix.cuh"
#include "detail/hc_combine/hc_bm32_nvfp4.cuh"
#include "detail/hc_combine/hc_bm32_mxfp8.cuh"
