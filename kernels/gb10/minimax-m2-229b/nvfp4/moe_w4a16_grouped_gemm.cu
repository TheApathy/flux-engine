// SPDX-License-Identifier: AGPL-3.0-only
// MiniMax-M2 grouped W4A16 MoE GEMM: shared NVFP4 base plus M=128 dynamic-SMEM extension.
#include "../../common/shared/moe_w4a16_grouped_gemm_nvfp4.cuh"

// ═══════════════════════════════════════════════════════════════════
// K64 fused gate+up MoE GEMM with M=128 tile.
//
// Doubles M_TILE from 64 → 128: halves block count for large prefill,
// reducing launch amortization on 48-SM GB10. Same K=64 cp.async pipeline
// + BF16→FP8 SMEM packing + m16n8k32 FP8 MMA as parent kernel.
//
// Block dim: 256 threads (8 warps × 32 lanes), each warp covers 16 M rows.
// Dynamic SMEM: 58,048 B (vs 39,360 B static at M=64) — exceeds 48 KiB static
// cap, so host must set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES >= 58048
// and pass shared_mem_bytes = 58048.
//
// On MiniMax-M2 (256 experts, top_k=8, N=1536 -> 24 fused N-tiles) at 4096-token
// prefill (~128 tokens/expert average):
//   M=64:  2 M-blocks/expert × 256 experts × 24 N-tiles = 12,288 blocks
//   M=128: 1 M-block/expert  × 256 experts × 24 N-tiles =  6,144 blocks
// Halved block count reduces launch overhead and weight re-reads.
// ═══════════════════════════════════════════════════════════════════

#define M_TILE_M128 128

extern "C" __global__ void moe_w4a16_fused_gate_up_t_k64_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ C_gate,
    __nv_bfloat16* __restrict__ C_up,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE_M128;
    if (cta_m_local >= M_expert) return;

    const unsigned int tiles_per_proj = (N + N_TILE_LG - 1) / N_TILE_LG;
    const bool is_up = (blockIdx.x >= tiles_per_proj);
    const unsigned int cta_n = (is_up ? (blockIdx.x - tiles_per_proj) : blockIdx.x) * N_TILE_LG;

    const unsigned char* B_expert;
    const unsigned char* S_expert;
    float scale2;
    __nv_bfloat16* C;
    if (is_up) {
        B_expert = (const unsigned char*)up_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)up_scale_ptrs[expert_id];
        scale2 = up_scale2_vals[expert_id];
        C = C_up;
    } else {
        B_expert = (const unsigned char*)gate_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)gate_scale_ptrs[expert_id];
        scale2 = gate_scale2_vals[expert_id];
        C = C_gate;
    }

    if (B_expert == 0) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int warp_id = threadIdx.x / 32;        // 0..7
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;     // 0,16,...,112 → 8×16=128
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Dynamic shared memory (58,048 B total, all slices 16-byte aligned):
    //   smem_A_m128:     2 * 128 * 72 * 2 = 36864 B (offset     0)
    //   smem_Bp_m128:    2 * 32  * 144    =  9216 B (offset 36864)
    //   smem_Bs_m128:    2 * 4   * 144    =  1152 B (offset 46080)
    //   smem_B_fp8_m128: 128 * 80         = 10240 B (offset 47232)
    //   smem_LUT_m128:   16 * 4           =    64 B (offset 57472)
    //   smem_tok_m128:   128 * 4          =   512 B (offset 57536)
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto smem_A_m128 = reinterpret_cast<__nv_bfloat16 (*)[M_TILE_M128][K_STEP_T64 + PAD_T64]>(smem_raw);
    auto smem_Bp_m128 = reinterpret_cast<unsigned char (*)[K_STEP_T64 / 2][N_TILE_LG + BP_PAD]>(smem_raw + 36864);
    auto smem_Bs_m128 = reinterpret_cast<unsigned char (*)[K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD]>(smem_raw + 46080);
    auto smem_B_fp8_m128 = reinterpret_cast<unsigned char (*)[K_STEP_T64 + 16]>(smem_raw + 47232);
    float* smem_LUT_m128 = reinterpret_cast<float*>(smem_raw + 57472);
    int* smem_tok_m128 = reinterpret_cast<int*>(smem_raw + 57536);

    if (threadIdx.x < 16) smem_LUT_m128[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE_M128) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_m128[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_m128[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast_m128 = K_STEP_T64 + PAD_T64;
    const unsigned int M_eff = (unsigned int)M_expert;

    // 256 threads × 4 rounds × 8-element cp.async = 8192 BF16 = M_TILE_M128 × K_STEP_T64.
    // Stride between rounds is 32 (vs 16 at M=64) since 256 threads have a_row_base 0..31.
    #define FGU64M128_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3;        /* 0..31 */ \
            unsigned int a_col = (threadIdx.x & 7) << 3;        /* 0,8,...,56 */ \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base;       /* 0..127 */ \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok_m128[row]; \
                moe_cp_async_pred_16(&smem_A_m128[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        /* B-load: 256 threads cover 32 K-pairs × 8 N chunks of 16-byte each = full K_STEP_T64/2 × N_TILE_LG. Single round. */ \
        { \
            unsigned int kp_cur = threadIdx.x >> 3;             /* 0..31 */ \
            unsigned int ns = (threadIdx.x & 7) << 4;           /* 0,16,...,112 */ \
            unsigned int gns = cta_n + ns; \
            unsigned int gke = (kb) + (kp_cur << 1); \
            moe_cp_async_pred_16(&smem_Bp_m128[(buf)][kp_cur][ns], \
                &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 < K) && (gns + 15 < N)); \
            if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                moe_cp_async_pred_16(&smem_Bs_m128[(buf)][kp_cur][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (sg < K / GROUP_SIZE) && (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant: 128 threads (my_n=0..127) drive the per-N column work; threads
    // 128..255 idle in dequant but participate in MMA. Same dequant wall-time
    // as M=64 case — dequant is not the bottleneck.
    #define FGU64M128_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        if (my_n < N_TILE_LG) { \
            __nv_fp8_e4m3 f0, f1, f2, f3; \
            *(unsigned char*)&f0 = smem_Bs_m128[(buf)][0][my_n]; \
            *(unsigned char*)&f1 = smem_Bs_m128[(buf)][1][my_n]; \
            *(unsigned char*)&f2 = smem_Bs_m128[(buf)][2][my_n]; \
            *(unsigned char*)&f3 = smem_Bs_m128[(buf)][3][my_n]; \
            float sv0 = (float)f0 * scale2; \
            float sv1 = (float)f1 * scale2; \
            float sv2 = (float)f2 * scale2; \
            float sv3 = (float)f3 * scale2; \
            _Pragma("unroll") \
            for (int kp = 0; kp < 8; kp++) { \
                unsigned char packed = smem_Bp_m128[(buf)][kp][my_n]; \
                float lo = smem_LUT_m128[packed & 0xF] * sv0; \
                float hi = smem_LUT_m128[packed >> 4] * sv0; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8_m128[my_n][kp * 2] = fp8_pair; \
            } \
            _Pragma("unroll") \
            for (int kp = 8; kp < 16; kp++) { \
                unsigned char packed = smem_Bp_m128[(buf)][kp][my_n]; \
                float lo = smem_LUT_m128[packed & 0xF] * sv1; \
                float hi = smem_LUT_m128[packed >> 4] * sv1; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8_m128[my_n][kp * 2] = fp8_pair; \
            } \
            _Pragma("unroll") \
            for (int kp = 16; kp < 24; kp++) { \
                unsigned char packed = smem_Bp_m128[(buf)][kp][my_n]; \
                float lo = smem_LUT_m128[packed & 0xF] * sv2; \
                float hi = smem_LUT_m128[packed >> 4] * sv2; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8_m128[my_n][kp * 2] = fp8_pair; \
            } \
            _Pragma("unroll") \
            for (int kp = 24; kp < 32; kp++) { \
                unsigned char packed = smem_Bp_m128[(buf)][kp][my_n]; \
                float lo = smem_LUT_m128[packed & 0xF] * sv3; \
                float hi = smem_LUT_m128[packed >> 4] * sv3; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8_m128[my_n][kp * 2] = fp8_pair; \
            } \
        } \
    } while(0)

    // MMA: 8 warps each cover 16 M rows, 16 N-strips per warp × m16n8k32 FP8 MMA.
    // Per K_STEP iteration: 2 inner K halves × 16 N-strips × 8 warps = 256 MMAs.
    #define FGU64M128_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_m128[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_m128 + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_m128 + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_m128 + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_m128 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_m128[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_m128[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
        unsigned int a4 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_m128 + 32 + tid * 4]); \
        unsigned int a5 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_m128 + 32 + tid * 4]); \
        unsigned int a6 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_m128 + 48 + tid * 4]); \
        unsigned int a7 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_m128 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_m128[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_m128[nc][48 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    FGU64M128_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    FGU64M128_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        FGU64M128_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        FGU64M128_COMPUTE_MMA(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        FGU64M128_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    FGU64M128_COMPUTE_MMA(cur);

    #undef FGU64M128_ISSUE_LOADS
    #undef FGU64M128_DEQUANT
    #undef FGU64M128_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}
