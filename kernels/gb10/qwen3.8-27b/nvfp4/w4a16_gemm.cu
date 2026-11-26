// SPDX-License-Identifier: AGPL-3.0-only

// Flux W4A16 GEMM — Qwen3.8-27B NVFP4 kernels.
//
// Optimizations:
// - w4a16_gemm_t: cp.async 2-stage double-buffered pipeline (overlaps next tile
//   loads with current tile compute), FP8 E4M3 dequant, BP_PAD bank conflict fix
// - Vectorized uint4 (128-bit) B_packed loads
// - Both-nibble extraction from packed bytes
// - N_TILE=128 for reduced A bandwidth

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef ATLAS_MMA_FP8_UNIT_M16N8K32
#if __CUDA_ARCH__ >= 1200
#define ATLAS_MMA_FP8_UNIT_M16N8K32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3) \
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{0,0},{%14},{0,0};" \
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3) \
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
         "f"(c0),"f"(c1),"f"(c2),"f"(c3),"r"(0x7fu))
#else
#define ATLAS_MMA_FP8_UNIT_M16N8K32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3) \
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
         "f"(c0),"f"(c1),"f"(c2),"f"(c3))
#endif
#endif

#define M_TILE 64
#define N_TILE_SM 64
#define N_TILE_LG 128
#define K_STEP 16
#define K_STEP_T 32
#define PAD 2
#define PAD_T 8        // cp.async needs 16-byte aligned rows: (32+8)*2=80, 80%16=0
#define BP_PAD 16      // smem_Bp row padding: stride 144 is 16-byte aligned, eliminates 4-way bank conflict
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Original layout w4a16_gemm: unchanged, N_TILE=64
extern "C" __global__ void w4a16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE_SM + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int ept = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[(unsigned long long)gr * K + gc] : __float2bfloat16(0.0f);
            }
        }
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE_SM;
                unsigned int n = idx % N_TILE_SM;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);
                    unsigned int sg = gk / GROUP_SIZE;
                    unsigned char sb = B_scale[(unsigned long long)gn * num_groups + sg];
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT[nibble] * (float)fp8 * scale2);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        unsigned int fc0 = tid * 2, fc1 = fc0 + 8;
        unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0];
        unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0];
        unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1];
        unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int nc = nt * 8 + group_id;
            unsigned int k0 = tid * 2, k1 = k0 + 8;
            unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[nt][3]);
    }
}
// cp.async helpers (SM80+)
__device__ __forceinline__ void cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// Windowed wait: block until at most N cp.async groups remain in flight.
// N is an immediate (PTX requires it), so we template on it.
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_pipe — cp.async 2-stage double-buffered EXACT shadow of
// w4a16_gemm (original-layout NVFP4 → BF16 GEMM).
//
// Preserves bit-identical arithmetic: same B_packed[N][K/2] / B_scale[N][K/16]
// gmem layout, same per-element dequant (E2M1_LUT[nibble] * (float)fp8 *
// scale2 → BF16, left-to-right), same smem_B BF16 layout (stride 66), same
// m16n8k16 f32.bf16.bf16.f32 MMA sequence and accumulator order. Only the
// LOAD path changes: A and B_packed move via cp.async into double buffers so
// stage N+1 loads overlap stage N compute; B_scale (2 bytes/column/stage) is
// staged through smem_Bs. K_STEP=32 with a kh∈{0,1} MMA loop preserves
// the exact per-accumulator K order of the K_STEP=16 parent while halving
// stage and barrier count (K_STEP=64 halves occupancy and regresses small-M
// prefill latency).
//
// Pipeline per stage: issue(N+1) || wait(cur) → dequant(cur) → MMA(cur).
// ═══════════════════════════════════════════════════════════════════

#define K_STEP_P 32          // pipeline stage width (2× parent K_STEP=16)
#define PAD_P 8              // (32+8)*2 = 80B rows → 16-byte aligned for cp.async
#define BP_P_STRIDE 16       // 16B per-column packed slots (32 K-rows) → 16-byte aligned
#define A_P_CHUNKS (M_TILE * K_STEP_P / 8 / 128)  // 16B chunks per thread (2)

// Byte-exact shadow of w4a16_gemm. See block comment above.
extern "C" __global__ void w4a16_gemm_pipe(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_P + PAD_P];
    __shared__ unsigned char smem_Bp[2][N_TILE_SM][BP_P_STRIDE];  // raw packed nibbles
    __shared__ unsigned char smem_Bs[2][N_TILE_SM][2];            // raw fp8 scale bytes (2 groups per 32-row stage)
    __shared__ __nv_bfloat16 smem_B[2][K_STEP_P][N_TILE_SM + PAD];// dequantized BF16
    __shared__ float smem_LUT[16];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP_P + PAD_P;  // 40
    const unsigned int b_stride = N_TILE_SM + PAD;   // 66

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    // Issue cp.async loads for stage at absolute K offset `kk` into buffer `buf`.
    // kk >= K issues 0-byte copies (no memory access) so the commit group still
    // fences the final real stage via wait_group<1>.
    #define ISSUE_P(buf, kk) do {                                                   \
        _Pragma("unroll")                                                           \
        for (int p = 0; p < A_P_CHUNKS; p++) {                                      \
            unsigned int c = threadIdx.x * A_P_CHUNKS + p;                          \
            unsigned int row = c / (K_STEP_P / 8);                                  \
            unsigned int col8 = (c % (K_STEP_P / 8)) << 3;                          \
            unsigned int gr = cta_m + row;                                          \
            cp_async_pred_16(&smem_A[(buf)][row][col8],                             \
                &A[(unsigned long long)gr * K + (kk) + col8], ((gr) < M) && ((kk) < K));                \
        }                                                                           \
        if (threadIdx.x < 64) {                                                     \
            unsigned int gn = cta_n + threadIdx.x;                                  \
            cp_async_pred_16(&smem_Bp[(buf)][threadIdx.x][0],                       \
                &B_packed[(unsigned long long)gn * half_K + ((kk) >> 1)],           \
                ((gn) < N) && ((kk) < K));                                          \
        }                                                                           \
        if (threadIdx.x < 64) {                                                     \
            unsigned int gn = cta_n + threadIdx.x;                                  \
            unsigned int sg = (kk) / GROUP_SIZE;                                    \
            _Pragma("unroll")                                                       \
            for (int g = 0; g < 2; g++) {                                           \
                unsigned char s = ((gn) < N && (kk) + g * GROUP_SIZE < K)           \
                    ? B_scale[(unsigned long long)gn * num_groups + sg + g] : 0;    \
                smem_Bs[(buf)][threadIdx.x][g] = s;                                 \
            }                                                                       \
        }                                                                           \
        cp_async_commit();                                                          \
    } while(0)

    // Dequant current stage into smem_B with the parent's exact arithmetic:
    // ((E2M1_LUT[nibble] * (float)fp8) * scale2) → BF16, left-to-right.
    #define DEQUANT_P(buf, kk) do {                                                 \
        _Pragma("unroll")                                                           \
        for (unsigned int i = 0; i < (K_STEP_P * N_TILE_SM / 128); i++) {           \
            unsigned int idx = threadIdx.x * (K_STEP_P * N_TILE_SM / 128) + i;      \
            unsigned int k = idx / N_TILE_SM;                                       \
            unsigned int n = idx % N_TILE_SM;                                       \
            unsigned char packed = smem_Bp[(buf)][n][k >> 1];                       \
            unsigned int nibble = ((kk) + k) & 1 ? (packed >> 4) : (packed & 0xF);  \
            unsigned char sb = smem_Bs[(buf)][n][k / GROUP_SIZE];                   \
            __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;                          \
            smem_B[(buf)][k][n] = __float2bfloat16(smem_LUT[nibble] * (float)fp8 * scale2); \
        }                                                                           \
    } while(0)

    ISSUE_P(0, 0);

    unsigned int cur = 0, nxt = 1;
    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_P) {
        ISSUE_P(nxt, k_base + K_STEP_P);
        cp_async_wait_group<1>();   // cur stage complete (next group may fly)
        __syncthreads();

        DEQUANT_P(cur, k_base);
        __syncthreads();

        // MMA on current stage — identical instruction sequence + accumulation
        // order to w4a16_gemm (kh loop preserves k_base+0..15,+16..31).
        const unsigned short* sA = (const unsigned short*)smem_A[cur];
        const unsigned short* sB = (const unsigned short*)smem_B[cur];
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            unsigned int fc0 = tid * 2 + kh * 16;
            unsigned int fc1 = fc0 + 8;
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0];
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0];
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1];
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned int nc = nt * 8 + group_id;
                unsigned int k0 = tid * 2 + kh * 16, k1 = k0 + 8;
                unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
                unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                     "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
            }
        }
        __syncthreads();
        unsigned int t = cur; cur = nxt; nxt = t;
    }

    // Epilogue — identical to w4a16_gemm.
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// Fused epilogue shadow for the dense Qwen3.8 prefill FFN up projection.
// The GEMM body is intentionally identical to w4a16_gemm_pipe. Its FP32
// accumulator is rounded to BF16 and converted back to FP32 before SiLU,
// preserving the separate up-GEMM -> BF16 buffer -> moe_silu_mul contract.
// `gate_in_out` is deliberately one in-place pointer rather than two
// `__restrict__` pointers: every element is read before its matching output is
// written, and no thread reads another thread's element.
extern "C" __global__ void w4a16_gemm_pipe_silu_mul(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ gate_in_out,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_P + PAD_P];
    __shared__ unsigned char smem_Bp[2][N_TILE_SM][BP_P_STRIDE];
    __shared__ unsigned char smem_Bs[2][N_TILE_SM][2];
    __shared__ __nv_bfloat16 smem_B[2][K_STEP_P][N_TILE_SM + PAD];
    __shared__ float smem_LUT[16];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP_P + PAD_P;
    const unsigned int b_stride = N_TILE_SM + PAD;

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    #define ISSUE_FUSED_EPILOGUE(buf, kk) do {                                     \
        _Pragma("unroll")                                                          \
        for (int p = 0; p < A_P_CHUNKS; p++) {                                    \
            unsigned int c = threadIdx.x * A_P_CHUNKS + p;                        \
            unsigned int row = c / (K_STEP_P / 8);                                \
            unsigned int col8 = (c % (K_STEP_P / 8)) << 3;                        \
            unsigned int gr = cta_m + row;                                        \
            cp_async_pred_16(&smem_A[(buf)][row][col8],                           \
                &A[(unsigned long long)gr * K + (kk) + col8], ((gr) < M) && ((kk) < K));              \
        }                                                                          \
        if (threadIdx.x < 64) {                                                    \
            unsigned int gn = cta_n + threadIdx.x;                                \
            cp_async_pred_16(&smem_Bp[(buf)][threadIdx.x][0],                     \
                &B_packed[(unsigned long long)gn * half_K + ((kk) >> 1)],         \
                ((gn) < N) && ((kk) < K));                                        \
            unsigned int sg = (kk) / GROUP_SIZE;                                  \
            _Pragma("unroll")                                                      \
            for (int g = 0; g < 2; g++) {                                         \
                unsigned char s = ((gn) < N && (kk) + g * GROUP_SIZE < K)         \
                    ? B_scale[(unsigned long long)gn * num_groups + sg + g] : 0;  \
                smem_Bs[(buf)][threadIdx.x][g] = s;                               \
            }                                                                      \
        }                                                                          \
        cp_async_commit();                                                         \
    } while(0)

    #define DEQUANT_FUSED_EPILOGUE(buf, kk) do {                                   \
        _Pragma("unroll")                                                          \
        for (unsigned int i = 0; i < (K_STEP_P * N_TILE_SM / 128); i++) {         \
            unsigned int idx = threadIdx.x * (K_STEP_P * N_TILE_SM / 128) + i;    \
            unsigned int k = idx / N_TILE_SM;                                     \
            unsigned int n = idx % N_TILE_SM;                                     \
            unsigned char packed = smem_Bp[(buf)][n][k >> 1];                     \
            unsigned int nibble = ((kk) + k) & 1 ? (packed >> 4) : (packed & 0xF);\
            unsigned char sb = smem_Bs[(buf)][n][k / GROUP_SIZE];                 \
            __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;                        \
            smem_B[(buf)][k][n] = __float2bfloat16(                              \
                smem_LUT[nibble] * (float)fp8 * scale2);                           \
        }                                                                          \
    } while(0)

    ISSUE_FUSED_EPILOGUE(0, 0);
    unsigned int cur = 0, nxt = 1;
    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_P) {
        ISSUE_FUSED_EPILOGUE(nxt, k_base + K_STEP_P);
        cp_async_wait_group<1>();
        __syncthreads();
        DEQUANT_FUSED_EPILOGUE(cur, k_base);
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A[cur];
        const unsigned short* sB = (const unsigned short*)smem_B[cur];
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            unsigned int fc0 = tid * 2 + kh * 16;
            unsigned int fc1 = fc0 + 8;
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0];
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0];
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1];
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned int nc = nt * 8 + group_id;
                unsigned int k0 = tid * 2 + kh * 16, k1 = k0 + 8;
                unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16)
                    | (unsigned int)sB[k0*b_stride+nc];
                unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16)
                    | (unsigned int)sB[k1*b_stride+nc];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                     "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
            }
        }
        __syncthreads();
        unsigned int t = cur; cur = nxt; nxt = t;
    }

    #define STORE_SILU_MUL(row, col, value) do {                                  \
        if ((row) < M && (col) < N) {                                             \
            unsigned long long out_idx = (unsigned long long)(row) * N + (col);   \
            __nv_bfloat16 up_bf16 = __float2bfloat16(value);                      \
            float g = __bfloat162float(gate_in_out[out_idx]);                      \
            float u = __bfloat162float(up_bf16);                                   \
            float sigmoid_g = 1.0f / (1.0f + __expf(-g));                         \
            gate_in_out[out_idx] = __float2bfloat16(g * sigmoid_g * u);            \
        }                                                                          \
    } while(0)

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        STORE_SILU_MUL(r0, c0, acc[nt][0]);
        STORE_SILU_MUL(r0, c1, acc[nt][1]);
        STORE_SILU_MUL(r1, c0, acc[nt][2]);
        STORE_SILU_MUL(r1, c1, acc[nt][3]);
    }

    #undef STORE_SILU_MUL
    #undef DEQUANT_FUSED_EPILOGUE
    #undef ISSUE_FUSED_EPILOGUE
}

// Exact dual large-M projection. Each side retains the w4a16_gemm_pipe
// dequantization and MMA order. `fuse_silu=0` writes two independent BF16
// outputs (attention K/V); `fuse_silu=1` rounds both through BF16 and writes
// only SiLU(first)*second to C0 (FFN gate/up). A and the LUT are loaded once.
extern "C" __global__ void w4a16_gemm_pipe_dual(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ gate_scale,
    const float gate_scale2,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float up_scale2,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* C1,
    unsigned int fuse_silu,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_P + PAD_P];
    __shared__ unsigned char smem_gate_p[2][N_TILE_SM][BP_P_STRIDE];
    __shared__ unsigned char smem_up_p[2][N_TILE_SM][BP_P_STRIDE];
    __shared__ unsigned char smem_gate_s[2][N_TILE_SM][2];
    __shared__ unsigned char smem_up_s[2][N_TILE_SM][2];
    __shared__ __nv_bfloat16 smem_B[2][K_STEP_P][N_TILE_SM + PAD];
    __shared__ float smem_LUT[16];

    float gate_acc[8][4];
    float up_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        gate_acc[i][0] = 0.0f; gate_acc[i][1] = 0.0f;
        gate_acc[i][2] = 0.0f; gate_acc[i][3] = 0.0f;
        up_acc[i][0] = 0.0f; up_acc[i][1] = 0.0f;
        up_acc[i][2] = 0.0f; up_acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP_P + PAD_P;
    const unsigned int b_stride = N_TILE_SM + PAD;
    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    #define ISSUE_GATEUP(buf, kk) do {                                            \
        _Pragma("unroll")                                                         \
        for (int p = 0; p < A_P_CHUNKS; p++) {                                   \
            unsigned int c = threadIdx.x * A_P_CHUNKS + p;                       \
            unsigned int row = c / (K_STEP_P / 8);                               \
            unsigned int col8 = (c % (K_STEP_P / 8)) << 3;                       \
            unsigned int gr = cta_m + row;                                       \
            cp_async_pred_16(&smem_A[(buf)][row][col8],                           \
                &A[(unsigned long long)gr * K + (kk) + col8], ((gr) < M) && ((kk) < K));             \
        }                                                                         \
        if (threadIdx.x < 64) {                                                   \
            unsigned int gn = cta_n + threadIdx.x;                               \
            cp_async_pred_16(&smem_gate_p[(buf)][threadIdx.x][0],                \
                &gate_packed[(unsigned long long)gn * half_K + ((kk) >> 1)],     \
                ((gn) < N) && ((kk) < K));                                       \
            cp_async_pred_16(&smem_up_p[(buf)][threadIdx.x][0],                  \
                &up_packed[(unsigned long long)gn * half_K + ((kk) >> 1)],       \
                ((gn) < N) && ((kk) < K));                                       \
            unsigned int sg = (kk) / GROUP_SIZE;                                 \
            _Pragma("unroll")                                                     \
            for (int g = 0; g < 2; g++) {                                        \
                bool valid = (gn) < N && (kk) + g * GROUP_SIZE < K;              \
                smem_gate_s[(buf)][threadIdx.x][g] = valid                        \
                    ? gate_scale[(unsigned long long)gn * num_groups + sg + g] : 0;\
                smem_up_s[(buf)][threadIdx.x][g] = valid                          \
                    ? up_scale[(unsigned long long)gn * num_groups + sg + g] : 0; \
            }                                                                     \
        }                                                                         \
        cp_async_commit();                                                        \
    } while(0)

    #define DEQUANT_GATEUP(buf, kk, packed_smem, scale_smem, scale_2) do {         \
        _Pragma("unroll")                                                         \
        for (unsigned int i = 0; i < (K_STEP_P * N_TILE_SM / 128); i++) {        \
            unsigned int idx = threadIdx.x * (K_STEP_P * N_TILE_SM / 128) + i;   \
            unsigned int k = idx / N_TILE_SM;                                    \
            unsigned int n = idx % N_TILE_SM;                                    \
            unsigned char packed = (packed_smem)[(buf)][n][k >> 1];              \
            unsigned int nibble = ((kk) + k) & 1 ? (packed >> 4) : (packed & 0xF);\
            unsigned char sb = (scale_smem)[(buf)][n][k / GROUP_SIZE];           \
            __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;                       \
            smem_B[(buf)][k][n] = __float2bfloat16(                              \
                smem_LUT[nibble] * (float)fp8 * (scale_2));                       \
        }                                                                         \
    } while(0)

    #define MMA_GATEUP(accum, buf) do {                                           \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)];          \
        const unsigned short* sB = (const unsigned short*)smem_B[(buf)];          \
        unsigned int fr0 = warp_m_offset + group_id;                              \
        unsigned int fr1 = fr0 + 8;                                               \
        _Pragma("unroll")                                                         \
        for (int kh = 0; kh < 2; kh++) {                                          \
            unsigned int fc0 = tid * 2 + kh * 16;                                \
            unsigned int fc1 = fc0 + 8;                                           \
            unsigned int a0 = *(const unsigned int*)&sA[fr0*a_stride+fc0];        \
            unsigned int a1 = *(const unsigned int*)&sA[fr1*a_stride+fc0];        \
            unsigned int a2 = *(const unsigned int*)&sA[fr0*a_stride+fc1];        \
            unsigned int a3 = *(const unsigned int*)&sA[fr1*a_stride+fc1];        \
            _Pragma("unroll")                                                     \
            for (int nt = 0; nt < 8; nt++) {                                     \
                unsigned int nc = nt * 8 + group_id;                             \
                unsigned int k0 = tid * 2 + kh * 16, k1 = k0 + 8;                \
                unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16)     \
                    | (unsigned int)sB[k0*b_stride+nc];                           \
                unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16)     \
                    | (unsigned int)sB[k1*b_stride+nc];                           \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"\
                    :"=f"((accum)[nt][0]),"=f"((accum)[nt][1]),                 \
                     "=f"((accum)[nt][2]),"=f"((accum)[nt][3])                  \
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),       \
                     "f"((accum)[nt][0]),"f"((accum)[nt][1]),                   \
                     "f"((accum)[nt][2]),"f"((accum)[nt][3]));                  \
            }                                                                     \
        }                                                                         \
    } while(0)

    ISSUE_GATEUP(0, 0);
    unsigned int cur = 0, nxt = 1;
    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_P) {
        ISSUE_GATEUP(nxt, k_base + K_STEP_P);
        cp_async_wait_group<1>();
        __syncthreads();
        DEQUANT_GATEUP(cur, k_base, smem_gate_p, smem_gate_s, gate_scale2);
        __syncthreads();
        MMA_GATEUP(gate_acc, cur);
        __syncthreads();
        DEQUANT_GATEUP(cur, k_base, smem_up_p, smem_up_s, up_scale2);
        __syncthreads();
        MMA_GATEUP(up_acc, cur);
        __syncthreads();
        unsigned int t = cur; cur = nxt; nxt = t;
    }

    #define STORE_GATEUP(row, col, gate_value, up_value) do {                     \
        if ((row) < M && (col) < N) {                                             \
            unsigned long long out_idx = (unsigned long long)(row) * N + (col);   \
            float g = __bfloat162float(__float2bfloat16(gate_value));             \
            float u = __bfloat162float(__float2bfloat16(up_value));               \
            if (fuse_silu != 0) {                                                 \
                float sigmoid_g = 1.0f / (1.0f + __expf(-g));                     \
                C0[out_idx] = __float2bfloat16(g * sigmoid_g * u);                \
            } else {                                                              \
                C0[out_idx] = __float2bfloat16(g);                                \
                C1[out_idx] = __float2bfloat16(u);                                \
            }                                                                     \
        }                                                                          \
    } while(0)
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        STORE_GATEUP(r0, c0, gate_acc[nt][0], up_acc[nt][0]);
        STORE_GATEUP(r0, c1, gate_acc[nt][1], up_acc[nt][1]);
        STORE_GATEUP(r1, c0, gate_acc[nt][2], up_acc[nt][2]);
        STORE_GATEUP(r1, c1, gate_acc[nt][3], up_acc[nt][3]);
    }

    #undef STORE_GATEUP
    #undef MMA_GATEUP
    #undef DEQUANT_GATEUP
    #undef ISSUE_GATEUP
}

// Eight-warp dual-projection shadow. Warps 0-3 own the first projection and
// warps 4-7 own the second, so each lane retains one 8x4 accumulator instead
// of both parent accumulator sets. A and the E2M1 LUT are still staged once;
// packed/scaled weights and dequantized BF16 B tiles remain independent.
//
// This symbol is intentionally not routed. Its launch contract is identical
// to w4a16_gemm_pipe_dual except that the block must contain 256 threads.
extern "C" __global__ __launch_bounds__(256, 1) void w4a16_gemm_pipe_dual_warp8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ gate_scale,
    const float gate_scale2,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float up_scale2,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* C1,
    unsigned int fuse_silu,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int side = warp_id >> 2;
    const unsigned int local_warp = warp_id & 3;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int warp_m_offset = local_warp * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A8[2][M_TILE][K_STEP_P + PAD_P];
    __shared__ unsigned char smem_p8[2][2][N_TILE_SM][BP_P_STRIDE];
    __shared__ unsigned char smem_s8[2][2][N_TILE_SM][2];
    __shared__ __nv_bfloat16 smem_B8[2][2][K_STEP_P][N_TILE_SM + PAD];
    // Fused-SiLU mode publishes only rounded gate values. Up remains in the
    // owning warp's registers, preserving both parent BF16 boundaries.
    __shared__ __nv_bfloat16 smem_gate_epilogue[M_TILE][N_TILE_SM];
    __shared__ float smem_LUT8[16];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP_P + PAD_P;
    const unsigned int b_stride = N_TILE_SM + PAD;
    if (threadIdx.x < 16) smem_LUT8[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    #define ISSUE_GATEUP8(buf, kk) do {                                             \
        unsigned int c = threadIdx.x;                                               \
        unsigned int row = c / (K_STEP_P / 8);                                     \
        unsigned int col8 = (c % (K_STEP_P / 8)) << 3;                             \
        unsigned int gr = cta_m + row;                                              \
        cp_async_pred_16(&smem_A8[(buf)][row][col8],                                \
            &A[(unsigned long long)gr * K + (kk) + col8], ((gr) < M) && ((kk) < K));                   \
        if (threadIdx.x < 128) {                                                    \
            unsigned int load_side = threadIdx.x >> 6;                             \
            unsigned int load_n = threadIdx.x & 63;                                \
            unsigned int gn = cta_n + load_n;                                      \
            const unsigned char* packed_base = load_side == 0                      \
                ? gate_packed : up_packed;                                          \
            const unsigned char* scale_base = load_side == 0                       \
                ? gate_scale : up_scale;                                            \
            cp_async_pred_16(&smem_p8[(buf)][load_side][load_n][0],                \
                &packed_base[(unsigned long long)gn * half_K + ((kk) >> 1)],        \
                ((gn) < N) && ((kk) < K));                                         \
            unsigned int sg = (kk) / GROUP_SIZE;                                   \
            _Pragma("unroll")                                                      \
            for (int g = 0; g < 2; g++) {                                          \
                bool valid = (gn) < N && (kk) + g * GROUP_SIZE < K;                \
                smem_s8[(buf)][load_side][load_n][g] = valid                       \
                    ? scale_base[(unsigned long long)gn * num_groups + sg + g] : 0;\
            }                                                                       \
        }                                                                           \
        cp_async_commit();                                                          \
    } while(0)

    #define DEQUANT_GATEUP8(buf, kk) do {                                           \
        _Pragma("unroll")                                                          \
        for (unsigned int i = 0; i <                                               \
                (2 * K_STEP_P * N_TILE_SM / 256); i++) {                           \
            unsigned int idx = threadIdx.x *                                       \
                (2 * K_STEP_P * N_TILE_SM / 256) + i;                              \
            unsigned int dequant_side = idx / (K_STEP_P * N_TILE_SM);              \
            unsigned int side_idx = idx % (K_STEP_P * N_TILE_SM);                  \
            unsigned int k = side_idx / N_TILE_SM;                                 \
            unsigned int n = side_idx % N_TILE_SM;                                 \
            unsigned char packed = smem_p8[(buf)][dequant_side][n][k >> 1];        \
            unsigned int nibble = ((kk) + k) & 1                                  \
                ? (packed >> 4) : (packed & 0xF);                                  \
            unsigned char sb = smem_s8[(buf)][dequant_side][n]                    \
                [k / GROUP_SIZE];                                                   \
            __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;                         \
            float scale_2 = dequant_side == 0 ? gate_scale2 : up_scale2;           \
            smem_B8[(buf)][dequant_side][k][n] = __float2bfloat16(                 \
                smem_LUT8[nibble] * (float)fp8 * scale_2);                         \
        }                                                                           \
    } while(0)

    #define MMA_GATEUP8(buf) do {                                                   \
        const unsigned short* sA = (const unsigned short*)smem_A8[(buf)];          \
        const unsigned short* sB = (const unsigned short*)smem_B8[(buf)][side];    \
        unsigned int fr0 = warp_m_offset + group_id;                               \
        unsigned int fr1 = fr0 + 8;                                                 \
        _Pragma("unroll")                                                          \
        for (int kh = 0; kh < 2; kh++) {                                           \
            unsigned int fc0 = tid * 2 + kh * 16;                                  \
            unsigned int fc1 = fc0 + 8;                                             \
            unsigned int a0 = *(const unsigned int*)&sA[fr0*a_stride+fc0];         \
            unsigned int a1 = *(const unsigned int*)&sA[fr1*a_stride+fc0];         \
            unsigned int a2 = *(const unsigned int*)&sA[fr0*a_stride+fc1];         \
            unsigned int a3 = *(const unsigned int*)&sA[fr1*a_stride+fc1];         \
            _Pragma("unroll")                                                      \
            for (int nt = 0; nt < 8; nt++) {                                       \
                unsigned int nc = nt * 8 + group_id;                               \
                unsigned int k0 = tid * 2 + kh * 16, k1 = k0 + 8;                 \
                unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16)      \
                    | (unsigned int)sB[k0*b_stride+nc];                             \
                unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16)      \
                    | (unsigned int)sB[k1*b_stride+nc];                             \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"\
                    :"=f"(acc[nt][0]),"=f"(acc[nt][1]),                         \
                     "=f"(acc[nt][2]),"=f"(acc[nt][3])                          \
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),   \
                     "f"(acc[nt][0]),"f"(acc[nt][1]),                           \
                     "f"(acc[nt][2]),"f"(acc[nt][3]));                          \
            }                                                                       \
        }                                                                           \
    } while(0)

    ISSUE_GATEUP8(0, 0);
    unsigned int cur = 0, nxt = 1;
    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_P) {
        ISSUE_GATEUP8(nxt, k_base + K_STEP_P);
        cp_async_wait_group<1>();
        __syncthreads();
        DEQUANT_GATEUP8(cur, k_base);
        __syncthreads();
        MMA_GATEUP8(cur);
        __syncthreads();
        unsigned int t = cur; cur = nxt; nxt = t;
    }

    if (fuse_silu != 0) {
        if (side == 0) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned int c0 = nt*8 + tid*2;
                unsigned int c1 = c0 + 1;
                unsigned int r0 = warp_m_offset + group_id;
                unsigned int r1 = r0 + 8;
                smem_gate_epilogue[r0][c0] = __float2bfloat16(acc[nt][0]);
                smem_gate_epilogue[r0][c1] = __float2bfloat16(acc[nt][1]);
                smem_gate_epilogue[r1][c0] = __float2bfloat16(acc[nt][2]);
                smem_gate_epilogue[r1][c1] = __float2bfloat16(acc[nt][3]);
            }
        }
        __syncthreads();
        if (side == 1) {
            #define STORE_SILU8(row, col, value) do {                               \
                unsigned int gr = cta_m + (row);                                   \
                unsigned int gc = cta_n + (col);                                   \
                if (gr < M && gc < N) {                                            \
                    unsigned long long out_idx = (unsigned long long)gr * N + gc;  \
                    float g = __bfloat162float(smem_gate_epilogue[(row)][(col)]);   \
                    float u = __bfloat162float(__float2bfloat16(value));            \
                    float sigmoid_g = 1.0f / (1.0f + __expf(-g));                  \
                    C0[out_idx] = __float2bfloat16(g * sigmoid_g * u);              \
                }                                                                   \
            } while(0)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned int c0 = nt*8 + tid*2;
                unsigned int c1 = c0 + 1;
                unsigned int r0 = warp_m_offset + group_id;
                unsigned int r1 = r0 + 8;
                STORE_SILU8(r0, c0, acc[nt][0]);
                STORE_SILU8(r0, c1, acc[nt][1]);
                STORE_SILU8(r1, c0, acc[nt][2]);
                STORE_SILU8(r1, c1, acc[nt][3]);
            }
            #undef STORE_SILU8
        }
    } else {
        __nv_bfloat16* output = side == 0 ? C0 : C1;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int c0 = cta_n + nt*8 + tid*2;
            unsigned int c1 = c0 + 1;
            unsigned int r0 = cta_m + warp_m_offset + group_id;
            unsigned int r1 = r0 + 8;
            if (r0 < M && c0 < N) output[(unsigned long long)r0*N+c0] = __float2bfloat16(acc[nt][0]);
            if (r0 < M && c1 < N) output[(unsigned long long)r0*N+c1] = __float2bfloat16(acc[nt][1]);
            if (r1 < M && c0 < N) output[(unsigned long long)r1*N+c0] = __float2bfloat16(acc[nt][2]);
            if (r1 < M && c1 < N) output[(unsigned long long)r1*N+c1] = __float2bfloat16(acc[nt][3]);
        }
    }

    #undef MMA_GATEUP8
    #undef DEQUANT_GATEUP8
    #undef ISSUE_GATEUP8
}

// Four-warp N32 dual-projection shadow. Halving the N tile halves each
// projection's live accumulator footprint while retaining one shared A stage
// for gate and up. The symbol is intentionally unrouted and requires a
// 128-thread block; grid.x must be ceil(N/32).
#define N_TILE_DUAL32 32
extern "C" __global__ __launch_bounds__(128, 4) void w4a16_gemm_pipe_dual_n32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ gate_scale,
    const float gate_scale2,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float up_scale2,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* C1,
    unsigned int fuse_silu,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_DUAL32;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A32[2][M_TILE][K_STEP_P + PAD_P];
    __shared__ unsigned char smem_p32[2][2][N_TILE_DUAL32][BP_P_STRIDE];
    __shared__ unsigned char smem_s32[2][2][N_TILE_DUAL32][2];
    __shared__ __nv_bfloat16 smem_B32[2][2][K_STEP_P][N_TILE_DUAL32 + PAD];
    __shared__ float smem_LUT32[16];

    float gate_acc[4][4];
    float up_acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        gate_acc[i][0] = 0.0f; gate_acc[i][1] = 0.0f;
        gate_acc[i][2] = 0.0f; gate_acc[i][3] = 0.0f;
        up_acc[i][0] = 0.0f; up_acc[i][1] = 0.0f;
        up_acc[i][2] = 0.0f; up_acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP_P + PAD_P;
    const unsigned int b_stride = N_TILE_DUAL32 + PAD;
    if (threadIdx.x < 16) smem_LUT32[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    #define ISSUE_GATEUP32(buf, kk) do {                                            \
        _Pragma("unroll")                                                          \
        for (int p = 0; p < A_P_CHUNKS; p++) {                                    \
            unsigned int c = threadIdx.x * A_P_CHUNKS + p;                        \
            unsigned int row = c / (K_STEP_P / 8);                                \
            unsigned int col8 = (c % (K_STEP_P / 8)) << 3;                        \
            unsigned int gr = cta_m + row;                                         \
            cp_async_pred_16(&smem_A32[(buf)][row][col8],                          \
                &A[(unsigned long long)gr * K + (kk) + col8], ((gr) < M) && ((kk) < K));              \
        }                                                                           \
        if (threadIdx.x < 64) {                                                     \
            unsigned int load_side = threadIdx.x >> 5;                            \
            unsigned int load_n = threadIdx.x & 31;                               \
            unsigned int gn = cta_n + load_n;                                      \
            const unsigned char* packed_base = load_side == 0                     \
                ? gate_packed : up_packed;                                          \
            const unsigned char* scale_base = load_side == 0                      \
                ? gate_scale : up_scale;                                            \
            cp_async_pred_16(&smem_p32[(buf)][load_side][load_n][0],              \
                &packed_base[(unsigned long long)gn * half_K + ((kk) >> 1)],        \
                ((gn) < N) && ((kk) < K));                                         \
            unsigned int sg = (kk) / GROUP_SIZE;                                   \
            _Pragma("unroll")                                                      \
            for (int g = 0; g < 2; g++) {                                          \
                bool valid = (gn) < N && (kk) + g * GROUP_SIZE < K;                \
                smem_s32[(buf)][load_side][load_n][g] = valid                     \
                    ? scale_base[(unsigned long long)gn * num_groups + sg + g] : 0;\
            }                                                                       \
        }                                                                           \
        cp_async_commit();                                                          \
    } while(0)

    #define DEQUANT_GATEUP32(buf, kk) do {                                          \
        _Pragma("unroll")                                                          \
        for (unsigned int i = 0; i <                                               \
                (2 * K_STEP_P * N_TILE_DUAL32 / 128); i++) {                       \
            unsigned int idx = threadIdx.x *                                       \
                (2 * K_STEP_P * N_TILE_DUAL32 / 128) + i;                          \
            unsigned int dequant_side = idx / (K_STEP_P * N_TILE_DUAL32);          \
            unsigned int side_idx = idx % (K_STEP_P * N_TILE_DUAL32);              \
            unsigned int k = side_idx / N_TILE_DUAL32;                             \
            unsigned int n = side_idx % N_TILE_DUAL32;                             \
            unsigned char packed = smem_p32[(buf)][dequant_side][n][k >> 1];       \
            unsigned int nibble = ((kk) + k) & 1                                  \
                ? (packed >> 4) : (packed & 0xF);                                  \
            unsigned char sb = smem_s32[(buf)][dequant_side][n]                   \
                [k / GROUP_SIZE];                                                   \
            __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;                         \
            float scale_2 = dequant_side == 0 ? gate_scale2 : up_scale2;           \
            smem_B32[(buf)][dequant_side][k][n] = __float2bfloat16(                \
                smem_LUT32[nibble] * (float)fp8 * scale_2);                        \
        }                                                                           \
    } while(0)

    #define MMA_GATEUP32(accum, buf, mma_side) do {                                 \
        const unsigned short* sA = (const unsigned short*)smem_A32[(buf)];         \
        const unsigned short* sB =                                                  \
            (const unsigned short*)smem_B32[(buf)][(mma_side)];                    \
        unsigned int fr0 = warp_m_offset + group_id;                               \
        unsigned int fr1 = fr0 + 8;                                                 \
        _Pragma("unroll")                                                          \
        for (int kh = 0; kh < 2; kh++) {                                           \
            unsigned int fc0 = tid * 2 + kh * 16;                                  \
            unsigned int fc1 = fc0 + 8;                                             \
            unsigned int a0 = *(const unsigned int*)&sA[fr0*a_stride+fc0];         \
            unsigned int a1 = *(const unsigned int*)&sA[fr1*a_stride+fc0];         \
            unsigned int a2 = *(const unsigned int*)&sA[fr0*a_stride+fc1];         \
            unsigned int a3 = *(const unsigned int*)&sA[fr1*a_stride+fc1];         \
            _Pragma("unroll")                                                      \
            for (int nt = 0; nt < 4; nt++) {                                       \
                unsigned int nc = nt * 8 + group_id;                               \
                unsigned int k0 = tid * 2 + kh * 16, k1 = k0 + 8;                 \
                unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16)      \
                    | (unsigned int)sB[k0*b_stride+nc];                             \
                unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16)      \
                    | (unsigned int)sB[k1*b_stride+nc];                             \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"\
                    :"=f"((accum)[nt][0]),"=f"((accum)[nt][1]),                 \
                     "=f"((accum)[nt][2]),"=f"((accum)[nt][3])                  \
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),   \
                     "f"((accum)[nt][0]),"f"((accum)[nt][1]),                   \
                     "f"((accum)[nt][2]),"f"((accum)[nt][3]));                  \
            }                                                                       \
        }                                                                           \
    } while(0)

    ISSUE_GATEUP32(0, 0);
    unsigned int cur = 0, nxt = 1;
    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_P) {
        ISSUE_GATEUP32(nxt, k_base + K_STEP_P);
        cp_async_wait_group<1>();
        __syncthreads();
        DEQUANT_GATEUP32(cur, k_base);
        __syncthreads();
        MMA_GATEUP32(gate_acc, cur, 0);
        MMA_GATEUP32(up_acc, cur, 1);
        __syncthreads();
        unsigned int t = cur; cur = nxt; nxt = t;
    }

    #define STORE_DUAL32(row, col, gate_value, up_value) do {                       \
        unsigned int gr = cta_m + (row);                                           \
        unsigned int gc = cta_n + (col);                                           \
        if (gr < M && gc < N) {                                                    \
            unsigned long long out_idx = (unsigned long long)gr * N + gc;          \
            float g = __bfloat162float(__float2bfloat16(gate_value));              \
            float u = __bfloat162float(__float2bfloat16(up_value));                \
            if (fuse_silu != 0) {                                                  \
                float sigmoid_g = 1.0f / (1.0f + __expf(-g));                     \
                C0[out_idx] = __float2bfloat16(g * sigmoid_g * u);                 \
            } else {                                                               \
                C0[out_idx] = __float2bfloat16(g);                                 \
                C1[out_idx] = __float2bfloat16(u);                                 \
            }                                                                      \
        }                                                                          \
    } while(0)
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        unsigned int c0 = nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        STORE_DUAL32(r0, c0, gate_acc[nt][0], up_acc[nt][0]);
        STORE_DUAL32(r0, c1, gate_acc[nt][1], up_acc[nt][1]);
        STORE_DUAL32(r1, c0, gate_acc[nt][2], up_acc[nt][2]);
        STORE_DUAL32(r1, c1, gate_acc[nt][3], up_acc[nt][3]);
    }

    #undef STORE_DUAL32
    #undef MMA_GATEUP32
    #undef DEQUANT_GATEUP32
    #undef ISSUE_GATEUP32
}
#undef N_TILE_DUAL32

// ═══════════════════════════════════════════════════════════════════
// FP8-MMA transposed dense GEMM.
//
// Dequant B to FP8 E4M3 (not BF16). Convert A from BF16→FP8 in
// registers. Use mma.sync.m16n8k32.e4m3.e4m3 — processes full K=32
// per instruction (2x fewer MMA instructions vs BF16 m16n8k16).
//
// Pipeline: load[nxt] || MMA[cur] → wait → dequant[nxt] → sync
//
// smem: A 2×64×40×2=10240B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//       B_fp8 128×32=4096B, LUT 64B = ~19.6KB
// ═══════════════════════════════════════════════════════════════════

// Convert 4 BF16 values from smem to packed uint32 of 4 E4M3 values
__device__ __forceinline__ unsigned int bf16x4_to_e4m3x4(const unsigned short* src) {
    unsigned int p0 = *(const unsigned int*)src;
    unsigned int p1 = *(const unsigned int*)(src + 2);
    unsigned short bf0 = (unsigned short)(p0 & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p0 >> 16);
    unsigned short bf2 = (unsigned short)(p1 & 0xFFFFu);
    unsigned short bf3 = (unsigned short)(p1 >> 16);
    float f0, f1, f2, f3;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f2) : "h"(bf2));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f3) : "h"(bf3));
    unsigned short h0, h1;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h0) : "f"(f1), "f"(f0));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h1) : "f"(f3), "f"(f2));
    return ((unsigned int)h1 << 16) | (unsigned int)h0;
}

extern "C" __global__
__launch_bounds__(128, 4)
void w4a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant B: FP4 → FP8 E4M3 (cvt.rn.satfinite.e4m3x2.f32)
    #define DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // FP8 MMA: convert A BF16→E4M3 in registers, single m16n8k32 per N-tile
    #define COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]); \
        } \
    } while(0)

    ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_T(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_T(nxt);
        __syncthreads();
        cur = nxt;
    }

    COMPUTE_MMA(cur);

    #undef ISSUE_LOADS
    #undef DEQUANT_T
    #undef COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Small-M FP8-MMA transposed GEMM.  M_TILE=16.
//
// Specialization of w4a16_gemm_t for K=γ verify (MTP K=3 → M=3,
// DFlash γ=16 → M=16-17). With M_TILE=64 the parent kernel discards
// 75-95% of MMA accumulator writes via `if (r < M)` guards AND every
// warp redundantly streams the same 128-N B tile to compute rows that
// don't exist.
//
// Redesign:
//   - 1 CTA covers 16 rows × 128 N cols (single CTA row when M ≤ 16).
//   - All 4 warps process the SAME 16 rows (warp_m_offset = 0).
//   - Warp w handles N sub-tiles [w*4 .. w*4+4) → 32 N columns per warp.
//   - 4 m16n8k32 MMAs per warp = 16 MMAs total (same as parent) but
//     parallelized 4-way instead of 4 warps × 16 MMA serialized.
//   - B/dequant tile shared across all warps via smem (same as parent).
//
// Grid: (ceil(N/128), ceil(M/16), 1)  Block: (128, 1, 1)
// SMEM: A 2×16×40×2=2560B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//       B_fp8 128×32=4096B, LUT 64B ≈ 11.9 KB → register-limited
//       at ≥6 CTAs/SM (same as parent — register count unchanged).
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void w4a16_gemm_t_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    constexpr unsigned int M_TILE_S = 16;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE_S;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    // All warps cover the same 16 M rows — no warp_m_offset (was warp_id*16).
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_S][K_STEP_T + PAD_T];        // 2×16×40×2 = 2560B
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD]; // 4608B
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD]; // 576B
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];               // 4096B
    __shared__ float smem_LUT[16];                                          //   64B

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Per-warp accumulator: 4 N sub-tiles × 4 fp32 = 16 fp32 (was 16×4=64 in parent).
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // A load: with 128 threads and M=16 rows of width 32 BF16 (16B per thread × 4 cols),
    // we only need (16 rows × 4 cols/row) = 64 threads. Threads 0-63 load; 64-127 idle for A.
    // B/Bs loads identical to parent — all 128 threads participate.
    #define ISSUE_LOADS_M16(buf, kb) do { \
        if (threadIdx.x < 64) { \
            unsigned int a_row = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            unsigned int gr = cta_m + a_row; \
            cp_async_pred_16(&smem_A[(buf)][a_row][a_col], \
                &A[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 7 < K)); \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant: identical to parent (FP4 → FP8 E4M3, all 128 threads, 1 N col each).
    #define DEQUANT_T_M16(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // FP8 MMA: each warp handles 4 N sub-tiles (32 N cols), rows 0..15 shared.
    // Per-warp N range: warp_id * 32 .. warp_id*32 + 32  (warp_id in 0..3).
    // Per-warp nt range: warp_id*4 .. warp_id*4 + 4.
    #define COMPUTE_MMA_M16(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id;          /* rows 0..7  */ \
        unsigned int fr1 = fr0 + 8;           /* rows 8..15 */ \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 4; sub++) { \
            unsigned int nt = warp_id * 4 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[sub][0], acc[sub][1], acc[sub][2], acc[sub][3], \
                a0, a1, a2, a3, b0, b1, \
                acc[sub][0], acc[sub][1], acc[sub][2], acc[sub][3]); \
        } \
    } while(0)

    ISSUE_LOADS_M16(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_T_M16(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS_M16(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA_M16(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_T_M16(nxt);
        __syncthreads();
        cur = nxt;
    }
    COMPUTE_MMA_M16(cur);

    #undef ISSUE_LOADS_M16
    #undef DEQUANT_T_M16
    #undef COMPUTE_MMA_M16

    // Output write: each warp writes its 4 N sub-tiles for rows 0..15.
    #pragma unroll
    for (int sub = 0; sub < 4; sub++) {
        unsigned int nt = warp_id * 4 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[sub][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[sub][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[sub][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[sub][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m16_n64 — small-M (M_TILE=16), small-N (N_TILE=64)
//
// Drop-in replacement for `w4a16_gemm_t_m16` tuned for the K=3 MTP
// verify path on dense Qwen3.8-27B (M=3, padded internally to MMA-16).
//
// The parent `w4a16_gemm_t_m16` uses N_TILE_LG=128 → at intermediate=17408
// the grid is only 136 CTAs/projection (~2.8 CTAs/SM on GB10's 48 SMs),
// leaving limited occupancy to hide BF16→FP8 dequant + MMA latency, so the
// path runs slower than the M=3 `w4a16_gemv_dual_batch3` GEMV that
// dispatches 8704 CTAs (measured: -23% mean tok/s when forward_k3
// was routed through this kernel on the 27B 5-prompt benchmark suite).
//
// This variant halves N_TILE to 64 — 272 CTAs/projection (~5.7 CTAs/SM),
// 2× more parallel CTAs at half the per-CTA work. Each warp still owns
// 4 N sub-tiles of 8 cols (32 cols/warp × 2 warps = 64 cols/CTA),
// reducing thread count from 128 → 64. The shared dequant pipeline,
// FP8 MMA, and cp.async double-buffer are unchanged.
//
// Geometry:
//   - Grid (ceil(N/64), ceil(M/16), 1)  Block (64, 1, 1) = 2 warps
//   - M_TILE_S = 16 (kernel-internal pad; output bounds-check writes 0..M-1)
//   - N_TILE_S2 = 64, K_STEP_T = 32 (unchanged)
//
// SMEM:
//   - A:        2 × 16 × 40 × 2  = 2560B
//   - Bp:       2 × 16 × 80      = 2560B  (halved from 4608B)
//   - Bs:       2 × 2  × 80      =  320B  (halved from 576B)
//   - B_fp8:    64 × 32          = 2048B  (halved from 4096B)
//   - LUT:                        =   64B
//                              ≈ 7.6 KB → ~8 CTAs/SM register budget
//
// Caller contract: identical to `w4a16_gemm_t_m16`. Accepts any M ≤ 16;
// the kernel pads with zeros in smem and the output bounds check
// discards padding rows.
//
// Measured outcome (Qwen3.8-27B long-output benchmark, 1024-token max):
// 18.91 mean tok/s vs 26.83 mean for the production GEMV
// path (`w4a16_gemv_dual_batch3`). The N=128 sibling measured 20.56 in
// the same session. Both TC variants lose because the K=3 FFN on this
// model is HBM-bandwidth-bound — 132 MB of NVFP4 weight reads/layer at
// LPDDR5X 273 GB/s = ~31 ms minimum FFN time per verify step regardless
// of kernel geometry. Tensor-core compute density (~16× CUDA-core FMA)
// cannot overcome the bandwidth floor when M=3 already amortizes
// inadequately (81% MMA-tile waste). Kept in tree as an opt-in artifact
// (`ATLAS_TC_NVFP4_K3=1`) for future hardware (HBM3e ≥ 3.35 TB/s would
// flip the ranking) or fused multi-layer experiments that preserve
// weights in L2 across layer-call boundaries.
// ═══════════════════════════════════════════════════════════════════

// Local constants — naming chosen to avoid colliding with the
// module-level N_TILE_LG (=128) used by the parent and by other kernels
// in this file.
#define N_TILE_S2 64

extern "C" __global__ void w4a16_gemm_t_m16_n64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    constexpr unsigned int M_TILE_S = 16;
    const unsigned int cta_n = blockIdx.x * N_TILE_S2;
    const unsigned int cta_m = blockIdx.y * M_TILE_S;
    const unsigned int warp_id = threadIdx.x / 32;  // 0..1
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;     // 0..7 (M row pair)
    const unsigned int tid = lane_id & 3;           // 0..3 (K stride)

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_S][K_STEP_T + PAD_T];        // 2560B
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_S2 + BP_PAD]; // 2560B
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_S2 + BP_PAD]; // 320B
    __shared__ unsigned char smem_B_fp8[N_TILE_S2][K_STEP_T];               // 2048B
    __shared__ float smem_LUT[16];                                          //   64B

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Per-warp accumulator: 4 N sub-tiles × 4 fp32 = 16 fp32 per warp.
    // 2 warps × 4 sub-tiles × 8 cols = 64 cols total.
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // A load: 16 rows × 4 cols × 8 BF16/col = 64 thread-loads needed.
    // 64 threads (full block) → exactly one round.
    // B load: 64 cols × K_STEP_T/2 (=16) packed bytes via 16-byte cp.async.
    //   16 K-pair rows × 4 col-stripes of 16 bytes each = 64 thread-loads.
    //   threadIdx.x maps to (kp = idx>>2, ns_stripe = idx&3 × 16).
    // Bs load: 2 scale rows × 64 cols / 16 bytes/load = 8 thread-loads.
    #define ISSUE_LOADS_M16_N64(buf, kb) do { \
        { \
            unsigned int a_row = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            unsigned int gr = cta_m + a_row; \
            cp_async_pred_16(&smem_A[(buf)][a_row & (M_TILE_S - 1)][a_col], \
                &A[(unsigned long long)gr * K + gc], \
                (a_row < M_TILE_S) && (gr < M) && (gc + 7 < K)); \
        } \
        { \
            unsigned int kp = threadIdx.x >> 2;          /* 0..15 */ \
            unsigned int ns = (threadIdx.x & 3) << 4;    /* 0/16/32/48 */ \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (threadIdx.x < (K_STEP_T / GROUP_SIZE) * (N_TILE_S2 / 16)) { \
                /* 2 scale rows × 4 16-byte stripes = 8 cp.async issues, */ \
                /* mapped onto threads 0..7 of warp 0. */ \
                unsigned int sg_row = threadIdx.x >> 2;           /* 0 or 1 */ \
                unsigned int sg_ns  = (threadIdx.x & 3) << 4;     /* 0/16/32/48 */ \
                unsigned int sg = (kb) / GROUP_SIZE + sg_row; \
                cp_async_pred_16(&smem_Bs[(buf)][sg_row][sg_ns], \
                    &B_scale[(unsigned long long)sg * N + cta_n + sg_ns], \
                    (cta_n + sg_ns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant: 64 threads, one col each, full K_STEP_T (32 K-values).
    // Identical math to parent (FP4 → FP8 E4M3) but operates on the 64-col tile.
    #define DEQUANT_T_M16_N64(buf) do { \
        unsigned int my_n = threadIdx.x; \
        if (my_n < N_TILE_S2) { \
            unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
            unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
            __nv_fp8_e4m3 f0, f1; \
            *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
            float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
            _Pragma("unroll") \
            for (int kp = 0; kp < 8; kp++) { \
                unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
                float lo = smem_LUT[packed & 0xF] * sv0; \
                float hi = smem_LUT[packed >> 4] * sv0; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
            } \
            _Pragma("unroll") \
            for (int kp = 8; kp < 16; kp++) { \
                unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
                float lo = smem_LUT[packed & 0xF] * sv1; \
                float hi = smem_LUT[packed >> 4] * sv1; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
            } \
        } \
    } while(0)

    // FP8 MMA: 2 warps × 4 N sub-tiles each (= 64 cols / CTA).
    // warp_id maps to N halves: warp 0 → sub-tiles 0..3 (cols 0..31),
    //                            warp 1 → sub-tiles 0..3 (cols 32..63).
    // Same in-tile math as parent (m16n8k32 e4m3 MMA), 16 rows shared.
    #define COMPUTE_MMA_M16_N64(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 4; sub++) { \
            unsigned int nt = warp_id * 4 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[sub][0], acc[sub][1], acc[sub][2], acc[sub][3], \
                a0, a1, a2, a3, b0, b1, \
                acc[sub][0], acc[sub][1], acc[sub][2], acc[sub][3]); \
        } \
    } while(0)

    ISSUE_LOADS_M16_N64(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_T_M16_N64(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS_M16_N64(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA_M16_N64(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_T_M16_N64(nxt);
        __syncthreads();
        cur = nxt;
    }
    COMPUTE_MMA_M16_N64(cur);

    #undef ISSUE_LOADS_M16_N64
    #undef DEQUANT_T_M16_N64
    #undef COMPUTE_MMA_M16_N64

    // Output write: each warp writes its 4 N sub-tiles for rows 0..15.
    // Identical to parent layout; only the cta_n stride changes.
    #pragma unroll
    for (int sub = 0; sub < 4; sub++) {
        unsigned int nt = warp_id * 4 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[sub][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[sub][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[sub][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[sub][3]);
    }
}

#undef N_TILE_S2

// ═══════════════════════════════════════════════════════════════════
// Pre-dequanted FP8 GEMM (prefill).
//
// B_fp8 is pre-dequanted at load time: NVFP4 → FP8 E4M3 once.
// Eliminates the per-inference DEQUANT phase entirely.
// B_fp8[N, K] layout — each row is one output neuron, K consecutive.
//
// Pipeline: LOAD(A+B_fp8) || COMPUTE_MMA — only 1 sync per K step.
//
// smem: A 2×64×40×2=10240B, B_fp8 2×128×32=8192B = ~18.4KB
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void fp8_gemm_t(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (BF16) + B (FP8, pre-dequanted) via cp.async
    #define FP8_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8 MMA — identical to w4a16_gemm_t COMPUTE_MMA
    #define FP8_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]); \
        } \
    } while(0)

    // Prolog: load first tile, wait, no dequant needed
    FP8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Main loop: LOAD(nxt) || COMPUTE(cur) → wait → sync
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FP8_LOADS(nxt, k_base);
        cp_async_commit();
        FP8_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FP8_COMPUTE(cur, cur);

    #undef FP8_LOADS
    #undef FP8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pre-dequant: NVFP4 [N, K/2] + scales [N, K/GROUP_SIZE] → FP8 [N, K]
//
// One-time conversion at model load. Each thread processes 1 packed
// byte (2 FP4 values) → 2 FP8 E4M3 values.
// Grid: (ceil(N * K/2 / 256), 1, 1)  Block: (256, 1, 1)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void predequant_nvfp4_to_fp8(
    const unsigned char* __restrict__ B_packed,  // [N, K/2]
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE]
    float scale2,
    unsigned char* __restrict__ B_fp8,           // [N, K]
    unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_K = K / 2;
    unsigned int total = N * half_K;
    if (idx >= total) return;

    unsigned int n = idx / half_K;
    unsigned int k_pair = idx % half_K;
    unsigned int k_even = k_pair * 2;

    unsigned char packed = B_packed[(unsigned long long)n * half_K + k_pair];
    unsigned int group = k_even / GROUP_SIZE;
    unsigned char sb = B_scale[(unsigned long long)n * (K / GROUP_SIZE) + group];
    __nv_fp8_e4m3 fp8_scale;
    *(unsigned char*)&fp8_scale = sb;
    float sv = (float)fp8_scale * scale2;

    float val_lo = E2M1_LUT[packed & 0xF] * sv;
    float val_hi = E2M1_LUT[packed >> 4] * sv;

    unsigned short fp8_pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(fp8_pair) : "f"(val_hi), "f"(val_lo));

    *(unsigned short*)&B_fp8[(unsigned long long)n * K + k_even] = fp8_pair;
}

// ═══════════════════════════════════════════════════════════════════
// BF16 → FP8 E4M3 activation conversion.
// Converts [M, K] BF16 activations to [M, K] FP8 E4M3 out-of-place only
// (src and dst must not alias). Grid: (ceil(M*K/2 / 256), 1, 1)  Block: (256, 1, 1)
// Each thread converts 2 BF16 values → 2 FP8 values via cvt.e4m3x2.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void bf16_to_fp8(
    const __nv_bfloat16* __restrict__ src,   // [M, K] BF16
    unsigned char* __restrict__ dst,          // [M, K] FP8 E4M3
    unsigned int total_elements               // M * K (must be even)
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    unsigned int p = *(const unsigned int*)&src[idx];
    unsigned short bf0 = (unsigned short)(p & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p >> 16);
    float f0, f1;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    unsigned short fp8_pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(fp8_pair) : "f"(f1), "f"(f0));
    *(unsigned short*)&dst[idx] = fp8_pair;
}

// ═══════════════════════════════════════════════════════════════════
// FP8×FP8 GEMM: A [M, K] FP8 E4M3 × B [N, K] FP8 E4M3 → C [M, N] BF16
//
// Both A and B are pre-converted to FP8. No BF16→FP8 conversion in
// the inner loop — pure cp.async loads + FP8 MMA.
// Same tiling as fp8_gemm_t: M_TILE=64, N_TILE=128, K_STEP=32.
// A smem is FP8 (half the size of BF16 variant), no PAD needed.
// Grid: (ceil(N/128), ceil(M/64))  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
#define A_FP8_STRIDE 32  // K_STEP_T = 32 bytes per row for FP8

extern "C" __global__ void fp8_fp8_gemm_t(
    const unsigned char* __restrict__ A_fp8,  // [M, K] FP8 E4M3
    const unsigned char* __restrict__ B_fp8,  // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A smem: FP8 [64][32] = 2 KB per buffer (vs 5 KB BF16)
    __shared__ unsigned char smem_Af[2][M_TILE][A_FP8_STRIDE];
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    // Load A (FP8) + B (FP8) via cp.async — both 1 byte per element
    #define FF_LOADS(buf, kb) do { \
        { \
            /* 128 threads load 64 rows × 32 bytes: each thread loads 16 bytes */ \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            unsigned int gr = cta_m + row; \
            cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 15 < K)); \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8×FP8 MMA — no conversion needed, read A directly as FP8
    #define FF_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        /* A fragments: 4 bytes = 4 FP8 elements per register, 4 regs (a0..a3 for m16×k32) */ \
        unsigned int a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        unsigned int a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        unsigned int a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        unsigned int a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]); \
        } \
    } while(0)

    // Prolog: load first tile, wait
    FF_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Main loop: LOAD(nxt) || COMPUTE(cur) → wait → sync
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FF_LOADS(nxt, k_base);
        cp_async_commit();
        FF_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FF_COMPUTE(cur, cur);

    #undef FF_LOADS
    #undef FF_COMPUTE

    // Write results
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// K64 FP8-MMA transposed dense GEMM — halves outer K-loop vs K32.
//
// Same algorithm as w4a16_gemm_t but K_STEP_T64=64: 32 outer iterations
// instead of 64 for K=2048. Two m16n8k32 MMAs per N-tile per step.
// Reduces loop overhead and better amortizes DMA startup cost.
//
// K must be divisible by 64.
//
// smem: A 2×64×72×2=18432B, Bp 2×32×144=9216B, Bs 2×4×144=1152B,
//       B_fp8 128×80=10240B, LUT 64B = 39,104 B (~38.2 KiB)
// ═══════════════════════════════════════════════════════════════════
#define K_STEP_T64 64
#define PAD_T64    8   // (64+8)*2=144, 144%16=0 ✓

extern "C" __global__ void w4a16_gemm_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // B_fp8 row stride 80 = K64+16: avoids 4-way bank conflicts.
    __shared__ __nv_bfloat16 smem_A_k64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_k64[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_k64[16];

    if (threadIdx.x < 16) smem_LUT_k64[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast64 = K_STEP_T64 + PAD_T64;

    // A: 4 rounds × 16 rows = 64 rows (M_TILE); each thread: 8 BF16 = 16 bytes.
    // Bp: 2 rounds × 16 rows = 32 rows (K64/2); each thread: 16 bytes per ns chunk.
    // Bs: inline with Bp when kp_cur < K_STEP_T64/GROUP_SIZE (4 scale groups).
    #define K64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_k64[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                cp_async_pred_16(&smem_Bp_k64[(buf)][kp_cur][ns], \
                    &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 <= K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    cp_async_pred_16(&smem_Bs_k64[(buf)][kp_cur][ns], \
                        &B_scale[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    // 4 scale groups, 32 dequant iters: sv0→K{0..15}, sv1→K{16..31},
    // sv2→K{32..47}, sv3→K{48..63}.
    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2, sv3 = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // Two m16n8k32 MMA calls per N-tile: first covers K=0..31, second K=32..63.
    #define K64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_k64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]); \
        } \
        unsigned int a4 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 32 + tid * 4]); \
        unsigned int a5 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 32 + tid * 4]); \
        unsigned int a6 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 48 + tid * 4]); \
        unsigned int a7 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][48 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], \
                a4, a5, a6, a7, b0, b1, \
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]); \
        } \
    } while(0)

    K64_ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    K64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        K64_ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        K64_COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        K64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    K64_COMPUTE_MMA(cur);

    #undef K64_ISSUE_LOADS
    #undef K64_DEQUANT
    #undef K64_COMPUTE_MMA
    #undef K_STEP_T64
    #undef PAD_T64

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant: 2 consecutive 64-row M-chunks per CTA.
//
// For large-M prefill (e.g. ISL=1016, N=12288):
//   M_TILE=64: grid=(96,16,1)=1536 blocks, 16 weight re-reads  → 227MB B DRAM
//   M_TILE2=128: grid=(96,8,1)=768 blocks, 8 weight re-reads   → 114MB B DRAM
//
// SMEM: A 2×128×40×2=20480B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//       B_fp8 128×32=4096B, LUT 64B ≈ 29.8KB → 3 blocks/SM.
//
// For qkvz (K=2048,N=12288): ~2× speedup at ISL>128 vs w4a16_gemm_t.
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);  // base row for this block
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A is 2× larger (128 rows instead of 64); B/LUT/dequant identical to w4a16_gemm_t.
    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];   // 20480 B
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD]; // 4608 B
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD]; // 576 B
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];             // 4096 B
    __shared__ float smem_LUT[16];                                         //   64 B
    // Total ≈ 29.8 KB → 3 blocks/SM

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Two sets of accumulators: chunk0 = rows [cta_m..cta_m+63],
    //                           chunk1 = rows [cta_m+64..cta_m+127].
    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (4 rounds → 128 rows) + B (same as w4a16_gemm_t).
    #define M128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant B tile: identical to w4a16_gemm_t's DEQUANT_T.
    #define M128_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4]  * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4]  * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // MMA for both M-chunks; B tile (smem_B_fp8) loaded once, reused by both.
    #define M128_COMPUTE(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc0[nt][0], acc0[nt][1], acc0[nt][2], acc0[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc0[nt][0], acc0[nt][1], acc0[nt][2], acc0[nt][3]); \
        } \
        /* Chunk 1: smem rows 64..127 (offset M_TILE=64) */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc1[nt][0], acc1[nt][1], acc1[nt][2], acc1[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc1[nt][0], acc1[nt][1], acc1[nt][2], acc1[nt][3]); \
        } \
    } while(0)

    // Pipeline: same double-buffer structure as w4a16_gemm_t.
    M128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    M128_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M128_LOADS(nxt, k_base);
        cp_async_commit();
        M128_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        M128_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M128_COMPUTE(cur);

    #undef M128_LOADS
    #undef M128_DEQUANT
    #undef M128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    {
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M) {
            __nv_bfloat16* C_r0 = C + (unsigned long long)r0 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r0[c0] = __float2bfloat16(acc0[nt][0]);
                if (c1 < N) C_r0[c1] = __float2bfloat16(acc0[nt][1]);
            }
        }
        if (r1 < M) {
            __nv_bfloat16* C_r1 = C + (unsigned long long)r1 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r1[c0] = __float2bfloat16(acc0[nt][2]);
                if (c1 < N) C_r1[c1] = __float2bfloat16(acc0[nt][3]);
            }
        }
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    {
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M) {
            __nv_bfloat16* C_r0 = C + (unsigned long long)r0 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r0[c0] = __float2bfloat16(acc1[nt][0]);
                if (c1 < N) C_r0[c1] = __float2bfloat16(acc1[nt][1]);
            }
        }
        if (r1 < M) {
            __nv_bfloat16* C_r1 = C + (unsigned long long)r1 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r1[c0] = __float2bfloat16(acc1[nt][2]);
                if (c1 < N) C_r1[c1] = __float2bfloat16(acc1[nt][3]);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant of fp8_gemm_t: BF16 A × FP8 B, 2 M-chunks per CTA.
//
// For out_proj (K=2048, N=2048) and paged Q/K/V: halves the number of
// times B is read from DRAM (8 m-tile groups vs 16 at M=1015).
//
// SMEM: A 2×128×40×2=20480B, B 2×128×32=8192B ≈ 28.7KB → 3 blocks/SM.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void fp8_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];  // 20480 B
    __shared__ unsigned char  smem_B[2][N_TILE_LG][K_STEP_T];            //  8192 B

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (BF16, 4 rounds → 128 rows) + B (FP8, same as fp8_gemm_t).
    #define FGM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8 MMA for both M-chunks; B tile loaded once and reused.
    #define FGM128_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc0[nt][0], acc0[nt][1], acc0[nt][2], acc0[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc0[nt][0], acc0[nt][1], acc0[nt][2], acc0[nt][3]); \
        } \
        /* Chunk 1: smem rows 64..127 */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc1[nt][0], acc1[nt][1], acc1[nt][2], acc1[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc1[nt][0], acc1[nt][1], acc1[nt][2], acc1[nt][3]); \
        } \
    } while(0)

    FGM128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FGM128_LOADS(nxt, k_base);
        cp_async_commit();
        FGM128_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FGM128_COMPUTE(cur, cur);

    #undef FGM128_LOADS
    #undef FGM128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    {
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M) {
            __nv_bfloat16* C_r0 = C + (unsigned long long)r0 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r0[c0] = __float2bfloat16(acc0[nt][0]);
                if (c1 < N) C_r0[c1] = __float2bfloat16(acc0[nt][1]);
            }
        }
        if (r1 < M) {
            __nv_bfloat16* C_r1 = C + (unsigned long long)r1 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r1[c0] = __float2bfloat16(acc0[nt][2]);
                if (c1 < N) C_r1[c1] = __float2bfloat16(acc0[nt][3]);
            }
        }
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    {
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M) {
            __nv_bfloat16* C_r0 = C + (unsigned long long)r0 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r0[c0] = __float2bfloat16(acc1[nt][0]);
                if (c1 < N) C_r0[c1] = __float2bfloat16(acc1[nt][1]);
            }
        }
        if (r1 < M) {
            __nv_bfloat16* C_r1 = C + (unsigned long long)r1 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r1[c0] = __float2bfloat16(acc1[nt][2]);
                if (c1 < N) C_r1[c1] = __float2bfloat16(acc1[nt][3]);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant of fp8_fp8_gemm_t: FP8 A × FP8 B, 2 M-chunks per CTA.
//
// For Q/K/V projections in cache-skip prefill path (FP8 activations):
// halves B re-reads. Uses 3 blocks/SM (not 6) to avoid register spilling:
// dual acc0+acc1 need ~145 regs/thread; 3 blocks allows 170 regs/thread.
//
// SMEM: Af 2×128×32=8192B, Bf 2×128×32=8192B ≈ 16KB, 3 blocks → 48KB/SM.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void fp8_fp8_gemm_t_m128(
    const unsigned char* __restrict__ A_fp8,  // [M, K] FP8 E4M3
    const unsigned char* __restrict__ B_fp8,  // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ unsigned char smem_Af[2][2 * M_TILE][A_FP8_STRIDE];  //  8192 B
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];        //  8192 B

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    // Load A (FP8, 2 rounds → 128 rows) + B (FP8, same as fp8_fp8_gemm_t).
    #define FFM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                    &A_fp8[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 15 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8×FP8 MMA for both M-chunks; B loaded once, reused by both.
    #define FFM128_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc0[nt][0], acc0[nt][1], acc0[nt][2], acc0[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc0[nt][0], acc0[nt][1], acc0[nt][2], acc0[nt][3]); \
        } \
        /* Chunk 1: smem rows 64..127 */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc1[nt][0], acc1[nt][1], acc1[nt][2], acc1[nt][3], \
                a0, a1, a2, a3, b0, b1, \
                acc1[nt][0], acc1[nt][1], acc1[nt][2], acc1[nt][3]); \
        } \
    } while(0)

    FFM128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FFM128_LOADS(nxt, k_base);
        cp_async_commit();
        FFM128_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FFM128_COMPUTE(cur, cur);

    #undef FFM128_LOADS
    #undef FFM128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    {
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M) {
            __nv_bfloat16* C_r0 = C + (unsigned long long)r0 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r0[c0] = __float2bfloat16(acc0[nt][0]);
                if (c1 < N) C_r0[c1] = __float2bfloat16(acc0[nt][1]);
            }
        }
        if (r1 < M) {
            __nv_bfloat16* C_r1 = C + (unsigned long long)r1 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r1[c0] = __float2bfloat16(acc0[nt][2]);
                if (c1 < N) C_r1[c1] = __float2bfloat16(acc0[nt][3]);
            }
        }
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    {
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M) {
            __nv_bfloat16* C_r0 = C + (unsigned long long)r0 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r0[c0] = __float2bfloat16(acc1[nt][0]);
                if (c1 < N) C_r0[c1] = __float2bfloat16(acc1[nt][1]);
            }
        }
        if (r1 < M) {
            __nv_bfloat16* C_r1 = C + (unsigned long long)r1 * N;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int c0 = cta_n + nt * 8 + tid * 2;
                unsigned int c1 = c0 + 1;
                if (c0 < N) C_r1[c0] = __float2bfloat16(acc1[nt][2]);
                if (c1 < N) C_r1[c1] = __float2bfloat16(acc1[nt][3]);
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m32_n64 — small-M (M_TILE=32), small-N (N_TILE=64)
//
// Purpose-built for DFlash K=γ+1=17 verify:
//   * `w4a16_gemm_t_m16` covers M=17 with TWO M-tile rows, and each
//     tile row re-reads the full B matrix → 2× weight DRAM traffic on
//     a memory-bound GEMM.
//   * `w4a16_gemm_t_m128` covers M=17 in one tile (single B read) but
//     at N_TILE=128 fields only ~136 CTAs at inter=17408 (~2.8 CTAs/SM
//     on GB10's 48 SMs), under-occupied and measured ~1.8× off the
//     bandwidth floor.
// This variant does BOTH: one 32-row M-tile (single B read for any
// M ≤ 32) × N_TILE=64 (272 CTAs at inter=17408 ≈ 5.7 CTAs/SM).
//
// Geometry:
//   - Grid (ceil(N/64), ceil(M/32), 1)  Block (128, 1, 1) = 4 warps
//   - Each warp owns 2 N sub-tiles (16 cols) and BOTH 16-row M
//     fragments → 4 MMAs per K-step per warp (same pipeline density
//     as the validated m16 kernel).
//   - A load: 32 rows × 4 chunks of 8 BF16 = 128 thread-loads (full
//     block, one round). B/Bs loads use threads 0..63 / 0..7 with the
//     m16_n64 mapping.
//   - Dequant: 2 threads per N col (128 threads / 64 cols), each
//     handling one 16-K half with its own scale group.
//
// SMEM:
//   - A:        2 × 32 × 40 × 2 = 5120B
//   - Bp:       2 × 16 × 80     = 2560B
//   - Bs:       2 × 2  × 80     =  320B
//   - B_fp8:    64 × 32         = 2048B
//   - LUT:                          64B            ≈ 10.1 KB
//
// Caller contract: same as `w4a16_gemm_t_m16` plus a 9th `unsigned int ldb`
// stride parameter; accepts any M ≤ 32, output bounds-check discards padding rows.
// ═══════════════════════════════════════════════════════════════════

#define N_TILE_S3 64

extern "C" __global__ void w4a16_gemm_t_m32_n64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    // B-row stride in elements. Equals N for tightly-packed weights; a
    // 64-padded value for odd-N weights (e.g. the 248077-vocab lm_head)
    // where a tight stride would break cp.async 16B alignment. C stores
    // remain guarded by the logical N.
    unsigned int ldb
) {
    constexpr unsigned int M_TILE_S = 32;
    const unsigned int cta_n = blockIdx.x * N_TILE_S3;
    const unsigned int cta_m = blockIdx.y * M_TILE_S;
    const unsigned int warp_id = threadIdx.x / 32;  // 0..3
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;     // 0..7
    const unsigned int tid = lane_id & 3;           // 0..3

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_S][K_STEP_T + PAD_T];        // 5120B
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_S3 + BP_PAD]; // 2560B
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_S3 + BP_PAD]; // 320B
    __shared__ unsigned char smem_B_fp8[N_TILE_S3][K_STEP_T];               // 2048B
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Per-warp accumulators: [m_frag 0..1][n_subtile 0..1][4 fp32].
    float acc[2][2][4];
    #pragma unroll
    for (int mf = 0; mf < 2; mf++)
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            acc[mf][sub][0] = 0.0f; acc[mf][sub][1] = 0.0f;
            acc[mf][sub][2] = 0.0f; acc[mf][sub][3] = 0.0f;
        }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define ISSUE_LOADS_M32_N64(buf, kb) do { \
        { \
            unsigned int a_row = threadIdx.x >> 2;        /* 0..31 */ \
            unsigned int a_col = (threadIdx.x & 3) << 3;  /* 0/8/16/24 */ \
            unsigned int gc = (kb) + a_col; \
            unsigned int gr = cta_m + a_row; \
            cp_async_pred_16(&smem_A[(buf)][a_row][a_col], \
                &A[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 7 < K)); \
        } \
        if (threadIdx.x < 64) { \
            unsigned int kp = threadIdx.x >> 2;          /* 0..15 */ \
            unsigned int ns = (threadIdx.x & 3) << 4;    /* 0/16/32/48 */ \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * ldb + gns], \
                (gke + 1 <= K) && (gns + 15 < ldb)); \
            if (threadIdx.x < (K_STEP_T / GROUP_SIZE) * (N_TILE_S3 / 16)) { \
                unsigned int sg_row = threadIdx.x >> 2; \
                unsigned int sg_ns  = (threadIdx.x & 3) << 4; \
                unsigned int sg = (kb) / GROUP_SIZE + sg_row; \
                cp_async_pred_16(&smem_Bs[(buf)][sg_row][sg_ns], \
                    &B_scale[(unsigned long long)sg * ldb + cta_n + sg_ns], \
                    (cta_n + sg_ns + 15 < ldb)); \
            } \
        } \
    } while(0)

    // Dequant: 2 threads per col; thread half h handles K-half h
    // (kp 8h..8h+7) with scale group h. 128 threads cover 64 cols.
    #define DEQUANT_T_M32_N64(buf) do { \
        unsigned int my_n = threadIdx.x >> 1;            /* 0..63 */ \
        unsigned int half = threadIdx.x & 1;             /* 0..1  */ \
        unsigned char sb = smem_Bs[(buf)][half][my_n]; \
        __nv_fp8_e4m3 f; \
        *(unsigned char*)&f = sb; \
        float sv = (float)f * scale2; \
        unsigned int kp0 = half << 3; \
        _Pragma("unroll") \
        for (unsigned int kp = kp0; kp < kp0 + 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv; \
            float hi = smem_LUT[packed >> 4] * sv; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // FP8 MMA: 4 warps × (2 M-frags × 2 N-subtiles). Warp w owns cols
    // [w*16, w*16+16).
    #define COMPUTE_MMA_M32_N64(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        unsigned int b0 = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + tid * 4]); \
        unsigned int b1 = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + tid * 4]); \
        unsigned int b2 = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + 16 + tid * 4]); \
        unsigned int b3 = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 2; sub++) { \
            unsigned int nt = warp_id * 2 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int v0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int v1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[0][sub][0], acc[0][sub][1], acc[0][sub][2], acc[0][sub][3], \
                a0, a1, a2, a3, v0, v1, \
                acc[0][sub][0], acc[0][sub][1], acc[0][sub][2], acc[0][sub][3]); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[1][sub][0], acc[1][sub][1], acc[1][sub][2], acc[1][sub][3], \
                b0, b1, b2, b3, v0, v1, \
                acc[1][sub][0], acc[1][sub][1], acc[1][sub][2], acc[1][sub][3]); \
        } \
    } while(0)

    ISSUE_LOADS_M32_N64(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_T_M32_N64(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS_M32_N64(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA_M32_N64(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_T_M32_N64(nxt);
        __syncthreads();
        cur = nxt;
    }
    COMPUTE_MMA_M32_N64(cur);

    #undef ISSUE_LOADS_M32_N64
    #undef DEQUANT_T_M32_N64
    #undef COMPUTE_MMA_M32_N64

    // Output: each warp writes 2 N sub-tiles × 4 row groups
    // (group_id, +8, +16, +24), bounds-checked against M.
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        unsigned int nt = warp_id * 2 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        unsigned int r2 = r0 + 16;
        unsigned int r3 = r0 + 24;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = __float2bfloat16(acc[0][sub][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = __float2bfloat16(acc[0][sub][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = __float2bfloat16(acc[0][sub][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = __float2bfloat16(acc[0][sub][3]);
        if (r2 < M && c0 < N) C[(unsigned long long)r2 *N+c0] = __float2bfloat16(acc[1][sub][0]);
        if (r2 < M && c1 < N) C[(unsigned long long)r2 *N+c1] = __float2bfloat16(acc[1][sub][1]);
        if (r3 < M && c0 < N) C[(unsigned long long)r3 *N+c0] = __float2bfloat16(acc[1][sub][2]);
        if (r3 < M && c1 < N) C[(unsigned long long)r3 *N+c1] = __float2bfloat16(acc[1][sub][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m32_n64_splitk — split-K variant of w4a16_gemm_t_m32_n64
//
// Purpose: the DFlash K=17 verify FFN
// `down_proj` GEMM has shape [M=17, N=5120, K=17408]. The base
// m32_n64 kernel fields grid (ceil(5120/64), 1) = 80 CTAs (~1.7 CTAs/SM
// on GB10's 48 SMs), yet each CTA grinds a 544-iteration K-loop
// (K=17408/K_STEP_T=32). Measured: down runs at ~91 GB/s vs the
// gate/up projections' ~163 GB/s on an identically-sized weight, i.e.
// it is OCCUPANCY-starved, not bandwidth-bound. gate/up have N=17408
// → 272 CTAs and saturate; down does not.
//
// Fix: add a gridDim.z = SPLITK dimension. Slice z owns K-range
// [z*Kc, (z+1)*Kc) (Kc = ceil(K / SPLITK), rounded to K_STEP_T). Each
// slice accumulates its partial [M,N] into a FP32 scratch row-band
// `Cpartial + z*M*N`. A companion `reduce_splitk_f32_to_bf16` sums the
// SPLITK partials and writes the BF16 output. This multiplies CTA
// count by SPLITK (80→320 at SPLITK=4) without atomics, restoring full
// occupancy.
//
// PRECISION: the partials are FP32, so nothing rounds to BF16 mid-
// accumulation. That is NOT the same as reproducing the single-slice
// result: this kernel restarts the accumulator at each slice boundary
// and reduce_splitk_f32_to_bf16 sums the slices afterwards, whereas the
// base kernel runs one left-to-right FP32 accumulation over all of K.
// FP32 addition is not associative, so the BF16 output can differ by 1 ULP
// vs the single-slice kernel (expected split-K reassociation).
//
// Geometry identical to w4a16_gemm_t_m32_n64 except:
//   - Grid (ceil(N/64), ceil(M/32), SPLITK)  Block (128,1,1)
//   - K-loop bounded to [k_lo, k_hi) for this z-slice
//   - Stores FP32 to Cpartial[z*M*N + r*N + c] (no bounds-add; plain
//     write, the reduce kernel reads all SPLITK bands).
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void w4a16_gemm_t_m32_n64_splitk(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ Cpartial,           // [SPLITK, M, N] FP32
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb,
    unsigned int splitk                     // number of K-slices (gridDim.z)
) {
    constexpr unsigned int M_TILE_S = 32;
    const unsigned int cta_n = blockIdx.x * N_TILE_S3;
    const unsigned int cta_m = blockIdx.y * M_TILE_S;
    const unsigned int zslice = blockIdx.z;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // K-range for this slice, snapped to K_STEP_T so group/scale indexing
    // stays aligned (GROUP_SIZE=16 divides K_STEP_T=32).
    unsigned int kc = (K + splitk - 1) / splitk;
    kc = ((kc + K_STEP_T - 1) / K_STEP_T) * K_STEP_T;   // round up to K_STEP_T
    unsigned int k_lo = zslice * kc;
    unsigned int k_hi = k_lo + kc;
    if (k_hi > K) k_hi = K;
    // Empty slice (can happen when splitk*kc overshoots): write zeros so the
    // reduce kernel reads a defined value, then bail.
    if (k_lo >= K) {
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            unsigned int nt = warp_id * 2 + sub;
            unsigned int c0 = cta_n + nt * 8 + tid * 2;
            unsigned int c1 = c0 + 1;
            unsigned int r0 = cta_m + group_id;
            unsigned int rows[4] = {r0, r0 + 8, r0 + 16, r0 + 24};
            float* base = Cpartial + (unsigned long long)zslice * M * N;
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                if (rows[rr] < M && c0 < N) base[rows[rr]*N + c0] = 0.0f;
                if (rows[rr] < M && c1 < N) base[rows[rr]*N + c1] = 0.0f;
            }
        }
        return;
    }

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_S][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_S3 + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_S3 + BP_PAD];
    __shared__ unsigned char smem_B_fp8[N_TILE_S3][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[2][2][4];
    #pragma unroll
    for (int mf = 0; mf < 2; mf++)
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            acc[mf][sub][0] = 0.0f; acc[mf][sub][1] = 0.0f;
            acc[mf][sub][2] = 0.0f; acc[mf][sub][3] = 0.0f;
        }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Identical load/dequant/MMA macros to w4a16_gemm_t_m32_n64, but the
    // global K-coordinate `gc/gke/sg` is offset by the absolute k position
    // (the macro arg `kb` is already absolute below) and predicates use the
    // full K (so the weight-row stride math is unchanged); the loop bounds
    // restrict to [k_lo, k_hi).
    #define ISSUE_LOADS_SK(buf, kb) do { \
        { \
            unsigned int a_row = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            unsigned int gr = cta_m + a_row; \
            cp_async_pred_16(&smem_A[(buf)][a_row][a_col], \
                &A[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 7 < K)); \
        } \
        if (threadIdx.x < 64) { \
            unsigned int kp = threadIdx.x >> 2; \
            unsigned int ns = (threadIdx.x & 3) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * ldb + gns], \
                (gke + 1 <= K) && (gns + 15 < ldb)); \
            if (threadIdx.x < (K_STEP_T / GROUP_SIZE) * (N_TILE_S3 / 16)) { \
                unsigned int sg_row = threadIdx.x >> 2; \
                unsigned int sg_ns  = (threadIdx.x & 3) << 4; \
                unsigned int sg = (kb) / GROUP_SIZE + sg_row; \
                cp_async_pred_16(&smem_Bs[(buf)][sg_row][sg_ns], \
                    &B_scale[(unsigned long long)sg * ldb + cta_n + sg_ns], \
                    (cta_n + sg_ns + 15 < ldb)); \
            } \
        } \
    } while(0)

    #define DEQUANT_SK(buf) do { \
        unsigned int my_n = threadIdx.x >> 1; \
        unsigned int half = threadIdx.x & 1; \
        unsigned char sb = smem_Bs[(buf)][half][my_n]; \
        __nv_fp8_e4m3 f; \
        *(unsigned char*)&f = sb; \
        float sv = (float)f * scale2; \
        unsigned int kp0 = half << 3; \
        _Pragma("unroll") \
        for (unsigned int kp = kp0; kp < kp0 + 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv; \
            float hi = smem_LUT[packed >> 4] * sv; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    #define COMPUTE_MMA_SK(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        unsigned int b0 = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + tid * 4]); \
        unsigned int b1 = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + tid * 4]); \
        unsigned int b2 = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + 16 + tid * 4]); \
        unsigned int b3 = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 2; sub++) { \
            unsigned int nt = warp_id * 2 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int v0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int v1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[0][sub][0], acc[0][sub][1], acc[0][sub][2], acc[0][sub][3], \
                a0, a1, a2, a3, v0, v1, \
                acc[0][sub][0], acc[0][sub][1], acc[0][sub][2], acc[0][sub][3]); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[1][sub][0], acc[1][sub][1], acc[1][sub][2], acc[1][sub][3], \
                b0, b1, b2, b3, v0, v1, \
                acc[1][sub][0], acc[1][sub][1], acc[1][sub][2], acc[1][sub][3]); \
        } \
    } while(0)

    ISSUE_LOADS_SK(0, k_lo);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_SK(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = k_lo + K_STEP_T; k_base < k_hi; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS_SK(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA_SK(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_SK(nxt);
        __syncthreads();
        cur = nxt;
    }
    COMPUTE_MMA_SK(cur);

    #undef ISSUE_LOADS_SK
    #undef DEQUANT_SK
    #undef COMPUTE_MMA_SK

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    float* base = Cpartial + (unsigned long long)zslice * M * N;
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        unsigned int nt = warp_id * 2 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        unsigned int r2 = r0 + 16;
        unsigned int r3 = r0 + 24;
        if (r0 < M && c0 < N) base[r0*N+c0] = acc[0][sub][0];
        if (r0 < M && c1 < N) base[r0*N+c1] = acc[0][sub][1];
        if (r1 < M && c0 < N) base[r1*N+c0] = acc[0][sub][2];
        if (r1 < M && c1 < N) base[r1*N+c1] = acc[0][sub][3];
        if (r2 < M && c0 < N) base[r2*N+c0] = acc[1][sub][0];
        if (r2 < M && c1 < N) base[r2*N+c1] = acc[1][sub][1];
        if (r3 < M && c0 < N) base[r3*N+c0] = acc[1][sub][2];
        if (r3 < M && c1 < N) base[r3*N+c1] = acc[1][sub][3];
    }
}

// Reduce the [SPLITK, M, N] FP32 partials produced by
// w4a16_gemm_t_m32_n64_splitk into a [M, N] BF16 output. One thread per
// (row, col) output element; sums the SPLITK bands. Grid covers M*N.
extern "C" __global__ void reduce_splitk_f32_to_bf16(
    const float* __restrict__ Cpartial,     // [SPLITK, M, N]
    __nv_bfloat16* __restrict__ C,           // [M, N]
    unsigned int M, unsigned int N, unsigned int splitk
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long mn = (unsigned long long)M * N;
    if (idx >= mn) return;
    const float* p = Cpartial + idx;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    float sum = 0.0f;
    for (unsigned int z = 0; z < splitk; z++) {
        sum += *p;
        p += mn;
    }
    C[idx] = __float2bfloat16(sum);
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m32_n64_gateup_silu — FUSED gate_proj + up_proj + SiLU·mul
//
// Fork of `w4a16_gemm_t_m32_n64`. The DFlash K=γ+1 verify FFN runs
// gate = gate_proj(x); up = up_proj(x); act = silu(gate)*up. The two
// projections share the SAME input `A` [M, K=hidden] and the SAME
// output shape [M, N=inter]. The baseline dispatches them as two
// separate m32_n64 GEMMs (each re-loading the A tile + writing a full
// [M,N] activation) followed by a third `moe_silu_mul` kernel that
// reads both [M,N] activations and writes one.
//
// This kernel loads A ONCE into SMEM (shared across both projections),
// streams BOTH B weights (gate + up) through their own double-buffered
// cp.async pipelines, keeps two independent accumulator sets, and at
// output computes silu(gate)*up in registers — writing only the single
// [M,N] activation. Eliminates:
//   * one full re-read of the [M, K] A tile (small: ~174 KB at M=17)
//   * the two [M,N] activation WRITES + two READS of the standalone
//     silu_mul (gate_out + up_out ≈ 2×[17,17408]×2B = 1.18 MB write +
//     1.18 MB read) → replaced by ONE [M,N] write
//   * two kernel launches (2 GEMM + 1 silu → 1 fused)
//
// NOTE: the dominant DRAM traffic is the two 47 MB weights
// (gate + up), which this kernel still reads in full — fusion cannot
// reduce weight bandwidth. The saved traffic (~2.2 MB activation +
// A re-read) is ~2% of the ~96 MB total, so the win is bounded by
// launch-overhead + activation-roundtrip elimination, NOT a bandwidth
// speedup. A/B measured, gated behind ATLAS_FFN_FUSED_GATEUP.
//
// SMEM (2× B pipeline vs the single-B m32_n64):
//   - A:        2 × 32 × 40 × 2          = 5120 B  (shared)
//   - Bp ×2:    2 × 2 × 16 × 80          = 5120 B
//   - Bs ×2:    2 × 2 × 2  × 80          =  640 B
//   - B_fp8 ×2: 2 × 64 × 32              = 4096 B
//   - LUT:                                    64 B  ≈ 15.0 KB
//
// Caller contract: identical geometry to w4a16_gemm_t_m32_n64 but with
// TWO weight triples (gate then up) and a shared ldb. Output C is the
// fused silu(gate)*up activation [M, N]. Accepts any M ≤ 32.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void w4a16_gemm_t_m32_n64_gateup_silu(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,          // fused silu(gate)*up [M,N]
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb
) {
    constexpr unsigned int M_TILE_S = 32;
    const unsigned int cta_n = blockIdx.x * N_TILE_S3;
    const unsigned int cta_m = blockIdx.y * M_TILE_S;
    const unsigned int warp_id = threadIdx.x / 32;  // 0..3
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;     // 0..7
    const unsigned int tid = lane_id & 3;           // 0..3

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_S][K_STEP_T + PAD_T];        // 5120B
    // Two B pipelines: [0]=gate, [1]=up.
    __shared__ unsigned char smem_Bp[2][2][K_STEP_T / 2][N_TILE_S3 + BP_PAD]; // 5120B
    __shared__ unsigned char smem_Bs[2][2][K_STEP_T / GROUP_SIZE][N_TILE_S3 + BP_PAD]; // 640B
    __shared__ unsigned char smem_B_fp8[2][N_TILE_S3][K_STEP_T];           // 4096B
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Per-warp accumulators for BOTH projections:
    // acc[w][m_frag 0..1][n_subtile 0..1][4 fp32]. w: 0=gate, 1=up.
    float acc[2][2][2][4];
    #pragma unroll
    for (int w = 0; w < 2; w++)
        #pragma unroll
        for (int mf = 0; mf < 2; mf++)
            #pragma unroll
            for (int sub = 0; sub < 2; sub++) {
                acc[w][mf][sub][0] = 0.0f; acc[w][mf][sub][1] = 0.0f;
                acc[w][mf][sub][2] = 0.0f; acc[w][mf][sub][3] = 0.0f;
            }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // A loads (shared) + BOTH B loads for the given weight index w.
    #define ISSUE_A_M32(buf, kb) do { \
        unsigned int a_row = threadIdx.x >> 2;        /* 0..31 */ \
        unsigned int a_col = (threadIdx.x & 3) << 3;  /* 0/8/16/24 */ \
        unsigned int gc = (kb) + a_col; \
        unsigned int gr = cta_m + a_row; \
        cp_async_pred_16(&smem_A[(buf)][a_row][a_col], \
            &A[(unsigned long long)gr * K + gc], \
            (gr < M) && (gc + 7 < K)); \
    } while(0)

    #define ISSUE_B_M32(w, buf, kb, Bp, Bs) do { \
        if (threadIdx.x < 64) { \
            unsigned int kp = threadIdx.x >> 2;          /* 0..15 */ \
            unsigned int ns = (threadIdx.x & 3) << 4;    /* 0/16/32/48 */ \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(w)][(buf)][kp][ns], \
                &Bp[(unsigned long long)(gke >> 1) * ldb + gns], \
                (gke + 1 <= K) && (gns + 15 < ldb)); \
            if (threadIdx.x < (K_STEP_T / GROUP_SIZE) * (N_TILE_S3 / 16)) { \
                unsigned int sg_row = threadIdx.x >> 2; \
                unsigned int sg_ns  = (threadIdx.x & 3) << 4; \
                unsigned int sg = (kb) / GROUP_SIZE + sg_row; \
                cp_async_pred_16(&smem_Bs[(w)][(buf)][sg_row][sg_ns], \
                    &Bs[(unsigned long long)sg * ldb + cta_n + sg_ns], \
                    (cta_n + sg_ns + 15 < ldb)); \
            } \
        } \
    } while(0)

    // Dequant weight w's B tile → smem_B_fp8[w].
    #define DEQUANT_M32(w, buf, sc2) do { \
        unsigned int my_n = threadIdx.x >> 1;            /* 0..63 */ \
        unsigned int half = threadIdx.x & 1;             /* 0..1  */ \
        unsigned char sb = smem_Bs[(w)][(buf)][half][my_n]; \
        __nv_fp8_e4m3 f; \
        *(unsigned char*)&f = sb; \
        float sv = (float)f * (sc2); \
        unsigned int kp0 = half << 3; \
        _Pragma("unroll") \
        for (unsigned int kp = kp0; kp < kp0 + 8; kp++) { \
            unsigned char packed = smem_Bp[(w)][(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv; \
            float hi = smem_LUT[packed >> 4] * sv; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[(w)][my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // MMA for weight w, using the shared A tile (a_buf) and B_fp8[w].
    #define COMPUTE_MMA_M32(w, a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        unsigned int b0 = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + tid * 4]); \
        unsigned int b1 = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + tid * 4]); \
        unsigned int b2 = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + 16 + tid * 4]); \
        unsigned int b3 = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 2; sub++) { \
            unsigned int nt = warp_id * 2 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int v0 = *(const unsigned int*)&smem_B_fp8[(w)][nc][4 * tid]; \
            unsigned int v1 = *(const unsigned int*)&smem_B_fp8[(w)][nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3], \
                a0, a1, a2, a3, v0, v1, \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3]); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3], \
                b0, b1, b2, b3, v0, v1, \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3]); \
        } \
    } while(0)

    // Prologue: load A + both B tiles for k_base=0.
    ISSUE_A_M32(0, 0);
    ISSUE_B_M32(0, 0, 0, Bg_packed, Bg_scale);
    ISSUE_B_M32(1, 0, 0, Bu_packed, Bu_scale);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_M32(0, 0, scale2_g);
    DEQUANT_M32(1, 0, scale2_u);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_A_M32(nxt, k_base);
        ISSUE_B_M32(0, nxt, k_base, Bg_packed, Bg_scale);
        ISSUE_B_M32(1, nxt, k_base, Bu_packed, Bu_scale);
        cp_async_commit();
        COMPUTE_MMA_M32(0, cur);
        COMPUTE_MMA_M32(1, cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_M32(0, nxt, scale2_g);
        DEQUANT_M32(1, nxt, scale2_u);
        __syncthreads();
        cur = nxt;
    }
    COMPUTE_MMA_M32(0, cur);
    COMPUTE_MMA_M32(1, cur);

    #undef ISSUE_A_M32
    #undef ISSUE_B_M32
    #undef DEQUANT_M32
    #undef COMPUTE_MMA_M32

    // Output: silu(gate)*up per element, single [M,N] write.
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        unsigned int nt = warp_id * 2 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        unsigned int r2 = r0 + 16;
        unsigned int r3 = r0 + 24;
        // SiLU(g)*u. Round the gate/up accumulators to BF16 and back to
        // FP32 FIRST — this reproduces the baseline flow byte-for-byte:
        // the standalone m32_n64 GEMMs write gate_out/up_out as BF16
        // (__float2bfloat16(acc)), then `moe_silu_mul` reloads them as
        // FP32 (__bfloat162float) and computes g*sigmoid(g)*u. Keeping the
        // raw FP32 accumulators would be *more* precise but would diverge
        // in the low bits → breaks the md5 gate. Match the baseline exactly.
        #define FUSE(gi, ui) ({ \
            float g_ = __bfloat162float(__float2bfloat16(gi)); \
            float u_ = __bfloat162float(__float2bfloat16(ui)); \
            float sig_ = 1.0f / (1.0f + __expf(-g_)); \
            __float2bfloat16(g_ * sig_ * u_); })
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = FUSE(acc[0][0][sub][0], acc[1][0][sub][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = FUSE(acc[0][0][sub][1], acc[1][0][sub][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = FUSE(acc[0][0][sub][2], acc[1][0][sub][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = FUSE(acc[0][0][sub][3], acc[1][0][sub][3]);
        if (r2 < M && c0 < N) C[(unsigned long long)r2 *N+c0] = FUSE(acc[0][1][sub][0], acc[1][1][sub][0]);
        if (r2 < M && c1 < N) C[(unsigned long long)r2 *N+c1] = FUSE(acc[0][1][sub][1], acc[1][1][sub][1]);
        if (r3 < M && c0 < N) C[(unsigned long long)r3 *N+c0] = FUSE(acc[0][1][sub][2], acc[1][1][sub][2]);
        if (r3 < M && c1 < N) C[(unsigned long long)r3 *N+c1] = FUSE(acc[0][1][sub][3], acc[1][1][sub][3]);
        #undef FUSE
    }
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m32_n64_gateup_silu_pipe3 — 3-stage triple-buffered fork.
//
// SAME math as w4a16_gemm_t_m32_n64_gateup_silu — identical load, DEQUANT
// (FP8 E4M3 conversion via cvt.rn.satfinite.e4m3x2), MMA, and FUSE (BF16
// round-trip SiLU·mul). SCHEDULING CHANGE ONLY, zero arithmetic change:
//
//   2-stage baseline: ISSUE(nxt); commit; MMA(cur); WAIT_ALL; DEQUANT(nxt).
//     WAIT_ALL (wait_group 0) drains EVERY outstanding group before dequant,
//     and DEQUANT sits on the critical path between the wait and the next
//     MMA — the ~36µs dequant is fully exposed and the next tile's loads
//     cannot overlap it.
//
//   3-stage pipe: keep TWO cp.async groups in flight. Producer issues stage
//     k+2's A+B loads, then `cp.async.wait_group<2>` blocks only until stage
//     k's loads land (≤2 groups remain). We then DEQUANT stage k and MMA
//     stage k while stage k+1 is already resident and stage k+2 streams in
//     the background. Dequant of k overlaps the in-flight load of k+2, and
//     MMA of k overlaps nothing extra but is no longer gated by a full drain.
//
// Ring depth 3 for A / Bp / Bs / B_fp8 (buf = k % 3). SMEM grows ~1.5× vs
// the 2-stage kernel (still well under the 100 KB SM_120 cap):
//   - A:        3 × 32 × 40 × 2      =  7680 B
//   - Bp ×2:    3 × 2 × 16 × 80      =  7680 B
//   - Bs ×2:    3 × 2 × 2  × 80      =   960 B
//   - B_fp8 ×2: 2 × 3 × 64 × 32      = 12288 B  ≈ 28.7 KB + LUT
// Caller contract identical to w4a16_gemm_t_m32_n64_gateup_silu.
// ═══════════════════════════════════════════════════════════════════
#define N_TILE_S3 64
extern "C" __global__ void w4a16_gemm_t_m32_n64_gateup_silu_pipe3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb
) {
    constexpr unsigned int M_TILE_S = 32;
    constexpr int STAGES = 3;
    const unsigned int cta_n = blockIdx.x * N_TILE_S3;
    const unsigned int cta_m = blockIdx.y * M_TILE_S;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[STAGES][M_TILE_S][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][STAGES][K_STEP_T / 2][N_TILE_S3 + BP_PAD];
    __shared__ unsigned char smem_Bs[2][STAGES][K_STEP_T / GROUP_SIZE][N_TILE_S3 + BP_PAD];
    __shared__ unsigned char smem_B_fp8[2][STAGES][N_TILE_S3][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[2][2][2][4];
    #pragma unroll
    for (int w = 0; w < 2; w++)
        #pragma unroll
        for (int mf = 0; mf < 2; mf++)
            #pragma unroll
            for (int sub = 0; sub < 2; sub++) {
                acc[w][mf][sub][0] = 0.0f; acc[w][mf][sub][1] = 0.0f;
                acc[w][mf][sub][2] = 0.0f; acc[w][mf][sub][3] = 0.0f;
            }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // A loads into ring slot `buf` for global k-offset `kb`.
    #define P3_ISSUE_A(buf, kb) do { \
        unsigned int a_row = threadIdx.x >> 2; \
        unsigned int a_col = (threadIdx.x & 3) << 3; \
        unsigned int gc = (kb) + a_col; \
        unsigned int gr = cta_m + a_row; \
        cp_async_pred_16(&smem_A[(buf)][a_row][a_col], \
            &A[(unsigned long long)gr * K + gc], \
            (gr < M) && (gc + 7 < K)); \
    } while(0)

    // B loads for weight w into ring slot `buf`.
    #define P3_ISSUE_B(w, buf, kb, Bp, Bs) do { \
        if (threadIdx.x < 64) { \
            unsigned int kp = threadIdx.x >> 2; \
            unsigned int ns = (threadIdx.x & 3) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(w)][(buf)][kp][ns], \
                &Bp[(unsigned long long)(gke >> 1) * ldb + gns], \
                (gke + 1 <= K) && (gns + 15 < ldb)); \
            if (threadIdx.x < (K_STEP_T / GROUP_SIZE) * (N_TILE_S3 / 16)) { \
                unsigned int sg_row = threadIdx.x >> 2; \
                unsigned int sg_ns  = (threadIdx.x & 3) << 4; \
                unsigned int sg = (kb) / GROUP_SIZE + sg_row; \
                cp_async_pred_16(&smem_Bs[(w)][(buf)][sg_row][sg_ns], \
                    &Bs[(unsigned long long)sg * ldb + cta_n + sg_ns], \
                    (cta_n + sg_ns + 15 < ldb)); \
            } \
        } \
    } while(0)

    // Issue A + both B tiles for one stage (one cp.async group).
    #define P3_ISSUE_STAGE(buf, kb) do { \
        P3_ISSUE_A((buf), (kb)); \
        P3_ISSUE_B(0, (buf), (kb), Bg_packed, Bg_scale); \
        P3_ISSUE_B(1, (buf), (kb), Bu_packed, Bu_scale); \
        cp_async_commit(); \
    } while(0)

    #define P3_DEQUANT(w, buf, sc2) do { \
        unsigned int my_n = threadIdx.x >> 1; \
        unsigned int half = threadIdx.x & 1; \
        unsigned char sb = smem_Bs[(w)][(buf)][half][my_n]; \
        __nv_fp8_e4m3 f; \
        *(unsigned char*)&f = sb; \
        float sv = (float)f * (sc2); \
        unsigned int kp0 = half << 3; \
        _Pragma("unroll") \
        for (unsigned int kp = kp0; kp < kp0 + 8; kp++) { \
            unsigned char packed = smem_Bp[(w)][(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv; \
            float hi = smem_LUT[packed >> 4] * sv; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[(w)][(buf)][my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    #define P3_MMA(w, buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        unsigned int fr0 = group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        unsigned int b0 = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + tid * 4]); \
        unsigned int b1 = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + tid * 4]); \
        unsigned int b2 = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + 16 + tid * 4]); \
        unsigned int b3 = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 2; sub++) { \
            unsigned int nt = warp_id * 2 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int v0 = *(const unsigned int*)&smem_B_fp8[(w)][(buf)][nc][4 * tid]; \
            unsigned int v1 = *(const unsigned int*)&smem_B_fp8[(w)][(buf)][nc][16 + 4 * tid]; \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3], \
                a0, a1, a2, a3, v0, v1, \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3]); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3], \
                b0, b1, b2, b3, v0, v1, \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3]); \
        } \
    } while(0)

    const unsigned int num_k = K / K_STEP_T;   // K is a multiple of K_STEP_T

    // Prologue: prime up to (STAGES-1)=2 cp.async groups in flight.
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if ((unsigned)s < num_k) P3_ISSUE_STAGE(s, (unsigned)s * K_STEP_T);
    }

    // Steady state.
    for (unsigned int k = 0; k < num_k; k++) {
        unsigned int buf = k % STAGES;
        // Prefetch stage k+2 (keeps 2 groups ahead) before consuming k.
        unsigned int pf = k + (STAGES - 1);
        bool prefetched = pf < num_k;
        if (prefetched) {
            P3_ISSUE_STAGE(pf % STAGES, pf * K_STEP_T);
        }
        // Block until stage k's loads have landed. In steady state exactly
        // STAGES-1 groups are ahead of stage k, so leaving ≤(STAGES-1) in
        // flight guarantees stage k is done. In the TAIL (no prefetch issued),
        // fewer than STAGES-1 groups remain — waiting for ≤STAGES-1 could
        // return WITHOUT completing stage k. Drain fully there: the remaining
        // resident stages stay valid (wait_group<0> only blocks, never evicts).
        if (prefetched) {
            cp_async_wait_group<STAGES - 1>();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();
        P3_DEQUANT(0, buf, scale2_g);
        P3_DEQUANT(1, buf, scale2_u);
        __syncthreads();
        P3_MMA(0, buf);
        P3_MMA(1, buf);
        __syncthreads();
    }

    #undef P3_ISSUE_A
    #undef P3_ISSUE_B
    #undef P3_ISSUE_STAGE
    #undef P3_DEQUANT
    #undef P3_MMA

    // Output: silu(gate)*up — identical FUSE to the 2-stage kernel.
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        unsigned int nt = warp_id * 2 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        unsigned int r2 = r0 + 16;
        unsigned int r3 = r0 + 24;
        #define FUSE(gi, ui) ({ \
            float g_ = __bfloat162float(__float2bfloat16(gi)); \
            float u_ = __bfloat162float(__float2bfloat16(ui)); \
            float sig_ = 1.0f / (1.0f + __expf(-g_)); \
            __float2bfloat16(g_ * sig_ * u_); })
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = FUSE(acc[0][0][sub][0], acc[1][0][sub][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = FUSE(acc[0][0][sub][1], acc[1][0][sub][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = FUSE(acc[0][0][sub][2], acc[1][0][sub][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = FUSE(acc[0][0][sub][3], acc[1][0][sub][3]);
        if (r2 < M && c0 < N) C[(unsigned long long)r2 *N+c0] = FUSE(acc[0][1][sub][0], acc[1][1][sub][0]);
        if (r2 < M && c1 < N) C[(unsigned long long)r2 *N+c1] = FUSE(acc[0][1][sub][1], acc[1][1][sub][1]);
        if (r3 < M && c0 < N) C[(unsigned long long)r3 *N+c0] = FUSE(acc[0][1][sub][2], acc[1][1][sub][2]);
        if (r3 < M && c1 < N) C[(unsigned long long)r3 *N+c1] = FUSE(acc[0][1][sub][3], acc[1][1][sub][3]);
        #undef FUSE
    }
}

#undef N_TILE_S3

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m32_n64_gateup_silu_pipe — DEQUANT-IN-REGISTERS fork of
// w4a16_gemm_t_m32_n64_gateup_silu (`ATLAS_DEQUANT_PIPE=1`).
//
// Same shape / caller contract / OUTPUT MATH as the 2-stage baseline
// (byte-exact — see the bit-exactness argument at the bottom). The ONLY
// change is WHERE the NVFP4→FP8 dequant lands and HOW the K-loop is
// scheduled:
//
//   Baseline 2-stage steady-state (per K-step):
//     ISSUE(nxt); commit; MMA_from_smem_fp8(cur);
//     cp.async.wait_group 0;  __syncthreads();
//     DEQUANT(cur→smem_B_fp8);  __syncthreads();
//   The wait_group 0 fully DRAINS every outstanding cp.async group, then
//   DEQUANT sits on the critical path between the drain and the next MMA,
//   fully exposed, gated by TWO __syncthreads per step. B is materialized
//   into a separate 4 KB smem_B_fp8[2][64][32] staging array.
//
//   This kernel: NO smem_B_fp8 staging. SMEM holds only the PACKED W4
//   bytes (smem_Bp) + FP8 scales (smem_Bs). The FP4→FP8 unpack+scale runs
//   in REGISTERS inside the MMA macro, immediately before each
//   mma.sync.e4m3, from the resident packed bytes of the CURRENT tile.
//   Meanwhile the NEXT tile's cp.async is already in flight and we only
//   `cp.async.wait_group 1` (leave 1 group ahead) instead of draining —
//   so the unpack ALU of tile k overlaps the memory latency of tile k+1.
//   One __syncthreads per step (the dequant→smem barrier is gone).
//
// Dequant-in-regs is the load-latency-hiding lever the SMEM-staged
// baseline cannot get: the baseline's DEQUANT must FINISH (2nd sync)
// before ANY MMA can read smem_B_fp8, so it is serialized after the
// drain. Here the per-column unpack is private to the consuming thread
// and interleaves with the MMA issue + the background load.
//
// SMEM (2-stage, NO fp8 staging → 4 KB smaller than the baseline fused
// kernel's ~15 KB → higher occupancy, the opposite of the pipe3 attempt
// which GREW smem and lost a block/SM):
//   - A:      2 × 32 × 40 × 2   = 5120 B  (shared)
//   - Bp ×2:  2 × 2 × 16 × 80   = 5120 B
//   - Bs ×2:  2 × 2 × 2  × 80   =  640 B
//   - LUT:                          64 B  ≈ 10.9 KB  (vs 15.0 KB baseline)
//
// Because the packed bytes stay in SMEM and only the transient FP8 lives
// in registers, the 4× SMEM saving vs the staged path preserves (indeed
// improves) blocks/SM — directly answering the pipe3 no-op: that fork
// LOST by growing SMEM 1.5×; this one SHRINKS it.
// ═══════════════════════════════════════════════════════════════════
#define N_TILE_S3 64
extern "C" __global__ void w4a16_gemm_t_m32_n64_gateup_silu_pipe(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,          // fused silu(gate)*up [M,N]
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb
) {
    constexpr unsigned int M_TILE_S = 32;
    const unsigned int cta_n = blockIdx.x * N_TILE_S3;
    const unsigned int cta_m = blockIdx.y * M_TILE_S;
    const unsigned int warp_id = threadIdx.x / 32;  // 0..3
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;     // 0..7
    const unsigned int tid = lane_id & 3;           // 0..3

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_S][K_STEP_T + PAD_T];        // 5120B
    // Two B pipelines: [0]=gate, [1]=up. Packed W4 + FP8 scales ONLY —
    // no dequanted smem_B_fp8 staging array (the register-dequant win).
    __shared__ unsigned char smem_Bp[2][2][K_STEP_T / 2][N_TILE_S3 + BP_PAD]; // 5120B
    __shared__ unsigned char smem_Bs[2][2][K_STEP_T / GROUP_SIZE][N_TILE_S3 + BP_PAD]; // 640B
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Per-warp accumulators for BOTH projections:
    // acc[w][m_frag 0..1][n_subtile 0..1][4 fp32]. w: 0=gate, 1=up.
    float acc[2][2][2][4];
    #pragma unroll
    for (int w = 0; w < 2; w++)
        #pragma unroll
        for (int mf = 0; mf < 2; mf++)
            #pragma unroll
            for (int sub = 0; sub < 2; sub++) {
                acc[w][mf][sub][0] = 0.0f; acc[w][mf][sub][1] = 0.0f;
                acc[w][mf][sub][2] = 0.0f; acc[w][mf][sub][3] = 0.0f;
            }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // A loads (shared) — identical to the baseline ISSUE_A_M32.
    #define PR_ISSUE_A(buf, kb) do { \
        unsigned int a_row = threadIdx.x >> 2;        /* 0..31 */ \
        unsigned int a_col = (threadIdx.x & 3) << 3;  /* 0/8/16/24 */ \
        unsigned int gc = (kb) + a_col; \
        unsigned int gr = cta_m + a_row; \
        cp_async_pred_16(&smem_A[(buf)][a_row][a_col], \
            &A[(unsigned long long)gr * K + gc], \
            (gr < M) && (gc + 7 < K)); \
    } while(0)

    // B loads for weight w — identical to the baseline ISSUE_B_M32.
    #define PR_ISSUE_B(w, buf, kb, Bp, Bs) do { \
        if (threadIdx.x < 64) { \
            unsigned int kp = threadIdx.x >> 2;          /* 0..15 */ \
            unsigned int ns = (threadIdx.x & 3) << 4;    /* 0/16/32/48 */ \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(w)][(buf)][kp][ns], \
                &Bp[(unsigned long long)(gke >> 1) * ldb + gns], \
                (gke + 1 <= K) && (gns + 15 < ldb)); \
            if (threadIdx.x < (K_STEP_T / GROUP_SIZE) * (N_TILE_S3 / 16)) { \
                unsigned int sg_row = threadIdx.x >> 2; \
                unsigned int sg_ns  = (threadIdx.x & 3) << 4; \
                unsigned int sg = (kb) / GROUP_SIZE + sg_row; \
                cp_async_pred_16(&smem_Bs[(w)][(buf)][sg_row][sg_ns], \
                    &Bs[(unsigned long long)sg * ldb + cta_n + sg_ns], \
                    (cta_n + sg_ns + 15 < ldb)); \
            } \
        } \
    } while(0)

    #define PR_ISSUE_STAGE(buf, kb) do { \
        PR_ISSUE_A((buf), (kb)); \
        PR_ISSUE_B(0, (buf), (kb), Bg_packed, Bg_scale); \
        PR_ISSUE_B(1, (buf), (kb), Bu_packed, Bu_scale); \
        cp_async_commit(); \
    } while(0)

    // Register dequant of ONE column `nc` of weight w, tile `buf`, into the
    // two uint32 FP8 fragments the mma.sync consumes: v0 = fp8[k=4tid..4tid+3],
    // v1 = fp8[k=16+4tid..16+4tid+3]. Bit-exact with the baseline DEQUANT +
    // smem_B_fp8 read:
    //   baseline: smem_B_fp8[nc][kp*2 (+1)] = cvt.e4m3x2.f32(hi,lo) with
    //             lo=LUT[packed&0xF]*sv, hi=LUT[packed>>4]*sv, sv=(f8_scale)*sc2;
    //             MMA then reads 4 consecutive fp8 bytes as one uint32.
    //   here:     reconstruct those same 4 bytes directly in a register.
    // v0 spans k=4tid..4tid+3 → packed bytes kp=2tid (k lo/hi) and kp=2tid+1;
    // both in the FIRST scale half (kp<8 for tid≤3) → scale group 0.
    // v1 spans k=16+4tid..16+4tid+3 → packed bytes kp=8+2tid and 8+2tid+1;
    // SECOND scale half → scale group 1.
    //
    // Byte order within the uint32 MUST match the smem layout: byte b of
    // smem_B_fp8[nc][base+b] = fp8 for k=base+b. cvt.e4m3x2.f32 %0,%hi,%lo
    // packs lo into the LOW byte and hi into the HIGH byte of the 16-bit
    // result — i.e. the pair (k even, k odd) lands (low, high), matching
    // *(unsigned short*)&smem_B_fp8[nc][kp*2]. Assembling two such pairs
    // little-endian into a uint32 reproduces the 4-byte smem read exactly.
    #define PR_DEQ_COL(w, buf, nc, sc2, v0, v1) do { \
        __nv_fp8_e4m3 f0_, f1_; \
        *(unsigned char*)&f0_ = smem_Bs[(w)][(buf)][0][(nc)]; \
        *(unsigned char*)&f1_ = smem_Bs[(w)][(buf)][1][(nc)]; \
        float sv0_ = (float)f0_ * (sc2); \
        float sv1_ = (float)f1_ * (sc2); \
        unsigned int p0_ = smem_Bp[(w)][(buf)][2*tid    ][(nc)]; \
        unsigned int p1_ = smem_Bp[(w)][(buf)][2*tid + 1][(nc)]; \
        unsigned int p2_ = smem_Bp[(w)][(buf)][8 + 2*tid    ][(nc)]; \
        unsigned int p3_ = smem_Bp[(w)][(buf)][8 + 2*tid + 1][(nc)]; \
        unsigned short h0_, h1_, h2_, h3_; \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h0_) \
            : "f"(smem_LUT[p0_ >> 4] * sv0_), "f"(smem_LUT[p0_ & 0xF] * sv0_)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h1_) \
            : "f"(smem_LUT[p1_ >> 4] * sv0_), "f"(smem_LUT[p1_ & 0xF] * sv0_)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h2_) \
            : "f"(smem_LUT[p2_ >> 4] * sv1_), "f"(smem_LUT[p2_ & 0xF] * sv1_)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h3_) \
            : "f"(smem_LUT[p3_ >> 4] * sv1_), "f"(smem_LUT[p3_ & 0xF] * sv1_)); \
        (v0) = ((unsigned int)h1_ << 16) | (unsigned int)h0_; \
        (v1) = ((unsigned int)h3_ << 16) | (unsigned int)h2_; \
    } while(0)

    // MMA for weight w with register-resident B dequant. Identical MMA
    // instruction + accumulator targets + A fragment path as the baseline
    // COMPUTE_MMA_M32; only the B operand source differs (regs, not smem).
    #define PR_MMA(w, buf, sc2) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        unsigned int fr0 = group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        unsigned int c0_ = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + tid * 4]); \
        unsigned int c1_ = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + tid * 4]); \
        unsigned int c2_ = bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + 16 + tid * 4]); \
        unsigned int c3_ = bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 2; sub++) { \
            unsigned int nt = warp_id * 2 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int v0, v1; \
            PR_DEQ_COL((w), (buf), nc, (sc2), v0, v1); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3], \
                a0, a1, a2, a3, v0, v1, \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3]); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3], \
                c0_, c1_, c2_, c3_, v0, v1, \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3]); \
        } \
    } while(0)

    const unsigned int num_k = K / K_STEP_T;   // K is a multiple of K_STEP_T

    // Prologue: prime the first tile (2-stage → 1 group ahead).
    PR_ISSUE_STAGE(0, 0);

    int cur = 0;
    for (unsigned int k = 0; k < num_k; k++) {
        unsigned int kb = k * K_STEP_T;
        int nxt = 1 - cur;
        // Prefetch next tile BEFORE consuming current — keeps 1 cp.async
        // group in flight so its memory latency overlaps this step's
        // register-dequant + MMA.
        bool has_next = (k + 1) < num_k;
        if (has_next) {
            PR_ISSUE_STAGE(nxt, kb + K_STEP_T);
        }
        // Leave ≤1 group in flight (the just-issued next tile) → block only
        // until the CURRENT tile's loads have landed. On the last step no
        // next tile was issued, so drain fully.
        if (has_next) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();
        // Dequant happens IN-LINE inside PR_MMA (register-resident); no
        // smem_B_fp8 write and no second __syncthreads.
        PR_MMA(0, cur, scale2_g);
        PR_MMA(1, cur, scale2_u);
        // Barrier before the next step overwrites smem[cur]: every thread
        // has finished reading smem_Bp/Bs[cur] in PR_MMA above.
        __syncthreads();
        cur = nxt;
    }

    #undef PR_ISSUE_A
    #undef PR_ISSUE_B
    #undef PR_ISSUE_STAGE
    #undef PR_DEQ_COL
    #undef PR_MMA

    // Output: silu(gate)*up — IDENTICAL FUSE to the 2-stage kernel (BF16
    // round-trip on each accumulator, __expf sigmoid). Byte-for-byte.
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        unsigned int nt = warp_id * 2 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        unsigned int r2 = r0 + 16;
        unsigned int r3 = r0 + 24;
        #define FUSE(gi, ui) ({ \
            float g_ = __bfloat162float(__float2bfloat16(gi)); \
            float u_ = __bfloat162float(__float2bfloat16(ui)); \
            float sig_ = 1.0f / (1.0f + __expf(-g_)); \
            __float2bfloat16(g_ * sig_ * u_); })
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = FUSE(acc[0][0][sub][0], acc[1][0][sub][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = FUSE(acc[0][0][sub][1], acc[1][0][sub][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = FUSE(acc[0][0][sub][2], acc[1][0][sub][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = FUSE(acc[0][0][sub][3], acc[1][0][sub][3]);
        if (r2 < M && c0 < N) C[(unsigned long long)r2 *N+c0] = FUSE(acc[0][1][sub][0], acc[1][1][sub][0]);
        if (r2 < M && c1 < N) C[(unsigned long long)r2 *N+c1] = FUSE(acc[0][1][sub][1], acc[1][1][sub][1]);
        if (r3 < M && c0 < N) C[(unsigned long long)r3 *N+c0] = FUSE(acc[0][1][sub][2], acc[1][1][sub][2]);
        if (r3 < M && c1 < N) C[(unsigned long long)r3 *N+c1] = FUSE(acc[0][1][sub][3], acc[1][1][sub][3]);
        #undef FUSE
    }
}

#undef N_TILE_S3

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64 — K_STEP=64 fork of
// w4a16_gemm_t_m32_n64_gateup_silu_pipe (ATLAS_GATEUP_K64=1).
//
// Doubles the K-step tile from 32 to 64 elements: 80 K-loop iterations
// vs 160, halving sync count and loop overhead. Each step issues 2×
// the cp.async traffic (8 KB A+gate+up vs 4 KB; 8.5 KB vs 4.25 KB with scales),
// so the background load overlaps more of the per-step dequant+double-MMA work.
//
// Mathematics: two consecutive m16n8k32 FP8 MMAs per accumulator per
// K-step (K[0..31] then K[32..63]), sharing the same FP32 accumulator.
// Byte-exact with the _pipe kernel: BF16 round-trip SiLU*mul at output.
//
// Loading (2-stage pipeline, wait_group<1>):
//   A: each thread issues 2 × cp.async 16B — cols [a_col] and [a_col+32]
//      for its row → 256 loads of 16B per stage = 4096 B.
//   B gate/up: all 128 threads (kp=0..31), 1 × 16B load per weight →
//      128 loads = 2048 B per weight; threads 0..15 also load 16B scale.
//
// SMEM (no smem_B_fp8 staging, a_stride=72 for 16B-aligned rows):
//   A: 2×32×72×2 = 9216 B  |  Bp×2: 2×2×32×80 = 10240 B
//   Bs×2: 2×2×4×80 = 1280 B  |  LUT: 64 B   ≈ 20.8 KB total
// ═══════════════════════════════════════════════════════════════════
#define N_TILE_K64  64
#define K_STEP_K64  64
#define A_PAD_K64    8   // (64+8)*2=144, 144%16=0 → 16B-aligned rows
#define BP_PAD_K64  16
extern "C" __global__ void w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb
) {
    constexpr unsigned int M_TILE_K  = 32;
    constexpr unsigned int K_GROUPS  = K_STEP_K64 / GROUP_SIZE;  // 4
    const unsigned int cta_n    = blockIdx.x * N_TILE_K64;
    const unsigned int cta_m    = blockIdx.y * M_TILE_K;
    const unsigned int warp_id  = threadIdx.x / 32;
    const unsigned int lane_id  = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;   // 0..7
    const unsigned int tid      = lane_id & 3;    // 0..3

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_K][K_STEP_K64 + A_PAD_K64];
    __shared__ unsigned char smem_Bp[2][2][K_STEP_K64 / 2][N_TILE_K64 + BP_PAD_K64];
    __shared__ unsigned char smem_Bs[2][2][K_GROUPS][N_TILE_K64 + BP_PAD_K64];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[2][2][2][4];
    #pragma unroll
    for (int w = 0; w < 2; w++)
        #pragma unroll
        for (int mf = 0; mf < 2; mf++)
            #pragma unroll
            for (int sub = 0; sub < 2; sub++) {
                acc[w][mf][sub][0] = 0.0f; acc[w][mf][sub][1] = 0.0f;
                acc[w][mf][sub][2] = 0.0f; acc[w][mf][sub][3] = 0.0f;
            }

    const unsigned int a_stride = K_STEP_K64 + A_PAD_K64;  // 72 uint16

    // A: each thread issues 2 cp.async — K[0..31] and K[32..63] of its row.
    #define K64_ISSUE_A(buf, kb) do { \
        unsigned int _ar = threadIdx.x >> 2; \
        unsigned int _ac = (threadIdx.x & 3) << 3; \
        unsigned int _gr = cta_m + _ar; \
        cp_async_pred_16(&smem_A[(buf)][_ar][_ac], \
            &A[(unsigned long long)_gr * K + (kb) + _ac], \
            (_gr < M) && ((kb) + _ac + 7 < K)); \
        unsigned int _ac2 = _ac + 32; \
        cp_async_pred_16(&smem_A[(buf)][_ar][_ac2], \
            &A[(unsigned long long)_gr * K + (kb) + _ac2], \
            (_gr < M) && ((kb) + _ac2 + 7 < K)); \
    } while(0)

    // B: all 128 threads, kp=0..31.  Threads 0..15 also load scale.
    #define K64_ISSUE_B(w, buf, kb, Bp, Bs) do { \
        unsigned int _kp  = threadIdx.x >> 2; \
        unsigned int _ns  = (threadIdx.x & 3) << 4; \
        unsigned int _gke = (kb) + (_kp << 1); \
        unsigned int _gns = cta_n + _ns; \
        cp_async_pred_16(&smem_Bp[(w)][(buf)][_kp][_ns], \
            &Bp[(unsigned long long)(_gke >> 1) * ldb + _gns], \
            (_gke + 1 <= K) && (_gns + 15 < ldb)); \
        if (threadIdx.x < K_GROUPS * (N_TILE_K64 / 16)) { \
            unsigned int _sgr = threadIdx.x >> 2; \
            unsigned int _sgn = (threadIdx.x & 3) << 4; \
            unsigned int _sg  = (kb) / GROUP_SIZE + _sgr; \
            cp_async_pred_16(&smem_Bs[(w)][(buf)][_sgr][_sgn], \
                &Bs[(unsigned long long)_sg * ldb + cta_n + _sgn], \
                (cta_n + _sgn + 15 < ldb)); \
        } \
    } while(0)

    #define K64_ISSUE_STAGE(buf, kb) do { \
        K64_ISSUE_A((buf), (kb)); \
        K64_ISSUE_B(0, (buf), (kb), Bg_packed, Bg_scale); \
        K64_ISSUE_B(1, (buf), (kb), Bu_packed, Bu_scale); \
        cp_async_commit(); \
    } while(0)

    // Register dequant of 64 K-elements of column nc for weight w tile buf.
    // Output fragments: v0_0,v1_0 = K[0..31]; v0_1,v1_1 = K[32..63].
    // Each uint32 packs 4 FP8 bytes in the layout consumed by m16n8k32:
    //   v0 = {bytes for k=4tid,4tid+1,4tid+2,4tid+3} (lower K-half of step)
    //   v1 = {bytes for k=16+4tid..16+4tid+3}         (upper K-half)
    // Bit-exact with PR_DEQ_COL in _pipe kernel per K-half.
    #define K64_DEQ_COL(w, buf, nc, sc2, v0_0, v1_0, v0_1, v1_1) do { \
        __nv_fp8_e4m3 _sf0, _sf1, _sf2, _sf3; \
        *(unsigned char*)&_sf0 = smem_Bs[(w)][(buf)][0][(nc)]; \
        *(unsigned char*)&_sf1 = smem_Bs[(w)][(buf)][1][(nc)]; \
        *(unsigned char*)&_sf2 = smem_Bs[(w)][(buf)][2][(nc)]; \
        *(unsigned char*)&_sf3 = smem_Bs[(w)][(buf)][3][(nc)]; \
        float _sv0 = (float)_sf0 * (sc2); \
        float _sv1 = (float)_sf1 * (sc2); \
        float _sv2 = (float)_sf2 * (sc2); \
        float _sv3 = (float)_sf3 * (sc2); \
        unsigned int _p00 = smem_Bp[(w)][(buf)][   2*tid  ][(nc)]; \
        unsigned int _p01 = smem_Bp[(w)][(buf)][   2*tid+1][(nc)]; \
        unsigned int _p10 = smem_Bp[(w)][(buf)][8+ 2*tid  ][(nc)]; \
        unsigned int _p11 = smem_Bp[(w)][(buf)][8+ 2*tid+1][(nc)]; \
        unsigned short _h00,_h01,_h10,_h11; \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_h00):"f"(smem_LUT[_p00>>4]*_sv0),"f"(smem_LUT[_p00&0xF]*_sv0)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_h01):"f"(smem_LUT[_p01>>4]*_sv0),"f"(smem_LUT[_p01&0xF]*_sv0)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_h10):"f"(smem_LUT[_p10>>4]*_sv1),"f"(smem_LUT[_p10&0xF]*_sv1)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_h11):"f"(smem_LUT[_p11>>4]*_sv1),"f"(smem_LUT[_p11&0xF]*_sv1)); \
        (v0_0) = ((unsigned int)_h01 << 16) | (unsigned int)_h00; \
        (v1_0) = ((unsigned int)_h11 << 16) | (unsigned int)_h10; \
        unsigned int _q00 = smem_Bp[(w)][(buf)][16+ 2*tid  ][(nc)]; \
        unsigned int _q01 = smem_Bp[(w)][(buf)][16+ 2*tid+1][(nc)]; \
        unsigned int _q10 = smem_Bp[(w)][(buf)][24+ 2*tid  ][(nc)]; \
        unsigned int _q11 = smem_Bp[(w)][(buf)][24+ 2*tid+1][(nc)]; \
        unsigned short _g00,_g01,_g10,_g11; \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_g00):"f"(smem_LUT[_q00>>4]*_sv2),"f"(smem_LUT[_q00&0xF]*_sv2)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_g01):"f"(smem_LUT[_q01>>4]*_sv2),"f"(smem_LUT[_q01&0xF]*_sv2)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_g10):"f"(smem_LUT[_q10>>4]*_sv3),"f"(smem_LUT[_q10&0xF]*_sv3)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_g11):"f"(smem_LUT[_q11>>4]*_sv3),"f"(smem_LUT[_q11&0xF]*_sv3)); \
        (v0_1) = ((unsigned int)_g01 << 16) | (unsigned int)_g00; \
        (v1_1) = ((unsigned int)_g11 << 16) | (unsigned int)_g10; \
    } while(0)

    // Two consecutive m16n8k32 MMAs per N-subtile per M-frag (K[0..31] + K[32..63]).
    #define K64_MMA(w, buf, sc2) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        unsigned int _fr0 = group_id; \
        unsigned int _fr1 = _fr0 + 8; \
        unsigned int _a0 = bf16x4_to_e4m3x4(&sA[_fr0      * a_stride +        tid * 4]); \
        unsigned int _a1 = bf16x4_to_e4m3x4(&sA[_fr1      * a_stride +        tid * 4]); \
        unsigned int _a2 = bf16x4_to_e4m3x4(&sA[_fr0      * a_stride + 16 +   tid * 4]); \
        unsigned int _a3 = bf16x4_to_e4m3x4(&sA[_fr1      * a_stride + 16 +   tid * 4]); \
        unsigned int _b0 = bf16x4_to_e4m3x4(&sA[(_fr0+16) * a_stride +        tid * 4]); \
        unsigned int _b1 = bf16x4_to_e4m3x4(&sA[(_fr1+16) * a_stride +        tid * 4]); \
        unsigned int _b2 = bf16x4_to_e4m3x4(&sA[(_fr0+16) * a_stride + 16 +   tid * 4]); \
        unsigned int _b3 = bf16x4_to_e4m3x4(&sA[(_fr1+16) * a_stride + 16 +   tid * 4]); \
        unsigned int _e0 = bf16x4_to_e4m3x4(&sA[_fr0      * a_stride + 32 +   tid * 4]); \
        unsigned int _e1 = bf16x4_to_e4m3x4(&sA[_fr1      * a_stride + 32 +   tid * 4]); \
        unsigned int _e2 = bf16x4_to_e4m3x4(&sA[_fr0      * a_stride + 48 +   tid * 4]); \
        unsigned int _e3 = bf16x4_to_e4m3x4(&sA[_fr1      * a_stride + 48 +   tid * 4]); \
        unsigned int _f0 = bf16x4_to_e4m3x4(&sA[(_fr0+16) * a_stride + 32 +   tid * 4]); \
        unsigned int _f1 = bf16x4_to_e4m3x4(&sA[(_fr1+16) * a_stride + 32 +   tid * 4]); \
        unsigned int _f2 = bf16x4_to_e4m3x4(&sA[(_fr0+16) * a_stride + 48 +   tid * 4]); \
        unsigned int _f3 = bf16x4_to_e4m3x4(&sA[(_fr1+16) * a_stride + 48 +   tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 2; sub++) { \
            unsigned int nt = warp_id * 2 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int _v00,_v10,_v01,_v11; \
            K64_DEQ_COL((w),(buf),nc,(sc2),_v00,_v10,_v01,_v11); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3], \
                _a0, _a1, _a2, _a3, _v00, _v10, \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3]); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3], \
                _b0, _b1, _b2, _b3, _v00, _v10, \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3]); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3], \
                _e0, _e1, _e2, _e3, _v01, _v11, \
                acc[(w)][0][sub][0], acc[(w)][0][sub][1], acc[(w)][0][sub][2], acc[(w)][0][sub][3]); \
            ATLAS_MMA_FP8_UNIT_M16N8K32( \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3], \
                _f0, _f1, _f2, _f3, _v01, _v11, \
                acc[(w)][1][sub][0], acc[(w)][1][sub][1], acc[(w)][1][sub][2], acc[(w)][1][sub][3]); \
        } \
    } while(0)

    const unsigned int num_k = K / K_STEP_K64;  // K must be a multiple of 64

    K64_ISSUE_STAGE(0, 0);
    int cur = 0;
    for (unsigned int k = 0; k < num_k; k++) {
        int nxt = 1 - cur;
        bool has_next = (k + 1) < num_k;
        if (has_next) K64_ISSUE_STAGE(nxt, (k + 1) * K_STEP_K64);
        if (has_next) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();
        K64_MMA(0, cur, scale2_g);
        K64_MMA(1, cur, scale2_u);
        __syncthreads();
        cur = nxt;
    }

    #undef K64_ISSUE_A
    #undef K64_ISSUE_B
    #undef K64_ISSUE_STAGE
    #undef K64_DEQ_COL
    #undef K64_MMA

    // Output: identical FUSE to _pipe (BF16 round-trip SiLU*mul).
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        unsigned int nt = warp_id * 2 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        unsigned int r2 = r0 + 16;
        unsigned int r3 = r0 + 24;
        #define FUSE_K64(gi, ui) ({ \
            float _g = __bfloat162float(__float2bfloat16(gi)); \
            float _u = __bfloat162float(__float2bfloat16(ui)); \
            float _s = 1.0f / (1.0f + __expf(-_g)); \
            __float2bfloat16(_g * _s * _u); })
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = FUSE_K64(acc[0][0][sub][0], acc[1][0][sub][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = FUSE_K64(acc[0][0][sub][1], acc[1][0][sub][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = FUSE_K64(acc[0][0][sub][2], acc[1][0][sub][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = FUSE_K64(acc[0][0][sub][3], acc[1][0][sub][3]);
        if (r2 < M && c0 < N) C[(unsigned long long)r2 *N+c0] = FUSE_K64(acc[0][1][sub][0], acc[1][1][sub][0]);
        if (r2 < M && c1 < N) C[(unsigned long long)r2 *N+c1] = FUSE_K64(acc[0][1][sub][1], acc[1][1][sub][1]);
        if (r3 < M && c0 < N) C[(unsigned long long)r3 *N+c0] = FUSE_K64(acc[0][1][sub][2], acc[1][1][sub][2]);
        if (r3 < M && c1 < N) C[(unsigned long long)r3 *N+c1] = FUSE_K64(acc[0][1][sub][3], acc[1][1][sub][3]);
        #undef FUSE_K64
    }
}

#undef N_TILE_K64
#undef K_STEP_K64
#undef A_PAD_K64
#undef BP_PAD_K64

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64 — N_TILE=32 wide-SM fork of
// w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64 (`ATLAS_GATEUP_N32`).
//
// On 188-SM GPUs (SM120 RTX PRO 6000 / SM103), Qwen3.8-27B's FFN gate+up
// (N=17408) fields only 272 CTAs at N_TILE=64 (1.45 CTAs/SM -> the tail wave leaves
// 104 of 188 SMs idle with zero multi-CTA latency hiding). Halving N_TILE to
// 32 doubles the grid to 544 CTAs (2.89 CTAs/SM, 89.4% wave efficiency),
// halves accumulator registers (16 vs 32), shrinks SMEM to 16.2 KB, and keeps
// the exact same K_STEP=64 register-dequant + cp.async.wait_group<1> loop.
//
// Byte-exact (0-ULP) with `w4a16_gemm_t_m32_n64_gateup_silu` and `_pipe_k64`:
// identical K[0..31]+K[32..63] m16n8k32 accumulation order and BF16 round-trip
// SiLU*mul epilogue. Zero extra VRAM, zero split-K reduce launch.
// Requires K % 64 == 0 and N % 32 == 0. Grid: (ceil(N/32), ceil(M/32), 1).
// ═══════════════════════════════════════════════════════════════════
#define N_TILE_N32  32
#define K_STEP_N32  64
#define A_PAD_N32    8
#define BP_PAD_N32  16   // 32 + 16 = 48 B rows, 16B-aligned
extern "C" __global__ void w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb
) {
    constexpr unsigned int M_TILE_K  = 32;
    constexpr unsigned int K_GROUPS  = K_STEP_N32 / GROUP_SIZE;  // 4
    const unsigned int cta_n    = blockIdx.x * N_TILE_N32;
    const unsigned int cta_m    = blockIdx.y * M_TILE_K;
    const unsigned int warp_id  = threadIdx.x / 32;
    const unsigned int lane_id  = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;   // 0..7
    const unsigned int tid      = lane_id & 3;    // 0..3

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_K][K_STEP_N32 + A_PAD_N32];
    __shared__ unsigned char smem_Bp[2][2][K_STEP_N32 / 2][N_TILE_N32 + BP_PAD_N32];
    __shared__ unsigned char smem_Bs[2][2][K_GROUPS][N_TILE_N32 + BP_PAD_N32];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // 1 N-subtile per warp (4 warps x 8 cols = 32 cols): acc[w=0..1][mf=0..1][4].
    float acc[2][2][4];
    #pragma unroll
    for (int w = 0; w < 2; w++)
        #pragma unroll
        for (int mf = 0; mf < 2; mf++) {
            acc[w][mf][0] = 0.0f; acc[w][mf][1] = 0.0f;
            acc[w][mf][2] = 0.0f; acc[w][mf][3] = 0.0f;
        }

    const unsigned int a_stride = K_STEP_N32 + A_PAD_N32;  // 72 uint16

    #define N32_ISSUE_A(buf, kb) do { \
        unsigned int _ar = threadIdx.x >> 2; \
        unsigned int _ac = (threadIdx.x & 3) << 3; \
        unsigned int _gr = cta_m + _ar; \
        cp_async_pred_16(&smem_A[(buf)][_ar][_ac], \
            &A[(unsigned long long)_gr * K + (kb) + _ac], \
            (_gr < M) && ((kb) + _ac + 7 < K)); \
        unsigned int _ac2 = _ac + 32; \
        cp_async_pred_16(&smem_A[(buf)][_ar][_ac2], \
            &A[(unsigned long long)_gr * K + (kb) + _ac2], \
            (_gr < M) && ((kb) + _ac2 + 7 < K)); \
    } while(0)

    // 32 K-pair rows x 2 16B chunks = 64 loads per weight:
    // threads 0..63 load Bg (gate), threads 64..127 load Bu (up).
    #define N32_ISSUE_B(buf, kb) do { \
        const unsigned int _w = threadIdx.x >> 6; \
        const unsigned int _t = threadIdx.x & 63u; \
        const unsigned int _kp = _t >> 1; \
        const unsigned int _ns = (_t & 1u) << 4; \
        const unsigned int _gke = (kb) + (_kp << 1); \
        const unsigned int _gns = cta_n + _ns; \
        const unsigned char* _Bp = (_w == 0u) ? Bg_packed : Bu_packed; \
        const unsigned char* _Bs = (_w == 0u) ? Bg_scale  : Bu_scale; \
        cp_async_pred_16(&smem_Bp[_w][(buf)][_kp][_ns], \
            &_Bp[(unsigned long long)(_gke >> 1) * ldb + _gns], \
            (_gke + 1 <= K) && (_gns + 15 < ldb)); \
        if (_t < K_GROUPS * (N_TILE_N32 / 16)) { \
            const unsigned int _sgr = _t >> 1; \
            const unsigned int _sgn = (_t & 1u) << 4; \
            const unsigned int _sg  = (kb) / GROUP_SIZE + _sgr; \
            cp_async_pred_16(&smem_Bs[_w][(buf)][_sgr][_sgn], \
                &_Bs[(unsigned long long)_sg * ldb + cta_n + _sgn], \
                (cta_n + _sgn + 15 < ldb)); \
        } \
    } while(0)

    #define N32_ISSUE_STAGE(buf, kb) do { \
        N32_ISSUE_A((buf), (kb)); \
        N32_ISSUE_B((buf), (kb)); \
        cp_async_commit(); \
    } while(0)

    #define N32_DEQ_COL(w, buf, nc, sc2, v0_0, v1_0, v0_1, v1_1) do { \
        __nv_fp8_e4m3 _sf0, _sf1, _sf2, _sf3; \
        *(unsigned char*)&_sf0 = smem_Bs[(w)][(buf)][0][(nc)]; \
        *(unsigned char*)&_sf1 = smem_Bs[(w)][(buf)][1][(nc)]; \
        *(unsigned char*)&_sf2 = smem_Bs[(w)][(buf)][2][(nc)]; \
        *(unsigned char*)&_sf3 = smem_Bs[(w)][(buf)][3][(nc)]; \
        float _sv0 = (float)_sf0 * (sc2); \
        float _sv1 = (float)_sf1 * (sc2); \
        float _sv2 = (float)_sf2 * (sc2); \
        float _sv3 = (float)_sf3 * (sc2); \
        unsigned int _p00 = smem_Bp[(w)][(buf)][   2*tid  ][(nc)]; \
        unsigned int _p01 = smem_Bp[(w)][(buf)][   2*tid+1][(nc)]; \
        unsigned int _p10 = smem_Bp[(w)][(buf)][8+ 2*tid  ][(nc)]; \
        unsigned int _p11 = smem_Bp[(w)][(buf)][8+ 2*tid+1][(nc)]; \
        unsigned short _h00,_h01,_h10,_h11; \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_h00):"f"(smem_LUT[_p00>>4]*_sv0),"f"(smem_LUT[_p00&0xF]*_sv0)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_h01):"f"(smem_LUT[_p01>>4]*_sv0),"f"(smem_LUT[_p01&0xF]*_sv0)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_h10):"f"(smem_LUT[_p10>>4]*_sv1),"f"(smem_LUT[_p10&0xF]*_sv1)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_h11):"f"(smem_LUT[_p11>>4]*_sv1),"f"(smem_LUT[_p11&0xF]*_sv1)); \
        (v0_0) = ((unsigned int)_h01 << 16) | (unsigned int)_h00; \
        (v1_0) = ((unsigned int)_h11 << 16) | (unsigned int)_h10; \
        unsigned int _q00 = smem_Bp[(w)][(buf)][16+ 2*tid  ][(nc)]; \
        unsigned int _q01 = smem_Bp[(w)][(buf)][16+ 2*tid+1][(nc)]; \
        unsigned int _q10 = smem_Bp[(w)][(buf)][24+ 2*tid  ][(nc)]; \
        unsigned int _q11 = smem_Bp[(w)][(buf)][24+ 2*tid+1][(nc)]; \
        unsigned short _g00,_g01,_g10,_g11; \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_g00):"f"(smem_LUT[_q00>>4]*_sv2),"f"(smem_LUT[_q00&0xF]*_sv2)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_g01):"f"(smem_LUT[_q01>>4]*_sv2),"f"(smem_LUT[_q01&0xF]*_sv2)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_g10):"f"(smem_LUT[_q10>>4]*_sv3),"f"(smem_LUT[_q10&0xF]*_sv3)); \
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(_g11):"f"(smem_LUT[_q11>>4]*_sv3),"f"(smem_LUT[_q11&0xF]*_sv3)); \
        (v0_1) = ((unsigned int)_g01 << 16) | (unsigned int)_g00; \
        (v1_1) = ((unsigned int)_g11 << 16) | (unsigned int)_g10; \
    } while(0)

    #define N32_MMA(w, buf, sc2) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        unsigned int _fr0 = group_id; \
        unsigned int _fr1 = _fr0 + 8; \
        unsigned int _a0 = bf16x4_to_e4m3x4(&sA[_fr0      * a_stride +        tid * 4]); \
        unsigned int _a1 = bf16x4_to_e4m3x4(&sA[_fr1      * a_stride +        tid * 4]); \
        unsigned int _a2 = bf16x4_to_e4m3x4(&sA[_fr0      * a_stride + 16 +   tid * 4]); \
        unsigned int _a3 = bf16x4_to_e4m3x4(&sA[_fr1      * a_stride + 16 +   tid * 4]); \
        unsigned int _b0 = bf16x4_to_e4m3x4(&sA[(_fr0+16) * a_stride +        tid * 4]); \
        unsigned int _b1 = bf16x4_to_e4m3x4(&sA[(_fr1+16) * a_stride +        tid * 4]); \
        unsigned int _b2 = bf16x4_to_e4m3x4(&sA[(_fr0+16) * a_stride + 16 +   tid * 4]); \
        unsigned int _b3 = bf16x4_to_e4m3x4(&sA[(_fr1+16) * a_stride + 16 +   tid * 4]); \
        unsigned int _e0 = bf16x4_to_e4m3x4(&sA[_fr0      * a_stride + 32 +   tid * 4]); \
        unsigned int _e1 = bf16x4_to_e4m3x4(&sA[_fr1      * a_stride + 32 +   tid * 4]); \
        unsigned int _e2 = bf16x4_to_e4m3x4(&sA[_fr0      * a_stride + 48 +   tid * 4]); \
        unsigned int _e3 = bf16x4_to_e4m3x4(&sA[_fr1      * a_stride + 48 +   tid * 4]); \
        unsigned int _f0 = bf16x4_to_e4m3x4(&sA[(_fr0+16) * a_stride + 32 +   tid * 4]); \
        unsigned int _f1 = bf16x4_to_e4m3x4(&sA[(_fr1+16) * a_stride + 32 +   tid * 4]); \
        unsigned int _f2 = bf16x4_to_e4m3x4(&sA[(_fr0+16) * a_stride + 48 +   tid * 4]); \
        unsigned int _f3 = bf16x4_to_e4m3x4(&sA[(_fr1+16) * a_stride + 48 +   tid * 4]); \
        unsigned int nc = warp_id * 8 + group_id; \
        unsigned int _v00,_v10,_v01,_v11; \
        N32_DEQ_COL((w),(buf),nc,(sc2),_v00,_v10,_v01,_v11); \
        ATLAS_MMA_FP8_UNIT_M16N8K32( \
            acc[(w)][0][0], acc[(w)][0][1], acc[(w)][0][2], acc[(w)][0][3], \
            _a0, _a1, _a2, _a3, _v00, _v10, \
            acc[(w)][0][0], acc[(w)][0][1], acc[(w)][0][2], acc[(w)][0][3]); \
        ATLAS_MMA_FP8_UNIT_M16N8K32( \
            acc[(w)][1][0], acc[(w)][1][1], acc[(w)][1][2], acc[(w)][1][3], \
            _b0, _b1, _b2, _b3, _v00, _v10, \
            acc[(w)][1][0], acc[(w)][1][1], acc[(w)][1][2], acc[(w)][1][3]); \
        ATLAS_MMA_FP8_UNIT_M16N8K32( \
            acc[(w)][0][0], acc[(w)][0][1], acc[(w)][0][2], acc[(w)][0][3], \
            _e0, _e1, _e2, _e3, _v01, _v11, \
            acc[(w)][0][0], acc[(w)][0][1], acc[(w)][0][2], acc[(w)][0][3]); \
        ATLAS_MMA_FP8_UNIT_M16N8K32( \
            acc[(w)][1][0], acc[(w)][1][1], acc[(w)][1][2], acc[(w)][1][3], \
            _f0, _f1, _f2, _f3, _v01, _v11, \
            acc[(w)][1][0], acc[(w)][1][1], acc[(w)][1][2], acc[(w)][1][3]); \
    } while(0)

    const unsigned int num_k = K / K_STEP_N32;

    N32_ISSUE_STAGE(0, 0);
    int cur = 0;
    for (unsigned int k = 0; k < num_k; k++) {
        int nxt = 1 - cur;
        bool has_next = (k + 1) < num_k;
        if (has_next) N32_ISSUE_STAGE(nxt, (k + 1) * K_STEP_N32);
        if (has_next) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();
        N32_MMA(0, cur, scale2_g);
        N32_MMA(1, cur, scale2_u);
        __syncthreads();
        cur = nxt;
    }

    #undef N32_ISSUE_A
    #undef N32_ISSUE_B
    #undef N32_ISSUE_STAGE
    #undef N32_DEQ_COL
    #undef N32_MMA

    {
        unsigned int c0 = cta_n + warp_id * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        unsigned int r2 = r0 + 16;
        unsigned int r3 = r0 + 24;
        #define FUSE_N32(gi, ui) ({ \
            float _g = __bfloat162float(__float2bfloat16(gi)); \
            float _u = __bfloat162float(__float2bfloat16(ui)); \
            float _s = 1.0f / (1.0f + __expf(-_g)); \
            __float2bfloat16(_g * _s * _u); })
        if (r0 < M && c0 < N) C[(unsigned long long)r0 *N+c0] = FUSE_N32(acc[0][0][0], acc[1][0][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 *N+c1] = FUSE_N32(acc[0][0][1], acc[1][0][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 *N+c0] = FUSE_N32(acc[0][0][2], acc[1][0][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 *N+c1] = FUSE_N32(acc[0][0][3], acc[1][0][3]);
        if (r2 < M && c0 < N) C[(unsigned long long)r2 *N+c0] = FUSE_N32(acc[0][1][0], acc[1][1][0]);
        if (r2 < M && c1 < N) C[(unsigned long long)r2 *N+c1] = FUSE_N32(acc[0][1][1], acc[1][1][1]);
        if (r3 < M && c0 < N) C[(unsigned long long)r3 *N+c0] = FUSE_N32(acc[0][1][2], acc[1][1][2]);
        if (r3 < M && c1 < N) C[(unsigned long long)r3 *N+c1] = FUSE_N32(acc[0][1][3], acc[1][1][3]);
        #undef FUSE_N32
    }
}

#undef N_TILE_N32
#undef K_STEP_N32
#undef A_PAD_N32
#undef BP_PAD_N32

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_pipe_m128n128 — byte-exact large-M shadow of w4a16_gemm.
//
// Same numerics contract as w4a16_gemm_pipe: original-layout NVFP4
// weights, per-element dequant ((E2M1_LUT[nib] * (float)fp8) * scale2)
// rounded to BF16, BF16 x BF16 m16n8k16 MMA with FP32 accumulate, K
// consumed in ascending 16-wide steps for every output element. Only the
// tiling changes: 128x128 CTA tile, 8 warps (4 along M x 2 along N, each
// 32x64), ldmatrix fragment loads, cp.async double-buffered A/B_packed,
// and a [n][k] BF16 dequant layout so the col-major B fragment is one
// ldmatrix. Each weight byte is dequantized M/128 times instead of M/64,
// and each A row is re-read N/128 times instead of N/64.
//
// Requirements (caller checks): K % 32 == 0, any M, any N.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (256, 1, 1)
// SMEM: A 2x128x40x2 = 20480 B, B 2x128x40x2 = 20480 B, LUT 64 B -> ~41.0 KB static.
// (B_packed and B_scale are register-loaded via __ldg, unified with qwen3.8-flash-next.)
// ═══════════════════════════════════════════════════════════════════

#define MM_M 128
#define MM_N 128
#define MM_K 32
#define MM_PAD 8           // (32+8)*2 = 80 B rows, 16-byte aligned
#define MM_STRIDE (MM_K + MM_PAD)
#define MM_THREADS 256

__device__ __forceinline__ void ldmatrix_x4(unsigned int& r0, unsigned int& r1,
                                            unsigned int& r2, unsigned int& r3,
                                            const void* smem_ptr) {
    unsigned int addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

extern "C" __global__ void __launch_bounds__(MM_THREADS, 2)
w4a16_gemm_pipe_m128n128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,   // [N, K/2]
    const unsigned char* __restrict__ B_scale,    // [N, K/16]
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * MM_M;
    const unsigned int cta_n = blockIdx.x * MM_N;
    if (cta_m >= M) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int warp_m = (warp_id & 3) * 32;   // 4 warps along M
    const unsigned int warp_n = (warp_id >> 2) * 64;  // 2 warps along N
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3;

    __shared__ __align__(16) __nv_bfloat16 smem_A[2][MM_M][MM_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[2][MM_N][MM_STRIDE];  // [n][k]
    __shared__ float smem_LUT[16];

    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; acc[i][j][2] = 0.f; acc[i][j][3] = 0.f;
        }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int deq_n = tid >> 1;
    const unsigned int deq_kh = tid & 1;
    const unsigned int deq_gn = cta_n + deq_n;

    if (tid < 16) smem_LUT[tid] = E2M1_LUT[tid];
    __syncthreads();

    #define MM_ISSUE_A(buf, kk) do {                                                  \
        _Pragma("unroll")                                                             \
        for (int p = 0; p < 2; p++) {                                                 \
            unsigned int c = tid + p * MM_THREADS;                                    \
            unsigned int row = c >> 2;                                                \
            unsigned int col8 = (c & 3) << 3;                                         \
            unsigned int gr = cta_m + row;                                            \
            cp_async_pred_16(&smem_A[(buf)][row][col8],                               \
                &A[(unsigned long long)gr * K + (kk) + col8], (gr < M) && ((kk) < K)); \
        }                                                                             \
        cp_async_commit();                                                            \
    } while (0)

    #define MM_LOAD_B_REG(kk, bp_out, bs_out) do {                                    \
        if ((deq_gn < N) && ((kk) < K)) {                                             \
            (bp_out) = __ldg(reinterpret_cast<const uint2*>(                          \
                &B_packed[(unsigned long long)deq_gn * half_K + ((kk) >> 1) + deq_kh * 8])); \
            (bs_out) = __ldg(&B_scale[(unsigned long long)deq_gn * num_groups + ((kk) / GROUP_SIZE) + deq_kh]); \
        } else {                                                                      \
            (bp_out) = make_uint2(0u, 0u);                                            \
            (bs_out) = 0u;                                                            \
        }                                                                             \
    } while (0)

    #define MM_DEQUANT_REG(buf, bp_in, bs_in) do {                                    \
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = (bs_in);                           \
        float fs = (float)fp8;                                                        \
        const unsigned char pb[8] = {                                                 \
            (unsigned char)((bp_in).x & 0xFFu),                                       \
            (unsigned char)(((bp_in).x >> 8) & 0xFFu),                                \
            (unsigned char)(((bp_in).x >> 16) & 0xFFu),                               \
            (unsigned char)((bp_in).x >> 24),                                         \
            (unsigned char)((bp_in).y & 0xFFu),                                       \
            (unsigned char)(((bp_in).y >> 8) & 0xFFu),                                \
            (unsigned char)(((bp_in).y >> 16) & 0xFFu),                               \
            (unsigned char)((bp_in).y >> 24)                                          \
        };                                                                            \
        __nv_bfloat16 tmp[16];                                                        \
        _Pragma("unroll")                                                             \
        for (int i = 0; i < 8; i++) {                                                 \
            unsigned char packed = pb[i];                                             \
            tmp[2 * i]     = __float2bfloat16(smem_LUT[packed & 0xF] * fs * scale2);  \
            tmp[2 * i + 1] = __float2bfloat16(smem_LUT[packed >> 4] * fs * scale2);   \
        }                                                                             \
        uint4* dst4 = reinterpret_cast<uint4*>(&smem_B[(buf)][deq_n][deq_kh * 16]);   \
        const uint4* src4 = reinterpret_cast<const uint4*>(tmp);                      \
        dst4[0] = src4[0];                                                            \
        dst4[1] = src4[1];                                                            \
    } while (0)

    MM_ISSUE_A(0, 0);
    {
        uint2 bp0;
        unsigned char bs0;
        MM_LOAD_B_REG(0u, bp0, bs0);
        MM_DEQUANT_REG(0, bp0, bs0);
    }

    const unsigned int a_row = lane & 15;
    const unsigned int a_koff = (lane >> 4) * 8;
    const unsigned int b_row = (lane & 7) + ((lane >> 4) * 8);
    const unsigned int b_koff = ((lane >> 3) & 1) * 8;

    unsigned int cur = 0, nxt = 1;
    for (unsigned int k_base = 0; k_base < K; k_base += MM_K) {
        cp_async_wait_group<0>();
        __syncthreads();

        const unsigned int k_next = k_base + MM_K;
        uint2 bp_nxt = make_uint2(0u, 0u);
        unsigned char bs_nxt = 0u;
        if (k_next < K) {
            MM_ISSUE_A(nxt, k_next);
            MM_LOAD_B_REG(k_next, bp_nxt, bs_nxt);
        }

        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            const unsigned int k0 = kh * 16;
            unsigned int a[2][4];
            #pragma unroll
            for (int mf = 0; mf < 2; mf++) {
                ldmatrix_x4(a[mf][0], a[mf][1], a[mf][2], a[mf][3],
                    &smem_A[cur][warp_m + mf * 16 + a_row][k0 + a_koff]);
            }
            #pragma unroll
            for (int np = 0; np < 4; np++) {
                unsigned int b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3,
                    &smem_B[cur][warp_n + np * 16 + b_row][k0 + b_koff]);
                #pragma unroll
                for (int mf = 0; mf < 2; mf++) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[mf][2*np][0]),"=f"(acc[mf][2*np][1]),"=f"(acc[mf][2*np][2]),"=f"(acc[mf][2*np][3])
                        :"r"(a[mf][0]),"r"(a[mf][1]),"r"(a[mf][2]),"r"(a[mf][3]),"r"(b0),"r"(b1),
                         "f"(acc[mf][2*np][0]),"f"(acc[mf][2*np][1]),"f"(acc[mf][2*np][2]),"f"(acc[mf][2*np][3]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[mf][2*np+1][0]),"=f"(acc[mf][2*np+1][1]),"=f"(acc[mf][2*np+1][2]),"=f"(acc[mf][2*np+1][3])
                        :"r"(a[mf][0]),"r"(a[mf][1]),"r"(a[mf][2]),"r"(a[mf][3]),"r"(b2),"r"(b3),
                         "f"(acc[mf][2*np+1][0]),"f"(acc[mf][2*np+1][1]),"f"(acc[mf][2*np+1][2]),"f"(acc[mf][2*np+1][3]));
                }
            }
        }

        if (k_next < K) {
            MM_DEQUANT_REG(nxt, bp_nxt, bs_nxt);
        }
        unsigned int t = cur; cur = nxt; nxt = t;
    }

    if (cta_n + MM_N <= N && (N & 7u) == 0u
        && ((reinterpret_cast<unsigned long long>(C) & 15ull) == 0ull)) {
        auto pack_quad = [&](unsigned int v0, unsigned int v1, unsigned int v2, unsigned int v3) -> uint4 {
            const unsigned int a0 = (tid4 & 1u) ? v1 : v0;
            const unsigned int a1 = (tid4 & 1u) ? v0 : v1;
            const unsigned int a2 = (tid4 & 1u) ? v3 : v2;
            const unsigned int a3 = (tid4 & 1u) ? v2 : v3;
            const unsigned int b1 = __shfl_xor_sync(0xffffffffu, a1, 1);
            const unsigned int b3 = __shfl_xor_sync(0xffffffffu, a3, 1);
            const unsigned int p0 = (tid4 & 1u) ? b1 : a0;
            const unsigned int p1 = (tid4 & 1u) ? a0 : b1;
            const unsigned int p2 = (tid4 & 1u) ? b3 : a2;
            const unsigned int p3 = (tid4 & 1u) ? a2 : b3;
            const unsigned int lo0 = (tid4 & 2u) ? p0 : p2;
            const unsigned int lo1 = (tid4 & 2u) ? p1 : p3;
            const unsigned int r0 = __shfl_xor_sync(0xffffffffu, lo0, 2);
            const unsigned int r1 = __shfl_xor_sync(0xffffffffu, lo1, 2);
            const unsigned int q0 = (tid4 & 2u) ? r0 : p0;
            const unsigned int q1 = (tid4 & 2u) ? r1 : p1;
            const unsigned int q2 = (tid4 & 2u) ? p2 : r0;
            const unsigned int q3 = (tid4 & 2u) ? p3 : r1;
            return make_uint4(q0, q1, q2, q3);
        };
        #pragma unroll
        for (int mf = 0; mf < 2; mf++) {
            const unsigned int r0 = cta_m + warp_m + mf * 16 + group_id;
            const unsigned int r1 = r0 + 8;
            #pragma unroll
            for (int b = 0; b < 2; b++) {
                unsigned int u0[4], u1[4];
                #pragma unroll
                for (int jj = 0; jj < 4; jj++) {
                    const int nf = 4 * b + jj;
                    const __nv_bfloat162 p0 = __floats2bfloat162_rn(acc[mf][nf][0], acc[mf][nf][1]);
                    const __nv_bfloat162 p1 = __floats2bfloat162_rn(acc[mf][nf][2], acc[mf][nf][3]);
                    u0[jj] = *reinterpret_cast<const unsigned int*>(&p0);
                    u1[jj] = *reinterpret_cast<const unsigned int*>(&p1);
                }
                const unsigned int col8 = cta_n + warp_n + b * 32u + tid4 * 8u;
                const uint4 w0 = pack_quad(u0[0], u0[1], u0[2], u0[3]);
                if (r0 < M) {
                    *reinterpret_cast<uint4*>(&C[(unsigned long long)r0 * N + col8]) = w0;
                }
                const uint4 w1 = pack_quad(u1[0], u1[1], u1[2], u1[3]);
                if (r1 < M) {
                    *reinterpret_cast<uint4*>(&C[(unsigned long long)r1 * N + col8]) = w1;
                }
            }
        }
    } else {
        #pragma unroll
        for (int mf = 0; mf < 2; mf++) {
            #pragma unroll
            for (int nf = 0; nf < 8; nf++) {
                unsigned int c0 = cta_n + warp_n + nf * 8 + tid4 * 2;
                unsigned int c1 = c0 + 1;
                unsigned int r0 = cta_m + warp_m + mf * 16 + group_id;
                unsigned int r1 = r0 + 8;
                if (r0 < M) {
                    if (c1 < N) {
                        __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[mf][nf][0], acc[mf][nf][1]);
                        *reinterpret_cast<__nv_bfloat162*>(&C[(unsigned long long)r0 * N + c0]) = v0;
                    } else if (c0 < N) {
                        C[(unsigned long long)r0 * N + c0] = __float2bfloat16(acc[mf][nf][0]);
                    }
                }
                if (r1 < M) {
                    if (c1 < N) {
                        __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[mf][nf][2], acc[mf][nf][3]);
                        *reinterpret_cast<__nv_bfloat162*>(&C[(unsigned long long)r1 * N + c0]) = v1;
                    } else if (c0 < N) {
                        C[(unsigned long long)r1 * N + c0] = __float2bfloat16(acc[mf][nf][2]);
                    }
                }
            }
        }
    }
    #undef MM_ISSUE_A
    #undef MM_LOAD_B_REG
    #undef MM_DEQUANT_REG
}
#undef MM_M
#undef MM_N
#undef MM_K
#undef MM_PAD
#undef MM_STRIDE
#undef MM_THREADS

// ═══════════════════════════════════════════════════════════════════
// 128x128-tile, 8-warp shadows of the FP8-MMA transposed GEMMs.
//
//   w4a16_gemm_t_m128n128_w8  <->  w4a16_gemm_t / w4a16_gemm_t_m128
//   fp8_gemm_t_m128n128_w8    <->  fp8_gemm_t   / fp8_gemm_t_m128
//
// Bit-identical per output element: the same DEQUANT_T dequant (FP4 * fp8
// scale * scale2 -> cvt.rn.satfinite.e4m3x2), the same bf16x4_to_e4m3x4
// activation conversion, the same m16n8k32.e4m3.e4m3.f32 MMA consumed in
// ascending 32-wide K steps, and the same BF16 epilogue rounding. Only the
// CTA tile (128 rows x 128 cols, 8 warps of 32x64) and per-thread work
// assignment change; each B tile is now shared by 128 rows instead of 64
// (t) or 128 rows with 4 warps (m128).
// Grid: (N/128, ceil(M/128), 1)  Block: (256, 1, 1). Requires N % 128 == 0,
// K % 32 == 0.
// ═══════════════════════════════════════════════════════════════════
#define W8_M 128
#define W8_N 128
#define W8_THREADS 256

extern "C" __global__ void __launch_bounds__(W8_THREADS, 2)
w4a16_gemm_t_m128n128_w8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,   // [K/2, N] transposed
    const unsigned char* __restrict__ B_scale,    // [K/16, N] transposed
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * W8_N;
    const unsigned int cta_m = blockIdx.y * W8_M;
    if (cta_m >= M) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31;
    const unsigned int warp_m = (warp_id & 3) * 32;
    const unsigned int warp_n = (warp_id >> 2) * 64;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid4 = lane_id & 3;

    __shared__ __align__(16) __nv_bfloat16 smem_A[2][W8_M][K_STEP_T + PAD_T];     // 20480 B
    __shared__ __align__(16) unsigned char smem_Bp[2][K_STEP_T / 2][W8_N + BP_PAD]; //  4608 B
    __shared__ __align__(16) unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][W8_N + BP_PAD]; // 576 B
    __shared__ __align__(16) unsigned char smem_B_fp8[W8_N][K_STEP_T];              //  4096 B
    __shared__ float smem_LUT[16];
    if (tid < 16) smem_LUT[tid] = E2M1_LUT[tid];

    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; acc[i][j][2] = 0.f; acc[i][j][3] = 0.f; }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // A: 128 rows x 4 x 16B chunks = 512 chunks, 2 per thread. B: as w4a16_gemm_t (threads 0..127).
    #define W8_LOADS(buf, kb) do { \
        _Pragma("unroll") \
        for (int p = 0; p < 2; p++) { \
            unsigned int c = tid + p * W8_THREADS; \
            unsigned int row = c >> 2; \
            unsigned int a_col = (c & 3) << 3; \
            unsigned int gr = cta_m + row; \
            unsigned int gc = (kb) + a_col; \
            cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                &A[(unsigned long long)gr * K + gc], (gr < M) && (gc + 7 < K)); \
        } \
        if (tid < 128) { \
            unsigned int kp  = tid >> 3; \
            unsigned int ns  = (tid & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], (gns + 15 < N)); \
            } \
        } \
    } while (0)

    // Dequant: thread -> column n = tid>>1, k-half = tid&1 (parent's exact
    // per-element arithmetic, split across two threads per column).
    #define W8_DEQUANT(buf) do { \
        unsigned int my_n = tid >> 1; \
        unsigned int half = tid & 1; \
        unsigned char sb = smem_Bs[(buf)][half][my_n]; \
        __nv_fp8_e4m3 f; *(unsigned char*)&f = sb; \
        float sv = (float)f * scale2; \
        _Pragma("unroll") \
        for (int i = 0; i < 8; i++) { \
            int kp = half * 8 + i; \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv; \
            float hi = smem_LUT[packed >> 4]  * sv; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while (0)

    #define W8_COMPUTE(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int a[2][4]; \
        _Pragma("unroll") \
        for (int mf = 0; mf < 2; mf++) { \
            unsigned int fr0 = warp_m + mf * 16 + group_id, fr1 = fr0 + 8; \
            a[mf][0] = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid4 * 4]); \
            a[mf][1] = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid4 * 4]); \
            a[mf][2] = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid4 * 4]); \
            a[mf][3] = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid4 * 4]); \
        } \
        _Pragma("unroll") \
        for (int nt = 0; nt < 8; nt++) { \
            unsigned int nc = warp_n + nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid4]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid4]; \
            _Pragma("unroll") \
            for (int mf = 0; mf < 2; mf++) { \
                ATLAS_MMA_FP8_UNIT_M16N8K32( \
                    acc[mf][nt][0], acc[mf][nt][1], acc[mf][nt][2], acc[mf][nt][3], \
                    a[mf][0], a[mf][1], a[mf][2], a[mf][3], b0, b1, \
                    acc[mf][nt][0], acc[mf][nt][1], acc[mf][nt][2], acc[mf][nt][3]); \
            } \
        } \
    } while (0)

    W8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    W8_DEQUANT(0);
    __syncthreads();
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        W8_LOADS(nxt, k_base);
        cp_async_commit();
        W8_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        W8_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    W8_COMPUTE(cur);
    #undef W8_LOADS
    #undef W8_DEQUANT
    #undef W8_COMPUTE

    #pragma unroll
    for (int mf = 0; mf < 2; mf++)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int c0 = cta_n + warp_n + nt * 8 + tid4 * 2, c1 = c0 + 1;
            unsigned int r0 = cta_m + warp_m + mf * 16 + group_id, r1 = r0 + 8;
            if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16(acc[mf][nt][0]);
            if (r0 < M && c1 < N) C[(unsigned long long)r0 * N + c1] = __float2bfloat16(acc[mf][nt][1]);
            if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16(acc[mf][nt][2]);
            if (r1 < M && c1 < N) C[(unsigned long long)r1 * N + c1] = __float2bfloat16(acc[mf][nt][3]);
        }
}

extern "C" __global__ void __launch_bounds__(W8_THREADS, 2)
fp8_gemm_t_m128n128_w8(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * W8_N;
    const unsigned int cta_m = blockIdx.y * W8_M;
    if (cta_m >= M) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31;
    const unsigned int warp_m = (warp_id & 3) * 32;
    const unsigned int warp_n = (warp_id >> 2) * 64;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid4 = lane_id & 3;

    __shared__ __align__(16) __nv_bfloat16 smem_A[2][W8_M][K_STEP_T + PAD_T];  // 20480 B
    __shared__ __align__(16) unsigned char smem_B[2][W8_N][K_STEP_T];            //  8192 B

    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; acc[i][j][2] = 0.f; acc[i][j][3] = 0.f; }
    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define F8_LOADS(buf, kb) do { \
        _Pragma("unroll") \
        for (int p = 0; p < 2; p++) { \
            unsigned int c = tid + p * W8_THREADS; \
            unsigned int row = c >> 2; \
            unsigned int a_col = (c & 3) << 3; \
            unsigned int gr = cta_m + row; \
            unsigned int gc = (kb) + a_col; \
            cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                &A[(unsigned long long)gr * K + gc], (gr < M) && (gc + 7 < K)); \
        } \
        { \
            unsigned int my_n = tid >> 1; \
            unsigned int half = tid & 1; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][half * 16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + half * 16], valid); \
        } \
    } while (0)

    #define F8_COMPUTE(buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        unsigned int a[2][4]; \
        _Pragma("unroll") \
        for (int mf = 0; mf < 2; mf++) { \
            unsigned int fr0 = warp_m + mf * 16 + group_id, fr1 = fr0 + 8; \
            a[mf][0] = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid4 * 4]); \
            a[mf][1] = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid4 * 4]); \
            a[mf][2] = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid4 * 4]); \
            a[mf][3] = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid4 * 4]); \
        } \
        _Pragma("unroll") \
        for (int nt = 0; nt < 8; nt++) { \
            unsigned int nc = warp_n + nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(buf)][nc][4 * tid4]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(buf)][nc][16 + 4 * tid4]; \
            _Pragma("unroll") \
            for (int mf = 0; mf < 2; mf++) { \
                ATLAS_MMA_FP8_UNIT_M16N8K32( \
                    acc[mf][nt][0], acc[mf][nt][1], acc[mf][nt][2], acc[mf][nt][3], \
                    a[mf][0], a[mf][1], a[mf][2], a[mf][3], b0, b1, \
                    acc[mf][nt][0], acc[mf][nt][1], acc[mf][nt][2], acc[mf][nt][3]); \
            } \
        } \
    } while (0)

    F8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        F8_LOADS(nxt, k_base);
        cp_async_commit();
        F8_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    F8_COMPUTE(cur);
    #undef F8_LOADS
    #undef F8_COMPUTE

    #pragma unroll
    for (int mf = 0; mf < 2; mf++)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int c0 = cta_n + warp_n + nt * 8 + tid4 * 2, c1 = c0 + 1;
            unsigned int r0 = cta_m + warp_m + mf * 16 + group_id, r1 = r0 + 8;
            if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16(acc[mf][nt][0]);
            if (r0 < M && c1 < N) C[(unsigned long long)r0 * N + c1] = __float2bfloat16(acc[mf][nt][1]);
            if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16(acc[mf][nt][2]);
            if (r1 < M && c1 < N) C[(unsigned long long)r1 * N + c1] = __float2bfloat16(acc[mf][nt][3]);
        }
}
#undef W8_M
#undef W8_N
#undef W8_THREADS

// ---------------------------------------------------------------------------
// dequant_nvfp4_to_bf16: materialise, once, the exact BF16 weight bytes that
// w4a16_gemm_pipe_m128n128 builds on the fly, laid out [N, K] row-major so
// cuBLASLt can take them as the A operand of a TN BF16 GEMM.
//
// Per element this is statement-for-statement that kernel's dequant:
//     dst = __float2bfloat16(LUT[code] * (float)block_scale_e4m3 * scale2)
// so the library multiplies the identical operands, and the ONLY difference
// from the hand-written path is the order in which the FP32 accumulator sums
// them. Unlike the SSM/e4m3 case that difference is NOT provably invisible:
// BF16 products carry 16 significant bits and K=5120 needs ~13 more, which
// exceeds FP32's 24-bit mantissa, so this route is gated and gate-tested.
//
// Attention weight layout (from that kernel's loader) differs from the SSM's:
// B_packed is row-major [N, K/2] and B_scale row-major [N, K/16], so one
// thread per packed byte gives fully coalesced reads and writes.
extern "C" __global__ void dequant_nvfp4_to_bf16(
    const unsigned char* __restrict__ B_packed,  // [N, K/2]
    const unsigned char* __restrict__ B_scale,   // [N, K/16] e4m3 bytes
    __nv_bfloat16* __restrict__ out,             // [N, K]
    float scale2,
    unsigned int N,
    unsigned int K
) {
    const unsigned long long half_k = K >> 1;
    const unsigned long long total = (unsigned long long)N * half_k;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const unsigned long long n = i / half_k;
        const unsigned long long kb = i - n * half_k;   // packed-byte index within the row
        const unsigned char packed = B_packed[i];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = B_scale[n * (K >> 4) + (kb >> 3)];
        const float fs = (float)fp8;
        __nv_bfloat16* dst = out + n * K + (kb << 1);
        dst[0] = __float2bfloat16(E2M1_LUT[packed & 0xF] * fs * scale2);
        dst[1] = __float2bfloat16(E2M1_LUT[packed >> 4] * fs * scale2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Coalesced-K single-sweep and intra-CTA Warp-Split-K row-major NVFP4
// Tensor-Core GEMM (BF16 x BF16 MMA, unclipped BF16 activations, M <= 128).
//
// Eliminates unscaled FP8 E4M3 activation clipping (±448) and global Split-K
// partial workspace traffic / reduce launches on small-M dense projections:
//   - Non-split-K tiers (`CO_N in {128, 64, 32, 16}`, 1 n8 tile per warp):
//     zero duplicate weight loads across warps, full smem_A reuse across warps,
//     zero smem_B, zero smem_LUT.
//   - Intra-CTA Warp-Split-K tiers (`n8_sk8_m16`, `n16_sk4_m16`, `n32_sk4_m16`,
//     `n64_sk4_m16`, `n32_sk4_m32`, `n64_sk2_m64`, plus v2 2-stage K=256
//     prefetch variants): saturate wide Blackwell SMs with zero global atomics,
//     zero workspace scratch, deterministic FP32 SMEM reduction, and full
//     CUDA-graph capture safety.
//   - Native BF16x2 register dequant (`prmt` + `mul.rn.bf16x2`) computes
//     `E2M1 * E4M3` bit-exact in BF16 (<= 7 significant bits) and hoists
//     `scale2` once to the FP32 epilogue (`acc * scale2`).
//   - Epilogue reuses smem as `smem.c` / `smem.red` for 100% coalesced
//     streaming stores (`st.global.cs`) safe for arbitrary or odd widths.
// ═══════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ void st_global_cs_u16(__nv_bfloat16* ptr, __nv_bfloat16 val) {
    const unsigned short bits = *reinterpret_cast<const unsigned short*>(&val);
    asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(ptr), "h"(bits));
}

__device__ __forceinline__ void st_global_cs_u32(__nv_bfloat162* ptr, __nv_bfloat162 val) {
    const unsigned int bits = *reinterpret_cast<const unsigned int*>(&val);
    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(ptr), "r"(bits));
}

__device__ __forceinline__ void st_global_cs_u64(void* ptr, uint2 val) {
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};" :: "l"(ptr), "r"(val.x), "r"(val.y));
}

__device__ __forceinline__ void st_global_cs_u128(void* ptr, uint4 val) {
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

// Bit-exact E2M1 x E4M3 -> BF16x2 dequantization without `scale2` (`scale2` is
// hoisted once to the FP32 epilogue). Because E2M1 (<= 6.0, 3 sig bits) times
// finite E4M3 (<= 448.0, 4 sig bits) has at most 7 significant bits and lies
// strictly in normal BF16 range [2^-10, 2688], `mul.rn.bf16x2` is exact with
// zero overflow/underflow.
__device__ __forceinline__ void dequant_nvfp4_pair_to_bf16x2_noscale2(
    unsigned int byte0,
    unsigned int byte1,
    unsigned int sb,
    unsigned int& b0,
    unsigned int& b1
) {
    // BF16 encodings of E2M1 magnitudes {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}:
    //   0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0
    constexpr unsigned int kLo0 = 0xC0800000u;
    constexpr unsigned int kLo1 = 0xC0804000u;
    constexpr unsigned int kHi0 = 0x3F3F3F00u;
    constexpr unsigned int kHi1 = 0x40404040u;
    constexpr unsigned int kSignLut = 0x00008000u;

    const unsigned int w4 = byte0 | (byte1 << 8);
    const unsigned int mag = w4 & 0x7777u;
    const unsigned int sgn = (w4 >> 3) & 0x1111u;

    unsigned int b_lo, b_hi, b_sg, e01, e23;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(b_lo) : "r"(kLo0), "r"(kLo1), "r"(mag));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(b_hi) : "r"(kHi0), "r"(kHi1), "r"(mag));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(b_sg) : "r"(kSignLut), "r"(0u), "r"(sgn));
    b_hi ^= b_sg;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(e01) : "r"(b_lo), "r"(b_hi), "r"(0x5140u));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(e23) : "r"(b_lo), "r"(b_hi), "r"(0x7362u));

    __nv_fp8_e4m3 fp8;
    *reinterpret_cast<unsigned char*>(&fp8) = static_cast<unsigned char>(sb);
    const __nv_bfloat162 fs2 = __float2bfloat162_rn(static_cast<float>(fp8));
    const unsigned int fs_u32 = *reinterpret_cast<const unsigned int*>(&fs2);

    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(b0) : "r"(e01), "r"(fs_u32));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(b1) : "r"(e23), "r"(fs_u32));
}

template <unsigned int TILE_M, unsigned int CO_N, bool DOUBLE_BUF_A_128>
__device__ __forceinline__ void w4a16_gemm_coalesced_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    static_assert(
        CO_N == 16u || CO_N == 32u || CO_N == 64u || CO_N == 128u,
        "CO_N must be 16, 32, 64, or 128");

    constexpr unsigned int NUM_WARPS = CO_N / 8u;
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int A_TILE_K = DOUBLE_BUF_A_128 ? 128u : 64u;
    constexpr unsigned int A_STRIDE = A_TILE_K + 8u;
    constexpr unsigned int C_STRIDE = CO_N + 8u;
    constexpr unsigned int A_CHUNKS_PER_ROW = A_TILE_K / 8u;
    static_assert(
        (TILE_M * A_CHUNKS_PER_ROW) % NUM_THREADS == 0u,
        "A tile chunks must divide evenly across CTA threads");
    static_assert(
        NUM_THREADS % A_CHUNKS_PER_ROW == 0u,
        "CTA threads must be a multiple of A chunks per row");
    constexpr unsigned int A_ITERS = (TILE_M * A_CHUNKS_PER_ROW) / NUM_THREADS;
    constexpr unsigned int A_ROWS_PER_ITER = NUM_THREADS / A_CHUNKS_PER_ROW;

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_n = blockIdx.x * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int byte_shift = tid4 * 8u;
    const unsigned int gn = cta_n + warp_id * 8u + group_id;

    union Smem {
        __nv_bfloat16 a[2][TILE_M][A_STRIDE];
        __nv_bfloat16 c[TILE_M][C_STRIDE];
    };
    __shared__ __align__(16) Smem smem;
    static_assert(sizeof(Smem) <= 49152u, "static SMEM must stay <= 48 KiB");

    float acc[M_FRAGS][4];
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        acc[mf][0] = 0.0f;
        acc[mf][1] = 0.0f;
        acc[mf][2] = 0.0f;
        acc[mf][3] = 0.0f;
    }

    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int a_row = lane & 15u;
    const unsigned int a_koff = (lane >> 4) * 8u;
    const unsigned int a_tid_row = tid / A_CHUNKS_PER_ROW;
    const unsigned int a_tid_col8 = (tid % A_CHUNKS_PER_ROW) * 8u;

    unsigned long long pol_b;
    if (gridDim.y == 1u) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_b));
    } else {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1000)
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_b));
#else
        asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(pol_b));
#endif
    }

    auto issue_a = [&](unsigned int buf, unsigned int kk) {
        const bool k_valid = (kk + a_tid_col8 < K);
        #pragma unroll
        for (unsigned int p = 0u; p < A_ITERS; ++p) {
            if (cta_m + p * A_ROWS_PER_ITER < M) {
                const unsigned int row = p * A_ROWS_PER_ITER + a_tid_row;
                const unsigned int gr = cta_m + row;
                const bool valid = (gr < M) && k_valid;
                const __nv_bfloat16* src_a =
                    valid ? &A[(unsigned long long)gr * K + kk + a_tid_col8] : A;
                cp_async_pred_16(&smem.a[buf][row][a_tid_col8], src_a, valid);
            }
        }
        cp_async_commit();
    };

    auto load_b = [&](unsigned int kk, uint4& bp_out, unsigned int& bs_out) {
        const unsigned int k_start = kk + tid4 * 32u;
        if (gn < N && k_start < K) {
            const unsigned char* bp_ptr =
                B_packed + (unsigned long long)gn * half_K + (k_start >> 1);
            asm volatile(
                "ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                : "=r"(bp_out.x), "=r"(bp_out.y), "=r"(bp_out.z), "=r"(bp_out.w)
                : "l"(bp_ptr), "l"(pol_b));
            const unsigned short s16 = __ldg(reinterpret_cast<const unsigned short*>(
                B_scale + (unsigned long long)gn * num_groups + (k_start >> 4)));
            bs_out = static_cast<unsigned int>(s16);
        } else {
            bp_out = make_uint4(0u, 0u, 0u, 0u);
            bs_out = 0u;
        }
    };

    auto mma_step = [&](unsigned int cur_buf, unsigned int k0, unsigned int b0, unsigned int b1) {
        #pragma unroll
        for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
            if (cta_m + mf * 16u < M) {
                unsigned int a0, a1, a2, a3;
                ldmatrix_x4(
                    a0, a1, a2, a3,
                    &smem.a[cur_buf][mf * 16u + a_row][k0 + a_koff]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc[mf][0]), "=f"(acc[mf][1]), "=f"(acc[mf][2]), "=f"(acc[mf][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc[mf][0]), "f"(acc[mf][1]), "f"(acc[mf][2]), "f"(acc[mf][3]));
            }
        }
    };

    if constexpr (DOUBLE_BUF_A_128) {
        issue_a(0u, 0u);
        uint4 bp_cur;
        unsigned int bs_cur;
        load_b(0u, bp_cur, bs_cur);

        unsigned int cur = 0u, nxt = 1u;
        for (unsigned int k_base = 0u; k_base < K; k_base += 128u) {
            cp_async_wait_group<0>();
            __syncthreads();

            const unsigned int k_next = k_base + 128u;
            uint4 bp_nxt = make_uint4(0u, 0u, 0u, 0u);
            unsigned int bs_nxt = 0u;
            if (k_next < K) {
                issue_a(nxt, k_next);
                load_b(k_next, bp_nxt, bs_nxt);
            }

            #pragma unroll
            for (int src = 0; src < 4; ++src) {
                if (src == 0 || k_base + static_cast<unsigned int>(src) * 32u < K) {
                    const unsigned int sb_pair = __shfl_sync(0xffffffffu, bs_cur, src, 4);
                    {
                        const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp_cur.x, src, 4);
                        const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp_cur.y, src, 4);
                        unsigned int b0, b1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pk_lo >> byte_shift) & 0xFFu,
                            (pk_hi >> byte_shift) & 0xFFu,
                            sb_pair & 0xFFu,
                            b0,
                            b1);
                        mma_step(cur, static_cast<unsigned int>(src * 32), b0, b1);
                    }
                    {
                        const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp_cur.z, src, 4);
                        const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp_cur.w, src, 4);
                        unsigned int b0, b1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pk_lo >> byte_shift) & 0xFFu,
                            (pk_hi >> byte_shift) & 0xFFu,
                            sb_pair >> 8,
                            b0,
                            b1);
                        mma_step(cur, static_cast<unsigned int>(src * 32 + 16), b0, b1);
                    }
                }
            }

            bp_cur = bp_nxt;
            bs_cur = bs_nxt;
            const unsigned int tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
    } else {
        issue_a(0u, 0u);
        uint4 bp_cur;
        unsigned int bs_cur;
        load_b(0u, bp_cur, bs_cur);

        for (unsigned int k_base = 0u; k_base < K; k_base += 128u) {
            cp_async_wait_group<0>();
            __syncthreads();

            if (k_base + 64u < K) {
                issue_a(1u, k_base + 64u);
            }
            uint4 bp_nxt = make_uint4(0u, 0u, 0u, 0u);
            unsigned int bs_nxt = 0u;
            if (k_base + 128u < K) {
                load_b(k_base + 128u, bp_nxt, bs_nxt);
            }

            #pragma unroll
            for (int src = 0; src < 2; ++src) {
                if (src == 0 || k_base + 32u < K) {
                    const unsigned int sb_pair = __shfl_sync(0xffffffffu, bs_cur, src, 4);
                    {
                        const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp_cur.x, src, 4);
                        const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp_cur.y, src, 4);
                        unsigned int b0, b1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pk_lo >> byte_shift) & 0xFFu,
                            (pk_hi >> byte_shift) & 0xFFu,
                            sb_pair & 0xFFu,
                            b0,
                            b1);
                        mma_step(0u, static_cast<unsigned int>(src * 32), b0, b1);
                    }
                    {
                        const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp_cur.z, src, 4);
                        const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp_cur.w, src, 4);
                        unsigned int b0, b1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pk_lo >> byte_shift) & 0xFFu,
                            (pk_hi >> byte_shift) & 0xFFu,
                            sb_pair >> 8,
                            b0,
                            b1);
                        mma_step(0u, static_cast<unsigned int>(src * 32 + 16), b0, b1);
                    }
                }
            }

            if (k_base + 64u < K) {
                cp_async_wait_group<0>();
                __syncthreads();
                if (k_base + 128u < K) {
                    issue_a(0u, k_base + 128u);
                }

                #pragma unroll
                for (int src = 2; src < 4; ++src) {
                    if (src == 2 || k_base + 96u < K) {
                        const unsigned int sb_pair = __shfl_sync(0xffffffffu, bs_cur, src, 4);
                        {
                            const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp_cur.x, src, 4);
                            const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp_cur.y, src, 4);
                            unsigned int b0, b1;
                            dequant_nvfp4_pair_to_bf16x2_noscale2(
                                (pk_lo >> byte_shift) & 0xFFu,
                                (pk_hi >> byte_shift) & 0xFFu,
                                sb_pair & 0xFFu,
                                b0,
                                b1);
                            mma_step(1u, static_cast<unsigned int>((src - 2) * 32), b0, b1);
                        }
                        {
                            const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp_cur.z, src, 4);
                            const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp_cur.w, src, 4);
                            unsigned int b0, b1;
                            dequant_nvfp4_pair_to_bf16x2_noscale2(
                                (pk_lo >> byte_shift) & 0xFFu,
                                (pk_hi >> byte_shift) & 0xFFu,
                                sb_pair >> 8,
                                b0,
                                b1);
                            mma_step(1u, static_cast<unsigned int>((src - 2) * 32 + 16), b0, b1);
                        }
                    }
                }
            }

            bp_cur = bp_nxt;
            bs_cur = bs_nxt;
        }
    }

    __syncthreads();
    const unsigned int smem_col = warp_id * 8u + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.c[r0][smem_col]) =
                __floats2bfloat162_rn(acc[mf][0] * scale2, acc[mf][1] * scale2);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.c[r1][smem_col]) =
                __floats2bfloat162_rn(acc[mf][2] * scale2, acc[mf][3] * scale2);
        }
    }
    __syncthreads();

    const unsigned int rows_in_cta = (M - cta_m) < TILE_M ? (M - cta_m) : TILE_M;
    const unsigned int cols_in_cta = (cta_n + CO_N <= N) ? CO_N : (N - cta_n);
    const unsigned long long c_addr = reinterpret_cast<unsigned long long>(C);
    if (cols_in_cta == CO_N && (N & 3u) == 0u && (c_addr & 7ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U64 = CO_N / 4u;
        constexpr unsigned int ROWS_PER_STEP_U64 = NUM_THREADS / THREADS_PER_ROW_U64;
        const unsigned int row_off = tid / THREADS_PER_ROW_U64;
        const unsigned int col = (tid % THREADS_PER_ROW_U64) * 4u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U64) {
            const uint2 v = *reinterpret_cast<const uint2*>(&smem.c[r][col]);
            st_global_cs_u64(&C[(unsigned long long)(cta_m + r) * N + cta_n + col], v);
        }
    } else if (cols_in_cta == CO_N && (N & 1u) == 0u && (c_addr & 3ull) == 0ull) {
        if constexpr (CO_N == 128u) {
            for (unsigned int r = warp_id; r < rows_in_cta; r += NUM_WARPS) {
                const unsigned int col = lane * 2u;
                st_global_cs_u32(
                    reinterpret_cast<__nv_bfloat162*>(
                        &C[(unsigned long long)(cta_m + r) * N + cta_n + col]),
                    *reinterpret_cast<const __nv_bfloat162*>(&smem.c[r][col]));
                st_global_cs_u32(
                    reinterpret_cast<__nv_bfloat162*>(
                        &C[(unsigned long long)(cta_m + r) * N + cta_n + col + 64u]),
                    *reinterpret_cast<const __nv_bfloat162*>(&smem.c[r][col + 64u]));
            }
        } else {
            constexpr unsigned int THREADS_PER_ROW_U32 = CO_N / 2u;
            constexpr unsigned int ROWS_PER_STEP_U32 = NUM_THREADS / THREADS_PER_ROW_U32;
            const unsigned int row_off = tid / THREADS_PER_ROW_U32;
            const unsigned int col = (tid % THREADS_PER_ROW_U32) * 2u;
            for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U32) {
                st_global_cs_u32(
                    reinterpret_cast<__nv_bfloat162*>(
                        &C[(unsigned long long)(cta_m + r) * N + cta_n + col]),
                    *reinterpret_cast<const __nv_bfloat162*>(&smem.c[r][col]));
            }
        }
    } else {
        if constexpr (CO_N >= 32u) {
            for (unsigned int r = warp_id; r < rows_in_cta; r += NUM_WARPS) {
                __nv_bfloat16* c_row = C + (unsigned long long)(cta_m + r) * N + cta_n;
                #pragma unroll
                for (unsigned int c = 0u; c < CO_N; c += 32u) {
                    if (c + lane < cols_in_cta) {
                        st_global_cs_u16(&c_row[c + lane], smem.c[r][c + lane]);
                    }
                }
            }
        } else {
            constexpr unsigned int ROWS_PER_STEP_U16 = NUM_THREADS / CO_N;
            const unsigned int row_off = tid / CO_N;
            const unsigned int col = tid % CO_N;
            for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U16) {
                if (col < cols_in_cta) {
                    st_global_cs_u16(
                        &C[(unsigned long long)(cta_m + r) * N + cta_n + col],
                        smem.c[r][col]);
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 3)
w4a16_gemm_coalesced_n64_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<32u, 64u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_n64_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<64u, 64u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_n64_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<128u, 64u, false>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 4)
w4a16_gemm_coalesced_n32_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<32u, 32u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 2)
w4a16_gemm_coalesced_n32_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<64u, 32u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 2)
w4a16_gemm_coalesced_n32_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<128u, 32u, false>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(64, 4)
w4a16_gemm_coalesced_n16_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<32u, 16u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(64, 2)
w4a16_gemm_coalesced_n16_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<64u, 16u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(64, 2)
w4a16_gemm_coalesced_n16_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_body<128u, 16u, false>(A, B_packed, B_scale, scale2, C, M, N, K);
}

template <
    unsigned int TILE_M,
    unsigned int CO_N,
    unsigned int SPLIT_K,
    unsigned int SUB_K>
__device__ __forceinline__ void w4a16_gemm_coalesced_splitk_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    static_assert(CO_N % 8u == 0u, "CO_N must be a multiple of 8");
    static_assert(SPLIT_K >= 2u, "SPLIT_K must be >= 2");
    static_assert(
        SUB_K == 16u || SUB_K == 32u || SUB_K == 64u || SUB_K == 128u,
        "SUB_K must be 16, 32, 64, or 128");

    constexpr unsigned int N_WARPS = CO_N / 8u;
    constexpr unsigned int NUM_WARPS = N_WARPS * SPLIT_K;
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int K_STEP_CTA = SPLIT_K * 128u;
    constexpr unsigned int NUM_SUBS = 128u / SUB_K;
    constexpr unsigned int MMAS_PER_SUB = SUB_K / 16u;
    constexpr unsigned int A_SLAB_K = SPLIT_K * SUB_K;
    constexpr unsigned int A_STRIDE = A_SLAB_K + 8u;
    constexpr unsigned int RED_STRIDE = CO_N + 4u;

    constexpr unsigned int THREADS_PER_SK = N_WARPS * 32u;
    constexpr unsigned int CHUNKS_PER_SUB = SUB_K / 8u;
    static_assert(
        THREADS_PER_SK % CHUNKS_PER_SUB == 0u,
        "threads per split-K group must be a multiple of A sub-slab chunks");
    constexpr unsigned int A_ROWS_PER_ITER = THREADS_PER_SK / CHUNKS_PER_SUB;
    static_assert(
        TILE_M % A_ROWS_PER_ITER == 0u,
        "TILE_M must be a positive multiple of A_ROWS_PER_ITER");
    constexpr unsigned int A_ITERS = TILE_M / A_ROWS_PER_ITER;
    static_assert(A_ITERS >= 1u, "A_ITERS must be >= 1");
    static_assert(NUM_THREADS % CO_N == 0u, "NUM_THREADS must be a multiple of CO_N");
    static_assert(NUM_THREADS % (CO_N / 2u) == 0u, "NUM_THREADS must be a multiple of CO_N/2");
    static_assert(NUM_THREADS % (CO_N / 4u) == 0u, "NUM_THREADS must be a multiple of CO_N/4");

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_n = blockIdx.x * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_k = warp_id % SPLIT_K;
    const unsigned int warp_n = warp_id / SPLIT_K;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int byte_shift = tid4 * 8u;
    const unsigned int gn = cta_n + warp_n * 8u + group_id;

    union Smem {
        __nv_bfloat16 a[2][TILE_M][A_STRIDE];
        float red[SPLIT_K][TILE_M][RED_STRIDE];
    };
    __shared__ __align__(16) Smem smem;
    static_assert(sizeof(Smem) <= 49152u, "static SMEM must stay <= 48 KiB");

    float acc[M_FRAGS][4];
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        acc[mf][0] = 0.0f;
        acc[mf][1] = 0.0f;
        acc[mf][2] = 0.0f;
        acc[mf][3] = 0.0f;
    }

    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int a_row = lane & 15u;
    const unsigned int a_koff = (lane >> 4) * 8u;

    const unsigned int tid_in_sk = warp_n * 32u + lane;
    const unsigned int a_tid_row = tid_in_sk / CHUNKS_PER_SUB;
    const unsigned int a_tid_sub_col8 = (tid_in_sk % CHUNKS_PER_SUB) * 8u;
    const unsigned int warp_k_smem_base = warp_k * SUB_K;
    const unsigned int a_smem_col = warp_k_smem_base + a_tid_sub_col8;

    unsigned long long pol_b;
    if (gridDim.y == 1u) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_b));
    } else {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1000)
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_b));
#else
        asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(pol_b));
#endif
    }

    auto issue_a = [&](unsigned int buf, unsigned int k_cta, unsigned int sub) {
        const unsigned int gk = k_cta + warp_k * 128u + sub * SUB_K + a_tid_sub_col8;
        const bool k_valid = (gk < K);
        #pragma unroll
        for (unsigned int p = 0u; p < A_ITERS; ++p) {
            if (cta_m + p * A_ROWS_PER_ITER < M) {
                const unsigned int row = p * A_ROWS_PER_ITER + a_tid_row;
                const unsigned int gr = cta_m + row;
                const bool valid = (gr < M) && k_valid;
                const __nv_bfloat16* src_a =
                    valid ? &A[(unsigned long long)gr * K + gk] : A;
                cp_async_pred_16(&smem.a[buf][row][a_smem_col], src_a, valid);
            }
        }
        cp_async_commit();
    };

    auto load_b = [&](unsigned int k_cta, uint4& bp_out, unsigned int& bs_out) {
        const unsigned int k_start = k_cta + warp_k * 128u + tid4 * 32u;
        if (gn < N && k_start < K) {
            const unsigned char* bp_ptr =
                B_packed + (unsigned long long)gn * half_K + (k_start >> 1);
            asm volatile(
                "ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                : "=r"(bp_out.x), "=r"(bp_out.y), "=r"(bp_out.z), "=r"(bp_out.w)
                : "l"(bp_ptr), "l"(pol_b));
            const unsigned short s16 = __ldg(reinterpret_cast<const unsigned short*>(
                B_scale + (unsigned long long)gn * num_groups + (k_start >> 4)));
            bs_out = static_cast<unsigned int>(s16);
        } else {
            bp_out = make_uint4(0u, 0u, 0u, 0u);
            bs_out = 0u;
        }
    };

    auto mma_step = [&](unsigned int cur_buf, unsigned int k0, unsigned int b0, unsigned int b1) {
        #pragma unroll
        for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
            if (cta_m + mf * 16u < M) {
                unsigned int a0, a1, a2, a3;
                ldmatrix_x4(
                    a0, a1, a2, a3,
                    &smem.a[cur_buf][mf * 16u + a_row][k0 + a_koff]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc[mf][0]), "=f"(acc[mf][1]), "=f"(acc[mf][2]), "=f"(acc[mf][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc[mf][0]), "f"(acc[mf][1]), "f"(acc[mf][2]), "f"(acc[mf][3]));
            }
        }
    };

    unsigned int cur = 0u, nxt = 1u;
    issue_a(0u, 0u, 0u);
    uint4 bp_cur;
    unsigned int bs_cur;
    load_b(0u, bp_cur, bs_cur);

    for (unsigned int k_cta = 0u; k_cta < K; k_cta += K_STEP_CTA) {
        uint4 bp_nxt = make_uint4(0u, 0u, 0u, 0u);
        unsigned int bs_nxt = 0u;
        const unsigned int k_next_cta = k_cta + K_STEP_CTA;
        const unsigned int k_warp = k_cta + warp_k * 128u;

        #pragma unroll
        for (unsigned int sub = 0u; sub < NUM_SUBS; ++sub) {
            cp_async_wait_group<0>();
            if constexpr (N_WARPS > 1u) {
                __syncthreads();
            } else {
                __syncwarp();
            }

            if (sub + 1u < NUM_SUBS) {
                issue_a(nxt, k_cta, sub + 1u);
            } else if (k_next_cta < K) {
                issue_a(nxt, k_next_cta, 0u);
            }
            if (sub == 0u && k_next_cta < K) {
                load_b(k_next_cta, bp_nxt, bs_nxt);
            }

            if (k_warp + sub * SUB_K < K) {
                #pragma unroll
                for (unsigned int mi = 0u; mi < MMAS_PER_SUB; ++mi) {
                    const unsigned int k_off = sub * SUB_K + mi * 16u;
                    if (k_warp + k_off < K) {
                        const int src = static_cast<int>(k_off >> 5);
                        const unsigned int hi_half = (k_off >> 4) & 1u;
                        const unsigned int sb_pair = __shfl_sync(0xffffffffu, bs_cur, src, 4);
                        const unsigned int pk_lo = __shfl_sync(
                            0xffffffffu, hi_half ? bp_cur.z : bp_cur.x, src, 4);
                        const unsigned int pk_hi = __shfl_sync(
                            0xffffffffu, hi_half ? bp_cur.w : bp_cur.y, src, 4);
                        const unsigned int sb = hi_half ? (sb_pair >> 8) : (sb_pair & 0xFFu);
                        unsigned int b0, b1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pk_lo >> byte_shift) & 0xFFu,
                            (pk_hi >> byte_shift) & 0xFFu,
                            sb,
                            b0,
                            b1);
                        mma_step(cur, warp_k_smem_base + mi * 16u, b0, b1);
                    }
                }
            }
            const unsigned int tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
        bp_cur = bp_nxt;
        bs_cur = bs_nxt;
    }

    __syncthreads();
    const unsigned int smem_col = warp_n * 8u + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<float2*>(&smem.red[warp_k][r0][smem_col]) =
                make_float2(acc[mf][0], acc[mf][1]);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<float2*>(&smem.red[warp_k][r1][smem_col]) =
                make_float2(acc[mf][2], acc[mf][3]);
        }
    }
    __syncthreads();

    const unsigned int rows_in_cta = (M - cta_m) < TILE_M ? (M - cta_m) : TILE_M;
    const unsigned int cols_in_cta = (cta_n + CO_N <= N) ? CO_N : (N - cta_n);
    const unsigned long long c_addr = reinterpret_cast<unsigned long long>(C);
    if (cols_in_cta == CO_N && (N & 3u) == 0u && (c_addr & 7ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U64 = CO_N / 4u;
        constexpr unsigned int ROWS_PER_STEP_U64 = NUM_THREADS / THREADS_PER_ROW_U64;
        const unsigned int row_off = tid / THREADS_PER_ROW_U64;
        const unsigned int col = (tid % THREADS_PER_ROW_U64) * 4u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U64) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            #pragma unroll
            for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                const float2 lo = *reinterpret_cast<const float2*>(&smem.red[sk][r][col]);
                const float2 hi = *reinterpret_cast<const float2*>(&smem.red[sk][r][col + 2u]);
                s0 += lo.x;
                s1 += lo.y;
                s2 += hi.x;
                s3 += hi.y;
            }
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            const __nv_bfloat162 p1 = __floats2bfloat162_rn(s2 * scale2, s3 * scale2);
            uint2 v;
            v.x = *reinterpret_cast<const unsigned int*>(&p0);
            v.y = *reinterpret_cast<const unsigned int*>(&p1);
            st_global_cs_u64(&C[(unsigned long long)(cta_m + r) * N + cta_n + col], v);
        }
    } else if (cols_in_cta == CO_N && (N & 1u) == 0u && (c_addr & 3ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U32 = CO_N / 2u;
        constexpr unsigned int ROWS_PER_STEP_U32 = NUM_THREADS / THREADS_PER_ROW_U32;
        const unsigned int row_off = tid / THREADS_PER_ROW_U32;
        const unsigned int col = (tid % THREADS_PER_ROW_U32) * 2u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U32) {
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll
            for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                const float2 lo = *reinterpret_cast<const float2*>(&smem.red[sk][r][col]);
                s0 += lo.x;
                s1 += lo.y;
            }
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            st_global_cs_u32(
                reinterpret_cast<__nv_bfloat162*>(
                    &C[(unsigned long long)(cta_m + r) * N + cta_n + col]),
                p0);
        }
    } else {
        constexpr unsigned int ROWS_PER_STEP_U16 = NUM_THREADS / CO_N;
        const unsigned int row_off = tid / CO_N;
        const unsigned int col = tid % CO_N;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U16) {
            if (col < cols_in_cta) {
                float s = 0.0f;
                #pragma unroll
                for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                    s += smem.red[sk][r][col];
                }
                const __nv_bfloat16 out_v = __float2bfloat16(s * scale2);
                st_global_cs_u16(
                    &C[(unsigned long long)(cta_m + r) * N + cta_n + col],
                    out_v);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 3)
w4a16_gemm_coalesced_n8_sk8_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_body<16u, 8u, 8u, 32u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 4)
w4a16_gemm_coalesced_n16_sk4_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_body<16u, 16u, 4u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 3)
w4a16_gemm_coalesced_n32_sk4_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_body<16u, 32u, 4u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(1024, 1)
w4a16_gemm_coalesced_n64_sk4_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_body<16u, 64u, 4u, 128u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 2)
w4a16_gemm_coalesced_n32_sk4_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_body<32u, 32u, 4u, 32u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 1)
w4a16_gemm_coalesced_n64_sk2_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_body<64u, 64u, 2u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

template <unsigned int TILE_M, unsigned int CO_N, bool DOUBLE_BUF_A_128>
struct W4a16CoalescedV2Smem {
    static constexpr unsigned int A_TILE_K = DOUBLE_BUF_A_128 ? 128u : 64u;
    static constexpr unsigned int A_STRIDE = A_TILE_K + 8u;
    static constexpr unsigned int C_STRIDE = CO_N + 8u;
    union Storage {
        __nv_bfloat16 a[2][TILE_M][A_STRIDE];
        __nv_bfloat16 c[TILE_M][C_STRIDE];
    };
};

template <unsigned int TILE_M, unsigned int CO_N, bool DOUBLE_BUF_A_128>
__device__ __forceinline__ void w4a16_gemm_coalesced_v2_body_impl(
    typename W4a16CoalescedV2Smem<TILE_M, CO_N, DOUBLE_BUF_A_128>::Storage& smem,
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    static_assert(
        CO_N == 32u || CO_N == 64u || CO_N == 128u,
        "v2 non-split-K CO_N must be 32, 64, or 128");

    constexpr unsigned int NUM_WARPS = CO_N / 8u;
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int A_TILE_K = DOUBLE_BUF_A_128 ? 128u : 64u;
    constexpr unsigned int A_CHUNKS_PER_ROW = A_TILE_K / 8u;
    static_assert(
        NUM_THREADS % A_CHUNKS_PER_ROW == 0u,
        "CTA threads must be a multiple of A chunks per row");
    constexpr unsigned int A_ROWS_PER_ITER = NUM_THREADS / A_CHUNKS_PER_ROW;
    constexpr unsigned int A_ITERS = (TILE_M + A_ROWS_PER_ITER - 1u) / A_ROWS_PER_ITER;
    static_assert(
        sizeof(typename W4a16CoalescedV2Smem<TILE_M, CO_N, DOUBLE_BUF_A_128>::Storage) <= 49152u,
        "static SMEM must stay <= 48 KiB");

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_n = blockIdx.x * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int byte_shift = tid4 * 8u;
    const unsigned int gn = cta_n + warp_id * 8u + group_id;

    float acc[M_FRAGS][4];
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        acc[mf][0] = 0.0f;
        acc[mf][1] = 0.0f;
        acc[mf][2] = 0.0f;
        acc[mf][3] = 0.0f;
    }

    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int a_row = lane & 15u;
    const unsigned int a_koff = (lane >> 4) * 8u;
    const unsigned int a_tid_row = tid / A_CHUNKS_PER_ROW;
    const unsigned int a_tid_col8 = (tid % A_CHUNKS_PER_ROW) * 8u;
    const bool scale_aligned4 =
        ((K & 63u) == 0u) && ((reinterpret_cast<unsigned long long>(B_scale) & 3ull) == 0ull);

    unsigned long long pol_b;
    if (gridDim.y == 1u) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_b));
    } else {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1000)
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_b));
#else
        asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(pol_b));
#endif
    }

    auto issue_a = [&](unsigned int buf, unsigned int kk) {
        const bool k_valid = (kk + a_tid_col8 < K);
        #pragma unroll
        for (unsigned int p = 0u; p < A_ITERS; ++p) {
            const unsigned int row = p * A_ROWS_PER_ITER + a_tid_row;
            if (row < TILE_M && cta_m + p * A_ROWS_PER_ITER < M) {
                const unsigned int gr = cta_m + row;
                const bool valid = (gr < M) && k_valid;
                const __nv_bfloat16* src_a =
                    valid ? &A[(unsigned long long)gr * K + kk + a_tid_col8] : A;
                cp_async_pred_16(&smem.a[buf][row][a_tid_col8], src_a, valid);
            }
        }
        cp_async_commit();
    };

    auto load_b_packed = [&](unsigned int kk, uint4& bp_out) {
        const unsigned int k_start = kk + tid4 * 32u;
        if (gn < N && k_start < K) {
            const unsigned char* bp_ptr =
                B_packed + (unsigned long long)gn * half_K + (k_start >> 1);
            asm volatile(
                "ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                : "=r"(bp_out.x), "=r"(bp_out.y), "=r"(bp_out.z), "=r"(bp_out.w)
                : "l"(bp_ptr), "l"(pol_b));
        } else {
            bp_out = make_uint4(0u, 0u, 0u, 0u);
        }
    };

    auto load_b_scale32 = [&](unsigned int kk, unsigned int& bs_out) {
        const unsigned int k_sc = kk + tid4 * 64u;
        if (gn < N && k_sc < K) {
            const unsigned char* bs_ptr =
                B_scale + (unsigned long long)gn * num_groups + (k_sc >> 4);
            if (scale_aligned4) {
                bs_out = __ldg(reinterpret_cast<const unsigned int*>(bs_ptr));
            } else {
                const unsigned int lo =
                    static_cast<unsigned int>(__ldg(reinterpret_cast<const unsigned short*>(bs_ptr)));
                const unsigned int hi = (k_sc + 32u < K)
                    ? static_cast<unsigned int>(
                          __ldg(reinterpret_cast<const unsigned short*>(bs_ptr + 2)))
                    : 0u;
                bs_out = lo | (hi << 16);
            }
        } else {
            bs_out = 0u;
        }
    };

    auto mma_step = [&](unsigned int cur_buf, unsigned int k0, unsigned int b0, unsigned int b1) {
        #pragma unroll
        for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
            if (cta_m + mf * 16u < M) {
                unsigned int a0, a1, a2, a3;
                ldmatrix_x4(
                    a0, a1, a2, a3,
                    &smem.a[cur_buf][mf * 16u + a_row][k0 + a_koff]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc[mf][0]), "=f"(acc[mf][1]), "=f"(acc[mf][2]), "=f"(acc[mf][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc[mf][0]), "f"(acc[mf][1]), "f"(acc[mf][2]), "f"(acc[mf][3]));
            }
        }
    };

    if constexpr (DOUBLE_BUF_A_128) {
        issue_a(0u, 0u);
        uint4 bp0, bp1;
        unsigned int bs_cur;
        load_b_packed(0u, bp0);
        load_b_packed(128u, bp1);
        load_b_scale32(0u, bs_cur);

        for (unsigned int k_base = 0u; k_base < K; k_base += 256u) {
            cp_async_wait_group<0>();
            __syncthreads();

            const bool has_half1 = (k_base + 128u < K);
            if (has_half1) {
                issue_a(1u, k_base + 128u);
            }
            uint4 bp_nxt0 = make_uint4(0u, 0u, 0u, 0u);
            uint4 bp_nxt1 = make_uint4(0u, 0u, 0u, 0u);
            unsigned int bs_nxt = 0u;
            if (k_base + 256u < K) {
                load_b_packed(k_base + 256u, bp_nxt0);
                load_b_scale32(k_base + 256u, bs_nxt);
            }

            #pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const unsigned int sb_word = __shfl_sync(0xffffffffu, bs_cur, pair, 4);
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int src = pair * 2 + sub;
                    if (src == 0 || k_base + static_cast<unsigned int>(src) * 32u < K) {
                        const unsigned int sb_pair =
                            (sub == 0) ? (sb_word & 0xFFFFu) : (sb_word >> 16);
                        {
                            const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp0.x, src, 4);
                            const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp0.y, src, 4);
                            unsigned int b0, b1;
                            dequant_nvfp4_pair_to_bf16x2_noscale2(
                                (pk_lo >> byte_shift) & 0xFFu,
                                (pk_hi >> byte_shift) & 0xFFu,
                                sb_pair & 0xFFu,
                                b0,
                                b1);
                            mma_step(0u, static_cast<unsigned int>(src * 32), b0, b1);
                        }
                        {
                            const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp0.z, src, 4);
                            const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp0.w, src, 4);
                            unsigned int b0, b1;
                            dequant_nvfp4_pair_to_bf16x2_noscale2(
                                (pk_lo >> byte_shift) & 0xFFu,
                                (pk_hi >> byte_shift) & 0xFFu,
                                sb_pair >> 8,
                                b0,
                                b1);
                            mma_step(0u, static_cast<unsigned int>(src * 32 + 16), b0, b1);
                        }
                    }
                }
            }

            if (has_half1) {
                cp_async_wait_group<0>();
                __syncthreads();

                if (k_base + 256u < K) {
                    issue_a(0u, k_base + 256u);
                }
                if (k_base + 384u < K) {
                    load_b_packed(k_base + 384u, bp_nxt1);
                }

                const unsigned int k_h1 = k_base + 128u;
                #pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const unsigned int sb_word = __shfl_sync(0xffffffffu, bs_cur, 2 + pair, 4);
                    #pragma unroll
                    for (int sub = 0; sub < 2; ++sub) {
                        const int src = pair * 2 + sub;
                        if (src == 0 || k_h1 + static_cast<unsigned int>(src) * 32u < K) {
                            const unsigned int sb_pair =
                                (sub == 0) ? (sb_word & 0xFFFFu) : (sb_word >> 16);
                            {
                                const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp1.x, src, 4);
                                const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp1.y, src, 4);
                                unsigned int b0, b1;
                                dequant_nvfp4_pair_to_bf16x2_noscale2(
                                    (pk_lo >> byte_shift) & 0xFFu,
                                    (pk_hi >> byte_shift) & 0xFFu,
                                    sb_pair & 0xFFu,
                                    b0,
                                    b1);
                                mma_step(1u, static_cast<unsigned int>(src * 32), b0, b1);
                            }
                            {
                                const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp1.z, src, 4);
                                const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp1.w, src, 4);
                                unsigned int b0, b1;
                                dequant_nvfp4_pair_to_bf16x2_noscale2(
                                    (pk_lo >> byte_shift) & 0xFFu,
                                    (pk_hi >> byte_shift) & 0xFFu,
                                    sb_pair >> 8,
                                    b0,
                                    b1);
                                mma_step(1u, static_cast<unsigned int>(src * 32 + 16), b0, b1);
                            }
                        }
                    }
                }
            }

            bp0 = bp_nxt0;
            bp1 = bp_nxt1;
            bs_cur = bs_nxt;
        }
    } else {
        auto compute_quarter = [&](
            unsigned int cur_buf,
            unsigned int k_q_base,
            const uint4& bp,
            unsigned int bs_word_src,
            int sc_lane,
            int src_base
        ) {
            const unsigned int sb_word = __shfl_sync(0xffffffffu, bs_word_src, sc_lane, 4);
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int src = src_base + sub;
                if (sub == 0 || k_q_base + 32u < K) {
                    const unsigned int sb_pair =
                        (sub == 0) ? (sb_word & 0xFFFFu) : (sb_word >> 16);
                    {
                        const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp.x, src, 4);
                        const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp.y, src, 4);
                        unsigned int b0, b1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pk_lo >> byte_shift) & 0xFFu,
                            (pk_hi >> byte_shift) & 0xFFu,
                            sb_pair & 0xFFu,
                            b0,
                            b1);
                        mma_step(cur_buf, static_cast<unsigned int>(sub * 32), b0, b1);
                    }
                    {
                        const unsigned int pk_lo = __shfl_sync(0xffffffffu, bp.z, src, 4);
                        const unsigned int pk_hi = __shfl_sync(0xffffffffu, bp.w, src, 4);
                        unsigned int b0, b1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pk_lo >> byte_shift) & 0xFFu,
                            (pk_hi >> byte_shift) & 0xFFu,
                            sb_pair >> 8,
                            b0,
                            b1);
                        mma_step(cur_buf, static_cast<unsigned int>(sub * 32 + 16), b0, b1);
                    }
                }
            }
        };

        issue_a(0u, 0u);
        uint4 bp0, bp1;
        unsigned int bs_cur;
        load_b_packed(0u, bp0);
        load_b_packed(128u, bp1);
        load_b_scale32(0u, bs_cur);

        for (unsigned int k_base = 0u; k_base < K; k_base += 256u) {
            cp_async_wait_group<0>();
            __syncthreads();

            if (k_base + 64u < K) {
                issue_a(1u, k_base + 64u);
            }
            uint4 bp_nxt0 = make_uint4(0u, 0u, 0u, 0u);
            uint4 bp_nxt1 = make_uint4(0u, 0u, 0u, 0u);
            unsigned int bs_nxt = 0u;
            if (k_base + 256u < K) {
                load_b_packed(k_base + 256u, bp_nxt0);
                load_b_scale32(k_base + 256u, bs_nxt);
            }

            compute_quarter(0u, k_base, bp0, bs_cur, 0, 0);

            if (k_base + 64u < K) {
                cp_async_wait_group<0>();
                __syncthreads();
                if (k_base + 128u < K) {
                    issue_a(0u, k_base + 128u);
                }
                compute_quarter(1u, k_base + 64u, bp0, bs_cur, 1, 2);
            }

            if (k_base + 128u < K) {
                cp_async_wait_group<0>();
                __syncthreads();
                if (k_base + 192u < K) {
                    issue_a(1u, k_base + 192u);
                }
                if (k_base + 384u < K) {
                    load_b_packed(k_base + 384u, bp_nxt1);
                }
                compute_quarter(0u, k_base + 128u, bp1, bs_cur, 2, 0);
            }

            if (k_base + 192u < K) {
                cp_async_wait_group<0>();
                __syncthreads();
                if (k_base + 256u < K) {
                    issue_a(0u, k_base + 256u);
                }
                compute_quarter(1u, k_base + 192u, bp1, bs_cur, 3, 2);
            }

            bp0 = bp_nxt0;
            bp1 = bp_nxt1;
            bs_cur = bs_nxt;
        }
    }

    __syncthreads();
    const unsigned int smem_col = warp_id * 8u + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.c[r0][smem_col]) =
                __floats2bfloat162_rn(acc[mf][0] * scale2, acc[mf][1] * scale2);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.c[r1][smem_col]) =
                __floats2bfloat162_rn(acc[mf][2] * scale2, acc[mf][3] * scale2);
        }
    }
    __syncthreads();

    const unsigned int rows_in_cta = (M - cta_m) < TILE_M ? (M - cta_m) : TILE_M;
    const unsigned int cols_in_cta = (cta_n + CO_N <= N) ? CO_N : (N - cta_n);
    const unsigned long long c_addr = reinterpret_cast<unsigned long long>(C);
    if (cols_in_cta == CO_N && (N & 7u) == 0u && (c_addr & 15ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U128 = CO_N / 8u;
        constexpr unsigned int ROWS_PER_STEP_U128 = NUM_THREADS / THREADS_PER_ROW_U128;
        const unsigned int row_off = tid / THREADS_PER_ROW_U128;
        const unsigned int col = (tid % THREADS_PER_ROW_U128) * 8u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U128) {
            const uint4 v = *reinterpret_cast<const uint4*>(&smem.c[r][col]);
            st_global_cs_u128(&C[(unsigned long long)(cta_m + r) * N + cta_n + col], v);
        }
    } else if (cols_in_cta == CO_N && (N & 3u) == 0u && (c_addr & 7ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U64 = CO_N / 4u;
        constexpr unsigned int ROWS_PER_STEP_U64 = NUM_THREADS / THREADS_PER_ROW_U64;
        const unsigned int row_off = tid / THREADS_PER_ROW_U64;
        const unsigned int col = (tid % THREADS_PER_ROW_U64) * 4u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U64) {
            const uint2 v = *reinterpret_cast<const uint2*>(&smem.c[r][col]);
            st_global_cs_u64(&C[(unsigned long long)(cta_m + r) * N + cta_n + col], v);
        }
    } else if (cols_in_cta == CO_N && (N & 1u) == 0u && (c_addr & 3ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U32 = CO_N / 2u;
        constexpr unsigned int ROWS_PER_STEP_U32 = NUM_THREADS / THREADS_PER_ROW_U32;
        const unsigned int row_off = tid / THREADS_PER_ROW_U32;
        const unsigned int col = (tid % THREADS_PER_ROW_U32) * 2u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U32) {
            st_global_cs_u32(
                reinterpret_cast<__nv_bfloat162*>(
                    &C[(unsigned long long)(cta_m + r) * N + cta_n + col]),
                *reinterpret_cast<const __nv_bfloat162*>(&smem.c[r][col]));
        }
    } else {
        for (unsigned int r = warp_id; r < rows_in_cta; r += NUM_WARPS) {
            __nv_bfloat16* c_row = C + (unsigned long long)(cta_m + r) * N + cta_n;
            #pragma unroll
            for (unsigned int c = 0u; c < CO_N; c += 32u) {
                if (c + lane < cols_in_cta) {
                    st_global_cs_u16(&c_row[c + lane], smem.c[r][c + lane]);
                }
            }
        }
    }
}

template <unsigned int TILE_M, unsigned int CO_N, bool DOUBLE_BUF_A_128>
__device__ __forceinline__ void w4a16_gemm_coalesced_v2_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16)
        typename W4a16CoalescedV2Smem<TILE_M, CO_N, DOUBLE_BUF_A_128>::Storage smem;
    w4a16_gemm_coalesced_v2_body_impl<TILE_M, CO_N, DOUBLE_BUF_A_128>(
        smem, A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 2)
w4a16_gemm_coalesced_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<32u, 128u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 1)
w4a16_gemm_coalesced_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<64u, 128u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 1)
w4a16_gemm_coalesced_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    union AdaptiveM128Smem {
        W4a16CoalescedV2Smem<80u, 128u, true>::Storage m80;
        W4a16CoalescedV2Smem<128u, 128u, false>::Storage m128;
    };
    __shared__ __align__(16) AdaptiveM128Smem smem;
    static_assert(sizeof(AdaptiveM128Smem) <= 49152u, "static SMEM must stay <= 48 KiB");
    if (blockIdx.y == 0u && M <= 80u) {
        w4a16_gemm_coalesced_v2_body_impl<80u, 128u, true>(
            smem.m80, A, B_packed, B_scale, scale2, C, M, N, K);
        return;
    }
    w4a16_gemm_coalesced_v2_body_impl<128u, 128u, false>(
        smem.m128, A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 4)
w4a16_gemm_coalesced_v2_n32_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<16u, 32u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 4)
w4a16_gemm_coalesced_v2_n64_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<16u, 64u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 4)
w4a16_gemm_coalesced_v2_n32_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<32u, 32u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 4)
w4a16_gemm_coalesced_v2_n64_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<32u, 64u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n64_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<64u, 64u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n64_m80(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<80u, 64u, true>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n64_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_v2_body<128u, 64u, false>(A, B_packed, B_scale, scale2, C, M, N, K);
}

template <
    unsigned int TILE_M,
    unsigned int CO_N,
    unsigned int SPLIT_K,
    unsigned int SUB_K>
__device__ __forceinline__ void w4a16_gemm_coalesced_splitk_v2_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    static_assert(CO_N % 8u == 0u, "CO_N must be a multiple of 8");
    static_assert(SPLIT_K >= 2u, "SPLIT_K must be >= 2");
    static_assert(
        SUB_K == 16u || SUB_K == 32u || SUB_K == 64u || SUB_K == 128u,
        "SUB_K must be 16, 32, 64, or 128");

    constexpr unsigned int N_WARPS = CO_N / 8u;
    constexpr unsigned int NUM_WARPS = N_WARPS * SPLIT_K;
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int K_STEP_CTA = SPLIT_K * 256u;
    constexpr unsigned int NUM_SUBS = 128u / SUB_K;
    constexpr unsigned int MMAS_PER_SUB = SUB_K / 16u;
    constexpr unsigned int A_SLAB_K = SPLIT_K * SUB_K;
    constexpr unsigned int A_STRIDE = A_SLAB_K + 8u;
    constexpr unsigned int RED_STRIDE = CO_N + 4u;

    constexpr unsigned int THREADS_PER_SK = N_WARPS * 32u;
    constexpr unsigned int CHUNKS_PER_SUB = SUB_K / 8u;
    static_assert(
        THREADS_PER_SK % CHUNKS_PER_SUB == 0u,
        "threads per split-K group must be a multiple of A sub-slab chunks");
    constexpr unsigned int A_ROWS_PER_ITER = THREADS_PER_SK / CHUNKS_PER_SUB;
    static_assert(
        TILE_M % A_ROWS_PER_ITER == 0u,
        "TILE_M must be a positive multiple of A_ROWS_PER_ITER");
    constexpr unsigned int A_ITERS = TILE_M / A_ROWS_PER_ITER;
    static_assert(A_ITERS >= 1u, "A_ITERS must be >= 1");
    static_assert(NUM_THREADS % CO_N == 0u, "NUM_THREADS must be a multiple of CO_N");
    static_assert(NUM_THREADS % (CO_N / 2u) == 0u, "NUM_THREADS must be a multiple of CO_N/2");
    static_assert(NUM_THREADS % (CO_N / 4u) == 0u, "NUM_THREADS must be a multiple of CO_N/4");

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_n = blockIdx.x * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_k = warp_id % SPLIT_K;
    const unsigned int warp_n = warp_id / SPLIT_K;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int byte_shift = tid4 * 8u;
    const unsigned int gn = cta_n + warp_n * 8u + group_id;

    union Smem {
        __nv_bfloat16 a[2][TILE_M][A_STRIDE];
        float red[SPLIT_K][TILE_M][RED_STRIDE];
    };
    __shared__ __align__(16) Smem smem;
    static_assert(sizeof(Smem) <= 49152u, "static SMEM must stay <= 48 KiB");

    float acc[M_FRAGS][4];
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        acc[mf][0] = 0.0f;
        acc[mf][1] = 0.0f;
        acc[mf][2] = 0.0f;
        acc[mf][3] = 0.0f;
    }

    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int a_row = lane & 15u;
    const unsigned int a_koff = (lane >> 4) * 8u;

    const unsigned int tid_in_sk = warp_n * 32u + lane;
    const unsigned int a_tid_row = tid_in_sk / CHUNKS_PER_SUB;
    const unsigned int a_tid_sub_col8 = (tid_in_sk % CHUNKS_PER_SUB) * 8u;
    const unsigned int warp_k_smem_base = warp_k * SUB_K;
    const unsigned int a_smem_col = warp_k_smem_base + a_tid_sub_col8;

    unsigned long long pol_b;
    if (gridDim.y == 1u) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_b));
    } else {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1000)
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_b));
#else
        asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(pol_b));
#endif
    }

    auto issue_a = [&](unsigned int buf, unsigned int k_cta, unsigned int half_sub) {
        const unsigned int gk = k_cta + warp_k * 256u + half_sub * SUB_K + a_tid_sub_col8;
        const bool k_valid = (gk < K);
        #pragma unroll
        for (unsigned int p = 0u; p < A_ITERS; ++p) {
            if (cta_m + p * A_ROWS_PER_ITER < M) {
                const unsigned int row = p * A_ROWS_PER_ITER + a_tid_row;
                const unsigned int gr = cta_m + row;
                const bool valid = (gr < M) && k_valid;
                const __nv_bfloat16* src_a =
                    valid ? &A[(unsigned long long)gr * K + gk] : A;
                cp_async_pred_16(&smem.a[buf][row][a_smem_col], src_a, valid);
            }
        }
        cp_async_commit();
    };

    auto load_b_packed = [&](unsigned int k_cta, unsigned int half_off, uint4& bp_out) {
        const unsigned int k_start = k_cta + warp_k * 256u + half_off + tid4 * 32u;
        if (gn < N && k_start < K) {
            const unsigned char* bp_ptr =
                B_packed + (unsigned long long)gn * half_K + (k_start >> 1);
            asm volatile(
                "ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                : "=r"(bp_out.x), "=r"(bp_out.y), "=r"(bp_out.z), "=r"(bp_out.w)
                : "l"(bp_ptr), "l"(pol_b));
        } else {
            bp_out = make_uint4(0u, 0u, 0u, 0u);
        }
    };

    auto load_b_scale32 = [&](unsigned int k_cta, unsigned int& bs_out) {
        const unsigned int k_sc = k_cta + warp_k * 256u + tid4 * 64u;
        if (gn < N && k_sc < K) {
            bs_out = __ldg(reinterpret_cast<const unsigned int*>(
                B_scale + (unsigned long long)gn * num_groups + (k_sc >> 4)));
        } else {
            bs_out = 0u;
        }
    };

    auto mma_step = [&](unsigned int cur_buf, unsigned int k0, unsigned int b0, unsigned int b1) {
        #pragma unroll
        for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
            if (cta_m + mf * 16u < M) {
                unsigned int a0, a1, a2, a3;
                ldmatrix_x4(
                    a0, a1, a2, a3,
                    &smem.a[cur_buf][mf * 16u + a_row][k0 + a_koff]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc[mf][0]), "=f"(acc[mf][1]), "=f"(acc[mf][2]), "=f"(acc[mf][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc[mf][0]), "f"(acc[mf][1]), "f"(acc[mf][2]), "f"(acc[mf][3]));
            }
        }
    };

    unsigned int cur = 0u, nxt = 1u;
    issue_a(0u, 0u, 0u);
    uint4 bp0, bp1;
    unsigned int bs_cur;
    load_b_packed(0u, 0u, bp0);
    load_b_packed(0u, 128u, bp1);
    load_b_scale32(0u, bs_cur);

    for (unsigned int k_cta = 0u; k_cta < K; k_cta += K_STEP_CTA) {
        uint4 bp_nxt0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 bp_nxt1 = make_uint4(0u, 0u, 0u, 0u);
        unsigned int bs_nxt = 0u;
        const unsigned int k_next_cta = k_cta + K_STEP_CTA;
        const unsigned int k_warp = k_cta + warp_k * 256u;
        const bool has_half1 = (k_cta + 128u < K);

        #pragma unroll
        for (unsigned int sub = 0u; sub < NUM_SUBS; ++sub) {
            cp_async_wait_group<0>();
            if constexpr (N_WARPS > 1u) {
                __syncthreads();
            } else {
                __syncwarp();
            }

            if (sub + 1u < NUM_SUBS) {
                issue_a(nxt, k_cta, sub + 1u);
            } else if (has_half1) {
                issue_a(nxt, k_cta, NUM_SUBS);
            }
            if (sub == 0u && k_next_cta < K) {
                load_b_packed(k_next_cta, 0u, bp_nxt0);
                load_b_scale32(k_next_cta, bs_nxt);
            }

            if (k_warp + sub * SUB_K < K) {
                #pragma unroll
                for (unsigned int mi = 0u; mi < MMAS_PER_SUB; ++mi) {
                    const unsigned int k_off = sub * SUB_K + mi * 16u;
                    if (k_warp + k_off < K) {
                        const int src = static_cast<int>(k_off >> 5);
                        const unsigned int hi_half = (k_off >> 4) & 1u;
                        const unsigned int sb_word = __shfl_sync(0xffffffffu, bs_cur, src >> 1, 4);
                        const unsigned int sb_pair =
                            (src & 1) ? (sb_word >> 16) : (sb_word & 0xFFFFu);
                        const unsigned int sb = hi_half ? (sb_pair >> 8) : (sb_pair & 0xFFu);
                        const unsigned int pk_lo = __shfl_sync(
                            0xffffffffu, hi_half ? bp0.z : bp0.x, src, 4);
                        const unsigned int pk_hi = __shfl_sync(
                            0xffffffffu, hi_half ? bp0.w : bp0.y, src, 4);
                        unsigned int b0, b1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pk_lo >> byte_shift) & 0xFFu,
                            (pk_hi >> byte_shift) & 0xFFu,
                            sb,
                            b0,
                            b1);
                        mma_step(cur, warp_k_smem_base + mi * 16u, b0, b1);
                    }
                }
            }
            const unsigned int tmp = cur;
            cur = nxt;
            nxt = tmp;
        }

        if (has_half1) {
            const unsigned int k_warp1 = k_warp + 128u;
            #pragma unroll
            for (unsigned int sub = 0u; sub < NUM_SUBS; ++sub) {
                cp_async_wait_group<0>();
                if constexpr (N_WARPS > 1u) {
                    __syncthreads();
                } else {
                    __syncwarp();
                }

                if (sub + 1u < NUM_SUBS) {
                    issue_a(nxt, k_cta, NUM_SUBS + sub + 1u);
                } else if (k_next_cta < K) {
                    issue_a(nxt, k_next_cta, 0u);
                }
                if (sub == 0u && k_next_cta < K) {
                    load_b_packed(k_next_cta, 128u, bp_nxt1);
                }

                if (k_warp1 + sub * SUB_K < K) {
                    #pragma unroll
                    for (unsigned int mi = 0u; mi < MMAS_PER_SUB; ++mi) {
                        const unsigned int k_off = sub * SUB_K + mi * 16u;
                        if (k_warp1 + k_off < K) {
                            const int src = static_cast<int>(k_off >> 5);
                            const unsigned int hi_half = (k_off >> 4) & 1u;
                            const unsigned int sb_word =
                                __shfl_sync(0xffffffffu, bs_cur, 2 + (src >> 1), 4);
                            const unsigned int sb_pair =
                                (src & 1) ? (sb_word >> 16) : (sb_word & 0xFFFFu);
                            const unsigned int sb = hi_half ? (sb_pair >> 8) : (sb_pair & 0xFFu);
                            const unsigned int pk_lo = __shfl_sync(
                                0xffffffffu, hi_half ? bp1.z : bp1.x, src, 4);
                            const unsigned int pk_hi = __shfl_sync(
                                0xffffffffu, hi_half ? bp1.w : bp1.y, src, 4);
                            unsigned int b0, b1;
                            dequant_nvfp4_pair_to_bf16x2_noscale2(
                                (pk_lo >> byte_shift) & 0xFFu,
                                (pk_hi >> byte_shift) & 0xFFu,
                                sb,
                                b0,
                                b1);
                            mma_step(cur, warp_k_smem_base + mi * 16u, b0, b1);
                        }
                    }
                }
                const unsigned int tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
        }

        bp0 = bp_nxt0;
        bp1 = bp_nxt1;
        bs_cur = bs_nxt;
    }

    __syncthreads();
    const unsigned int smem_col = warp_n * 8u + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<float2*>(&smem.red[warp_k][r0][smem_col]) =
                make_float2(acc[mf][0], acc[mf][1]);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<float2*>(&smem.red[warp_k][r1][smem_col]) =
                make_float2(acc[mf][2], acc[mf][3]);
        }
    }
    __syncthreads();

    const unsigned int rows_in_cta = (M - cta_m) < TILE_M ? (M - cta_m) : TILE_M;
    const unsigned int cols_in_cta = (cta_n + CO_N <= N) ? CO_N : (N - cta_n);
    const unsigned long long c_addr = reinterpret_cast<unsigned long long>(C);
    if (cols_in_cta == CO_N && (N & 3u) == 0u && (c_addr & 7ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U64 = CO_N / 4u;
        constexpr unsigned int ROWS_PER_STEP_U64 = NUM_THREADS / THREADS_PER_ROW_U64;
        const unsigned int row_off = tid / THREADS_PER_ROW_U64;
        const unsigned int col = (tid % THREADS_PER_ROW_U64) * 4u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U64) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            #pragma unroll
            for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                const float2 lo = *reinterpret_cast<const float2*>(&smem.red[sk][r][col]);
                const float2 hi = *reinterpret_cast<const float2*>(&smem.red[sk][r][col + 2u]);
                s0 += lo.x;
                s1 += lo.y;
                s2 += hi.x;
                s3 += hi.y;
            }
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            const __nv_bfloat162 p1 = __floats2bfloat162_rn(s2 * scale2, s3 * scale2);
            uint2 v;
            v.x = *reinterpret_cast<const unsigned int*>(&p0);
            v.y = *reinterpret_cast<const unsigned int*>(&p1);
            st_global_cs_u64(&C[(unsigned long long)(cta_m + r) * N + cta_n + col], v);
        }
    } else if (cols_in_cta == CO_N && (N & 1u) == 0u && (c_addr & 3ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U32 = CO_N / 2u;
        constexpr unsigned int ROWS_PER_STEP_U32 = NUM_THREADS / THREADS_PER_ROW_U32;
        const unsigned int row_off = tid / THREADS_PER_ROW_U32;
        const unsigned int col = (tid % THREADS_PER_ROW_U32) * 2u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U32) {
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll
            for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                const float2 lo = *reinterpret_cast<const float2*>(&smem.red[sk][r][col]);
                s0 += lo.x;
                s1 += lo.y;
            }
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            st_global_cs_u32(
                reinterpret_cast<__nv_bfloat162*>(
                    &C[(unsigned long long)(cta_m + r) * N + cta_n + col]),
                p0);
        }
    } else {
        constexpr unsigned int ROWS_PER_STEP_U16 = NUM_THREADS / CO_N;
        const unsigned int row_off = tid / CO_N;
        const unsigned int col = tid % CO_N;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U16) {
            if (col < cols_in_cta) {
                float s = 0.0f;
                #pragma unroll
                for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                    s += smem.red[sk][r][col];
                }
                const __nv_bfloat16 out_v = __float2bfloat16(s * scale2);
                st_global_cs_u16(
                    &C[(unsigned long long)(cta_m + r) * N + cta_n + col],
                    out_v);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 3)
w4a16_gemm_coalesced_v2_n16_sk4_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_body<16u, 16u, 4u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 2)
w4a16_gemm_coalesced_v2_n32_sk4_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_body<16u, 32u, 4u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 2)
w4a16_gemm_coalesced_v2_n32_sk4_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_body<32u, 32u, 4u, 32u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n32_sk2_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_body<64u, 32u, 2u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n32_sk2_m80(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_body<80u, 32u, 2u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n16_sk4_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_body<32u, 16u, 4u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n16_sk4_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_body<64u, 16u, 4u, 32u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused Coalesced Warp-Split-K BF16xBF16 MMA Gate+Up+SiLU*Mul (SM120/SM103)
//
// Reads A[TILE_M, K] ONCE into SMEM (shared by both gate & up MMAs via single
// `ldmatrix_x4`), streams row-major coalesced 128-bit NVFP4 `Bg` and `Bu` with
// `L2::evict_first` + 128-K half-pipelining (zero register spills, <= 18.5 KiB
// SMEM), reduces `SPLIT_K` FP32 partials in SMEM, and writes `silu(g)*u` in BF16.
// Eliminates M_TILE=32 waste, eliminates FP8 activation clipping, and replaces
// 3 launches (gate + up + silu_mul) with 1 bandwidth-saturated launch.
// ═══════════════════════════════════════════════════════════════════════════
template <
    unsigned int TILE_M,
    unsigned int CO_N,
    unsigned int SPLIT_K,
    unsigned int SUB_K>
__device__ __forceinline__ void w4a16_gemm_coalesced_splitk_v2_gateup_silu_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    static_assert(CO_N % 8u == 0u, "CO_N must be a multiple of 8");
    static_assert(SPLIT_K >= 2u, "SPLIT_K must be >= 2");
    static_assert(
        SUB_K == 16u || SUB_K == 32u || SUB_K == 64u || SUB_K == 128u,
        "SUB_K must be 16, 32, 64, or 128");

    constexpr unsigned int N_WARPS = CO_N / 8u;
    constexpr unsigned int NUM_WARPS = N_WARPS * SPLIT_K;
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int K_STEP_CTA = SPLIT_K * 256u;
    constexpr unsigned int NUM_SUBS = 128u / SUB_K;
    constexpr unsigned int MMAS_PER_SUB = SUB_K / 16u;
    constexpr unsigned int A_SLAB_K = SPLIT_K * SUB_K;
    constexpr unsigned int A_STRIDE = A_SLAB_K + 8u;
    constexpr unsigned int RED_STRIDE = CO_N + 4u;

    constexpr unsigned int THREADS_PER_SK = N_WARPS * 32u;
    constexpr unsigned int CHUNKS_PER_SUB = SUB_K / 8u;
    static_assert(THREADS_PER_SK % CHUNKS_PER_SUB == 0u, "threads/sk must divide chunks");
    constexpr unsigned int A_ROWS_PER_ITER = THREADS_PER_SK / CHUNKS_PER_SUB;
    static_assert(TILE_M % A_ROWS_PER_ITER == 0u, "TILE_M must be multiple of A_ROWS_PER_ITER");
    constexpr unsigned int A_ITERS = TILE_M / A_ROWS_PER_ITER;
    static_assert(NUM_THREADS % CO_N == 0u, "NUM_THREADS must divide CO_N");
    static_assert(NUM_THREADS % (CO_N / 2u) == 0u, "NUM_THREADS must divide CO_N/2");
    static_assert(NUM_THREADS % (CO_N / 4u) == 0u, "NUM_THREADS must divide CO_N/4");

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_n = blockIdx.x * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_k = warp_id % SPLIT_K;
    const unsigned int warp_n = warp_id / SPLIT_K;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int byte_shift = tid4 * 8u;
    const unsigned int gn = cta_n + warp_n * 8u + group_id;

    union Smem {
        __nv_bfloat16 a[2][TILE_M][A_STRIDE];
        float red[2][SPLIT_K][TILE_M][RED_STRIDE];
    };
    __shared__ __align__(16) Smem smem;
    static_assert(sizeof(Smem) <= 49152u, "static SMEM must stay <= 48 KiB");

    float acc_g[M_FRAGS][4];
    float acc_u[M_FRAGS][4];
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        acc_g[mf][0] = 0.0f; acc_g[mf][1] = 0.0f; acc_g[mf][2] = 0.0f; acc_g[mf][3] = 0.0f;
        acc_u[mf][0] = 0.0f; acc_u[mf][1] = 0.0f; acc_u[mf][2] = 0.0f; acc_u[mf][3] = 0.0f;
    }

    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int a_row = lane & 15u;
    const unsigned int a_koff = (lane >> 4) * 8u;

    const unsigned int tid_in_sk = warp_n * 32u + lane;
    const unsigned int a_tid_row = tid_in_sk / CHUNKS_PER_SUB;
    const unsigned int a_tid_sub_col8 = (tid_in_sk % CHUNKS_PER_SUB) * 8u;
    const unsigned int warp_k_smem_base = warp_k * SUB_K;
    const unsigned int a_smem_col = warp_k_smem_base + a_tid_sub_col8;

    unsigned long long pol_b;
    if (gridDim.y == 1u) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_b));
    } else {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1000)
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_b));
#else
        asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(pol_b));
#endif
    }

    auto issue_a = [&](unsigned int buf, unsigned int k_cta, unsigned int half_sub) {
        const unsigned int gk = k_cta + warp_k * 256u + half_sub * SUB_K + a_tid_sub_col8;
        const bool k_valid = (gk < K);
        #pragma unroll
        for (unsigned int p = 0u; p < A_ITERS; ++p) {
            if (cta_m + p * A_ROWS_PER_ITER < M) {
                const unsigned int row = p * A_ROWS_PER_ITER + a_tid_row;
                const unsigned int gr = cta_m + row;
                const bool valid = (gr < M) && k_valid;
                const __nv_bfloat16* src_a =
                    valid ? &A[(unsigned long long)gr * K + gk] : A;
                cp_async_pred_16(&smem.a[buf][row][a_smem_col], src_a, valid);
            }
        }
        cp_async_commit();
    };

    auto load_b_packed = [&](const unsigned char* Bp, unsigned int k_cta, unsigned int half_off, uint4& bp_out) {
        const unsigned int k_start = k_cta + warp_k * 256u + half_off + tid4 * 32u;
        if (gn < N && k_start < K) {
            const unsigned char* bp_ptr =
                Bp + (unsigned long long)gn * half_K + (k_start >> 1);
            asm volatile(
                "ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                : "=r"(bp_out.x), "=r"(bp_out.y), "=r"(bp_out.z), "=r"(bp_out.w)
                : "l"(bp_ptr), "l"(pol_b));
        } else {
            bp_out = make_uint4(0u, 0u, 0u, 0u);
        }
    };

    auto load_b_scale32 = [&](const unsigned char* Bs, unsigned int k_cta, unsigned int& bs_out) {
        const unsigned int k_sc = k_cta + warp_k * 256u + tid4 * 64u;
        if (gn < N && k_sc < K) {
            bs_out = __ldg(reinterpret_cast<const unsigned int*>(
                Bs + (unsigned long long)gn * num_groups + (k_sc >> 4)));
        } else {
            bs_out = 0u;
        }
    };

    auto mma_step_dual = [&](unsigned int cur_buf, unsigned int k0,
                             unsigned int bg0, unsigned int bg1,
                             unsigned int bu0, unsigned int bu1) {
        #pragma unroll
        for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
            if (cta_m + mf * 16u < M) {
                unsigned int a0, a1, a2, a3;
                ldmatrix_x4(
                    a0, a1, a2, a3,
                    &smem.a[cur_buf][mf * 16u + a_row][k0 + a_koff]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc_g[mf][0]), "=f"(acc_g[mf][1]), "=f"(acc_g[mf][2]), "=f"(acc_g[mf][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bg0), "r"(bg1),
                      "f"(acc_g[mf][0]), "f"(acc_g[mf][1]), "f"(acc_g[mf][2]), "f"(acc_g[mf][3]));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc_u[mf][0]), "=f"(acc_u[mf][1]), "=f"(acc_u[mf][2]), "=f"(acc_u[mf][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bu0), "r"(bu1),
                      "f"(acc_u[mf][0]), "f"(acc_u[mf][1]), "f"(acc_u[mf][2]), "f"(acc_u[mf][3]));
            }
        }
    };

    unsigned int cur = 0u, nxt = 1u;
    issue_a(0u, 0u, 0u);
    uint4 bpg_cur, bpu_cur;
    unsigned int bsg_cur, bsu_cur;
    load_b_packed(Bg_packed, 0u, 0u, bpg_cur);
    load_b_packed(Bu_packed, 0u, 0u, bpu_cur);
    load_b_scale32(Bg_scale, 0u, bsg_cur);
    load_b_scale32(Bu_scale, 0u, bsu_cur);

    for (unsigned int k_cta = 0u; k_cta < K; k_cta += K_STEP_CTA) {
        uint4 bpg_nxt = make_uint4(0u, 0u, 0u, 0u);
        uint4 bpu_nxt = make_uint4(0u, 0u, 0u, 0u);
        unsigned int bsg_nxt = 0u;
        unsigned int bsu_nxt = 0u;
        const unsigned int k_next_cta = k_cta + K_STEP_CTA;
        const unsigned int k_warp = k_cta + warp_k * 256u;
        const bool has_half1 = (k_cta + 128u < K);

        #pragma unroll
        for (unsigned int sub = 0u; sub < NUM_SUBS; ++sub) {
            cp_async_wait_group<0>();
            if constexpr (N_WARPS > 1u) {
                __syncthreads();
            } else {
                __syncwarp();
            }

            if (sub + 1u < NUM_SUBS) {
                issue_a(nxt, k_cta, sub + 1u);
            } else if (has_half1) {
                issue_a(nxt, k_cta, NUM_SUBS);
            }
            if (sub == 0u && has_half1) {
                load_b_packed(Bg_packed, k_cta, 128u, bpg_nxt);
                load_b_packed(Bu_packed, k_cta, 128u, bpu_nxt);
            }

            if (k_warp + sub * SUB_K < K) {
                #pragma unroll
                for (unsigned int mi = 0u; mi < MMAS_PER_SUB; ++mi) {
                    const unsigned int k_off = sub * SUB_K + mi * 16u;
                    if (k_warp + k_off < K) {
                        const int src = static_cast<int>(k_off >> 5);
                        const unsigned int hi_half = (k_off >> 4) & 1u;

                        const unsigned int sbg_word = __shfl_sync(0xffffffffu, bsg_cur, src >> 1, 4);
                        const unsigned int sbg_pair =
                            (src & 1) ? (sbg_word >> 16) : (sbg_word & 0xFFFFu);
                        const unsigned int sbg = hi_half ? (sbg_pair >> 8) : (sbg_pair & 0xFFu);
                        const unsigned int pkg_lo = __shfl_sync(
                            0xffffffffu, hi_half ? bpg_cur.z : bpg_cur.x, src, 4);
                        const unsigned int pkg_hi = __shfl_sync(
                            0xffffffffu, hi_half ? bpg_cur.w : bpg_cur.y, src, 4);
                        unsigned int bg0, bg1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pkg_lo >> byte_shift) & 0xFFu,
                            (pkg_hi >> byte_shift) & 0xFFu,
                            sbg, bg0, bg1);

                        const unsigned int sbu_word = __shfl_sync(0xffffffffu, bsu_cur, src >> 1, 4);
                        const unsigned int sbu_pair =
                            (src & 1) ? (sbu_word >> 16) : (sbu_word & 0xFFFFu);
                        const unsigned int sbu = hi_half ? (sbu_pair >> 8) : (sbu_pair & 0xFFu);
                        const unsigned int pku_lo = __shfl_sync(
                            0xffffffffu, hi_half ? bpu_cur.z : bpu_cur.x, src, 4);
                        const unsigned int pku_hi = __shfl_sync(
                            0xffffffffu, hi_half ? bpu_cur.w : bpu_cur.y, src, 4);
                        unsigned int bu0, bu1;
                        dequant_nvfp4_pair_to_bf16x2_noscale2(
                            (pku_lo >> byte_shift) & 0xFFu,
                            (pku_hi >> byte_shift) & 0xFFu,
                            sbu, bu0, bu1);

                        mma_step_dual(cur, warp_k_smem_base + mi * 16u, bg0, bg1, bu0, bu1);
                    }
                }
            }
            const unsigned int tmp = cur;
            cur = nxt;
            nxt = tmp;
        }

        if (has_half1) {
            bpg_cur = bpg_nxt;
            bpu_cur = bpu_nxt;
            const unsigned int k_warp1 = k_warp + 128u;
            #pragma unroll
            for (unsigned int sub = 0u; sub < NUM_SUBS; ++sub) {
                cp_async_wait_group<0>();
                if constexpr (N_WARPS > 1u) {
                    __syncthreads();
                } else {
                    __syncwarp();
                }

                if (sub + 1u < NUM_SUBS) {
                    issue_a(nxt, k_cta, NUM_SUBS + sub + 1u);
                } else if (k_next_cta < K) {
                    issue_a(nxt, k_next_cta, 0u);
                }
                if (sub == 0u && k_next_cta < K) {
                    load_b_packed(Bg_packed, k_next_cta, 0u, bpg_nxt);
                    load_b_packed(Bu_packed, k_next_cta, 0u, bpu_nxt);
                    load_b_scale32(Bg_scale, k_next_cta, bsg_nxt);
                    load_b_scale32(Bu_scale, k_next_cta, bsu_nxt);
                }

                if (k_warp1 + sub * SUB_K < K) {
                    #pragma unroll
                    for (unsigned int mi = 0u; mi < MMAS_PER_SUB; ++mi) {
                        const unsigned int k_off = sub * SUB_K + mi * 16u;
                        if (k_warp1 + k_off < K) {
                            const int src = static_cast<int>(k_off >> 5);
                            const unsigned int hi_half = (k_off >> 4) & 1u;

                            const unsigned int sbg_word =
                                __shfl_sync(0xffffffffu, bsg_cur, 2 + (src >> 1), 4);
                            const unsigned int sbg_pair =
                                (src & 1) ? (sbg_word >> 16) : (sbg_word & 0xFFFFu);
                            const unsigned int sbg = hi_half ? (sbg_pair >> 8) : (sbg_pair & 0xFFu);
                            const unsigned int pkg_lo = __shfl_sync(
                                0xffffffffu, hi_half ? bpg_cur.z : bpg_cur.x, src, 4);
                            const unsigned int pkg_hi = __shfl_sync(
                                0xffffffffu, hi_half ? bpg_cur.w : bpg_cur.y, src, 4);
                            unsigned int bg0, bg1;
                            dequant_nvfp4_pair_to_bf16x2_noscale2(
                                (pkg_lo >> byte_shift) & 0xFFu,
                                (pkg_hi >> byte_shift) & 0xFFu,
                                sbg, bg0, bg1);

                            const unsigned int sbu_word =
                                __shfl_sync(0xffffffffu, bsu_cur, 2 + (src >> 1), 4);
                            const unsigned int sbu_pair =
                                (src & 1) ? (sbu_word >> 16) : (sbu_word & 0xFFFFu);
                            const unsigned int sbu = hi_half ? (sbu_pair >> 8) : (sbu_pair & 0xFFu);
                            const unsigned int pku_lo = __shfl_sync(
                                0xffffffffu, hi_half ? bpu_cur.z : bpu_cur.x, src, 4);
                            const unsigned int pku_hi = __shfl_sync(
                                0xffffffffu, hi_half ? bpu_cur.w : bpu_cur.y, src, 4);
                            unsigned int bu0, bu1;
                            dequant_nvfp4_pair_to_bf16x2_noscale2(
                                (pku_lo >> byte_shift) & 0xFFu,
                                (pku_hi >> byte_shift) & 0xFFu,
                                sbu, bu0, bu1);

                            mma_step_dual(cur, warp_k_smem_base + mi * 16u, bg0, bg1, bu0, bu1);
                        }
                    }
                }
                const unsigned int tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
            bpg_cur = bpg_nxt;
            bpu_cur = bpu_nxt;
            bsg_cur = bsg_nxt;
            bsu_cur = bsu_nxt;
        }
    }

    __syncthreads();
    const unsigned int smem_col = warp_n * 8u + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<float2*>(&smem.red[0][warp_k][r0][smem_col]) =
                make_float2(acc_g[mf][0], acc_g[mf][1]);
            *reinterpret_cast<float2*>(&smem.red[1][warp_k][r0][smem_col]) =
                make_float2(acc_u[mf][0], acc_u[mf][1]);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<float2*>(&smem.red[0][warp_k][r1][smem_col]) =
                make_float2(acc_g[mf][2], acc_g[mf][3]);
            *reinterpret_cast<float2*>(&smem.red[1][warp_k][r1][smem_col]) =
                make_float2(acc_u[mf][2], acc_u[mf][3]);
        }
    }
    __syncthreads();

    auto silu_mul_f = [&](float sg, float su) -> float {
        const float g = sg * scale2_g;
        const float u = su * scale2_u;
        return (g / (1.0f + __expf(-g))) * u;
    };

    const unsigned int rows_in_cta = (M - cta_m) < TILE_M ? (M - cta_m) : TILE_M;
    const unsigned int cols_in_cta = (cta_n + CO_N <= N) ? CO_N : (N - cta_n);
    const unsigned long long c_addr = reinterpret_cast<unsigned long long>(C);
    if (cols_in_cta == CO_N && (N & 3u) == 0u && (c_addr & 7ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U64 = CO_N / 4u;
        constexpr unsigned int ROWS_PER_STEP_U64 = NUM_THREADS / THREADS_PER_ROW_U64;
        const unsigned int row_off = tid / THREADS_PER_ROW_U64;
        const unsigned int col = (tid % THREADS_PER_ROW_U64) * 4u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U64) {
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
            #pragma unroll
            for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                const float2 glo = *reinterpret_cast<const float2*>(&smem.red[0][sk][r][col]);
                const float2 ghi = *reinterpret_cast<const float2*>(&smem.red[0][sk][r][col + 2u]);
                const float2 ulo = *reinterpret_cast<const float2*>(&smem.red[1][sk][r][col]);
                const float2 uhi = *reinterpret_cast<const float2*>(&smem.red[1][sk][r][col + 2u]);
                g0 += glo.x; g1 += glo.y; g2 += ghi.x; g3 += ghi.y;
                u0 += ulo.x; u1 += ulo.y; u2 += uhi.x; u3 += uhi.y;
            }
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(silu_mul_f(g0, u0), silu_mul_f(g1, u1));
            const __nv_bfloat162 p1 = __floats2bfloat162_rn(silu_mul_f(g2, u2), silu_mul_f(g3, u3));
            uint2 v;
            v.x = *reinterpret_cast<const unsigned int*>(&p0);
            v.y = *reinterpret_cast<const unsigned int*>(&p1);
            st_global_cs_u64(&C[(unsigned long long)(cta_m + r) * N + cta_n + col], v);
        }
    } else if (cols_in_cta == CO_N && (N & 1u) == 0u && (c_addr & 3ull) == 0ull) {
        constexpr unsigned int THREADS_PER_ROW_U32 = CO_N / 2u;
        constexpr unsigned int ROWS_PER_STEP_U32 = NUM_THREADS / THREADS_PER_ROW_U32;
        const unsigned int row_off = tid / THREADS_PER_ROW_U32;
        const unsigned int col = (tid % THREADS_PER_ROW_U32) * 2u;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U32) {
            float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
            #pragma unroll
            for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                const float2 glo = *reinterpret_cast<const float2*>(&smem.red[0][sk][r][col]);
                const float2 ulo = *reinterpret_cast<const float2*>(&smem.red[1][sk][r][col]);
                g0 += glo.x; g1 += glo.y;
                u0 += ulo.x; u1 += ulo.y;
            }
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(silu_mul_f(g0, u0), silu_mul_f(g1, u1));
            st_global_cs_u32(
                reinterpret_cast<__nv_bfloat162*>(
                    &C[(unsigned long long)(cta_m + r) * N + cta_n + col]),
                p0);
        }
    } else {
        constexpr unsigned int ROWS_PER_STEP_U16 = NUM_THREADS / CO_N;
        const unsigned int row_off = tid / CO_N;
        const unsigned int col = tid % CO_N;
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U16) {
            if (col < cols_in_cta) {
                float sg = 0.0f, su = 0.0f;
                #pragma unroll
                for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                    sg += smem.red[0][sk][r][col];
                    su += smem.red[1][sk][r][col];
                }
                const __nv_bfloat16 out_v = __float2bfloat16(silu_mul_f(sg, su));
                st_global_cs_u16(
                    &C[(unsigned long long)(cta_m + r) * N + cta_n + col],
                    out_v);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 3)
w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_gateup_silu_body<16u, 16u, 4u, 64u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 2)
w4a16_gemm_coalesced_v2_n32_sk4_m16_gateup_silu(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_gateup_silu_body<16u, 32u, 4u, 64u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n16_sk4_m32_gateup_silu(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_gateup_silu_body<32u, 16u, 4u, 64u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n16_sk4_m64_gateup_silu(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_gateup_silu_body<64u, 16u, 4u, 32u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_v2_n32_sk2_m64_gateup_silu(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bg_packed,
    const unsigned char* __restrict__ Bg_scale,
    const float scale2_g,
    const unsigned char* __restrict__ Bu_packed,
    const unsigned char* __restrict__ Bu_scale,
    const float scale2_u,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemm_coalesced_splitk_v2_gateup_silu_body<64u, 32u, 2u, 64u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}
