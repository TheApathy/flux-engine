// SPDX-License-Identifier: AGPL-3.0-only

// Use the generic GB10 W4A16 module that also exports the FP8 staging helpers
// required by the shared MoE constructor.  No dimensions are compiled into
// these kernels; Flash-Next supplies its 2560/640 geometry at launch time.
#include "../../nvfp4/w4a16_gemm.cu"

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

#define K_STEP_P 32          // pipeline stage width (2x parent K_STEP=16)
#define PAD_P 8              // (32+8)*2 = 80B rows -> 16-byte aligned for cp.async
#define BP_P_STRIDE 16       // 16B per-column packed slots (32 K-rows) -> 16-byte aligned
#define A_P_CHUNKS (M_TILE * K_STEP_P / 8 / 128)  // 16B chunks per thread (2)

// Byte-exact shadow of w4a16_gemm.
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
    const unsigned int a_stride = K_STEP_P + PAD_P;  // 40
    const unsigned int b_stride = N_TILE_SM + PAD;   // 66

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    #define ISSUE_P(buf, kk) do {                                                   \
        _Pragma("unroll")                                                           \
        for (int p = 0; p < A_P_CHUNKS; p++) {                                      \
            unsigned int c = threadIdx.x * A_P_CHUNKS + p;                          \
            unsigned int row = c / (K_STEP_P / 8);                                  \
            unsigned int col8 = (c % (K_STEP_P / 8)) << 3;                          \
            unsigned int gr = cta_m + row;                                          \
            cp_async_pred_16(&smem_A[(buf)][row][col8],                             \
                &A[(unsigned long long)gr * K + (kk) + col8], ((gr) < M) && ((kk) < K)); \
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
        cp_async_wait_group<1>();
        __syncthreads();

        DEQUANT_P(cur, k_base);
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

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
    #undef ISSUE_P
    #undef DEQUANT_P
}
#undef K_STEP_P
#undef PAD_P
#undef BP_P_STRIDE
#undef A_P_CHUNKS

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

// ═══════════════════════════════════════════════════════════════════════════
// Coalesced-K single-sweep and intra-CTA Warp-Split-K row-major NVFP4
// Tensor-Core GEMM (M <= 128).
//
// Eliminates the strided 16B/row uncoalesced B loads of w4a16_gemm_pipe*:
//   - Non-split-K tiers (`CO_N in {128, 64, 32, 16}`, 1 n8 tile per warp):
//     zero duplicate weight loads across warps, full smem_A reuse across warps,
//     zero smem_B, zero smem_LUT.
//   - Intra-CTA Warp-Split-K tiers (`n8_sk8_m16`, `n16_sk4_m16`, `n32_sk4_m16`,
//     `n64_sk4_m16`, `n32_sk4_m32`, `n64_sk2_m64`):
//     * Saturate all 48 SMs on narrow/medium projections (`hc.down N=320`,
//       `v_router N=512`, `ssm.out`/`attn.oproj N=2560`) with zero global
//       atomics, zero scratch, deterministic FP32 SMEM reduction, and full
//       CUDA-graph capture safety.
//     * Each warp loads full 64B (`uint4`, 128 K-elements) weight bursts with
//       warp-coalesced A sub-slabs (`N_WARPS == 1` uses barrier-free
//       `__syncwarp()` in the K-loop).
//   - Native BF16x2 register dequant (`prmt` + `mul.rn.bf16x2`) computes
//     `E2M1 * E4M3` bit-exact in BF16 (<= 7 significant bits) and hoists
//     `scale2` once to the FP32 epilogue (`acc * scale2`), halving inner-loop
//     multiply instructions while improving scale2 precision.
//   - Epilogue reuses smem as `smem.c` / `smem.red` for 100% coalesced
//     streaming stores (`st.global.cs`) that stay alignment-safe for arbitrary
//     or odd widths (e.g. 248077).
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

__device__ __forceinline__ void cp_async_cg_pred_16(
    void* dst_smem,
    const void* src_gmem,
    bool pred
) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16u : 0u;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
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

// Note: `w4a16_gemm_coalesced_m{32,64,128}` are defined below after
// `w4a16_gemm_coalesced_v2_body` so they inherit 2-stage K=256 prefetch,
// `st.global.cs.v4.u32` (16B) streaming stores, and double-buffered `M<=80`
// acceleration in-place while preserving exact launch geometry (`CO_N=128, blk=512`).

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

// ═══════════════════════════════════════════════════════════════════════════
// Intra-CTA Warp-Split-K coalesced NVFP4 Tensor-Core GEMM.
//
// Each CTA partitions K across `SPLIT_K` warp groups (each owning `128`
// contiguous K-elements per CTA step via `uint4` 64B weight bursts) and
// reduces partials deterministically in FP32 inside `smem.red` before applying
// `scale2` once and streaming BF16 output with `st.global.cs`.
// When `CO_N == 8` (`N_WARPS == 1`), each warp exclusively writes and reads
// its own `smem.a` column window, eliminating `__syncthreads()` in the K-loop.
// ═══════════════════════════════════════════════════════════════════════════
// Exact 0-ULP twin of `qwen4_hc_silu_div`: `bf16(x / (1 + expf(-x)))` where
// `x = bf16_to_f32(raw) / silu_divisor` on the already BF16-rounded GEMM output.
__device__ __forceinline__ __nv_bfloat16 w4a16_silu_div_bf16(
    const __nv_bfloat16 raw,
    const float silu_divisor
) {
    const float x = __bfloat162float(raw) / silu_divisor;
    return __float2bfloat16(x / (1.0f + expf(-x)));
}

__device__ __forceinline__ __nv_bfloat162 w4a16_silu_div_bf16x2(
    const __nv_bfloat162 raw,
    const float silu_divisor
) {
    const __nv_bfloat16 lo = w4a16_silu_div_bf16(__low2bfloat16(raw), silu_divisor);
    const __nv_bfloat16 hi = w4a16_silu_div_bf16(__high2bfloat16(raw), silu_divisor);
    return __halves2bfloat162(lo, hi);
}

template <
    unsigned int TILE_M,
    unsigned int CO_N,
    unsigned int SPLIT_K,
    unsigned int SUB_K,
    bool SILU_DIV = false>
__device__ __forceinline__ void w4a16_gemm_coalesced_splitk_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    const float silu_divisor = 1.0f,
    unsigned int bx = blockIdx.x,
    bool apply_silu = SILU_DIV
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
    const unsigned int cta_n = bx * CO_N;
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
            __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            __nv_bfloat162 p1 = __floats2bfloat162_rn(s2 * scale2, s3 * scale2);
            if constexpr (SILU_DIV) {
                if (apply_silu) {
                    p0 = w4a16_silu_div_bf16x2(p0, silu_divisor);
                    p1 = w4a16_silu_div_bf16x2(p1, silu_divisor);
                }
            }
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
            __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            if constexpr (SILU_DIV) {
                if (apply_silu) {
                    p0 = w4a16_silu_div_bf16x2(p0, silu_divisor);
                }
            }
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
                __nv_bfloat16 out_v = __float2bfloat16(s * scale2);
                if constexpr (SILU_DIV) {
                    if (apply_silu) {
                        out_v = w4a16_silu_div_bf16(out_v, silu_divisor);
                    }
                }
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

extern "C" __global__ void __launch_bounds__(256, 3)
w4a16_gemm_coalesced_n8_sk8_m16_silu_div(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    const float silu_divisor
) {
    w4a16_gemm_coalesced_splitk_body<16u, 8u, 8u, 32u, true>(
        A, B_packed, B_scale, scale2, C, M, N, K, silu_divisor);
}

extern "C" __global__ void __launch_bounds__(256, 4)
w4a16_gemm_coalesced_n16_sk4_m16_silu_div(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    const float silu_divisor
) {
    w4a16_gemm_coalesced_splitk_body<16u, 16u, 4u, 64u, true>(
        A, B_packed, B_scale, scale2, C, M, N, K, silu_divisor);
}

extern "C" __global__ void __launch_bounds__(512, 3)
w4a16_gemm_coalesced_n32_sk4_m16_silu_div(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    const float silu_divisor
) {
    w4a16_gemm_coalesced_splitk_body<16u, 32u, 4u, 64u, true>(
        A, B_packed, B_scale, scale2, C, M, N, K, silu_divisor);
}

extern "C" __global__ void __launch_bounds__(1024, 1)
w4a16_gemm_coalesced_n64_sk4_m16_silu_div(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    const float silu_divisor
) {
    w4a16_gemm_coalesced_splitk_body<16u, 64u, 4u, 128u, true>(
        A, B_packed, B_scale, scale2, C, M, N, K, silu_divisor);
}

// Single-sweep dual `hc.down + silu_div` (`blockIdx.x < div_ceil(N1, 32)`) +
// `hc.inject` (`blockIdx.x >= div_ceil(N1, 32)`, no `silu_div`) using the fast
// v1 `CO_N = 32, SPLIT_K = 4, SUB_K = 64, TILE_M = 16` K-loop (`blk = 512`,
// `16,896 B` SMEM):
//   `grid = [div_ceil(N1, 32) + div_ceil(N2, 32), div_ceil(M, 16), 1]`
//   (= `[11, div_ceil(M, 16), 1]` for `N1 = 320, N2 = 4`, or `[10, ...]` when `N2 = 0`).
// Bounds activation `X` rereads to 11 N-blocks (1 single wave at `M = 25..64`).
extern "C" __global__ void __launch_bounds__(512, 3)
w4a16_gemm_coalesced_n32_sk4_m16_dual_silu_div(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N2,
    unsigned int M,
    unsigned int K,
    const float silu_divisor
) {
    const unsigned int blocks1 = (N1 + 31u) >> 5;
    const bool is_b1 = (blockIdx.x < blocks1);
    const unsigned int bx = is_b1 ? blockIdx.x : (blockIdx.x - blocks1);
    w4a16_gemm_coalesced_splitk_body<16u, 32u, 4u, 64u, true>(
        A,
        is_b1 ? B1_packed : B2_packed,
        is_b1 ? B1_scale : B2_scale,
        is_b1 ? scale2_1 : scale2_2,
        is_b1 ? C1 : C2,
        M,
        is_b1 ? N1 : N2,
        K,
        silu_divisor,
        bx,
        is_b1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Coalesced NVFP4 W4A16 GEMM v2 (`ATLAS_QWEN4_COALESCED_V2 != 0`):
//   - 2-stage K-prefetch (`K_STEP = 256` per warp: `bp0, bp1` + `uint32`
//     4-byte `B_scale` load): full 128B cache-line bursts, halves scale sector
//     overfetch, doubles in-flight bytes per warp to satisfy GB10 Little's Law.
//   - Double-buffered `TILE_M = 80` (`M_FRAGS = 5`, `43,520 B <= 48 KiB`):
//     eliminates `DOUBLE_BUF_A_128=false` across the entire `C=16 K=5` (`M<=80`)
//     envelope while keeping 2 CTAs/SM residency (`85 KiB <= 100 KiB`).
//   - Shape-matched 2-CTA/SM tiles (`blk = 256`):
//       * `CO_N = 64` for wide `N > 4096` (`ssm_qkvz`: 256 CTAs, 2 CTAs/SM)
//       * `CO_N = 32, SPLIT_K = 2` for `N = 2560, K = 6144` (80 CTAs, 0 idle SMs)
//       * `TILE_M = 16` (`n32_m16`, `n32_sk4_m16`, `n64_m16`) for `M = 9..16`
//   - Fused single-sweep `hc.down + silu_div + hc.inject` (`n8_sk8_m32_dual_silu_div`,
//     0 `__syncthreads()` in the `K=10240` loop, 33 KiB SMEM, 2 CTAs/SM).
//   - Fused `hc.up + mix_save_inject` (`qwen4_hc_upmix_coalesced_m16/m32/m64/m80`,
//     16 hidden cols x 4 streams per CTA, eliminates `mix_logits` staging).
// Precondition for all v2 / 2-stage prefetch entrypoints: `K % 64 == 0` and
// `B_scale` is 4-byte aligned (guaranteed for all canonical Flash-Next shapes).
// ═══════════════════════════════════════════════════════════════════════════

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
    unsigned int SUB_K,
    bool SILU_DIV = false>
__device__ __forceinline__ void w4a16_gemm_coalesced_splitk_v2_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int bx,
    bool apply_silu = SILU_DIV,
    const float silu_divisor = 1.0f
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
    const unsigned int cta_n = bx * CO_N;
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
            __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            __nv_bfloat162 p1 = __floats2bfloat162_rn(s2 * scale2, s3 * scale2);
            if constexpr (SILU_DIV) {
                if (apply_silu) {
                    p0 = w4a16_silu_div_bf16x2(p0, silu_divisor);
                    p1 = w4a16_silu_div_bf16x2(p1, silu_divisor);
                }
            }
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
            __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            if constexpr (SILU_DIV) {
                if (apply_silu) {
                    p0 = w4a16_silu_div_bf16x2(p0, silu_divisor);
                }
            }
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
                __nv_bfloat16 out_v = __float2bfloat16(s * scale2);
                if constexpr (SILU_DIV) {
                    if (apply_silu) {
                        out_v = w4a16_silu_div_bf16(out_v, silu_divisor);
                    }
                }
                st_global_cs_u16(
                    &C[(unsigned long long)(cta_m + r) * N + cta_n + col],
                    out_v);
            }
        }
    }
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
    w4a16_gemm_coalesced_splitk_v2_body<16u, 32u, 4u, 64u, false>(
        A, B_packed, B_scale, scale2, C, M, N, K, blockIdx.x);
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
    w4a16_gemm_coalesced_splitk_v2_body<32u, 32u, 4u, 32u, false>(
        A, B_packed, B_scale, scale2, C, M, N, K, blockIdx.x);
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
    w4a16_gemm_coalesced_splitk_v2_body<64u, 32u, 2u, 64u, false>(
        A, B_packed, B_scale, scale2, C, M, N, K, blockIdx.x);
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
    w4a16_gemm_coalesced_splitk_v2_body<80u, 32u, 2u, 64u, false>(
        A, B_packed, B_scale, scale2, C, M, N, K, blockIdx.x);
}

// Single-sweep dual `hc.down + silu_div` (`blockIdx.x < div_ceil(N1, 8)`) +
// `hc.inject` (`blockIdx.x >= div_ceil(N1, 8)`, no `silu_div`):
//   CO_N = 8, SPLIT_K = 8, SUB_K = 32, TILE_M = 32 (`blk = 256`, `N_WARPS = 1`
//   -> zero `__syncthreads()` in the entire K-loop, 33,792 B SMEM -> 2 CTAs/SM).
//   `grid = [div_ceil(N1, 8) + div_ceil(N2, 8), div_ceil(M, 32), 1]`
//   (= `[41, div_ceil(M, 32), 1]` for `N1 = 320, N2 = 4`, or `[40, ...]` when `N2 = 0`).
extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N2,
    unsigned int M,
    unsigned int K,
    const float silu_divisor
) {
    const unsigned int blocks1 = (N1 + 7u) >> 3;
    const bool is_b1 = (blockIdx.x < blocks1);
    const unsigned int bx = is_b1 ? blockIdx.x : (blockIdx.x - blocks1);
    w4a16_gemm_coalesced_splitk_v2_body<32u, 8u, 8u, 32u, true>(
        A,
        is_b1 ? B1_packed : B2_packed,
        is_b1 ? B1_scale : B2_scale,
        is_b1 ? scale2_1 : scale2_2,
        is_b1 ? C1 : C2,
        M,
        is_b1 ? N1 : N2,
        K,
        bx,
        is_b1,
        silu_divisor);
}

// Single-sweep v2 dual `hc.down + silu_div` (`blockIdx.x < div_ceil(N1, 16)`) +
// `hc.inject` (`blockIdx.x >= div_ceil(N1, 16)`, no `silu_div`) with 2-stage
// K-prefetch (`K_STEP = 256` per warp + `uint32` 4-byte `B_scale` loads):
//   CO_N = 16, SPLIT_K = 4, SUB_K = 64, TILE_M = 16 (`blk = 256`, `N_WARPS = 2`
//   x `SPLIT_K = 4` = `8` warps, `<= 48` regs, `16,896 B` SMEM -> `5 CTAs/SM`).
//   `grid = [div_ceil(N1, 16) + div_ceil(N2, 16), div_ceil(M, 16), 1]`
//   (= `[21, 3, 1]` = `63 CTAs` at `M = 40`, `[21, 5, 1]` = `105 CTAs` at
//   `M = 80` for `N1 = 320, N2 = 4`, saturating all 48 GB10 SMs).
extern "C" __global__ void __launch_bounds__(256, 5)
w4a16_gemm_coalesced_n16_sk4_m16_dual_silu_div_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N2,
    unsigned int M,
    unsigned int K,
    const float silu_divisor
) {
    const unsigned int blocks1 = (N1 + 15u) >> 4;
    const bool is_b1 = (blockIdx.x < blocks1);
    const unsigned int bx = is_b1 ? blockIdx.x : (blockIdx.x - blocks1);
    w4a16_gemm_coalesced_splitk_v2_body<16u, 16u, 4u, 64u, true>(
        A,
        is_b1 ? B1_packed : B2_packed,
        is_b1 ? B1_scale : B2_scale,
        is_b1 ? scale2_1 : scale2_2,
        is_b1 ? C1 : C2,
        M,
        is_b1 ? N1 : N2,
        K,
        bx,
        is_b1,
        silu_divisor);
}

// ═══════════════════════════════════════════════════════════════════════════
// High-Occupancy Grid-Split-K Dual `hc.down + silu_div + hc.inject` Kernels
// (SM120/SM103 188-SM verify saturation for `M = 25..=256`):
//   - Supports BOTH `gridDim.z == 1` (zero-scratch direct BF16 epilogue,
//     0-ULP vs intra-CTA baseline) AND `gridDim.z > 1` (inter-CTA K-splitting
//     with L2-pinned `float4` partials and self-resetting deterministic
//     last-CTA reduction — 1 launch, 0 extra memset/reduce kernels, 100%
//     CUDA-graph replay safe).
//   - Cuts `down` weight re-reads by 2x-4x (`TILE_M = 32/64`) and `residual`
//     L2 activation re-reads by up to 6.8x (`CO_N = 16/32/64`) while filling
//     all 188 SMs across `M = 32..=256`.
// ═══════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void st_global_f4_hint(
    float* ptr,
    float4 val,
    unsigned long long pol
) {
    asm volatile(
        "st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;\n"
        :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w), "l"(pol)
        : "memory");
}

__device__ __forceinline__ float4 ld_global_cg_f4_hint(
    const float* ptr,
    unsigned long long pol
) {
    float4 val;
    asm volatile(
        "ld.global.cg.L2::cache_hint.v4.f32 {%0, %1, %2, %3}, [%4], %5;\n"
        : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        : "l"(ptr), "l"(pol));
    return val;
}

template <
    unsigned int TILE_M,
    unsigned int CO_N,
    unsigned int SPLIT_K,
    unsigned int SUB_K,
    bool SILU_DIV = true>
__device__ __forceinline__ void w4a16_gemm_coalesced_splitk_grid_v2_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int bx,
    bool apply_silu,
    const float silu_divisor,
    float* __restrict__ splitk_partials,
    unsigned int* __restrict__ splitk_counters
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    static_assert(CO_N % 8u == 0u, "CO_N must be a multiple of 8");
    static_assert(SPLIT_K >= 1u, "SPLIT_K must be >= 1");
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
    const unsigned int cta_n = bx * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_k = warp_id % SPLIT_K;
    const unsigned int warp_n = warp_id / SPLIT_K;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int byte_shift = tid4 * 8u;
    const unsigned int warp_n_col = cta_n + warp_n * 8u;
    const bool warp_n_active = (warp_n_col < N);
    const unsigned int gn = warp_n_col + group_id;

    union Smem {
        __nv_bfloat16 a[2][TILE_M][A_STRIDE];
        struct {
            float red[SPLIT_K][TILE_M][RED_STRIDE];
            unsigned int is_last;
        } ep;
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

    unsigned int k_begin = 0u;
    unsigned int k_end = K;
    if (gridDim.z > 1u) {
        if ((K & 255u) == 0u) {
            const unsigned int units = K >> 8;
            k_begin = ((units * blockIdx.z) / gridDim.z) << 8;
            k_end = ((units * (blockIdx.z + 1u)) / gridDim.z) << 8;
        } else {
            const unsigned int units = K >> 6;
            k_begin = ((units * blockIdx.z) / gridDim.z) << 6;
            k_end = ((units * (blockIdx.z + 1u)) / gridDim.z) << 6;
        }
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
        const bool k_valid = (gk < k_end);
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
        if (gn < N && k_start < k_end) {
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
        if (gn < N && k_sc < k_end) {
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

    if (k_begin < k_end) {
        unsigned int cur = 0u, nxt = 1u;
        issue_a(0u, k_begin, 0u);
        uint4 bp0, bp1;
        unsigned int bs_cur;
        load_b_packed(k_begin, 0u, bp0);
        load_b_packed(k_begin, 128u, bp1);
        load_b_scale32(k_begin, bs_cur);

        for (unsigned int k_cta = k_begin; k_cta < k_end; k_cta += K_STEP_CTA) {
            uint4 bp_nxt0 = make_uint4(0u, 0u, 0u, 0u);
            uint4 bp_nxt1 = make_uint4(0u, 0u, 0u, 0u);
            unsigned int bs_nxt = 0u;
            const unsigned int k_next_cta = k_cta + K_STEP_CTA;
            const unsigned int k_warp = k_cta + warp_k * 256u;
            const bool has_half1 = (k_cta + 128u < k_end);

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
                if (sub == 0u && k_next_cta < k_end) {
                    load_b_packed(k_next_cta, 0u, bp_nxt0);
                    load_b_scale32(k_next_cta, bs_nxt);
                }

                if (warp_n_active && k_warp + sub * SUB_K < k_end) {
                    #pragma unroll
                    for (unsigned int mi = 0u; mi < MMAS_PER_SUB; ++mi) {
                        const unsigned int k_off = sub * SUB_K + mi * 16u;
                        if (k_warp + k_off < k_end) {
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
                    } else if (k_next_cta < k_end) {
                        issue_a(nxt, k_next_cta, 0u);
                    }
                    if (sub == 0u && k_next_cta < k_end) {
                        load_b_packed(k_next_cta, 128u, bp_nxt1);
                    }

                    if (warp_n_active && k_warp1 + sub * SUB_K < k_end) {
                        #pragma unroll
                        for (unsigned int mi = 0u; mi < MMAS_PER_SUB; ++mi) {
                            const unsigned int k_off = sub * SUB_K + mi * 16u;
                            if (k_warp1 + k_off < k_end) {
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
    }

    __syncthreads();
    const unsigned int smem_col = warp_n * 8u + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<float2*>(&smem.ep.red[warp_k][r0][smem_col]) =
                make_float2(acc[mf][0], acc[mf][1]);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<float2*>(&smem.ep.red[warp_k][r1][smem_col]) =
                make_float2(acc[mf][2], acc[mf][3]);
        }
    }
    __syncthreads();

    const unsigned int rows_in_cta = (M - cta_m) < TILE_M ? (M - cta_m) : TILE_M;
    const unsigned int cols_in_cta = (cta_n + CO_N <= N) ? CO_N : (N - cta_n);
    const unsigned long long c_addr = reinterpret_cast<unsigned long long>(C);

    constexpr unsigned int THREADS_PER_ROW_U64 = CO_N / 4u;
    constexpr unsigned int ROWS_PER_STEP_U64 = NUM_THREADS / THREADS_PER_ROW_U64;
    const unsigned int row_off = tid / THREADS_PER_ROW_U64;
    const unsigned int col = (tid % THREADS_PER_ROW_U64) * 4u;

    if (gridDim.z == 1u) {
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U64) {
            if (col >= cols_in_cta) continue;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            #pragma unroll
            for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                const float2 lo = *reinterpret_cast<const float2*>(&smem.ep.red[sk][r][col]);
                const float2 hi = *reinterpret_cast<const float2*>(&smem.ep.red[sk][r][col + 2u]);
                s0 += lo.x;
                s1 += lo.y;
                s2 += hi.x;
                s3 += hi.y;
            }
            if (col + 4u <= cols_in_cta && (N & 3u) == 0u && (c_addr & 7ull) == 0ull) {
                __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
                __nv_bfloat162 p1 = __floats2bfloat162_rn(s2 * scale2, s3 * scale2);
                if constexpr (SILU_DIV) {
                    if (apply_silu) {
                        p0 = w4a16_silu_div_bf16x2(p0, silu_divisor);
                        p1 = w4a16_silu_div_bf16x2(p1, silu_divisor);
                    }
                }
                uint2 v;
                v.x = *reinterpret_cast<const unsigned int*>(&p0);
                v.y = *reinterpret_cast<const unsigned int*>(&p1);
                st_global_cs_u64(&C[(unsigned long long)(cta_m + r) * N + cta_n + col], v);
            } else {
                const float vals[4] = {s0, s1, s2, s3};
                #pragma unroll
                for (unsigned int j = 0u; j < 4u; ++j) {
                    if (col + j < cols_in_cta) {
                        __nv_bfloat16 out_v = __float2bfloat16(vals[j] * scale2);
                        if constexpr (SILU_DIV) {
                            if (apply_silu) {
                                out_v = w4a16_silu_div_bf16(out_v, silu_divisor);
                            }
                        }
                        st_global_cs_u16(
                            &C[(unsigned long long)(cta_m + r) * N + cta_n + col + j],
                            out_v);
                    }
                }
            }
        }
        return;
    }

    // Inter-CTA Split-K path (`gridDim.z > 1`):
    // Store L2-pinned float4 partials, fence + atomic semaphore, self-reset counter on last CTA.
    const unsigned int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int num_tiles = gridDim.x * gridDim.y;
    constexpr unsigned int TILE_ELEMS = TILE_M * CO_N;
    const unsigned long long tile_stride = (unsigned long long)num_tiles * TILE_ELEMS;
    float* const base_partial = splitk_partials + (unsigned long long)tile_idx * TILE_ELEMS;
    float* const my_partial = base_partial + (unsigned long long)blockIdx.z * tile_stride;

    unsigned long long pol_el, pol_ef;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_el));
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_ef));

    for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U64) {
        if (col >= cols_in_cta) continue;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        #pragma unroll
        for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
            const float2 lo = *reinterpret_cast<const float2*>(&smem.ep.red[sk][r][col]);
            const float2 hi = *reinterpret_cast<const float2*>(&smem.ep.red[sk][r][col + 2u]);
            s0 += lo.x;
            s1 += lo.y;
            s2 += hi.x;
            s3 += hi.y;
        }
        st_global_f4_hint(my_partial + r * CO_N + col, make_float4(s0, s1, s2, s3), pol_el);
    }

    __threadfence();
    __syncthreads();
    if (tid == 0u) {
        const unsigned int prev = atomicAdd(&splitk_counters[tile_idx], 1u);
        const unsigned int last = (prev == gridDim.z - 1u) ? 1u : 0u;
        smem.ep.is_last = last;
        if (last) {
            splitk_counters[tile_idx] = 0u;
        }
    }
    __syncthreads();
    if (!smem.ep.is_last) return;

    const unsigned int gz = gridDim.z;
    for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP_U64) {
        if (col >= cols_in_cta) continue;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (unsigned int z = 0u; z < gz; ++z) {
            const float4 p = ld_global_cg_f4_hint(
                base_partial + (unsigned long long)z * tile_stride + r * CO_N + col, pol_ef);
            s0 += p.x;
            s1 += p.y;
            s2 += p.z;
            s3 += p.w;
        }
        if (col + 4u <= cols_in_cta && (N & 3u) == 0u && (c_addr & 7ull) == 0ull) {
            __nv_bfloat162 p0 = __floats2bfloat162_rn(s0 * scale2, s1 * scale2);
            __nv_bfloat162 p1 = __floats2bfloat162_rn(s2 * scale2, s3 * scale2);
            if constexpr (SILU_DIV) {
                if (apply_silu) {
                    p0 = w4a16_silu_div_bf16x2(p0, silu_divisor);
                    p1 = w4a16_silu_div_bf16x2(p1, silu_divisor);
                }
            }
            uint2 v;
            v.x = *reinterpret_cast<const unsigned int*>(&p0);
            v.y = *reinterpret_cast<const unsigned int*>(&p1);
            st_global_cs_u64(&C[(unsigned long long)(cta_m + r) * N + cta_n + col], v);
        } else {
            const float vals[4] = {s0, s1, s2, s3};
            #pragma unroll
            for (unsigned int j = 0u; j < 4u; ++j) {
                if (col + j < cols_in_cta) {
                    __nv_bfloat16 out_v = __float2bfloat16(vals[j] * scale2);
                    if constexpr (SILU_DIV) {
                        if (apply_silu) {
                            out_v = w4a16_silu_div_bf16(out_v, silu_divisor);
                        }
                    }
                    st_global_cs_u16(
                        &C[(unsigned long long)(cta_m + r) * N + cta_n + col + j],
                        out_v);
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N2,
    unsigned int M,
    unsigned int K,
    const float silu_divisor,
    float* __restrict__ splitk_partials,
    unsigned int* __restrict__ splitk_counters
) {
    const unsigned int blocks1 = (N1 + 7u) >> 3;
    const bool is_b1 = (blockIdx.x < blocks1);
    const unsigned int bx = is_b1 ? blockIdx.x : (blockIdx.x - blocks1);
    w4a16_gemm_coalesced_splitk_grid_v2_body<32u, 8u, 8u, 32u, true>(
        A,
        is_b1 ? B1_packed : B2_packed,
        is_b1 ? B1_scale : B2_scale,
        is_b1 ? scale2_1 : scale2_2,
        is_b1 ? C1 : C2,
        M,
        is_b1 ? N1 : N2,
        K,
        bx,
        is_b1,
        silu_divisor,
        splitk_partials,
        splitk_counters);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_n16_sk4_m64_dual_silu_div_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N2,
    unsigned int M,
    unsigned int K,
    const float silu_divisor,
    float* __restrict__ splitk_partials,
    unsigned int* __restrict__ splitk_counters
) {
    const unsigned int blocks1 = (N1 + 15u) >> 4;
    const bool is_b1 = (blockIdx.x < blocks1);
    const unsigned int bx = is_b1 ? blockIdx.x : (blockIdx.x - blocks1);
    w4a16_gemm_coalesced_splitk_grid_v2_body<64u, 16u, 4u, 32u, true>(
        A,
        is_b1 ? B1_packed : B2_packed,
        is_b1 ? B1_scale : B2_scale,
        is_b1 ? scale2_1 : scale2_2,
        is_b1 ? C1 : C2,
        M,
        is_b1 ? N1 : N2,
        K,
        bx,
        is_b1,
        silu_divisor,
        splitk_partials,
        splitk_counters);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_n32_sk2_m64_dual_silu_div_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N2,
    unsigned int M,
    unsigned int K,
    const float silu_divisor,
    float* __restrict__ splitk_partials,
    unsigned int* __restrict__ splitk_counters
) {
    const unsigned int blocks1 = (N1 + 31u) >> 5;
    const bool is_b1 = (blockIdx.x < blocks1);
    const unsigned int bx = is_b1 ? blockIdx.x : (blockIdx.x - blocks1);
    w4a16_gemm_coalesced_splitk_grid_v2_body<64u, 32u, 2u, 64u, true>(
        A,
        is_b1 ? B1_packed : B2_packed,
        is_b1 ? B1_scale : B2_scale,
        is_b1 ? scale2_1 : scale2_2,
        is_b1 ? C1 : C2,
        M,
        is_b1 ? N1 : N2,
        K,
        bx,
        is_b1,
        silu_divisor,
        splitk_partials,
        splitk_counters);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_gemm_coalesced_n64_sk1_m64_dual_silu_div_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N2,
    unsigned int M,
    unsigned int K,
    const float silu_divisor,
    float* __restrict__ splitk_partials,
    unsigned int* __restrict__ splitk_counters
) {
    const unsigned int blocks1 = (N1 + 63u) >> 6;
    const bool is_b1 = (blockIdx.x < blocks1);
    const unsigned int bx = is_b1 ? blockIdx.x : (blockIdx.x - blocks1);
    w4a16_gemm_coalesced_splitk_grid_v2_body<64u, 64u, 1u, 128u, true>(
        A,
        is_b1 ? B1_packed : B2_packed,
        is_b1 ? B1_scale : B2_scale,
        is_b1 ? scale2_1 : scale2_2,
        is_b1 ? C1 : C2,
        M,
        is_b1 ? N1 : N2,
        K,
        bx,
        is_b1,
        silu_divisor,
        splitk_partials,
        splitk_counters);
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused Coalesced NVFP4 `hc.up + mix_save_inject` (`M <= 80`, `hc_count == 4`):
//   - Each CTA owns `16` hidden columns across all `4` hyper-connection streams
//     (= `64` rows of `up`, `grid = [hidden_size / 16, div_ceil(M, TILE_M), 1]`,
//     `blk = 256`, `8` warps = `4` streams x `2` col-groups of `8`).
//   - Double-buffered `A` SMEM (`<= 43,520 B` at `TILE_M = 80`) -> 2 CTAs/SM.
//   - Epilogue rounds `mix_logits` to BF16 in SMEM (`160 B/row`), reads L2-hot
//     `residual[m, s * hidden_size + h]` once per stream in exact `s = 0..3` order,
//     writes `mixed[m, h]` (skipping `mix_logits` DRAM traffic completely), and in
//     `blockIdx.x == gridDim.x - 1` (after `__syncthreads()`) transforms raw
//     `inject_logits` (`bf16(2 / (1 + exp(-x/4)))`) and saves them into each row's
//     `residual` tail, matching `qwen4_hc_mix_save_inject` bit-for-bit.
// ═══════════════════════════════════════════════════════════════════════════
template <unsigned int TILE_M>
__device__ __forceinline__ void qwen4_hc_upmix_coalesced_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ inject_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int M,
    unsigned int hidden_size,
    unsigned int K
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    constexpr unsigned int NUM_THREADS = 256u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int A_TILE_K = 128u;
    constexpr unsigned int A_STRIDE = A_TILE_K + 8u;
    constexpr unsigned int A_CHUNKS_PER_ROW = A_TILE_K / 8u;
    constexpr unsigned int A_ROWS_PER_ITER = NUM_THREADS / A_CHUNKS_PER_ROW;
    constexpr unsigned int A_ITERS = TILE_M / A_ROWS_PER_ITER;

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_h = blockIdx.x * 16u;
    if (cta_m >= M || cta_h >= hidden_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_s = warp_id >> 1;
    const unsigned int warp_h = (warp_id & 1u) * 8u;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int byte_shift = tid4 * 8u;
    const unsigned int local_h_lane = warp_h + group_id;
    const unsigned int gh_lane = cta_h + local_h_lane;
    const unsigned int gn = warp_s * hidden_size + gh_lane;

    union Smem {
        __nv_bfloat16 a[2][TILE_M][A_STRIDE];
        __nv_bfloat16 g[TILE_M][4][20];
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

    auto load_b_packed = [&](unsigned int kk, uint4& bp_out) {
        const unsigned int k_start = kk + tid4 * 32u;
        if (gh_lane < hidden_size && k_start < K) {
            const unsigned char* bp_ptr =
                up_packed + (unsigned long long)gn * half_K + (k_start >> 1);
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
        if (gh_lane < hidden_size && k_sc < K) {
            bs_out = __ldg(reinterpret_cast<const unsigned int*>(
                up_scale + (unsigned long long)gn * num_groups + (k_sc >> 4)));
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

    __syncthreads();
    const unsigned int smem_h = warp_h + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.g[r0][warp_s][smem_h]) =
                __floats2bfloat162_rn(acc[mf][0] * scale2, acc[mf][1] * scale2);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.g[r1][warp_s][smem_h]) =
                __floats2bfloat162_rn(acc[mf][2] * scale2, acc[mf][3] * scale2);
        }
    }
    __syncthreads();

    const unsigned int rows_in_cta = (M - cta_m) < TILE_M ? (M - cta_m) : TILE_M;
    const unsigned int r_width = hidden_size * 4u;
    const unsigned int local_h = tid & 15u;
    const unsigned int row_off = tid >> 4;
    const unsigned int gh = cta_h + local_h;
    if (gh < hidden_size) {
        for (unsigned int r = row_off; r < rows_in_cta; r += 16u) {
            const unsigned long long row_res_base =
                (unsigned long long)(cta_m + r) * r_width + gh;
            float mix_acc = 0.0f;
            #pragma unroll
            for (unsigned int s = 0u; s < 4u; ++s) {
                const float g0 = __bfloat162float(smem.g[r][s][local_h]);
                const float g = 1.0f / (1.0f + expf(-g0));
                const float rv = __bfloat162float(
                    residual[row_res_base + (unsigned long long)s * hidden_size]);
                mix_acc += g * rv;
            }
            mixed[(unsigned long long)(cta_m + r) * hidden_size + gh] =
                __float2bfloat16(mix_acc / 4.0f);
        }
    }

    if (inject_logits != nullptr && blockIdx.x == gridDim.x - 1u) {
        __syncthreads();
        for (unsigned int t = tid; t < rows_in_cta * 4u; t += NUM_THREADS) {
            const unsigned int r = t >> 2;
            const unsigned int ss = t & 3u;
            const unsigned int row = cta_m + r;
            const unsigned long long idx = (unsigned long long)row * 4ull + ss;
            const float x = __bfloat162float(inject_logits[idx]) / 4.0f;
            const __nv_bfloat16 scale = __float2bfloat16(2.0f / (1.0f + expf(-x)));
            inject_logits[idx] = scale;
            residual[(unsigned long long)row * r_width + (r_width - 4u) + ss] = scale;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
qwen4_hc_upmix_coalesced_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ inject_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int M, unsigned int hidden_size, unsigned int K
) {
    qwen4_hc_upmix_coalesced_body<16u>(
        A, up_packed, up_scale, scale2, residual, inject_logits, mixed, M, hidden_size, K);
}

extern "C" __global__ void __launch_bounds__(256, 4)
qwen4_hc_upmix_coalesced_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ inject_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int M, unsigned int hidden_size, unsigned int K
) {
    qwen4_hc_upmix_coalesced_body<32u>(
        A, up_packed, up_scale, scale2, residual, inject_logits, mixed, M, hidden_size, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
qwen4_hc_upmix_coalesced_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ inject_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int M, unsigned int hidden_size, unsigned int K
) {
    qwen4_hc_upmix_coalesced_body<64u>(
        A, up_packed, up_scale, scale2, residual, inject_logits, mixed, M, hidden_size, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
qwen4_hc_upmix_coalesced_m80(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ inject_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int M, unsigned int hidden_size, unsigned int K
) {
    qwen4_hc_upmix_coalesced_body<80u>(
        A, up_packed, up_scale, scale2, residual, inject_logits, mixed, M, hidden_size, K);
}

// ═══════════════════════════════════════════════════════════════════════════
// High-CTA Transposed NVFP4 W4A16 Shared-Expert Decode Kernels (`rows = 2..128`):
//
// Consumes the unified transposed shared-expert buffers (`shared_gate_t`,
// `shared_up_t`: packed `[K/2, N]`, flat E4M3 scale `[K/16, N]`; `shared_down_t`:
// packed `[K/2, N]`, flat E4M3 scale `[K/16, N]`) directly from BF16 activations
// without MXFP8 quantization or low-CTA `v28_m32` (5-20 CTA) under-occupancy:
//   - Gate+Up + fused `SiLU(gate) * up` (`N = 640, K = 2560`): `CO_N = 16`
//     (40 N-CTAs x M-tiles), intra-CTA `SPLIT_K in {2, 4}`, `cp.async.cg.16`
//     pipelined weight/scale streaming, FP32 deterministic reduction, L2-caching
//     BF16 writeback stores (immediately consumed by `w4a16_shared_down_t_n32_*`).
//   - Down (`N = 2560, K = 640`): `CO_N = 32` (80 N-CTAs, saturates all 48 SMs),
//     128 threads/CTA (2-4 CTAs/SM), `SUB_K in {64, 128}`, L2-caching writeback
//     stores (immediately consumed by the fused MoE blend epilogue).
// Preconditions: 16B alignment on all global pointers, `N % 16 == 0` (gate+up) /
// `N % 32 == 0` (down), `K % 16 == 0`.
// ═══════════════════════════════════════════════════════════════════════════

template <unsigned int TILE_M, unsigned int SPLIT_K, unsigned int SUB_K, unsigned int STAGES>
__device__ __forceinline__ void w4a16_shared_gateup_silu_t_n16_body(
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
    static_assert(SPLIT_K == 2u || SPLIT_K == 4u, "SPLIT_K must be 2 or 4");
    static_assert(SUB_K == 64u, "SUB_K must be 64");
    static_assert(STAGES >= 2u, "STAGES must be >= 2");

    constexpr unsigned int CO_N = 16u;
    constexpr unsigned int N_WARPS = 2u;
    constexpr unsigned int NUM_WARPS = N_WARPS * SPLIT_K;
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int K_STEP_CTA = SPLIT_K * SUB_K;
    constexpr unsigned int A_SLAB_K = SPLIT_K * SUB_K;
    constexpr unsigned int A_STRIDE = A_SLAB_K + 8u;
    constexpr unsigned int THREADS_PER_SK = N_WARPS * 32u;
    constexpr unsigned int CHUNKS_PER_SUB = SUB_K / 8u;
    constexpr unsigned int A_ROWS_PER_ITER = THREADS_PER_SK / CHUNKS_PER_SUB;
    static_assert(
        TILE_M % A_ROWS_PER_ITER == 0u,
        "TILE_M must be a multiple of A_ROWS_PER_ITER");
    constexpr unsigned int A_ITERS = TILE_M / A_ROWS_PER_ITER;
    constexpr unsigned int KP_PER_SUB = SUB_K / 2u;
    constexpr unsigned int SG_PER_SUB = SUB_K / 16u;

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
    const unsigned int local_n = warp_n * 8u + group_id;

    struct StageSmem {
        __nv_bfloat16 a[STAGES][TILE_M][A_STRIDE];
        unsigned char bg[STAGES][SPLIT_K][KP_PER_SUB][CO_N];
        unsigned char bu[STAGES][SPLIT_K][KP_PER_SUB][CO_N];
        unsigned char bgs[STAGES][SPLIT_K][SG_PER_SUB][CO_N];
        unsigned char bus[STAGES][SPLIT_K][SG_PER_SUB][CO_N];
    };
    union Smem {
        StageSmem st;
        float red[SPLIT_K][2][TILE_M][CO_N];
    };
    __shared__ __align__(16) Smem smem;
    static_assert(sizeof(Smem) <= 49152u, "static SMEM must stay <= 48 KiB");

    float acc_g[M_FRAGS][4];
    float acc_u[M_FRAGS][4];
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        acc_g[mf][0] = 0.0f;
        acc_g[mf][1] = 0.0f;
        acc_g[mf][2] = 0.0f;
        acc_g[mf][3] = 0.0f;
        acc_u[mf][0] = 0.0f;
        acc_u[mf][1] = 0.0f;
        acc_u[mf][2] = 0.0f;
        acc_u[mf][3] = 0.0f;
    }

    const unsigned int a_row = lane & 15u;
    const unsigned int a_koff = (lane >> 4) * 8u;
    const unsigned int tid_in_sk = warp_n * 32u + lane;
    const unsigned int a_tid_row = tid_in_sk / CHUNKS_PER_SUB;
    const unsigned int a_tid_col8 = (tid_in_sk % CHUNKS_PER_SUB) * 8u;
    const unsigned int warp_k_smem_base = warp_k * SUB_K;
    const unsigned int a_smem_col = warp_k_smem_base + a_tid_col8;
    const bool n_full = (cta_n + CO_N <= N);

    auto issue_stage = [&](unsigned int buf, unsigned int k_cta) {
        const unsigned int k_warp = k_cta + warp_k * SUB_K;
        const unsigned int gk = k_warp + a_tid_col8;
        const bool k_valid = (gk < K);
        #pragma unroll
        for (unsigned int p = 0u; p < A_ITERS; ++p) {
            if (cta_m + p * A_ROWS_PER_ITER < M) {
                const unsigned int row = p * A_ROWS_PER_ITER + a_tid_row;
                const unsigned int gr = cta_m + row;
                const bool valid = (gr < M) && k_valid;
                const __nv_bfloat16* src_a =
                    valid ? &A[(unsigned long long)gr * K + gk] : A;
                cp_async_pred_16(&smem.st.a[buf][row][a_smem_col], src_a, valid);
            }
        }
        const bool is_up = (tid_in_sk >= 32u);
        const unsigned int kp_loc = tid_in_sk & 31u;
        const bool bp_valid = n_full && (k_warp + (kp_loc << 1) < K);
        const unsigned int kp = (k_warp >> 1) + kp_loc;
        const unsigned char* src_b =
            (is_up ? Bu_packed : Bg_packed) + (unsigned long long)kp * N + cta_n;
        void* dst_b = is_up
            ? static_cast<void*>(&smem.st.bu[buf][warp_k][kp_loc][0])
            : static_cast<void*>(&smem.st.bg[buf][warp_k][kp_loc][0]);
        cp_async_cg_pred_16(dst_b, bp_valid ? src_b : Bg_packed, bp_valid);
        if (tid_in_sk < 8u) {
            const bool sc_up = (tid_in_sk >= 4u);
            const unsigned int sg_loc = tid_in_sk & 3u;
            const bool bs_valid = n_full && (k_warp + (sg_loc << 4) < K);
            const unsigned int sg = (k_warp >> 4) + sg_loc;
            const unsigned char* src_s =
                (sc_up ? Bu_scale : Bg_scale) + (unsigned long long)sg * N + cta_n;
            void* dst_s = sc_up
                ? static_cast<void*>(&smem.st.bus[buf][warp_k][sg_loc][0])
                : static_cast<void*>(&smem.st.bgs[buf][warp_k][sg_loc][0]);
            cp_async_cg_pred_16(dst_s, bs_valid ? src_s : Bg_scale, bs_valid);
        }
        cp_async_commit();
    };

    const unsigned int total_iters = (K + K_STEP_CTA - 1u) / K_STEP_CTA;
    #pragma unroll
    for (unsigned int s = 0u; s < STAGES - 1u; ++s) {
        if (s < total_iters) {
            issue_stage(s, s * K_STEP_CTA);
        }
    }

    for (unsigned int it = 0u; it < total_iters; ++it) {
        const unsigned int cur = it % STAGES;
        const unsigned int fetch_it = it + (STAGES - 1u);
        if (fetch_it < total_iters) {
            issue_stage(fetch_it % STAGES, fetch_it * K_STEP_CTA);
            cp_async_wait_group<STAGES - 1u>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const unsigned int k_warp = it * K_STEP_CTA + warp_k * SUB_K;
        if (k_warp < K) {
            #pragma unroll
            for (unsigned int mi = 0u; mi < SUB_K / 16u; ++mi) {
                if (k_warp + mi * 16u < K) {
                    unsigned int g0, g1, u0, u1;
                    dequant_nvfp4_pair_to_bf16x2_noscale2(
                        smem.st.bg[cur][warp_k][mi * 8u + tid4][local_n],
                        smem.st.bg[cur][warp_k][mi * 8u + 4u + tid4][local_n],
                        smem.st.bgs[cur][warp_k][mi][local_n],
                        g0,
                        g1);
                    dequant_nvfp4_pair_to_bf16x2_noscale2(
                        smem.st.bu[cur][warp_k][mi * 8u + tid4][local_n],
                        smem.st.bu[cur][warp_k][mi * 8u + 4u + tid4][local_n],
                        smem.st.bus[cur][warp_k][mi][local_n],
                        u0,
                        u1);
                    const unsigned int k0_smem = warp_k_smem_base + mi * 16u;
                    #pragma unroll
                    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
                        if (cta_m + mf * 16u < M) {
                            unsigned int a0, a1, a2, a3;
                            ldmatrix_x4(
                                a0, a1, a2, a3,
                                &smem.st.a[cur][mf * 16u + a_row][k0_smem + a_koff]);
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                                : "=f"(acc_g[mf][0]), "=f"(acc_g[mf][1]),
                                  "=f"(acc_g[mf][2]), "=f"(acc_g[mf][3])
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(g0), "r"(g1),
                                  "f"(acc_g[mf][0]), "f"(acc_g[mf][1]),
                                  "f"(acc_g[mf][2]), "f"(acc_g[mf][3]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                                : "=f"(acc_u[mf][0]), "=f"(acc_u[mf][1]),
                                  "=f"(acc_u[mf][2]), "=f"(acc_u[mf][3])
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(u0), "r"(u1),
                                  "f"(acc_u[mf][0]), "f"(acc_u[mf][1]),
                                  "f"(acc_u[mf][2]), "f"(acc_u[mf][3]));
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    const unsigned int smem_col = warp_n * 8u + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<float2*>(&smem.red[warp_k][0][r0][smem_col]) =
                make_float2(acc_g[mf][0], acc_g[mf][1]);
            *reinterpret_cast<float2*>(&smem.red[warp_k][1][r0][smem_col]) =
                make_float2(acc_u[mf][0], acc_u[mf][1]);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<float2*>(&smem.red[warp_k][0][r1][smem_col]) =
                make_float2(acc_g[mf][2], acc_g[mf][3]);
            *reinterpret_cast<float2*>(&smem.red[warp_k][1][r1][smem_col]) =
                make_float2(acc_u[mf][2], acc_u[mf][3]);
        }
    }
    __syncthreads();

    const unsigned int rows_in_cta = (M - cta_m) < TILE_M ? (M - cta_m) : TILE_M;
    constexpr unsigned int THREADS_PER_ROW = CO_N / 4u;
    constexpr unsigned int ROWS_PER_STEP = NUM_THREADS / THREADS_PER_ROW;
    const unsigned int row_off = tid / THREADS_PER_ROW;
    const unsigned int col = (tid % THREADS_PER_ROW) * 4u;
    if (cta_n + col + 4u <= N) {
        for (unsigned int r = row_off; r < rows_in_cta; r += ROWS_PER_STEP) {
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
            #pragma unroll
            for (unsigned int sk = 0u; sk < SPLIT_K; ++sk) {
                const float2 gl = *reinterpret_cast<const float2*>(&smem.red[sk][0][r][col]);
                const float2 gh = *reinterpret_cast<const float2*>(&smem.red[sk][0][r][col + 2u]);
                const float2 ul = *reinterpret_cast<const float2*>(&smem.red[sk][1][r][col]);
                const float2 uh = *reinterpret_cast<const float2*>(&smem.red[sk][1][r][col + 2u]);
                g0 += gl.x;
                g1 += gl.y;
                g2 += gh.x;
                g3 += gh.y;
                u0 += ul.x;
                u1 += ul.y;
                u2 += uh.x;
                u3 += uh.y;
            }
            g0 *= scale2_g;
            g1 *= scale2_g;
            g2 *= scale2_g;
            g3 *= scale2_g;
            u0 *= scale2_u;
            u1 *= scale2_u;
            u2 *= scale2_u;
            u3 *= scale2_u;
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(
                (g0 / (1.0f + expf(-g0))) * u0,
                (g1 / (1.0f + expf(-g1))) * u1);
            const __nv_bfloat162 p1 = __floats2bfloat162_rn(
                (g2 / (1.0f + expf(-g2))) * u2,
                (g3 / (1.0f + expf(-g3))) * u3);
            uint2 v;
            v.x = *reinterpret_cast<const unsigned int*>(&p0);
            v.y = *reinterpret_cast<const unsigned int*>(&p1);
            *reinterpret_cast<uint2*>(
                &C[(unsigned long long)(cta_m + r) * N + cta_n + col]) = v;
        }
    }
}

template <unsigned int TILE_M, unsigned int SUB_K>
__device__ __forceinline__ void w4a16_shared_down_t_n32_body(
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
    static_assert(SUB_K == 64u || SUB_K == 128u, "SUB_K must be 64 or 128");

    constexpr unsigned int CO_N = 32u;
    constexpr unsigned int NUM_THREADS = 128u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int A_STRIDE = SUB_K + 8u;
    constexpr unsigned int CHUNKS_PER_SUB = SUB_K / 8u;
    constexpr unsigned int A_ROWS_PER_ITER = NUM_THREADS / CHUNKS_PER_SUB;
    static_assert(
        TILE_M % A_ROWS_PER_ITER == 0u,
        "TILE_M must be a multiple of A_ROWS_PER_ITER");
    constexpr unsigned int A_ITERS = TILE_M / A_ROWS_PER_ITER;
    constexpr unsigned int KP_PER_SUB = SUB_K / 2u;
    constexpr unsigned int SG_PER_SUB = SUB_K / 16u;
    constexpr unsigned int B_CHUNKS_PER_ROW = CO_N / 16u;
    constexpr unsigned int BP_TOTAL_CHUNKS = KP_PER_SUB * B_CHUNKS_PER_ROW;
    constexpr unsigned int BS_TOTAL_CHUNKS = SG_PER_SUB * B_CHUNKS_PER_ROW;
    constexpr unsigned int C_STRIDE = CO_N + 8u;

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_n = blockIdx.x * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_n = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int local_n = warp_n * 8u + group_id;

    struct StageSmem {
        __nv_bfloat16 a[2][TILE_M][A_STRIDE];
        unsigned char bp[2][KP_PER_SUB][CO_N];
        unsigned char bs[2][SG_PER_SUB][CO_N];
    };
    union Smem {
        StageSmem st;
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

    const unsigned int a_row = lane & 15u;
    const unsigned int a_koff = (lane >> 4) * 8u;
    const unsigned int a_tid_row = tid / CHUNKS_PER_SUB;
    const unsigned int a_tid_col8 = (tid % CHUNKS_PER_SUB) * 8u;

    auto issue_stage = [&](unsigned int buf, unsigned int k_cta) {
        const unsigned int gk = k_cta + a_tid_col8;
        const bool k_valid = (gk < K);
        #pragma unroll
        for (unsigned int p = 0u; p < A_ITERS; ++p) {
            if (cta_m + p * A_ROWS_PER_ITER < M) {
                const unsigned int row = p * A_ROWS_PER_ITER + a_tid_row;
                const unsigned int gr = cta_m + row;
                const bool valid = (gr < M) && k_valid;
                cp_async_pred_16(
                    &smem.st.a[buf][row][a_tid_col8],
                    valid ? &A[(unsigned long long)gr * K + gk] : A,
                    valid);
            }
        }
        if constexpr (BP_TOTAL_CHUNKS == 128u) {
            const unsigned int kp_loc = tid >> 1;
            const unsigned int col16 = (tid & 1u) * 16u;
            const bool bp_valid = (cta_n + col16 + 16u <= N) && (k_cta + (kp_loc << 1) < K);
            const unsigned int kp = (k_cta >> 1) + kp_loc;
            cp_async_cg_pred_16(
                &smem.st.bp[buf][kp_loc][col16],
                bp_valid ? (B_packed + (unsigned long long)kp * N + cta_n + col16) : B_packed,
                bp_valid);
            if (tid < BS_TOTAL_CHUNKS) {
                const unsigned int sg_loc = tid >> 1;
                const unsigned int scol16 = (tid & 1u) * 16u;
                const bool bs_valid = (cta_n + scol16 + 16u <= N) && (k_cta + (sg_loc << 4) < K);
                const unsigned int sg = (k_cta >> 4) + sg_loc;
                cp_async_cg_pred_16(
                    &smem.st.bs[buf][sg_loc][scol16],
                    bs_valid ? (B_scale + (unsigned long long)sg * N + cta_n + scol16) : B_scale,
                    bs_valid);
            }
        } else {
            if (tid < BP_TOTAL_CHUNKS) {
                const unsigned int kp_loc = tid >> 1;
                const unsigned int col16 = (tid & 1u) * 16u;
                const bool bp_valid = (cta_n + col16 + 16u <= N) && (k_cta + (kp_loc << 1) < K);
                const unsigned int kp = (k_cta >> 1) + kp_loc;
                cp_async_cg_pred_16(
                    &smem.st.bp[buf][kp_loc][col16],
                    bp_valid ? (B_packed + (unsigned long long)kp * N + cta_n + col16) : B_packed,
                    bp_valid);
            } else if (tid < BP_TOTAL_CHUNKS + BS_TOTAL_CHUNKS) {
                const unsigned int sid = tid - BP_TOTAL_CHUNKS;
                const unsigned int sg_loc = sid >> 1;
                const unsigned int col16 = (sid & 1u) * 16u;
                const bool bs_valid = (cta_n + col16 + 16u <= N) && (k_cta + (sg_loc << 4) < K);
                const unsigned int sg = (k_cta >> 4) + sg_loc;
                cp_async_cg_pred_16(
                    &smem.st.bs[buf][sg_loc][col16],
                    bs_valid ? (B_scale + (unsigned long long)sg * N + cta_n + col16) : B_scale,
                    bs_valid);
            }
        }
        cp_async_commit();
    };

    unsigned int cur = 0u;
    unsigned int nxt = 1u;
    issue_stage(0u, 0u);
    for (unsigned int k_cta = 0u; k_cta < K; k_cta += SUB_K) {
        if (k_cta + SUB_K < K) {
            issue_stage(nxt, k_cta + SUB_K);
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int mi = 0u; mi < SUB_K / 16u; ++mi) {
            if (k_cta + mi * 16u < K) {
                unsigned int b0, b1;
                dequant_nvfp4_pair_to_bf16x2_noscale2(
                    smem.st.bp[cur][mi * 8u + tid4][local_n],
                    smem.st.bp[cur][mi * 8u + 4u + tid4][local_n],
                    smem.st.bs[cur][mi][local_n],
                    b0,
                    b1);
                #pragma unroll
                for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
                    if (cta_m + mf * 16u < M) {
                        unsigned int a0, a1, a2, a3;
                        ldmatrix_x4(
                            a0, a1, a2, a3,
                            &smem.st.a[cur][mf * 16u + a_row][mi * 16u + a_koff]);
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                            : "=f"(acc[mf][0]), "=f"(acc[mf][1]),
                              "=f"(acc[mf][2]), "=f"(acc[mf][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                              "f"(acc[mf][0]), "f"(acc[mf][1]),
                              "f"(acc[mf][2]), "f"(acc[mf][3]));
                    }
                }
            }
        }
        __syncthreads();
        const unsigned int tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    const unsigned int smem_col = warp_n * 8u + tid4 * 2u;
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
    const unsigned int row_off = tid >> 2;
    const unsigned int col = (tid & 3u) * 8u;
    if (cta_n + col + 8u <= N) {
        for (unsigned int r = row_off; r < rows_in_cta; r += 32u) {
            const uint4 v = *reinterpret_cast<const uint4*>(&smem.c[r][col]);
            *reinterpret_cast<uint4*>(
                &C[(unsigned long long)(cta_m + r) * N + cta_n + col]) = v;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_shared_gateup_silu_t_n16_sk4_m16(
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
    w4a16_shared_gateup_silu_t_n16_body<16u, 4u, 64u, 3u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2)
w4a16_shared_gateup_silu_t_n16_sk4_m32(
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
    w4a16_shared_gateup_silu_t_n16_body<32u, 4u, 64u, 2u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 4)
w4a16_shared_gateup_silu_t_n16_sk2_m32(
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
    w4a16_shared_gateup_silu_t_n16_body<32u, 2u, 64u, 2u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 2)
w4a16_shared_gateup_silu_t_n16_sk2_m64(
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
    w4a16_shared_gateup_silu_t_n16_body<64u, 2u, 64u, 2u>(
        A, Bg_packed, Bg_scale, scale2_g, Bu_packed, Bu_scale, scale2_u, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 4)
w4a16_shared_down_t_n32_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_shared_down_t_n32_body<16u, 128u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 4)
w4a16_shared_down_t_n32_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_shared_down_t_n32_body<32u, 128u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 2)
w4a16_shared_down_t_n32_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_shared_down_t_n32_body<64u, 128u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 2)
w4a16_shared_down_t_n32_m80(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_shared_down_t_n32_body<80u, 128u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 2)
w4a16_shared_down_t_n32_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_shared_down_t_n32_body<128u, 64u>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// =============================================================================
// Coalesced-K single-sweep BF16 Tensor Core GEMM for target BF16 lm_head.
//
// C[M, N] = A[M, K] @ B[N, K]^T  (BF16 in, FP32 acc, BF16 out, row-major).
// Supports arbitrary N (including odd vocab N=248077) directly into C with
// zero scratch buffer and zero copy, while streaming B at full 16B coalesced
// bandwidth (L2::evict_first when gridDim.y == 1) directly into m16n8k16 MMA
// B-fragments via 4-lane sub-warp shuffles.
// Preconditions: K > 0 && K % 64 == 0, A and B 16B-aligned, C 2B-aligned.
// =============================================================================

__device__ __forceinline__ unsigned int bf16_shfl_u4_lane_word(
    const uint4& v,
    int src_lane,
    unsigned int word_idx
) {
    const unsigned int x = __shfl_sync(0xffffffffu, v.x, src_lane, 4);
    const unsigned int y = __shfl_sync(0xffffffffu, v.y, src_lane, 4);
    const unsigned int z = __shfl_sync(0xffffffffu, v.z, src_lane, 4);
    const unsigned int w = __shfl_sync(0xffffffffu, v.w, src_lane, 4);
    const unsigned int lo = (word_idx & 1u) ? y : x;
    const unsigned int hi = (word_idx & 1u) ? w : z;
    return (word_idx & 2u) ? hi : lo;
}

template <unsigned int TILE_M, unsigned int CO_N>
__device__ __forceinline__ void bf16_gemm_coalesced_v2_body(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    static_assert(CO_N == 64u || CO_N == 128u, "CO_N must be 64 or 128");

    constexpr unsigned int NUM_WARPS = CO_N / 8u;
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int A_TILE_K = 64u;
    constexpr unsigned int A_CHUNKS_PER_ROW = A_TILE_K / 8u;
    static_assert(
        NUM_THREADS % A_CHUNKS_PER_ROW == 0u,
        "CTA threads must be a multiple of A chunks per row");
    constexpr unsigned int A_ROWS_PER_ITER = NUM_THREADS / A_CHUNKS_PER_ROW;
    constexpr unsigned int A_ITERS = (TILE_M + A_ROWS_PER_ITER - 1u) / A_ROWS_PER_ITER;

    __shared__ __align__(16)
        typename W4a16CoalescedV2Smem<TILE_M, CO_N, false>::Storage smem;
    static_assert(
        sizeof(typename W4a16CoalescedV2Smem<TILE_M, CO_N, false>::Storage) <= 49152u,
        "static SMEM must stay <= 48 KiB");

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_n = blockIdx.x * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int gn = cta_n + warp_id * 8u + group_id;

    float acc[M_FRAGS][4];
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        acc[mf][0] = 0.0f;
        acc[mf][1] = 0.0f;
        acc[mf][2] = 0.0f;
        acc[mf][3] = 0.0f;
    }

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

    auto load_b_bf16 = [&](unsigned int kk, uint4& bv_out) {
        const unsigned int k_start = kk + tid4 * 8u;
        if (gn < N && k_start < K) {
            const __nv_bfloat16* b_ptr = B + (unsigned long long)gn * K + k_start;
            asm volatile(
                "ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                : "=r"(bv_out.x), "=r"(bv_out.y), "=r"(bv_out.z), "=r"(bv_out.w)
                : "l"(b_ptr), "l"(pol_b));
        } else {
            bv_out = make_uint4(0u, 0u, 0u, 0u);
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

    issue_a(0u, 0u);
    uint4 bv0, bv1;
    load_b_bf16(0u, bv0);
    load_b_bf16(32u, bv1);
    unsigned int cur_buf = 0u;

    for (unsigned int k_base = 0u; k_base < K; k_base += 64u) {
        cp_async_wait_group<0>();
        __syncthreads();

        const unsigned int next_buf = cur_buf ^ 1u;
        uint4 bv_nxt0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 bv_nxt1 = make_uint4(0u, 0u, 0u, 0u);
        if (k_base + 64u < K) {
            issue_a(next_buf, k_base + 64u);
            load_b_bf16(k_base + 64u, bv_nxt0);
            load_b_bf16(k_base + 96u, bv_nxt1);
        }

        {
            const unsigned int b0 = bf16_shfl_u4_lane_word(bv0, 0, tid4);
            const unsigned int b1 = bf16_shfl_u4_lane_word(bv0, 1, tid4);
            mma_step(cur_buf, 0u, b0, b1);
        }
        {
            const unsigned int b0 = bf16_shfl_u4_lane_word(bv0, 2, tid4);
            const unsigned int b1 = bf16_shfl_u4_lane_word(bv0, 3, tid4);
            mma_step(cur_buf, 16u, b0, b1);
        }
        {
            const unsigned int b0 = bf16_shfl_u4_lane_word(bv1, 0, tid4);
            const unsigned int b1 = bf16_shfl_u4_lane_word(bv1, 1, tid4);
            mma_step(cur_buf, 32u, b0, b1);
        }
        {
            const unsigned int b0 = bf16_shfl_u4_lane_word(bv1, 2, tid4);
            const unsigned int b1 = bf16_shfl_u4_lane_word(bv1, 3, tid4);
            mma_step(cur_buf, 48u, b0, b1);
        }

        bv0 = bv_nxt0;
        bv1 = bv_nxt1;
        cur_buf = next_buf;
    }

    __syncthreads();
    const unsigned int smem_col = warp_id * 8u + tid4 * 2u;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.c[r0][smem_col]) =
                __floats2bfloat162_rn(acc[mf][0], acc[mf][1]);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.c[r1][smem_col]) =
                __floats2bfloat162_rn(acc[mf][2], acc[mf][3]);
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

extern "C" __global__ void __launch_bounds__(512, 2)
bf16_gemm_coalesced_m16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    bf16_gemm_coalesced_v2_body<16u, 128u>(A, B, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 2)
bf16_gemm_coalesced_m32(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    bf16_gemm_coalesced_v2_body<32u, 128u>(A, B, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 1)
bf16_gemm_coalesced_m64(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    bf16_gemm_coalesced_v2_body<64u, 128u>(A, B, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 1)
bf16_gemm_coalesced_m128(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    bf16_gemm_coalesced_v2_body<128u, 128u>(A, B, C, M, N, K);
}

// =============================================================================
// Coalesced-K single-sweep FP8 E4M3 (per-row scaled) Tensor Core GEMM for
// target FP8 lm_head.
//
// C[M, N] = bf16((A[M, K] @ dequant_e4m3(B[N, K])^T) * row_scale[N])
//   A:         [M, K] BF16 (row-major)
//   B:         [N, K] FP8 E4M3 bytes (row-major, from quantize_bf16_to_fp8)
//   row_scale: [N] FP32 (per-row dequant scale)
//   C:         [M, N] BF16 (row-major)
//
// Reads half the weight bytes of BF16 (1 B/weight = 635 MB for N=248077,
// K=2560) in a single coalesced 16B/lane sweep (1 uint4 per lane per K_STEP=64,
// L2::evict_first when gridDim.y == 1) and converts E4M3 -> BF16 losslessly
// in-register before m16n8k16 FP32 MMA. Supports arbitrary odd N directly into
// C with zero scratch buffer and zero copy.
// Preconditions: K > 0 && K % 64 == 0, A and B 16B-aligned, row_scale 4B-aligned,
//                C 2B-aligned.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned int fp8x2_e4m3_to_bf16x2_bits(unsigned int two_bytes) {
    const __half2_raw h2r = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(two_bytes & 0xFFFFu),
        __NV_E4M3);
    const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&h2r));
    const __nv_bfloat162 bf2 = __floats2bfloat162_rn(f2.x, f2.y);
    return *reinterpret_cast<const unsigned int*>(&bf2);
}

template <unsigned int TILE_M, unsigned int CO_N>
__device__ __forceinline__ void fp8_gemm_coalesced_v2_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    static_assert(TILE_M % 16u == 0u, "TILE_M must be a multiple of 16");
    static_assert(CO_N == 64u || CO_N == 128u, "CO_N must be 64 or 128");

    constexpr unsigned int NUM_WARPS = CO_N / 8u;
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32u;
    constexpr unsigned int M_FRAGS = TILE_M / 16u;
    constexpr unsigned int A_TILE_K = 64u;
    constexpr unsigned int A_CHUNKS_PER_ROW = A_TILE_K / 8u;
    static_assert(
        NUM_THREADS % A_CHUNKS_PER_ROW == 0u,
        "CTA threads must be a multiple of A chunks per row");
    constexpr unsigned int A_ROWS_PER_ITER = NUM_THREADS / A_CHUNKS_PER_ROW;
    constexpr unsigned int A_ITERS = (TILE_M + A_ROWS_PER_ITER - 1u) / A_ROWS_PER_ITER;

    __shared__ __align__(16)
        typename W4a16CoalescedV2Smem<TILE_M, CO_N, false>::Storage smem;
    static_assert(
        sizeof(typename W4a16CoalescedV2Smem<TILE_M, CO_N, false>::Storage) <= 49152u,
        "static SMEM must stay <= 48 KiB");

    const unsigned int cta_m = blockIdx.y * TILE_M;
    const unsigned int cta_n = blockIdx.x * CO_N;
    if (cta_m >= M || cta_n >= N) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid4 = lane & 3u;
    const unsigned int gn = cta_n + warp_id * 8u + group_id;

    float acc[M_FRAGS][4];
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        acc[mf][0] = 0.0f;
        acc[mf][1] = 0.0f;
        acc[mf][2] = 0.0f;
        acc[mf][3] = 0.0f;
    }

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

    // Each lane in the 4-lane quad loads 16 FP8 weights (16 bytes):
    //   tid4=0: k in [kk +  0, kk + 15]
    //   tid4=1: k in [kk + 16, kk + 31]
    //   tid4=2: k in [kk + 32, kk + 47]
    //   tid4=3: k in [kk + 48, kk + 63]
    // Together the 4 lanes load all 64 weights for row `gn` in one 64B burst.
    auto load_b_fp8 = [&](unsigned int kk, uint4& bv_out) {
        const unsigned int k_start = kk + tid4 * 16u;
        if (gn < N && k_start < K) {
            const unsigned char* b_ptr = B + (unsigned long long)gn * K + k_start;
            asm volatile(
                "ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
                : "=r"(bv_out.x), "=r"(bv_out.y), "=r"(bv_out.z), "=r"(bv_out.w)
                : "l"(b_ptr), "l"(pol_b));
        } else {
            bv_out = make_uint4(0u, 0u, 0u, 0u);
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

    auto mma_fp8_k16 = [&](unsigned int cur_buf, unsigned int k0, int src_lane, const uint4& bv) {
        const unsigned int w_lo_01 = __shfl_sync(0xffffffffu, bv.x, src_lane, 4);
        const unsigned int w_lo_23 = __shfl_sync(0xffffffffu, bv.y, src_lane, 4);
        const unsigned int w_hi_01 = __shfl_sync(0xffffffffu, bv.z, src_lane, 4);
        const unsigned int w_hi_23 = __shfl_sync(0xffffffffu, bv.w, src_lane, 4);
        const unsigned int w0 = (tid4 & 2u) ? w_lo_23 : w_lo_01;
        const unsigned int w1 = (tid4 & 2u) ? w_hi_23 : w_hi_01;
        const unsigned int shift = (tid4 & 1u) << 4;
        const unsigned int b0 = fp8x2_e4m3_to_bf16x2_bits(w0 >> shift);
        const unsigned int b1 = fp8x2_e4m3_to_bf16x2_bits(w1 >> shift);
        mma_step(cur_buf, k0, b0, b1);
    };

    issue_a(0u, 0u);
    uint4 bv;
    load_b_fp8(0u, bv);
    unsigned int cur_buf = 0u;

    for (unsigned int k_base = 0u; k_base < K; k_base += 64u) {
        cp_async_wait_group<0>();
        __syncthreads();

        const unsigned int next_buf = cur_buf ^ 1u;
        uint4 bv_nxt = make_uint4(0u, 0u, 0u, 0u);
        if (k_base + 64u < K) {
            issue_a(next_buf, k_base + 64u);
            load_b_fp8(k_base + 64u, bv_nxt);
        }

        mma_fp8_k16(cur_buf, 0u, 0, bv);
        mma_fp8_k16(cur_buf, 16u, 1, bv);
        mma_fp8_k16(cur_buf, 32u, 2, bv);
        mma_fp8_k16(cur_buf, 48u, 3, bv);

        bv = bv_nxt;
        cur_buf = next_buf;
    }

    __syncthreads();
    const unsigned int smem_col = warp_id * 8u + tid4 * 2u;
    const unsigned int c_n0 = cta_n + smem_col;
    const unsigned int c_n1 = c_n0 + 1u;
    const float rs0 = (c_n0 < N) ? __ldg(&row_scale[c_n0]) : 0.0f;
    const float rs1 = (c_n1 < N) ? __ldg(&row_scale[c_n1]) : 0.0f;
    #pragma unroll
    for (unsigned int mf = 0u; mf < M_FRAGS; ++mf) {
        const unsigned int r0 = mf * 16u + group_id;
        const unsigned int r1 = r0 + 8u;
        if (cta_m + r0 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.c[r0][smem_col]) =
                __floats2bfloat162_rn(acc[mf][0] * rs0, acc[mf][1] * rs1);
        }
        if (cta_m + r1 < M) {
            *reinterpret_cast<__nv_bfloat162*>(&smem.c[r1][smem_col]) =
                __floats2bfloat162_rn(acc[mf][2] * rs0, acc[mf][3] * rs1);
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

extern "C" __global__ void __launch_bounds__(512, 2)
fp8_gemm_coalesced_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    fp8_gemm_coalesced_v2_body<16u, 128u>(A, B, row_scale, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 2)
fp8_gemm_coalesced_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    fp8_gemm_coalesced_v2_body<32u, 128u>(A, B, row_scale, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 1)
fp8_gemm_coalesced_m64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    fp8_gemm_coalesced_v2_body<64u, 128u>(A, B, row_scale, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(512, 1)
fp8_gemm_coalesced_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    fp8_gemm_coalesced_v2_body<128u, 128u>(A, B, row_scale, C, M, N, K);
}


