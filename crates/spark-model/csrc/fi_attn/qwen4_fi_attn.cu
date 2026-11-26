// SPDX-License-Identifier: AGPL-3.0-only
//
// FlashInfer FA2 ragged prefill attention for Flash-Next wide-chunk prefill
// (`ATLAS_QWEN4_PREFILL_ATTN_FI=1`): BF16, head_dim 256, GQA, causal, one
// sequence. The kernel and the PrefillPlan scheduler are FlashInfer's
// (Apache-2.0, headers taken from FLASHINFER_HOME at build time, not vendored);
// the plan / params plumbing follows FlashInfer's ragged-prefill wrapper pattern.
//
// A continuation chunk attends over every earlier chunk: the prefix (and the
// chunk itself, already written to the paged BF16 cache) is gathered into
// contiguous [kv_len, nkv, hd] K/V scratch, and FlashInfer's causal mask is
// bottom-right aligned, so query row i sits at absolute position
// kv_len - rows + i.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <mutex>

#include <flashinfer/allocator.h>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>

namespace {

using flashinfer::BatchPrefillRaggedParams;
using flashinfer::DefaultAttention;
using flashinfer::GetPtrFromBaseOffset;
using flashinfer::MaskMode;
using flashinfer::PosEncodingMode;
using flashinfer::PrefillPlan;
using flashinfer::PrefillPlanInfo;

using Params = BatchPrefillRaggedParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
using Attention = DefaultAttention<false, false, false, false>;
constexpr uint32_t kHeadDim = 256;

constexpr size_t kFloatWsBytesDefault = 128ull << 20;
// High-SM GPUs (e.g. SM120 RTX PRO 6000 with 188 SMs) let FlashInfer's
// PrefillPlan split short/medium GQA head_dim=256 prefills across many more
// CTAs, pushing batch_prefill_tmp_v past 128 MiB (observed 141.6 MB at 314
// tokens on Qwen3.8-27B with nq=24). Scale workspace on >64-SM devices while
// keeping GB10 (<=64 SMs) at 128 MiB.
constexpr size_t kFloatWsBytesHighSm = 384ull << 20;
constexpr size_t kIntWsBytes = 8ull << 20;

// Gather absolute rows [0, kv_len) of one layer's paged BF16 cache
// ([num_blocks][block_size][row_elems]) into contiguous [kv_len][row_elems].
// Grid: (kv_len, 1, 1)  Block: (128, 1, 1); 16-byte vectors.
__global__ void k_gather_kv(const uint4* __restrict__ k_pool, const uint4* __restrict__ v_pool,
                            const uint32_t* __restrict__ block_table, uint32_t block_size,
                            uint32_t row_vecs, uint4* __restrict__ k_out,
                            uint4* __restrict__ v_out) {
  const uint32_t row = blockIdx.x;
  const uint64_t block = block_table[row / block_size];
  const uint64_t src = (block * block_size + row % block_size) * row_vecs;
  const uint64_t dst = (uint64_t)row * row_vecs;
  for (uint32_t i = threadIdx.x; i < row_vecs; i += blockDim.x) {
    k_out[dst + i] = k_pool[src + i];
    v_out[dst + i] = v_pool[src + i];
  }
}

__global__ void k_set_indptr(int32_t* qo, int32_t* kv, int32_t rows, int32_t kv_len) {
  qo[0] = 0;
  qo[1] = rows;
  kv[0] = 0;
  kv[1] = kv_len;
}

struct State {
  std::mutex mu;
  void* float_ws = nullptr;
  size_t float_ws_bytes = 0;
  void* int_ws = nullptr;
  void* pinned_ws = nullptr;
  int32_t* indptr_d = nullptr;  // [qo0, qo1, kv0, kv1]
  void* kv_scratch = nullptr;
  size_t kv_scratch_bytes = 0;
  bool planned = false;
  uint32_t plan_rows = 0, plan_kv = 0, plan_nq = 0, plan_nkv = 0;
  PrefillPlanInfo plan_info;
};

State& state() {
  static State s;
  return s;
}

size_t choose_float_ws_bytes() {
  int dev = 0;
  int sm_count = 0;
  if (cudaGetDevice(&dev) == cudaSuccess &&
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
      sm_count > 64) {
    return kFloatWsBytesHighSm;
  }
  return kFloatWsBytesDefault;
}

int ensure_alloc(State& s) {
  if (!s.float_ws) {
    const size_t bytes = choose_float_ws_bytes();
    if (cudaMalloc(&s.float_ws, bytes) != cudaSuccess) return -10;
    s.float_ws_bytes = bytes;
  }
  if (!s.int_ws && cudaMalloc(&s.int_ws, kIntWsBytes) != cudaSuccess) return -11;
  if (!s.pinned_ws && cudaHostAlloc(&s.pinned_ws, kIntWsBytes, cudaHostAllocDefault) != cudaSuccess)
    return -12;
  if (!s.indptr_d && cudaMalloc(&s.indptr_d, 4 * sizeof(int32_t)) != cudaSuccess) return -13;
  return 0;
}

template <uint32_t CTA_TILE_Q, MaskMode MASK, typename P>
static auto call_ragged_dispatched(P& params, __nv_bfloat16* tmp_v, float* tmp_s,
                                   cudaStream_t stream, int)
    -> decltype(flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
                CTA_TILE_Q, kHeadDim, kHeadDim, PosEncodingMode::kNone, false, MASK, false,
                Attention, P>(params, tmp_v, tmp_s, false, stream)) {
  return flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
      CTA_TILE_Q, kHeadDim, kHeadDim, PosEncodingMode::kNone, false, MASK, false, Attention, P>(
      params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
}

template <uint32_t CTA_TILE_Q, MaskMode MASK, typename P>
static auto call_ragged_dispatched(P& params, __nv_bfloat16* tmp_v, float* tmp_s,
                                   cudaStream_t stream, long)
    -> decltype(flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
                CTA_TILE_Q, kHeadDim, kHeadDim, PosEncodingMode::kNone, false, MASK, Attention, P>(
        params, tmp_v, tmp_s, false, stream)) {
  return flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
      CTA_TILE_Q, kHeadDim, kHeadDim, PosEncodingMode::kNone, false, MASK, Attention, P>(
      params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
}

template <MaskMode MASK>
cudaError_t dispatch(Params& params, __nv_bfloat16* tmp_v, float* tmp_s, int64_t cta_tile_q,
                     cudaStream_t stream) {
  cudaError_t status = cudaSuccess;
  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    status = call_ragged_dispatched<CTA_TILE_Q, MASK>(params, tmp_v, tmp_s, stream, 0);
  });
  return status;
}

template <typename S>
static auto call_prefill_plan(S& st_ref, int32_t* qo, int32_t* kv, uint32_t r, uint32_t nq_,
                              uint32_t nkv_, cudaStream_t stream_, int)
    -> decltype(PrefillPlan<int32_t>(
        st_ref.float_ws, st_ref.float_ws_bytes, st_ref.int_ws, st_ref.pinned_ws, kIntWsBytes,
        st_ref.plan_info, qo, kv, r, 1, nq_, nkv_, kHeadDim, kHeadDim, 1, false,
        sizeof(__nv_bfloat16), -1, -1, false, 0, int64_t{0}, false, stream_)) {
  return PrefillPlan<int32_t>(st_ref.float_ws, st_ref.float_ws_bytes, st_ref.int_ws,
                              st_ref.pinned_ws, kIntWsBytes, st_ref.plan_info, qo, kv, r, 1, nq_,
                              nkv_, kHeadDim, kHeadDim, 1, false, sizeof(__nv_bfloat16), -1, -1,
                              false, 0, int64_t{0}, false, stream_);
}

template <typename S>
static auto call_prefill_plan(S& st_ref, int32_t* qo, int32_t* kv, uint32_t r, uint32_t nq_,
                              uint32_t nkv_, cudaStream_t stream_, long)
    -> decltype(PrefillPlan<int32_t>(
        st_ref.float_ws, st_ref.float_ws_bytes, st_ref.int_ws, st_ref.pinned_ws, kIntWsBytes,
        st_ref.plan_info, qo, kv, r, 1, nq_, nkv_, kHeadDim, kHeadDim, 1, false,
        sizeof(__nv_bfloat16), -1, -1, false, 0, stream_)) {
  return PrefillPlan<int32_t>(st_ref.float_ws, st_ref.float_ws_bytes, st_ref.int_ws,
                              st_ref.pinned_ws, kIntWsBytes, st_ref.plan_info, qo, kv, r, 1, nq_,
                              nkv_, kHeadDim, kHeadDim, 1, false, sizeof(__nv_bfloat16), -1, -1,
                              false, 0, stream_);
}

}  // namespace

extern "C" {

// q: [rows, nq, 256] BF16 contiguous. k_chunk / v_chunk: [rows, nkv, 256] BF16
// (used only when start == 0). k_pool / v_pool: this layer's paged BF16 cache;
// block_table: the sequence's u32 block ids. o: [rows, nq, 256] BF16.
// Returns 0 on success, a negative wrapper code, or a cudaError_t value.
int32_t atlas_qwen4_fi_attn_prefill(const void* q, const void* k_chunk, const void* v_chunk,
                                    const void* k_pool, const void* v_pool,
                                    const uint32_t* block_table, uint32_t block_size,
                                    uint32_t start, uint32_t rows, uint32_t nq, uint32_t nkv,
                                    float sm_scale, void* o, void* stream_raw) {
  if (rows == 0 || nkv == 0 || nq % nkv != 0) return -2;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  try {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);
    int rc = ensure_alloc(s);
    if (rc) return rc;
    const uint32_t kv_len = start + rows;
    const uint32_t row_elems = nkv * kHeadDim;
    const void* k = k_chunk;
    const void* v = v_chunk;
    if (start > 0) {
      if (!k_pool || !v_pool || !block_table || block_size == 0) return -3;
      const size_t need = (size_t)kv_len * row_elems * sizeof(__nv_bfloat16) * 2;
      if (need > s.kv_scratch_bytes) {
        // Grow-only, never freed mid-request (the GB10 free-under-traffic
        // class): the old block is retired, not freed. Doubling bounds the
        // retired total below the live block (with the QSA core this path only
        // sees chunks ending inside the dense window, a few MB).
        const size_t cap = need > 2 * s.kv_scratch_bytes ? need : 2 * s.kv_scratch_bytes;
        s.kv_scratch = nullptr;
        s.kv_scratch_bytes = 0;
        if (cudaMalloc(&s.kv_scratch, cap) != cudaSuccess) return -14;
        s.kv_scratch_bytes = cap;
      }
      uint4* k_out = static_cast<uint4*>(s.kv_scratch);
      uint4* v_out = k_out + (size_t)kv_len * row_elems / 8;
      k_gather_kv<<<kv_len, 128, 0, stream>>>(static_cast<const uint4*>(k_pool),
                                              static_cast<const uint4*>(v_pool), block_table,
                                              block_size, row_elems / 8, k_out, v_out);
      k = k_out;
      v = v_out;
    }

    // PrefillPlan writes the pinned buffer and enqueues an async copy from it:
    // drain the stream before re-planning so no pending copy of the previous
    // plan reads the rewritten bytes. One sync per new (rows, kv_len) shape.
    if (!s.planned || s.plan_rows != rows || s.plan_kv != kv_len || s.plan_nq != nq ||
        s.plan_nkv != nkv) {
      cudaError_t st = cudaStreamSynchronize(stream);
      if (st != cudaSuccess) return static_cast<int32_t>(st);
      int32_t qo_indptr_h[2] = {0, (int32_t)rows};
      int32_t kv_indptr_h[2] = {0, (int32_t)kv_len};
      st = call_prefill_plan(s, qo_indptr_h, kv_indptr_h, rows, nq, nkv, stream, 0);
      if (st != cudaSuccess) return static_cast<int32_t>(st);
      k_set_indptr<<<1, 1, 0, stream>>>(s.indptr_d, s.indptr_d + 2, (int32_t)rows,
                                        (int32_t)kv_len);
      s.planned = true;
      s.plan_rows = rows;
      s.plan_kv = kv_len;
      s.plan_nq = nq;
      s.plan_nkv = nkv;
    }

    Params params;
    params.q = static_cast<__nv_bfloat16*>(const_cast<void*>(q));
    params.k = static_cast<__nv_bfloat16*>(const_cast<void*>(k));
    params.v = static_cast<__nv_bfloat16*>(const_cast<void*>(v));
    params.o = static_cast<__nv_bfloat16*>(o);
    params.lse = nullptr;
    params.q_indptr = s.indptr_d;
    params.kv_indptr = s.indptr_d + 2;
    params.num_qo_heads = nq;
    params.num_kv_heads = nkv;
    params.group_size = flashinfer::uint_fastdiv(nq / nkv);
    params.q_stride_n = nq * kHeadDim;
    params.q_stride_h = kHeadDim;
    params.k_stride_n = row_elems;
    params.k_stride_h = kHeadDim;
    params.v_stride_n = row_elems;
    params.v_stride_h = kHeadDim;
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = sm_scale;

    const PrefillPlanInfo& pi = s.plan_info;
    params.request_indices = GetPtrFromBaseOffset<int32_t>(s.int_ws, pi.request_indices_offset);
    params.qo_tile_indices = GetPtrFromBaseOffset<int32_t>(s.int_ws, pi.qo_tile_indices_offset);
    params.kv_tile_indices = GetPtrFromBaseOffset<int32_t>(s.int_ws, pi.kv_tile_indices_offset);
    params.o_indptr = GetPtrFromBaseOffset<int32_t>(s.int_ws, pi.o_indptr_offset);
    params.kv_chunk_size_ptr =
        GetPtrFromBaseOffset<int32_t>(s.int_ws, pi.kv_chunk_size_ptr_offset);
    __nv_bfloat16* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (pi.split_kv) {
      params.merge_indptr = GetPtrFromBaseOffset<int32_t>(s.int_ws, pi.merge_indptr_offset);
      tmp_v = GetPtrFromBaseOffset<__nv_bfloat16>(s.float_ws, pi.v_offset);
      tmp_s = GetPtrFromBaseOffset<float>(s.float_ws, pi.s_offset);
    }
    params.padded_batch_size = pi.padded_batch_size;
    params.max_total_num_rows = pi.total_num_rows;

    cudaError_t st = dispatch<MaskMode::kCausal>(params, tmp_v, tmp_s, pi.cta_tile_q, stream);
    if (st != cudaSuccess) return static_cast<int32_t>(st);
    st = cudaGetLastError();
    return st == cudaSuccess ? 0 : static_cast<int32_t>(st);
  } catch (...) {
    return -4;
  }
}

}  // extern "C"
