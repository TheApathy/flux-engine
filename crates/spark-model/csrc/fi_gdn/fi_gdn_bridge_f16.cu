// SPDX-License-Identifier: AGPL-3.0-only
//
// Hand-maintained bridge for the FP16 FI GDN export
// (atlas_fi_gdn_sm121a_prefill_f16.o,
// repro-kit/source-closure/v0.9.13-r8/gdn-aot/fp16/export_fp16.py: q/k/v/o in
// float16, `_f16` suffix). Extends the base bridge with the strip API, 32x33
// shared-memory transpose, and L2-discard helpers.
//
// Bridge: FlashInfer 0.6.14 SM121a chunked-delta-rule GDN prefill AOT object
// behind the Rust engine's native buffer layouts.
//
// Why a bridge at all: the exported CuTe-DSL object is a whole *host* launcher
// (marshalling + cudaLibraryLoadData + launch), not just a cubin we can hand to
// the engine's PTX registry. Linking its host code is the only proven path
// (the Python/cutlass ExternalBinaryModule route returns rc=98 / segfaults).
//
// Layout translation performed here (all coordinates verified against the
// in-process DSL on identical inputs):
//
//   engine                          AOT object expects                 action
//   ------------------------------  --------------------------------  -----------
//   qkv: one fp16 buffer, row       descriptors carry the row stride  zero-copy
//   stride conv_dim, Q|K|V at       (proven bit-exact), so the packed
//   0 / KD / 2*KD                   slices are described directly
//   output: [T, HV*D] fp16          [T, HV, D] contiguous              zero-copy
//   gate_beta: interleaved fp32     two contiguous fp32 [T*HV]        de-interleave
//   [T, 2*HV] (gate cols 0..HV-1,   (rank-1 descriptor has no stride)
//    beta cols HV..2HV-1)
//   state: fp32 [HV][K][V]          fp32 [HV][V][K]                   transpose
//
// Initial state: the artifact is exported WITH a real `init_state` tensor, so
// `needs_init_state` is true and the kernel reads the incoming recurrent state.
// That is what makes this route usable on every chunk of a chunked prefill, not
// just the first (zero-state) one. The engine's fp32 [HV][K][V] state is
// transposed into the kernel's [HV][V][K] layout and passed as init_state; the
// kernel writes its final state into a separate scratch buffer, which is
// transposed back into the engine buffer.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <cstdlib>
#include <cstring>

#include "atlas_fi_gdn_sm121a_prefill_f16.h"

namespace {

// Names this bridge in its FI_GDN_SCRATCH_GROWN line.
constexpr const char *kBridge = "fi_gdn_bridge_f16";

// ATLAS_PREFILL_SCRATCH_INIT (see spark_runtime::scratch_init): a newly allocated block is
// zeroed (default), filled with 0xFF (poison) or left as the driver returned it (off), so no
// kernel can read a previous owner's bytes on the first request large enough to grow it.
int init_new(void *p, size_t bytes) {
  const char *mode = std::getenv("ATLAS_PREFILL_SCRATCH_INIT");
  if (mode && std::strcmp(mode, "off") == 0) return 0;
  const int byte = (mode && std::strcmp(mode, "poison") == 0) ? 0xFF : 0;
  if (cudaMemset(p, byte, bytes) != cudaSuccess) return -8;
  // The prefill stream is not ordered with the legacy stream: finish before first use.
  return cudaDeviceSynchronize() == cudaSuccess ? 0 : -8;
}

constexpr int kHv = 48;
constexpr int kD = 128;
constexpr int kMaxStateElems = kHv * kD * kD;
constexpr int kTensormapBytes = 48 * 128;

bool resolve_heads_from_conv_dim(int32_t conv_dim, int32_t *hq_out, int32_t *hv_out) {
  if (conv_dim == 10240) {
    *hq_out = 16;
    *hv_out = 48;
    return true;
  }
  if (conv_dim == 5120) {
    *hq_out = 8;
    *hv_out = 24;
    return true;
  }
  if (conv_dim == 2560) {
    *hq_out = 4;
    *hv_out = 12;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------- kernels

// gate_beta is [T, 2*HV] fp32: element (t, vh) of the gate lives at flat
// t*2*HV + vh and the corresponding beta at t*2*HV + HV + vh. Pack them into
// two contiguous [T*HV] arrays indexed by i = t*HV + vh, and write
// cu_seqlens = {0, total_len} from thread 0 in the same launch.
__global__ void k_deinterleave_gate_beta(const float *__restrict__ gb,
                                         float *__restrict__ gate,
                                         float *__restrict__ beta, long n,
                                         int hv,
                                         int64_t *__restrict__ cu,
                                         int64_t total_len) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0 && cu != nullptr) {
    cu[0] = 0;
    cu[1] = total_len;
  }
  if (i < n) {
    long t = i / hv;
    long vh = i - t * hv;
    long row = t * 2 * hv;
    gate[i] = gb[row + vh];
    beta[i] = gb[row + hv + vh];
  }
}

// Per-head 128x128 transpose: dst[h][i][j] = src[h][j][i].
// Uses a padded 32x33 shared-memory tile with a (32, 8, 1) CTA so every warp
// issues full 128-byte cache-line global loads and stores with zero bank conflicts.
__global__ void k_transpose_state(const float *__restrict__ src,
                                  float *__restrict__ dst, int n,
                                  int64_t *__restrict__ cu,
                                  int64_t total_len,
                                  const unsigned char *__restrict__ discard_ptr = nullptr,
                                  size_t discard_lines = 0) {
  __shared__ float tile[32][33];
  int h = blockIdx.z;
  int bx = blockIdx.x * 32;
  int by = blockIdx.y * 32;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  if (cu != nullptr && h == 0 && bx == 0 && by == 0 && tx == 0 && ty == 0) {
    cu[0] = 0;
    cu[1] = total_len;
  }
  if (discard_ptr != nullptr) {
    const size_t cta_id = ((size_t)h * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t tid_global = cta_id * 256u + (size_t)ty * 32u + (size_t)tx;
    const size_t total_threads = (size_t)gridDim.x * gridDim.y * gridDim.z * 256u;
    for (size_t line = tid_global; line < discard_lines; line += total_threads) {
      asm volatile("discard.global.L2 [%0], 128;\n" :: "l"(discard_ptr + line * 128u) : "memory");
    }
  }
  long base = (long)h * n * n;
  int c_in = by + tx;
  #pragma unroll
  for (int s = 0; s < 32; s += 8) {
    int r_in = bx + ty + s;
    if (r_in < n && c_in < n) {
      float v;
      const float *p = src + base + (long)r_in * n + c_in;
      asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p));
      tile[ty + s][tx] = v;
    }
  }
  __syncthreads();
  int c_out = bx + tx;
  #pragma unroll
  for (int s = 0; s < 32; s += 8) {
    int r_out = by + ty + s;
    if (r_out < n && c_out < n) {
      float v = tile[tx][ty + s];
      float *p = dst + base + (long)r_out * n + c_out;
      if (total_len > 0) {
        asm volatile("st.global.wb.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
      } else {
        asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
      }
    }
  }
}

// cu_seqlens = {0, total}: written on the stream instead of an H2D copy from
// a pageable host array, which stages through a host bounce buffer on every
// call and is not capturable.
__global__ void k_set_cu_seqlens(int64_t *cu, int64_t total) {
  cu[0] = 0;
  cu[1] = total;
}

// ---------------------------------------------------------------- state

struct Scratch {
  std::mutex mu;
  bool module_loaded = false;
  atlas_fi_gdn_sm121a_prefill_f16_Kernel_Module_t mod{};
  float *gate = nullptr;
  float *beta = nullptr;
  float *state_fi = nullptr;
  float *state_init_fi = nullptr;
  void *tensormaps = nullptr;
  int64_t *cu = nullptr;
  int64_t cap_tokens = 0;
  int64_t last_cu_len = -1;
  // Stream of the previous call (see the stream-switch drain in the entry).
  cudaStream_t last_stream = nullptr;
  bool has_last_stream = false;
};

Scratch &scratch() {
  static Scratch s;
  return s;
}

// Grow-only, never freed: a cudaFree here runs mid-request while the previous
// chunk's kernels may still be queued (the GB10 free-under-traffic class). A
// grown block retires the old one (kept, not freed); the engine reserves its
// largest chunk at load through the `_reserve` entry, so requests never grow it.
int ensure_capacity(Scratch &s, int64_t tokens) {
  if (tokens <= s.cap_tokens && s.gate) return 0;
  if (s.gate)
    std::fprintf(stderr, "FI_GDN_SCRATCH_GROWN bridge=%s tokens=%lld retired_tokens=%lld freed=0\n",
                 kBridge, (long long)tokens, (long long)s.cap_tokens);
  s.gate = s.beta = nullptr;
  s.cap_tokens = 0;
  size_t n = (size_t)tokens * kHv;
  if (cudaMalloc(&s.gate, n * 4) != cudaSuccess) return -1;
  if (cudaMalloc(&s.beta, n * 4) != cudaSuccess) return -2;
  if (init_new(s.gate, n * 4) || init_new(s.beta, n * 4)) return -8;
  s.cap_tokens = tokens;
  return 0;
}

int ensure_once(Scratch &s) {
  if (!s.module_loaded) {
    atlas_fi_gdn_sm121a_prefill_f16_Kernel_Module_Load(&s.mod);
    s.module_loaded = true;
  }
  const bool new_state = !s.state_fi, new_init = !s.state_init_fi, new_tm = !s.tensormaps,
             new_cu = !s.cu;
  if (!s.state_fi && cudaMalloc(&s.state_fi, (size_t)kMaxStateElems * 4) != cudaSuccess)
    return -3;
  if (!s.state_init_fi &&
      cudaMalloc(&s.state_init_fi, (size_t)kMaxStateElems * 4) != cudaSuccess)
    return -4;
  if (!s.tensormaps && cudaMalloc(&s.tensormaps, kTensormapBytes) != cudaSuccess)
    return -5;
  if (!s.cu && cudaMalloc(&s.cu, 2 * 8) != cudaSuccess) return -6;
  if ((new_state && init_new(s.state_fi, (size_t)kMaxStateElems * 4)) ||
      (new_init && init_new(s.state_init_fi, (size_t)kMaxStateElems * 4)) ||
      (new_tm && init_new(s.tensormaps, kTensormapBytes)) || (new_cu && init_new(s.cu, 2 * 8)))
    return -8;
  return 0;
}

}  // namespace

extern "C" {

// Return codes.
enum {
  ATLAS_GDN_FI_OK = 0,
  ATLAS_GDN_FI_BAD_SHAPE = 2,
  ATLAS_GDN_FI_ERR = -1,
};

// Shapes: q/k/v packed in `qkv` with row stride `conv_dim` ELEMENTS and the
// slices at 0 / kd / 2*kd. `state` is the engine's fp32 [HV][K][V].
// Supports TP=1 (hq=16, hv=48, conv_dim=10240), TP=2 (hq=8, hv=24, conv_dim=5120),
// and TP=4 (hq=4, hv=12, conv_dim=2560).
// Returns ATLAS_GDN_FI_OK, ATLAS_GDN_FI_BAD_SHAPE, or a negative error.
int32_t atlas_gdn_fi_sm121a_prefill_f16(void *qkv, void *gate_beta, void *output,
                                    void *state, int32_t total_len,
                                    int32_t conv_dim, void *stream) {
  Scratch &s = scratch();
  std::lock_guard<std::mutex> lock(s.mu);
  const bool planar_gb = (conv_dim < 0);
  if (planar_gb) conv_dim = -conv_dim;
  int32_t hq = 0, hv = 0;
  if (total_len <= 0 || !resolve_heads_from_conv_dim(conv_dim, &hq, &hv))
    return ATLAS_GDN_FI_BAD_SHAPE;
  if (!qkv || !gate_beta || !output || !state) return ATLAS_GDN_FI_ERR;
  const int32_t kd = hq * kD;
  const int32_t vd = hv * kD;
  const int32_t state_elems = hv * kD * kD;
  const int32_t tm_bytes = hv * 128;

  int rc = ensure_once(s);
  if (rc) return rc;
  if (!planar_gb) {
    rc = ensure_capacity(s, total_len);
    if (rc) return rc;
  }

  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  if (s.has_last_stream && s.last_stream != st &&
      cudaStreamSynchronize(s.last_stream) != cudaSuccess)
    return -1;
  s.last_stream = st;
  s.has_last_stream = true;
  s.last_cu_len = total_len;

  // ---- de-interleave gate/beta (unless already planar) + set cu_seqlens ----
  long n = (long)total_len * hv;
  float *gate_ptr = s.gate;
  float *beta_ptr = s.beta;
  if (planar_gb) {
    gate_ptr = reinterpret_cast<float *>(gate_beta);
    beta_ptr = gate_ptr + n;
  } else {
    k_deinterleave_gate_beta<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(
        reinterpret_cast<const float *>(gate_beta), s.gate, s.beta, n, hv,
        s.cu, (int64_t)total_len);
  }

  // ---- state [HV][K][V] -> init scratch [HV][V][K] (the kernel's read layout) ----
  dim3 tblock(32, 8, 1);
  dim3 tgrid(kD / 32, kD / 32, hv);
  k_transpose_state<<<tgrid, tblock, 0, st>>>(
      reinterpret_cast<const float *>(state), s.state_init_fi, kD,
      planar_gb ? s.cu : nullptr, (int64_t)total_len);

  // ---- marshal the AOT call ----
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_q_t gq{
      qkv, {total_len, kD, hq}, {(int64_t)conv_dim, kD}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_k_t gk{
      static_cast<uint16_t *>(qkv) + kd, {kD, total_len, hq},
      {(int64_t)conv_dim, kD}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_v_t gv{
      static_cast<uint16_t *>(qkv) + 2 * kd, {kD, total_len, hv},
      {(int64_t)conv_dim, kD}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_o_t go{
      output, {kD, total_len, hv}, {(int64_t)vd, kD}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_alpha_t ga{gate_ptr, {total_len * hv}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_beta_t gb{beta_ptr, {total_len * hv}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_state_t gst{s.state_fi, {state_elems}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_init_state_t ginit{s.state_init_fi,
                                                       {state_elems}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_tensormaps_t gtm{s.tensormaps,
                                                        {tm_bytes}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_cu_seqlens_t gcu{s.cu, {2}};

  float scale = 1.0f / sqrtf((float)kD);
  int32_t nq = hq, nk = hq, nv = hv, nsab = hv, nseq = 1, nchk = 1,
          chkevery = 0;
  int32_t gx = nseq * nsab;
  int32_t ret = -1;
  void *packed[21] = {&gq, &gk, &gv, &go, &ga, &gb, &gst, &ginit, &gtm, &gcu,
                      &scale, &nq, &nk, &nv, &nsab, &nseq, &nchk, &chkevery,
                      &gx, &st, &ret};
  _mlir_atlas_fi_gdn_sm121a_prefill_f16__mlir_ciface_cutlass___call___flashinfergdn_kernelsdelta_rule_dsldelta_rule_sm120_FullyFusedDeltaRuleSm120_object_at__Tensorgmemoi641i64_Tensorgmemo1i64i64_Tensorgmemo1i64i64_Tensorgmemo1i64i64(packed, 21);
  if (ret != 0) return ATLAS_GDN_FI_ERR;

  // ---- state scratch [HV][V][K] -> engine [HV][K][V] ----
  k_transpose_state<<<tgrid, tblock, 0, st>>>(
      s.state_fi, reinterpret_cast<float *>(state), kD, nullptr, 0);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ATLAS_GDN_FI_ERR;
  return ATLAS_GDN_FI_OK;
}

// Strip-mined L2-resident FlashInfer GDN step (`strip_flags`: bit 0 = first strip,
// bit 1 = last strip). When `beta_in != nullptr`, `gate_in` and `beta_in` point directly
// to planar [strip_len * HV] slices; otherwise `gate_in` holds interleaved [strip_len, 2*HV]
// rows. Intermediate strips swap `(s.state_init_fi, s.state_fi)` in-place without transpose
// launches, and the final strip's state transpose also discards the L2-resident `qkv` buffer.
int32_t atlas_gdn_fi_sm121a_prefill_f16_strip(void *qkv, void *gate_in, void *beta_in,
                                              void *output, void *state,
                                              int32_t strip_len, int32_t conv_dim,
                                              int32_t strip_flags, void *stream) {
  Scratch &s = scratch();
  std::lock_guard<std::mutex> lock(s.mu);
  if (conv_dim < 0) conv_dim = -conv_dim;
  int32_t hq = 0, hv = 0;
  if (strip_len <= 0 || !resolve_heads_from_conv_dim(conv_dim, &hq, &hv))
    return ATLAS_GDN_FI_BAD_SHAPE;
  if (!qkv || !gate_in || !output || !state) return ATLAS_GDN_FI_ERR;
  const int32_t kd = hq * kD;
  const int32_t vd = hv * kD;
  const int32_t state_elems = hv * kD * kD;
  const int32_t tm_bytes = hv * 128;

  const bool first_strip = (strip_flags & 1) != 0;
  const bool last_strip = (strip_flags & 2) != 0;
  const bool planar_gb = (beta_in != nullptr);

  int rc = ensure_once(s);
  if (rc) return rc;
  if (!planar_gb) {
    rc = ensure_capacity(s, strip_len);
    if (rc) return rc;
  }

  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  if (s.has_last_stream && s.last_stream != st) {
    if (cudaStreamSynchronize(s.last_stream) != cudaSuccess) return -1;
    s.last_cu_len = -1;
  }
  s.last_stream = st;
  s.has_last_stream = true;

  long n = (long)strip_len * hv;
  float *gate_ptr = s.gate;
  float *beta_ptr = s.beta;
  if (planar_gb) {
    gate_ptr = reinterpret_cast<float *>(gate_in);
    beta_ptr = reinterpret_cast<float *>(beta_in);
  } else {
    k_deinterleave_gate_beta<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(
        reinterpret_cast<const float *>(gate_in), s.gate, s.beta, n, hv,
        s.cu, (int64_t)strip_len);
    s.last_cu_len = strip_len;
  }

  dim3 tblock(32, 8, 1);
  dim3 tgrid(kD / 32, kD / 32, hv);
  if (first_strip) {
    k_transpose_state<<<tgrid, tblock, 0, st>>>(
        reinterpret_cast<const float *>(state), s.state_init_fi, kD,
        planar_gb ? s.cu : nullptr, (int64_t)strip_len);
    if (planar_gb) s.last_cu_len = strip_len;
  } else if (planar_gb && s.last_cu_len != strip_len) {
    k_set_cu_seqlens<<<1, 1, 0, st>>>(s.cu, (int64_t)strip_len);
    s.last_cu_len = strip_len;
  }

  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_q_t gq{
      qkv, {strip_len, kD, hq}, {(int64_t)conv_dim, kD}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_k_t gk{
      static_cast<uint16_t *>(qkv) + kd, {kD, strip_len, hq},
      {(int64_t)conv_dim, kD}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_v_t gv{
      static_cast<uint16_t *>(qkv) + 2 * kd, {kD, strip_len, hv},
      {(int64_t)conv_dim, kD}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_o_t go{
      output, {kD, strip_len, hv}, {(int64_t)vd, kD}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_alpha_t ga{gate_ptr, {strip_len * hv}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_beta_t gb{beta_ptr, {strip_len * hv}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_state_t gst{s.state_fi, {state_elems}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_init_state_t ginit{s.state_init_fi,
                                                       {state_elems}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_g_tensormaps_t gtm{s.tensormaps,
                                                        {tm_bytes}};
  atlas_fi_gdn_sm121a_prefill_f16_Tensor_cu_seqlens_t gcu{s.cu, {2}};

  float scale = 1.0f / sqrtf((float)kD);
  int32_t nq = hq, nk = hq, nv = hv, nsab = hv, nseq = 1, nchk = 1,
          chkevery = 0;
  int32_t gx = nseq * nsab;
  int32_t ret = -1;
  void *packed[21] = {&gq, &gk, &gv, &go, &ga, &gb, &gst, &ginit, &gtm, &gcu,
                      &scale, &nq, &nk, &nv, &nsab, &nseq, &nchk, &chkevery,
                      &gx, &st, &ret};
  _mlir_atlas_fi_gdn_sm121a_prefill_f16__mlir_ciface_cutlass___call___flashinfergdn_kernelsdelta_rule_dsldelta_rule_sm120_FullyFusedDeltaRuleSm120_object_at__Tensorgmemoi641i64_Tensorgmemo1i64i64_Tensorgmemo1i64i64_Tensorgmemo1i64i64(packed, 21);
  if (ret != 0) return ATLAS_GDN_FI_ERR;

  if (last_strip) {
    const unsigned char *disc_p =
        ((reinterpret_cast<uintptr_t>(qkv) & 127u) == 0u)
            ? reinterpret_cast<const unsigned char *>(qkv)
            : nullptr;
    const size_t disc_lines =
        disc_p ? (((size_t)strip_len * (size_t)conv_dim * 2u) >> 7) : 0u;
    k_transpose_state<<<tgrid, tblock, 0, st>>>(
        s.state_fi, reinterpret_cast<float *>(state), kD, nullptr, 0,
        disc_p, disc_lines);
  } else {
    std::swap(s.state_init_fi, s.state_fi);
  }

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ATLAS_GDN_FI_ERR;
  return ATLAS_GDN_FI_OK;
}

// Load-time sizing (ATLAS_PREFILL_SCRATCH_INIT applies): loads the module and
// allocates the state, tensormap, cu_seqlens and gate/beta buffers for `tokens`
// rows, so no prefill call allocates. Returns 0 or a negative error.
int32_t atlas_gdn_fi_sm121a_prefill_f16_reserve(int64_t tokens) {
  Scratch &s = scratch();
  std::lock_guard<std::mutex> lock(s.mu);
  if (tokens <= 0) return -7;
  int rc = ensure_once(s);
  if (rc) return rc;
  return ensure_capacity(s, tokens);
}

// Diagnostic used by the Rust-side engagement receipt.
const char *atlas_gdn_fi_sm121a_f16_name(void) { return "flashinfer-gdn-sm121a-f16-aot"; }

}  // extern "C"
