// SPDX-License-Identifier: AGPL-3.0-only
//
// ABI shim for the exported GDN object.
//
// The CuTe-DSL host code (MLIR runtime) references CUDA symbols with a leading
// underscore — `_cudaLaunchKernelEx`, `_cudaGetDevice`, … — and targets the
// CUDA 12.x runtime ABI. Two mismatches on this toolchain:
//   1. the leading underscore, and
//   2. CUDA 13.0 removed `cudaLaunchKernelEx` in favour of `cudaLaunchKernelExC`.
//
// Each shim is a real function whose prototype is taken from the CUDA headers
// and whose body forwards to the current API, so the compiler checks the
// signature instead of us guessing it.
//
// On `_cudaLaunchKernelEx`: the CUDA 12 form was
// (config, func, args, extra) and CUDA 13's `cudaLaunchKernelExC` dropped the
// trailing reserved `extra`. Forwarding the first three arguments is correct —
// `extra` was required to be NULL and is simply left unread in the callee's
// register slot on aarch64.

#include <cuda_runtime.h>
#include <cuda.h>

extern "C" {

cudaError_t _cudaGetDevice(int *device) { return cudaGetDevice(device); }

cudaError_t _cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr,
                                    int device) {
  return cudaDeviceGetAttribute(value, attr, device);
}

cudaError_t _cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr,
                                  int value) {
  return cudaFuncSetAttribute(func, attr, value);
}

cudaError_t _cudaKernelSetAttributeForDevice(cudaKernel_t kernel,
                                              enum cudaFuncAttribute attr,
                                              int value, int device) {
  return cudaKernelSetAttributeForDevice(kernel, attr, value, device);
}

cudaError_t _cudaLibraryLoadData(cudaLibrary_t *library, const void *code,
                                 enum cudaJitOption *jitOptions,
                                 void **jitOptionsValues,
                                 unsigned int numJitOptions,
                                 enum cudaLibraryOption *libraryOptions,
                                 void **libraryOptionValues,
                                 unsigned int numLibraryOptions) {
  return cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues,
                             numJitOptions, libraryOptions,
                             libraryOptionValues, numLibraryOptions);
}

cudaError_t _cudaLibraryGetKernel(cudaKernel_t *pKernel, cudaLibrary_t library,
                                  const char *name) {
  return cudaLibraryGetKernel(pKernel, library, name);
}

cudaError_t _cudaLaunchKernelEx(const cudaLaunchConfig_t *config,
                                const void *func, void **args, void **extra) {
  (void)extra;
  return cudaLaunchKernelExC(config, func, args);
}

CUresult _cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
                               CUkernel kernel, CUdevice dev) {
  return cuKernelGetAttribute(pi, attrib, kernel, dev);
}

}  // extern "C"