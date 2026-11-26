// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    // libcuda / libcublasLt / libcudart are only needed when the cuda feature is
    // enabled (i.e. when AtlasCudaBackend and cublaslt FFI are compiled in).
    // The metal feature build on Apple Silicon must not request CUDA libraries.
    // Note: ATLAS_SKIP_BUILD=1 only suppresses nvcc kernel compilation in
    // atlas-kernels / spark-storage / spark-model; rustc still compiles the
    // extern "C" CUDA/cuBLASLt bindings and needs these dylib link flags so
    // `cargo test` links cleanly against real or CI-stub shared libraries.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    // Link libcuda for AtlasCudaBackend's raw CUDA driver API calls.
    println!("cargo:rustc-link-lib=dylib=cuda");
    // cublaslt.rs FFI references cublasLt* symbols that must resolve at link time.
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    // cuda_backend/gpu_impl.rs's copy_d2d_2d_async references cudaMemcpy2DAsync
    // from the CUDA runtime library.
    println!("cargo:rustc-link-lib=dylib=cudart");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME")
        && !cuda_path.trim().is_empty()
    {
        let cuda_path = cuda_path.trim();
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
    }
    // Standard CUDA locations
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    match std::env::var("CARGO_CFG_TARGET_ARCH")
        .unwrap_or_else(|_| std::env::consts::ARCH.to_owned())
        .as_str()
    {
        "aarch64" => {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/sbsa-linux/lib/stubs");
            println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
        }
        "x86_64" => {
            println!(
                "cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib/stubs"
            );
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        }
        _ => {}
    }
}
