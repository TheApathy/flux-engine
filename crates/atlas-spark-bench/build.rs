// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    // Link libcuda for raw CUDA driver API calls in kernel benchmarks (needed at
    // link time even under ATLAS_SKIP_BUILD=1 when building/testing against CI stubs).
    println!("cargo:rustc-link-lib=dylib=cuda");

    if let Ok(cuda_path) = std::env::var("CUDA_HOME")
        && !cuda_path.trim().is_empty()
    {
        let cuda_path = cuda_path.trim();
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
    }
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    match std::env::var("CARGO_CFG_TARGET_ARCH")
        .unwrap_or_else(|_| std::env::consts::ARCH.to_owned())
        .as_str()
    {
        "aarch64" => println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu"),
        "x86_64" => println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu"),
        _ => {}
    }
}
