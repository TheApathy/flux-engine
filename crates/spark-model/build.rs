// SPDX-License-Identifier: AGPL-3.0-only
//
// Optionally builds and links the FlashInfer GDN prefill bridge (`csrc/fi_gdn/`)
// when user-supplied CuTe-DSL AOT object files are present in `csrc/fi_gdn/`.
//
// Prebuilt `.o` binaries are intentionally not distributed in the open-source
// tree; when absent, this build script skips `libatlas_fi_gdn.a` and omits
// `cfg(atlas_fi_gdn)`, cleanly falling back to the native CUDA GDN prefill
// kernels on all targets (`gb10`, `rtx6000`, etc.).
//
// Non-CUDA builds (the Metal feature) skip this entirely.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../build-support/native_command_trace.rs"]
mod native_command_trace;
#[path = "../../build-support/release_path_remap.rs"]
mod release_path_remap;

fn find_tool(explicit_env: &str, candidates: &[PathBuf], fallback: &str) -> String {
    if native_command_trace::enabled() && matches!(explicit_env, "ATLAS_NVCC" | "ATLAS_CXX") {
        return native_command_trace::pinned_tool(explicit_env)
            .display()
            .to_string();
    }
    if let Ok(v) = std::env::var(explicit_env)
        && !v.is_empty()
    {
        return v;
    }
    for c in candidates {
        if c.exists() {
            return c.display().to_string();
        }
    }
    fallback.to_string()
}

fn run(cmd: &mut Command, what: &str) {
    let display = format!("{cmd:?}");
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("spark-model build: failed to spawn {what}: {e}\n{display}"));
    if !out.status.success() {
        panic!(
            "spark-model build: {what} failed (status {:?})\n{display}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
}

fn run_native(source: &Path, output: &Path, role: &'static str, cmd: &mut Command, what: &str) {
    let trace = native_command_trace::prepare(cmd, source, output, role);
    run(cmd, what);
    native_command_trace::finish(trace, cmd, true);
}

fn compiler_with_release_path_remap(tool: &str, nvcc: bool) -> Command {
    release_path_remap::compiler(tool, nvcc)
}

fn canonical_hw_name(raw: &str) -> &str {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("sm120") {
        "rtx6000"
    } else if trimmed.eq_ignore_ascii_case("sm121") {
        "gb10"
    } else if trimmed.eq_ignore_ascii_case("sm103") {
        "gb300"
    } else {
        trimmed
    }
}

fn parse_hardware_arch(text: &str) -> Option<String> {
    let mut in_hardware = false;
    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_hardware = line == "[hardware]";
            continue;
        }
        if !in_hardware {
            continue;
        }
        if let Some((key, val)) = line.split_once('=')
            && key.trim() == "arch"
        {
            let arch = val.trim().trim_matches('"').trim();
            if !arch.is_empty() {
                return Some(arch.to_owned());
            }
        }
    }
    None
}

fn find_fi_gdn_aot_objects(csrc: &Path, arch: &str, target_arch: &str) -> Option<[PathBuf; 3]> {
    let arch_compact = arch.replace('_', "");
    let mut prefixes = Vec::with_capacity(3);
    prefixes.push(format!("atlas_fi_gdn_{arch_compact}_{target_arch}_prefill"));
    let arch_a = format!("{}a", arch_compact.trim_end_matches(['a', 'f']));
    let arch_a_prefix = format!("atlas_fi_gdn_{arch_a}_{target_arch}_prefill");
    if arch_a_prefix != prefixes[0] {
        prefixes.push(arch_a_prefix);
    }
    if arch.starts_with("sm_121") && target_arch == "aarch64" {
        prefixes.push("atlas_fi_gdn_sm121a_prefill".to_owned());
    }

    for prefix in prefixes {
        let objs = [
            csrc.join(format!("{prefix}.o")),
            csrc.join(format!("{prefix}_f16.o")),
            csrc.join(format!("{prefix}_f16_nf8.o")),
        ];
        if objs.iter().all(|p| p.is_file()) {
            return Some(objs);
        }
    }
    None
}

fn main() {
    release_path_remap::configure();
    native_command_trace::configure();
    println!("cargo:rustc-check-cfg=cfg(atlas_qwen4_fi_attn)");
    println!("cargo:rustc-check-cfg=cfg(atlas_fi_gdn)");
    println!("cargo:rerun-if-env-changed=FLASHINFER_HOME");
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=ATLAS_NVCC");
    println!("cargo:rerun-if-env-changed=ATLAS_CXX");
    println!("cargo:rerun-if-env-changed=ATLAS_FI_GDN_ARCH");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    println!("cargo:rerun-if-env-changed=FLUX_TARGET_HW");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if matches!(
        std::env::var("ATLAS_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        return;
    }
    // Only the CUDA build can link the cubin-only AOT object.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let csrc = manifest_dir.join("csrc").join("fi_gdn");
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("spark-model must live at <workspace>/crates/spark-model")
        .to_path_buf();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")
        .unwrap_or_else(|_| std::env::consts::ARCH.to_owned());

    // ── Compute target: mirror atlas-kernels' HARDWARE.toml arch ──
    let arch = if let Ok(a) = std::env::var("ATLAS_FI_GDN_ARCH")
        && !a.trim().is_empty()
    {
        a.trim().to_owned()
    } else {
        let raw_hw = std::env::var("ATLAS_TARGET_HW")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                std::env::var("FLUX_TARGET_HW")
                    .ok()
                    .filter(|v| !v.trim().is_empty())
            })
            .unwrap_or_else(|| "gb10".into());
        let hw = canonical_hw_name(&raw_hw);
        let hw_toml = workspace_root
            .join("kernels")
            .join(hw)
            .join("HARDWARE.toml");
        // Spelled under the manifest dir so cargo stores it package-relative:
        // an absolute path into one worktree would read as unchanged from
        // another worktree sharing the target dir (atlas-kernels B1).
        let rel_hw_toml = manifest_dir
            .join("../../kernels")
            .join(hw)
            .join("HARDWARE.toml");
        if rel_hw_toml.exists() {
            println!("cargo:rerun-if-changed={}", rel_hw_toml.display());
        }
        std::fs::read_to_string(&hw_toml)
            .ok()
            .and_then(|text| parse_hardware_arch(&text))
            .unwrap_or_else(|| "sm_121a".into())
    };

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let cuda_libdir = format!("{cuda_home}/lib64");
    let cuda_inc = format!("{cuda_home}/include");

    let cuda13 = PathBuf::from("/usr/local/cuda-13.0/bin/nvcc");
    let nvcc = find_tool(
        "ATLAS_NVCC",
        &[
            PathBuf::from(&cuda_home).join("bin/nvcc"),
            cuda13,
            PathBuf::from("/usr/local/cuda/bin/nvcc"),
        ],
        "nvcc",
    );
    let cxx = find_tool("ATLAS_CXX", &[PathBuf::from("/usr/bin/g++")], "g++");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if csrc.is_dir() {
        println!("cargo:rerun-if-changed={}", csrc.display());
    }

    if let Some([aot_obj, aot_f16_obj, aot_f16_nf8_obj]) =
        find_fi_gdn_aot_objects(&csrc, &arch, &target_arch)
    {
        if !Path::new(&cuda_inc).exists() {
            panic!("spark-model build: CUDA include dir {cuda_inc} not found; set CUDA_HOME");
        }
        let bridge_cu = csrc.join("fi_gdn_bridge.cu");
        let shim_cpp = csrc.join("abi_shim.cpp");
        // Hand-maintained FP16 export bridge (q/k/v/o float16, `_f16` suffix),
        // including the strip API, 32x33 transpose, and L2-discard helpers.
        let bridge_f16_cu = csrc.join("fi_gdn_bridge_f16.cu");
        // Norm->e4m3 epilogue variant of the FP16 export and its bridge.
        let bridge_f16_nf8_cu = csrc.join("fi_gdn_bridge_f16_nf8.cu");
        for f in [
            &bridge_cu,
            &shim_cpp,
            &aot_obj,
            &bridge_f16_cu,
            &aot_f16_obj,
            &bridge_f16_nf8_cu,
            &aot_f16_nf8_obj,
        ] {
            assert!(f.exists(), "spark-model build: missing {}", f.display());
            println!("cargo:rerun-if-changed={}", f.display());
        }

        let bridge_o = out_dir.join("fi_gdn_bridge.o");
        let bridge_f16_o = out_dir.join("fi_gdn_bridge_f16.o");
        let bridge_f16_nf8_o = out_dir.join("fi_gdn_bridge_f16_nf8.o");
        let shim_o = out_dir.join("abi_shim.o");

        run_native(
            &bridge_cu,
            &bridge_o,
            "nvcc",
            compiler_with_release_path_remap(&nvcc, true)
                .arg("-O2")
                .arg("-std=c++17")
                .arg(format!("-arch={arch}"))
                .arg("-I")
                .arg(&cuda_inc)
                .arg("-I")
                .arg(&csrc)
                .arg("-c")
                .arg(&bridge_cu)
                .arg("-o")
                .arg(&bridge_o),
            "nvcc fi_gdn_bridge.cu",
        );

        run_native(
            &bridge_f16_cu,
            &bridge_f16_o,
            "nvcc",
            compiler_with_release_path_remap(&nvcc, true)
                .arg("-O2")
                .arg("-std=c++17")
                .arg(format!("-arch={arch}"))
                .arg("-I")
                .arg(&cuda_inc)
                .arg("-I")
                .arg(&csrc)
                .arg("-c")
                .arg(&bridge_f16_cu)
                .arg("-o")
                .arg(&bridge_f16_o),
            "nvcc fi_gdn_bridge_f16.cu",
        );

        run_native(
            &bridge_f16_nf8_cu,
            &bridge_f16_nf8_o,
            "nvcc",
            compiler_with_release_path_remap(&nvcc, true)
                .arg("-O2")
                .arg("-std=c++17")
                .arg(format!("-arch={arch}"))
                .arg("-I")
                .arg(&cuda_inc)
                .arg("-I")
                .arg(&csrc)
                .arg("-c")
                .arg(&bridge_f16_nf8_cu)
                .arg("-o")
                .arg(&bridge_f16_nf8_o),
            "nvcc fi_gdn_bridge_f16_nf8.cu",
        );

        run_native(
            &shim_cpp,
            &shim_o,
            "cxx",
            compiler_with_release_path_remap(&cxx, false)
                .arg("-O2")
                .arg("-std=c++17")
                .arg("-I")
                .arg(&cuda_inc)
                .arg("-c")
                .arg(&shim_cpp)
                .arg("-o")
                .arg(&shim_o),
            "g++ abi_shim.cpp",
        );

        // Archive the bridge, the shim, and the prebuilt AOT object together so
        // the linker pulls them as one unit with correct ordering vs -lcudart.
        let lib = out_dir.join("libatlas_fi_gdn.a");
        let _ = std::fs::remove_file(&lib);
        run(
            Command::new("ar")
                .arg("rcs")
                .arg(&lib)
                .arg(&bridge_o)
                .arg(&shim_o)
                .arg(&aot_obj)
                .arg(&bridge_f16_o)
                .arg(&aot_f16_obj)
                .arg(&bridge_f16_nf8_o)
                .arg(&aot_f16_nf8_obj),
            "ar rcs libatlas_fi_gdn.a",
        );

        println!("cargo:rustc-link-lib=static=atlas_fi_gdn");
        println!("cargo:rustc-cfg=atlas_fi_gdn");
    } else {
        println!(
            "cargo:warning=spark-model: no FlashInfer GDN AOT objects for arch={arch} target_arch={target_arch}; skipping libatlas_fi_gdn.a (native CUDA GDN prefill fallback active)"
        );
    }

    // FlashInfer FA2 ragged prefill for Flash-Next wide-chunk attention
    // (`csrc/fi_attn/`), when FLASHINFER_HOME (or bundled `/opt/flux/fi_home`)
    // points at a FlashInfer tree with `include/` and its pinned `3rdparty/cccl`
    // (which must precede the CUDA 13 toolkit CCCL).
    let fi_home_opt = std::env::var_os("FLASHINFER_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            let bundled = Path::new("/opt/flux/fi_home");
            if bundled.join("include").is_dir() && bundled.join("3rdparty/cccl").is_dir() {
                Some(bundled.to_path_buf())
            } else {
                None
            }
        });
    if let Some(fi_home) = fi_home_opt {
        if !Path::new(&cuda_inc).exists() {
            panic!("spark-model build: CUDA include dir {cuda_inc} not found; set CUDA_HOME");
        }
        let fi_cu = manifest_dir
            .join("csrc")
            .join("fi_attn")
            .join("qwen4_fi_attn.cu");
        println!("cargo:rerun-if-changed={}", fi_cu.display());
        let cccl = fi_home.join("3rdparty").join("cccl");
        let fi_o = out_dir.join("qwen4_fi_attn.o");
        let fi_home_str = fi_home.display().to_string();
        let mut fi_cmd = compiler_with_release_path_remap(&nvcc, true);
        for opt in [
            "-ffile-prefix-map",
            "-fmacro-prefix-map",
            "-fdebug-prefix-map",
        ] {
            fi_cmd
                .arg("-Xcompiler")
                .arg(format!("{opt}={fi_home_str}=/opt/flux/fi_home"));
        }
        run_native(
            &fi_cu,
            &fi_o,
            "nvcc",
            fi_cmd
                .arg("-c")
                .arg("-O3")
                .arg("-std=c++17")
                .arg("--expt-relaxed-constexpr")
                .arg("-Xcompiler")
                .arg("-fPIC")
                .arg(format!("-arch={arch}"))
                .arg("-isystem")
                .arg(cccl.join("libcudacxx").join("include"))
                .arg("-isystem")
                .arg(cccl.join("cub"))
                .arg("-isystem")
                .arg(cccl.join("thrust"))
                .arg(format!("-I{}", fi_home.join("include").display()))
                .arg(&fi_cu)
                .arg("-o")
                .arg(&fi_o),
            "nvcc qwen4_fi_attn.cu",
        );
        let fi_lib = out_dir.join("libatlas_qwen4_fi_attn.a");
        let _ = std::fs::remove_file(&fi_lib);
        run(
            Command::new("ar").arg("rcs").arg(&fi_lib).arg(&fi_o),
            "ar rcs libatlas_qwen4_fi_attn.a",
        );
        println!("cargo:rustc-link-lib=static=atlas_qwen4_fi_attn");
        println!("cargo:rustc-cfg=atlas_qwen4_fi_attn");
    }

    // CUDA runtime + driver, and the C++ runtime the AOT host code needs.
    println!("cargo:rustc-link-search=native={cuda_libdir}");
    println!("cargo:rustc-link-search=native={cuda_libdir}/stubs");
    match target_arch.as_str() {
        "aarch64" => println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu"),
        "x86_64" => println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu"),
        _ => {}
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
