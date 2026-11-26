// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::{BTreeSet, HashMap};
use std::env;
use std::path::PathBuf;

#[path = "build_target_selector.rs"]
mod build_target_selector;
use build_target_selector::resolve_model_names;

#[path = "build_deps.rs"]
mod build_deps;

#[path = "build_hw.rs"]
mod build_hw;

#[path = "build_source_root.rs"]
mod build_source_root;

/// Per-category sampling defaults parsed from MODEL.toml `[sampling.*]`.
#[derive(Debug, Clone)]
struct SamplingCat {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    // DRY sampler params (see SamplingCategory in atlas-kernels/src/lib.rs
    // for full rationale). Defaults disable DRY; individual MODEL.toml
    // `[sampling.*]` tables opt in when needed.
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: u32,
    // LZ penalty (arXiv:2504.20131). Frequency-weighted n-gram penalty
    // over the recent token window. 0.0 = disabled. 0.2 is the SGLang
    // reference value; lossless on AIME/GPQA at that strength.
    lz_penalty: f32,
    // None means MODEL.toml is silent, so --default-min-p still owns it.
    // Some(0.0) is intentionally distinct from absence.
    min_p: Option<f32>,
    // None means MODEL.toml is silent, so --default-top-n-sigma still owns it.
    // Some(0.0) is intentionally distinct from absence.
    top_n_sigma: Option<f32>,
}

impl Default for SamplingCat {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            min_p: None,
            top_n_sigma: None,
        }
    }
}

/// A `(model_type, optional hidden_size)` pair declaring which models a kernel target supports.
#[derive(Debug, Clone)]
struct ModelTypeMatch {
    model_type: String,
    hidden_size: Option<usize>,
}

/// A resolved (hw, model, quant) compilation target.
struct Target {
    hw: String,
    model: String,
    quant: String,
    arch: String,
    /// Ordered hardware kernel directories (child first, base ancestor(s) last).
    hw_dirs: Vec<PathBuf>,
    /// Ordered `-I` include directories for inherited targets (empty when non-inherited).
    include_dirs: Vec<PathBuf>,
    extra_flags: Vec<String>,
    module_overrides: HashMap<String, String>,
    sampling_thinking_text: SamplingCat,
    sampling_thinking_coding: SamplingCat,
    sampling_non_thinking: SamplingCat,
    sampling_tools: SamplingCat,
    behavior_thinking_in_tools: bool,
    behavior_max_thinking_budget: u32,
    behavior_thinking_default: bool,
    behavior_honor_eos_inside_thinking: bool,
    behavior_fp8_kv_calibration_tokens: usize,
    behavior_default_kv_dtype: String,
    behavior_default_num_drafts: u32,
    behavior_disable_tool_steering: bool,
    behavior_tool_call_parser: String,
    behavior_enable_loop_watchdog: bool,
    // Extended behavior fields (13 fields) used by build_codegen.rs. Default
    // to safe values (false / 0 / empty) until a model wants to customize.
    behavior_think_loop_min_repeats: u32,
    behavior_think_loop_scan_window: u32,
    behavior_confidence_early_stop: bool,
    behavior_confidence_run_length: u32,
    behavior_fuzzy_repeat_tolerance_div: u32,
    behavior_max_inter_tool_prose: u32,
    behavior_tscg: bool,
    behavior_disable_tool_grammar: bool,
    behavior_rollback_resteer: bool,
    behavior_rom_head: String,
    behavior_preserve_thinking: Option<bool>,
    behavior_thinking_answer_reserve: u32,
    behavior_thinking_sampling_in_tools: bool,
    /// Which `(model_type, hidden_size)` pairs this kernel target supports.
    /// Parsed from `[[model_types]]` in MODEL.toml.
    model_type_matches: Vec<ModelTypeMatch>,
    /// `[dflash]` section if present in MODEL.toml — drafter pairing for
    /// block-diffusion speculative decoding. `None` when the model has no
    /// associated DFlash drafter checkpoint.
    dflash: Option<DflashRaw>,
}

#[derive(Default, Clone)]
struct DflashRaw {
    draft_model: String,
    gamma: usize,
    window_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
}

// Cargo treats a missing watched path as permanently stale. Common-only
// targets may have no model override directory; watch its existing ancestor
// so creating an override still invalidates the build without constant rebuilds.
fn nearest_existing_path(path: &std::path::Path) -> Option<PathBuf> {
    let mut watched = path;
    loop {
        if watched.exists() {
            return Some(watched.to_path_buf());
        }
        watched = watched.parent()?;
    }
}

fn rerun_if_changed(path: &std::path::Path) {
    if let Some(watched) = nearest_existing_path(path) {
        println!("cargo:rerun-if-changed={}", watched.display());
    }
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=ATLAS_NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    // Without these the cargo cache short-circuits when only the target
    // selection env vars change, leaving an out-of-date kernel registry
    // baked into the binary (e.g. defaulting to qwen3-next-only after a
    // prior `ATLAS_TARGET_MODEL=*` build).
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_MODEL");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_QUANT");
    println!("cargo:rerun-if-env-changed=FLUX_TARGET_HW");
    println!("cargo:rerun-if-env-changed=FLUX_TARGET_MODEL");
    println!("cargo:rerun-if-env-changed=FLUX_TARGET_QUANT");
    println!("cargo:rerun-if-env-changed=ATLAS_EXLLAMAV3_SOURCE");
    println!("cargo:rerun-if-env-changed=ATLAS_OBFUSCATE_PTX");
    println!("cargo:rerun-if-env-changed=ATLAS_EMBED_CUBIN");
    println!("cargo:rerun-if-env-changed=ATLAS_EMBED_AOT_CUBIN");
    // Set per tree by .cargo/config.toml: a shared CARGO_TARGET_DIR reruns
    // this script when another worktree builds (see build_source_root.rs).
    println!(
        "cargo:rerun-if-env-changed={}",
        build_source_root::SOURCE_ROOT_ENV
    );
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    {
        let declared = env::var(build_source_root::SOURCE_ROOT_ENV).ok();
        if let Some(warning) =
            build_source_root::source_root_warning(declared.as_deref(), workspace_root)
        {
            println!("cargo:warning=atlas-kernels: {warning}");
        }
    }

    // Resolve hardware target + any `extends` base chain once so drafter
    // profiles, vendor detection, and target resolution all share the exact
    // same alias canonicalization and inheritance chain.
    let hw = build_hw::resolve_hw_env(
        env::var("ATLAS_TARGET_HW").ok().as_deref(),
        env::var("FLUX_TARGET_HW").ok().as_deref(),
    );
    let kernels_root = workspace_root.join("kernels");
    let hw_chain = build_hw::resolve_hw_chain(&kernels_root, &hw)
        .unwrap_or_else(|e| panic!("atlas-kernels: {e}"));
    for toml_path in &hw_chain.hw_toml_paths {
        println!("cargo:rerun-if-changed={}", toml_path.display());
    }

    // The drafter-profile table needs no nvcc, so it is generated before the
    // ATLAS_SKIP_BUILD early return: stub builds (cargo test on CPU hosts)
    // admit and refuse drafters exactly as a real build does.
    {
        let profiles = match hw_chain.hw_dirs.as_slice() {
            [single] => build_profile::generate_drafter_profiles_rs(single),
            layered => build_profile::generate_drafter_profiles_rs_layered(layered),
        };
        std::fs::write(out_dir.join("drafter_profiles.rs"), profiles)
            .expect("write drafter_profiles.rs");
    }
    if matches!(
        env::var("ATLAS_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    ) {
        let stub = "// Auto-generated by build.rs (ATLAS_SKIP_BUILD stub — no nvcc invoked).\n\
            pub(crate) fn register_embedded_aot_cubins() {}\n\
            pub fn ptx_modules() -> Vec<(&'static str, &'static str)> {\n\
                register_embedded_aot_cubins();\n\
                Vec::new()\n\
            }\n\
            pub fn all_ptx_sets() -> Vec<TargetPtxSet> {\n\
                register_embedded_aot_cubins();\n\
                Vec::new()\n\
            }\n";
        std::fs::write(out_dir.join("target_ptx.rs"), stub)
            .expect("write ATLAS_SKIP_BUILD stub target_ptx.rs");
        println!("cargo:rustc-env=ATLAS_PTX_DIR={}", out_dir.display());
        return;
    }

    // ── Resolve targets (supports wildcards and `extends` inheritance) ──
    let targets = resolve_targets(&hw_chain);

    assert!(
        !targets.is_empty(),
        "No kernel targets resolved. Check ATLAS_TARGET_* env vars."
    );

    // ── Resolve compute target (compiler) from HARDWARE.toml vendor ──
    // This abstraction supports NVIDIA (nvcc→PTX), AMD (hipcc→HSACO),
    // Apple (xcrun→metallib), Intel (icpx→SPIR-V). Only NVIDIA and Apple
    // are implemented.
    let compute_target = resolve_compute_target(hw_chain.vendor.as_deref());
    let output_ext = compute_target.output_extension();
    // Compiled kernels and their signatures, per source tree: the mtime
    // cache below must never compare this tree's sources to another tree's PTX.
    let ptx_dir = build_source_root::ptx_dir(&out_dir, workspace_root);
    std::fs::create_dir_all(&ptx_dir)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", ptx_dir.display()));

    let obfuscate = matches!(
        env::var("ATLAS_OBFUSCATE_PTX").as_deref(),
        Ok("1") | Ok("true")
    );
    let embed_cubin_env = env::var("ATLAS_EMBED_CUBIN")
        .ok()
        .or_else(|| env::var("ATLAS_EMBED_AOT_CUBIN").ok());
    let ptxas_path = find_ptxas();
    let mut ptxas_missing_warned = false;

    // Per-target compiled module descriptors (stem, module_name, optional AOT CUBIN).
    let mut all_target_modules: Vec<Vec<ModuleArtifact>> = Vec::new();

    for (idx, target) in targets.iter().enumerate() {
        let source_ext = compute_target.source_extension();
        let cu_files = build_hw::collect_kernel_files_layered(
            &target.hw_dirs,
            &target.model,
            &target.quant,
            source_ext,
        );
        assert!(
            !cu_files.is_empty(),
            "No .{} files found for target ({}, {}, {})",
            source_ext,
            target.hw,
            target.model,
            target.quant,
        );

        // Compile all kernel source files via the ComputeTarget abstraction.
        // The NvidiaTarget uses nvcc; future targets (AMD, Apple, Intel) would
        // use their respective compilers.
        //
        // PTX-cache freshness check: skip nvcc when an existing PTX is newer
        // than every potentially-relevant source (.cu + every .cuh in the
        // model and common dirs) AND the target-identity signature for this
        // `t{idx}__` slot matches the prior build. The signature guards
        // against the case where ATLAS_TARGET_MODEL changes and re-numbers
        // the targets — without it we'd happily reuse a stale PTX from a
        // different model that happened to land at the same `idx`.
        // The nvcc flags are part of the identity too: a KERNEL.toml flag
        // change (or the fast-math opt-out) touches no .cu/.cuh, so without
        // them here every cached PTX would be reused, compiled under the OLD
        // flags, while cargo reports success.
        let signature = format!(
            "{}|{}|{}|{}|{}",
            target.hw,
            target.model,
            target.quant,
            target.arch,
            target.extra_flags.join(" "),
        );
        let signature_file = ptx_dir.join(format!("t{idx}__signature"));
        let signature_matches = std::fs::read_to_string(&signature_file)
            .map(|s| s == signature)
            .unwrap_or(false);
        if !signature_matches {
            let _ = std::fs::write(&signature_file, &signature);
        }
        let header_floor = target_header_floor(
            &target.hw_dirs,
            &target.model,
            &target.quant,
            &signature_file,
        );

        let mut errors = Vec::new();
        let mut compiled = 0usize;
        let mut cached = 0usize;
        // Files each `.cu` reaches through quoted #includes (wrappers around
        // another target's source or inherited headers): they gate the cache
        // and cargo's rerun.
        let include_deps: Vec<Vec<PathBuf>> = cu_files
            .iter()
            .map(|f| {
                if target.include_dirs.is_empty() {
                    build_deps::include_closure(f)
                } else {
                    build_deps::include_closure_with_dirs(f, &target.include_dirs)
                }
            })
            .collect();
        for (cu_file, deps) in cu_files.iter().zip(&include_deps) {
            let stem = cu_file.file_stem().unwrap().to_str().unwrap().to_string();
            let out_file = ptx_dir.join(format!("t{idx}__{stem}.{output_ext}"));
            let deps_file = ptx_dir.join(format!("t{idx}__{stem}.{output_ext}.deps"));
            let deps_manifest = build_deps::dependency_manifest(cu_file, deps).ok();
            let deps_match = deps_manifest.as_ref().is_some_and(|manifest| {
                std::fs::read_to_string(&deps_file).is_ok_and(|old| old == *manifest)
            });

            // Per-file freshness: skip nvcc when output exists and is
            // newer than max(this .cu's mtime, target header floor).
            // The header floor folds in every `.cuh` in the model+common
            // dirs plus the signature file mtime, so touching a header
            // invalidates every `.cu` in the target while touching a
            // single `.cu` only invalidates itself. Conservative —
            // false negatives (recompile when unchanged) are a minor
            // perf miss; false positives (skip when changed) would
            // silently ship stale PTX.
            if signature_matches && deps_match && compute_target.allow_cached_output() {
                let own_mt = std::fs::metadata(cu_file).and_then(|m| m.modified()).ok();
                let cu_mt = match (own_mt, build_deps::latest_mtime(deps)) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                };
                let out_mt = std::fs::metadata(&out_file).and_then(|m| m.modified()).ok();
                let src_floor = match (cu_mt, header_floor) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
                if let (Some(out_mt), Some(src_mt)) = (out_mt, src_floor)
                    && out_mt > src_mt
                {
                    cached += 1;
                    continue;
                }
            }

            // Invalidate before compiling: a failed compiler may leave a partial
            // output, which must never retain a successful dependency receipt.
            if let Err(error) = std::fs::remove_file(&deps_file)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                errors.push(format!(
                    "Failed to invalidate {}: {error}",
                    deps_file.display()
                ));
                continue;
            }
            if let Err(e) =
                compute_target.compile(cu_file, &out_file, &target.arch, &target.extra_flags)
            {
                errors.push(e);
            } else {
                compiled += 1;
                if let Some(manifest) = deps_manifest
                    && let Err(error) = std::fs::write(&deps_file, manifest)
                {
                    errors.push(format!("Failed to write {}: {error}", deps_file.display()));
                }
            }
        }
        if !errors.is_empty() {
            panic!("Kernel compilation failed:\n{}", errors.join("\n"));
        }
        if cached > 0 {
            println!(
                "cargo:warning=atlas-kernels: target {idx} ({}, {}, {}): {} cached, {} compiled",
                target.hw, target.model, target.quant, cached, compiled
            );
        }

        for dep in include_deps
            .iter()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>()
        {
            println!("cargo:rerun-if-changed={}", dep.display());
        }
        for cu_file in &cu_files {
            println!("cargo:rerun-if-changed={}", cu_file.display());
            if obfuscate {
                let stem = cu_file.file_stem().unwrap().to_str().unwrap();
                let out_file = ptx_dir.join(format!("t{idx}__{stem}.{output_ext}"));
                let obf_file = ptx_dir.join(format!("t{idx}__{stem}.{output_ext}.obf"));
                let raw = std::fs::read(&out_file)
                    .unwrap_or_else(|e| panic!("Failed to read {}: {e}", out_file.display()));
                let obf = build_codegen::obfuscate_ptx_bytes(&raw);
                std::fs::write(&obf_file, &obf)
                    .unwrap_or_else(|e| panic!("Failed to write {}: {e}", obf_file.display()));
            }
        }

        // Collect module descriptors sorted by module_name.
        let mut modules: Vec<ModuleArtifact> = cu_files
            .iter()
            .map(|f| {
                let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
                let module_name = target
                    .module_overrides
                    .get(&stem)
                    .cloned()
                    .unwrap_or_else(|| stem.clone());
                ModuleArtifact {
                    stem,
                    module_name,
                    aot_cubin: None,
                }
            })
            .collect();
        modules.sort_by(|a, b| a.module_name.cmp(&b.module_name));

        // Optional build-time AOT CUBIN assembly via parallel `ptxas` for
        // single-target NVIDIA builds (or explicit ATLAS_EMBED_CUBIN=1).
        if build_codegen::should_embed_cubin(
            compute_target.output_is_text(),
            targets.len(),
            &target.arch,
            embed_cubin_env.as_deref(),
        ) {
            match ptxas_path.as_deref() {
                Some(ptxas) => {
                    compile_aot_cubins_parallel(
                        ptxas,
                        &ptx_dir,
                        idx,
                        output_ext,
                        target,
                        signature_matches,
                        compute_target.allow_cached_output(),
                        obfuscate,
                        &mut modules,
                    );
                }
                None => {
                    if !ptxas_missing_warned {
                        println!(
                            "cargo:warning=atlas-kernels: ptxas not found; skipping AOT CUBIN embedding (runtime PTX fallback active)"
                        );
                        ptxas_missing_warned = true;
                    }
                }
            }
        }

        all_target_modules.push(modules);

        let mut watched_dirs = BTreeSet::new();
        for hw_dir in &target.hw_dirs {
            for candidate in [
                hw_dir.join(&target.model).join(&target.quant),
                hw_dir.join(&target.quant),
                hw_dir.join("common"),
            ] {
                if let Some(existing) = nearest_existing_path(&candidate) {
                    watched_dirs.insert(existing);
                }
            }
        }
        for dir in watched_dirs {
            println!("cargo:rerun-if-changed={}", dir.display());
        }
        let override_stems: BTreeSet<String> = target
            .hw_dirs
            .iter()
            .flat_map(|hw_dir| {
                build_hw::find_kernel_files(
                    &hw_dir.join(&target.model).join(&target.quant),
                    source_ext,
                )
            })
            .filter_map(|f| f.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
            .collect();
        let n_overrides = override_stems.len();
        println!(
            "cargo:warning=atlas-kernels: compiled {} kernels for target {} ({}, {}, {}){}",
            cu_files.len(),
            idx,
            target.hw,
            target.model,
            target.quant,
            if n_overrides > 0 {
                format!(" ({n_overrides} model-specific overrides)")
            } else {
                String::new()
            },
        );
    }

    // ── Generate target_ptx.rs ──
    let output_is_text_val = compute_target.output_is_text();
    let generated = generate_target_ptx_rs(
        &targets,
        &all_target_modules,
        output_ext,
        output_is_text_val,
    );
    let gen_path = out_dir.join("target_ptx.rs");
    std::fs::write(&gen_path, &generated)
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", gen_path.display()));

    println!("cargo:rustc-env=ATLAS_PTX_DIR={}", ptx_dir.display());
}

/// Resolve all compilation targets from env vars, expanding wildcards and
/// honoring `[hardware].extends` inheritance chains.
fn resolve_targets(hw_chain: &build_hw::HwChain) -> Vec<Target> {
    let model_spec = env::var("ATLAS_TARGET_MODEL")
        .or_else(|_| env::var("FLUX_TARGET_MODEL"))
        .unwrap_or_else(|_| "qwen3.8-flash-next".into());
    let quant_spec = env::var("ATLAS_TARGET_QUANT")
        .or_else(|_| env::var("FLUX_TARGET_QUANT"))
        .unwrap_or_else(|_| "nvfp4".into());

    // Propagate hardware config flags to Rust code via compile-time env vars
    if let Some(fp32_res) = hw_chain.use_fp32_residual {
        println!(
            "cargo:rustc-env=ATLAS_HW_FP32_RESIDUAL={}",
            if fp32_res { "true" } else { "false" }
        );
    }

    // Resolve wildcard, singular, or strict comma-separated exact model names.
    // Only directories with MODEL.toml are registered model targets; common
    // quant and tooling directories are never admitted by an exact selector.
    let mut available_set = BTreeSet::new();
    for hw_dir in &hw_chain.hw_dirs {
        for d in list_subdirs(hw_dir) {
            if hw_dir.join(&d).join("MODEL.toml").is_file() {
                available_set.insert(d);
            }
        }
    }
    let available_models: Vec<String> = available_set.into_iter().collect();
    let models = resolve_model_names(&model_spec, available_models)
        .unwrap_or_else(|e| panic!("Invalid ATLAS_TARGET_MODEL={model_spec:?}: {e}"));

    let vendor = hw_chain.vendor.as_deref().unwrap_or("nvidia");
    let child_hw_dir = &hw_chain.hw_dirs[0];
    let mut targets = Vec::new();
    for model in &models {
        let model_meta_dir = build_hw::resolve_model_meta_dir(&hw_chain.hw_dirs, model)
            .unwrap_or_else(|| {
                panic!(
                    "Model kernel directory with MODEL.toml not found for {model:?} in {:?}",
                    hw_chain.hw_dirs
                )
            });
        if hw_chain.is_inherited() {
            rerun_if_changed(&child_hw_dir.join(model).join("MODEL.toml"));
        }

        // Expand quant wildcard across all hardware dirs in the inheritance chain.
        let quants: Vec<String> = if quant_spec == "*" {
            // A quant target has a KERNEL.toml of its own or a common quant dir. DeepSeek-V4.1
            // keeps gate/tooling dirs (attn/, fp8gemm/) beside its one target (cb3/); those
            // are not targets.
            let mut quant_set = BTreeSet::new();
            for hw_dir in &hw_chain.hw_dirs {
                let model_dir = hw_dir.join(model);
                if !model_dir.is_dir() {
                    continue;
                }
                for q in list_subdirs(&model_dir) {
                    let has_toml = hw_chain
                        .hw_dirs
                        .iter()
                        .any(|d| d.join(model).join(&q).join("KERNEL.toml").is_file());
                    let has_quant_common = hw_chain.hw_dirs.iter().any(|d| d.join(&q).is_dir());
                    if has_toml || has_quant_common {
                        quant_set.insert(q);
                    }
                }
            }
            quant_set.into_iter().collect()
        } else {
            vec![quant_spec.clone()]
        };

        for quant in &quants {
            let has_any_kernel_dir = hw_chain.hw_dirs.iter().any(|hw_dir| {
                hw_dir.join(model).join(quant).is_dir()
                    || hw_dir.join(quant).is_dir()
                    || hw_dir.join("common").is_dir()
            });
            assert!(
                has_any_kernel_dir,
                "No kernel directory found for ({model}, {quant}) across {:?}.",
                hw_chain.hw_dirs,
            );

            // KERNEL.toml: prefer model-specific (child -> base), fall back to quant-common (child -> base).
            let toml_dir = build_hw::resolve_kernel_toml_dir(&hw_chain.hw_dirs, model, quant)
                .unwrap_or_else(|| {
                    panic!(
                        "No KERNEL.toml found for ({model}, {quant}) across {:?}",
                        hw_chain.hw_dirs
                    )
                });
            if hw_chain.is_inherited() {
                rerun_if_changed(&child_hw_dir.join(model).join(quant).join("KERNEL.toml"));
                rerun_if_changed(&child_hw_dir.join(quant).join("KERNEL.toml"));
            }

            let (mut extra_flags, module_overrides) = parse_kernel_toml(&toml_dir, vendor);
            if hw_chain.includes_hw("gb10") && model == "glm5.3-flash" && quant == "exl3" {
                extra_flags.extend(pinned_exllamav3_flags());
            }

            // For inherited hardware targets, append child-first `-I` include paths
            // so child `.cuh` overrides win cleanly and inherited headers resolve.
            let include_dirs = if hw_chain.is_inherited() {
                let dirs = build_hw::collect_include_dirs(&hw_chain.hw_dirs, model, quant);
                for dir in &dirs {
                    extra_flags.push(format!("-I{}", dir.display()));
                }
                dirs
            } else {
                Vec::new()
            };

            let (s_tt, s_tc, s_nt, s_tools) = parse_sampling_presets(&model_meta_dir);
            let pb = parse_behavior(&model_meta_dir);
            let b_thinking_in_tools = pb.thinking_in_tools;
            let b_max_thinking_budget = pb.max_thinking_budget;
            let b_thinking_default = pb.thinking_default;
            let b_fp8_kv_cal = pb.fp8_kv_calibration_tokens;
            let b_default_kv_dtype = pb.default_kv_dtype.clone();
            let b_default_num_drafts = pb.default_num_drafts;
            let b_disable_tool_steering = pb.disable_tool_steering;
            let b_tool_call_parser = pb.tool_call_parser.clone();
            let b_enable_loop_watchdog = pb.enable_loop_watchdog;
            let model_type_matches = parse_model_types(&model_meta_dir);
            let dflash = parse_dflash(&model_meta_dir);

            targets.push(Target {
                hw: hw_chain.hw.clone(),
                model: model.clone(),
                quant: quant.clone(),
                arch: hw_chain.arch.clone(),
                hw_dirs: hw_chain.hw_dirs.clone(),
                include_dirs,
                extra_flags,
                module_overrides,
                sampling_thinking_text: s_tt,
                sampling_thinking_coding: s_tc,
                sampling_non_thinking: s_nt,
                sampling_tools: s_tools,
                behavior_thinking_in_tools: b_thinking_in_tools,
                behavior_max_thinking_budget: b_max_thinking_budget,
                behavior_thinking_default: b_thinking_default,
                behavior_honor_eos_inside_thinking: pb.honor_eos_inside_thinking,
                behavior_fp8_kv_calibration_tokens: b_fp8_kv_cal,
                behavior_default_kv_dtype: b_default_kv_dtype,
                behavior_default_num_drafts: b_default_num_drafts,
                behavior_disable_tool_steering: b_disable_tool_steering,
                behavior_tool_call_parser: b_tool_call_parser,
                behavior_enable_loop_watchdog: b_enable_loop_watchdog,
                behavior_think_loop_min_repeats: pb.think_loop_min_repeats,
                behavior_think_loop_scan_window: pb.think_loop_scan_window,
                behavior_confidence_early_stop: pb.confidence_early_stop,
                behavior_confidence_run_length: pb.confidence_run_length,
                behavior_fuzzy_repeat_tolerance_div: pb.fuzzy_repeat_tolerance_div,
                behavior_max_inter_tool_prose: pb.max_inter_tool_prose,
                behavior_tscg: pb.tscg,
                behavior_disable_tool_grammar: pb.disable_tool_grammar,
                behavior_rollback_resteer: pb.rollback_resteer,
                behavior_rom_head: pb.rom_head.clone(),
                behavior_preserve_thinking: pb.preserve_thinking,
                behavior_thinking_answer_reserve: pb.thinking_answer_reserve,
                behavior_thinking_sampling_in_tools: pb.thinking_sampling_in_tools,
                model_type_matches,
                dflash,
            });
        }
    }

    // Sort by (model, quant) for deterministic ordering
    targets.sort_by(|a, b| (&a.model, &a.quant).cmp(&(&b.model, &b.quant)));
    targets
}

/// Admit the exact ExLlamaV3 source used by the published one-Spark recipe.
///
/// Atlas intentionally does not fall back to a system installation or a
/// floating checkout: EXL3's packed layout and generated symbol ABI are tied
/// to this revision. The thin model compunits include the admitted sources.
fn pinned_exllamav3_flags() -> Vec<String> {
    const REVISION: &str = "e648f1a131365aae15920073e761a3fa5a527654";
    const LICENSE_SHA256: &str = "27a32b6263fcd96c79d3beeecf221c4366780bdf15ad51986f48650bd7369bff";

    let checkout = PathBuf::from(env::var("ATLAS_EXLLAMAV3_SOURCE").unwrap_or_else(|_| {
        panic!(
            "GLM-5.3-Flash EXL3 requires ATLAS_EXLLAMAV3_SOURCE pointing to clean revision {REVISION}"
        )
    }))
    .canonicalize()
    .unwrap_or_else(|e| panic!("cannot resolve ATLAS_EXLLAMAV3_SOURCE: {e}"));

    let git_output = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(&checkout)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("cannot inspect pinned ExLlamaV3 checkout: {e}"));
        assert!(
            output.status.success(),
            "git inspection failed for {}: {}",
            checkout.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
        String::from_utf8(output.stdout)
            .expect("git output is not UTF-8")
            .trim()
            .to_owned()
    };
    let head = git_output(&["rev-parse", "HEAD"]);
    assert_eq!(head, REVISION, "unapproved ExLlamaV3 revision");
    let dirty = git_output(&["status", "--porcelain", "--untracked-files=no"]);
    assert!(dirty.is_empty(), "pinned ExLlamaV3 checkout is dirty");

    let license = checkout.join("LICENSE");
    let digest = std::process::Command::new("sha256sum")
        .arg(&license)
        .output()
        .unwrap_or_else(|e| panic!("cannot hash {}: {e}", license.display()));
    assert!(
        digest.status.success(),
        "sha256sum failed for ExLlamaV3 LICENSE"
    );
    let digest = String::from_utf8(digest.stdout)
        .expect("sha256sum output is not UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum returned no digest")
        .to_owned();
    assert_eq!(digest, LICENSE_SHA256, "unapproved ExLlamaV3 license bytes");

    let extension = checkout.join("exllamav3/exllamav3_ext");
    for bits in [2, 3, 4, 5] {
        let compunit = extension.join(format!("quant/comp_units/exl3_comp_unit_{bits}_cb2.cu"));
        assert!(
            compunit.is_file(),
            "pinned ExLlamaV3 extension source is missing: {}",
            compunit.display()
        );
    }
    let moe_compunit = extension.join("quant/comp_units/exl3_moe_inst_k2_cb2.cu");
    assert!(
        moe_compunit.is_file(),
        "pinned ExLlamaV3 MoE source is missing: {}",
        moe_compunit.display()
    );
    let gemv_kernel = extension.join("quant/exl3_gemv_kernel.cuh");
    assert!(
        gemv_kernel.is_file(),
        "pinned ExLlamaV3 GEMV source is missing: {}",
        gemv_kernel.display()
    );
    println!("cargo:rerun-if-changed={}", license.display());
    println!("cargo:rerun-if-changed={}", extension.display());
    vec![
        "-std=c++17".to_owned(),
        format!("-I{}", extension.display()),
        "-Xcudafe".to_owned(),
        "--diag_suppress=20012".to_owned(),
    ]
}

/// List subdirectory names (not files) in a directory, sorted.
fn list_subdirs(dir: &std::path::Path) -> Vec<String> {
    let mut dirs: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            // Follow symlinks: variant targets (qwen3.6-35b-a3b-abl,
            // qwen3.5-122b-a10b-heretic) symlink their quant dir to the
            // canonical target's, and `DirEntry::file_type` does not follow
            // links, so a wildcard build silently compiled zero kernels for
            // them.
            if std::fs::metadata(entry.path()).ok()?.is_dir() {
                Some(entry.file_name().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    dirs.sort();
    dirs
}

#[path = "build_parse.rs"]
mod build_parse;
use build_parse::{
    parse_behavior, parse_dflash, parse_kernel_toml, parse_model_types, parse_sampling_presets,
};

/// Latest mtime over all `.cuh` files (headers only) in a directory.
/// Used by the PTX-cache freshness check alongside `build_deps::include_closure_with_dirs`
/// as an additional conservative shared-header guard: any `.cuh` touch in a target
/// directory invalidates cached PTX in that target while single-`.cu` edits recompile
/// only the affected file and its transitive `#include` dependents.
fn latest_header_mtime(dir: &std::path::Path) -> Option<std::time::SystemTime> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut latest: Option<std::time::SystemTime> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cuh") {
            continue;
        }
        if let Ok(mt) = entry.metadata().and_then(|m| m.modified()) {
            latest = Some(latest.map_or(mt, |prev| prev.max(mt)));
        }
    }
    latest
}

/// Combined header-mtime for a target across every directory in `hw_dirs`
/// (model/quant, quant, and shared common) plus the target-identity signature
/// file. Per-`.cu` cache decisions then take
/// `max(this_cu_mtime, this_target_header_floor)`.
///
/// The signature file lets us invalidate cached PTX when the
/// `(hw, model, quant)` tuple at this `idx` slot changes from a prior
/// build (e.g. user toggled `ATLAS_TARGET_MODEL` between a wildcard
/// and a single model — same `t{idx}__` prefix would otherwise
/// collide with stale PTX).
fn target_header_floor(
    hw_dirs: &[PathBuf],
    model: &str,
    quant: &str,
    signature_file: &std::path::Path,
) -> Option<std::time::SystemTime> {
    let mut latest: Option<std::time::SystemTime> = None;
    for hw_dir in hw_dirs {
        for dir in [
            hw_dir.join(model).join(quant),
            hw_dir.join(quant),
            hw_dir.join("common"),
        ] {
            if let Some(t) = latest_header_mtime(&dir) {
                latest = Some(latest.map_or(t, |p| p.max(t)));
            }
        }
    }
    if let Ok(mt) = std::fs::metadata(signature_file).and_then(|m| m.modified()) {
        latest = Some(latest.map_or(mt, |p| p.max(mt)));
    }
    latest
}

/// Locate `ptxas` alongside `nvcc` or on `PATH`. Returns `None` if unavailable
/// so builds gracefully skip AOT CUBIN embedding instead of failing.
fn find_ptxas() -> Option<PathBuf> {
    if let Some(nvcc) = env::var_os("ATLAS_NVCC") {
        let candidate = PathBuf::from(nvcc).with_file_name("ptxas");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let candidate = PathBuf::from(cuda_home).join("bin/ptxas");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    for path in &[
        "/usr/local/cuda",
        "/usr/local/cuda-13.0",
        "/usr/local/cuda-13",
    ] {
        let candidate = PathBuf::from(path).join("bin/ptxas");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    if let Some(path_var) = env::var_os("PATH") {
        for dir in env::split_paths(&path_var) {
            let candidate = dir.join("ptxas");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

enum CubinOutcome {
    Cached(AotCubinArtifact),
    Compiled(AotCubinArtifact),
    Skipped(String),
}

struct CubinJob {
    mod_idx: usize,
    stem: String,
    ptx_file: PathBuf,
    cubin_file: PathBuf,
    ptx_size_hint: u64,
    is_cached: bool,
}

/// Assemble `.ptx` modules into `.cubin` ELF binaries in parallel using `ptxas`.
///
/// Honors incremental mtime caching (`cubin_mtime > ptx_mtime` when
/// `signature_matches && allow_cached`), matches each PTX file's `.target sm_*`
/// directive, forwards `-Xptxas` options, and obfuscates `.cubin.obf` when
/// `ATLAS_OBFUSCATE_PTX=1`. Any per-module `ptxas` failure emits a cargo
/// warning and omits only that module's CUBIN (automatic runtime PTX fallback).
fn compile_aot_cubins_parallel(
    ptxas: &std::path::Path,
    ptx_dir: &std::path::Path,
    target_idx: usize,
    output_ext: &str,
    target: &Target,
    signature_matches: bool,
    allow_cached: bool,
    obfuscate: bool,
    modules: &mut [ModuleArtifact],
) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    if modules.is_empty() {
        return;
    }

    let ptxas_extra_flags = build_codegen::extract_ptxas_flags(&target.extra_flags);

    let mut jobs: Vec<CubinJob> = modules
        .iter()
        .enumerate()
        .map(|(mod_idx, m)| {
            let ptx_file = ptx_dir.join(format!("t{target_idx}__{}.{output_ext}", m.stem));
            let cubin_file = ptx_dir.join(format!("t{target_idx}__{}.cubin", m.stem));
            let ptx_meta = std::fs::metadata(&ptx_file).ok();
            let ptx_size_hint = ptx_meta.as_ref().map_or(0, std::fs::Metadata::len);
            let is_cached = signature_matches
                && allow_cached
                && std::fs::metadata(&cubin_file)
                    .ok()
                    .filter(|meta| meta.len() > 0)
                    .and_then(|meta| meta.modified().ok())
                    .zip(ptx_meta.as_ref().and_then(|meta| meta.modified().ok()))
                    .is_some_and(|(cubin_mt, ptx_mt)| cubin_mt > ptx_mt);
            CubinJob {
                mod_idx,
                stem: m.stem.clone(),
                ptx_file,
                cubin_file,
                ptx_size_hint,
                is_cached,
            }
        })
        .collect();

    // Schedule uncached compilations before cached hashes, largest PTX first
    // (LPT scheduling) so multi-second modules begin at t=0.
    jobs.sort_by(|a, b| {
        a.is_cached
            .cmp(&b.is_cached)
            .then_with(|| b.ptx_size_hint.cmp(&a.ptx_size_hint))
            .then_with(|| a.mod_idx.cmp(&b.mod_idx))
    });

    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let worker_count = jobs.len().min(available.clamp(1, 16));
    let next_idx = AtomicUsize::new(0);
    let fallback_arch = target.arch.as_str();

    let mut outcomes: Vec<(usize, CubinOutcome)> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let jobs_ref = &jobs;
            let next_ref = &next_idx;
            let flags_ref = &ptxas_extra_flags;
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                loop {
                    let i = next_ref.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs_ref.get(i) else {
                        break;
                    };
                    let outcome = run_single_cubin_job(
                        ptxas,
                        ptx_dir,
                        target_idx,
                        fallback_arch,
                        flags_ref,
                        obfuscate,
                        job,
                    );
                    local.push((job.mod_idx, outcome));
                }
                local
            }));
        }
        let mut all = Vec::with_capacity(jobs.len());
        for handle in handles {
            all.extend(handle.join().expect("ptxas worker thread panicked"));
        }
        all
    });

    outcomes.sort_by_key(|(mod_idx, _)| *mod_idx);

    let mut cubin_cached = 0usize;
    let mut cubin_compiled = 0usize;
    let mut cubin_skipped = 0usize;
    for (mod_idx, outcome) in outcomes {
        match outcome {
            CubinOutcome::Cached(artifact) => {
                modules[mod_idx].aot_cubin = Some(artifact);
                cubin_cached += 1;
            }
            CubinOutcome::Compiled(artifact) => {
                modules[mod_idx].aot_cubin = Some(artifact);
                cubin_compiled += 1;
            }
            CubinOutcome::Skipped(reason) => {
                println!(
                    "cargo:warning=atlas-kernels: skipping AOT CUBIN for {} (PTX fallback): {reason}",
                    modules[mod_idx].module_name
                );
                cubin_skipped += 1;
            }
        }
    }

    println!(
        "cargo:warning=atlas-kernels: target {target_idx} ({}, {}, {}): AOT CUBIN {cubin_cached} cached, {cubin_compiled} compiled, {cubin_skipped} skipped",
        target.hw, target.model, target.quant
    );
}

fn run_single_cubin_job(
    ptxas: &std::path::Path,
    ptx_dir: &std::path::Path,
    target_idx: usize,
    fallback_arch: &str,
    ptxas_extra_flags: &[String],
    obfuscate: bool,
    job: &CubinJob,
) -> CubinOutcome {
    let raw_ptx = match std::fs::read(&job.ptx_file) {
        Ok(bytes) => bytes,
        Err(e) => return CubinOutcome::Skipped(format!("failed to read PTX: {e}")),
    };
    let ptx_len = raw_ptx.len();
    let digest = ring::digest::digest(&ring::digest::SHA256, &raw_ptx);
    let mut ptx_sha256 = [0u8; 32];
    ptx_sha256.copy_from_slice(digest.as_ref());

    if !job.is_cached {
        let effective_arch = build_codegen::parse_ptx_target_arch(&raw_ptx)
            .unwrap_or_else(|| fallback_arch.to_owned());
        let tmp_cubin = ptx_dir.join(format!("t{target_idx}__{}.cubin.tmp", job.stem));
        let mut cmd = std::process::Command::new(ptxas);
        cmd.arg(format!("-arch={effective_arch}")).arg("-O3");
        cmd.args(ptxas_extra_flags);
        cmd.arg(&job.ptx_file).arg("-o").arg(&tmp_cubin);

        match cmd.output() {
            Ok(output) if output.status.success() => {
                if let Err(e) = std::fs::rename(&tmp_cubin, &job.cubin_file) {
                    let _ = std::fs::remove_file(&tmp_cubin);
                    let _ = std::fs::remove_file(&job.cubin_file);
                    return CubinOutcome::Skipped(format!("failed to finalize cubin: {e}"));
                }
            }
            Ok(output) => {
                let _ = std::fs::remove_file(&tmp_cubin);
                let _ = std::fs::remove_file(&job.cubin_file);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let first_line = stderr
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("(no stderr)")
                    .trim();
                return CubinOutcome::Skipped(format!(
                    "ptxas -arch={effective_arch} exited with {}: {first_line}",
                    output.status
                ));
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_cubin);
                let _ = std::fs::remove_file(&job.cubin_file);
                return CubinOutcome::Skipped(format!("failed to invoke ptxas: {e}"));
            }
        }
    }

    if obfuscate {
        let raw_cubin = match std::fs::read(&job.cubin_file) {
            Ok(bytes) => bytes,
            Err(e) => return CubinOutcome::Skipped(format!("failed to read cubin for obf: {e}")),
        };
        let obf = build_codegen::obfuscate_ptx_bytes(&raw_cubin);
        let tmp_obf = ptx_dir.join(format!("t{target_idx}__{}.cubin.obf.tmp", job.stem));
        let obf_file = ptx_dir.join(format!("t{target_idx}__{}.cubin.obf", job.stem));
        if let Err(e) = std::fs::write(&tmp_obf, &obf) {
            let _ = std::fs::remove_file(&tmp_obf);
            return CubinOutcome::Skipped(format!("failed to write cubin.obf: {e}"));
        }
        if let Err(e) = std::fs::rename(&tmp_obf, &obf_file) {
            let _ = std::fs::remove_file(&tmp_obf);
            return CubinOutcome::Skipped(format!("failed to finalize cubin.obf: {e}"));
        }
    }

    let artifact = AotCubinArtifact {
        ptx_len,
        ptx_sha256,
    };
    if job.is_cached {
        CubinOutcome::Cached(artifact)
    } else {
        CubinOutcome::Compiled(artifact)
    }
}

#[path = "build_profile.rs"]
mod build_profile;

#[path = "build_codegen.rs"]
mod build_codegen;
use build_codegen::{AotCubinArtifact, ModuleArtifact, generate_target_ptx_rs};

#[path = "build_target.rs"]
mod build_target;
use build_target::resolve_compute_target;
