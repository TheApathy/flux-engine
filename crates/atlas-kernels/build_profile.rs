// SPDX-License-Identifier: AGPL-3.0-only
//
// `[[dflash.profile]]` rows from every MODEL.toml under `kernels/$hw/`, emitted
// as the static `DRAFTER_PROFILES` table (see `src/drafter_profile.rs`).
//
// Every model directory is read, not only the compiled targets: drafter
// checkpoint admission is target-agnostic (it runs before the target is
// paired), so a single-target binary must admit and refuse exactly what a
// universal one does. A malformed row fails the build; an unknown key is an
// error rather than a silently unchecked field.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const KINDS: &[(&str, &str)] = &[
    ("dflash", "Dflash"),
    ("dflash2", "Dflash2"),
    ("dspark", "Dspark"),
    ("mtp", "Mtp"),
];
const ADMISSIONS: &[(&str, &str)] = &[
    ("exact", "Exact"),
    ("quorum", "Quorum"),
    ("generic", "Generic"),
    ("refused", "Refused"),
];
const CAPTURE_MODES: &[(&str, &str)] = &[
    ("residual_slice", "ResidualSlice"),
    ("residual_slice_f32_downcast", "ResidualSliceF32Downcast"),
    ("qwen4_hyper_projected", "Qwen4HyperProjected"),
    ("qwen4_layer_attn_hc", "Qwen4LayerAttnHc"),
    ("qwen4_hc_mean", "Qwen4HcMean"),
];

/// Keys a row may carry beyond the drafter identity.
const POLICY_KEYS: &[&str] = &[
    "name",
    "kind",
    "admission",
    "qualified",
    "launcher_default",
    "capture_mode",
    "verify_k",
    "wide_verify_k",
    "proposal_vocab",
    "allow_modelopt_nvfp4",
    "allow_rtncal_w4a4",
    "allow_markov",
    "target_hidden_size",
    "target_residual_width",
    "target_num_layers",
    "target_vocab_sizes",
];

/// Drafter checkpoint identity keys. Only `exact` and `quorum` rows may set
/// them: a `generic` or `refused` row that pinned geometry would read as
/// enforced while nothing checks it.
const IDENTITY_KEYS: &[&str] = &[
    "architecture",
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_target_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "draft_vocab_size",
    "block_size",
    "root_block_size_explicit",
    "mask_token_id",
    "target_layer_ids",
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
    "selector_vocab_size",
    "forbid_unknown_fields",
    "layer_types",
    "sliding_window",
    "is_causal",
    "tie_word_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "hidden_act",
    "tensor_count",
    "parameter_count",
    "bare_tensor_names",
];

/// Parse every `[[dflash.profile]]` row under `hw_dir` and render the
/// generated Rust table. Panics on any malformed row.
pub(super) fn generate_drafter_profiles_rs(hw_dir: &Path) -> String {
    generate_drafter_profiles_rs_layered(&[hw_dir.to_path_buf()])
}

/// Parse every `[[dflash.profile]]` row across an ordered hardware inheritance
/// chain (`hw_dirs` ordered child-first: `[child, base, ...]`).
///
/// When both a child and a base directory define `<model>/MODEL.toml`, the
/// child file overrides the base file for that model. Panics on any malformed
/// row.
pub(super) fn generate_drafter_profiles_rs_layered(hw_dirs: &[PathBuf]) -> String {
    let mut model_tomls: BTreeMap<String, PathBuf> = BTreeMap::new();
    // Iterate base -> child so child MODEL.toml overrides base MODEL.toml.
    for hw_dir in hw_dirs.iter().rev() {
        println!("cargo:rerun-if-changed={}", hw_dir.display());
        for entry in std::fs::read_dir(hw_dir)
            .unwrap_or_else(|e| panic!("{}: {e}", hw_dir.display()))
            .flatten()
        {
            let path = entry.path();
            let toml_path = path.join("MODEL.toml");
            if toml_path.is_file() {
                let model = entry.file_name().to_string_lossy().into_owned();
                model_tomls.insert(model, toml_path);
            }
        }
    }

    let mut rows = Vec::new();
    for (model, path) in &model_tomls {
        println!("cargo:rerun-if-changed={}", path.display());
        // If inherited from a base dir, watch any existing child model dir so
        // adding a child MODEL.toml override invalidates the build.
        if let Some(child_hw) = hw_dirs.first() {
            let child_model_dir = child_hw.join(model);
            if child_model_dir.is_dir() && child_model_dir.join("MODEL.toml") != *path {
                println!("cargo:rerun-if-changed={}", child_model_dir.display());
            }
        }
        let content =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let toml: toml::Value = toml::from_str(&content)
            .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
        let Some(profiles) = toml.get("dflash").and_then(|d| d.get("profile")) else {
            continue;
        };
        let profiles = profiles.as_array().unwrap_or_else(|| {
            panic!(
                "{}: dflash.profile must be [[dflash.profile]]",
                path.display()
            )
        });
        let mut names = BTreeSet::new();
        let mut defaults = 0usize;
        for row in profiles {
            let table = row.as_table().unwrap_or_else(|| {
                panic!("{}: dflash.profile rows must be tables", path.display())
            });
            let rendered = render_row(model, table)
                .unwrap_or_else(|e| panic!("{} [[dflash.profile]]: {e}", path.display()));
            let name = table["name"].as_str().expect("checked by render_row");
            assert!(
                names.insert(name.to_owned()),
                "{}: duplicate [[dflash.profile]] name {name:?}",
                path.display()
            );
            if bool_key(table, "launcher_default").expect("checked") == Some(true) {
                defaults += 1;
            }
            rows.push(rendered);
        }
        assert!(
            defaults <= 1,
            "{}: at most one [[dflash.profile]] row may set launcher_default",
            path.display()
        );
    }

    let mut g = String::new();
    g.push_str("// Auto-generated by build.rs from kernels/$hw/*/MODEL.toml [[dflash.profile]] — do not edit.\n");
    g.push_str("static DRAFTER_PROFILES: &[DrafterProfile] = &[\n");
    for row in rows {
        g.push_str(&row);
    }
    g.push_str("];\n");
    g
}

type Table = toml::map::Map<String, toml::Value>;

fn render_row(model: &str, t: &Table) -> Result<String, String> {
    for key in t.keys() {
        if !POLICY_KEYS.contains(&key.as_str()) && !IDENTITY_KEYS.contains(&key.as_str()) {
            return Err(format!("unknown key {key:?}"));
        }
    }
    let name = str_key(t, "name")?.ok_or("missing `name`")?;
    let at = |e: String| format!("{model}/{name}: {e}");
    let kind = enum_key(t, "kind", KINDS)?.ok_or_else(|| at("missing `kind`".into()))?;
    let admission =
        enum_key(t, "admission", ADMISSIONS)?.ok_or_else(|| at("missing `admission`".into()))?;
    let capture_mode = enum_key(t, "capture_mode", CAPTURE_MODES)?.unwrap_or("ResidualSlice");
    let qualified = bool_key(t, "qualified")?.unwrap_or(false);
    let launcher_default = bool_key(t, "launcher_default")?.unwrap_or(false);
    let verify_k = usize_list(t, "verify_k")?;
    let wide_verify_k = usize_key(t, "wide_verify_k")?;
    let allow_modelopt = bool_key(t, "allow_modelopt_nvfp4")?.unwrap_or(false);
    let allow_rtncal = bool_key(t, "allow_rtncal_w4a4")?.unwrap_or(false);
    let allow_markov = bool_key(t, "allow_markov")?.unwrap_or(false);

    let pinned = matches!(admission, "Exact" | "Quorum");
    let identity_keys: Vec<&str> = t
        .keys()
        .map(String::as_str)
        .filter(|key| IDENTITY_KEYS.contains(key))
        .collect();
    if !pinned && !identity_keys.is_empty() {
        return Err(at(format!(
            "{admission} rows admit through the generic validator and must not pin identity keys {identity_keys:?}"
        )));
    }
    if launcher_default && !qualified {
        return Err(at("launcher_default requires qualified = true".into()));
    }
    if pinned {
        for key in [
            "architecture",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "block_size",
            "mask_token_id",
            "target_layer_ids",
        ] {
            if !t.contains_key(key) {
                return Err(at(format!("{admission} rows must pin `{key}`")));
            }
        }
        if kind != "Dspark" && !t.contains_key("is_causal") {
            return Err(at(format!("{admission} rows must pin `is_causal`")));
        }
        if !matches!(kind, "Dflash" | "Dflash2" | "Dspark") {
            return Err(at(
                "only dflash/dflash2/dspark rows can be exact or quorum".into()
            ));
        }
        let block = usize_key(t, "block_size")?.expect("checked");
        let max_verify_k = match kind {
            "Dspark" | "Dflash2" => block + 1,
            _ => block,
        };
        match verify_k.as_deref() {
            Some([lo, hi])
                if 2 <= *lo
                    && lo <= hi
                    && ((*hi == block && kind != "Dspark") || *hi == max_verify_k) => {}
            _ => {
                return Err(at(format!(
                    "verify_k must be [min, max] with 2 <= min <= max and max matching block_size={block}"
                )));
            }
        }
        if wide_verify_k.is_some_and(|wide| wide <= block) {
            return Err(at("wide_verify_k must exceed block_size".into()));
        }
    } else if verify_k.is_some() || wide_verify_k.is_some() {
        return Err(at(
            "verify_k/wide_verify_k need an exact or quorum row".into()
        ));
    }
    if usize_key(t, "mask_token_id")?.is_some_and(|mask| u32::try_from(mask).is_err()) {
        return Err(at("mask_token_id must fit in u32".into()));
    }
    if admission == "Quorum" && !t.contains_key("num_target_layers") {
        return Err(at("quorum rows must pin `num_target_layers`".into()));
    }
    // DFlash/DFlash2 quorum rows require an explicit noncausal declaration;
    // all-full-attention DSpark checkpoints omit `is_causal`.
    if admission == "Quorum" && kind != "Dspark" && bool_key(t, "is_causal")? != Some(false) {
        return Err(at(
            "quorum rows are block-diffusion drafters: pin is_causal = false".into(),
        ));
    }
    if kind == "Dflash2" && pinned {
        for key in [
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ] {
            if !t.contains_key(key) {
                return Err(at(format!("dflash2 rows must pin `{key}`")));
            }
        }
    }
    if allow_modelopt
        && (admission != "Exact"
            || usize_key(t, "conv_group_size")? != Some(16)
            || usize_key(t, "selector_rank")? != Some(256)
            || usize_key(t, "num_hidden_layers")? != Some(5))
    {
        // `qwen38_dflash2_modelopt` encodes exactly this census.
        return Err(at(
            "allow_modelopt_nvfp4 needs an exact 5-layer row with conv_group_size=16, selector_rank=256"
                .into(),
        ));
    }
    if matches!(
        capture_mode,
        "Qwen4HyperProjected" | "Qwen4LayerAttnHc" | "Qwen4HcMean"
    ) && (usize_key(t, "target_hidden_size")?.is_none()
        || usize_key(t, "target_residual_width")?.is_none()
        || usize_key(t, "target_num_layers")?.is_none()
        || usize_list(t, "target_vocab_sizes")?.is_none())
    {
        return Err(at(format!(
            "{capture_mode} rows must pin the target identity"
        )));
    }

    let mut r = String::new();
    let opt = |v: Option<usize>| v.map_or("None".into(), |v| format!("Some({v})"));
    let opt_str = |v: Option<&str>| v.map_or("None".into(), |v| format!("Some({v:?})"));
    let opt_bool = |v: Option<bool>| v.map_or("None".into(), |v| format!("Some({v})"));
    let opt_f32 = |v: Option<f32>| v.map_or("None".into(), |v| format!("Some({v:?}_f32)"));
    let opt_list = |v: Option<Vec<usize>>| {
        v.map_or("None".into(), |v| {
            format!(
                "Some(&[{}])",
                v.iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    };
    writeln!(r, "    DrafterProfile {{").unwrap();
    writeln!(r, "        target_model: {model:?},").unwrap();
    writeln!(r, "        name: {name:?},").unwrap();
    writeln!(r, "        kind: DrafterKind::{kind},").unwrap();
    writeln!(r, "        admission: DrafterAdmission::{admission},").unwrap();
    writeln!(r, "        qualified: {qualified},").unwrap();
    writeln!(r, "        launcher_default: {launcher_default},").unwrap();
    writeln!(
        r,
        "        capture_mode: DrafterCaptureMode::{capture_mode},"
    )
    .unwrap();
    writeln!(
        r,
        "        verify_k: {},",
        verify_k.map_or("None".into(), |v| format!("Some(({}, {}))", v[0], v[1]))
    )
    .unwrap();
    writeln!(r, "        wide_verify_k: {},", opt(wide_verify_k)).unwrap();
    writeln!(
        r,
        "        proposal_vocab: {},",
        opt(usize_key(t, "proposal_vocab")?)
    )
    .unwrap();
    writeln!(r, "        allow_modelopt_nvfp4: {allow_modelopt},").unwrap();
    writeln!(r, "        allow_rtncal_w4a4: {allow_rtncal},").unwrap();
    let _ = writeln!(r, "        allow_markov: {allow_markov},");
    writeln!(r, "        target: TargetIdentity {{").unwrap();
    writeln!(
        r,
        "            hidden_size: {},",
        opt(usize_key(t, "target_hidden_size")?)
    )
    .unwrap();
    writeln!(
        r,
        "            residual_width: {},",
        opt(usize_key(t, "target_residual_width")?)
    )
    .unwrap();
    writeln!(
        r,
        "            num_layers: {},",
        opt(usize_key(t, "target_num_layers")?)
    )
    .unwrap();
    writeln!(
        r,
        "            vocab_sizes: {},",
        opt_list(usize_list(t, "target_vocab_sizes")?)
    )
    .unwrap();
    writeln!(r, "        }},").unwrap();
    writeln!(r, "        drafter: DrafterIdentity {{").unwrap();
    writeln!(
        r,
        "            architecture: {},",
        opt_str(str_key(t, "architecture")?)
    )
    .unwrap();
    writeln!(
        r,
        "            model_type: {},",
        opt_str(str_key(t, "model_type")?)
    )
    .unwrap();
    for key in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_target_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    ] {
        writeln!(r, "            {key}: {},", opt(usize_key(t, key)?)).unwrap();
    }
    writeln!(
        r,
        "            draft_vocab_size: {},",
        vocab_pin(usize_key(t, "draft_vocab_size")?)
    )
    .unwrap();
    writeln!(
        r,
        "            block_size: {},",
        opt(usize_key(t, "block_size")?)
    )
    .unwrap();
    writeln!(
        r,
        "            root_block_size_explicit: {},",
        bool_key(t, "root_block_size_explicit")?.unwrap_or(false)
    )
    .unwrap();
    writeln!(
        r,
        "            mask_token_id: {},",
        usize_key(t, "mask_token_id")?.map_or("None".into(), |v| format!("Some({v})"))
    )
    .unwrap();
    writeln!(
        r,
        "            target_layer_ids: {},",
        opt_list(usize_list(t, "target_layer_ids")?)
    )
    .unwrap();
    for key in [
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ] {
        writeln!(r, "            {key}: {},", opt(usize_key(t, key)?)).unwrap();
    }
    writeln!(
        r,
        "            selector_vocab_size: {},",
        vocab_pin(usize_key(t, "selector_vocab_size")?)
    )
    .unwrap();
    writeln!(
        r,
        "            forbid_unknown_fields: {},",
        bool_key(t, "forbid_unknown_fields")?.unwrap_or(false)
    )
    .unwrap();
    writeln!(
        r,
        "            layer_types: {},",
        str_list(t, "layer_types")?.map_or("None".into(), |v| format!(
            "Some(&[{}])",
            v.iter()
                .map(|s| format!("{s:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    )
    .unwrap();
    writeln!(
        r,
        "            sliding_window: {},",
        opt(usize_key(t, "sliding_window")?)
    )
    .unwrap();
    writeln!(
        r,
        "            is_causal: {},",
        opt_bool(bool_key(t, "is_causal")?)
    )
    .unwrap();
    writeln!(
        r,
        "            tie_word_embeddings: {},",
        opt_bool(bool_key(t, "tie_word_embeddings")?)
    )
    .unwrap();
    writeln!(
        r,
        "            rope_theta: {},",
        opt_f32(f32_key(t, "rope_theta")?)
    )
    .unwrap();
    writeln!(
        r,
        "            rms_norm_eps: {},",
        opt_f32(f32_key(t, "rms_norm_eps")?)
    )
    .unwrap();
    writeln!(
        r,
        "            hidden_act: {},",
        opt_str(str_key(t, "hidden_act")?)
    )
    .unwrap();
    writeln!(
        r,
        "            tensor_count: {},",
        opt(usize_key(t, "tensor_count")?)
    )
    .unwrap();
    writeln!(
        r,
        "            parameter_count: {},",
        opt(usize_key(t, "parameter_count")?)
    )
    .unwrap();
    writeln!(
        r,
        "            bare_tensor_names: {},",
        bool_key(t, "bare_tensor_names")?.unwrap_or(false)
    )
    .unwrap();
    writeln!(r, "        }},").unwrap();
    writeln!(r, "    }},").unwrap();
    Ok(r)
}

/// `0` pins "the checkpoint must not declare this field"; any other value
/// pins it to that number.
fn vocab_pin(value: Option<usize>) -> String {
    match value {
        None => "VocabPin::Unchecked".into(),
        Some(0) => "VocabPin::Undeclared".into(),
        Some(v) => format!("VocabPin::Equals({v})"),
    }
}

fn str_key<'a>(t: &'a Table, key: &str) -> Result<Option<&'a str>, String> {
    t.get(key)
        .map(|v| v.as_str().ok_or(format!("`{key}` must be a string")))
        .transpose()
}

fn bool_key(t: &Table, key: &str) -> Result<Option<bool>, String> {
    t.get(key)
        .map(|v| v.as_bool().ok_or(format!("`{key}` must be a boolean")))
        .transpose()
}

fn usize_key(t: &Table, key: &str) -> Result<Option<usize>, String> {
    t.get(key)
        .map(|v| {
            v.as_integer()
                .and_then(|i| usize::try_from(i).ok())
                .ok_or(format!("`{key}` must be a non-negative integer"))
        })
        .transpose()
}

fn f32_key(t: &Table, key: &str) -> Result<Option<f32>, String> {
    t.get(key)
        .map(|v| {
            v.as_float()
                .map(|f| f as f32)
                .ok_or(format!("`{key}` must be a float"))
        })
        .transpose()
}

fn usize_list(t: &Table, key: &str) -> Result<Option<Vec<usize>>, String> {
    t.get(key)
        .map(|v| {
            v.as_array()
                .ok_or(format!("`{key}` must be an array"))?
                .iter()
                .map(|x| {
                    x.as_integer()
                        .and_then(|i| usize::try_from(i).ok())
                        .ok_or(format!("`{key}` entries must be non-negative integers"))
                })
                .collect()
        })
        .transpose()
}

fn str_list(t: &Table, key: &str) -> Result<Option<Vec<String>>, String> {
    t.get(key)
        .map(|v| {
            v.as_array()
                .ok_or(format!("`{key}` must be an array"))?
                .iter()
                .map(|x| {
                    x.as_str()
                        .map(str::to_owned)
                        .ok_or(format!("`{key}` entries must be strings"))
                })
                .collect()
        })
        .transpose()
}

fn enum_key(
    t: &Table,
    key: &str,
    variants: &[(&str, &'static str)],
) -> Result<Option<&'static str>, String> {
    str_key(t, key)?
        .map(|value| {
            variants
                .iter()
                .find(|(spelling, _)| *spelling == value)
                .map(|(_, variant)| *variant)
                .ok_or(format!(
                    "`{key}` = {value:?}; expected one of {:?}",
                    variants.iter().map(|(s, _)| *s).collect::<Vec<_>>()
                ))
        })
        .transpose()
}
