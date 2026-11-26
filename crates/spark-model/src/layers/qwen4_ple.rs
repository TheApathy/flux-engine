// SPDX-License-Identifier: AGPL-3.0-only

//! Exact Qwen4-Exp PLE n-gram row selection.

use anyhow::{Context, Result, ensure};

pub const QWEN4_PLE_HEADS: usize = 16;

const PLE_CONV_HISTORY: usize = 9;
const PLE_MAX_SPEC_ROWS: usize = 32;

/// Resolve the PLE offload manifest path, honoring an optional
/// `ATLAS_PLE_OFFLOAD_DIR` override (e.g., `/dev/shm/flux-ple-offload/<hash>`
/// for RAM-staged PLE on read-only model mounts).
///
/// When `override_dir` is set and non-empty, the resolved manifest file must
/// exist; missing overrides fail fast instead of silently falling back to disk.
pub fn resolve_ple_offload_manifest_path_from(
    default_manifest: &str,
    override_dir: Option<&str>,
) -> Result<std::path::PathBuf> {
    let Some(trimmed) = override_dir.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(std::path::PathBuf::from(default_manifest));
    };
    let candidate = std::path::Path::new(trimmed);
    let resolved = if candidate.extension().and_then(|ext| ext.to_str()) == Some("json") {
        candidate.to_path_buf()
    } else {
        candidate.join("manifest.json")
    };
    ensure!(
        resolved.is_file(),
        "ATLAS_PLE_OFFLOAD_DIR={trimmed:?} does not contain a readable PLE manifest at {}",
        resolved.display()
    );
    Ok(resolved)
}

/// Persistent, request-scaled PLE history allocations retained by the layer.
///
/// The speculative intermediate width is intentionally fixed at
/// [`PLE_MAX_SPEC_ROWS`]. Reducing it based on the selected speculation mode
/// would change the existing allocation contract and needs separate runtime
/// qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4PleAllocationBytes {
    pub state: usize,
    pub checkpoint: usize,
    pub intermediate: usize,
    pub total: usize,
}

/// Exact byte stride of one sequence slot's live PLE convolution history
/// (`residual_width * PLE_CONV_HISTORY * sizeof(bf16)` = `180 KiB` per PLE layer
/// at `residual_width = 10,240`, or `8.64 MiB` across 48 PLE layers).
pub fn checked_ple_conv_slot_bytes(residual_width: usize) -> Result<usize> {
    residual_width
        .checked_mul(PLE_CONV_HISTORY)
        .and_then(|items| items.checked_mul(std::mem::size_of::<u16>()))
        .context("Qwen4 PLE history state stride overflow")
}

/// Calculate the exact request-scaled PLE history bytes allocated by the
/// runtime without permitting any intermediate arithmetic to wrap. Each slot
/// retains the live state, one checkpoint, and 32 speculative intermediate
/// rows.
pub fn checked_ple_allocation_bytes(
    max_batch_size: usize,
    residual_width: usize,
) -> Result<Qwen4PleAllocationBytes> {
    let state_stride = checked_ple_conv_slot_bytes(residual_width)?;
    let state = max_batch_size
        .checked_mul(state_stride)
        .context("Qwen4 PLE live state allocation overflow")?;
    let checkpoint = state;
    let intermediate_rows = max_batch_size
        .checked_mul(PLE_MAX_SPEC_ROWS)
        .context("Qwen4 PLE intermediate row count overflow")?;
    let intermediate = intermediate_rows
        .checked_mul(state_stride)
        .context("Qwen4 PLE intermediate state allocation overflow")?;
    let total = state
        .checked_add(checkpoint)
        .and_then(|bytes| bytes.checked_add(intermediate))
        .context("Qwen4 PLE total history allocation overflow")?;
    Ok(Qwen4PleAllocationBytes {
        state,
        checkpoint,
        intermediate,
        total,
    })
}

/// Returns `true` when `name` is one of the offloaded PLE n-gram embedding table
/// tensors (`{table_prefix}.{weight,weight_scale,weight_scale_2,input_scale}`)
/// that are read directly from the binary PLE offload shards at runtime and must
/// not be uploaded into unified memory during safetensors loading.
pub fn skip_offloaded_ple_tensor(table_prefix: &str, name: &str) -> bool {
    let rest = name.strip_prefix(table_prefix).or_else(|| {
        let rel_prefix = table_prefix.strip_prefix("model.")?;
        if !rel_prefix.starts_with("layers.") {
            return None;
        }
        ["model.language_model.", "language_model.model."]
            .into_iter()
            .find_map(|p| name.strip_prefix(p)?.strip_prefix(rel_prefix))
    });
    let Some(rest) = rest else {
        return false;
    };
    let rest = if let Some(after_shard) = rest.strip_prefix(".shard_") {
        let digits_len = after_shard
            .bytes()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if digits_len > 0 {
            &after_shard[digits_len..]
        } else {
            return false;
        }
    } else {
        rest
    };
    matches!(
        rest,
        ".weight" | ".weight_scale" | ".weight_scale_2" | ".input_scale"
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleRowSelection {
    pub shard: usize,
    pub row: usize,
}

/// Exact identity of one cross-chunk PLE read-ahead operation.
///
/// Keeping the token bytes in the key is intentional: a recycled sequence
/// slot must never consume a detached read from an earlier request, even when
/// its chunk geometry happens to match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlePrefetchKey {
    chunk_start: usize,
    prior_tokens: Vec<u32>,
    current_tokens: Vec<u32>,
}

impl PlePrefetchKey {
    pub fn new(chunk_start: usize, current_tokens: &[u32], prior_tokens: &[u32]) -> Self {
        Self {
            chunk_start,
            prior_tokens: prior_tokens.to_vec(),
            current_tokens: current_tokens.to_vec(),
        }
    }

    pub fn matches(
        &self,
        chunk_start: usize,
        current_tokens: &[u32],
        prior_tokens: &[u32],
    ) -> bool {
        self.chunk_start == chunk_start
            && self.current_tokens == current_tokens
            && self.prior_tokens == prior_tokens
    }
}

/// CPU row planner. It runs as soon as the current token is known, allowing
/// the 16 sparse reads to overlap embedding lookup and layer 0.
pub struct Qwen4PleHasher {
    multipliers: [i64; 3],
    vocab_sizes: [i64; QWEN4_PLE_HEADS],
    offsets: [i64; QWEN4_PLE_HEADS],
    eos_token_id: u32,
    shard_rows: usize,
}

impl Qwen4PleHasher {
    pub fn new(
        multipliers: [i64; 3],
        vocab_sizes: [i64; QWEN4_PLE_HEADS],
        offsets: [i64; QWEN4_PLE_HEADS],
        eos_token_id: u32,
        shard_rows: usize,
    ) -> Result<Self> {
        ensure!(shard_rows > 0, "PLE shard row count must be positive");
        ensure!(
            multipliers.iter().all(|value| value % 2 != 0),
            "PLE hash multipliers must be odd"
        );
        ensure!(
            vocab_sizes.iter().all(|value| *value > 0),
            "PLE head vocabulary sizes must be positive"
        );
        ensure!(offsets[0] == 0, "PLE first head offset must be zero");
        for head in 1..QWEN4_PLE_HEADS {
            ensure!(
                offsets[head] == offsets[head - 1] + vocab_sizes[head - 1],
                "PLE head offsets are not contiguous at head {head}"
            );
        }
        Ok(Self {
            multipliers,
            vocab_sizes,
            offsets,
            eos_token_id,
            shard_rows,
        })
    }

    pub fn select_decode(
        &self,
        current_token: u32,
        prior_tokens: &[u32],
    ) -> [PleRowSelection; QWEN4_PLE_HEADS] {
        let mut segment = prior_tokens
            .iter()
            .rev()
            .take_while(|&&token| token != self.eos_token_id);
        let previous = segment.next().copied().unwrap_or(self.eos_token_id);
        let previous_2 = segment.next().copied().unwrap_or(self.eos_token_id);
        let bigram = (current_token as i64).wrapping_mul(self.multipliers[0])
            ^ (previous as i64).wrapping_mul(self.multipliers[1]);
        let trigram = bigram ^ (previous_2 as i64).wrapping_mul(self.multipliers[2]);
        std::array::from_fn(|head| {
            let mixed = if head < 8 { bigram } else { trigram };
            let global = mixed.rem_euclid(self.vocab_sizes[head]) + self.offsets[head];
            let global = global as usize;
            PleRowSelection {
                shard: global / self.shard_rows,
                row: global % self.shard_rows,
            }
        })
    }

    /// Plan every sparse row needed by a contiguous prefill chunk. The
    /// returned layout is token-major, then head-major, matching the batched
    /// dequantizer. This is deliberately the same rolling decode contract:
    /// token `t` can see the supplied prefix and chunk rows `[0, t)` only.
    pub fn select_prefill(
        &self,
        current_tokens: &[u32],
        prior_tokens: &[u32],
    ) -> Vec<PleRowSelection> {
        let prior_tail = &prior_tokens[prior_tokens.len().saturating_sub(2)..];
        let mut window = [self.eos_token_id; 2];
        for &token in prior_tail {
            window[0] = window[1];
            window[1] = token;
        }
        let mut selections = Vec::with_capacity(current_tokens.len() * QWEN4_PLE_HEADS);
        for &token in current_tokens {
            selections.extend_from_slice(&self.select_decode(token, &window));
            window[0] = window[1];
            window[1] = token;
        }
        selections
    }
}

#[cfg(test)]
mod planner_tests {
    use super::*;

    fn hasher() -> Qwen4PleHasher {
        Qwen4PleHasher::new(
            [10007, 10009, 10037],
            std::array::from_fn(|head| 997 + head as i64),
            std::array::from_fn(|head| (0..head).map(|prior| 997 + prior as i64).sum()),
            99,
            251,
        )
        .unwrap()
    }

    #[test]
    fn allocation_bytes_cover_c1_and_c128_history_geometry() {
        let c1 = checked_ple_allocation_bytes(1, 10_240).unwrap();
        assert_eq!(c1.state, 184_320);
        assert_eq!(c1.checkpoint, 184_320);
        assert_eq!(c1.intermediate, 5_898_240);
        assert_eq!(c1.total, 6_266_880);

        let c128 = checked_ple_allocation_bytes(128, 10_240).unwrap();
        assert_eq!(c128.state, 23_592_960);
        assert_eq!(c128.checkpoint, 23_592_960);
        assert_eq!(c128.intermediate, 754_974_720);
        assert_eq!(c128.total, 802_160_640);
        assert_eq!(c128.total, c1.total * 128);
    }

    #[test]
    fn allocation_bytes_fail_closed_on_stride_and_batch_overflow() {
        let stride = checked_ple_allocation_bytes(1, usize::MAX).unwrap_err();
        assert!(
            stride.to_string().contains("history state stride overflow"),
            "wrong error: {stride:#}"
        );

        let batch = checked_ple_allocation_bytes(usize::MAX, 10_240).unwrap_err();
        assert!(
            batch.to_string().contains("live state allocation overflow"),
            "wrong error: {batch:#}"
        );
    }

    #[test]
    fn prefill_plan_is_exact_rolling_decode() {
        let hasher = hasher();
        let prefix = [8, 9, 99, 11, 12];
        let tokens = [13, 14, 99, 15, 16, 17];
        let batched = hasher.select_prefill(&tokens, &prefix);
        let mut history = prefix.to_vec();
        let mut serial = Vec::new();
        for token in tokens {
            serial.extend_from_slice(&hasher.select_decode(token, &history));
            history.push(token);
        }
        assert_eq!(batched, serial);
        assert_eq!(batched.len(), tokens.len() * QWEN4_PLE_HEADS);
    }

    #[test]
    fn cross_chunk_prefetch_key_requires_exact_prompt_identity() {
        let key = PlePrefetchKey::new(4, &[5, 6, 7, 8], &[1, 2, 3, 4]);
        assert!(key.matches(4, &[5, 6, 7, 8], &[1, 2, 3, 4]));
        assert!(!key.matches(0, &[5, 6, 7, 8], &[1, 2, 3, 4]));
        assert!(!key.matches(4, &[5, 6, 7, 9], &[1, 2, 3, 4]));
        assert!(!key.matches(4, &[5, 6, 7, 8], &[1, 2, 3, 9]));
        assert!(!key.matches(4, &[5, 6, 7], &[1, 2, 3, 4]));
    }

    #[test]
    fn offload_skip_is_exact_and_hostile_names_stay_resident() {
        let prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
        for suffix in ["weight", "weight_scale", "weight_scale_2", "input_scale"] {
            assert!(super::skip_offloaded_ple_tensor(
                prefix,
                &format!("{prefix}.{suffix}")
            ));
        }
        for resident in [
            "model.language_model.layers.1.ple.key_proj.weight",
            "model.language_model.layers.1.ple.value_proj.weight",
            "model.language_model.layers.1.ple.norm_key.weight",
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight.extra",
            "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
        ] {
            assert!(
                !super::skip_offloaded_ple_tensor(prefix, resident),
                "unexpectedly skipped {resident}"
            );
        }
    }

    #[test]
    fn pre_autodetect_prefix_and_shards_are_skipped() {
        let pre_detect_prefix = "model.layers.1.ple.ple_embedding.ngram_embedding";
        for root in ["model.language_model.", "language_model.model.", "model."] {
            for suffix in [
                ".weight",
                ".weight_scale",
                ".weight_scale_2",
                ".input_scale",
            ] {
                let unsharded = format!("{root}layers.1.ple.ple_embedding.ngram_embedding{suffix}");
                assert!(
                    super::skip_offloaded_ple_tensor(pre_detect_prefix, &unsharded),
                    "{unsharded}"
                );
                for shard in [0, 1, 15, 127] {
                    let sharded = format!(
                        "{root}layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}{suffix}"
                    );
                    assert!(
                        super::skip_offloaded_ple_tensor(pre_detect_prefix, &sharded),
                        "{sharded}"
                    );
                }
            }
            for keep in [
                format!("{root}layers.1.ple.ple_gate.weight"),
                format!("{root}layers.1.ple.ple_proj.weight"),
                format!("{root}layers.1.ple.ple_embedding.ngram_embedding.shard_.weight"),
                format!("{root}layers.1.ple.ple_embedding.ngram_embedding.shard_0.other"),
                format!("{root}layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight"),
            ] {
                assert!(
                    !super::skip_offloaded_ple_tensor(pre_detect_prefix, &keep),
                    "{keep}"
                );
            }
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod runtime {
    use super::*;
    use atlas_core::config::ModelConfig;
    use parking_lot::Mutex;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle};
    use spark_runtime::kernel_args::KernelLaunch;
    use spark_runtime::weights::{WeightDtype, WeightStore};
    use spark_storage::ple_offload::{ParkedPleCache, PleIoMode, PleNvfp4Row, PleOffloadReader};
    use std::ffi::CString;
    use std::fs::{self, DirBuilder, File, OpenOptions};
    use std::io::{Read, Write};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
    use std::os::unix::io::AsRawFd;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::layers::ops;
    use crate::weight_map::{DenseWeight, dense, dense_f32_safe};

    mod read_affinity;
    mod verify_batch;
    pub use verify_batch::{
        PleActiveDecodeRow, PleWarmSeq, PleWarmTicket, ple_verify_batch_selected,
    };

    const PLE_EMBED: usize = 2560;
    const PLE_RECORD_BYTES: usize = 90;
    const PLE_IO_QUEUE_DEPTH: usize = 256;
    const PLE_IO_QUEUE_DEPTH_ENV: &str = "ATLAS_QWEN4_PLE_IO_QUEUE_DEPTH";
    const PLE_IO_TIMEOUT_MS_ENV: &str = "ATLAS_QWEN4_PLE_IO_TIMEOUT_MS";
    const PLE_PARITY_WIDTH: usize = 10_240;
    const PLE_PARITY_ENABLE: &str = "ATLAS_QWEN4_PLE_PARITY_CAPTURE";
    const PLE_PARITY_ROOT: &str = "ATLAS_QWEN4_PLE_PARITY_ROOT";
    const PLE_PARITY_NONCE: &str = "ATLAS_QWEN4_PLE_PARITY_NONCE";

    fn parse_ple_io_queue_depth(value: Option<&str>) -> Result<usize> {
        match value {
            None | Some("256") => Ok(PLE_IO_QUEUE_DEPTH),
            Some("32") => Ok(32),
            Some(_) => anyhow::bail!("{PLE_IO_QUEUE_DEPTH_ENV} must be exact value 32 or 256"),
        }
    }

    fn configured_ple_io_queue_depth() -> Result<usize> {
        let value = std::env::var_os(PLE_IO_QUEUE_DEPTH_ENV);
        let value = value
            .as_deref()
            .map(|value| {
                value
                    .to_str()
                    .with_context(|| format!("{PLE_IO_QUEUE_DEPTH_ENV} is not UTF-8"))
            })
            .transpose()?;
        parse_ple_io_queue_depth(value)
    }

    fn parse_ple_io_timeout_ms(value: Option<&str>) -> Result<std::time::Duration> {
        let value = value.context(format!(
            "{PLE_IO_TIMEOUT_MS_ENV} is required for bounded PLE storage waits"
        ))?;
        let millis = value
            .parse::<u64>()
            .with_context(|| format!("{PLE_IO_TIMEOUT_MS_ENV} must be an integer"))?;
        ensure!(
            (100..=300_000).contains(&millis),
            "{PLE_IO_TIMEOUT_MS_ENV} must be between 100 and 300000 milliseconds"
        );
        Ok(std::time::Duration::from_millis(millis))
    }

    fn configured_ple_io_timeout() -> Result<std::time::Duration> {
        let value = std::env::var_os(PLE_IO_TIMEOUT_MS_ENV);
        let value = value
            .as_deref()
            .map(|value| {
                value
                    .to_str()
                    .with_context(|| format!("{PLE_IO_TIMEOUT_MS_ENV} is not UTF-8"))
            })
            .transpose()?;
        parse_ple_io_timeout_ms(value)
    }

    #[cfg(test)]
    mod io_queue_depth_tests {
        use super::*;

        #[test]
        fn absent_defaults_to_wide_queue_and_exact_controls_are_admitted() {
            assert_eq!(parse_ple_io_queue_depth(None).unwrap(), 256);
            assert_eq!(parse_ple_io_queue_depth(Some("32")).unwrap(), 32);
            assert_eq!(parse_ple_io_queue_depth(Some("256")).unwrap(), 256);
        }

        #[test]
        fn ambiguous_or_unbounded_queue_depths_reject() {
            for value in ["", "0", "031", "64", "255", "257", " 256", "256 "] {
                assert!(parse_ple_io_queue_depth(Some(value)).is_err(), "{value}");
            }
        }

        #[test]
        fn ple_io_timeout_is_explicit_and_bounded() {
            assert!(parse_ple_io_timeout_ms(None).is_err());
            assert!(parse_ple_io_timeout_ms(Some("0")).is_err());
            assert!(parse_ple_io_timeout_ms(Some("99")).is_err());
            assert!(parse_ple_io_timeout_ms(Some("300001")).is_err());
            assert!(parse_ple_io_timeout_ms(Some("30s")).is_err());
            assert_eq!(
                parse_ple_io_timeout_ms(Some("30000")).unwrap(),
                std::time::Duration::from_secs(30)
            );
        }
    }

    // SAFETY: `renameat2(2)` is a standard Linux glibc/kernel C ABI symbol on
    // `target_os = "linux"` with signature `(c_int, *const c_char, c_int, *const c_char, c_uint) -> c_int`.
    unsafe extern "C" {
        fn renameat2(
            olddirfd: std::ffi::c_int,
            oldpath: *const std::ffi::c_char,
            newdirfd: std::ffi::c_int,
            newpath: *const std::ffi::c_char,
            flags: std::ffi::c_uint,
        ) -> std::ffi::c_int;
    }

    fn random_private_suffix() -> Result<String> {
        let mut bytes = [0u8; 16];
        File::open("/dev/urandom")
            .context("open OS randomness for private PLE parity staging")?
            .read_exact(&mut bytes)
            .context("read OS randomness for private PLE parity staging")?;
        Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    fn rename_noreplace(source: &Path, destination: &Path) -> Result<()> {
        const AT_FDCWD: std::ffi::c_int = -100;
        const RENAME_NOREPLACE: std::ffi::c_uint = 1;
        let source_c = CString::new(source.as_os_str().as_bytes())
            .context("PLE parity rename source contains NUL")?;
        let destination_c = CString::new(destination.as_os_str().as_bytes())
            .context("PLE parity rename destination contains NUL")?;
        // SAFETY: `source_c` and `destination_c` are valid, NUL-terminated
        // `CString`s alive for the entire call, and `AT_FDCWD`/`RENAME_NOREPLACE`
        // match the Linux `renameat2(2)` constants.
        let status = unsafe {
            renameat2(
                AT_FDCWD,
                source_c.as_ptr(),
                AT_FDCWD,
                destination_c.as_ptr(),
                RENAME_NOREPLACE,
            )
        };
        if status != 0 {
            return Err(std::io::Error::last_os_error()).with_context(|| {
                format!(
                    "atomic no-replace PLE parity rename {} -> {}",
                    source.display(),
                    destination.display()
                )
            });
        }
        Ok(())
    }

    fn private_staging_path(final_path: &Path) -> Result<PathBuf> {
        let parent = final_path
            .parent()
            .context("PLE parity final frame has no parent")?;
        let final_name = final_path
            .file_name()
            .and_then(|name| name.to_str())
            .context("PLE parity final frame name is not UTF-8")?;
        Ok(parent.join(format!(
            ".{final_name}.staging-{}",
            random_private_suffix()?
        )))
    }

    struct PleParityCaptureConfig {
        root: PathBuf,
        root_dev: u64,
        root_ino: u64,
        nonce: String,
        selector: u8,
    }

    struct PleParityArtifact {
        name: &'static str,
        bytes: usize,
        sha256: String,
        dev: u64,
        ino: u64,
    }

    fn parity_capture_config() -> Result<Option<PleParityCaptureConfig>> {
        const ALLOWED: [&[u8]; 3] = [
            PLE_PARITY_ENABLE.as_bytes(),
            PLE_PARITY_ROOT.as_bytes(),
            PLE_PARITY_NONCE.as_bytes(),
        ];
        let prefix = b"ATLAS_QWEN4_PLE_PARITY_";
        for (key, _) in std::env::vars_os() {
            let raw = key.as_os_str().as_bytes();
            ensure!(
                !raw.starts_with(prefix) || ALLOWED.contains(&raw),
                "unknown Qwen4 PLE parity environment key"
            );
        }
        let enabled = std::env::var_os(PLE_PARITY_ENABLE);
        if enabled.is_none() {
            ensure!(
                std::env::var_os(PLE_PARITY_ROOT).is_none()
                    && std::env::var_os(PLE_PARITY_NONCE).is_none(),
                "partial Qwen4 PLE parity configuration while capture is disabled"
            );
            return Ok(None);
        }
        ensure!(
            enabled
                .and_then(|value| value.into_string().ok())
                .as_deref()
                == Some("1"),
            "{PLE_PARITY_ENABLE} must be exact value 1"
        );
        let nonce = std::env::var(PLE_PARITY_NONCE)
            .with_context(|| format!("missing or non-UTF8 {PLE_PARITY_NONCE}"))?;
        ensure!(valid_parity_nonce(&nonce), "invalid PLE parity nonce");
        let root_text = std::env::var(PLE_PARITY_ROOT)
            .with_context(|| format!("missing or non-UTF8 {PLE_PARITY_ROOT}"))?;
        let root = PathBuf::from(&root_text);
        ensure!(root.is_absolute(), "PLE parity root must be absolute");
        let canonical = root
            .canonicalize()
            .with_context(|| format!("canonicalize PLE parity root {root_text}"))?;
        ensure!(canonical == root, "PLE parity root is not canonical");
        let metadata = fs::symlink_metadata(&root)
            .with_context(|| format!("stat PLE parity root {root_text}"))?;
        ensure!(metadata.is_dir(), "PLE parity root is not a directory");
        ensure!(
            metadata.permissions().mode() & 0o777 == 0o700,
            "PLE parity root mode must be 0700"
        );
        let selector = match std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH").as_deref() {
            Ok("0") => 0,
            Ok("1") => 1,
            _ => anyhow::bail!(
                "PLE parity capture requires exact ATLAS_QWEN4_PLE_PREFILL_BATCH=0 or 1"
            ),
        };
        Ok(Some(PleParityCaptureConfig {
            root,
            root_dev: metadata.dev(),
            root_ino: metadata.ino(),
            nonce,
            selector,
        }))
    }

    fn valid_parity_nonce(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn sha256_hex(bytes: &[u8]) -> Result<String> {
        const INITIAL: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        fn compress(state: &mut [u32; 8], block: &[u8]) {
            let mut words = [0u32; 64];
            for (index, word) in words[..16].iter_mut().enumerate() {
                *word = u32::from_be_bytes(
                    block[index * 4..index * 4 + 4]
                        .try_into()
                        .expect("fixed SHA256 word"),
                );
            }
            for index in 16..64 {
                let s0 = words[index - 15].rotate_right(7)
                    ^ words[index - 15].rotate_right(18)
                    ^ (words[index - 15] >> 3);
                let s1 = words[index - 2].rotate_right(17)
                    ^ words[index - 2].rotate_right(19)
                    ^ (words[index - 2] >> 10);
                words[index] = words[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(words[index - 7])
                    .wrapping_add(s1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
            for index in 0..64 {
                let upper = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let choose = (e & f) ^ (!e & g);
                let first = h
                    .wrapping_add(upper)
                    .wrapping_add(choose)
                    .wrapping_add(K[index])
                    .wrapping_add(words[index]);
                let lower = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let majority = (a & b) ^ (a & c) ^ (b & c);
                let second = lower.wrapping_add(majority);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(first);
                d = c;
                c = b;
                b = a;
                a = first.wrapping_add(second);
            }
            for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h].into_iter()) {
                *slot = slot.wrapping_add(value);
            }
        }

        let byte_len = u64::try_from(bytes.len()).context("SHA256 input length exceeds u64")?;
        let bit_len = byte_len
            .checked_mul(8)
            .context("SHA256 bit length overflow")?;
        let mut state = INITIAL;
        let mut chunks = bytes.chunks_exact(64);
        for block in &mut chunks {
            compress(&mut state, block);
        }
        let remainder = chunks.remainder();
        let mut tail = [0u8; 128];
        tail[..remainder.len()].copy_from_slice(remainder);
        tail[remainder.len()] = 0x80;
        let padded = if remainder.len() < 56 { 64 } else { 128 };
        tail[padded - 8..padded].copy_from_slice(&bit_len.to_be_bytes());
        for block in tail[..padded].chunks_exact(64) {
            compress(&mut state, block);
        }
        Ok(state.iter().map(|word| format!("{word:08x}")).collect())
    }

    fn token_sha256(tokens: &[u32]) -> Result<String> {
        token_parts_sha256(&[tokens])
    }

    fn token_parts_sha256(parts: &[&[u32]]) -> Result<String> {
        let token_count = parts.iter().try_fold(0usize, |count, part| {
            count
                .checked_add(part.len())
                .context("PLE parity token count overflow")
        })?;
        let capacity = token_count
            .checked_mul(std::mem::size_of::<u32>())
            .context("PLE parity token extent overflow")?;
        let mut bytes = Vec::with_capacity(capacity);
        for part in parts {
            for token in *part {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        sha256_hex(&bytes)
    }

    fn root_is_stable(config: &PleParityCaptureConfig) -> Result<()> {
        let metadata = fs::symlink_metadata(&config.root).context("re-stat PLE parity root")?;
        ensure!(
            metadata.is_dir()
                && metadata.dev() == config.root_dev
                && metadata.ino() == config.root_ino
                && metadata.permissions().mode() & 0o777 == 0o700,
            "PLE parity root identity or mode changed"
        );
        Ok(())
    }

    struct OwnedCreatedFile {
        path: PathBuf,
        file: Option<File>,
        identity: Option<(u64, u64)>,
        keep: bool,
    }

    impl OwnedCreatedFile {
        fn new(path: PathBuf) -> Self {
            Self {
                path,
                file: None,
                identity: None,
                keep: false,
            }
        }

        fn recover_identity(&mut self) {
            if self.identity.is_some() {
                return;
            }
            self.identity = self.file.as_ref().and_then(|file| {
                file.metadata()
                    .or_else(|_| fs::metadata(format!("/proc/self/fd/{}", file.as_raw_fd())))
                    .ok()
                    .map(|metadata| (metadata.dev(), metadata.ino()))
            });
        }
    }

    impl Drop for OwnedCreatedFile {
        fn drop(&mut self) {
            if self.keep {
                return;
            }
            self.recover_identity();
            drop(self.file.take());
            if let Some(identity) = self.identity
                && fs::symlink_metadata(&self.path)
                    .is_ok_and(|current| current.dev() == identity.0 && current.ino() == identity.1)
            {
                let _ = fs::remove_file(&self.path);
            }
        }
    }

    fn write_sealed_artifact_with_admission<F>(
        frame: &Path,
        name: &'static str,
        bytes: &[u8],
        admit: F,
    ) -> Result<PleParityArtifact>
    where
        F: FnOnce(&File) -> std::io::Result<fs::Metadata>,
    {
        let path = frame.join(name);
        let mut created = OwnedCreatedFile::new(path.clone());
        created.file = Some(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
                .with_context(|| format!("create PLE parity artifact {}", path.display()))?,
        );
        let opened = admit(created.file.as_ref().expect("created file is retained"))
            .with_context(|| format!("stat new PLE parity artifact {}", path.display()))?;
        ensure!(
            opened.is_file() && opened.nlink() == 1,
            "new PLE parity artifact is not a private regular inode"
        );
        let identity = (opened.dev(), opened.ino());
        created.identity = Some(identity);
        let result = (|| -> Result<PleParityArtifact> {
            let file = created.file.as_mut().expect("created file is retained");
            file.write_all(bytes)
                .with_context(|| format!("write PLE parity artifact {}", path.display()))?;
            file.sync_all()
                .with_context(|| format!("sync PLE parity artifact {}", path.display()))?;
            file.set_permissions(fs::Permissions::from_mode(0o444))
                .with_context(|| format!("seal PLE parity artifact {}", path.display()))?;
            file.sync_all()
                .with_context(|| format!("sync sealed PLE parity artifact {}", path.display()))?;
            let sealed = fs::symlink_metadata(&path)
                .with_context(|| format!("re-stat PLE parity artifact {}", path.display()))?;
            ensure!(
                sealed.is_file()
                    && sealed.dev() == identity.0
                    && sealed.ino() == identity.1
                    && sealed.nlink() == 1
                    && sealed.len() == bytes.len() as u64
                    && sealed.permissions().mode() & 0o777 == 0o444,
                "PLE parity artifact identity, length, or mode changed"
            );
            Ok(PleParityArtifact {
                name,
                bytes: bytes.len(),
                sha256: sha256_hex(bytes)?,
                dev: sealed.dev(),
                ino: sealed.ino(),
            })
        })();
        if result.is_ok() {
            created.keep = true;
        }
        result
    }

    fn write_sealed_artifact(
        frame: &Path,
        name: &'static str,
        bytes: &[u8],
    ) -> Result<PleParityArtifact> {
        write_sealed_artifact_with_admission(frame, name, bytes, File::metadata)
    }

    fn cleanup_owned_artifacts(frame: &Path, artifacts: &[PleParityArtifact]) {
        for artifact in artifacts.iter().rev() {
            let path = frame.join(artifact.name);
            if fs::symlink_metadata(&path)
                .is_ok_and(|current| current.dev() == artifact.dev && current.ino() == artifact.ino)
            {
                let _ = fs::remove_file(path);
            }
        }
    }

    struct OwnedPleParityFrame {
        path: PathBuf,
        final_path: PathBuf,
        handle: Option<File>,
        identity: Option<(u64, u64)>,
        artifacts: Vec<PleParityArtifact>,
        committed: bool,
    }

    impl OwnedPleParityFrame {
        fn new(path: PathBuf, final_path: PathBuf) -> Self {
            Self {
                path,
                final_path,
                handle: None,
                identity: None,
                artifacts: Vec::with_capacity(4),
                committed: false,
            }
        }

        fn recover_identity(&mut self) {
            if self.identity.is_some() {
                return;
            }
            self.identity = self.handle.as_ref().and_then(|handle| {
                handle
                    .metadata()
                    .or_else(|_| fs::metadata(format!("/proc/self/fd/{}", handle.as_raw_fd())))
                    .ok()
                    .map(|metadata| (metadata.dev(), metadata.ino()))
            });
        }

        fn identity(&self) -> (u64, u64) {
            self.identity.expect("created frame identity is admitted")
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn publish_noreplace(&mut self) -> Result<()> {
            let published_path = self.final_path.clone();
            rename_noreplace(&self.path, &published_path)?;
            // The rename already succeeded. This assignment performs no
            // allocation and keeps rollback bound to the published inode.
            self.path = published_path;
            Ok(())
        }

        fn handle_is_owned(&self) -> bool {
            self.handle.as_ref().is_some_and(|handle| {
                let metadata = handle
                    .metadata()
                    .or_else(|_| fs::metadata(format!("/proc/self/fd/{}", handle.as_raw_fd())));
                metadata.is_ok_and(|metadata| {
                    self.identity
                        .is_some_and(|identity| (metadata.dev(), metadata.ino()) == identity)
                })
            })
        }
    }

    impl Drop for OwnedPleParityFrame {
        fn drop(&mut self) {
            if self.committed {
                return;
            }
            self.recover_identity();
            if self.handle_is_owned()
                && let Some(handle) = &self.handle
            {
                let _ = handle.set_permissions(fs::Permissions::from_mode(0o700));
            }
            cleanup_owned_artifacts(&self.path, &self.artifacts);
            drop(self.handle.take());
            if let Some(identity) = self.identity
                && fs::symlink_metadata(&self.path)
                    .is_ok_and(|current| current.dev() == identity.0 && current.ino() == identity.1)
            {
                let _ = fs::remove_dir(&self.path);
            }
        }
    }

    fn create_owned_frame_with_open_and_admission<O, A>(
        final_path: PathBuf,
        root_dev: u64,
        open: O,
        admit: A,
    ) -> Result<OwnedPleParityFrame>
    where
        O: FnOnce(&Path) -> std::io::Result<File>,
        A: FnOnce(&Path) -> std::io::Result<fs::Metadata>,
    {
        let staging_path = private_staging_path(&final_path)?;
        let mut owned = OwnedPleParityFrame::new(staging_path.clone(), final_path);
        let mut builder = DirBuilder::new();
        builder.mode(0o700);
        builder.create(&staging_path).with_context(|| {
            format!(
                "create private PLE parity staging frame {}",
                staging_path.display()
            )
        })?;
        owned.handle = Some(open(&staging_path).with_context(|| {
            format!(
                "open private PLE parity staging frame {}",
                staging_path.display()
            )
        })?);
        let handle_metadata = owned
            .handle
            .as_ref()
            .expect("created frame handle is retained")
            .metadata()
            .context("stat created PLE parity frame descriptor")?;
        ensure!(
            handle_metadata.is_dir() && handle_metadata.dev() == root_dev,
            "created PLE parity frame descriptor is invalid"
        );
        owned.identity = Some((handle_metadata.dev(), handle_metadata.ino()));
        owned
            .handle
            .as_ref()
            .expect("created frame handle is retained")
            .set_permissions(fs::Permissions::from_mode(0o700))
            .context("set PLE parity frame mode0700")?;
        let admitted = admit(&staging_path).context("stat new PLE parity staging frame")?;
        let identity = owned.identity();
        ensure!(
            admitted.is_dir()
                && admitted.dev() == root_dev
                && admitted.dev() == identity.0
                && admitted.ino() == identity.1
                && admitted.permissions().mode() & 0o777 == 0o700,
            "new PLE parity frame identity or mode is invalid"
        );
        Ok(owned)
    }

    fn create_owned_frame_with_admission<F>(
        path: PathBuf,
        root_dev: u64,
        admit: F,
    ) -> Result<OwnedPleParityFrame>
    where
        F: FnOnce(&Path) -> std::io::Result<fs::Metadata>,
    {
        create_owned_frame_with_open_and_admission(path, root_dev, |path| File::open(path), admit)
    }

    fn create_owned_frame(path: PathBuf, root_dev: u64) -> Result<OwnedPleParityFrame> {
        create_owned_frame_with_admission(path, root_dev, |path| fs::symlink_metadata(path))
    }

    fn verify_sealed_artifact(frame: &Path, artifact: &PleParityArtifact) -> Result<()> {
        let path = frame.join(artifact.name);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("verify PLE parity artifact {}", path.display()))?;
        ensure!(
            metadata.is_file()
                && metadata.dev() == artifact.dev
                && metadata.ino() == artifact.ino
                && metadata.len() == artifact.bytes as u64
                && metadata.permissions().mode() & 0o777 == 0o444,
            "PLE parity artifact final identity, length, or mode changed"
        );
        let final_bytes = fs::read(&path)
            .with_context(|| format!("re-read PLE parity artifact {}", path.display()))?;
        ensure!(
            sha256_hex(&final_bytes)? == artifact.sha256,
            "PLE parity artifact final SHA256 changed"
        );
        Ok(())
    }

    struct Qwen4PlePrefillScratch {
        max_tokens: usize,
        records_dev: DevicePtr,
        scales_dev: DevicePtr,
        embedding_dev: DevicePtr,
        key_dev: DevicePtr,
        value_dev: DevicePtr,
        /// fp32 `[M, 4H]`; null under the fused join (`ATLAS_QWEN4_PLE_PREFILL_FUSED`).
        gated_value_dev: DevicePtr,
        /// fp32 `[M, 4H]`; null under the fused join.
        conv_input_dev: DevicePtr,
        dequant_k: KernelHandle,
        prepare_k: KernelHandle,
        conv_inject_k: KernelHandle,
        commit_state_k: KernelHandle,
        /// The fused join (`qwen4_ple_fused`), armed at load.
        fused: Option<Qwen4PleFusedJoin>,
    }

    /// `ATLAS_QWEN4_PLE_PREFILL_FUSED=1`: the gate scalars `[M, 4, 2]` f32 and
    /// the three kernels that replace prepare + conv_inject + commit.
    struct Qwen4PleFusedJoin {
        gate_dev: DevicePtr,
        gate_k: KernelHandle,
        inject_k: KernelHandle,
        commit_k: KernelHandle,
    }

    /// The fused join's kernels, or the refusal reason (old PTX without them).
    fn fused_join_kernels(gpu: &dyn GpuBackend) -> std::result::Result<[KernelHandle; 3], String> {
        use crate::layers::qwen4_ple_fused as fused;
        let lookup = |name: &str| {
            gpu.kernel(fused::MODULE, name)
                .map_err(|error| format!("kernel_missing kernel={name} error={error:#}"))
        };
        Ok([
            lookup(fused::GATE_KERNEL)?,
            lookup(fused::INJECT_KERNEL)?,
            lookup(fused::COMMIT_KERNEL)?,
        ])
    }

    impl Qwen4PlePrefillScratch {
        /// `fused_kernels`: arm the fused join (no fp32 intermediates).
        fn allocate(
            gpu: &dyn GpuBackend,
            max_tokens: usize,
            residual_width: usize,
            fused_kernels: Option<[KernelHandle; 3]>,
        ) -> Result<Self> {
            use crate::layers::qwen4_ple_fused::{self as fused, Buffer};
            ensure!(
                max_tokens > 0 && max_tokens <= u32::MAX as usize,
                "Qwen4 PLE prefill arena token bound is invalid: {max_tokens}"
            );
            let dequant_items = max_tokens
                .checked_mul(PLE_EMBED)
                .context("Qwen4 PLE prefill dequant extent overflow")?;
            ensure!(
                dequant_items <= u32::MAX as usize,
                "Qwen4 PLE prefill dequant extent {dequant_items} exceeds u32 grid contract"
            );
            let extents = fused::scratch_extents(
                max_tokens,
                QWEN4_PLE_HEADS,
                PLE_RECORD_BYTES,
                PLE_EMBED,
                residual_width,
                fused_kernels.is_some(),
            )
            .map_err(anyhow::Error::msg)?;
            let mut allocated: Vec<(Buffer, DevicePtr)> = Vec::with_capacity(extents.len());
            let result = (|| -> Result<Self> {
                for &(buffer, extent) in &extents {
                    allocated.push((buffer, gpu.alloc(extent)?));
                }
                let ptr = |want: Buffer| {
                    allocated
                        .iter()
                        .find(|(buffer, _)| *buffer == want)
                        .map_or(DevicePtr::NULL, |(_, ptr)| *ptr)
                };
                let fused = fused_kernels.map(|[gate_k, inject_k, commit_k]| Qwen4PleFusedJoin {
                    gate_dev: ptr(Buffer::Gate),
                    gate_k,
                    inject_k,
                    commit_k,
                });
                Ok(Self {
                    max_tokens,
                    records_dev: ptr(Buffer::Records),
                    scales_dev: ptr(Buffer::Scales),
                    embedding_dev: ptr(Buffer::Embedding),
                    key_dev: ptr(Buffer::Key),
                    value_dev: ptr(Buffer::Value),
                    gated_value_dev: ptr(Buffer::GatedValue),
                    conv_input_dev: ptr(Buffer::ConvInput),
                    dequant_k: gpu.kernel("qwen4_hyper", "qwen4_ple_dequant_prefill")?,
                    prepare_k: gpu.kernel("qwen4_hyper", "qwen4_ple_prepare_prefill")?,
                    conv_inject_k: gpu.kernel("qwen4_hyper", "qwen4_ple_conv_inject_prefill")?,
                    commit_state_k: gpu.kernel("qwen4_hyper", "qwen4_ple_commit_prefill_state")?,
                    fused,
                })
            })();
            match result {
                Ok(scratch) => Ok(scratch),
                Err(error) => {
                    let mut cleanup_errors = Vec::new();
                    for (_, ptr) in allocated.into_iter().rev() {
                        if let Err(cleanup) = gpu.free(ptr) {
                            cleanup_errors.push(format!("{ptr}: {cleanup:#}"));
                        }
                    }
                    if cleanup_errors.is_empty() {
                        Err(error.context("Qwen4 PLE prefill scratch allocation rolled back"))
                    } else {
                        Err(error.context(format!(
                            "Qwen4 PLE prefill scratch rollback failures: {}",
                            cleanup_errors.join("; ")
                        )))
                    }
                }
            }
        }

        /// Every allocated buffer (the fp32 intermediates are absent under
        /// the fused join, which has its gate scalars instead).
        fn device_buffers(&self) -> Vec<DevicePtr> {
            [
                self.records_dev,
                self.scales_dev,
                self.embedding_dev,
                self.key_dev,
                self.value_dev,
                self.gated_value_dev,
                self.conv_input_dev,
                self.fused.as_ref().map_or(DevicePtr::NULL, |f| f.gate_dev),
            ]
            .into_iter()
            .filter(|ptr| !ptr.is_null())
            .collect()
        }

        fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
            let mut errors = Vec::new();
            for ptr in self.device_buffers().into_iter().rev() {
                if let Err(error) = gpu.free(ptr) {
                    errors.push(format!("{ptr}: {error:#}"));
                }
            }
            ensure!(
                errors.is_empty(),
                "Qwen4 PLE prefill scratch release failures: {}",
                errors.join("; ")
            );
            Ok(())
        }
    }

    /// `ATLAS_QWEN4_PLE_PREFILL_STREAM`: 1 = prefill rows read with the ring
    /// kept full (`read_rows_streamed`), 2 = that plus the current read started
    /// on a helper thread before layer 0, 3 = level 2 plus exact next-chunk
    /// read-ahead while the current chunk is still executing. Same bytes as the
    /// windowed read in every mode.
    pub fn prefill_stream_level() -> u8 {
        static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        *LEVEL.get_or_init(
            || match std::env::var("ATLAS_QWEN4_PLE_PREFILL_STREAM").as_deref() {
                Ok("1") => 1,
                Ok("2") => 2,
                Ok("3") => 3,
                _ => 0,
            },
        )
    }

    fn prefill_stream_selected() -> bool {
        prefill_stream_level() >= 1
    }

    fn read_prefill_rows(
        hasher: &Qwen4PleHasher,
        reader: &Mutex<PleOffloadReader>,
        current_tokens: &[u32],
        prior_tokens: &[u32],
        streamed: bool,
        io_timeout: std::time::Duration,
        poisoned: &AtomicBool,
    ) -> Result<Vec<PleNvfp4Row>> {
        ensure!(
            !poisoned.load(Ordering::Acquire),
            "Qwen4 PLE I/O is poisoned; process restart is required"
        );
        let request: Vec<_> = hasher
            .select_prefill(current_tokens, prior_tokens)
            .iter()
            .map(|selection| (selection.shard, selection.row))
            .collect();
        ensure!(
            request.len() == current_tokens.len() * QWEN4_PLE_HEADS,
            "Qwen4 PLE prefill selection extent mismatch"
        );
        let Some(mut reader) = reader.try_lock_for(io_timeout) else {
            poisoned.store(true, Ordering::Release);
            anyhow::bail!(
                "Qwen4 PLE reader lock exceeded its deadline; PLE is poisoned and process restart is required"
            );
        };
        if streamed {
            let mut rows = vec![
                PleNvfp4Row {
                    record: [0u8; PLE_RECORD_BYTES],
                    scale2: 0.0,
                };
                request.len()
            ];
            reader.visit_rows_streamed(&request, &mut |idx, row| {
                rows[idx] = row;
                Ok(())
            })?;
            Ok(rows)
        } else {
            reader.read_rows_windowed(&request)
        }
    }

    fn visit_prefill_rows(
        hasher: &Qwen4PleHasher,
        reader: &Mutex<PleOffloadReader>,
        current_tokens: &[u32],
        prior_tokens: &[u32],
        io_timeout: std::time::Duration,
        poisoned: &AtomicBool,
        visit: &mut dyn FnMut(usize, PleNvfp4Row) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !poisoned.load(Ordering::Acquire),
            "Qwen4 PLE I/O is poisoned; process restart is required"
        );
        let request: Vec<_> = hasher
            .select_prefill(current_tokens, prior_tokens)
            .iter()
            .map(|selection| (selection.shard, selection.row))
            .collect();
        ensure!(
            request.len() == current_tokens.len() * QWEN4_PLE_HEADS,
            "Qwen4 PLE prefill selection extent mismatch"
        );
        let Some(mut reader) = reader.try_lock_for(io_timeout) else {
            poisoned.store(true, Ordering::Release);
            anyhow::bail!(
                "Qwen4 PLE reader lock exceeded its deadline; PLE is poisoned and process restart is required"
            );
        };
        reader.visit_rows_streamed(&request, visit)
    }

    /// The longest any wait on PLE storage may last: twice the I/O timeout
    /// plus 3 s (`ATLAS_QWEN4_PLE_IO_TIMEOUT_MS`).
    fn io_retire_grace(io_timeout: std::time::Duration) -> std::time::Duration {
        io_timeout
            .saturating_mul(2)
            .saturating_add(std::time::Duration::from_secs(3))
    }

    /// Join a PLE read thread within [`io_retire_grace`]. A reader that does not
    /// retire, or panics, may still own live O_DIRECT state, so the process
    /// exits without unwinding rather than drop it.
    fn join_read_thread<T>(
        handle: std::thread::JoinHandle<Result<T>>,
        io_timeout: std::time::Duration,
        poisoned: &AtomicBool,
    ) -> Result<T> {
        let grace = io_retire_grace(io_timeout);
        let deadline = std::time::Instant::now() + grace;
        while !handle.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        if !handle.is_finished() {
            poisoned.store(true, Ordering::Release);
            eprintln!(
                "fatal: Qwen4 PLE prefill reader did not retire within {grace:?}; exiting without unwinding before live I/O state can be dropped"
            );
            std::process::exit(70);
        }
        match handle.join() {
            Ok(result) => result,
            Err(_) => {
                poisoned.store(true, Ordering::Release);
                eprintln!(
                    "fatal: Qwen4 PLE prefill read thread panicked; exiting without unwinding because I/O retirement cannot be proven"
                );
                std::process::exit(70)
            }
        }
    }

    struct PackedPleBlock {
        num_rows: usize,
        records: Box<[u8]>,
        scales: Box<[u8]>,
    }

    impl PackedPleBlock {
        fn from_rows(rows: &[PleNvfp4Row]) -> Self {
            let scale_bytes = std::mem::size_of::<f32>();
            let mut records = vec![0u8; rows.len() * PLE_RECORD_BYTES];
            let mut scales = vec![0u8; rows.len() * scale_bytes];
            for (index, row) in rows.iter().enumerate() {
                let rec = index * PLE_RECORD_BYTES;
                records[rec..rec + PLE_RECORD_BYTES].copy_from_slice(&row.record);
                let sc = index * scale_bytes;
                scales[sc..sc + scale_bytes].copy_from_slice(&row.scale2.to_le_bytes());
            }
            Self {
                num_rows: rows.len(),
                records: records.into_boxed_slice(),
                scales: scales.into_boxed_slice(),
            }
        }

        fn append_to_rows(&self, out: &mut Vec<PleNvfp4Row>) {
            let scale_bytes = std::mem::size_of::<f32>();
            out.reserve(self.num_rows);
            for index in 0..self.num_rows {
                let mut record = [0u8; PLE_RECORD_BYTES];
                let rec = index * PLE_RECORD_BYTES;
                record.copy_from_slice(&self.records[rec..rec + PLE_RECORD_BYTES]);
                let sc = index * scale_bytes;
                let scale2 = f32::from_le_bytes(
                    self.scales[sc..sc + scale_bytes]
                        .try_into()
                        .expect("4-byte scale2"),
                );
                out.push(PleNvfp4Row { record, scale2 });
            }
        }
    }

    struct CachedPleChunkBlocks {
        hash: u64,
        current: Vec<u32>,
        prior: Vec<u32>,
        blocks: Arc<Vec<Arc<PackedPleBlock>>>,
    }

    fn hash_ple_chunk_key(current: &[u32], prior: &[u32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        current.hash(&mut h);
        prior.hash(&mut h);
        h.finish()
    }

    /// Default cap on cached PLE chunk rows: 2,097,152 rows (131,072 tokens, 188 MiB host RAM).
    const DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS: usize = 131_072 * QWEN4_PLE_HEADS;
    /// Approximate host-RAM footprint per cached PLE row (90 B packed record + 4 B scale2).
    const PLE_CHUNK_CACHE_BYTES_PER_ROW: usize = PLE_RECORD_BYTES + std::mem::size_of::<f32>();

    /// Resolve the chunk-block cache row limit from `ATLAS_PLE_CACHE_MB`.
    ///
    /// - Explicitly `"0"` disables chunk-block caching (`0` rows).
    /// - Positive integer MiB scales proportionally up to
    ///   [`DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS`].
    /// - Unset or unparseable preserves [`DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS`].
    fn parse_ple_chunk_cache_max_rows(raw: Option<&str>) -> usize {
        let Some(val) = raw.map(str::trim) else {
            return DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS;
        };
        let Ok(mb) = val.parse::<usize>() else {
            return DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS;
        };
        if mb == 0 {
            return 0;
        }
        let budget_bytes = mb.saturating_mul(1024 * 1024);
        (budget_bytes / PLE_CHUNK_CACHE_BYTES_PER_ROW).min(DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS)
    }

    fn ple_chunk_block_cache_max_rows() -> usize {
        static MAX_ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *MAX_ROWS.get_or_init(|| {
            parse_ple_chunk_cache_max_rows(std::env::var("ATLAS_PLE_CACHE_MB").ok().as_deref())
        })
    }

    fn ple_chunk_block_cache() -> &'static Mutex<std::collections::VecDeque<CachedPleChunkBlocks>> {
        static CACHE: std::sync::OnceLock<Mutex<std::collections::VecDeque<CachedPleChunkBlocks>>> =
            std::sync::OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(std::collections::VecDeque::with_capacity(32)))
    }

    /// Read a chunk's rows on a helper thread, publishing each complete block
    /// of `BLOCK_TOKENS` tokens (selections are token-major, 16 per token).
    /// Row ranges of the per-block reader-lock holds of a yielding read: one
    /// `BLOCK_TOKENS`-token block each, in order, covering every row once.
    pub(crate) fn yielding_block_ranges(
        total_rows: usize,
        rows_per_block: usize,
    ) -> Vec<std::ops::Range<usize>> {
        (0..total_rows)
            .step_by(rows_per_block.max(1))
            .map(|start| start..(start + rows_per_block).min(total_rows))
            .collect()
    }

    #[cfg(test)]
    mod yielding_tests {
        use super::yielding_block_ranges;

        #[test]
        fn yielding_ranges_cover_every_row_once_in_block_order() {
            let rows_per_block = 1024 * 16;
            for total in [0usize, 16, 16_384, 16_400, 262_144, 196_608 + 48] {
                let ranges = yielding_block_ranges(total, rows_per_block);
                assert_eq!(ranges.len(), total.div_ceil(rows_per_block), "{total}");
                let mut next = 0;
                for range in &ranges {
                    assert_eq!(range.start, next, "{total}: contiguous, in order");
                    assert!(range.len() <= rows_per_block && !range.is_empty());
                    next = range.end;
                }
                assert_eq!(next, total, "{total}: every row once");
            }
            // Only the last block may be short.
            let ranges = yielding_block_ranges(16_384 * 2 + 32, 16_384);
            assert_eq!(
                ranges.iter().map(|r| r.len()).collect::<Vec<_>>(),
                [16_384, 16_384, 32]
            );
        }

        #[test]
        fn chunk_cache_row_limit_honors_ple_cache_mb_zero_and_scales() {
            assert_eq!(super::parse_ple_chunk_cache_max_rows(Some("0")), 0);
            assert_eq!(super::parse_ple_chunk_cache_max_rows(Some(" 0 ")), 0);
            assert_eq!(
                super::parse_ple_chunk_cache_max_rows(None),
                super::DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS
            );
            assert_eq!(
                super::parse_ple_chunk_cache_max_rows(Some("invalid")),
                super::DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS
            );
            let at_64mb = super::parse_ple_chunk_cache_max_rows(Some("64"));
            assert!(at_64mb > 0 && at_64mb < super::DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS);
            assert_eq!(
                super::parse_ple_chunk_cache_max_rows(Some("1024")),
                super::DEFAULT_PLE_CHUNK_CACHE_MAX_ROWS
            );
        }
    }

    /// `yielding` (`ATLAS_QWEN4_PLE_QUEUE_PREFETCH`'s background read): take the
    /// reader lock per block and hand it on fairly between blocks, so a
    /// running request's decode reads (16 rows per token, same reader) wait
    /// at most one block read instead of the whole chunk's.
    fn spawn_block_read(
        hasher: Arc<Qwen4PleHasher>,
        reader: Arc<Mutex<PleOffloadReader>>,
        current: Vec<u32>,
        prior: Vec<u32>,
        io_timeout: std::time::Duration,
        poisoned: Arc<AtomicBool>,
        yielding: bool,
    ) -> Result<PleReadKind> {
        use crate::layers::qwen4_ple_pipeline::BLOCK_TOKENS;
        let blocks = current.len().div_ceil(BLOCK_TOKENS);
        let (tx, rx) = std::sync::mpsc::channel();
        let max_cached_rows = ple_chunk_block_cache_max_rows();
        let key_hash = if max_cached_rows > 0 {
            hash_ple_chunk_key(&current, &prior)
        } else {
            0
        };
        let cached = if max_cached_rows > 0 {
            let mut cache = ple_chunk_block_cache().lock();
            if let Some(idx) = cache.iter().position(|entry| {
                entry.hash == key_hash && entry.current == current && entry.prior == prior
            }) {
                let entry = cache.remove(idx).expect("checked index");
                let blocks_arc = Arc::clone(&entry.blocks);
                cache.push_back(entry);
                Some(blocks_arc)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(cached_blocks) = cached {
            for (block_idx, block_rows) in cached_blocks.iter().enumerate() {
                let _ = tx.send((block_idx, Arc::clone(block_rows)));
            }
            return Ok(PleReadKind::Blocks(PleBlockRead {
                rx,
                thread: None,
                blocks,
                io_timeout,
                poisoned,
            }));
        }
        let thread_poisoned = Arc::clone(&poisoned);
        let thread = std::thread::Builder::new()
            .name("ple-prefill-blocks".into())
            .spawn(move || -> Result<()> {
                // ATLAS_PLE_READ_AFFINITY (K10 item A): placement only; read rings inherit it.
                let placement = read_affinity::begin();
                let total_rows = current.len() * QWEN4_PLE_HEADS;
                let request: Vec<_> = hasher
                    .select_prefill(&current, &prior)
                    .iter()
                    .map(|selection| (selection.shard, selection.row))
                    .collect();
                ensure!(
                    request.len() == total_rows,
                    "Qwen4 PLE prefill selection extent mismatch"
                );
                let rows_per_block = BLOCK_TOKENS * QWEN4_PLE_HEADS;
                let mut collected_blocks = if max_cached_rows > 0 {
                    Vec::with_capacity(blocks)
                } else {
                    Vec::new()
                };
                let lock_reader = || {
                    reader.try_lock_for(io_timeout).ok_or_else(|| {
                        thread_poisoned.store(true, Ordering::Release);
                        anyhow::anyhow!(
                            "Qwen4 PLE reader lock exceeded its deadline; PLE is poisoned and process restart is required"
                        )
                    })
                };
                let result = if yielding {
                    let mut result = Ok(());
                    let mut abandoned = false;
                    for (block, range) in yielding_block_ranges(request.len(), rows_per_block)
                        .into_iter()
                        .enumerate()
                    {
                        let mut guard = lock_reader()?;
                        result = guard.read_rows_streamed_blocks(
                            &request[range],
                            rows_per_block,
                            &mut |_, rows| {
                                let packed = Arc::new(PackedPleBlock::from_rows(&rows));
                                if max_cached_rows > 0 {
                                    collected_blocks.push(Arc::clone(&packed));
                                }
                                abandoned |= tx.send((block, packed)).is_err();
                                Ok(())
                            },
                        );
                        // Hand the reader straight to a waiting decode read.
                        parking_lot::MutexGuard::unlock_fair(guard);
                        // A discarded background read stops at the block it is
                        // on instead of reading the rest of the chunk.
                        if result.is_err() || abandoned {
                            break;
                        }
                    }
                    result
                } else {
                    let mut reader = lock_reader()?;
                    reader.read_rows_streamed_blocks(&request, rows_per_block, &mut |block, rows| {
                        let packed = Arc::new(PackedPleBlock::from_rows(&rows));
                        if max_cached_rows > 0 {
                            collected_blocks.push(Arc::clone(&packed));
                        }
                        // A consumer that went away (a discarded prefetch) needs no rows.
                        let _ = tx.send((block, packed));
                        Ok(())
                    })
                };
                read_affinity::end(placement, request.len());
                if max_cached_rows > 0 && result.is_ok() && collected_blocks.len() == blocks {
                    let mut cache = ple_chunk_block_cache().lock();
                    let mut current_rows: usize = cache
                        .iter()
                        .map(|e| e.current.len() * QWEN4_PLE_HEADS)
                        .sum();
                    while current_rows.saturating_add(total_rows) > max_cached_rows {
                        let Some(evicted) = cache.pop_front() else {
                            break;
                        };
                        current_rows =
                            current_rows.saturating_sub(evicted.current.len() * QWEN4_PLE_HEADS);
                    }
                    if current_rows.saturating_add(total_rows) <= max_cached_rows {
                        cache.push_back(CachedPleChunkBlocks {
                            hash: key_hash,
                            current,
                            prior,
                            blocks: Arc::new(collected_blocks),
                        });
                    }
                }
                result
            })?;
        Ok(PleReadKind::Blocks(PleBlockRead {
            rx,
            thread: Some(thread),
            blocks,
            io_timeout,
            poisoned,
        }))
    }

    /// PLE rows being read on a helper thread (see `begin_prefill_read`): the
    /// whole chunk at once, or (`ATLAS_QWEN4_PLE_PREFILL_PIPELINE`) in
    /// published blocks of `BLOCK_TOKENS` tokens. Every wait on it is bounded
    /// by [`io_retire_grace`].
    pub struct PlePrefillRead {
        kind: Result<PleReadKind>,
        io_timeout: std::time::Duration,
        poisoned: Arc<AtomicBool>,
    }

    enum PleReadKind {
        Whole(std::thread::JoinHandle<Result<Vec<PleNvfp4Row>>>),
        Blocks(PleBlockRead),
    }

    /// Blocks arrive in order on `rx`; the thread's result carries any I/O error.
    struct PleBlockRead {
        rx: std::sync::mpsc::Receiver<(usize, Arc<PackedPleBlock>)>,
        thread: Option<std::thread::JoinHandle<Result<()>>>,
        blocks: usize,
        io_timeout: std::time::Duration,
        poisoned: Arc<AtomicBool>,
    }

    impl PleBlockRead {
        /// Block `expected`, waiting for it if `wait`; `None` if not yet published.
        fn take(&self, expected: usize, wait: bool) -> Result<Option<Arc<PackedPleBlock>>> {
            let received = if wait {
                let grace = io_retire_grace(self.io_timeout);
                match self.rx.recv_timeout(grace) {
                    Ok(block) => Some(block),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => None,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // The reader may still own live O_DIRECT state: the rule of
                        // a read thread that does not retire (`join_read_thread`).
                        self.poisoned.store(true, Ordering::Release);
                        eprintln!(
                            "fatal: Qwen4 PLE block {expected} was not published within {grace:?}; exiting without unwinding before live I/O state can be dropped"
                        );
                        std::process::exit(70);
                    }
                }
            } else {
                match self.rx.try_recv() {
                    Ok(block) => Some(block),
                    Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(None),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
                }
            };
            let (index, rows) = received
                .with_context(|| format!("Qwen4 PLE block read ended before block {expected}"))?;
            ensure!(
                index == expected,
                "Qwen4 PLE block read published block {index}, expected {expected}"
            );
            Ok(Some(rows))
        }

        fn finish(self) -> Result<()> {
            if let Some(thread) = self.thread {
                join_read_thread(thread, self.io_timeout, &self.poisoned)
            } else {
                Ok(())
            }
        }

        /// Every row, in order (the read thread's error wins over a short read).
        fn join_all(self) -> Result<Vec<PleNvfp4Row>> {
            let mut rows = Vec::new();
            for block in 0..self.blocks {
                match self.take(block, true) {
                    Ok(Some(block_rows)) => block_rows.append_to_rows(&mut rows),
                    Ok(None) => unreachable!("a blocking take always yields a block"),
                    Err(short) => {
                        self.finish()?;
                        return Err(short);
                    }
                }
            }
            self.finish()?;
            Ok(rows)
        }
    }

    /// How an async PLE prefill consumes its rows.
    enum PleRowsPlan {
        /// Every row is (or, for a short chunk, is waited to be) available.
        Whole(Vec<PleNvfp4Row>),
        /// A block read still in flight, with the blocks already published.
        Pipelined {
            read: PleBlockRead,
            ready: Vec<Arc<PackedPleBlock>>,
        },
    }

    impl PleRowsPlan {
        fn from_read(read: PlePrefillRead, num_tokens: usize) -> Result<Self> {
            let PlePrefillRead {
                kind,
                io_timeout,
                poisoned,
            } = read;
            let read = match kind? {
                PleReadKind::Whole(thread) => {
                    return Ok(Self::Whole(join_read_thread(
                        thread, io_timeout, &poisoned,
                    )?));
                }
                PleReadKind::Blocks(read) => read,
            };
            ensure!(
                read.blocks == num_tokens.div_ceil(crate::layers::qwen4_ple_pipeline::BLOCK_TOKENS),
                "Qwen4 PLE block read does not match the chunk"
            );
            let mut ready = Vec::new();
            while ready.len() < read.blocks {
                match read.take(ready.len(), false) {
                    Ok(Some(rows)) => ready.push(rows),
                    Ok(None) => break,
                    Err(short) => {
                        read.finish()?;
                        return Err(short);
                    }
                }
            }
            if read.blocks < 2 {
                return Ok(Self::Whole(Self::finish_whole(read, ready)?));
            }
            Ok(Self::Pipelined { read, ready })
        }

        /// The remaining blocks, waited for, appended to `ready`.
        fn finish_whole(
            read: PleBlockRead,
            ready: Vec<Arc<PackedPleBlock>>,
        ) -> Result<Vec<PleNvfp4Row>> {
            let mut block = ready.len();
            let mut rows = Vec::new();
            for b in ready {
                b.append_to_rows(&mut rows);
            }
            while block < read.blocks {
                match read.take(block, true) {
                    Ok(Some(block_rows)) => block_rows.append_to_rows(&mut rows),
                    Ok(None) => unreachable!("a blocking take always yields a block"),
                    Err(short) => {
                        read.finish()?;
                        return Err(short);
                    }
                }
                block += 1;
            }
            read.finish()?;
            Ok(rows)
        }
    }

    impl PlePrefillRead {
        pub(crate) fn join(self) -> Result<Vec<PleNvfp4Row>> {
            match self.kind? {
                PleReadKind::Whole(thread) => {
                    join_read_thread(thread, self.io_timeout, &self.poisoned)
                }
                PleReadKind::Blocks(read) => read.join_all(),
            }
        }
    }

    /// One detached next-chunk read, retained by sequence slot until the
    /// scheduler re-enters `prefill_chunk` for that exact token span.
    pub struct PleChunkPrefetch {
        key: PlePrefetchKey,
        read: PlePrefillRead,
    }

    impl PleChunkPrefetch {
        pub(crate) fn new(key: PlePrefetchKey, read: PlePrefillRead) -> Self {
            Self { key, read }
        }

        pub(crate) fn matches(
            &self,
            chunk_start: usize,
            current_tokens: &[u32],
            prior_tokens: &[u32],
        ) -> bool {
            self.key.matches(chunk_start, current_tokens, prior_tokens)
        }

        pub(crate) fn into_read(self) -> PlePrefillRead {
            self.read
        }

        pub(crate) fn finish_discard(self) -> Result<()> {
            self.read.join().map(|_| ())
        }
    }

    /// Atlas-native sparse PLE runtime. Only the 16 selected NVFP4 rows are
    /// staged per token; the 320M-row table remains in the O_DIRECT sidecar.
    pub struct Qwen4PleLayer {
        hasher: Arc<Qwen4PleHasher>,
        reader: Arc<Mutex<PleOffloadReader>>,
        /// Shared lock-free direct-mapped parked PLE record cache owned alongside
        /// `reader` so decode/verify hits skip `reader.lock()` completely.
        parked_cache: Arc<ParkedPleCache>,
        /// True when speculative background `DraftAhead` warm-ups are enabled.
        /// Defaults to `reader.has_io_uring()` so fast async `io_uring` hosts
        /// overlap bonus-token reads with drafting, while non-`io_uring`
        /// container runtimes avoid stalling foreground verify on `reader.lock()`.
        draft_ahead_enabled: bool,
        io_timeout: std::time::Duration,
        io_poisoned: Arc<AtomicBool>,
        scratch_submit: Mutex<()>,
        scratch_event: u64,
        scratch_poisoned: AtomicBool,
        key_proj: DenseWeight,
        value_proj: DenseWeight,
        norm_key: DenseWeight,
        norm_query: DenseWeight,
        norm_conv: DenseWeight,
        conv_weight: DenseWeight,
        records_dev: DevicePtr,
        scales_dev: DevicePtr,
        embedding_dev: DevicePtr,
        key_dev: DevicePtr,
        value_dev: DevicePtr,
        conv_state_dev: DevicePtr,
        conv_checkpoint_dev: DevicePtr,
        conv_intermediate_dev: DevicePtr,
        dequant_k: KernelHandle,
        fuse_k: KernelHandle,
        /// `qwen4_ple_fuse_verify_rows` (0 when absent): fused multi-row verify PLE join + intermediate state capture.
        fuse_verify_rows_k: KernelHandle,
        /// `qwen4_ple_fuse_verify_rows_ragged` (0 when absent): 1-launch ragged multi-sequence verify PLE join + intermediate state capture.
        fuse_verify_rows_ragged_k: KernelHandle,
        dense_gemv_k: KernelHandle,
        /// `dense_gemv_bf16_batchn_tf` (0 when absent): batched verify-row projections.
        dense_gemv_batchn_tf_k: KernelHandle,
        prefill: Option<Qwen4PlePrefillScratch>,
        /// Batched speculative-verify/native-decode staging, reserved during
        /// model initialization and serialized under `scratch_submit`.
        verify: Mutex<Option<verify_batch::Qwen4PleVerifyScratch>>,
        /// Lazily spawned persistent worker overlapping cross-sequence verify
        /// PLE reads with embedding + layer 0 without per-step thread spawns.
        warm_worker: Mutex<Option<verify_batch::PleWarmWorker>>,
        hidden_size: usize,
        hc_count: usize,
        max_batch_size: usize,
        state_allocation: super::Qwen4PleAllocationBytes,
        eps: f32,
    }

    impl Qwen4PleLayer {
        pub(crate) fn batch_verify_lane_ranges(
            &self,
            slot: usize,
        ) -> Result<Vec<(DevicePtr, usize, DevicePtr, usize)>> {
            ensure!(
                slot < self.max_batch_size,
                "PLE verify slot is outside actual state allocations"
            );
            let actual = self.state_allocation;
            ensure!(
                actual
                    == super::checked_ple_allocation_bytes(
                        self.max_batch_size,
                        self.residual_width(),
                    )?,
                "PLE verify allocation metadata drifted"
            );
            let live = actual
                .state
                .checked_div(self.max_batch_size)
                .context("PLE verify live span division failed")?;
            let intermediate = live
                .checked_mul(super::PLE_MAX_SPEC_ROWS)
                .context("PLE verify intermediate span overflow")?;
            let mut ranges = Vec::with_capacity(3);
            for (base, bytes, stride) in [
                (self.conv_state_dev, actual.state, live),
                (self.conv_checkpoint_dev, actual.checkpoint, live),
                (
                    self.conv_intermediate_dev,
                    actual.intermediate,
                    intermediate,
                ),
            ] {
                let offset = slot
                    .checked_mul(stride)
                    .context("PLE verify span offset overflow")?;
                ensure!(
                    base.0 != 0 && offset.checked_add(stride).is_some_and(|end| end <= bytes),
                    "PLE verify span exceeds actual allocation"
                );
                ranges.push((base, bytes, base.offset(offset), stride));
            }
            Ok(ranges)
        }

        pub fn load(
            store: &WeightStore,
            config: &ModelConfig,
            gpu: &dyn GpuBackend,
            max_batch_size: usize,
        ) -> Result<Option<Self>> {
            if !config.is_qwen4_exp() || config.ple_layer_ids.is_empty() {
                return Ok(None);
            }
            let manifest = config.ple_offload_manifest.as_deref().context(
                "Qwen4 PLE checkpoint is missing ple-offload/manifest.json; refusing to run without the position-learning enhancement table",
            )?;
            let manifest_override = std::env::var("ATLAS_PLE_OFFLOAD_DIR").ok();
            let manifest_path = super::resolve_ple_offload_manifest_path_from(
                manifest,
                manifest_override.as_deref(),
            )?;
            if manifest_override
                .as_deref()
                .is_some_and(|v| !v.trim().is_empty())
            {
                tracing::info!(
                    selector = "ATLAS_PLE_OFFLOAD_DIR",
                    manifest = %manifest_path.display(),
                    default_manifest = manifest,
                    "QWEN4_PLE_OFFLOAD_DIR_OVERRIDE"
                );
            }
            ensure!(
                config.hidden_size == PLE_EMBED,
                "Qwen4 PLE hidden size mismatch"
            );
            ensure!(config.hc_count == 4, "Qwen4 PLE requires hc_count=4");
            let residual = config.residual_width();
            let allocation = checked_ple_allocation_bytes(max_batch_size, residual)?;
            let prefix = format!("{}.layers.1.ple", config.weight_prefix);
            check_tensor(
                store,
                &format!("{prefix}.key_proj.weight"),
                &[10240, 2560],
                WeightDtype::BF16,
            )?;
            check_tensor(
                store,
                &format!("{prefix}.value_proj.weight"),
                &[2560, 2560],
                WeightDtype::BF16,
            )?;
            check_tensor(
                store,
                &format!("{prefix}.conv1d.weight"),
                &[10240, 1, 4],
                WeightDtype::BF16,
            )?;
            for name in ["norm_key.weight", "norm_query.weight", "norm_conv.weight"] {
                let full_name = format!("{prefix}.{name}");
                let w = store.get(&full_name)?;
                ensure!(
                    w.shape == [10240] && matches!(w.dtype, WeightDtype::BF16 | WeightDtype::FP32),
                    "Qwen4 PLE tensor {full_name} must be BF16/FP32 [10240], got {:?} {:?}",
                    w.dtype,
                    w.shape
                );
            }

            let ep = format!("{prefix}.ple_embedding");
            let multipliers = read_i64::<3>(store, &format!("{ep}.layer_multipliers"), gpu)?;
            let vocab_sizes =
                read_i64::<QWEN4_PLE_HEADS>(store, &format!("{ep}.ngram_heads_vocab_sizes"), gpu)?;
            let offsets =
                read_i64::<QWEN4_PLE_HEADS>(store, &format!("{ep}.ngram_heads_offsets"), gpu)?;
            let hasher = Qwen4PleHasher::new(
                multipliers,
                vocab_sizes,
                offsets,
                config.eos_token_id,
                2_500_012,
            )?;

            // The published NVFP4 recipe explicitly ignores `*.ple.*`.
            // Preserve those checkpoint projections in BF16 rather than
            // introducing a second, unsupported quantization pass.
            let key_proj = dense(store, &format!("{prefix}.key_proj.weight"))?;
            let value_proj = dense(store, &format!("{prefix}.value_proj.weight"))?;
            let page_cache = std::env::var("ATLAS_PLE_PAGE_CACHE").ok().as_deref() == Some("1");
            let io_mode = if page_cache {
                PleIoMode::PageCache
            } else {
                PleIoMode::Direct
            };
            // Flash-Next already leaves only a few GiB of unified-memory
            // headroom on one Spark. Keep the `PleOffloadReader` private row
            // cache as an explicit opt-in (`cache_mb = 0` when unset; a
            // 10-request Weschera gate found no decode benefit and extra
            // pressure could stall later requests). The separate prefill
            // chunk-block cache defaults to 188 MiB unless
            // `ATLAS_PLE_CACHE_MB=0` is set explicitly.
            let cache_mb = std::env::var("ATLAS_PLE_CACHE_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            let io_queue_depth = configured_ple_io_queue_depth()?;
            let io_timeout = configured_ple_io_timeout()?;
            tracing::info!(
                ple_io_queue_depth = io_queue_depth,
                ple_io_timeout_ms = io_timeout.as_millis(),
                ple_io_mode = ?io_mode,
                "configured Qwen4 PLE offload reader"
            );
            let reader = PleOffloadReader::open_with_mode(
                manifest_path.as_path(),
                io_queue_depth,
                cache_mb * 1024 * 1024,
                io_mode,
                io_timeout,
            )
            .with_context(|| {
                format!(
                    "open Qwen4 PLE offload manifest {}",
                    manifest_path.display()
                )
            })?;
            // ATLAS_QWEN4_PLE_WARM_READ=<rows per shard> (default off): touch the reader once at
            // load (io_uring rings, registered buffers, SSD out of idle) so the first prompt does
            // not pay the cold-reader cost (~286 ms measured on the first FN request).
            let mut reader = reader;
            if let Some(per_shard) = std::env::var("ATLAS_QWEN4_PLE_WARM_READ")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
            {
                let t0 = std::time::Instant::now();
                let selections: Vec<(usize, usize)> = reader
                    .shard_rows()
                    .iter()
                    .enumerate()
                    .flat_map(|(shard, &rows)| {
                        let n = per_shard.min(rows.max(1));
                        (0..n).map(move |i| (shard, i * rows / n))
                    })
                    .collect();
                // One read per queue-depth batch: a single request may not exceed the reader's
                // queue depth in unique pages (64 rows x 128 shards = 8192 > 256 failed the build).
                for batch in selections.chunks(reader.queue_depth().max(1)) {
                    reader.read_rows(batch)?;
                }
                tracing::info!(
                    "QWEN4_PLE_WARM_READ_ENGAGED rows={} ms={:.1}",
                    selections.len(),
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            let has_io_uring = reader.has_io_uring();
            let draft_ahead_enabled = match std::env::var("ATLAS_QWEN4_PLE_DRAFT_AHEAD")
                .ok()
                .as_deref()
                .map(str::trim)
            {
                Some("1") => true,
                Some("0") => false,
                _ => has_io_uring,
            };
            if !draft_ahead_enabled {
                tracing::info!(
                    has_io_uring,
                    "QWEN4_PLE_DRAFT_AHEAD_SKIPPED reason={} (override with ATLAS_QWEN4_PLE_DRAFT_AHEAD=1)",
                    if has_io_uring {
                        "env_disabled"
                    } else {
                        "no_io_uring"
                    }
                );
            }
            let parked_cache = reader.parked_cache();
            let layer = Self {
                hasher: Arc::new(hasher),
                reader: Arc::new(Mutex::new(reader)),
                parked_cache,
                draft_ahead_enabled,
                io_timeout,
                io_poisoned: Arc::new(AtomicBool::new(false)),
                scratch_submit: Mutex::new(()),
                scratch_event: gpu.create_event()?,
                scratch_poisoned: AtomicBool::new(false),
                key_proj,
                value_proj,
                norm_key: dense_f32_safe(store, &format!("{prefix}.norm_key.weight"), gpu)?,
                norm_query: dense_f32_safe(store, &format!("{prefix}.norm_query.weight"), gpu)?,
                norm_conv: dense_f32_safe(store, &format!("{prefix}.norm_conv.weight"), gpu)?,
                conv_weight: dense(store, &format!("{prefix}.conv1d.weight"))?,
                records_dev: gpu.alloc(QWEN4_PLE_HEADS * PLE_RECORD_BYTES)?,
                scales_dev: gpu.alloc(QWEN4_PLE_HEADS * 4)?,
                embedding_dev: gpu.alloc(PLE_EMBED * 2)?,
                key_dev: gpu.alloc(residual * 2)?,
                value_dev: gpu.alloc(PLE_EMBED * 2)?,
                conv_state_dev: gpu.alloc(allocation.state)?,
                conv_checkpoint_dev: gpu.alloc(allocation.checkpoint)?,
                state_allocation: allocation,
                conv_intermediate_dev: gpu.alloc(allocation.intermediate)?,
                dequant_k: gpu.kernel("qwen4_hyper", "qwen4_ple_dequant_rows")?,
                fuse_k: gpu.kernel("qwen4_hyper", "qwen4_ple_fuse_decode")?,
                fuse_verify_rows_k: crate::layers::try_kernel(
                    gpu,
                    "qwen4_hyper",
                    "qwen4_ple_fuse_verify_rows",
                ),
                fuse_verify_rows_ragged_k: crate::layers::try_kernel(
                    gpu,
                    "qwen4_hyper",
                    "qwen4_ple_fuse_verify_rows_ragged",
                ),
                dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
                dense_gemv_batchn_tf_k: crate::layers::try_kernel(
                    gpu,
                    "gemv",
                    "dense_gemv_bf16_batchn_tf",
                ),
                prefill: None,
                verify: Mutex::new(None),
                warm_worker: Mutex::new(None),
                hidden_size: config.hidden_size,
                hc_count: config.hc_count,
                max_batch_size,
                eps: config.rms_norm_eps as f32,
            };
            gpu.memset(layer.conv_state_dev, 0, allocation.state)?;
            gpu.memset(layer.conv_checkpoint_dev, 0, allocation.checkpoint)?;
            gpu.memset(layer.conv_intermediate_dev, 0, allocation.intermediate)?;
            tracing::info!(
                manifest = %manifest_path.display(),
                cache_mb,
                ?io_mode,
                "Qwen4 PLE sparse NVFP4 offload enabled"
            );
            Ok(Some(layer))
        }

        /// Allocate the opt-in whole-prompt arena only after the enclosing
        /// model is fully constructed. A later factory error then runs the
        /// model's synchronized Drop path instead of leaking raw pointers.
        pub fn initialize_prefill(
            &mut self,
            gpu: &dyn GpuBackend,
            max_prefill_tokens: usize,
        ) -> Result<()> {
            use crate::layers::qwen4_ple_fused as fused;
            let fused_selected = fused::selected().map_err(anyhow::Error::msg)?;
            if std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH")
                .ok()
                .as_deref()
                != Some("1")
            {
                if fused_selected {
                    tracing::info!("{}", fused::refused_line("prefill_batch_off"));
                }
                return Ok(());
            }
            ensure!(
                self.prefill.is_none(),
                "Qwen4 PLE prefill scratch is already initialized"
            );
            let fused_kernels = if fused_selected {
                match fused::check_geometry(self.hidden_size, self.hc_count)
                    .map_err(|why| format!("geometry {why}"))
                    .and_then(|()| fused_join_kernels(gpu))
                {
                    Ok(kernels) => Some(kernels),
                    Err(reason) => {
                        tracing::warn!("{}", fused::refused_line(&reason));
                        None
                    }
                }
            } else {
                None
            };
            self.prefill = Some(Qwen4PlePrefillScratch::allocate(
                gpu,
                max_prefill_tokens,
                self.residual_width(),
                fused_kernels,
            )?);
            if self
                .prefill
                .as_ref()
                .is_some_and(|scratch| scratch.fused.is_some())
            {
                let bytes = |fused_join: bool| {
                    fused::scratch_extents(
                        max_prefill_tokens,
                        QWEN4_PLE_HEADS,
                        PLE_RECORD_BYTES,
                        PLE_EMBED,
                        self.residual_width(),
                        fused_join,
                    )
                    .map(|extents| fused::total_bytes(&extents))
                    .map_err(anyhow::Error::msg)
                };
                let (unfused, armed) = (bytes(false)?, bytes(true)?);
                tracing::info!(
                    "{}",
                    fused::engaged_line(max_prefill_tokens, armed, unfused.saturating_sub(armed))
                );
            }
            Ok(())
        }

        /// Fetch, project, and inject one token. `prior_tokens` excludes the
        /// current input token, matching the official n-gram shift contract.
        pub fn forward_token(
            &self,
            current_token: u32,
            prior_tokens: &[u32],
            hyper: DevicePtr,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            // records/scales/embedding/key/value are shared staging buffers.
            // Serialize host enqueue order and chain their GPU use across
            // streams; a warm host-page hit can otherwise overtake unfinished
            // work that cold O_DIRECT latency happened to hide.
            let _submit = self.scratch_submit.lock();
            ensure!(
                !self.scratch_poisoned.load(Ordering::Acquire),
                "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
            );
            let selections = self.hasher.select_decode(current_token, prior_tokens);
            let request: [(usize, usize); QWEN4_PLE_HEADS] =
                std::array::from_fn(|head| (selections[head].shard, selections[head].row));
            ensure!(
                !self.io_poisoned.load(Ordering::Acquire),
                "Qwen4 PLE I/O is poisoned; process restart is required"
            );
            let mut records = [0u8; QWEN4_PLE_HEADS * PLE_RECORD_BYTES];
            let mut scales = [0u8; QWEN4_PLE_HEADS * 4];
            if !self
                .parked_cache
                .try_fill_staging(&request, &mut records, &mut scales)?
            {
                let Some(mut reader) = self.reader.try_lock_for(self.io_timeout) else {
                    self.io_poisoned.store(true, Ordering::Release);
                    anyhow::bail!(
                        "Qwen4 PLE reader lock exceeded its deadline; PLE is poisoned and process restart is required"
                    );
                };
                let rows = reader.read_rows(&request)?;
                drop(reader);
                for (head, row) in rows.iter().enumerate() {
                    let start = head * PLE_RECORD_BYTES;
                    records[start..start + PLE_RECORD_BYTES].copy_from_slice(&row.record);
                    scales[head * 4..head * 4 + 4].copy_from_slice(&row.scale2.to_le_bytes());
                }
            }
            if let Err(error) = gpu.stream_wait_event(stream, self.scratch_event) {
                self.scratch_poisoned.store(true, Ordering::Release);
                return Err(error.context(
                    "Qwen4 PLE scratch predecessor wait failed; shared scratch poisoned",
                ));
            }
            let execution = (|| -> Result<()> {
                gpu.copy_h2d_group_on_stream(
                    &[
                        HostToDeviceCopy::new(&records, self.records_dev),
                        HostToDeviceCopy::new(&scales, self.scales_dev),
                    ],
                    stream,
                )?;
                self.launch_decode_row(
                    self.records_dev,
                    self.scales_dev,
                    hyper,
                    slot_idx,
                    reset_state,
                    gpu,
                    stream,
                )
            })();
            if let Err(error) = execution {
                let fence = gpu
                    .synchronize(stream)
                    .and_then(|()| gpu.record_event(self.scratch_event, stream));
                return match fence {
                    Ok(()) => Err(error.context(
                        "Qwen4 PLE decode execution failed after shared scratch was fenced",
                    )),
                    Err(fence_error) => {
                        self.scratch_poisoned.store(true, Ordering::Release);
                        Err(error.context(format!(
                            "Qwen4 PLE decode execution failed and fencing failed ({fence_error:#}); shared scratch poisoned"
                        )))
                    }
                };
            }
            if let Err(error) = gpu.record_event(self.scratch_event, stream) {
                self.scratch_poisoned.store(true, Ordering::Release);
                return Err(error.context(
                    "Qwen4 PLE post-success scratch fence failed; shared scratch poisoned",
                ));
            }
            Ok(())
        }

        /// One decode row's PLE kernels: dequantize the 16 staged records,
        /// key/value projections, then the conv/fuse update of `hyper` and the
        /// slot's conv history. The batched verify path issues the same three
        /// launches (`launch_dequant` / `launch_projections` or the bit-identical
        /// batched projection / `launch_fuse`) per row.
        #[allow(clippy::too_many_arguments)]
        fn launch_decode_row(
            &self,
            records_dev: DevicePtr,
            scales_dev: DevicePtr,
            hyper: DevicePtr,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            self.launch_dequant(records_dev, scales_dev, self.embedding_dev, gpu, stream)?;
            self.launch_projections(
                self.embedding_dev,
                self.key_dev,
                self.value_dev,
                gpu,
                stream,
            )?;
            self.launch_fuse(
                hyper,
                self.key_dev,
                self.value_dev,
                slot_idx,
                reset_state,
                gpu,
                stream,
            )
        }

        /// Dequantize one row's 16 staged records into a `[PLE_EMBED]` BF16 embedding.
        fn launch_dequant(
            &self,
            records_dev: DevicePtr,
            scales_dev: DevicePtr,
            embedding: DevicePtr,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            KernelLaunch::new(gpu, self.dequant_k)
                .grid([10, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(records_dev)
                .arg_ptr(scales_dev)
                .arg_ptr(embedding)
                .launch(stream)
        }

        /// One row's key (`[residual]`) and value (`[PLE_EMBED]`) projections.
        fn launch_projections(
            &self,
            embedding: DevicePtr,
            key: DevicePtr,
            value: DevicePtr,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                embedding,
                &self.key_proj,
                key,
                self.residual_width() as u32,
                PLE_EMBED as u32,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                embedding,
                &self.value_proj,
                value,
                PLE_EMBED as u32,
                PLE_EMBED as u32,
                stream,
            )
        }

        /// Conv/fuse update of one row's `hyper` streams and the slot's conv history.
        #[allow(clippy::too_many_arguments)]
        fn launch_fuse(
            &self,
            hyper: DevicePtr,
            key: DevicePtr,
            value: DevicePtr,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            let state_stride = self.residual_width() * PLE_CONV_HISTORY * 2;
            KernelLaunch::new(gpu, self.fuse_k)
                .grid([self.hc_count as u32, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(key)
                .arg_ptr(value)
                .arg_ptr(self.norm_key.weight)
                .arg_ptr(self.norm_query.weight)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(self.conv_weight.weight)
                .arg_ptr(self.conv_state_dev.offset(slot_idx * state_stride))
                .arg_u32(self.hidden_size as u32)
                .arg_f32(self.eps)
                .arg_u32(u32::from(reset_state))
                .launch(stream)
        }

        /// Multi-row speculative verify PLE join + intermediate state capture in one launch.
        #[allow(clippy::too_many_arguments)]
        fn launch_fuse_verify_rows(
            &self,
            hyper: DevicePtr,
            hyper_row_elems: u32,
            key_rows: DevicePtr,
            value_rows: DevicePtr,
            slot_idx: usize,
            rows: u32,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            let state_stride = self.residual_width() * PLE_CONV_HISTORY * 2;
            KernelLaunch::new(gpu, self.fuse_verify_rows_k)
                .grid([self.hc_count as u32, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(hyper)
                .arg_u32(hyper_row_elems)
                .arg_ptr(key_rows)
                .arg_ptr(value_rows)
                .arg_ptr(self.norm_key.weight)
                .arg_ptr(self.norm_query.weight)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(self.conv_weight.weight)
                .arg_ptr(self.conv_state_dev.offset(slot_idx * state_stride))
                .arg_ptr(
                    self.conv_intermediate_dev
                        .offset(slot_idx * PLE_MAX_SPEC_ROWS * state_stride),
                )
                .arg_u32(self.hidden_size as u32)
                .arg_f32(self.eps)
                .arg_u32(rows)
                .arg_u32(PLE_MAX_SPEC_ROWS as u32)
                .arg_u64(0)
                .launch(stream)
        }

        /// Multi-sequence, multi-row speculative verify PLE join + intermediate state capture in a single launch.
        #[allow(clippy::too_many_arguments)]
        fn launch_fuse_verify_rows_multi(
            &self,
            hyper: DevicePtr,
            hyper_row_elems: u32,
            key_rows: DevicePtr,
            value_rows: DevicePtr,
            num_seqs: u32,
            rows_per_seq: u32,
            packed_slots: u64,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            KernelLaunch::new(gpu, self.fuse_verify_rows_k)
                .grid([self.hc_count as u32, num_seqs, 1])
                .block([1024, 1, 1])
                .arg_ptr(hyper)
                .arg_u32(hyper_row_elems)
                .arg_ptr(key_rows)
                .arg_ptr(value_rows)
                .arg_ptr(self.norm_key.weight)
                .arg_ptr(self.norm_query.weight)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(self.conv_weight.weight)
                .arg_ptr(self.conv_state_dev)
                .arg_ptr(self.conv_intermediate_dev)
                .arg_u32(self.hidden_size as u32)
                .arg_f32(self.eps)
                .arg_u32(rows_per_seq)
                .arg_u32(PLE_MAX_SPEC_ROWS as u32)
                .arg_u64(packed_slots)
                .launch(stream)
        }

        /// Ragged multi-sequence, multi-row speculative verify PLE join + intermediate state capture in a single launch.
        #[allow(clippy::too_many_arguments)]
        fn launch_fuse_verify_rows_ragged(
            &self,
            hyper: DevicePtr,
            hyper_row_elems: u32,
            key_rows: DevicePtr,
            value_rows: DevicePtr,
            num_seqs: u32,
            seq_descs_dev: DevicePtr,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            KernelLaunch::new(gpu, self.fuse_verify_rows_ragged_k)
                .grid([self.hc_count as u32, num_seqs, 1])
                .block([1024, 1, 1])
                .arg_ptr(hyper)
                .arg_u32(hyper_row_elems)
                .arg_ptr(key_rows)
                .arg_ptr(value_rows)
                .arg_ptr(self.norm_key.weight)
                .arg_ptr(self.norm_query.weight)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(self.conv_weight.weight)
                .arg_ptr(self.conv_state_dev)
                .arg_ptr(self.conv_intermediate_dev)
                .arg_u32(self.hidden_size as u32)
                .arg_f32(self.eps)
                .arg_u32(PLE_MAX_SPEC_ROWS as u32)
                .arg_ptr(seq_descs_dev)
                .launch(stream)
        }

        /// Start reading a prefill chunk's PLE rows on a helper thread. The rows
        /// depend only on token ids, so the caller issues this before layer 0
        /// and the O_DIRECT reads overlap the GPU work that precedes the PLE
        /// join (embedding + layer 0). Pass the handle to [`Self::forward_prefill`].
        pub fn begin_prefill_read(
            &self,
            current_tokens: &[u32],
            prior_tokens: &[u32],
        ) -> PlePrefillRead {
            self.begin_prefill_read_with(current_tokens, prior_tokens, false)
        }

        /// [`Self::begin_prefill_read`] for a background read issued while
        /// another request runs: per-block reader-lock holds (see
        /// `spawn_block_read`). `None` without the block pipeline
        /// (`ATLAS_QWEN4_PLE_PREFILL_PIPELINE=1`), whose whole-chunk read cannot
        /// yield the lock.
        pub fn begin_prefill_read_yielding(
            &self,
            current_tokens: &[u32],
            prior_tokens: &[u32],
        ) -> Option<PlePrefillRead> {
            matches!(
                crate::layers::qwen4_ple_pipeline::pipeline_selected(),
                Ok(true)
            )
            .then(|| self.begin_prefill_read_with(current_tokens, prior_tokens, true))
        }

        fn begin_prefill_read_with(
            &self,
            current_tokens: &[u32],
            prior_tokens: &[u32],
            yielding: bool,
        ) -> PlePrefillRead {
            let hasher = Arc::clone(&self.hasher);
            let reader = Arc::clone(&self.reader);
            let io_timeout = self.io_timeout;
            let poisoned = Arc::clone(&self.io_poisoned);
            let current = current_tokens.to_vec();
            let prior = prior_tokens[prior_tokens.len().saturating_sub(2)..].to_vec();
            if poisoned.load(Ordering::Acquire) {
                return PlePrefillRead {
                    kind: Err(anyhow::anyhow!(
                        "Qwen4 PLE I/O is poisoned; process restart is required"
                    )),
                    io_timeout,
                    poisoned,
                };
            }
            let kind = match crate::layers::qwen4_ple_pipeline::pipeline_selected() {
                Ok(true) => spawn_block_read(
                    hasher,
                    reader,
                    current,
                    prior,
                    io_timeout,
                    Arc::clone(&poisoned),
                    yielding,
                ),
                Ok(false) => {
                    let thread_poisoned = Arc::clone(&poisoned);
                    std::thread::Builder::new()
                        .name("ple-prefill-read".into())
                        .spawn(move || {
                            let placement = read_affinity::begin();
                            let rows = read_prefill_rows(
                                &hasher,
                                &reader,
                                &current,
                                &prior,
                                true,
                                io_timeout,
                                &thread_poisoned,
                            );
                            read_affinity::end(placement, current.len() * QWEN4_PLE_HEADS);
                            rows
                        })
                        .map(PleReadKind::Whole)
                        .map_err(anyhow::Error::from)
                }
                Err(error) => Err(error),
            };
            PlePrefillRead {
                kind,
                io_timeout,
                poisoned,
            }
        }

        /// Fetch and inject a complete contiguous prefill chunk. Sparse row
        /// I/O remains host-driven, but every dense projection and PLE join is
        /// issued once for the whole chunk instead of once per token.
        /// `prefetched` must come from [`Self::begin_prefill_read`] for the same
        /// tokens; without it the rows are read here.
        #[allow(clippy::too_many_arguments)]
        pub fn forward_prefill(
            &self,
            current_tokens: &[u32],
            prior_tokens: &[u32],
            hyper: DevicePtr,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
            prefetched: Option<PlePrefillRead>,
        ) -> Result<()> {
            if crate::layers::qwen4_ple_pipeline::async_selected()? {
                return self.forward_prefill_async(
                    current_tokens,
                    prior_tokens,
                    hyper,
                    slot_idx,
                    reset_state,
                    gpu,
                    stream,
                    prefetched,
                );
            }
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(!current_tokens.is_empty(), "PLE prefill chunk is empty");
            let scratch = self.prefill.as_ref().context(
                "Qwen4 PLE whole-prompt path requires ATLAS_QWEN4_PLE_PREFILL_BATCH=1 at model load",
            )?;
            let num_tokens = current_tokens.len();
            ensure!(
                num_tokens <= scratch.max_tokens,
                "Qwen4 PLE prefill chunk {num_tokens} exceeds scratch bound {}",
                scratch.max_tokens
            );
            let num_tokens_u32 =
                u32::try_from(num_tokens).context("Qwen4 PLE prefill token count exceeds u32")?;

            // Every mutable staging allocation is shared between requests.
            // Keep row reads, H2D and GPU consumption in one event-chained
            // critical section just like the single-token path.
            let _submit = self.scratch_submit.lock();
            ensure!(
                !self.scratch_poisoned.load(Ordering::Acquire),
                "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
            );
            let total_rows = num_tokens * QWEN4_PLE_HEADS;
            let mut records = vec![0u8; total_rows * PLE_RECORD_BYTES];
            let mut scales = vec![0u8; total_rows * std::mem::size_of::<f32>()];
            match prefetched {
                Some(read) => {
                    let rows = read.join()?;
                    ensure!(
                        rows.len() == total_rows,
                        "Qwen4 PLE prefill read extent mismatch"
                    );
                    for (index, row) in rows.iter().enumerate() {
                        let record_start = index * PLE_RECORD_BYTES;
                        records[record_start..record_start + PLE_RECORD_BYTES]
                            .copy_from_slice(&row.record);
                        let scale_start = index * std::mem::size_of::<f32>();
                        scales[scale_start..scale_start + std::mem::size_of::<f32>()]
                            .copy_from_slice(&row.scale2.to_le_bytes());
                    }
                }
                None if prefill_stream_selected() => {
                    visit_prefill_rows(
                        &self.hasher,
                        &self.reader,
                        current_tokens,
                        prior_tokens,
                        self.io_timeout,
                        &self.io_poisoned,
                        &mut |index, row| {
                            let record_start = index * PLE_RECORD_BYTES;
                            records[record_start..record_start + PLE_RECORD_BYTES]
                                .copy_from_slice(&row.record);
                            let scale_start = index * std::mem::size_of::<f32>();
                            scales[scale_start..scale_start + std::mem::size_of::<f32>()]
                                .copy_from_slice(&row.scale2.to_le_bytes());
                            Ok(())
                        },
                    )?;
                }
                None => {
                    let rows = read_prefill_rows(
                        &self.hasher,
                        &self.reader,
                        current_tokens,
                        prior_tokens,
                        false,
                        self.io_timeout,
                        &self.io_poisoned,
                    )?;
                    ensure!(
                        rows.len() == total_rows,
                        "Qwen4 PLE prefill read extent mismatch"
                    );
                    for (index, row) in rows.iter().enumerate() {
                        let record_start = index * PLE_RECORD_BYTES;
                        records[record_start..record_start + PLE_RECORD_BYTES]
                            .copy_from_slice(&row.record);
                        let scale_start = index * std::mem::size_of::<f32>();
                        scales[scale_start..scale_start + std::mem::size_of::<f32>()]
                            .copy_from_slice(&row.scale2.to_le_bytes());
                    }
                }
            }

            let dequant_items = num_tokens_u32
                .checked_mul(PLE_EMBED as u32)
                .context("Qwen4 PLE dequant grid overflow")?;
            let state_stride = self.state_stride_bytes();
            let state = self.conv_state_dev.offset(slot_idx * state_stride);
            if let Err(error) = gpu.stream_wait_event(stream, self.scratch_event) {
                self.scratch_poisoned.store(true, Ordering::Release);
                return Err(error.context(
                    "Qwen4 PLE scratch predecessor wait failed; shared scratch poisoned",
                ));
            }
            let execution = (|| -> Result<()> {
                gpu.copy_h2d_group_on_stream(
                    &[
                        HostToDeviceCopy::new(&records, scratch.records_dev),
                        HostToDeviceCopy::new(&scales, scratch.scales_dev),
                    ],
                    stream,
                )?;
                KernelLaunch::new(gpu, scratch.dequant_k)
                    .grid([dequant_items.div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(scratch.records_dev)
                    .arg_ptr(scratch.scales_dev)
                    .arg_ptr(scratch.embedding_dev)
                    .arg_u32(num_tokens_u32)
                    .launch(stream)?;
                ops::cublas_bf16_proj_dense(
                    scratch.embedding_dev,
                    &self.key_proj,
                    scratch.key_dev,
                    num_tokens_u32,
                    self.residual_width() as u32,
                    PLE_EMBED as u32,
                    stream,
                )?;
                ops::cublas_bf16_proj_dense(
                    scratch.embedding_dev,
                    &self.value_proj,
                    scratch.value_dev,
                    num_tokens_u32,
                    PLE_EMBED as u32,
                    PLE_EMBED as u32,
                    stream,
                )?;
                if let Some(fused_join) = &scratch.fused {
                    self.launch_prefill_fused_join(
                        scratch,
                        fused_join,
                        hyper,
                        state,
                        0..num_tokens,
                        reset_state,
                        gpu,
                        stream,
                    )?;
                    self.launch_prefill_fused_commit(
                        scratch,
                        fused_join,
                        state,
                        num_tokens_u32,
                        reset_state,
                        gpu,
                        stream,
                    )?;
                    return gpu.synchronize(stream);
                }
                KernelLaunch::new(gpu, scratch.prepare_k)
                    .grid([num_tokens_u32, self.hc_count as u32, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(hyper)
                    .arg_ptr(scratch.key_dev)
                    .arg_ptr(scratch.value_dev)
                    .arg_ptr(self.norm_key.weight)
                    .arg_ptr(self.norm_query.weight)
                    .arg_ptr(self.norm_conv.weight)
                    .arg_ptr(scratch.gated_value_dev)
                    .arg_ptr(scratch.conv_input_dev)
                    .arg_u32(num_tokens_u32)
                    .arg_u32(self.hidden_size as u32)
                    .arg_f32(self.eps)
                    .launch(stream)?;
                KernelLaunch::new(gpu, scratch.conv_inject_k)
                    .grid([
                        num_tokens_u32,
                        self.hc_count as u32,
                        (self.hidden_size as u32).div_ceil(256),
                    ])
                    .block([256, 1, 1])
                    .arg_ptr(hyper)
                    .arg_ptr(scratch.gated_value_dev)
                    .arg_ptr(scratch.conv_input_dev)
                    .arg_ptr(self.conv_weight.weight)
                    .arg_ptr(state)
                    .arg_u32(num_tokens_u32)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(u32::from(reset_state))
                    .launch(stream)?;
                KernelLaunch::new(gpu, scratch.commit_state_k)
                    .grid([(self.residual_width() as u32).div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(scratch.conv_input_dev)
                    .arg_ptr(state)
                    .arg_u32(num_tokens_u32)
                    .arg_u32(self.residual_width() as u32)
                    .arg_u32(u32::from(reset_state))
                    .launch(stream)?;
                gpu.synchronize(stream)
            })();
            if let Err(error) = execution {
                let fence = gpu
                    .synchronize(stream)
                    .and_then(|()| gpu.record_event(self.scratch_event, stream));
                return match fence {
                    Ok(()) => Err(error.context(
                        "Qwen4 PLE whole-prompt execution failed after shared scratch was fenced",
                    )),
                    Err(fence_error) => {
                        self.scratch_poisoned.store(true, Ordering::Release);
                        Err(error.context(format!(
                            "Qwen4 PLE whole-prompt execution failed and fencing failed ({fence_error:#}); shared scratch poisoned"
                        )))
                    }
                };
            }
            if let Err(error) = gpu.record_event(self.scratch_event, stream) {
                self.scratch_poisoned.store(true, Ordering::Release);
                return Err(error.context(
                    "Qwen4 PLE post-success scratch fence failed; shared scratch poisoned",
                ));
            }
            if scratch.fused.is_some() {
                crate::layers::qwen4_ple_pipeline::log_once(
                    crate::layers::qwen4_ple_fused::ran_line("sync"),
                );
            }
            tracing::info!(
                target: "atlas::qwen4_prefill",
                family = "ple",
                selector = "ATLAS_QWEN4_PLE_PREFILL_BATCH",
                projection = "cublaslt_bf16_non_bit_exact",
                projection_parity_required = true,
                performance_claim_allowed = false,
                num_tokens,
                heads = QWEN4_PLE_HEADS,
                hidden_size = self.hidden_size,
                hc_count = self.hc_count,
                reset_state,
                "QWEN4_PREFILL_ENGAGED"
            );
            Ok(())
        }

        /// [`Self::forward_prefill`] with pinned async staging and no host
        /// syncs (`ATLAS_QWEN4_PLE_PREFILL_ASYNC`); a block read still in
        /// flight is consumed block by block (`ATLAS_QWEN4_PLE_PREFILL_PIPELINE`).
        /// Same kernels on the same rows: per block, the projections use the
        /// whole chunk's cuBLASLt algorithm and the conv reads the previous
        /// block through the same BF16 history the chunk-end commit writes.
        #[allow(clippy::too_many_arguments)]
        fn forward_prefill_async(
            &self,
            current_tokens: &[u32],
            prior_tokens: &[u32],
            hyper: DevicePtr,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
            prefetched: Option<PlePrefillRead>,
        ) -> Result<()> {
            use crate::layers::qwen4_ple_pipeline::{
                self as pipeline, ASYNC_SELECTOR, BLOCK_TOKENS, BlockReadSchedule,
                PIPELINE_SELECTOR, StagingLayout,
            };
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(!current_tokens.is_empty(), "PLE prefill chunk is empty");
            let scratch = self.prefill.as_ref().context(
                "Qwen4 PLE whole-prompt path requires ATLAS_QWEN4_PLE_PREFILL_BATCH=1 at model load",
            )?;
            let num_tokens = current_tokens.len();
            ensure!(
                num_tokens <= scratch.max_tokens,
                "Qwen4 PLE prefill chunk {num_tokens} exceeds scratch bound {}",
                scratch.max_tokens
            );
            let _submit = self.scratch_submit.lock();
            ensure!(
                !self.scratch_poisoned.load(Ordering::Acquire),
                "Qwen4 PLE shared scratch is poisoned after an unfenceable GPU failure"
            );
            let read = match prefetched {
                Some(read) => read,
                None => self.begin_prefill_read(current_tokens, prior_tokens),
            };
            let plan = PleRowsPlan::from_read(read, num_tokens)?;
            let level = pipeline::pipeline_level()?;
            let tuned = std::env::var("ATLAS_CUBLAS_TUNED").ok().as_deref() == Some("1");
            let plan = match plan {
                PleRowsPlan::Pipelined { read, ready } if tuned && level == 1 => {
                    pipeline::log_once(format!(
                        "QWEN4_PLE_PREFILL_PIPELINE_REFUSED selector={PIPELINE_SELECTOR} reason=cublas_tuned (whole-chunk projections)"
                    ));
                    PleRowsPlan::Whole(PleRowsPlan::finish_whole(read, ready)?)
                }
                plan => plan,
            };
            let layout = StagingLayout {
                record_bytes: PLE_RECORD_BYTES,
                max_rows: scratch.max_tokens * QWEN4_PLE_HEADS,
            };
            let history_bytes = self.residual_width() * PLE_CONV_HISTORY * 2;
            let mut staging_guard = pipeline::staging(gpu, layout, history_bytes)?;
            let staging = staging_guard
                .as_mut()
                .context("Qwen4 PLE prefill staging missing")?;
            let half = staging.acquire(gpu)?;
            if let Err(error) = gpu.stream_wait_event(stream, self.scratch_event) {
                self.scratch_poisoned.store(true, Ordering::Release);
                return Err(error.context(
                    "Qwen4 PLE scratch predecessor wait failed; shared scratch poisoned",
                ));
            }
            let state = self
                .conv_state_dev
                .offset(slot_idx * self.state_stride_bytes());
            let residual_width = u32::try_from(self.residual_width())?;
            let num_tokens_u32 = u32::try_from(num_tokens)?;
            let mut engaged: Option<(usize, usize, BlockReadSchedule)> = None;
            let execution = (|| -> Result<()> {
                let history = staging.history_dev;
                match plan {
                    PleRowsPlan::Whole(rows) => {
                        self.stage_prefill_rows(
                            scratch,
                            staging,
                            half,
                            &rows,
                            0..num_tokens,
                            gpu,
                            stream,
                        )?;
                        self.launch_prefill_dequant(scratch, 0..num_tokens, gpu, stream)?;
                        self.launch_prefill_project_inject(
                            scratch,
                            hyper,
                            state,
                            history,
                            0..num_tokens,
                            reset_state,
                            None,
                            gpu,
                            stream,
                        )?;
                    }
                    PleRowsPlan::Pipelined { read, ready } => {
                        let ranges = pipeline::block_ranges(num_tokens, BLOCK_TOKENS);
                        let ready_at_entry = ready.len();
                        let schedule =
                            pipeline::schedule_block_read(ready_at_entry, ranges.len(), level);
                        match schedule {
                            BlockReadSchedule::WholeChunkCoalesced => {
                                self.stage_prefill_packed_blocks(
                                    scratch, staging, half, &ready, &ranges, gpu, stream,
                                )?;
                                read.finish()?;
                                self.launch_prefill_dequant(scratch, 0..num_tokens, gpu, stream)?;
                                self.launch_prefill_project_inject(
                                    scratch,
                                    hyper,
                                    state,
                                    history,
                                    0..num_tokens,
                                    reset_state,
                                    None,
                                    gpu,
                                    stream,
                                )?;
                            }
                            BlockReadSchedule::Streaming {
                                ready_prefix,
                                per_block_projection,
                            } => {
                                if ready_prefix > 0 {
                                    let prefix_tokens = 0..ranges[ready_prefix - 1].end;
                                    self.stage_prefill_packed_blocks(
                                        scratch,
                                        staging,
                                        half,
                                        &ready[..ready_prefix],
                                        &ranges[..ready_prefix],
                                        gpu,
                                        stream,
                                    )?;
                                    self.launch_prefill_dequant(
                                        scratch,
                                        prefix_tokens.clone(),
                                        gpu,
                                        stream,
                                    )?;
                                    if per_block_projection {
                                        self.launch_prefill_project_inject(
                                            scratch,
                                            hyper,
                                            state,
                                            history,
                                            prefix_tokens,
                                            reset_state,
                                            Some(num_tokens_u32),
                                            gpu,
                                            stream,
                                        )?;
                                    }
                                }
                                let mut remaining_ready = ready.into_iter().skip(ready_prefix);
                                for (block, tokens) in ranges.iter().enumerate().skip(ready_prefix)
                                {
                                    let rows = match remaining_ready.next() {
                                        Some(rows) => rows,
                                        None => match read.take(block, true) {
                                            Ok(Some(rows)) => rows,
                                            Ok(None) => {
                                                unreachable!(
                                                    "a blocking take always yields a block"
                                                )
                                            }
                                            Err(short) => {
                                                read.finish()?;
                                                return Err(short);
                                            }
                                        },
                                    };
                                    self.stage_prefill_packed_block(
                                        scratch,
                                        staging,
                                        half,
                                        &rows,
                                        tokens.clone(),
                                        gpu,
                                        stream,
                                    )?;
                                    self.launch_prefill_dequant(
                                        scratch,
                                        tokens.clone(),
                                        gpu,
                                        stream,
                                    )?;
                                    if per_block_projection {
                                        self.launch_prefill_project_inject(
                                            scratch,
                                            hyper,
                                            state,
                                            history,
                                            tokens.clone(),
                                            reset_state,
                                            Some(num_tokens_u32),
                                            gpu,
                                            stream,
                                        )?;
                                    }
                                }
                                read.finish()?;
                                if !per_block_projection {
                                    self.launch_prefill_project_inject(
                                        scratch,
                                        hyper,
                                        state,
                                        history,
                                        0..num_tokens,
                                        reset_state,
                                        None,
                                        gpu,
                                        stream,
                                    )?;
                                }
                            }
                        }
                        engaged = Some((ranges.len(), ready_at_entry, schedule));
                    }
                }
                staging.release(gpu, half, stream)?;
                if let Some(fused_join) = &scratch.fused {
                    return self.launch_prefill_fused_commit(
                        scratch,
                        fused_join,
                        state,
                        num_tokens_u32,
                        reset_state,
                        gpu,
                        stream,
                    );
                }
                KernelLaunch::new(gpu, scratch.commit_state_k)
                    .grid([residual_width.div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(scratch.conv_input_dev)
                    .arg_ptr(state)
                    .arg_u32(num_tokens_u32)
                    .arg_u32(residual_width)
                    .arg_u32(u32::from(reset_state))
                    .launch(stream)
            })();
            if let Err(error) = execution {
                let fence = gpu
                    .synchronize(stream)
                    .and_then(|()| gpu.record_event(self.scratch_event, stream))
                    .and_then(|()| staging.release(gpu, half, stream));
                return match fence {
                    Ok(()) => Err(error
                        .context("Qwen4 PLE async prefill failed after shared scratch was fenced")),
                    Err(fence_error) => {
                        self.scratch_poisoned.store(true, Ordering::Release);
                        Err(error.context(format!(
                            "Qwen4 PLE async prefill failed and fencing failed ({fence_error:#}); shared scratch poisoned"
                        )))
                    }
                };
            }
            if let Err(error) = gpu.record_event(self.scratch_event, stream) {
                self.scratch_poisoned.store(true, Ordering::Release);
                return Err(error.context(
                    "Qwen4 PLE post-success scratch fence failed; shared scratch poisoned",
                ));
            }
            if scratch.fused.is_some() {
                pipeline::log_once(crate::layers::qwen4_ple_fused::ran_line(
                    if engaged.is_some() {
                        "async_blocks"
                    } else {
                        "async"
                    },
                ));
            }
            match engaged {
                Some((blocks, ready_at_entry, schedule)) => tracing::info!(
                    "QWEN4_PLE_PREFILL_PIPELINE_ENGAGED selector={PIPELINE_SELECTOR} level={level} rows={num_tokens} blocks={blocks} ready_at_entry={ready_at_entry} projections={}",
                    schedule.projection_label()
                ),
                None => pipeline::log_once(format!(
                    "QWEN4_PLE_PREFILL_ASYNC_ENGAGED selector={ASYNC_SELECTOR} staging=pinned_double_buffer host_syncs=0"
                )),
            }
            tracing::info!(
                target: "atlas::qwen4_prefill",
                family = "ple",
                selector = "ATLAS_QWEN4_PLE_PREFILL_BATCH",
                projection = "cublaslt_bf16_non_bit_exact",
                projection_parity_required = true,
                performance_claim_allowed = false,
                num_tokens,
                heads = QWEN4_PLE_HEADS,
                hidden_size = self.hidden_size,
                hc_count = self.hc_count,
                reset_state,
                "QWEN4_PREFILL_ENGAGED"
            );
            Ok(())
        }

        /// Copy a contiguous slice of pre-deinterleaved `PackedPleBlock`s into pinned
        /// staging half `half` and issue one coalesced H2D copy each for records and
        /// scales into the matching device row arenas.
        #[allow(clippy::too_many_arguments)]
        fn stage_prefill_packed_blocks(
            &self,
            scratch: &Qwen4PlePrefillScratch,
            staging: &mut crate::layers::qwen4_ple_pipeline::Staging,
            half: usize,
            blocks: &[Arc<PackedPleBlock>],
            ranges: &[std::ops::Range<usize>],
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                !blocks.is_empty() && blocks.len() == ranges.len(),
                "Qwen4 PLE coalesced staging block/range count mismatch ({} vs {})",
                blocks.len(),
                ranges.len()
            );
            let first_token = ranges[0].start;
            let last_token = ranges[ranges.len() - 1].end;
            ensure!(
                last_token <= scratch.max_tokens && first_token < last_token,
                "Qwen4 PLE coalesced staging token span {first_token}..{last_token} exceeds scratch bound {}",
                scratch.max_tokens
            );
            let first_sel = first_token
                .checked_mul(QWEN4_PLE_HEADS)
                .context("Qwen4 PLE staged selection start overflow")?;
            let last_sel = last_token
                .checked_mul(QWEN4_PLE_HEADS)
                .context("Qwen4 PLE staged selection end overflow")?;
            ensure!(
                last_sel <= staging.layout.max_rows
                    && staging.layout.record_bytes == PLE_RECORD_BYTES,
                "Qwen4 PLE staging layout ({} rows, {} B/record) cannot hold {} selections",
                staging.layout.max_rows,
                staging.layout.record_bytes,
                last_sel
            );
            let layout = staging.layout;
            let (all_records, all_scales) = layout.ranges(first_sel..last_sel);
            let scale_bytes = std::mem::size_of::<f32>();
            let buffer = staging.buffer(half);
            let mut expected_start = first_token;
            for (rows, tokens) in blocks.iter().zip(ranges.iter()) {
                ensure!(
                    tokens.start == expected_start && tokens.start < tokens.end,
                    "Qwen4 PLE coalesced staging ranges are not contiguous at {}..{} (expected start {expected_start})",
                    tokens.start,
                    tokens.end
                );
                expected_start = tokens.end;
                let selections = tokens.start * QWEN4_PLE_HEADS..tokens.end * QWEN4_PLE_HEADS;
                ensure!(
                    rows.num_rows == selections.len(),
                    "Qwen4 PLE staged block extent mismatch"
                );
                let (records, scales) = layout.ranges(selections);
                ensure!(
                    rows.records.len() == records.len() && rows.scales.len() == scales.len(),
                    "Qwen4 PLE staged block byte length mismatch"
                );
                buffer[records].copy_from_slice(&rows.records);
                buffer[scales].copy_from_slice(&rows.scales);
            }
            gpu.copy_h2d_async(
                &buffer[all_records],
                scratch.records_dev.offset(first_sel * PLE_RECORD_BYTES),
                stream,
            )?;
            gpu.copy_h2d_async(
                &buffer[all_scales],
                scratch.scales_dev.offset(first_sel * scale_bytes),
                stream,
            )
        }

        /// Copy pre-deinterleaved `PackedPleBlock` into pinned staging half `half`
        /// and copy them to the same offsets of the device row arenas.
        #[allow(clippy::too_many_arguments)]
        fn stage_prefill_packed_block(
            &self,
            scratch: &Qwen4PlePrefillScratch,
            staging: &mut crate::layers::qwen4_ple_pipeline::Staging,
            half: usize,
            block: &PackedPleBlock,
            tokens: std::ops::Range<usize>,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            let selections = tokens.start * QWEN4_PLE_HEADS..tokens.end * QWEN4_PLE_HEADS;
            ensure!(
                block.num_rows == selections.len(),
                "Qwen4 PLE staged block extent mismatch"
            );
            let (records, scales) = staging.layout.ranges(selections.clone());
            let scale_bytes = std::mem::size_of::<f32>();
            let buffer = staging.buffer(half);
            buffer[records.clone()].copy_from_slice(&block.records);
            buffer[scales.clone()].copy_from_slice(&block.scales);
            gpu.copy_h2d_async(
                &buffer[records],
                scratch
                    .records_dev
                    .offset(selections.start * PLE_RECORD_BYTES),
                stream,
            )?;
            gpu.copy_h2d_async(
                &buffer[scales],
                scratch.scales_dev.offset(selections.start * scale_bytes),
                stream,
            )
        }

        /// Pack rows of chunk tokens `tokens` into pinned staging half `half`
        /// and copy them to the same offsets of the device row arenas.
        #[allow(clippy::too_many_arguments)]
        fn stage_prefill_rows(
            &self,
            scratch: &Qwen4PlePrefillScratch,
            staging: &mut crate::layers::qwen4_ple_pipeline::Staging,
            half: usize,
            rows: &[PleNvfp4Row],
            tokens: std::ops::Range<usize>,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            let selections = tokens.start * QWEN4_PLE_HEADS..tokens.end * QWEN4_PLE_HEADS;
            ensure!(
                rows.len() == selections.len(),
                "Qwen4 PLE staged block extent mismatch"
            );
            let (records, scales) = staging.layout.ranges(selections.clone());
            let scale_bytes = std::mem::size_of::<f32>();
            let buffer = staging.buffer(half);
            for (index, row) in rows.iter().enumerate() {
                let record = records.start + index * PLE_RECORD_BYTES;
                buffer[record..record + PLE_RECORD_BYTES].copy_from_slice(&row.record);
                let scale = scales.start + index * scale_bytes;
                buffer[scale..scale + scale_bytes].copy_from_slice(&row.scale2.to_le_bytes());
            }
            gpu.copy_h2d_async(
                &buffer[records],
                scratch
                    .records_dev
                    .offset(selections.start * PLE_RECORD_BYTES),
                stream,
            )?;
            gpu.copy_h2d_async(
                &buffer[scales],
                scratch.scales_dev.offset(selections.start * scale_bytes),
                stream,
            )
        }

        /// Dequantize the staged rows of chunk tokens `tokens`.
        fn launch_prefill_dequant(
            &self,
            scratch: &Qwen4PlePrefillScratch,
            tokens: std::ops::Range<usize>,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            let start = tokens.start;
            let rows = u32::try_from(tokens.len())?;
            let dequant_items = rows
                .checked_mul(PLE_EMBED as u32)
                .context("Qwen4 PLE dequant grid overflow")?;
            KernelLaunch::new(gpu, scratch.dequant_k)
                .grid([dequant_items.div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(
                    scratch
                        .records_dev
                        .offset(start * QWEN4_PLE_HEADS * PLE_RECORD_BYTES),
                )
                .arg_ptr(scratch.scales_dev.offset(start * QWEN4_PLE_HEADS * 4))
                .arg_ptr(scratch.embedding_dev.offset(start * PLE_EMBED * 2))
                .arg_u32(rows)
                .launch(stream)
        }

        /// Project, prepare and conv-inject chunk tokens `tokens` (the
        /// chunk-end state commit is the caller's). `ref_m`: project with the
        /// algorithm of a `ref_m`-row GEMM (a block of that chunk); `None`:
        /// the whole-chunk call the synchronous path makes.
        #[allow(clippy::too_many_arguments)]
        fn launch_prefill_project_inject(
            &self,
            scratch: &Qwen4PlePrefillScratch,
            hyper: DevicePtr,
            state: DevicePtr,
            history: DevicePtr,
            tokens: std::ops::Range<usize>,
            reset_state: bool,
            ref_m: Option<u32>,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            let start = tokens.start;
            let rows = u32::try_from(tokens.len())?;
            let width = self.residual_width();
            let width_u32 = u32::try_from(width)?;
            let embedding = scratch.embedding_dev.offset(start * PLE_EMBED * 2);
            let key = scratch.key_dev.offset(start * width * 2);
            let value = scratch.value_dev.offset(start * PLE_EMBED * 2);
            let gated_value = scratch.gated_value_dev.offset(start * width * 4);
            let conv_input = scratch.conv_input_dev.offset(start * width * 4);
            let hyper_rows = hyper.offset(start * width * 2);
            let project = |weight: &DenseWeight, out: DevicePtr, n: u32| -> Result<()> {
                match ref_m {
                    None => ops::cublas_bf16_proj_dense(
                        embedding,
                        weight,
                        out,
                        rows,
                        n,
                        PLE_EMBED as u32,
                        stream,
                    ),
                    Some(ref_m) => spark_runtime::cublaslt::bf16_gemm_act_weight_t_ref_m(
                        embedding.0,
                        weight.weight.0,
                        out.0,
                        rows,
                        n,
                        PLE_EMBED as u32,
                        ref_m,
                        stream,
                    ),
                }
            };
            project(&self.key_proj, key, width_u32)?;
            project(&self.value_proj, value, PLE_EMBED as u32)?;
            if let Some(fused_join) = &scratch.fused {
                // Chunk-relative taps: the block history commit below is not needed.
                return self.launch_prefill_fused_join(
                    scratch,
                    fused_join,
                    hyper,
                    state,
                    tokens,
                    reset_state,
                    gpu,
                    stream,
                );
            }
            KernelLaunch::new(gpu, scratch.prepare_k)
                .grid([rows, self.hc_count as u32, 1])
                .block([1024, 1, 1])
                .arg_ptr(hyper_rows)
                .arg_ptr(key)
                .arg_ptr(value)
                .arg_ptr(self.norm_key.weight)
                .arg_ptr(self.norm_query.weight)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(gated_value)
                .arg_ptr(conv_input)
                .arg_u32(rows)
                .arg_u32(self.hidden_size as u32)
                .arg_f32(self.eps)
                .launch(stream)?;
            // Tokens before this block come from the chunk's own prepared rows:
            // the commit kernel writes their BF16 history exactly as the
            // chunk-end commit would, into the block history scratch.
            let (entry, entry_reset) = if start == 0 {
                (state, u32::from(reset_state))
            } else {
                KernelLaunch::new(gpu, scratch.commit_state_k)
                    .grid([width_u32.div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(scratch.conv_input_dev)
                    .arg_ptr(history)
                    .arg_u32(u32::try_from(start)?)
                    .arg_u32(width_u32)
                    .arg_u32(0)
                    .launch(stream)?;
                (history, 0)
            };
            KernelLaunch::new(gpu, scratch.conv_inject_k)
                .grid([
                    rows,
                    self.hc_count as u32,
                    (self.hidden_size as u32).div_ceil(256),
                ])
                .block([256, 1, 1])
                .arg_ptr(hyper_rows)
                .arg_ptr(gated_value)
                .arg_ptr(conv_input)
                .arg_ptr(self.conv_weight.weight)
                .arg_ptr(entry)
                .arg_u32(rows)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(entry_reset)
                .launch(stream)
        }

        /// Fused join of chunk tokens `tokens` (`ATLAS_QWEN4_PLE_PREFILL_FUSED`):
        /// the gate scalars, then conv + injection whose taps come from the
        /// chunk's own rows (chunk-relative token >= 0) or the slot's entry
        /// `state`. Bit-identical to prepare + conv_inject over the same
        /// tokens (and to the block pipeline's history-commit variant); the
        /// chunk-end commit is the caller's.
        #[allow(clippy::too_many_arguments)]
        fn launch_prefill_fused_join(
            &self,
            scratch: &Qwen4PlePrefillScratch,
            fused_join: &Qwen4PleFusedJoin,
            hyper: DevicePtr,
            state: DevicePtr,
            tokens: std::ops::Range<usize>,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            use crate::layers::qwen4_ple_fused as fused;
            let start = u32::try_from(tokens.start)?;
            let rows = u32::try_from(tokens.len())?;
            let hidden = u32::try_from(self.hidden_size)?;
            let residual = u32::try_from(self.residual_width())?;
            KernelLaunch::new(gpu, fused_join.gate_k)
                .grid(fused::gate_grid(rows).map_err(anyhow::Error::msg)?)
                .block([fused::GATE_THREADS, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(scratch.key_dev)
                .arg_ptr(scratch.value_dev)
                .arg_ptr(self.norm_key.weight)
                .arg_ptr(self.norm_query.weight)
                .arg_ptr(fused_join.gate_dev)
                .arg_u32(start)
                .arg_u32(rows)
                .arg_u32(hidden)
                .arg_f32(self.eps)
                .launch(stream)?;
            KernelLaunch::new(gpu, fused_join.inject_k)
                .grid(fused::inject_grid(rows, residual).map_err(anyhow::Error::msg)?)
                .block([fused::INJECT_THREADS, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(scratch.value_dev)
                .arg_ptr(fused_join.gate_dev)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(self.conv_weight.weight)
                .arg_ptr(state)
                .arg_u32(start)
                .arg_u32(rows)
                .arg_u32(hidden)
                .arg_u32(u32::from(reset_state))
                .launch(stream)
        }

        /// Chunk-end conv-history commit of the fused join (after every
        /// [`Self::launch_prefill_fused_join`] of the chunk).
        #[allow(clippy::too_many_arguments)]
        fn launch_prefill_fused_commit(
            &self,
            scratch: &Qwen4PlePrefillScratch,
            fused_join: &Qwen4PleFusedJoin,
            state: DevicePtr,
            num_tokens: u32,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            use crate::layers::qwen4_ple_fused as fused;
            KernelLaunch::new(gpu, fused_join.commit_k)
                .grid(fused::commit_grid(u32::try_from(self.residual_width())?))
                .block([fused::INJECT_THREADS, 1, 1])
                .arg_ptr(scratch.value_dev)
                .arg_ptr(fused_join.gate_dev)
                .arg_ptr(self.norm_conv.weight)
                .arg_ptr(state)
                .arg_u32(num_tokens)
                .arg_u32(u32::try_from(self.hidden_size)?)
                .arg_u32(u32::from(reset_state))
                .launch(stream)
        }

        pub fn residual_width(&self) -> usize {
            self.hidden_size * self.hc_count
        }

        pub fn destroy_owned_resources(&self, gpu: &dyn GpuBackend) -> Result<()> {
            let _submit = self.scratch_submit.lock();
            ensure!(
                !self.scratch_poisoned.load(Ordering::Acquire),
                "Qwen4 PLE scratch is poisoned; refusing unsafe device-buffer teardown"
            );
            let teardown_stream = gpu.default_stream();
            gpu.stream_wait_event(teardown_stream, self.scratch_event)
                .context("wait for final Qwen4 PLE scratch owner before teardown")?;
            gpu.synchronize(teardown_stream)
                .context("synchronize final Qwen4 PLE scratch owner before teardown")?;
            let mut errors = Vec::new();
            if let Some(scratch) = &self.prefill
                && let Err(error) = scratch.release(gpu)
            {
                errors.push(format!("prefill scratch: {error:#}"));
            }
            if let Some(scratch) = self.verify.lock().take()
                && let Err(error) = scratch.release(gpu)
            {
                errors.push(format!("verify batch scratch: {error:#}"));
            }
            self.warm_worker.lock().take();
            if let Err(error) = gpu.destroy_event(self.scratch_event) {
                errors.push(format!("scratch event: {error:#}"));
            }
            ensure!(
                errors.is_empty(),
                "Qwen4 PLE owned-resource release failures: {}",
                errors.join("; ")
            );
            Ok(())
        }

        fn state_stride_bytes(&self) -> usize {
            self.residual_width() * PLE_CONV_HISTORY * 2
        }

        pub(crate) fn conv_state_stride_bytes(&self) -> usize {
            self.state_stride_bytes()
        }

        pub(crate) fn conv_state_base_ptr(&self) -> DevicePtr {
            self.conv_state_dev
        }

        pub(crate) fn conv_checkpoint_base_ptr(&self) -> DevicePtr {
            self.conv_checkpoint_dev
        }

        pub(crate) fn max_batch_slots(&self) -> usize {
            self.max_batch_size
        }

        /// Diagnostic-only exact snapshot of the slot-local live/checkpoint
        /// histories. The caller supplies the producer stream whose writes
        /// must complete before either pageable-host read.
        pub(crate) fn parity_snapshot(
            &self,
            slot_idx: usize,
            gpu: &dyn GpuBackend,
            producer_stream: u64,
        ) -> Result<(usize, Vec<u8>, Vec<u8>)> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE parity slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            ensure!(stride > 0, "PLE parity state stride is zero");
            gpu.synchronize(producer_stream)?;
            let mut live = vec![0u8; stride];
            let mut checkpoint = vec![0u8; stride];
            gpu.copy_d2h(self.conv_state_dev.offset(slot_idx * stride), &mut live)?;
            gpu.copy_d2h(
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                &mut checkpoint,
            )?;
            Ok((stride, live, checkpoint))
        }

        /// Capture the exact post-PLE boundary for serial-vs-whole-prompt
        /// qualification. This is inert unless the complete parity-specific
        /// environment is present. A receipt is the final commit marker; no
        /// timing or throughput field is produced here.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn capture_post_prefill_parity(
            &self,
            hidden: DevicePtr,
            request_tokens: &[u32],
            prior_tokens: &[u32],
            current_tokens: &[u32],
            chunk_start: usize,
            slot_idx: usize,
            reset_state: bool,
            gpu: &dyn GpuBackend,
            producer_stream: u64,
        ) -> Result<()> {
            let Some(config) = parity_capture_config()? else {
                return Ok(());
            };
            ensure!(
                cfg!(target_endian = "little"),
                "PLE parity BF16LE capture requires a little-endian host"
            );
            ensure!(
                self.residual_width() == PLE_PARITY_WIDTH,
                "PLE parity residual width {} != {PLE_PARITY_WIDTH}",
                self.residual_width()
            );
            ensure!(!current_tokens.is_empty(), "PLE parity chunk is empty");
            let chunk_end = chunk_start
                .checked_add(current_tokens.len())
                .context("PLE parity chunk range overflow")?;
            ensure!(
                chunk_end <= request_tokens.len()
                    && request_tokens[chunk_start..chunk_end] == *current_tokens,
                "PLE parity chunk is not the exact ordered request-token slice"
            );
            ensure!(
                reset_state == (chunk_start == 0),
                "PLE parity reset/continuation marker disagrees with chunk start"
            );
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE parity slot {slot_idx} out of range"
            );
            let hidden_bytes_len = current_tokens
                .len()
                .checked_mul(PLE_PARITY_WIDTH)
                .and_then(|elements| elements.checked_mul(2))
                .context("PLE parity hidden extent overflow")?;
            let state_bytes_len = self.state_stride_bytes();
            ensure!(
                state_bytes_len == PLE_PARITY_WIDTH * PLE_CONV_HISTORY * 2,
                "PLE parity state extent {state_bytes_len} is not exact [10240,9] BF16"
            );
            let state_offset = slot_idx
                .checked_mul(state_bytes_len)
                .context("PLE parity slot state offset overflow")?;

            gpu.synchronize(producer_stream)
                .context("synchronize actual post-PLE producer stream")?;
            let mut hidden_bytes = vec![0u8; hidden_bytes_len];
            let mut live_bytes = vec![0u8; state_bytes_len];
            let mut checkpoint_bytes = vec![0u8; state_bytes_len];
            gpu.copy_d2h(hidden, &mut hidden_bytes)
                .context("capture exact post-PLE hidden BF16 bytes")?;
            gpu.copy_d2h(self.conv_state_dev.offset(state_offset), &mut live_bytes)
                .context("capture exact post-PLE live state BF16 bytes")?;
            gpu.copy_d2h(
                self.conv_checkpoint_dev.offset(state_offset),
                &mut checkpoint_bytes,
            )
            .context("capture exact post-PLE checkpoint state BF16 bytes")?;

            root_is_stable(&config)?;
            let arm = if config.selector == 0 {
                "serial0"
            } else {
                "whole_prompt1"
            };
            let phase = if reset_state { "reset" } else { "continuation" };
            let frame_name = format!(
                "frame-{}-{arm}-m{}-s{chunk_start}-n{}-{phase}",
                config.nonce,
                request_tokens.len(),
                current_tokens.len()
            );
            let frame = config.root.join(&frame_name);
            let mut owned_frame = create_owned_frame(frame.clone(), config.root_dev)?;
            let staging_frame = owned_frame.path().to_path_buf();
            let frame_identity = owned_frame.identity();
            let publication = (|| -> Result<()> {
                owned_frame.artifacts.push(write_sealed_artifact(
                    &staging_frame,
                    "post_ple_hidden.bf16le",
                    &hidden_bytes,
                )?);
                owned_frame.artifacts.push(write_sealed_artifact(
                    &staging_frame,
                    "ple_live.bf16le",
                    &live_bytes,
                )?);
                owned_frame.artifacts.push(write_sealed_artifact(
                    &staging_frame,
                    "ple_checkpoint.bf16le",
                    &checkpoint_bytes,
                )?);
                for artifact in &owned_frame.artifacts {
                    verify_sealed_artifact(&staging_frame, artifact)?;
                }
                let hidden_artifact = &owned_frame.artifacts[0];
                let live_artifact = &owned_frame.artifacts[1];
                let checkpoint_artifact = &owned_frame.artifacts[2];
                let root_receipt = config
                    .root
                    .to_str()
                    .context("canonical PLE parity root is not UTF-8")?;
                let hidden_receipt = serde_json::json!({
                    "file": hidden_artifact.name,
                    "dtype": "bf16le",
                    "shape": [current_tokens.len(), PLE_PARITY_WIDTH],
                    "bytes": hidden_artifact.bytes,
                    "sha256": hidden_artifact.sha256.as_str(),
                    "dev": hidden_artifact.dev,
                    "ino": hidden_artifact.ino,
                    "mode": "0444"
                });
                let live_receipt = serde_json::json!({
                    "file": live_artifact.name,
                    "dtype": "bf16le",
                    "shape": [PLE_PARITY_WIDTH, PLE_CONV_HISTORY],
                    "bytes": live_artifact.bytes,
                    "sha256": live_artifact.sha256.as_str(),
                    "dev": live_artifact.dev,
                    "ino": live_artifact.ino,
                    "mode": "0444"
                });
                let checkpoint_receipt = serde_json::json!({
                    "file": checkpoint_artifact.name,
                    "dtype": "bf16le",
                    "shape": [PLE_PARITY_WIDTH, PLE_CONV_HISTORY],
                    "bytes": checkpoint_artifact.bytes,
                    "sha256": checkpoint_artifact.sha256.as_str(),
                    "dev": checkpoint_artifact.dev,
                    "ino": checkpoint_artifact.ino,
                    "mode": "0444"
                });
                let artifact_receipts = serde_json::json!({
                    "post_ple_hidden": hidden_receipt,
                    "ple_live": live_receipt,
                    "ple_checkpoint": checkpoint_receipt
                });
                let receipt = serde_json::json!({
                    "schema": "atlas-qwen38-flash-next-post-ple-parity-frame-v1",
                    "boundary": "post_ple_pre_layer1",
                    "performance_claim_allowed": false,
                    "producer_stream_synchronized": true,
                    "producer_stream": producer_stream,
                    "pid": std::process::id(),
                    "nonce": config.nonce.as_str(),
                    "capture_root": root_receipt,
                    "capture_root_dev": config.root_dev,
                    "capture_root_ino": config.root_ino,
                    "frame": frame_name.as_str(),
                    "frame_dev": frame_identity.0,
                    "frame_ino": frame_identity.1,
                    "frame_commit_mode": "0500",
                    "arm": arm,
                    "selector": config.selector,
                    "request_m": request_tokens.len(),
                    "request_tokens_encoding": "u32le",
                    "request_tokens_sha256": token_sha256(request_tokens)?,
                    "ple_prior_m": prior_tokens.len(),
                    "ple_prior_tokens_sha256": token_sha256(prior_tokens)?,
                    "ple_ordered_m": prior_tokens.len().checked_add(current_tokens.len())
                        .context("PLE parity ordered token count overflow")?,
                    "ple_ordered_tokens_sha256": token_parts_sha256(&[prior_tokens, current_tokens])?,
                    "chunk_start": chunk_start,
                    "chunk_m": current_tokens.len(),
                    "chunk_tokens_encoding": "u32le",
                    "chunk_tokens_sha256": token_sha256(current_tokens)?,
                    "slot_idx": slot_idx,
                    "reset_state": reset_state,
                    "continuation": !reset_state,
                    "artifacts": artifact_receipts
                });
                let mut receipt_bytes = serde_json::to_vec(&receipt)
                    .context("serialize canonical PLE parity receipt")?;
                receipt_bytes.push(b'\n');
                owned_frame.artifacts.push(write_sealed_artifact(
                    &staging_frame,
                    "receipt.json",
                    &receipt_bytes,
                )?);
                for artifact in &owned_frame.artifacts {
                    verify_sealed_artifact(&staging_frame, artifact)?;
                }
                owned_frame
                    .handle
                    .as_ref()
                    .expect("created frame handle is retained")
                    .sync_all()
                    .context("sync private PLE parity staging frame")?;
                File::open(&config.root)
                    .context("open PLE parity root for durability")?
                    .sync_all()
                    .context("sync PLE parity root")?;
                root_is_stable(&config)?;
                let final_frame = fs::symlink_metadata(&staging_frame)
                    .context("re-stat private PLE parity staging frame before publish")?;
                ensure!(
                    final_frame.is_dir()
                        && final_frame.dev() == frame_identity.0
                        && final_frame.ino() == frame_identity.1
                        && final_frame.permissions().mode() & 0o777 == 0o700,
                    "PLE parity staging frame identity or pre-publish mode changed"
                );
                for artifact in &owned_frame.artifacts {
                    verify_sealed_artifact(&staging_frame, artifact)?;
                }
                // Seal the descriptor before the sole no-replace publication.
                // Any later failure remains rollback-safe because the owned
                // path is switched to the published name immediately after
                // the atomic rename.
                owned_frame
                    .handle
                    .as_ref()
                    .expect("created frame handle is retained")
                    .set_permissions(fs::Permissions::from_mode(0o500))
                    .context("seal PLE parity staging frame mode0500")?;
                owned_frame.publish_noreplace().context(
                    "atomically publish PLE parity frame without replacing a canonical path",
                )?;
                File::open(&config.root)
                    .context("open PLE parity root after publication")?
                    .sync_all()
                    .context("sync published PLE parity directory entry")?;
                root_is_stable(&config)?;
                let published =
                    fs::symlink_metadata(&frame).context("re-stat published PLE parity frame")?;
                ensure!(
                    published.is_dir()
                        && published.dev() == frame_identity.0
                        && published.ino() == frame_identity.1
                        && published.permissions().mode() & 0o777 == 0o500,
                    "published PLE parity frame identity or mode changed"
                );
                for artifact in &owned_frame.artifacts {
                    verify_sealed_artifact(&frame, artifact)?;
                }
                // No fallible operation follows this success assignment.
                owned_frame.committed = true;
                Ok(())
            })();
            if let Err(error) = publication {
                return Err(error).context("publish exact post-PLE parity frame");
            }
            tracing::info!(
                target: "atlas::qwen4_ple_parity",
                frame = %frame.display(),
                arm,
                request_m = request_tokens.len(),
                chunk_start,
                chunk_m = current_tokens.len(),
                reset_state,
                performance_claim_allowed = false,
                "QWEN4_PLE_PARITY_CAPTURED"
            );
            Ok(())
        }

        /// Restore only the live PLE history captured after a wide verify.
        /// The normal commit still selects and mirrors the accepted
        /// intermediate; this merely reconstructs the verifier's pre-commit
        /// live-state identity after the serial diagnostic replay.
        pub(crate) fn parity_restore_live(
            &self,
            slot_idx: usize,
            stride: usize,
            live: &[u8],
            gpu: &dyn GpuBackend,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE parity restore slot {slot_idx} out of range"
            );
            ensure!(
                stride == self.state_stride_bytes() && live.len() == stride,
                "PLE parity restore stride/extent mismatch"
            );
            gpu.copy_h2d(live, self.conv_state_dev.offset(slot_idx * stride))
        }

        /// Diagnostic-only snapshot of one verify intermediate consumed by
        /// `rollback_and_checkpoint`.  A serial oracle may reuse PLE scratch
        /// between verify and commit, so commit-parity diagnostics preserve
        /// this production-owned source explicitly.
        pub(crate) fn parity_intermediate_snapshot(
            &self,
            slot_idx: usize,
            row: usize,
            gpu: &dyn GpuBackend,
            producer_stream: u64,
        ) -> Result<(usize, Vec<u8>)> {
            ensure!(
                slot_idx < self.max_batch_size && row < PLE_MAX_SPEC_ROWS,
                "PLE parity intermediate slot/row out of range"
            );
            let stride = self.state_stride_bytes();
            ensure!(stride > 0, "PLE parity intermediate stride is zero");
            gpu.synchronize(producer_stream)?;
            let mut bytes = vec![0u8; stride];
            gpu.copy_d2h(
                self.conv_intermediate_dev
                    .offset((slot_idx * PLE_MAX_SPEC_ROWS + row) * stride),
                &mut bytes,
            )?;
            Ok((stride, bytes))
        }

        /// Restore the exact verify intermediate captured above.  This does
        /// not touch live/checkpoint state; the ordinary production commit
        /// remains the sole owner of those destinations.
        pub(crate) fn parity_restore_intermediate(
            &self,
            slot_idx: usize,
            row: usize,
            stride: usize,
            bytes: &[u8],
            gpu: &dyn GpuBackend,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size && row < PLE_MAX_SPEC_ROWS,
                "PLE parity intermediate restore slot/row out of range"
            );
            ensure!(
                stride == self.state_stride_bytes() && bytes.len() == stride,
                "PLE parity intermediate restore stride/extent mismatch"
            );
            gpu.copy_h2d(
                bytes,
                self.conv_intermediate_dev
                    .offset((slot_idx * PLE_MAX_SPEC_ROWS + row) * stride),
            )
        }

        /// Clear every request-local PLE history buffer for a reused slot.
        /// PLE state is allocated outside `SsmStatePool`, so the pool's slot
        /// reset cannot cover it.
        pub fn zero_slot(&self, slot_idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.memset_async(
                self.conv_state_dev.offset(slot_idx * stride),
                0,
                stride,
                stream,
            )?;
            gpu.memset_async(
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                0,
                stride,
                stream,
            )?;
            gpu.memset_async(
                self.conv_intermediate_dev
                    .offset(slot_idx * PLE_MAX_SPEC_ROWS * stride),
                0,
                PLE_MAX_SPEC_ROWS * stride,
                stream,
            )
        }

        /// Move every request-owned PLE history buffer during scheduler slot
        /// compaction.  PLE storage is allocated outside `SsmStatePool`, so a
        /// pool-only move leaves the live sequence reading the destination
        /// slot's previous history after `swap_remove` churn.
        pub fn copy_slot(
            &self,
            old_slot: usize,
            new_slot: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                old_slot < self.max_batch_size && new_slot < self.max_batch_size,
                "PLE slot move {old_slot}->{new_slot} is out of range"
            );
            if old_slot == new_slot {
                return Ok(());
            }
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_state_dev.offset(old_slot * stride),
                self.conv_state_dev.offset(new_slot * stride),
                stride,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_checkpoint_dev.offset(old_slot * stride),
                self.conv_checkpoint_dev.offset(new_slot * stride),
                stride,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_intermediate_dev
                    .offset(old_slot * PLE_MAX_SPEC_ROWS * stride),
                self.conv_intermediate_dev
                    .offset(new_slot * PLE_MAX_SPEC_ROWS * stride),
                PLE_MAX_SPEC_ROWS * stride,
                stream,
            )
        }

        /// Slots the request-owned history buffers hold (`max_batch_size`).
        pub(crate) fn seqstate_slot_count(&self) -> usize {
            self.max_batch_size
        }

        /// `SeqStateBlob` ranges of a slot's request-owned history: the live
        /// state and the speculative checkpoint (the intermediates are
        /// per-verify scratch, written before they are read).
        pub(crate) fn seqstate_slot_regions(
            &self,
            slot_idx: usize,
        ) -> Result<[(DevicePtr, usize); 2]> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            Ok([
                (self.conv_state_dev.offset(slot_idx * stride), stride),
                (self.conv_checkpoint_dev.offset(slot_idx * stride), stride),
            ])
        }

        /// Byte stride per sequence slot in `conv_state_dev`.
        pub fn state_stride(&self) -> usize {
            self.state_stride_bytes()
        }

        /// Copy canonical PLE `conv_state` for `slot_idx` to host memory for sequence swap-out.
        pub fn save_slot_host(&self, slot_idx: usize, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE swap save slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            let mut buf = vec![0u8; stride];
            gpu.copy_d2h(self.conv_state_dev.offset(slot_idx * stride), &mut buf)?;
            Ok(buf)
        }

        /// Restore canonical PLE `conv_state` (and mirror `conv_checkpoint`) for `slot_idx` on swap-in.
        pub fn restore_slot_host(
            &self,
            slot_idx: usize,
            data: &[u8],
            gpu: &dyn GpuBackend,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE swap restore slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            ensure!(
                data.len() == stride,
                "PLE swap restore size mismatch: got {}, expected {stride}",
                data.len()
            );
            gpu.copy_h2d(data, self.conv_state_dev.offset(slot_idx * stride))?;
            gpu.copy_h2d(data, self.conv_checkpoint_dev.offset(slot_idx * stride))
        }

        /// Save the canonical PLE convolution history at a speculative boundary.
        pub fn checkpoint(&self, slot_idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_state_dev.offset(slot_idx * stride),
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        /// Preserve the state after one verify row for partial acceptance.
        pub fn save_intermediate(
            &self,
            slot_idx: usize,
            row: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(
                row < PLE_MAX_SPEC_ROWS,
                "PLE speculative row {row} exceeds {PLE_MAX_SPEC_ROWS}"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_state_dev.offset(slot_idx * stride),
                self.conv_intermediate_dev
                    .offset((slot_idx * PLE_MAX_SPEC_ROWS + row) * stride),
                stride,
                stream,
            )
        }

        /// Restore the pre-verify state before a verify replay.
        pub fn restore_checkpoint(
            &self,
            slot_idx: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            let stride = self.state_stride_bytes();
            gpu.copy_d2d_async(
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                self.conv_state_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        /// Commit a partial rollback and make it the next checkpoint.
        pub fn rollback_and_checkpoint(
            &self,
            slot_idx: usize,
            num_accepted: usize,
            gpu: &dyn GpuBackend,
            stream: u64,
        ) -> Result<()> {
            ensure!(
                slot_idx < self.max_batch_size,
                "PLE slot {slot_idx} out of range"
            );
            ensure!(
                num_accepted <= PLE_MAX_SPEC_ROWS,
                "Qwen4 PLE rollback accepts at most {PLE_MAX_SPEC_ROWS} rows, got {num_accepted}"
            );
            let stride = self.state_stride_bytes();
            let src = if num_accepted == 0 {
                self.conv_checkpoint_dev.offset(slot_idx * stride)
            } else {
                self.conv_intermediate_dev
                    .offset((slot_idx * PLE_MAX_SPEC_ROWS + num_accepted - 1) * stride)
            };
            let live = self.conv_state_dev.offset(slot_idx * stride);
            gpu.copy_d2d_async(src, live, stride, stream)?;
            gpu.copy_d2d_async(
                live,
                self.conv_checkpoint_dev.offset(slot_idx * stride),
                stride,
                stream,
            )
        }

        pub fn owned_device_buffers(&self) -> [DevicePtr; 2] {
            [self.records_dev, self.scales_dev]
        }

        pub fn scratch_device_buffers(&self) -> [DevicePtr; 6] {
            [
                self.embedding_dev,
                self.key_dev,
                self.value_dev,
                self.conv_state_dev,
                self.conv_checkpoint_dev,
                self.conv_intermediate_dev,
            ]
        }

        pub fn prefill_device_buffers(&self) -> Option<Vec<DevicePtr>> {
            self.prefill
                .as_ref()
                .map(Qwen4PlePrefillScratch::device_buffers)
        }
    }

    fn check_tensor(
        store: &WeightStore,
        name: &str,
        shape: &[usize],
        dtype: WeightDtype,
    ) -> Result<()> {
        let tensor = store.get(name)?;
        ensure!(
            tensor.shape == shape,
            "{name} shape {:?} != {shape:?}",
            tensor.shape
        );
        ensure!(
            tensor.dtype == dtype,
            "{name} dtype {:?} != {dtype:?}",
            tensor.dtype
        );
        Ok(())
    }

    fn read_i64<const N: usize>(
        store: &WeightStore,
        name: &str,
        gpu: &dyn GpuBackend,
    ) -> Result<[i64; N]> {
        check_tensor(store, name, &[N], WeightDtype::Int64)?;
        let mut bytes = vec![0u8; N * 8];
        gpu.copy_d2h(store.get(name)?.ptr, &mut bytes)?;
        Ok(std::array::from_fn(|i| {
            i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("fixed i64 slice"))
        }))
    }

    #[cfg(test)]
    mod parity_capture_tests {
        use super::*;

        fn private_test_root(label: &str) -> PathBuf {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "atlas-qwen4-ple-{label}-{}-{stamp}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
            root
        }

        #[test]
        fn dependency_free_sha256_matches_standard_vectors() {
            assert_eq!(
                sha256_hex(b"").unwrap(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            );
            assert_eq!(
                sha256_hex(b"abc").unwrap(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );
            assert_eq!(
                sha256_hex(&[b'a'; 64]).unwrap(),
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
            );
            assert_eq!(
                token_sha256(&[1, 2]).unwrap(),
                "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f"
            );
            assert_eq!(
                token_parts_sha256(&[&[1], &[2]]).unwrap(),
                "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f"
            );
        }

        #[test]
        fn parity_nonce_is_exact_lowercase_hex64() {
            assert!(valid_parity_nonce(&"a5".repeat(32)));
            assert!(!valid_parity_nonce(&"A5".repeat(32)));
            assert!(!valid_parity_nonce(&"a5".repeat(31)));
            assert!(!valid_parity_nonce(&format!("{}g", "a".repeat(63))));
            assert_eq!(PLE_PARITY_WIDTH * PLE_CONV_HISTORY * 2, 184_320);
            assert_eq!(38 * PLE_PARITY_WIDTH * 2, 778_240);
            assert_eq!(2013 * PLE_PARITY_WIDTH * 2, 41_226_240);
        }

        #[test]
        fn first_artifact_metadata_failure_is_absent_and_retryable() {
            let root = private_test_root("artifact-admission");
            let frame = root.join("frame");
            fs::create_dir(&frame).unwrap();
            fs::set_permissions(&frame, fs::Permissions::from_mode(0o700)).unwrap();
            let path = frame.join("post_ple_hidden.bf16le");
            let failed = write_sealed_artifact_with_admission(
                &frame,
                "post_ple_hidden.bf16le",
                b"first",
                |_| Err(std::io::Error::other("injected first metadata failure")),
            );
            assert!(failed.is_err());
            assert!(!path.exists());
            assert!(write_sealed_artifact(&frame, "post_ple_hidden.bf16le", b"retry").is_ok());
            assert_eq!(fs::read(&path).unwrap(), b"retry");
            fs::remove_file(path).unwrap();
            fs::remove_dir(frame).unwrap();
            fs::remove_dir(root).unwrap();
        }

        #[test]
        fn frame_admission_failures_are_absent_and_retryable() {
            let root = private_test_root("frame-admission");
            let root_dev = fs::symlink_metadata(&root).unwrap().dev();
            let frame = root.join("same-frame-name");
            let failed = create_owned_frame_with_admission(frame.clone(), root_dev, |_| {
                Err(std::io::Error::other("injected frame metadata failure"))
            });
            assert!(failed.is_err());
            assert!(!frame.exists());

            let mismatch = create_owned_frame_with_admission(frame.clone(), root_dev, |_| {
                fs::symlink_metadata(&root)
            });
            assert!(mismatch.is_err());
            assert!(!frame.exists());

            let retry = create_owned_frame(frame.clone(), root_dev).unwrap();
            assert!(!frame.exists());
            assert!(retry.path().is_dir());
            drop(retry);
            assert!(!frame.exists());
            fs::remove_dir(root).unwrap();
        }

        #[test]
        fn post_mkdir_open_failure_is_absent_and_retryable() {
            let root = private_test_root("frame-open");
            let root_dev = fs::symlink_metadata(&root).unwrap().dev();
            let frame = root.join("same-frame-name");
            let failed = create_owned_frame_with_open_and_admission(
                frame.clone(),
                root_dev,
                |_| Err(std::io::Error::other("injected frame open failure")),
                |path| fs::symlink_metadata(path),
            );
            assert!(failed.is_err());
            assert!(!frame.exists());
            let quarantined: Vec<PathBuf> = fs::read_dir(&root)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect();
            assert_eq!(quarantined.len(), 1);
            assert!(
                quarantined[0]
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(".same-frame-name.staging-")
            );
            assert!(quarantined[0].is_dir());
            let retry = create_owned_frame(frame.clone(), root_dev).unwrap();
            assert!(!frame.exists());
            assert!(retry.path().is_dir());
            drop(retry);
            assert!(!frame.exists());
            fs::remove_dir(&quarantined[0]).unwrap();
            fs::remove_dir(root).unwrap();
        }

        #[test]
        fn foreign_replacements_survive_cleanup_guards() {
            let root = private_test_root("foreign-replacement");
            let artifact_frame = root.join("artifact-frame");
            fs::create_dir(&artifact_frame).unwrap();
            fs::set_permissions(&artifact_frame, fs::Permissions::from_mode(0o700)).unwrap();
            let artifact = artifact_frame.join("receipt.json");
            let displaced_artifact = artifact_frame.join("owned-displaced");
            let failed = write_sealed_artifact_with_admission(
                &artifact_frame,
                "receipt.json",
                b"owned",
                |_| {
                    fs::rename(&artifact, &displaced_artifact)?;
                    fs::write(&artifact, b"foreign")?;
                    Err(std::io::Error::other("injected replacement"))
                },
            );
            assert!(failed.is_err());
            assert_eq!(fs::read(&artifact).unwrap(), b"foreign");
            fs::remove_file(artifact).unwrap();
            fs::remove_file(displaced_artifact).unwrap();
            fs::remove_dir(artifact_frame).unwrap();

            let root_dev = fs::symlink_metadata(&root).unwrap().dev();
            let preopen_frame = root.join("preopen-frame");
            fs::create_dir(&preopen_frame).unwrap();
            fs::set_permissions(&preopen_frame, fs::Permissions::from_mode(0o711)).unwrap();
            let mut unpublished = create_owned_frame(preopen_frame.clone(), root_dev).unwrap();
            let private_stage = unpublished.path().to_path_buf();
            assert!(unpublished.publish_noreplace().is_err());
            assert_eq!(
                fs::symlink_metadata(&preopen_frame)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o711
            );
            drop(unpublished);
            assert!(!private_stage.exists());
            fs::remove_dir(preopen_frame).unwrap();

            let frame = root.join("frame");
            let displaced_frame = root.join("owned-staging-displaced");
            let mut foreign_stage = None;
            let failed = create_owned_frame_with_admission(frame.clone(), root_dev, |stage| {
                fs::rename(stage, &displaced_frame)?;
                fs::create_dir(stage)?;
                fs::set_permissions(stage, fs::Permissions::from_mode(0o711))?;
                foreign_stage = Some(stage.to_path_buf());
                Err(std::io::Error::other("injected frame replacement"))
            });
            assert!(failed.is_err());
            assert!(!frame.exists());
            assert!(displaced_frame.is_dir());
            let foreign_stage = foreign_stage.unwrap();
            assert_eq!(
                fs::symlink_metadata(&foreign_stage)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o711
            );
            fs::remove_dir(foreign_stage).unwrap();
            fs::remove_dir(displaced_frame).unwrap();
            fs::remove_dir(root).unwrap();
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use runtime::{
    PleActiveDecodeRow, PleChunkPrefetch, PlePrefillRead, PleWarmSeq, PleWarmTicket, Qwen4PleLayer,
    ple_verify_batch_selected, prefill_stream_level,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn hasher() -> Qwen4PleHasher {
        Qwen4PleHasher::new(
            [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ],
            [
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275,
            ],
            248_044,
            2_500_012,
        )
        .unwrap()
    }

    #[test]
    fn selections_match_official_torch_reference() {
        let got = hasher().select_decode(3, &[1, 2]);
        let expected = [
            (5, 2_105_657),
            (9, 1_910_767),
            (19, 1_813_339),
            (28, 2_177_092),
            (34, 1_060_412),
            (41, 1_521_518),
            (52, 963_195),
            (58, 1_885_506),
            (64, 1_522_309),
            (78, 938_190),
            (87, 1_923_521),
            (88, 810_726),
            (99, 518_953),
            (110, 227_210),
            (113, 1_796_643),
            (123, 2_143_747),
        ];
        for (selection, &(shard, row)) in got.iter().zip(&expected) {
            assert_eq!(*selection, PleRowSelection { shard, row });
        }
    }

    #[test]
    fn eos_resets_ngram_context() {
        assert_eq!(
            hasher().select_decode(3, &[1, 248_044]),
            hasher().select_decode(3, &[])
        );
    }

    #[test]
    fn skip_offloaded_ple_tensor_handles_pre_autodetect_and_sharded_names() {
        let pre_detect_prefix = "model.layers.1.ple.ple_embedding.ngram_embedding";
        for root in ["model.language_model.", "language_model.model.", "model."] {
            for suffix in [
                ".weight",
                ".weight_scale",
                ".weight_scale_2",
                ".input_scale",
            ] {
                let unsharded = format!("{root}layers.1.ple.ple_embedding.ngram_embedding{suffix}");
                assert!(
                    super::skip_offloaded_ple_tensor(pre_detect_prefix, &unsharded),
                    "{unsharded}"
                );
                for shard in [0, 1, 15, 127] {
                    let sharded = format!(
                        "{root}layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}{suffix}"
                    );
                    assert!(
                        super::skip_offloaded_ple_tensor(pre_detect_prefix, &sharded),
                        "{sharded}"
                    );
                }
            }
            // Hostile near-misses (non-table PLE projection weights must NEVER be skipped!)
            for keep in [
                format!("{root}layers.1.ple.ple_gate.weight"),
                format!("{root}layers.1.ple.ple_proj.weight"),
                format!("{root}layers.1.ple.ple_embedding.ngram_embedding.shard_.weight"),
                format!("{root}layers.1.ple.ple_embedding.ngram_embedding.shard_0.other"),
                format!("{root}layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight"),
            ] {
                assert!(
                    !super::skip_offloaded_ple_tensor(pre_detect_prefix, &keep),
                    "{keep}"
                );
            }
        }
    }

    #[test]
    fn resolve_ple_offload_manifest_path_honors_override_and_rejects_missing() {
        let default = "/models/demo/ple-offload/manifest.json";
        assert_eq!(
            super::resolve_ple_offload_manifest_path_from(default, None).unwrap(),
            std::path::PathBuf::from(default)
        );
        assert_eq!(
            super::resolve_ple_offload_manifest_path_from(default, Some("   ")).unwrap(),
            std::path::PathBuf::from(default)
        );

        let uniq = format!(
            "atlas-ple-override-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(uniq);
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("manifest.json");
        std::fs::write(&manifest, b"{}").unwrap();

        let by_dir = super::resolve_ple_offload_manifest_path_from(default, dir.to_str()).unwrap();
        assert_eq!(by_dir, manifest);

        let by_file =
            super::resolve_ple_offload_manifest_path_from(default, manifest.to_str()).unwrap();
        assert_eq!(by_file, manifest);

        let missing = dir.join("nonexistent-ple-stage");
        let err = super::resolve_ple_offload_manifest_path_from(default, missing.to_str())
            .expect_err("missing override dir must fail fast");
        assert!(
            format!("{err:#}").contains("ATLAS_PLE_OFFLOAD_DIR"),
            "unexpected error: {err:#}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
