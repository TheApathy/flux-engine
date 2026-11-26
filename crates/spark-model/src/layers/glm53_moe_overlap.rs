// SPDX-License-Identifier: AGPL-3.0-only

//! Admission and execution receipt for prompt routed/shared MoE overlap.

use std::ffi::OsStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;

const ENV: &str = "ATLAS_GLM53_EXL3_MOE_SHARED_OVERLAP";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Setting {
    Off,
    On,
}

pub(super) fn parse(value: Option<&OsStr>) -> Result<Setting> {
    match text(ENV, value)? {
        None | Some("0") => Ok(Setting::Off),
        Some("1") => Ok(Setting::On),
        Some(_) => bail!("{ENV} must be exactly 0 or 1"),
    }
}

#[derive(Clone, Copy)]
pub(super) struct Admission<'a> {
    pub request: Option<&'a OsStr>,
    pub private_prefill: bool,
    pub staged_loaded: bool,
    pub reconstruct: Option<&'a OsStr>,
    pub reconstruct_dtype: Option<&'a OsStr>,
    pub reconstruct_overlap: Option<&'a OsStr>,
    pub chunk_rows: Option<&'a OsStr>,
    pub direct_combine: Option<&'a OsStr>,
    pub layer_major: Option<&'a OsStr>,
    pub layer_major_rows: Option<&'a OsStr>,
}

pub(super) fn admit(input: Admission<'_>) -> Result<Setting> {
    let setting = parse(input.request)?;
    if setting == Setting::Off {
        return Ok(setting);
    }
    ensure!(
        input.private_prefill && input.staged_loaded,
        "{ENV}=1 requires private-prefill routing and staged MoE"
    );
    require(
        "ATLAS_GLM53_EXL3_PREFILL_RECONSTRUCT",
        input.reconstruct,
        "1",
    )?;
    require(
        "ATLAS_GLM53_EXL3_RECONSTRUCT_DTYPE",
        input.reconstruct_dtype,
        "bf16",
    )?;
    match text(
        "ATLAS_GLM53_EXL3_RECONSTRUCT_OVERLAP",
        input.reconstruct_overlap,
    )? {
        None | Some("0") => {}
        Some(_) => {
            bail!("{ENV}=1 excludes ATLAS_GLM53_EXL3_RECONSTRUCT_OVERLAP; isolate one overlap axis")
        }
    }
    require("ATLAS_GLM53_EXL3_MOE_CHUNK_ROWS", input.chunk_rows, "64k2")?;
    require(
        "ATLAS_GLM53_EXL3_MOE_PREFILL_DIRECT_COMBINE",
        input.direct_combine,
        "1",
    )?;
    require("ATLAS_GLM53_LAYER_MAJOR_PREFILL", input.layer_major, "1")?;
    require(
        "ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS",
        input.layer_major_rows,
        "2048",
    )?;
    Ok(setting)
}

fn require(name: &str, value: Option<&OsStr>, expected: &str) -> Result<()> {
    ensure!(
        text(name, value)? == Some(expected),
        "{ENV}=1 requires {name}={expected}"
    );
    Ok(())
}

fn text<'a>(name: &str, value: Option<&'a OsStr>) -> Result<Option<&'a str>> {
    value
        .map(|v| v.to_str().with_context(|| format!("{name} must be UTF-8")))
        .transpose()
}

pub struct Glm53MoeOverlap {
    side_stream: u64,
    fork: u64,
    join: u64,
    executions: AtomicU64,
    busy: AtomicBool,
}

/// Owns the single fork/join pair through the caller's oracle and combine seam.
pub(super) struct Joined<'a> {
    owner: &'a Glm53MoeOverlap,
    gpu: &'a dyn GpuBackend,
    main_stream: u64,
    work_in_flight: bool,
    completed: bool,
}

impl Joined<'_> {
    /// Mark a layer successful only after its final combine was enqueued.
    pub(super) fn complete(mut self, rows: u32) {
        self.owner.record_success(rows);
        self.completed = true;
    }
}

impl Drop for Joined<'_> {
    fn drop(&mut self) {
        if self.work_in_flight && !self.completed {
            self.owner.quiesce(self.gpu, self.main_stream);
        }
        self.owner.busy.store(false, Ordering::Release);
    }
}

impl Glm53MoeOverlap {
    pub(crate) fn from_env(
        gpu: &dyn GpuBackend,
        private_prefill: bool,
        staged_loaded: bool,
    ) -> Result<Option<Self>> {
        initialize(setting_from_env(private_prefill, staged_loaded)?, || {
            Self::new(gpu)
        })
    }

    pub(crate) fn validate_env(
        owner: Option<&Self>,
        private_prefill: bool,
        staged_loaded: bool,
    ) -> Result<()> {
        validate_owner_state(
            setting_from_env(private_prefill, staged_loaded)?,
            owner.is_some(),
        )
    }

    pub(super) fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        let side_stream = gpu.create_stream()?;
        spark_runtime::cublaslt::register_stream_workspace(side_stream)?;
        let fork = gpu.create_event()?;
        let join = match gpu.create_event() {
            Ok(event) => event,
            Err(error) => {
                let _ = gpu.destroy_event(fork);
                return Err(error);
            }
        };
        let overlap = Self {
            side_stream,
            fork,
            join,
            executions: AtomicU64::new(0),
            busy: AtomicBool::new(false),
        };
        eprintln!(
            "GLM53_MOE_SHARED_OVERLAP ARMED side_stream={} recipe=2048/64k2/bf16",
            overlap.side_stream
        );
        Ok(overlap)
    }

    pub(super) fn run<'a>(
        &'a self,
        gpu: &'a dyn GpuBackend,
        main_stream: u64,
        rows: u32,
        shared: impl FnOnce(u64) -> Result<()>,
        routed: impl FnOnce() -> Result<()>,
    ) -> Result<Joined<'a>> {
        ensure!(
            self.busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "{ENV}=1 refuses concurrent use of its single fork/join event pair"
        );
        let mut joined = Joined {
            owner: self,
            gpu,
            main_stream,
            work_in_flight: false,
            completed: false,
        };
        ensure!(
            (1024..=2048).contains(&rows) && !gpu.stream_is_capturing(main_stream),
            "GLM routed/shared overlap requires eager prompt rows 1024..=2048"
        );
        gpu.record_event(self.fork, main_stream)?;
        joined.work_in_flight = true;
        gpu.stream_wait_event(self.side_stream, self.fork)
            .context("GLM shared expert overlap fork wait failed")?;
        shared(self.side_stream).context("GLM shared expert overlap enqueue failed")?;
        gpu.record_event(self.join, self.side_stream)
            .context("GLM shared expert overlap join record failed")?;
        routed().context("GLM routed MoE overlap enqueue failed")?;
        gpu.stream_wait_event(main_stream, self.join)
            .context("GLM routed/shared overlap join wait failed")?;
        Ok(joined)
    }

    /// Count only a layer whose post-join observation and final combine both succeeded.
    pub(super) fn record_success(&self, rows: u32) -> u64 {
        let count = self.executions.fetch_add(1, Ordering::Relaxed) + 1;
        if receipt_due(count) {
            eprintln!(
                "GLM53_MOE_SHARED_OVERLAP EXECUTED layers={count} fork_events={count} join_events={count} rows={rows}"
            );
        }
        count
    }

    /// Synchronize the overlap side stream so no in-flight shared-expert work
    /// remains before freeing device buffers.
    pub(crate) fn drain(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.synchronize(self.side_stream)
            .context("GLM routed/shared overlap side-stream drain failed")
    }

    /// Synchronize the side stream and destroy the fork/join CUDA events.
    pub(crate) fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        self.drain(gpu)?;
        let fork_res = gpu
            .destroy_event(self.fork)
            .context("GLM routed/shared overlap fork event destroy failed");
        let join_res = gpu
            .destroy_event(self.join)
            .context("GLM routed/shared overlap join event destroy failed");
        fork_res?;
        join_res
    }

    fn quiesce(&self, gpu: &dyn GpuBackend, main_stream: u64) {
        let _ = gpu.synchronize(main_stream);
        let _ = gpu.synchronize(self.side_stream);
    }
}

fn setting_from_env(private_prefill: bool, staged_loaded: bool) -> Result<Setting> {
    let request = std::env::var_os(ENV);
    let reconstruct = std::env::var_os("ATLAS_GLM53_EXL3_PREFILL_RECONSTRUCT");
    let reconstruct_dtype = std::env::var_os("ATLAS_GLM53_EXL3_RECONSTRUCT_DTYPE");
    let reconstruct_overlap = std::env::var_os("ATLAS_GLM53_EXL3_RECONSTRUCT_OVERLAP");
    let chunk_rows = std::env::var_os("ATLAS_GLM53_EXL3_MOE_CHUNK_ROWS");
    let direct_combine = std::env::var_os("ATLAS_GLM53_EXL3_MOE_PREFILL_DIRECT_COMBINE");
    let layer_major = std::env::var_os("ATLAS_GLM53_LAYER_MAJOR_PREFILL");
    let layer_major_rows = std::env::var_os("ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS");
    admit(Admission {
        request: request.as_deref(),
        private_prefill,
        staged_loaded,
        reconstruct: reconstruct.as_deref(),
        reconstruct_dtype: reconstruct_dtype.as_deref(),
        reconstruct_overlap: reconstruct_overlap.as_deref(),
        chunk_rows: chunk_rows.as_deref(),
        direct_combine: direct_combine.as_deref(),
        layer_major: layer_major.as_deref(),
        layer_major_rows: layer_major_rows.as_deref(),
    })
}

fn initialize<T>(setting: Setting, create: impl FnOnce() -> Result<T>) -> Result<Option<T>> {
    match setting {
        Setting::Off => Ok(None),
        Setting::On => create().map(Some),
    }
}

fn validate_owner_state(setting: Setting, owner_present: bool) -> Result<()> {
    ensure!(
        (setting == Setting::On) == owner_present,
        "{ENV} changed after model construction"
    );
    Ok(())
}

const fn receipt_due(count: u64) -> bool {
    count == 1 || count.is_multiple_of(42)
}

pub(super) const fn scope(rows: u32) -> bool {
    rows >= 1024 && rows <= 2048
}

#[cfg(test)]
#[path = "glm53_moe_overlap_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm53_moe_overlap_lifecycle_tests.rs"]
mod lifecycle_tests;
