// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::c_void;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::KernelHandle;

pub(super) const NQ: u32 = 24;
pub(super) const NKV: u32 = 4;
pub(super) const HD: u32 = 256;
pub(super) const CACHE_BLOCK: u32 = 16;
pub(super) const REDZONE: usize = 4 * 1024;
pub(super) const CANARY: u8 = 0xa5;
pub(super) const TIMING_PAIRS: usize = 24;

#[derive(Clone, Copy)]
pub(super) struct Case {
    pub(super) q_len: u32,
    pub(super) q_offset: u32,
    pub(super) sliding_window: u32,
}

impl Case {
    pub(super) const fn new(q_len: u32, q_offset: u32, sliding_window: u32) -> Self {
        Self {
            q_len,
            q_offset,
            sliding_window,
        }
    }

    pub(super) fn label(self) -> String {
        format!(
            "q{}_off{}_win{}",
            self.q_len, self.q_offset, self.sliding_window
        )
    }

    pub(super) fn kv_len(self) -> u32 {
        self.q_offset
            .checked_add(self.q_len)
            .expect("admitted case")
    }
}

pub(super) const CONTINUATION_CASES: [Case; 3] = [
    Case {
        q_len: 8192,
        q_offset: 8192,
        sliding_window: 0,
    },
    Case {
        q_len: 8192,
        q_offset: 16384,
        sliding_window: 0,
    },
    Case {
        q_len: 8192,
        q_offset: 24576,
        sliding_window: 0,
    },
];

pub(super) const SMOKE_CASES: [Case; 2] = [Case::new(2048, 8192, 0), Case::new(2113, 97, 33)];

pub(super) const FULL_CASES: [Case; 18] = [
    Case::new(1, 0, 0),
    Case::new(31, 15, 1),
    Case::new(32, 16, 31),
    Case::new(63, 17, 32),
    Case::new(64, 31, 33),
    Case::new(65, 32, 0),
    Case::new(127, 33, 4096),
    Case::new(128, 0, 32),
    Case::new(129, 17, 33),
    Case::new(255, 31, 0),
    Case::new(256, 32, 4096),
    Case::new(2047, 33, 31),
    Case::new(2048, 0, 0),
    Case::new(8192, 8192, 4096),
    CONTINUATION_CASES[0],
    CONTINUATION_CASES[1],
    CONTINUATION_CASES[2],
    Case::new(2113, 97, 33),
];

pub(super) fn strict_switch(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => bail!("{name} must be exactly 0 or 1"),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be valid UTF-8"),
    }
}

pub(super) fn require_target() -> Result<()> {
    ensure!(
        std::env::var("ATLAS_TARGET_MODEL").as_deref() == Ok("qwen3.8-27b"),
        "requires ATLAS_TARGET_MODEL=qwen3.8-27b"
    );
    ensure!(
        std::env::var("ATLAS_TARGET_QUANT").as_deref() == Ok("nvfp4"),
        "requires ATLAS_TARGET_QUANT=nvfp4"
    );
    Ok(())
}

pub(super) fn exact_modules() -> Result<Vec<(&'static str, &'static str)>> {
    let mut sets: Vec<_> = atlas_kernels::available_targets()
        .into_iter()
        .filter(|set| {
            set.target.arch == "sm_121"
                && set.target.model == "qwen3.8-27b"
                && set.target.quant == "nvfp4"
        })
        .collect();
    ensure!(
        sets.len() == 1,
        "requires exactly one embedded SM121 Qwen3.8 NVFP4 bundle"
    );
    Ok(sets.pop().context("target bundle disappeared")?.modules)
}

#[derive(Debug)]
pub(super) struct Resources {
    pub(super) max_threads: i32,
    pub(super) shared_bytes: i32,
    pub(super) local_bytes: i32,
    pub(super) registers: i32,
    pub(super) max_dynamic_shared_bytes: i32,
}

// SAFETY: CUDA Driver API symbols from `cuda.h` linked via `spark-runtime`.
unsafe extern "C" {
    fn cuCtxGetDevice(device: *mut i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: u32, device: i32) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attribute: u32, function: *mut c_void) -> i32;
    fn cuFuncSetAttribute(function: *mut c_void, attribute: u32, value: i32) -> i32;
}

pub(super) fn require_gb10() -> Result<()> {
    let mut device = -1;
    // SAFETY: Active CUDA context exists after `AtlasCudaBackend::new`; `&mut device` is valid.
    ensure!(
        unsafe { cuCtxGetDevice(&mut device) } == 0 && device >= 0,
        "CUDA context/device unavailable"
    );
    let attr = |kind| {
        let mut value = -1;
        // SAFETY: `device` is a valid CUDA ordinal from `cuCtxGetDevice` and `&mut value` is valid.
        ensure!(
            unsafe { cuDeviceGetAttribute(&mut value, kind, device) } == 0,
            "CUDA attribute {kind} failed"
        );
        Ok::<_, anyhow::Error>(value)
    };
    ensure!(
        attr(75)? == 12 && attr(76)? == 1 && attr(16)? == 48,
        "requires GB10 SM121 with 48 SMs"
    );
    Ok(())
}

pub(super) fn function_resources(kernel: KernelHandle) -> Result<Resources> {
    let function = kernel.0 as usize as *mut c_void;
    let dynamic_shared_bytes = 95_808;
    // SAFETY: `function` is a loaded `CUfunction` handle in the active CUDA context.
    let set_status = unsafe { cuFuncSetAttribute(function, 8, dynamic_shared_bytes) };
    ensure!(
        set_status == 0,
        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={dynamic_shared_bytes}) failed: {set_status}"
    );
    let attr = |kind| {
        let mut value = -1;
        // SAFETY: `function` is a loaded `CUfunction` handle and `&mut value` is valid.
        let status = unsafe { cuFuncGetAttribute(&mut value, kind, function) };
        ensure!(
            status == 0 && value >= 0,
            "cuFuncGetAttribute({kind}) failed: {status}"
        );
        Ok::<_, anyhow::Error>(value)
    };
    Ok(Resources {
        max_threads: attr(0)?,
        shared_bytes: attr(1)?,
        local_bytes: attr(3)?,
        registers: attr(4)?,
        max_dynamic_shared_bytes: attr(8)?,
    })
}
