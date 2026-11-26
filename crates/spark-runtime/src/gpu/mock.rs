// SPDX-License-Identifier: AGPL-3.0-only

//! Mock GPU backend for unit tests (no GPU required).

use super::*;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

mod backend_impl;
mod carveout;

#[derive(Debug)]
pub struct MockAlloc {
    pub bytes: usize,
    pub data: Vec<u8>,
}

/// Records kernel launches and memory operations for test assertions.
pub struct MockGpuBackend {
    allocs: Mutex<HashMap<u64, MockAlloc>>,
    next_ptr: Mutex<u64>,
    launches: Mutex<Vec<MockLaunch>>,
    memset_count: Mutex<usize>,
    /// Allocation receipt counters this mock feeds, if any. Tests give
    /// each mock its own instance so parallel tests do not mix counts.
    alloc_counters: Option<&'static crate::alloc_receipt::AllocCounters>,
    alloc_failure_countdown: Mutex<Option<usize>>,
    invalid_frees: AtomicUsize,
    /// Lendable carveout ([`Self::with_carveout`]); `None` = no carveout.
    carveout: Mutex<Option<super::carveout::CarveoutArena>>,
}

#[derive(Debug, Clone)]
pub struct MockLaunch {
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            allocs: Mutex::new(HashMap::new()),
            next_ptr: Mutex::new(0x1000_0000),
            launches: Mutex::new(Vec::new()),
            memset_count: Mutex::new(0),
            alloc_counters: None,
            alloc_failure_countdown: Mutex::new(None),
            invalid_frees: AtomicUsize::new(0),
            carveout: Mutex::new(None),
        }
    }

    /// A mock whose `alloc`/`free` feed `counters`, like the CUDA backend
    /// feeds [`crate::alloc_receipt::GLOBAL`].
    pub fn with_alloc_counters(counters: &'static crate::alloc_receipt::AllocCounters) -> Self {
        Self {
            alloc_counters: Some(counters),
            ..Self::new()
        }
    }

    /// Inject one allocation failure after `count` succeeding allocations.
    pub fn fail_alloc_after(&self, count: usize) {
        *self.alloc_failure_countdown.lock() = Some(count);
    }

    pub(super) fn check_alloc_failure(&self) -> Result<()> {
        let mut countdown = self.alloc_failure_countdown.lock();
        match *countdown {
            Some(0) => {
                *countdown = None;
                anyhow::bail!("mock allocation failed (injected)")
            }
            Some(n) => {
                *countdown = Some(n - 1);
                Ok(())
            }
            None => Ok(()),
        }
    }

    pub fn invalid_free_count(&self) -> usize {
        self.invalid_frees.load(Ordering::Relaxed)
    }

    pub fn alloc_count(&self) -> usize {
        self.allocs.lock().len()
    }

    pub fn launch_count(&self) -> usize {
        self.launches.lock().len()
    }

    pub fn memset_count(&self) -> usize {
        *self.memset_count.lock()
    }

    pub fn read_alloc(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.allocs.lock().get(&ptr.0).map(|a| a.data.clone())
    }
}

/// Find the allocation containing `ptr` (supports offset pointers).
pub(super) fn find_alloc(
    allocs: &HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &MockAlloc)> {
    for (&base, alloc) in allocs.iter() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

/// Mutable version of find_alloc.
pub(super) fn find_alloc_mut(
    allocs: &mut HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &mut MockAlloc)> {
    for (&base, alloc) in allocs.iter_mut() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}
