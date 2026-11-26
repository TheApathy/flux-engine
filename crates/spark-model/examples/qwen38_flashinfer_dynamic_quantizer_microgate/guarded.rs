// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::contract::REDZONE;

pub(super) struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    expected: Vec<u8>,
    immutable: bool,
}

impl Guarded {
    fn create(gpu: &dyn GpuBackend, payload: Vec<u8>, salt: u8, immutable: bool) -> Result<Self> {
        let mut expected = vec![0xa5 ^ salt; REDZONE + payload.len() + REDZONE];
        expected[REDZONE..REDZONE + payload.len()].copy_from_slice(&payload);
        expected[REDZONE + payload.len()..].fill(0x5a ^ salt);
        let allocation = gpu.alloc(expected.len())?;
        gpu.copy_h2d(&expected, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            expected,
            immutable,
        })
    }

    pub(super) fn input(gpu: &dyn GpuBackend, payload: Vec<u8>, salt: u8) -> Result<Self> {
        Self::create(gpu, payload, salt, true)
    }

    pub(super) fn output(gpu: &dyn GpuBackend, bytes: usize, salt: u8) -> Result<Self> {
        Self::create(gpu, output_sentinel(bytes, salt), salt, false)
    }

    pub(super) fn replace_payload(&self, gpu: &dyn GpuBackend, payload: &[u8]) -> Result<()> {
        ensure!(!self.immutable, "cannot overwrite immutable input buffer");
        ensure!(
            payload.len() == self.payload_len,
            "replacement extent changed"
        );
        gpu.copy_h2d(payload, self.ptr())
    }

    pub(super) fn ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn image(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut image = vec![0; self.expected.len()];
        gpu.copy_d2h(self.allocation, &mut image)?;
        Ok(image)
    }

    pub(super) fn payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let image = self.image(gpu)?;
        ensure!(
            image[..REDZONE] == self.expected[..REDZONE],
            "{label}: prefix redzone changed"
        );
        let suffix = REDZONE + self.payload_len;
        ensure!(
            image[suffix..] == self.expected[suffix..],
            "{label}: suffix redzone changed"
        );
        Ok(image[REDZONE..suffix].to_vec())
    }

    pub(super) fn check_immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        ensure!(self.immutable, "{label}: buffer is not marked immutable");
        ensure!(self.image(gpu)? == self.expected, "{label}: input changed");
        Ok(())
    }

    pub(super) fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

pub(super) fn output_sentinel(len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|index| {
            let mut value = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((value ^ (value >> 31)) as u8) ^ salt
        })
        .collect()
}

pub(super) fn disjoint_sentinel(expected: &[u8], salt: u8) -> Vec<u8> {
    let mut sentinel = output_sentinel(expected.len(), salt);
    for (actual, expected) in sentinel.iter_mut().zip(expected) {
        if actual == expected {
            *actual ^= 0xff;
        }
    }
    sentinel
}
