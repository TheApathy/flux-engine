// SPDX-License-Identifier: AGPL-3.0-only

//! Linear construction values sealed by the pre-effect context guard.

use anyhow::{Result, ensure};

use super::runtime::ContextRuntimeMode;

/// Opaque, non-cloneable authority consumed exactly once by model construction.
#[must_use = "context admission must be consumed by model construction"]
pub(crate) struct ContextAdmissionReceipt {
    mode: ContextRuntimeMode,
    extended: bool,
    unit_yarn_speculative: bool,
}

/// Construction values released only by consuming a still-current receipt.
pub(crate) struct ContextBuildValues {
    mode: ContextRuntimeMode,
}

impl ContextAdmissionReceipt {
    pub(super) fn mint(mode: ContextRuntimeMode, extended: bool) -> Self {
        Self {
            mode,
            extended,
            unit_yarn_speculative: false,
        }
    }

    pub(super) fn mint_unit_yarn(mode: ContextRuntimeMode, extended: bool) -> Self {
        Self {
            mode,
            extended,
            unit_yarn_speculative: true,
        }
    }

    pub(crate) fn consume(self, observed_config_capacity: usize) -> Result<ContextBuildValues> {
        ensure!(
            observed_config_capacity == self.mode.config_capacity,
            "context admission is stale: config capacity changed from {} to {}",
            self.mode.config_capacity,
            observed_config_capacity,
        );
        ensure!(
            self.mode.max_seq_len > 0 && self.mode.max_seq_len <= observed_config_capacity,
            "context admission carries an invalid effective sequence extent"
        );
        if self.extended {
            ensure!(
                (self.mode.max_batch_size == 1
                    || (self.unit_yarn_speculative && self.mode.max_batch_size >= 1))
                    && (!self.mode.speculative || self.unit_yarn_speculative)
                    && (!self.mode.dflash || self.unit_yarn_speculative)
                    && !self.mode.self_speculative
                    && !self.mode.ngram_speculative,
                "extended context admission lost its target-only C1 authority"
            );
        }
        Ok(ContextBuildValues { mode: self.mode })
    }
}

impl ContextBuildValues {
    pub(crate) fn block_size(&self) -> usize {
        self.mode.block_size
    }

    pub(crate) fn max_seq_len(&self) -> usize {
        self.mode.max_seq_len
    }

    pub(crate) fn max_batch_size(&self) -> usize {
        self.mode.max_batch_size
    }

    /// Whether the operator explicitly requested the checkpoint-native MTP
    /// proposer. DFlash has its own construction path and must not implicitly
    /// construct (then demote) native MTP merely because both use the target's
    /// speculative-verify plumbing.
    pub(crate) fn native_mtp_requested(&self) -> bool {
        self.mode.speculative
    }

    pub(crate) fn self_speculative(&self) -> bool {
        self.mode.self_speculative || self.mode.ngram_speculative
    }

    pub(crate) fn dflash(&self) -> bool {
        self.mode.dflash
    }

    pub(crate) fn hss_cache_blocks_per_seq(&self) -> Option<u32> {
        self.mode
            .high_speed_swap
            .then_some(self.mode.hss_cache_blocks_per_seq)
    }
}

#[cfg(test)]
#[path = "context_extension_admission_tests.rs"]
mod tests;
