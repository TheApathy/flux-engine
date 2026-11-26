// SPDX-License-Identifier: AGPL-3.0-only

//! Host oracle and typed ABI for sampled DFlash2 proposals.
//!
//! The production CUDA selector returns a typed sparse proposal distribution,
//! and the narrow scheduler transaction consumes it with exact target-p/q
//! acceptance. Runtime admission remains fail-closed until the candidate's
//! compile, GPU, endpoint, and bounded OFF/ON/OFF gates are sealed; source
//! presence never substitutes for qualification evidence.

mod acceptance;
mod receipt;
mod selector;
mod target;
mod types;

pub use acceptance::accept_sparse_dflash;
pub use receipt::{
    DFLASH_SAMPLING_RECEIPT_PREFIX, DFLASH_SAMPLING_RECEIPT_SCHEMA, DflashAcceptanceRule,
    DflashCandidateSource, DflashCanonicalCommitRule, DflashCommitReceipt,
    DflashNextTokenDisposition, DflashQNormalization, DflashResidualRule, DflashRngRule,
    DflashSamplingReceipt, DflashTargetDistribution,
};
pub use selector::{
    decode_dflash2_device_proposal, select_dflash2_path, select_dflash2_path_with_q_policy,
};
pub use target::{
    DflashTargetEvidence, DflashTargetLogitsFormat, DflashTargetProbabilityEntry,
    DflashTargetProbabilityRow, evaluate_sampled_dflash, evaluate_sampled_dflash_with_evidence,
    prepare_dflash_target_probabilities,
};
pub use types::{
    DFLASH2_MAX_Q_MODIFIERS, DFLASH2_SELECTOR_TOP_K, DflashDraftSampleMethod,
    DflashDraftSamplingRequest, DflashResidualBranch, DflashRngSeedSource, DflashSampledAcceptance,
    DflashSampledProposal, DflashSampledProposalFrame, DflashSparseQRow, DflashTargetPolicyPhase,
    RequestSamplerReceipt, dflash_history_sha256, dflash_target_policy_sha256,
    validate_dflash_runtime_admission,
};

#[cfg(test)]
mod device_transport_tests;
#[cfg(test)]
mod tests;
