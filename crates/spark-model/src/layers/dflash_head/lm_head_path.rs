// SPDX-License-Identifier: AGPL-3.0-only

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LmHeadPath {
    Nvfp4,
    Fp8,
    Bf16,
}

pub(super) fn resolve_lm_head_path(
    nvfp4_requested: bool,
    nvfp4_weight_present: bool,
    nvfp4_kernel_present: bool,
    fp8_present: bool,
) -> Result<LmHeadPath, &'static str> {
    if nvfp4_requested {
        if !nvfp4_weight_present {
            return Err("DFlash NVFP4 LM head requested but its transposed weight is absent");
        }
        if !nvfp4_kernel_present {
            return Err("DFlash NVFP4 LM head requested but its GEMM kernel is absent");
        }
        return Ok(LmHeadPath::Nvfp4);
    }

    Ok(if fp8_present {
        LmHeadPath::Fp8
    } else {
        LmHeadPath::Bf16
    })
}

impl LmHeadPath {
    pub(super) fn uses_fp8_input_norm(self) -> bool {
        self == Self::Fp8
    }

    pub(super) fn restores_unscaled_norm_for_selector(self, selector_present: bool) -> bool {
        self.uses_fp8_input_norm() && selector_present
    }
}

#[cfg(test)]
#[path = "lm_head_path_tests.rs"]
mod tests;
