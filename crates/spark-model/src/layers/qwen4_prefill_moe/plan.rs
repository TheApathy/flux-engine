// SPDX-License-Identifier: AGPL-3.0-only

//! Pure admission and scratch layout for the isolated grouped-MoE experiment.

pub(crate) fn parse_selector(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("selector must be absent, 0, or 1"),
    }
}

pub(crate) fn validate_conflict(value: Option<&str>, tile: bool) -> Result<(), &'static str> {
    match (value, tile) {
        (None, _) | (Some("0"), false) | (Some("1"), true) => Ok(()),
        _ => Err("competing prefill selector must be disabled for MoE-only batching"),
    }
}

pub(crate) fn validate_ple_composition(
    ple_value: Option<&str>,
    composition_value: Option<&str>,
) -> Result<(), &'static str> {
    let ple = parse_selector(ple_value)?;
    let composition = parse_selector(composition_value)?;
    if ple == composition {
        Ok(())
    } else {
        Err("whole-prompt PLE with F12 requires both composition selectors to be exact value 1")
    }
}

pub(crate) fn validate_exact_qkv16_composition(
    value: Option<&str>,
    k16_value: Option<&str>,
    f12_selected: bool,
    exact_hyper_selected: bool,
    start: usize,
    rows: usize,
    multi_chunk: bool,
) -> Result<(), &'static str> {
    if !parse_selector(value)? {
        return Ok(());
    }
    if !parse_selector(k16_value)? || !f12_selected || !exact_hyper_selected {
        return Err("exact QKV16 prefill requires F12, exact HC, and exact K16 selectors");
    }
    let min_rows = if super::geometry_or_legacy().tail {
        2
    } else {
        16
    };
    if !(min_rows..=super::max_chunk_rows()).contains(&rows)
        || !super::exact_chunk_window_for_selection(rows, start, multi_chunk)
    {
        return Err("exact QKV16 prefill requires an admitted <=max-chunk-row window");
    }
    Ok(())
}

pub(crate) fn validate_exact_o16_composition(
    value: Option<&str>,
    exact_qkv16_selected: bool,
) -> Result<(), &'static str> {
    if parse_selector(value)? && !exact_qkv16_selected {
        Err("exact O16 prefill requires the exact QKV16 route")
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub capacity: usize,
    pub qkv: usize,
    pub norm: usize,
    pub hidden: usize,
    pub residual: usize,
    pub output: usize,
}

#[derive(Debug)]
pub(crate) struct Plan {
    pub row_bytes: usize,
    pub core_bytes: usize,
    pub staging_offset: usize,
    pub input_bytes: usize,
}

impl Plan {
    pub(crate) fn new(
        rows: usize,
        seq_len_start: usize,
        multi_chunk: bool,
        hidden: usize,
        residual_width: usize,
        limits: Limits,
    ) -> Result<Self, &'static str> {
        let max_rows = if seq_len_start == 0
            && let Some(mixed) = super::active_mixed_moe_rows()
            && mixed.total_rows() == Some(rows)
            && mixed.prefill_rows <= super::max_chunk_rows()
        {
            limits.capacity
        } else {
            super::max_chunk_rows().min(limits.capacity)
        };
        if !(2..=max_rows).contains(&rows) || rows > limits.capacity {
            return Err("MoE-only prefill requires 2..=max-chunk rows within arena capacity");
        }
        if !super::chunk_window_for_selection(rows, seq_len_start, multi_chunk) {
            return Err("MoE-only prefill requires an admitted <=max-chunk-row window");
        }
        if hidden == 0 || residual_width == 0 {
            return Err("MoE-only prefill requires nonzero row widths");
        }
        let row_bytes = residual_width.checked_mul(2).ok_or("row size overflow")?;
        let core_bytes = hidden.checked_mul(2).ok_or("core size overflow")?;
        let input_bytes = rows.checked_mul(core_bytes).ok_or("input size overflow")?;
        let hidden_bytes = rows.checked_mul(row_bytes).ok_or("hidden size overflow")?;
        let end = row_bytes
            .checked_add(input_bytes)
            .ok_or("staging size overflow")?;
        if end > limits.qkv || input_bytes > limits.norm || input_bytes > limits.output {
            return Err("MoE-only staging exceeds QKV, norm, or MoE output arena");
        }
        if hidden_bytes > limits.hidden || hidden_bytes > limits.residual {
            return Err("MoE-only rows exceed hidden or residual arena");
        }
        Ok(Self {
            row_bytes,
            core_bytes,
            staging_offset: row_bytes,
            input_bytes,
        })
    }
}

#[cfg(test)]
#[path = "plan/layout_tests.rs"]
mod layout_tests;

#[cfg(test)]
mod tests {
    use super::super::{chunk_window_for_selection, exact_chunk_window_for_selection};
    use super::{
        validate_exact_o16_composition, validate_exact_qkv16_composition, validate_ple_composition,
    };

    #[test]
    fn multi_chunk_window_is_explicit_aligned_and_bounded() {
        assert!(chunk_window_for_selection(1024, 1024, false));
        assert!(!exact_chunk_window_for_selection(1024, 1024, false));
        for start in [0, 2048, 4096, 6144] {
            assert!(exact_chunk_window_for_selection(2048, start, true));
        }
        for (rows, start, selected) in [
            (2048, 2048, false),
            (2047, 2048, true),
            (2048, 2049, true),
            (2048, 8192, true),
            (usize::MAX, 1, true),
        ] {
            assert!(!exact_chunk_window_for_selection(rows, start, selected));
        }
    }

    #[test]
    fn o16_requires_the_exact_qkv16_route() {
        assert!(validate_exact_o16_composition(None, false).is_ok());
        assert!(validate_exact_o16_composition(Some("0"), true).is_ok());
        assert!(validate_exact_o16_composition(Some("1"), true).is_ok());
        for case in [(Some("1"), false), (Some("true"), true)] {
            assert!(validate_exact_o16_composition(case.0, case.1).is_err());
        }
    }

    #[test]
    fn qkv16_requires_all_exact_selectors_and_initial_window() {
        assert!(validate_exact_qkv16_composition(None, None, false, false, 0, 2048, false).is_ok());
        assert!(
            validate_exact_qkv16_composition(Some("0"), None, true, true, 17, 32, false).is_ok()
        );
        assert!(
            validate_exact_qkv16_composition(Some("1"), Some("1"), true, true, 0, 2048, false,)
                .is_ok()
        );
        assert!(
            validate_exact_qkv16_composition(Some("1"), Some("1"), true, true, 6144, 2048, true,)
                .is_ok()
        );

        for case in [
            (Some("1"), None, true, true, 0, 2048, false),
            (Some("1"), Some("0"), true, true, 0, 2048, false),
            (Some("1"), Some("1"), false, true, 0, 2048, false),
            (Some("1"), Some("1"), true, false, 0, 2048, false),
            (Some("1"), Some("1"), true, true, 1, 2047, false),
            (Some("1"), Some("1"), true, true, 0, 15, false),
            (Some("1"), Some("1"), true, true, 0, 2049, false),
            (Some("1"), Some("1"), true, true, 2048, 2048, false),
            (Some("1"), Some("1"), true, true, 6145, 2047, true),
            (Some("true"), Some("1"), true, true, 0, 2048, false),
        ] {
            assert!(
                validate_exact_qkv16_composition(
                    case.0, case.1, case.2, case.3, case.4, case.5, case.6,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn ple_composition_requires_two_exact_opt_ins() {
        assert!(validate_ple_composition(None, None).is_ok());
        assert!(validate_ple_composition(Some("0"), Some("0")).is_ok());
        assert!(validate_ple_composition(Some("1"), Some("1")).is_ok());

        for (ple, composition) in [
            (Some("1"), None),
            (Some("1"), Some("0")),
            (None, Some("1")),
            (Some("0"), Some("1")),
            (Some("true"), Some("1")),
            (Some("1"), Some("true")),
        ] {
            assert!(validate_ple_composition(ple, composition).is_err());
        }
    }

    #[test]
    fn mixed_moe_rows_guard_admits_prefill_plus_decode_within_arena_capacity() {
        let max_p = super::super::max_chunk_rows();
        let decode_d = 128usize;
        let total = max_p + decode_d;
        let hidden = 2560usize;
        let residual_width = 10_240usize;
        let row_bytes = residual_width * 2;
        let core_bytes = hidden * 2;
        let limits = super::Limits {
            capacity: total,
            qkv: row_bytes + total * core_bytes,
            norm: total * core_bytes,
            hidden: total * row_bytes,
            residual: total * row_bytes,
            output: total * core_bytes,
        };

        // Without guard, P + D exceeds max_chunk_rows() and is rejected.
        assert!(super::Plan::new(total, 0, false, hidden, residual_width, limits).is_err());

        // With MixedMoeRowsGuard::enter(P, D), P + D is admitted and sized for all rows.
        {
            let _guard = super::super::MixedMoeRowsGuard::enter(max_p, decode_d);
            let plan = super::Plan::new(total, 0, false, hidden, residual_width, limits).unwrap();
            assert_eq!(plan.row_bytes, row_bytes);
            assert_eq!(plan.core_bytes, core_bytes);
            assert_eq!(plan.input_bytes, total * core_bytes);

            // Exceeding arena capacity still fails closed.
            let tight = super::Limits {
                capacity: total - 1,
                ..limits
            };
            assert!(super::Plan::new(total, 0, false, hidden, residual_width, tight).is_err());
        }

        // Prefill P > max_chunk_rows() fails closed even under the guard.
        {
            let _oversized = super::super::MixedMoeRowsGuard::enter(max_p + 1, decode_d);
            let huge = super::Limits {
                capacity: total + 1,
                qkv: row_bytes + (total + 1) * core_bytes,
                norm: (total + 1) * core_bytes,
                hidden: (total + 1) * row_bytes,
                residual: (total + 1) * row_bytes,
                output: (total + 1) * core_bytes,
            };
            assert!(super::Plan::new(total + 1, 0, false, hidden, residual_width, huge).is_err());
        }
    }
}
