// SPDX-License-Identifier: AGPL-3.0-only

//! Common-Random-Numbers (CRN) noise-coupled Gumbel-Max drafting for Qwen4 MTP.
//!
//! When `FLUX_SPEC_VERIFY=gumbel` and `temperature > 0` (unless disabled via
//! `ATLAS_QWEN4_MTP_GUMBEL_COUPLED=0`), the MTP draft head perturbs its compact
//! shortlist logits with `gumbel_f32(seed, abs_pos, global_token_id)` using the
//! exact same SplitMix64 seed and position key as the 248K target verifier.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle};
use spark_runtime::sampler::gumbel::{
    GumbelDraftSpec, gumbel_sampling_enabled, lookup_gumbel_draft_spec,
    select_gumbel_shortlist_bf16,
};

/// Whether noise-coupled Gumbel MTP drafting is enabled (`ATLAS_QWEN4_MTP_GUMBEL_COUPLED != 0`).
pub(super) fn coupled_drafting_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let disabled = std::env::var("ATLAS_QWEN4_MTP_GUMBEL_COUPLED")
            .ok()
            .as_deref()
            .map(str::trim)
            == Some("0");
        !disabled && gumbel_sampling_enabled()
    })
}

/// Resolve the active CRN Gumbel spec for `propose` at `position`, if enabled.
pub(super) fn active_spec_for(position: usize) -> Option<GumbelDraftSpec> {
    if !coupled_drafting_enabled() {
        return None;
    }
    lookup_gumbel_draft_spec(position)
}

/// Return the shared sampling parameters when every sequence in `specs` has an
/// active CRN Gumbel spec with identical `(temperature, top_k, top_p, min_p)`.
/// Distinct per-sequence `seed` and `step0_abs_pos` values are supported in a
/// single batched kernel launch.
pub(super) fn uniform_batch_gumbel_spec(
    specs: &[Option<GumbelDraftSpec>],
) -> Option<GumbelDraftSpec> {
    let first = (*specs.first()?)?;
    if !first.is_active() {
        return None;
    }
    let all_match = specs[1..].iter().all(|opt| {
        opt.is_some_and(|s| {
            s.is_active()
                && s.temperature.to_bits() == first.temperature.to_bits()
                && s.top_k == first.top_k
                && s.top_p.to_bits() == first.top_p.to_bits()
                && s.min_p.to_bits() == first.min_p.to_bits()
        })
    });
    all_match.then_some(first)
}

fn read_shortlist_ids_on_stream(
    gpu: &dyn GpuBackend,
    shortlist_ids_dev: DevicePtr,
    vocab: usize,
    stream: u64,
) -> Result<Option<Vec<u32>>> {
    if shortlist_ids_dev.is_null() {
        return Ok(None);
    }
    let id_bytes = vocab
        .checked_mul(4)
        .context("MTP Gumbel shortlist ID byte length overflow")?;
    let mut raw_ids = vec![0u8; id_bytes];
    gpu.copy_d2h_on_stream(shortlist_ids_dev, &mut raw_ids, stream)?;
    let ids = raw_ids
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(Some(ids))
}

/// Launch single-row `mtp_gumbel_argmax_bf16` on device, or fall back to host shortlist
/// selection if the optional kernel handle is unavailable (e.g. CPU mock tests).
#[allow(clippy::too_many_arguments)]
pub(super) fn select_chain_gumbel_step(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    shortlist_ids_dev: DevicePtr,
    argmax_out: DevicePtr,
    prob_out: Option<DevicePtr>,
    vocab: usize,
    spec: GumbelDraftSpec,
    abs_pos: u64,
    stream: u64,
) -> Result<()> {
    let vocab_u32 = u32::try_from(vocab).context("MTP Gumbel vocab exceeds u32")?;
    if kernel.0 != 0 {
        spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(shortlist_ids_dev)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(argmax_out)
            .arg_ptr(prob_out.unwrap_or(DevicePtr::NULL))
            .arg_u32(vocab_u32)
            .arg_u64(spec.seed)
            .arg_u64(abs_pos)
            .arg_f32(spec.temperature)
            .arg_u32(spec.top_k)
            .arg_f32(spec.top_p)
            .arg_f32(spec.min_p)
            .launch(stream)?;
        return Ok(());
    }

    let bf16_bytes = vocab
        .checked_mul(2)
        .context("MTP Gumbel BF16 shortlist byte length overflow")?;
    let mut raw_bf16 = vec![0u8; bf16_bytes];
    gpu.copy_d2h_on_stream(logits, &mut raw_bf16, stream)?;
    let shortlist_ids = read_shortlist_ids_on_stream(gpu, shortlist_ids_dev, vocab, stream)?;
    let (tok, prob) =
        select_gumbel_shortlist_bf16(&raw_bf16, shortlist_ids.as_deref(), spec, abs_pos)
            .context("MTP Gumbel host shortlist selection returned no finite candidate")?;
    let tok_bytes = tok.to_le_bytes();
    let prob_bytes = prob.to_le_bytes();
    if let Some(p_ptr) = prob_out {
        gpu.copy_h2d_group_on_stream(
            &[
                HostToDeviceCopy::new(&tok_bytes, argmax_out),
                HostToDeviceCopy::new(&prob_bytes, p_ptr),
            ],
            stream,
        )?;
    } else {
        gpu.copy_h2d_group_on_stream(&[HostToDeviceCopy::new(&tok_bytes, argmax_out)], stream)?;
    }
    Ok(())
}

/// Launch batched `b`-row `mtp_gumbel_argmax_bf16` on device (one 256-thread CTA per
/// sequence row with per-row `seeds_dev[row]` and `positions_dev[row]`), or fall back
/// to host shortlist selection if the optional kernel handle is unavailable (CPU mock tests).
#[allow(clippy::too_many_arguments)]
pub(super) fn select_chain_gumbel_batch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    shortlist_ids_dev: DevicePtr,
    seeds_dev: DevicePtr,
    positions_dev: DevicePtr,
    argmax_out: DevicePtr,
    prob_out: Option<DevicePtr>,
    batch: usize,
    vocab: usize,
    shared_spec: GumbelDraftSpec,
    host_specs: &[GumbelDraftSpec],
    step: u64,
    stream: u64,
) -> Result<()> {
    if batch == 0 {
        return Ok(());
    }
    let batch_u32 = u32::try_from(batch).context("MTP Gumbel batch exceeds u32")?;
    let vocab_u32 = u32::try_from(vocab).context("MTP Gumbel vocab exceeds u32")?;
    if kernel.0 != 0 {
        spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
            .grid([batch_u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(shortlist_ids_dev)
            .arg_ptr(seeds_dev)
            .arg_ptr(positions_dev)
            .arg_ptr(argmax_out)
            .arg_ptr(prob_out.unwrap_or(DevicePtr::NULL))
            .arg_u32(vocab_u32)
            .arg_u64(shared_spec.seed)
            .arg_u64(shared_spec.step0_abs_pos + step)
            .arg_f32(shared_spec.temperature)
            .arg_u32(shared_spec.top_k)
            .arg_f32(shared_spec.top_p)
            .arg_f32(shared_spec.min_p)
            .launch(stream)?;
        return Ok(());
    }

    anyhow::ensure!(
        host_specs.len() == batch,
        "MTP Gumbel batch specs length mismatch ({} != {batch})",
        host_specs.len()
    );
    let row_bytes = vocab
        .checked_mul(2)
        .context("MTP Gumbel BF16 shortlist row byte length overflow")?;
    let total_bytes = row_bytes
        .checked_mul(batch)
        .context("MTP Gumbel BF16 shortlist batch byte length overflow")?;
    let mut raw_bf16 = vec![0u8; total_bytes];
    gpu.copy_d2h_on_stream(logits, &mut raw_bf16, stream)?;
    let shortlist_ids = read_shortlist_ids_on_stream(gpu, shortlist_ids_dev, vocab, stream)?;
    let mut tok_bytes = Vec::with_capacity(batch * 4);
    let mut prob_bytes = Vec::with_capacity(if prob_out.is_some() { batch * 4 } else { 0 });
    for (row, &spec) in host_specs.iter().enumerate() {
        let row_slice = &raw_bf16[row * row_bytes..(row + 1) * row_bytes];
        let abs_pos = spec
            .step0_abs_pos
            .checked_add(step)
            .context("MTP Gumbel batch abs_pos overflow")?;
        let (tok, prob) =
            select_gumbel_shortlist_bf16(row_slice, shortlist_ids.as_deref(), spec, abs_pos)
                .with_context(|| {
                    format!(
                        "MTP Gumbel batch host shortlist selection returned no finite candidate for row {row}"
                    )
                })?;
        tok_bytes.extend_from_slice(&tok.to_le_bytes());
        if prob_out.is_some() {
            prob_bytes.extend_from_slice(&prob.to_le_bytes());
        }
    }
    if let Some(p_ptr) = prob_out {
        gpu.copy_h2d_group_on_stream(
            &[
                HostToDeviceCopy::new(&tok_bytes, argmax_out),
                HostToDeviceCopy::new(&prob_bytes, p_ptr),
            ],
            stream,
        )?;
    } else {
        gpu.copy_h2d_group_on_stream(&[HostToDeviceCopy::new(&tok_bytes, argmax_out)], stream)?;
    }
    Ok(())
}

/// Select a Gumbel-perturbed draft token on the synchronous single-step path (`chain = None`).
#[allow(clippy::too_many_arguments)]
pub(super) fn select_sync_gumbel_step(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    scratch_out: DevicePtr,
    vocab: usize,
    spec: GumbelDraftSpec,
    abs_pos: u64,
    stream: u64,
) -> Result<u32> {
    if kernel.0 != 0 {
        select_chain_gumbel_step(
            gpu,
            kernel,
            logits,
            DevicePtr::NULL,
            scratch_out,
            None,
            vocab,
            spec,
            abs_pos,
            stream,
        )?;
        let mut out = [0u8; 4];
        gpu.copy_d2h_on_stream(scratch_out, &mut out, stream)?;
        return Ok(u32::from_le_bytes(out));
    }
    let bf16_bytes = vocab
        .checked_mul(2)
        .context("MTP Gumbel BF16 shortlist byte length overflow")?;
    let mut raw_bf16 = vec![0u8; bf16_bytes];
    gpu.copy_d2h_on_stream(logits, &mut raw_bf16, stream)?;
    let (tok, _) = select_gumbel_shortlist_bf16(&raw_bf16, None, spec, abs_pos)
        .context("MTP Gumbel sync host shortlist selection returned no finite candidate")?;
    Ok(tok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn host_fallback_writes_coupled_token_and_confidence_to_device_buffers() {
        let gpu = MockGpuBackend::new();
        let vocab = 8usize;
        let logits_ptr = gpu.alloc(vocab * 2).unwrap();
        let argmax_ptr = gpu.alloc(4).unwrap();
        let prob_ptr = gpu.alloc(4).unwrap();

        let vals = [-4.0f32, 3.5, 3.25, -2.0, -5.0, 1.0, -3.0, -6.0];
        let bf16: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        gpu.copy_h2d(&bf16, logits_ptr).unwrap();

        let spec = GumbelDraftSpec {
            seed: 42,
            temperature: 0.8,
            top_k: 3,
            top_p: 0.95,
            min_p: 0.05,
            step0_abs_pos: 17,
        };
        select_chain_gumbel_step(
            &gpu,
            KernelHandle(0),
            logits_ptr,
            DevicePtr::NULL,
            argmax_ptr,
            Some(prob_ptr),
            vocab,
            spec,
            17,
            0,
        )
        .unwrap();

        let mut tok_b = [0u8; 4];
        let mut prob_b = [0u8; 4];
        gpu.copy_d2h(argmax_ptr, &mut tok_b).unwrap();
        gpu.copy_d2h(prob_ptr, &mut prob_b).unwrap();
        let tok = u32::from_le_bytes(tok_b);
        let prob = f32::from_le_bytes(prob_b);
        let expected = select_gumbel_shortlist_bf16(&bf16, None, spec, 17).unwrap();
        assert_eq!(tok, expected.0);
        assert!((prob - expected.1).abs() < 1e-6);
    }

    #[test]
    fn uniform_batch_gumbel_spec_accepts_distinct_seeds_and_positions_only() {
        let base = GumbelDraftSpec {
            seed: 101,
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            step0_abs_pos: 10,
        };
        let peer = GumbelDraftSpec {
            seed: 202,
            step0_abs_pos: 999,
            ..base
        };
        assert_eq!(
            uniform_batch_gumbel_spec(&[Some(base), Some(peer)]),
            Some(base)
        );
        assert_eq!(uniform_batch_gumbel_spec(&[]), None);
        assert_eq!(uniform_batch_gumbel_spec(&[Some(base), None]), None);
        assert_eq!(
            uniform_batch_gumbel_spec(&[
                Some(base),
                Some(GumbelDraftSpec {
                    temperature: 0.8,
                    ..peer
                }),
            ]),
            None
        );
        assert_eq!(
            uniform_batch_gumbel_spec(&[Some(base), Some(GumbelDraftSpec { top_k: 40, ..peer }),]),
            None
        );
        assert_eq!(
            uniform_batch_gumbel_spec(&[Some(base), Some(GumbelDraftSpec { top_p: 0.9, ..peer }),]),
            None
        );
        assert_eq!(
            uniform_batch_gumbel_spec(&[
                Some(base),
                Some(GumbelDraftSpec {
                    min_p: 0.05,
                    ..peer
                }),
            ]),
            None
        );
        assert_eq!(
            uniform_batch_gumbel_spec(&[Some(GumbelDraftSpec {
                temperature: 0.0,
                ..base
            })]),
            None
        );
    }

    #[test]
    fn host_fallback_batch_writes_distinct_seed_and_step0_abs_pos_rows() {
        let gpu = MockGpuBackend::new();
        let batch = 3usize;
        let vocab = 8usize;
        let logits_ptr = gpu.alloc(batch * vocab * 2).unwrap();
        let argmax_ptr = gpu.alloc(batch * 4).unwrap();
        let prob_ptr = gpu.alloc(batch * 4).unwrap();

        let rows = [
            [-4.0f32, 3.5, 3.25, -2.0, -5.0, 1.0, -3.0, -6.0],
            [1.0f32, 2.5, 2.75, 2.5, -1.0, 0.5, 2.25, -3.0],
            [0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        ];
        let bf16_rows: Vec<Vec<u8>> = rows
            .iter()
            .map(|r| {
                r.iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect()
            })
            .collect();
        let packed_bf16: Vec<u8> = bf16_rows.iter().flatten().copied().collect();
        gpu.copy_h2d(&packed_bf16, logits_ptr).unwrap();

        let specs = [
            GumbelDraftSpec {
                seed: 11,
                temperature: 1.0,
                top_k: 4,
                top_p: 0.95,
                min_p: 0.0,
                step0_abs_pos: 40,
            },
            GumbelDraftSpec {
                seed: 22,
                temperature: 1.0,
                top_k: 4,
                top_p: 0.95,
                min_p: 0.0,
                step0_abs_pos: 105,
            },
            GumbelDraftSpec {
                seed: 33,
                temperature: 1.0,
                top_k: 4,
                top_p: 0.95,
                min_p: 0.0,
                step0_abs_pos: 512,
            },
        ];
        let step = 2u64;
        select_chain_gumbel_batch(
            &gpu,
            KernelHandle(0),
            logits_ptr,
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
            argmax_ptr,
            Some(prob_ptr),
            batch,
            vocab,
            specs[0],
            &specs,
            step,
            0,
        )
        .unwrap();

        let mut tok_b = vec![0u8; batch * 4];
        let mut prob_b = vec![0u8; batch * 4];
        gpu.copy_d2h(argmax_ptr, &mut tok_b).unwrap();
        gpu.copy_d2h(prob_ptr, &mut prob_b).unwrap();
        for row in 0..batch {
            let tok = u32::from_le_bytes(tok_b[row * 4..row * 4 + 4].try_into().unwrap());
            let prob = f32::from_le_bytes(prob_b[row * 4..row * 4 + 4].try_into().unwrap());
            let expected = select_gumbel_shortlist_bf16(
                &bf16_rows[row],
                None,
                specs[row],
                specs[row].step0_abs_pos + step,
            )
            .unwrap();
            assert_eq!(tok, expected.0, "row={row}");
            assert!((prob - expected.1).abs() < 1e-6, "row={row}");
        }
    }

    #[test]
    fn host_fallback_respects_shortlist_ids_and_propagates_empty_candidates() {
        let gpu = MockGpuBackend::new();
        let vocab = 4usize;
        let logits_ptr = gpu.alloc(vocab * 2).unwrap();
        let shortlist_ptr = gpu.alloc(vocab * 4).unwrap();
        let argmax_ptr = gpu.alloc(4).unwrap();
        let prob_ptr = gpu.alloc(4).unwrap();

        let vals = [-2.0f32, 3.0, 2.5, -1.0];
        let bf16: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let shortlist_ids = [100u32, 50_001, 200_002, 777];
        let shortlist_bytes: Vec<u8> = shortlist_ids
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        gpu.copy_h2d(&bf16, logits_ptr).unwrap();
        gpu.copy_h2d(&shortlist_bytes, shortlist_ptr).unwrap();

        let spec = GumbelDraftSpec {
            seed: 99,
            temperature: 0.9,
            top_k: 4,
            top_p: 0.95,
            min_p: 0.0,
            step0_abs_pos: 23,
        };
        select_chain_gumbel_step(
            &gpu,
            KernelHandle(0),
            logits_ptr,
            shortlist_ptr,
            argmax_ptr,
            Some(prob_ptr),
            vocab,
            spec,
            23,
            7,
        )
        .unwrap();

        let mut tok_b = [0u8; 4];
        let mut prob_b = [0u8; 4];
        gpu.copy_d2h(argmax_ptr, &mut tok_b).unwrap();
        gpu.copy_d2h(prob_ptr, &mut prob_b).unwrap();
        let expected = select_gumbel_shortlist_bf16(&bf16, Some(&shortlist_ids), spec, 23).unwrap();
        assert_eq!(u32::from_le_bytes(tok_b), expected.0);
        assert!(shortlist_ids.contains(&expected.0));
        assert!((f32::from_le_bytes(prob_b) - expected.1).abs() < 1e-6);

        // All-NaN logits must fail closed instead of substituting token 0.
        let nan_bf16: Vec<u8> = [0x7fc0u16; 4]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        gpu.copy_h2d(&nan_bf16, logits_ptr).unwrap();
        assert!(
            select_chain_gumbel_step(
                &gpu,
                KernelHandle(0),
                logits_ptr,
                shortlist_ptr,
                argmax_ptr,
                Some(prob_ptr),
                vocab,
                spec,
                23,
                7,
            )
            .is_err()
        );
        assert!(
            select_sync_gumbel_step(
                &gpu,
                KernelHandle(0),
                logits_ptr,
                argmax_ptr,
                vocab,
                spec,
                23,
                7
            )
            .is_err()
        );
    }
}
