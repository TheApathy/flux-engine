// SPDX-License-Identifier: AGPL-3.0-only

//! Raw CUDA parity gate for the sampled DFlash2 candidate selector.
//!
//! This intentionally loads no model. It proves that the production CUDA
//! symbol exports the exact candidate rows and normalized sparse-q values,
//! and that every returned token is selected by the receipted f32 uniform
//! using the production strict-`>` CDF rule. The cases cover pair-adjusted
//! sequential walks, exact ties, extreme scores with trailing zero q, and
//! multiple deterministic input seeds.
//!
//! Run with the Qwen3.8-27B kernel target:
//! `ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b \
//!  ATLAS_TARGET_QUANT=nvfp4 cargo run --release -p spark-model \
//!  --example dflash2_probabilistic_selector_microtest`

use anyhow::{Context, Result, ensure};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const TOP_K: usize = 16;
const Q_SUM_TOLERANCE: f32 = 2.0e-5;
const Q_ELEMENT_TOLERANCE: f32 = 2.0e-5;

#[derive(Debug)]
struct SelectorCase {
    name: String,
    gamma: usize,
    rank: usize,
    vocab: usize,
    temperature: f32,
    top_k: u32,
    min_p: f32,
    top_p: f32,
    anchor: u32,
    unary: Vec<f32>,
    candidates: Vec<u32>,
    hidden: Vec<u16>,
    predecessor: Vec<u16>,
    successor: Vec<u16>,
    uniforms: Vec<f32>,
    logit_bias: Vec<(u32, f32)>,
    forbidden_tokens: Vec<u32>,
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = if (bits >> 23) & 0xff == 0xff {
        bits
    } else {
        bits + 0x7fff + ((bits >> 16) & 1)
    };
    (rounded >> 16) as u16
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn next_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 8) & 0xff_ffff) as f32 / (1u32 << 24) as f32
}

fn random_bf16(state: &mut u64, count: usize, scale: f32) -> Vec<u16> {
    (0..count)
        .map(|_| f32_to_bf16((next_f32(state) - 0.5) * 2.0 * scale))
        .collect()
}

fn as_bytes_u16(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn as_bytes_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn as_bytes_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn decode_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn select_strict_f32(probabilities: &[f32], uniform: f32) -> usize {
    let mut cumulative = 0.0f32;
    let mut last_nonzero = 0usize;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability > 0.0 {
            last_nonzero = index;
        }
        cumulative += probability;
        if cumulative > uniform {
            return index;
        }
    }
    last_nonzero
}

fn filtered_softmax_f32(
    scores: &[f32; TOP_K],
    candidates: &[u32],
    temperature: f32,
    top_k: u32,
    min_p: f32,
    top_p: f32,
) -> [f32; TOP_K] {
    let mut order = (0..TOP_K)
        .filter(|&index| scores[index].is_finite())
        .collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        if scores[left] > scores[right] {
            std::cmp::Ordering::Less
        } else if scores[left] < scores[right] {
            std::cmp::Ordering::Greater
        } else {
            candidates[left].cmp(&candidates[right])
        }
    });
    if top_k > 0 {
        order.truncate(order.len().min(top_k as usize));
    }
    let maximum = scores[order[0]];
    let mut weights = [0.0f32; TOP_K];
    for &index in &order {
        weights[index] = ((scores[index] - maximum) / temperature).exp();
    }
    let min_weight = min_p * weights[order[0]];
    order.retain(|&index| weights[index] >= min_weight);
    if top_p < 1.0 {
        let sum = order.iter().map(|&index| weights[index]).sum::<f32>();
        let mut cumulative = 0.0f32;
        let mut keep = order.len();
        for (rank, &index) in order.iter().enumerate() {
            cumulative += weights[index] / sum;
            if cumulative >= top_p {
                keep = rank + 1;
                break;
            }
        }
        order.truncate(keep);
    }
    let sum = order.iter().map(|&index| weights[index]).sum::<f32>();
    let mut probabilities = [0.0f32; TOP_K];
    for &index in &order {
        probabilities[index] = weights[index] / sum;
    }
    probabilities
}

fn cpu_reference(case: &SelectorCase) -> (Vec<u32>, Vec<f32>) {
    let mut path = Vec::with_capacity(case.gamma);
    let mut q = Vec::with_capacity(case.gamma * TOP_K);
    let mut previous = case.anchor as usize;
    for position in 0..case.gamma {
        let mut scores = [0.0f32; TOP_K];
        for (candidate_index, score) in scores.iter_mut().enumerate() {
            let candidate = case.candidates[position * TOP_K + candidate_index] as usize;
            let mut pair_adjustment = 0.0f32;
            for rank_index in 0..case.rank {
                pair_adjustment += bf16_to_f32(case.predecessor[previous * case.rank + rank_index])
                    * bf16_to_f32(case.hidden[position * case.rank + rank_index])
                    * bf16_to_f32(case.successor[candidate * case.rank + rank_index]);
            }
            *score = case.unary[position * TOP_K + candidate_index] + pair_adjustment;
            for &(token, bias) in &case.logit_bias {
                if token as usize == candidate {
                    *score += bias;
                }
            }
            if case
                .forbidden_tokens
                .binary_search(&(candidate as u32))
                .is_ok()
            {
                *score = f32::NEG_INFINITY;
            }
        }
        let row_candidates = &case.candidates[position * TOP_K..(position + 1) * TOP_K];
        let probabilities = filtered_softmax_f32(
            &scores,
            row_candidates,
            case.temperature,
            case.top_k,
            case.min_p,
            case.top_p,
        );
        let selected = select_strict_f32(&probabilities, case.uniforms[position]);
        previous = case.candidates[position * TOP_K + selected] as usize;
        path.push(previous as u32);
        q.extend_from_slice(&probabilities);
    }
    (path, q)
}

fn random_case(seed: u64) -> SelectorCase {
    let gamma = 8;
    let rank = 32;
    let vocab = 257;
    let mut state = seed;
    let unary = (0..gamma * TOP_K)
        .map(|_| (next_f32(&mut state) - 0.5) * 3.0)
        .collect();
    let candidate_offset = (seed as usize) % (vocab - TOP_K);
    let candidates = (0..gamma)
        .flat_map(|position| {
            (0..TOP_K)
                .map(move |index| ((candidate_offset + position * TOP_K + index) % vocab) as u32)
        })
        .collect();
    let hidden = random_bf16(&mut state, gamma * rank, 0.5);
    let predecessor = random_bf16(&mut state, vocab * rank, 0.25);
    let successor = random_bf16(&mut state, vocab * rank, 0.25);
    let uniforms = (0..gamma)
        .map(|_| 0.01 + next_f32(&mut state) * 0.98)
        .collect();
    SelectorCase {
        name: format!("pair-adjusted-seed-{seed:016x}"),
        gamma,
        rank,
        vocab,
        temperature: 0.85,
        top_k: 0,
        min_p: 0.0,
        top_p: 1.0,
        anchor: ((seed >> 32) as usize % vocab) as u32,
        unary,
        candidates,
        hidden,
        predecessor,
        successor,
        uniforms,
        logit_bias: Vec::new(),
        forbidden_tokens: Vec::new(),
    }
}

fn zero_pair_case(
    name: &str,
    unary: Vec<f32>,
    uniforms: Vec<f32>,
    temperature: f32,
) -> SelectorCase {
    let gamma = uniforms.len();
    let rank = 4;
    let vocab = gamma * TOP_K + 1;
    let candidates = (0..gamma * TOP_K).map(|token| token as u32).collect();
    SelectorCase {
        name: name.to_owned(),
        gamma,
        rank,
        vocab,
        temperature,
        top_k: 0,
        min_p: 0.0,
        top_p: 1.0,
        anchor: (vocab - 1) as u32,
        unary,
        candidates,
        hidden: vec![f32_to_bf16(0.0); gamma * rank],
        predecessor: vec![f32_to_bf16(0.0); vocab * rank],
        successor: vec![f32_to_bf16(0.0); vocab * rank],
        uniforms,
        logit_bias: Vec::new(),
        forbidden_tokens: Vec::new(),
    }
}

fn tie_case() -> SelectorCase {
    let uniforms = vec![0.0, 0.0625, 0.125, 0.9375, 0.999_999_94];
    zero_pair_case(
        "exact-ties-and-cdf-boundaries",
        vec![0.0; uniforms.len() * TOP_K],
        uniforms,
        1.0,
    )
}

fn extreme_case() -> SelectorCase {
    let uniforms = vec![0.0, 0.5, 0.999_999_94, 0.25];
    let maxima = [0usize, 7, 11, 15];
    let mut unary = vec![-200.0; uniforms.len() * TOP_K];
    for (position, maximum) in maxima.into_iter().enumerate() {
        unary[position * TOP_K + maximum] = 0.0;
    }
    zero_pair_case("extreme-underflow-and-last-nonzero", unary, uniforms, 0.25)
}

fn sparse_q_policy_case() -> SelectorCase {
    let mut case = zero_pair_case(
        "ordered-logit-bias-and-hard-mask",
        vec![0.0; 2 * TOP_K],
        vec![0.0, 0.0],
        1.0,
    );
    case.logit_bias = vec![(0, -100.0), (1, 2.0), (2, 100.0), (3, -0.5), (3, -0.5)];
    case.forbidden_tokens = vec![2, 16];
    case
}

fn scalar_sampler_policy_case() -> SelectorCase {
    let mut unary = vec![-100.0; 2 * TOP_K];
    for row in 0..2 {
        unary[row * TOP_K + 5] = 0.0;
        unary[row * TOP_K + 1] = 0.2f32.ln();
        unary[row * TOP_K + 9] = 0.19f32.ln();
    }
    let mut case = zero_pair_case(
        "top-k-softmax-min-p-top-p",
        unary,
        vec![0.0, 0.999_999_94],
        1.0,
    );
    case.top_k = 3;
    case.min_p = 0.195;
    case.top_p = 0.82;
    case
}

fn run_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernel: KernelHandle,
    case: &SelectorCase,
) -> Result<f32> {
    ensure!(case.temperature.is_finite() && case.temperature > 0.0);
    ensure!(case.min_p.is_finite() && (0.0..=1.0).contains(&case.min_p));
    ensure!(case.top_p.is_finite() && case.top_p > 0.0 && case.top_p <= 1.0);
    ensure!((case.anchor as usize) < case.vocab);
    ensure!(case.uniforms.len() == case.gamma);
    ensure!(
        case.uniforms
            .iter()
            .all(|uniform| uniform.is_finite() && (0.0..1.0).contains(uniform))
    );

    let unary = upload(gpu, &as_bytes_f32(&case.unary))?;
    let candidates = upload(gpu, &as_bytes_u32(&case.candidates))?;
    let hidden = upload(gpu, &as_bytes_u16(&case.hidden))?;
    let predecessor = upload(gpu, &as_bytes_u16(&case.predecessor))?;
    let successor = upload(gpu, &as_bytes_u16(&case.successor))?;
    let uniforms = upload(gpu, &as_bytes_f32(&case.uniforms))?;
    let mut modifier_ids = case
        .logit_bias
        .iter()
        .map(|&(token, _)| token)
        .collect::<Vec<_>>();
    let mut modifier_values = case
        .logit_bias
        .iter()
        .map(|&(_, bias)| bias)
        .collect::<Vec<_>>();
    modifier_ids.extend(case.forbidden_tokens.iter().copied());
    modifier_values.extend(case.forbidden_tokens.iter().map(|_| f32::NEG_INFINITY));
    let modifier_count = modifier_ids.len();
    ensure!(modifier_count <= 64, "microtest q policy exceeds CUDA ABI");
    if modifier_ids.is_empty() {
        modifier_ids.push(0);
        modifier_values.push(0.0);
    }
    let q_modifier_ids = upload(gpu, &as_bytes_u32(&modifier_ids))?;
    let q_modifier_values = upload(gpu, &as_bytes_f32(&modifier_values))?;
    let path = gpu.alloc(case.gamma * size_of::<u32>())?;
    let q_probabilities = gpu.alloc(case.gamma * TOP_K * size_of::<f32>())?;
    let q_candidates = gpu.alloc(case.gamma * TOP_K * size_of::<u32>())?;

    ops::dflash2_selector_walk_probabilistic(
        gpu,
        kernel,
        unary,
        candidates,
        hidden,
        predecessor,
        successor,
        path,
        uniforms,
        q_modifier_ids,
        q_modifier_values,
        modifier_count as i32,
        case.temperature,
        case.top_k,
        case.min_p,
        case.top_p,
        q_probabilities,
        q_candidates,
        case.anchor,
        case.gamma as i32,
        case.rank as i32,
        stream,
    )?;

    let mut path_bytes = vec![0u8; case.gamma * size_of::<u32>()];
    let mut candidate_bytes = vec![0u8; case.gamma * TOP_K * size_of::<u32>()];
    let mut probability_bytes = vec![0u8; case.gamma * TOP_K * size_of::<f32>()];
    gpu.copy_d2h_pair_on_stream(
        path,
        &mut path_bytes,
        q_candidates,
        &mut candidate_bytes,
        stream,
    )?;
    gpu.copy_d2h(q_probabilities, &mut probability_bytes)?;

    for ptr in [
        unary,
        candidates,
        hidden,
        predecessor,
        successor,
        uniforms,
        q_modifier_ids,
        q_modifier_values,
        path,
        q_probabilities,
        q_candidates,
    ] {
        gpu.free(ptr)?;
    }

    let actual_path = decode_u32(&path_bytes);
    let actual_candidates = decode_u32(&candidate_bytes);
    let actual_q = decode_f32(&probability_bytes);
    ensure!(
        actual_candidates == case.candidates,
        "{}: exported candidate IDs differ from selector input",
        case.name
    );

    let (expected_path, expected_q) = cpu_reference(case);
    ensure!(
        actual_path == expected_path,
        "{}: sampled path mismatch: device={actual_path:?} host={expected_path:?}",
        case.name
    );
    let mut max_q_error = 0.0f32;
    for position in 0..case.gamma {
        let row_start = position * TOP_K;
        let row = &actual_q[row_start..row_start + TOP_K];
        ensure!(
            row.iter()
                .all(|probability| probability.is_finite() && *probability >= 0.0),
            "{}: non-finite or negative q at row {position}",
            case.name
        );
        let row_sum = row.iter().sum::<f32>();
        ensure!(
            (row_sum - 1.0).abs() <= Q_SUM_TOLERANCE,
            "{}: q row {position} sums to {row_sum}",
            case.name
        );
        let replay_index = select_strict_f32(row, case.uniforms[position]);
        ensure!(
            actual_path[position] == actual_candidates[row_start + replay_index],
            "{}: row {position} token disagrees with exported q and f32 uniform",
            case.name
        );
        for candidate_index in 0..TOP_K {
            let candidate = actual_candidates[row_start + candidate_index];
            if case.forbidden_tokens.binary_search(&candidate).is_ok() {
                ensure!(
                    row[candidate_index] == 0.0,
                    "{}: hard-masked candidate {candidate} retained q mass",
                    case.name
                );
            }
            let absolute_error =
                (row[candidate_index] - expected_q[row_start + candidate_index]).abs();
            max_q_error = max_q_error.max(absolute_error);
            ensure!(
                absolute_error <= Q_ELEMENT_TOLERANCE,
                "{}: q mismatch at row {position}, candidate {candidate_index}: device={} host={} abs_error={absolute_error}",
                case.name,
                row[candidate_index],
                expected_q[row_start + candidate_index]
            );
        }
    }
    println!(
        "sampled selector case: PASS name={} gamma={} rank={} temperature={} max_q_abs_error={max_q_error:.9}",
        case.name, case.gamma, case.rank, case.temperature
    );
    Ok(max_q_error)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with qwen3.8-27b kernels")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernel = gpu
        .kernel("dflash2_selector", "dflash2_selector_walk_probabilistic")
        .context("resolve dflash2_selector_walk_probabilistic")?;
    ensure!(kernel.0 != 0, "sampled selector kernel handle is null");

    let cases = [
        random_case(0x0123_4567_89ab_cdef),
        random_case(0xa5a5_5a5a_f00d_cafe),
        random_case(0xd1f1_a520_0000_0003),
        tie_case(),
        extreme_case(),
        sparse_q_policy_case(),
        scalar_sampler_policy_case(),
    ];
    let mut max_q_error = 0.0f32;
    for case in &cases {
        max_q_error = max_q_error.max(run_case(gpu, stream, kernel, case)?);
    }
    println!(
        "DFLASH2_PROBABILISTIC_SELECTOR_RAW_PARITY=PASS cases={} max_q_abs_error={max_q_error:.9}",
        cases.len()
    );
    Ok(())
}
