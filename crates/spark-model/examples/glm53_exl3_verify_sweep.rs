// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 EXL3 verify-cost sweep: the 1-row walk against k-row exact verify.
//!
//! Phase A prefills a fixed prompt and greedy-decodes `generate` tokens one row at a
//! time, timing each walk. Phase B re-prefills the same prompt for every width
//! k and walks the recorded greedy path in k-row exact verify chunks, so every
//! chunk is a fully accepted DFlash2 step (anchor plus k-1 correct drafts) at
//! the real positions and routing. Each chunk's oracle must equal the walk's
//! next tokens: a mismatch means the verify is not bit-identical to the walk.
//!
//! Timing is host wall time around calls that end in a stream synchronize. It
//! covers the verify body only: no DSA snapshot, drafter, policy or readback.
//! Levers are process-start environment switches, so run one process per arm.

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use spark_model::model::glm53::Glm53Exl3Model;
use spark_model::traits::Model;
use spark_model::weight_loader::{
    Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights, admit_glm53_exl3_files, load_glm53_exl3_store,
    materialize_glm53_exl3_native,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

const RESERVE_BYTES: usize = 8 * 1024 * 1024 * 1024;
const CAPACITY: u32 = 2048;
/// A fixed id sequence (the K8 probe's context) so every arm sees the same prompt.
const PROMPT_SEED: [u32; 8] = [151_331, 963, 8_568, 14, 21, 8, 374, 372];

struct Args {
    root: std::path::PathBuf,
    arm: String,
    prompt_tokens: usize,
    generate: usize,
    widths: Vec<usize>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let usage = "usage: glm53_exl3_verify_sweep <exl3-dir> <arm-label> [prompt_tokens=256] \
                 [generate=160] [widths=2,3,4,5,6,7,8]";
    let root = it.next().context(usage)?.into();
    let arm = it.next().context(usage)?;
    let prompt_tokens = it.next().map_or(Ok(256), |v| v.parse()).context(usage)?;
    let generate = it.next().map_or(Ok(160), |v| v.parse()).context(usage)?;
    let widths = match it.next() {
        Some(list) => list
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<usize>, _>>()
            .context(usage)?,
        None => (2..=8).collect(),
    };
    ensure!(
        widths.iter().all(|k| (2..=8).contains(k)),
        "verify widths must be 2..=8"
    );
    ensure!(
        prompt_tokens >= PROMPT_SEED.len() && prompt_tokens + generate + 8 <= CAPACITY as usize,
        "prompt_tokens + generate must fit the {CAPACITY}-position probe capacity"
    );
    Ok(Args {
        root,
        arm,
        prompt_tokens,
        generate,
        widths,
    })
}

/// The seed repeated to length: a long, deterministic, in-vocabulary prompt.
fn prompt(len: usize) -> Vec<u32> {
    PROMPT_SEED.iter().copied().cycle().take(len).collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let files = admit_glm53_exl3_files(&args.root)?;
    let backend: std::sync::Arc<dyn GpuBackend> =
        std::sync::Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let gpu: &dyn GpuBackend = backend.as_ref();
    let store = match load_glm53_exl3_store(&files, gpu, RESERVE_BYTES) {
        Ok(store) => store,
        Err(error) => match error.retry_cleanup(gpu) {
            Ok(primary) => return Err(primary),
            Err(retained) => bail!("EXL3 load and cleanup failed: {retained}"),
        },
    };
    let catalog = Glm53Exl3TargetCatalog::new(&files, &store)?;
    let native =
        materialize_glm53_exl3_native(&catalog, gpu).map_err(|error| anyhow::anyhow!("{error}"))?;
    let weights = Glm53Exl3TargetWeights::new(&catalog, &native)?;
    drop(catalog);
    let model = Glm53Exl3Model::new(backend, store, native, weights, CAPACITY)?;
    let stream = model.gpu().default_stream();
    let prompt = prompt(args.prompt_tokens);

    // Phase A: the 1-row walk, which also records the greedy path.
    let mut seq = model.alloc_sequence()?;
    let logits = model.prefill(&prompt, &mut seq, stream)?;
    let mut path = vec![model.argmax_host(logits)?];
    let mut walk_ms = Vec::with_capacity(args.generate);
    for _ in 0..args.generate {
        let token = *path.last().unwrap();
        model.gpu().synchronize(stream)?;
        let started = Instant::now();
        let logits = model.decode_token(token, stream)?;
        walk_ms.push(started.elapsed().as_secs_f64() * 1e3);
        path.push(model.argmax_host(logits)?);
    }
    model.free_sequence(&mut seq)?;
    // Skip the first walks: graph capture and allocator warm-up land there.
    let warm = walk_ms.len().min(8);
    let t1 = median(&mut walk_ms[warm..].to_vec());
    println!(
        "{{\"arm\":\"{}\",\"kind\":\"walk\",\"k\":1,\"calls\":{},\"median_ms\":{t1:.3}}}",
        args.arm,
        walk_ms.len() - warm
    );

    // Phase B: k-row exact verify over the same greedy path.
    let mut total_mismatched = 0usize;
    for &k in &args.widths {
        let mut seq = model.alloc_sequence()?;
        model.prefill(&prompt, &mut seq, stream)?;
        let mut costs = Vec::new();
        let mut mismatched = 0usize;
        let mut position = 0usize;
        // path[p] is the anchor; path[p+1..p+k] are drafts; path[p+k] is the
        // walk's token after the chunk, which the last oracle row must equal.
        while position + k < path.len() {
            let block = &path[position..position + k];
            model.gpu().synchronize(stream)?;
            let started = Instant::now();
            let logits = model.verify_tokens_full_exact(block, stream)?;
            costs.push(started.elapsed().as_secs_f64() * 1e3);
            let oracle = model.argmax_rows_host(logits, k)?;
            if oracle[..] != path[position + 1..=position + k] {
                mismatched += 1;
            }
            position += k;
        }
        model.free_sequence(&mut seq)?;
        total_mismatched += mismatched;
        let calls = costs.len();
        let warm = calls.min(2);
        let vk = median(&mut costs[warm..].to_vec());
        println!(
            "{{\"arm\":\"{}\",\"kind\":\"verify\",\"k\":{k},\"calls\":{},\"median_ms\":{vk:.3},\
             \"ratio_to_walk\":{:.4},\"marginal_ms_per_row\":{:.3},\"oracle_mismatched_calls\":{mismatched}}}",
            args.arm,
            calls - warm,
            vk / t1,
            (vk - t1) / (k as f64 - 1.0),
        );
    }
    model.free()?;
    ensure!(
        total_mismatched == 0,
        "FAIL: {total_mismatched} verify chunk(s) mismatched 1-row walk oracle"
    );
    println!(
        "RESULT: PASS arm={} prompt_tokens={} generate={}",
        args.arm, args.prompt_tokens, args.generate
    );
    Ok(())
}
