// SPDX-License-Identifier: AGPL-3.0-only
//! Drop gate for the served DeepSeek-V4.1 model (TUI model swap): build `Dsv41Model`, run one
//! short prefill so a sequence exists, DROP it, and require that at least 95% of the memory the
//! model took comes back in MemAvailable.
//!
//! `--control leak` `mem::forget`s the model instead. It must FAIL: that proves the gate can see
//! a leak (the model's device memory is unified memory on GB10, so it is visible to the host).
//!
//! ```text
//! ATLAS_DSV41_PACKED_KEEP=6 cargo run -p spark-model --release --example dsv41_drop_gate \
//!   --features cuda,gpu-examples -- [--control leak]
//! ```

use anyhow::{Context, Result, bail, ensure};
use std::path::Path;
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::model::dsv41::Dsv41Model;
use spark_model::traits::Model;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR_ENV: &str = "FLUX_DSV41_MODEL_DIR";

fn resolve_model_dir() -> Result<String> {
    std::env::var(MODEL_DIR_ENV)
        .with_context(|| format!("set {MODEL_DIR_ENV} to the DeepSeek-V4.1 checkpoint directory"))
}

fn mem_available_kb() -> Result<u64> {
    let s = std::fs::read_to_string("/proc/meminfo")?;
    s.lines()
        .find_map(|l| {
            l.strip_prefix("MemAvailable:")
                .map(|r| r.trim().trim_end_matches(" kB").trim().parse::<u64>())
        })
        .context("no MemAvailable")?
        .map_err(anyhow::Error::from)
}

fn settle() -> Result<u64> {
    // MemAvailable moves for a moment after large frees; take the max of a few samples.
    let mut best = 0;
    for _ in 0..10 {
        best = best.max(mem_available_kb()?);
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    Ok(best)
}

fn main() -> Result<()> {
    let leak = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--control") => match std::env::args().nth(2).as_deref() {
            Some("leak") => true,
            o => bail!("--control leak, got {o:?}"),
        },
        Some(o) => bail!("unknown argument {o}"),
    };
    let model_dir = resolve_model_dir()?;
    let config = parse_config(&std::fs::read_to_string(format!(
        "{model_dir}/config.json"
    ))?)?;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    // A second handle on the same device, kept outside the model, to read cudaMemGetInfo after
    // the model (and its backend) are gone. MemAvailable includes page cache that moves on its
    // own; the device-side free number is what says the allocations were released.
    let probe = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(
        spark_model::weight_loader::deepseek_v41::skip_tensor_for_serving,
    ));
    let store = loader.load(Path::new(&model_dir), &gpu, 0)?;
    let before = settle()?;
    let dev_before = probe.free_memory()?;
    println!(
        "MemAvailable before model: {:.2} GiB (store resident: {:.2} GiB)",
        before as f64 / 1_048_576.0,
        store.total_bytes() as f64 / 1_073_741_824.0
    );

    let model = Dsv41Model::new(
        &config,
        &store,
        std::sync::Arc::new(gpu),
        Path::new(&model_dir),
        8192,
        512,
    )?;
    let mut seq = model.alloc_sequence()?;
    let ids: Vec<u32> = (0..64).map(|i| 1000 + i).collect();
    model.prefill(&ids, &mut seq, 0)?;
    let loaded = settle()?;
    let dev_loaded = probe.free_memory()?;
    let took = before.saturating_sub(loaded);
    println!(
        "MemAvailable after load + 1 prefill: {:.2} GiB (model took {:.2} GiB)",
        loaded as f64 / 1_048_576.0,
        took as f64 / 1_048_576.0
    );
    ensure!(
        took > 1024 * 1024,
        "the model took under 1 GiB — the gate would be meaningless"
    );

    if leak {
        println!("[control leak] mem::forget(model): the gate MUST FAIL");
        std::mem::forget(seq);
        std::mem::forget(model);
    } else {
        model.free_sequence(&mut seq)?;
        drop(model);
    }
    let after = settle()?;
    let dev_after = probe.free_memory()?;
    let dev_took = dev_before.saturating_sub(dev_loaded);
    let dev_back = dev_after.saturating_sub(dev_loaded);
    let dev_frac = dev_back as f64 / dev_took.max(1) as f64;
    println!(
        "cudaMemGetInfo free: before {:.2} / loaded {:.2} / after drop {:.2} GiB -> device returned {:.1}% of {:.2} GiB",
        dev_before as f64 / 1_073_741_824.0,
        dev_loaded as f64 / 1_073_741_824.0,
        dev_after as f64 / 1_073_741_824.0,
        dev_frac * 100.0,
        dev_took as f64 / 1_073_741_824.0
    );
    let back = after.saturating_sub(loaded);
    let frac = back as f64 / took as f64;
    println!(
        "MemAvailable after drop: {:.2} GiB (returned {:.2} GiB = {:.1}% of what the model took)",
        after as f64 / 1_048_576.0,
        back as f64 / 1_048_576.0,
        frac * 100.0
    );
    let passed = frac >= 0.95 && dev_frac >= 0.95;
    if passed {
        println!(
            "DROP GATE PASS{}",
            if leak {
                " — BUT THIS WAS THE LEAK CONTROL: THE GATE CANNOT FAIL"
            } else {
                ""
            }
        );
    } else {
        println!(
            "DROP GATE FAIL{}",
            if leak {
                " (expected: leak control)"
            } else {
                ""
            }
        );
    }
    drop(store);
    ensure!(
        if leak { !passed } else { passed },
        "DROP GATE FAIL (leak={leak}, frac={frac:.3}, dev_frac={dev_frac:.3})"
    );
    Ok(())
}
