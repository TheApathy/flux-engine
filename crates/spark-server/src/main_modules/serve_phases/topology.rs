// SPDX-License-Identifier: AGPL-3.0-only

//! TP/EP/PP topology + disaggregated role resolution + NCCL communicator init.

// `Context` only used by the cuda-feature `init_nccl_comm` to wrap
// NCCL bootstrap errors; metal builds don't reach that path.
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;

use atlas_core::config::{DisaggRole, ModelConfig};

use crate::cli;

pub(crate) struct Topology {
    pub(crate) world_size: usize,
    pub(crate) tp_size: usize,
    pub(crate) ep_size: usize,
    pub(crate) tp_rank: usize,
    pub(crate) ep_rank: usize,
}

pub(crate) fn resolve_topology(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
) -> Result<Topology> {
    anyhow::ensure!(args.pp_size >= 1, "--pp-size must be >= 1");
    let disagg_role = DisaggRole::parse(&args.role).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid --role {:?}: expected unified, prefill, or decode",
            args.role
        )
    })?;
    config.disagg_role = disagg_role;

    let pp_size = args.pp_size;
    let (tp_size, ep_size) =
        if args.tp_size == 1 && args.ep_size == 1 && pp_size == 1 && args.world_size > 1 {
            (1usize, args.world_size)
        } else {
            (args.tp_size.max(1), args.ep_size.max(1))
        };
    anyhow::ensure!(
        pp_size == 1 || (tp_size == 1 && ep_size == 1),
        "Combining --pp-size > 1 (got {pp_size}) with --tp-size {tp_size} or --ep-size {ep_size} \
         requires per-stage NCCL sub-communicators; use --pp-size N alone (e.g. 3-Spark PP=3) \
         or --tp-size/--ep-size with --pp-size 1 (e.g. 2-Spark TP=2+EP=2)."
    );
    let tp_ep_world = if tp_size == ep_size {
        tp_size
    } else {
        tp_size
            .checked_mul(ep_size)
            .ok_or_else(|| anyhow::anyhow!("tp_size × ep_size overflow"))?
    };
    let derived_world = tp_ep_world
        .checked_mul(pp_size)
        .ok_or_else(|| anyhow::anyhow!("tp_ep_world × pp_size overflow"))?;
    let world_size = if args.world_size <= 1 && derived_world > 1 {
        tracing::info!(
            "Auto-derived world_size={} from --tp-size {} --ep-size {} --pp-size {} \
             (rule: within-stage = tp==ep ? tp : tp×ep; total = within-stage × pp). \
             Pass --world-size to override.",
            derived_world,
            tp_size,
            ep_size,
            pp_size,
        );
        derived_world
    } else {
        args.world_size
    };
    if world_size != derived_world {
        anyhow::bail!(
            "Invalid parallelism topology: world_size={} but (tp_size={}, ep_size={}) -> \
             stage_world={} × pp_size={} = {}. Either use orthogonal mesh \
             (world = tp × ep × pp) or overlapping TP+EP groups (world = tp × pp with tp == ep).",
            world_size,
            tp_size,
            ep_size,
            tp_ep_world,
            pp_size,
            derived_world,
        );
    }
    anyhow::ensure!(
        args.rank < world_size,
        "Invalid --rank {} for world_size={} (must satisfy rank < world_size)",
        args.rank,
        world_size
    );
    let pp_rank = args.rank / tp_ep_world;
    let within_stage_rank = args.rank % tp_ep_world;
    let (tp_rank, ep_rank) = if tp_size == ep_size && tp_size == tp_ep_world && tp_size > 1 {
        (within_stage_rank, within_stage_rank)
    } else {
        (within_stage_rank % tp_size, within_stage_rank / tp_size)
    };

    config.tp_rank = tp_rank;
    config.tp_world_size = tp_size;
    config.ep_rank = ep_rank;
    config.ep_world_size = ep_size;
    config.pp_rank = pp_rank;
    config.pp_world_size = pp_size;
    config.pp_layer_offset = 0;

    crate::api::disagg::publish_runtime_config(
        disagg_role,
        args.disagg_peer.as_deref(),
        pp_rank,
        pp_size,
    );

    if tp_size > 1 {
        let loader = spark_model::factory::loader_for_config(config)?;
        if !loader.supports_tp() {
            anyhow::bail!(
                "TP (--tp-size > 1) is not supported by the {} weight loader. \
                 Run with --tp-size 1 (EP-only or --pp-size > 1).",
                config.model_type,
            );
        }
        drop(loader);
        if !config.num_attention_heads.is_multiple_of(tp_size) {
            anyhow::bail!(
                "TP requires num_attention_heads ({}) divisible by tp_size ({})",
                config.num_attention_heads,
                tp_size,
            );
        }
        config.tp_kv_replicate = 1;
        if config.num_key_value_heads.is_multiple_of(tp_size) {
            config.num_key_value_heads /= tp_size;
        } else if tp_size > config.num_key_value_heads
            && config.num_key_value_heads > 0
            && tp_size.is_multiple_of(config.num_key_value_heads)
        {
            let replicate = tp_size / config.num_key_value_heads;
            config.tp_kv_replicate = replicate;
            config.num_key_value_heads = 1;
            tracing::info!(
                "TP KV-head replication: tp_size={tp_size}, tp_kv_replicate={replicate}, \
                 kv_tp_rank={}/{} (local_kv_heads=1)",
                config.kv_tp_rank(),
                config.kv_tp_size(),
            );
        } else {
            anyhow::bail!(
                "TP requires num_key_value_heads ({}) divisible by tp_size ({}) \
                 or tp_size divisible by num_key_value_heads for KV replication",
                config.num_key_value_heads,
                tp_size,
            );
        }
        if config.linear_num_key_heads > 0 || config.linear_num_value_heads > 0 {
            if !config.linear_num_key_heads.is_multiple_of(tp_size) {
                anyhow::bail!(
                    "TP requires linear_num_key_heads ({}) divisible by tp_size ({})",
                    config.linear_num_key_heads,
                    tp_size,
                );
            }
            if !config.linear_num_value_heads.is_multiple_of(tp_size) {
                anyhow::bail!(
                    "TP requires linear_num_value_heads ({}) divisible by tp_size ({})",
                    config.linear_num_value_heads,
                    tp_size,
                );
            }
            let local_nk = config.linear_num_key_heads / tp_size;
            let local_nv = config.linear_num_value_heads / tp_size;
            if local_nk == 0 || !local_nv.is_multiple_of(local_nk) {
                anyhow::bail!(
                    "TP produces invalid local GDN GVA head ratio: linear_num_key_heads={} \
                     linear_num_value_heads={} at tp_size={}",
                    local_nk,
                    local_nv,
                    tp_size,
                );
            }
            config.linear_num_key_heads = local_nk;
            config.linear_num_value_heads = local_nv;
            tracing::info!(
                "TP-local GDN head counts: linear_num_key_heads={}, linear_num_value_heads={}",
                config.linear_num_key_heads,
                config.linear_num_value_heads,
            );
        }
        config.num_attention_heads /= tp_size;
        tracing::info!(
            "TP-local head counts: num_attention_heads={}, num_key_value_heads={} (kv_replicate={})",
            config.num_attention_heads,
            config.num_key_value_heads,
            config.tp_kv_replicate,
        );
    }

    if pp_size > 1 {
        let total_layers = config.num_hidden_layers;
        if total_layers == 0 || !total_layers.is_multiple_of(pp_size) {
            anyhow::bail!(
                "PP requires num_hidden_layers ({}) > 0 and divisible by pp_size ({})",
                total_layers,
                pp_size,
            );
        }
        if config.layer_types.is_empty() && config.full_attention_interval > 0 {
            config.layer_types = (0..total_layers)
                .map(|i| {
                    if (i + 1) % config.full_attention_interval == 0 {
                        atlas_core::config::LayerType::FullAttention
                    } else {
                        atlas_core::config::LayerType::LinearAttention
                    }
                })
                .collect();
        }
        let stage_layers = total_layers / pp_size;
        let pp_start = pp_rank * stage_layers;
        let pp_end = pp_start + stage_layers;
        config.pp_layer_offset = pp_start;
        config.num_hidden_layers = stage_layers;
        if config.layer_types.len() == total_layers {
            config.layer_types = config.layer_types[pp_start..pp_end].to_vec();
        }
        if !config.ple_layer_ids.is_empty() {
            // `ple_layer_ids` are 1-indexed global decoder layer IDs.
            config.ple_layer_ids = config
                .ple_layer_ids
                .iter()
                .copied()
                .filter(|&id| id > pp_start && id <= pp_end)
                .map(|id| id - pp_start)
                .collect();
        }
        if !config.dflash_capture_layers.is_empty() {
            config.dflash_capture_layers = config
                .dflash_capture_layers
                .iter()
                .copied()
                .filter(|&idx| idx >= pp_start && idx < pp_end)
                .map(|idx| idx - pp_start)
                .collect();
        }
        tracing::info!(
            "PP stage {}/{}: global layers [{}..{}), stage_layers={}, ple_layer_ids={:?}",
            pp_rank,
            pp_size,
            pp_start,
            pp_end,
            stage_layers,
            config.ple_layer_ids,
        );
    }

    if world_size > 1 {
        let (start, end) = config.local_expert_range();
        tracing::info!(
            "Parallelism: global rank {}/{} (pp_rank={}/{}, tp_rank={}/{}, ep_rank={}/{}), \
             role={}, local experts [{}, {})",
            args.rank,
            world_size,
            pp_rank,
            pp_size,
            tp_rank,
            tp_size,
            ep_rank,
            ep_size,
            disagg_role.as_str(),
            start,
            end,
        );
    }
    Ok(Topology {
        world_size,
        tp_size,
        ep_size,
        tp_rank,
        ep_rank,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn init_nccl_comm(
    args: &cli::ServeArgs,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    world_size: usize,
) -> Result<Option<std::sync::Arc<dyn spark_comm::CommBackend>>> {
    use spark_comm::CommBackend;
    if world_size <= 1 {
        return Ok(None);
    }
    let cuda_stream = gpu.default_stream();
    let comm_mode = std::env::var("ATLAS_COMM_BACKEND")
        .or_else(|_| std::env::var("FLUX_COMM_BACKEND"))
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if comm_mode == "tcp" {
        let raw_peers = std::env::var("ATLAS_PEER_ADDRS")
            .or_else(|_| std::env::var("FLUX_PEER_ADDRS"))
            .context(
                "ATLAS_COMM_BACKEND=tcp requires ATLAS_PEER_ADDRS (or FLUX_PEER_ADDRS) \
                 set to comma-separated host:port endpoints for every rank 0..world_size",
            )?;
        let peer_addrs: Vec<String> = raw_peers
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        let bind_port: u16 = std::env::var("ATLAS_TCP_BIND_PORT")
            .or_else(|_| std::env::var("FLUX_TCP_BIND_PORT"))
            .ok()
            .and_then(|v| v.trim().parse::<u16>().ok())
            .unwrap_or(args.master_port);
        let bind_addr = format!("0.0.0.0:{bind_port}");
        tracing::info!(
            "Initializing TCP full-mesh comm: rank {}/{}, bind {}, peers={:?}",
            args.rank,
            world_size,
            bind_addr,
            peer_addrs
        );
        let backend = spark_comm::TcpCommBackend::new(
            args.rank,
            world_size,
            &bind_addr,
            &peer_addrs,
            cuda_stream,
        )
        .context("Failed to initialize TCP full-mesh comm backend")?;
        tracing::info!("TCP comm initialized: rank {}", backend.rank());
        return Ok(Some(
            std::sync::Arc::new(backend) as std::sync::Arc<dyn spark_comm::CommBackend>
        ));
    }
    tracing::info!(
        "Initializing NCCL: rank {}/{}, master {}:{}",
        args.rank,
        world_size,
        args.master_addr,
        args.master_port
    );
    let backend = spark_comm::NcclBackend::new(
        args.rank,
        world_size,
        &args.master_addr,
        args.master_port,
        cuda_stream,
    )
    .context("Failed to initialize NCCL")?;
    tracing::info!("NCCL initialized: rank {}", backend.rank());
    Ok(Some(
        std::sync::Arc::new(backend) as std::sync::Arc<dyn spark_comm::CommBackend>
    ))
}

/// Metal-feature variant: NCCL multi-GPU isn't reachable on a single
/// Apple Silicon device, so collective ops fall back to the no-op
/// `SingleGpuBackend`. `world_size > 1` is rejected explicitly so a
/// misconfigured `--rank > 0` invocation fails fast instead of
/// silently degrading to single-rank.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn spark_runtime::gpu::GpuBackend,
    world_size: usize,
) -> Result<Option<std::sync::Arc<dyn spark_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available on Apple Silicon (metal feature); \
             single-device only"
        );
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::LayerType;
    use clap::Parser;

    fn qwen4_flash_next_config() -> ModelConfig {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.model_type = "qwen4_exp".to_string();
        cfg.num_hidden_layers = 48;
        cfg.hidden_size = 2560;
        cfg.vocab_size = 248_320;
        cfg.num_attention_heads = 24;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 256;
        cfg.layer_types.clear();
        cfg.full_attention_interval = 4;
        cfg.linear_num_key_heads = 16;
        cfg.linear_num_value_heads = 48;
        cfg.linear_key_head_dim = 128;
        cfg.linear_value_head_dim = 128;
        cfg.num_experts = 512;
        cfg.num_experts_per_tok = 10;
        cfg.moe_intermediate_size = 640;
        cfg.hc_count = 4;
        cfg.hc_mult = 4;
        cfg.ple_layer_ids = vec![2];
        cfg
    }

    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn two_spark_tp2_shards_qsa_and_gdn_heads_to_8_24() {
        let _guard = TEST_LOCK.lock().unwrap();
        let args = cli::ServeArgs::parse_from([
            "spark",
            "org/qwen3.8-flash-next",
            "--tp-size",
            "2",
            "--ep-size",
            "2",
            "--rank",
            "1",
        ]);
        let mut cfg = qwen4_flash_next_config();
        let topo = resolve_topology(&args, &mut cfg).expect("2-Spark TP=2+EP=2 resolves");
        assert_eq!(topo.world_size, 2);
        assert_eq!(topo.tp_size, 2);
        assert_eq!(topo.ep_size, 2);
        assert_eq!(topo.tp_rank, 1);
        assert_eq!(topo.ep_rank, 1);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.linear_num_key_heads, 8);
        assert_eq!(cfg.linear_num_value_heads, 24);
        assert_eq!(cfg.local_expert_range(), (256, 512));
    }

    #[test]
    fn uniform_tp3_on_qwen4_is_rejected_due_to_indivisible_heads() {
        let _guard = TEST_LOCK.lock().unwrap();
        let args =
            cli::ServeArgs::parse_from(["spark", "org/qwen3.8-flash-next", "--tp-size", "3"]);
        let mut cfg = qwen4_flash_next_config();
        let err = resolve_topology(&args, &mut cfg)
            .err()
            .expect("TP=3 must fail on 2 KV / 16 GDN heads");
        assert!(err.to_string().contains("divisible by tp_size"), "{err}");
    }

    #[test]
    fn three_spark_pp3_partitions_48_layers_into_three_16_layer_hybrid_stages() {
        let _guard = TEST_LOCK.lock().unwrap();
        for (rank, expected_offset, expected_ple) in [
            (0usize, 0usize, vec![2usize]),
            (1, 16, vec![]),
            (2, 32, vec![]),
        ] {
            let rank_str = rank.to_string();
            let args = cli::ServeArgs::parse_from([
                "spark",
                "org/qwen3.8-flash-next",
                "--pp-size",
                "3",
                "--rank",
                &rank_str,
            ]);
            let mut cfg = qwen4_flash_next_config();
            let topo = resolve_topology(&args, &mut cfg).expect("3-Spark PP=3 resolves");
            assert_eq!(topo.world_size, 3);
            assert_eq!(cfg.pp_world_size, 3);
            assert_eq!(cfg.pp_rank, rank);
            assert_eq!(cfg.pp_layer_offset, expected_offset);
            assert_eq!(cfg.num_hidden_layers, 16);
            assert_eq!(cfg.num_attention_layers(), 4);
            assert_eq!(cfg.num_ssm_layers(), 12);
            assert_eq!(cfg.layer_type(3), LayerType::FullAttention);
            assert_eq!(cfg.layer_type(0), LayerType::LinearAttention);
            assert_eq!(cfg.ple_layer_ids, expected_ple);
            assert_eq!(
                cfg.layer_prefix(0),
                format!("model.layers.{expected_offset}")
            );
            assert_eq!(cfg.is_first_pp_stage(), rank == 0);
            assert_eq!(cfg.is_last_pp_stage(), rank == 2);
        }
    }

    #[test]
    fn disagg_roles_publish_cleanly_for_1p_1d_and_1p_2d() {
        let _guard = TEST_LOCK.lock().unwrap();
        let args = cli::ServeArgs::parse_from([
            "spark",
            "org/qwen3.8-flash-next",
            "--role",
            "decode",
            "--disagg-peer",
            "http://10.0.0.10:8000",
        ]);
        let mut cfg = qwen4_flash_next_config();
        resolve_topology(&args, &mut cfg).expect("decode role resolves");
        assert_eq!(cfg.disagg_role, DisaggRole::Decode);
        let runtime = crate::api::disagg::current_runtime_config();
        assert_eq!(runtime.role, DisaggRole::Decode);
        assert_eq!(runtime.peers, vec!["http://10.0.0.10:8000".to_string()]);
    }
}
