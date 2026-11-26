# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only unit tests for discrete SM120 (RTX 50-series / RTX PRO 5000-6000, 1x/2x/4x)
and SM103 (B300 / GB300) hardware detection, VRAM budgeting, single-host multi-GPU
launch snippet generation, container tag selection, and docker/sm120/Dockerfile hygiene."""

import importlib.machinery
import importlib.util
import io
import json
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase, mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"
SM120_DOCKERFILE = REPO_ROOT / "docker" / "sm120" / "Dockerfile"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader(
        "flux_autodetect_sm120_sm103_test", str(SCRIPT)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"could not import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


fa = _load_flux_autodetect()


def _arg_value(command, flag):
    return command[command.index(flag) + 1]


def _make_hw_via_smi(smi_output: str, host_ram_gib: float = 256.0):
    """Run real detect_hardware_topology() against mocked nvidia-smi and /proc/meminfo."""
    meminfo = (
        f"MemTotal:       {int(host_ram_gib * 1024 * 1024)} kB\n"
        f"MemAvailable:   {int((host_ram_gib - 16.0) * 1024 * 1024)} kB\n"
        f"Buffers:        1048576 kB\n"
        f"Cached:         4194304 kB\n"
    )
    with (
        mock.patch.object(fa.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        mock.patch.object(fa.subprocess, "check_output", return_value=smi_output),
        mock.patch("builtins.open", mock.mock_open(read_data=meminfo)),
    ):
        return fa.detect_hardware_topology()


def _flash_next_model(*, with_dflash: bool = True):
    spec = (
        {
            "mode": "dflash",
            "drafter_path": "/models/Qwen3.8-Flash-Next-DFlash2",
            "gamma": 7,
            "block_size": 8,
        }
        if with_dflash
        else {"mode": "none", "drafter_path": None, "gamma": 0}
    )
    return {
        "kernel_target": "qwen3.8-flash-next",
        "selected_binary": "/opt/flux/bin/flux-server-real",
        "path": "/models/Qwen3.8-Flash-Next-NVFP4",
        "name": "Qwen3.8-Flash-Next-NVFP4",
        "family": "Qwen3.8-Flash-Next Hybrid DeltaNet-MoE (QSA / PLE)",
        "quantization": "NVFP4",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "ple_dir": "/models/Qwen3.8-Flash-Next-NVFP4/ple-offload",
        "speculative": spec,
        "requires_qwen4_qsa": True,
        "recommended_max_seq_len": 262144,
        "max_position_embeddings": 262144,
        "recommended_tool_call_parser": "qwen3_coder",
        "is_drafter": False,
        "broken_symlinks": [],
        "status": "ready",
    }


def _qwen27b_model(*, with_dflash: bool = True):
    spec = (
        {
            "mode": "dflash",
            "drafter_path": "/models/Qwen3.8-27B-DFlash2",
            "gamma": 7,
            "block_size": 8,
        }
        if with_dflash
        else {"mode": "none", "drafter_path": None, "gamma": 0}
    )
    return {
        "kernel_target": "qwen3.8-27b",
        "selected_binary": "/opt/flux/bin/flux-server-universal",
        "path": "/models/Qwen3.8-27B-NVFP4",
        "name": "Qwen3.8-27B-NVFP4",
        "family": "Qwen3.8-27B Dense Hybrid Gated-DeltaNet",
        "quantization": "NVFP4",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_experts": 0,
        "ple_dir": None,
        "speculative": spec,
        "requires_qwen4_qsa": False,
        "recommended_max_seq_len": 262144,
        "max_position_embeddings": 262144,
        "recommended_tool_call_parser": "qwen3_coder",
        "is_drafter": False,
        "broken_symlinks": [],
        "status": "ready",
    }


def _small_14b_nvfp4_model():
    return {
        "kernel_target": "qwen3.6-35b-a3b",
        "selected_binary": "/opt/flux/bin/flux-server-universal",
        "path": "/models/Qwen3-14B-NVFP4",
        "name": "Qwen3-14B-NVFP4",
        "family": "Auto-Mapped (Qwen3ForCausalLM)",
        "quantization": "NVFP4",
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_experts": 0,
        "vocab_size": 151936,
        "ple_dir": None,
        "speculative": {"mode": "none", "drafter_path": None, "gamma": 0},
        "requires_qwen4_qsa": False,
        "recommended_max_seq_len": 32768,
        "max_position_embeddings": 131072,
        "recommended_tool_call_parser": "hermes",
        "status": "ready",
    }


def _dsv41_model():
    return {
        "kernel_target": "deepseek-v4.1",
        "selected_binary": "/opt/flux/bin/flux-server-universal",
        "path": "/models/DeepSeek-V4.1-Flash-Next",
        "name": "DeepSeek-V4.1-Flash-Next",
        "family": "DeepSeek-V4.1 Flash-Next (CB3 K124 MoE)",
        "quantization": "NVFP4",
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_experts": 128,
        "ple_dir": None,
        "speculative": {"mode": "none", "drafter_path": None, "gamma": 0},
        "requires_qwen4_qsa": False,
        "recommended_max_seq_len": 65536,
        "max_position_embeddings": 131072,
        "recommended_tool_call_parser": "dsml_v41",
        "status": "ready",
    }


def _glm53_model():
    return {
        "kernel_target": "glm5.3-flash",
        "selected_binary": "/opt/flux/bin/flux-server-glm53",
        "path": "/models/GLM-5.3-Flash-EXL3",
        "name": "GLM-5.3-Flash-EXL3",
        "family": "GLM-4.7 / GLM-5.3 MoE",
        "quantization": "EXL3",
        "hidden_size": 4096,
        "num_hidden_layers": 45,
        "num_experts": 128,
        "ple_dir": None,
        "speculative": {"mode": "none", "drafter_path": None, "gamma": 0},
        "requires_qwen4_qsa": False,
        "recommended_max_seq_len": 36864,
        "max_position_embeddings": 131072,
        "recommended_tool_call_parser": "glm_xml",
        "status": "ready",
    }


def _qwen80b_a3b_model():
    return {
        "kernel_target": "qwen3-next-80b-a3b",
        "selected_binary": "/opt/flux/bin/flux-server-universal",
        "path": "/models/Qwen3-Next-80B-A3B-NVFP4",
        "name": "Qwen3-Next-80B-A3B-NVFP4",
        "family": "Qwen3-Next-80B-A3B Hybrid MoE",
        "quantization": "NVFP4",
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "ple_dir": None,
        "speculative": {"mode": "none", "drafter_path": None, "gamma": 0},
        "requires_qwen4_qsa": False,
        "recommended_max_seq_len": 131072,
        "max_position_embeddings": 262144,
        "recommended_tool_call_parser": "hermes",
        "status": "ready",
    }


def _plan_on_hw(model_info, hw, *, mode="auto", env=None, extra_args=None):
    with (
        mock.patch.dict(os.environ, env or {}, clear=True),
        mock.patch.object(fa, "detect_hardware_topology", return_value=hw),
        mock.patch.object(
            fa,
            "detect_intranode_nccl_env",
            return_value=(
                {
                    "NCCL_P2P_DISABLE": "0",
                    "NCCL_P2P_LEVEL": "SYS",
                    "NCCL_SHM_DISABLE": "0",
                    "NCCL_IB_DISABLE": "1",
                },
                "pcie-p2p-shm",
            ),
        ),
    ):
        return fa.build_launch_plan(model_info, mode, list(extra_args or []))


class HardwareDetectionSM120SM103Tests(TestCase):
    """detect_hardware_topology accurately separates GB10 SM121a UMA,
    discrete SM120a (1x/2x/4x), SM103a (B300/GB300), and legacy GPUs."""

    def test_gb10_sm121a_uma_detection(self):
        hw = _make_hw_via_smi("NVIDIA GB10, 128000, 12.1\n", host_ram_gib=128.0)
        self.assertEqual(hw["sm_family"], "gb10-sm121a")
        self.assertEqual(hw["arch_target"], "sm_121a")
        self.assertTrue(hw["unified_memory_architecture"])
        self.assertFalse(hw["single_host_multi_gpu"])
        self.assertEqual(hw["container_image"], "ghcr.io/theapathy/flux-engine:gb10")
        self.assertEqual(hw["hardware_recipe"], "flux-nextgen-max-prefill-decode.env")

        # GB10 with [N/A] memory.total falls back to UMA host RAM
        hw_na = _make_hw_via_smi("NVIDIA GB10, [N/A], 12.1\n", host_ram_gib=128.0)
        self.assertEqual(hw_na["sm_family"], "gb10-sm121a")
        self.assertTrue(hw_na["unified_memory_architecture"])
        self.assertEqual(hw_na["effective_memory_gib"], 128.0)

    def test_discrete_sm120a_single_gpu_tiers_never_use_host_ram(self):
        cases = (
            ("NVIDIA GeForce RTX 5060 Ti, 16311, 12.0\n", 15.9, "1x-sm120-16gb"),
            ("NVIDIA GeForce RTX 5070 Ti, 16303, 12.0\n", 15.9, "1x-sm120-16gb"),
            ("NVIDIA GeForce RTX 5080, 16303, 12.0\n", 15.9, "1x-sm120-16gb"),
            ("NVIDIA GeForce RTX 5090, 32607, 12.0\n", 31.8, "1x-sm120-32gb"),
            ("NVIDIA RTX PRO 5000 Blackwell, 49140, 12.0\n", 48.0, "1x-sm120-48gb"),
            ("NVIDIA RTX PRO 5000 Blackwell, 73728, 12.0\n", 72.0, "1x-sm120-72gb"),
            ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 12.0\n", 95.6, "1x-sm120-96gb"),
        )
        for smi_line, exp_gib, exp_tier in cases:
            with self.subTest(tier=exp_tier):
                # Deliberately give the host 512 GiB RAM to verify discrete VRAM never mixes with host RAM.
                hw = _make_hw_via_smi(smi_line, host_ram_gib=512.0)
                self.assertEqual(hw["sm_family"], "blackwell-sm120a")
                self.assertEqual(hw["arch_target"], "sm_120a")
                self.assertFalse(hw["unified_memory_architecture"])
                self.assertFalse(hw["single_host_multi_gpu"])
                self.assertEqual(hw["gpu_count"], 1)
                self.assertAlmostEqual(hw["per_gpu_vram_gib"], exp_gib, places=1)
                self.assertAlmostEqual(hw["total_vram_gib"], exp_gib, places=1)
                self.assertAlmostEqual(hw["effective_memory_gib"], exp_gib, places=1)
                self.assertEqual(hw["gpu_tier"], exp_tier)
                self.assertEqual(hw["container_image"], "ghcr.io/theapathy/flux-engine:sm120")

    def test_discrete_sm120a_multi_gpu_single_host_detection(self):
        smi_2x_5090 = "NVIDIA GeForce RTX 5090, 32607, 12.0\n" * 2
        hw2 = _make_hw_via_smi(smi_2x_5090, host_ram_gib=256.0)
        self.assertEqual(hw2["sm_family"], "blackwell-sm120a")
        self.assertFalse(hw2["unified_memory_architecture"])
        self.assertTrue(hw2["single_host_multi_gpu"])
        self.assertEqual(hw2["gpu_count"], 2)
        self.assertEqual(hw2["vram_per_gpu_gib"], [31.8, 31.8])
        self.assertEqual(hw2["per_gpu_vram_gib"], 31.8)
        self.assertAlmostEqual(hw2["total_vram_gib"], 63.7, places=1)
        self.assertEqual(hw2["recommended_tp_size"], 2)
        self.assertEqual(hw2["container_image"], "ghcr.io/theapathy/flux-engine:sm120")

        smi_4x_6000 = "NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 4
        hw4 = _make_hw_via_smi(smi_4x_6000, host_ram_gib=512.0)
        self.assertEqual(hw4["sm_family"], "blackwell-sm120a")
        self.assertFalse(hw4["unified_memory_architecture"])
        self.assertTrue(hw4["single_host_multi_gpu"])
        self.assertEqual(hw4["gpu_count"], 4)
        self.assertEqual(hw4["vram_per_gpu_gib"], [95.6, 95.6, 95.6, 95.6])
        self.assertEqual(hw4["per_gpu_vram_gib"], 95.6)
        self.assertAlmostEqual(hw4["total_vram_gib"], 382.4, places=1)
        self.assertEqual(hw4["recommended_tp_size"], 4)
        self.assertEqual(hw4["container_image"], "ghcr.io/theapathy/flux-engine:sm120")

    def test_sm103a_b300_and_gb300_detection(self):
        for smi_line, exp_gib in (
            ("NVIDIA B300 SXM6 AC, 294912, 10.3\n", 288.0),
            ("NVIDIA GB300, 275456, 10.3\n", 269.0),
        ):
            with self.subTest(smi=smi_line.strip()):
                hw = _make_hw_via_smi(smi_line, host_ram_gib=512.0)
                self.assertEqual(hw["sm_family"], "blackwell-sm103a")
                self.assertEqual(hw["arch_target"], "sm_103a")
                self.assertFalse(hw["unified_memory_architecture"])
                self.assertEqual(hw["gpu_count"], 1)
                self.assertAlmostEqual(hw["per_gpu_vram_gib"], exp_gib, places=1)
                self.assertAlmostEqual(hw["total_vram_gib"], exp_gib, places=1)
                self.assertEqual(hw["container_image"], "ghcr.io/theapathy/flux-engine:sm103")
                self.assertEqual(hw["hardware_max_seq_len_fp8_kv"], 1048576)

    def test_other_gpu_fallback_detection(self):
        hw_h100 = _make_hw_via_smi("NVIDIA H100 80GB HBM3, 81559, 9.0\n", host_ram_gib=256.0)
        self.assertEqual(hw_h100["sm_family"], "hopper-sm90")
        self.assertFalse(hw_h100["unified_memory_architecture"])
        self.assertEqual(hw_h100["hardware_recipe"], "nvidia-hopper-sm90.env")

        hw_4090 = _make_hw_via_smi("NVIDIA GeForce RTX 4090, 24564, 8.9\n", host_ram_gib=128.0)
        self.assertEqual(hw_4090["sm_family"], "cuda-sm89")
        self.assertFalse(hw_4090["unified_memory_architecture"])
        self.assertAlmostEqual(hw_4090["per_gpu_vram_gib"], 24.0, places=1)


class DiscreteVramBudgetAndFitTests(TestCase):
    """Per-tier discrete SM120 / SM103 VRAM fit, refusal, and KV-pool sizing."""

    def test_1x_16gb_fits_small_14b_and_refuses_27b_and_80b_cleanly(self):
        hw_16 = _make_hw_via_smi("NVIDIA GeForce RTX 5060 Ti, 16311, 12.0\n", host_ram_gib=256.0)

        # 14B NVFP4 fits cleanly on 16 GB with healthy KV pool and 1024 MB OOM guard
        plan_14b = _plan_on_hw(_small_14b_nvfp4_model(), hw_16)
        self.assertTrue(plan_14b["vram_budget"]["fits"])
        self.assertGreater(plan_14b["vram_budget"]["free_vram_for_kv_per_gpu_gib"], 2.5)
        self.assertEqual(plan_14b["container_image"], "ghcr.io/theapathy/flux-engine:sm120")
        self.assertEqual(_arg_value(plan_14b["command"], "--oom-guard-mb"), "1024")
        self.assertEqual(_arg_value(plan_14b["command"], "--gpu-memory-utilization"), "0.90")

        # 27B NVFP4 (~15.5 GB + overhead) exceeds 14.3 GiB usable on 16 GB -> clean refusal
        with self.assertRaisesRegex(ValueError, r"Insufficient discrete VRAM.*16 GB"):
            _plan_on_hw(_qwen27b_model(with_dflash=False), hw_16)

        # 80B / Flash-Next exceeds 16 GB -> clean refusal
        with self.assertRaisesRegex(ValueError, r"Insufficient discrete VRAM"):
            _plan_on_hw(_flash_next_model(), hw_16)
        with self.assertRaisesRegex(ValueError, r"Insufficient discrete VRAM"):
            _plan_on_hw(_qwen80b_a3b_model(), hw_16)

    def test_1x_32gb_rtx5090_fits_qwen27b_with_10gb_plus_free_and_refuses_flash_next(self):
        hw_32 = _make_hw_via_smi("NVIDIA GeForce RTX 5090, 32607, 12.0\n", host_ram_gib=128.0)

        plan_27b = _plan_on_hw(_qwen27b_model(with_dflash=True), hw_32)
        vb = plan_27b["vram_budget"]
        self.assertTrue(vb["fits"])
        self.assertEqual(vb["model_weights_per_gpu_gib"], 15.5)
        # Even with 32607 MiB (31.8 GiB) and DFlash2 (+1.0 GiB), >= 10.0 GiB remains for KV!
        self.assertGreaterEqual(vb["free_vram_for_kv_per_gpu_gib"], 10.0)
        self.assertGreaterEqual(vb["recommended_kv_pool_tokens"], 262144)
        self.assertEqual(plan_27b["max_seq_len"], 262144)
        self.assertEqual(plan_27b["gpu_memory_utilization"], 0.9)
        self.assertEqual(plan_27b["container_image"], "ghcr.io/theapathy/flux-engine:sm120")
        # Non-Flash-Next profile must not inherit Qwen4-specific overlay vars
        self.assertNotIn("ATLAS_QWEN4_PREFILL_WIDE", plan_27b["env"])

        with self.assertRaisesRegex(ValueError, r"Insufficient discrete VRAM"):
            _plan_on_hw(_flash_next_model(), hw_32)

    def test_1x_48gb_and_72gb_rtx_pro_5000_27b_massive_kv_and_flash_next_notes_2x(self):
        for vram_mib, min_pool in ((49140, 524288), (73728, 1048576)):
            with self.subTest(vram_mib=vram_mib):
                hw = _make_hw_via_smi(f"NVIDIA RTX PRO 5000 Blackwell, {vram_mib}, 12.0\n", host_ram_gib=256.0)
                plan_27b = _plan_on_hw(_qwen27b_model(with_dflash=True), hw)
                vb = plan_27b["vram_budget"]
                self.assertTrue(vb["fits"])
                self.assertGreaterEqual(vb["recommended_kv_pool_tokens"], min_pool)
                self.assertGreaterEqual(plan_27b["max_seq_len"], 262144)

                # 80B / Flash-Next on 1x 48/72 GB cleanly refuses and notes 2x GPUs needed
                with self.assertRaisesRegex(ValueError, r"Requires 2x (48|72) GB GPUs \(TP=2 \+ EP=2\)"):
                    _plan_on_hw(_flash_next_model(), hw)

                # Verify recommendation helper also notes 2x needed for Flash-Next
                recs = fa.recommend_models_for_hardware(hw)
                self.assertTrue(any("2x" in n for n in recs["notes"]))

    def test_1x_96gb_rtx_pro_6000_fits_flash_next_with_dflash2_and_healthy_kv_pool(self):
        # Test with low host RAM (32 GB) to prove discrete 96 GB does NOT get clamped by GB10 tight_uma!
        hw_96 = _make_hw_via_smi(
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 12.0\n",
            host_ram_gib=32.0,
        )
        self.assertEqual(hw_96["hardware_max_seq_len_fp8_kv"], 1048576)
        plan_c1 = _plan_on_hw(_flash_next_model(with_dflash=True), hw_96)
        vb = plan_c1["vram_budget"]
        self.assertTrue(vb["fits"])
        # Lossless TS4E (-3.37 GiB) + Qwen4ReleaseMode::Dflash (-1.18 GiB BF16 lm_head) reclaim 4.55 GiB.
        self.assertAlmostEqual(vb["model_weights_per_gpu_gib"], 69.45, places=2)
        self.assertGreaterEqual(vb["free_vram_for_kv_per_gpu_gib"], 14.0)
        self.assertGreaterEqual(vb["recommended_kv_pool_tokens"], 524288)
        self.assertEqual(plan_c1["max_seq_len"], 262144)
        self.assertEqual(plan_c1["kv_cache_dtype"], "bf16")
        self.assertIn("--dflash", plan_c1["command"])
        self.assertEqual(plan_c1["env"]["ATLAS_MOE_SCALE_TS4E"], "1")
        self.assertEqual(plan_c1["env"]["ATLAS_QWEN4_DFLASH_PCUM_THRESH"], "0")
        self.assertEqual(plan_c1["env"]["ATLAS_QWEN4_C1_DENSE_ATTN"], "1")
        self.assertEqual(plan_c1["env"]["ATLAS_QWEN4_SHARED_OVERLAP_THIN_SK"], "1")
        self.assertEqual(plan_c1["container_image"], "ghcr.io/theapathy/flux-engine:sm120")

        # C=16 BF16 accounts for per-slot SSM/QSA reserve + FP8 weight cache (fits real ~418K cap!)
        plan_c16_bf16 = _plan_on_hw(
            _flash_next_model(with_dflash=True),
            hw_96,
            env={"FLUX_CONCURRENCY": "16"},
        )
        self.assertEqual(plan_c16_bf16["kv_cache_dtype"], "bf16")
        c16_pool = int(_arg_value(plan_c16_bf16["command"], "--kv-pool-tokens"))
        self.assertGreaterEqual(c16_pool, 262144)
        self.assertLessEqual(c16_pool, 393216)

        # C=4 high-kv (NVFP4 KV) unlocks >=1.3M pool (hp=auto) and >=2M pool (hp=0) cleanly
        plan_c4_nvfp4 = _plan_on_hw(
            _flash_next_model(with_dflash=True),
            hw_96,
            mode="high-kv",
            env={"FLUX_CONCURRENCY": "4"},
        )
        self.assertEqual(plan_c4_nvfp4["kv_cache_dtype"], "nvfp4")
        self.assertEqual(plan_c4_nvfp4["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertEqual(plan_c4_nvfp4["env"]["ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE"], "0")
        self.assertGreaterEqual(int(_arg_value(plan_c4_nvfp4["command"], "--kv-pool-tokens")), 1310720)
        self.assertEqual(_arg_value(plan_c4_nvfp4["command"], "--gpu-memory-utilization"), "0.92")
        self.assertEqual(_arg_value(plan_c4_nvfp4["command"], "--speculative-cuda-headroom-mb"), "2048")

    def test_1x_sm103_b300_gb300_fits_flash_next_dsv41_and_glm53_on_1_gpu(self):
        for smi_line in (
            "NVIDIA B300 SXM6 AC, 294912, 10.3\n",
            "NVIDIA GB300, 275456, 10.3\n",
        ):
            hw = _make_hw_via_smi(smi_line, host_ram_gib=512.0)
            with self.subTest(gpu=smi_line.split(",")[0]):
                # 1. Qwen3.8-Flash-Next with massive multi-million-token BF16 KV pool
                plan_fn = _plan_on_hw(
                    _flash_next_model(with_dflash=True),
                    hw,
                    env={"FLUX_CONCURRENCY": "8"},
                )
                self.assertTrue(plan_fn["vram_budget"]["fits"])
                self.assertEqual(plan_fn["world_size"], 1)
                self.assertEqual(plan_fn["env"]["FLUX_ARCH_TARGET"], "sm_103a")
                self.assertEqual(plan_fn["container_image"], "ghcr.io/theapathy/flux-engine:sm103")
                self.assertGreaterEqual(plan_fn["vram_budget"]["recommended_kv_pool_tokens"], 4194304)
                self.assertGreaterEqual(int(_arg_value(plan_fn["command"], "--kv-pool-tokens")), 4194304)

                # 2. DeepSeek-V4.1 on 1 GPU with massive >=1M BF16 KV pool
                plan_ds = _plan_on_hw(_dsv41_model(), hw)
                self.assertTrue(plan_ds["vram_budget"]["fits"])
                self.assertEqual(plan_ds["world_size"], 1)
                self.assertEqual(plan_ds["env"]["FLUX_ARCH_TARGET"], "sm_103a")
                self.assertGreaterEqual(plan_ds["vram_budget"]["recommended_kv_pool_tokens"], 1048576)
                self.assertNotIn("ATLAS_QWEN4_PREFILL_WIDE", plan_ds["env"])

                # 3. GLM-5.3 on 1 GPU with massive >=2M BF16 KV pool
                plan_glm = _plan_on_hw(_glm53_model(), hw)
                self.assertTrue(plan_glm["vram_budget"]["fits"])
                self.assertEqual(plan_glm["world_size"], 1)
                self.assertEqual(plan_glm["env"]["FLUX_ARCH_TARGET"], "sm_103a")
                self.assertGreaterEqual(plan_glm["vram_budget"]["recommended_kv_pool_tokens"], 2097152)
                self.assertNotIn("ATLAS_QWEN4_PREFILL_WIDE", plan_glm["env"])


class SingleHostMultiGpuLaunchTests(TestCase):
    """Single-host 2x/4x discrete SM120 TP=2 / TP=4 launch plans & shell snippets."""

    def test_2x_rtx5090_32gb_tp2_dense_and_moe_launch_plans(self):
        hw_2x32 = _make_hw_via_smi("NVIDIA GeForce RTX 5090, 32607, 12.0\n" * 2, host_ram_gib=256.0)

        # Dense 27B -> auto TP=2, EP=1, world_size=2
        plan_27b = _plan_on_hw(_qwen27b_model(with_dflash=True), hw_2x32)
        self.assertEqual(plan_27b["world_size"], 2)
        self.assertEqual(plan_27b["tp_size"], 2)
        self.assertEqual(plan_27b["ep_size"], 1)
        self.assertEqual(_arg_value(plan_27b["command"], "--world-size"), "2")
        self.assertEqual(_arg_value(plan_27b["command"], "--tp-size"), "2")
        self.assertEqual(_arg_value(plan_27b["command"], "--gpu-ordinal"), "0")
        self.assertEqual(_arg_value(plan_27b["command"], "--master-addr"), "127.0.0.1")
        self.assertEqual(_arg_value(plan_27b["command"], "--master-port"), "29500")
        self.assertEqual(plan_27b["env"]["NCCL_P2P_LEVEL"], "SYS")
        mg = plan_27b["multi_gpu_launch"]
        self.assertEqual(mg["mode"], "single-host-multi-gpu")
        self.assertEqual(len(mg["rank_commands"]), 2)
        self.assertEqual(_arg_value(mg["rank_commands"][0], "--gpu-ordinal"), "0")
        self.assertEqual(_arg_value(mg["rank_commands"][1], "--gpu-ordinal"), "1")
        self.assertEqual(_arg_value(mg["rank_commands"][1], "--rank"), "1")
        self.assertIn("FLUX_NODE_RANK=$r", mg["shell_snippet"])
        self.assertIn('--gpu-ordinal "$r"', mg["shell_snippet"])
        self.assertIn("--world-size 2", mg["shell_snippet"])
        self.assertIn("--master-addr 127.0.0.1", mg["shell_snippet"])
        self.assertIn("--master-port 29500", mg["shell_snippet"])
        self.assertIn("NCCL_P2P_LEVEL=SYS", mg["shell_snippet"])
        self.assertIn("ghcr.io/theapathy/flux-engine:sm120", mg["docker_snippet"])

        # MoE 80B-A3B -> auto TP=2, EP=2, world_size=2 (~24.5 GiB/GPU fits 32 GB!)
        plan_80b = _plan_on_hw(_qwen80b_a3b_model(), hw_2x32)
        self.assertTrue(plan_80b["vram_budget"]["fits"])
        self.assertEqual(plan_80b["tp_size"], 2)
        self.assertEqual(plan_80b["ep_size"], 2)
        self.assertEqual(plan_80b["world_size"], 2)

    def test_2x_rtx_pro_5000_48gb_and_72gb_tp2_ep2_flash_next(self):
        hw_2x48 = _make_hw_via_smi("NVIDIA RTX PRO 5000 Blackwell, 49140, 12.0\n" * 2, host_ram_gib=256.0)
        plan_48 = _plan_on_hw(_flash_next_model(with_dflash=True), hw_2x48)
        self.assertTrue(plan_48["vram_budget"]["fits"])
        self.assertEqual(plan_48["world_size"], 2)
        self.assertEqual(plan_48["tp_size"], 2)
        self.assertEqual(plan_48["ep_size"], 2)
        # With TS4E + Qwen4ReleaseMode::Dflash + QSA_FP8 + FP8 fast-proj cache accounted for,
        # 2x 48 GB fits lossless BF16 >=262K cleanly across the 2 GPUs without overshooting VRAM.
        self.assertEqual(plan_48["kv_cache_dtype"], "bf16")
        self.assertGreaterEqual(plan_48["max_seq_len"], 262144)
        self.assertGreaterEqual(int(_arg_value(plan_48["command"], "--kv-pool-tokens")), 262144)
        self.assertIn("multi_gpu_launch", plan_48)

        # Even tighter 2x 42 GB tier auto-switches to NVFP4 KV when BF16 falls below 131K tokens.
        hw_2x42 = _make_hw_via_smi("NVIDIA RTX PRO 5000 Blackwell, 43008, 12.0\n" * 2, host_ram_gib=256.0)
        plan_42 = _plan_on_hw(_flash_next_model(with_dflash=True), hw_2x42)
        self.assertTrue(plan_42["vram_budget"]["fits"])
        self.assertEqual(plan_42["kv_cache_dtype"], "nvfp4")
        self.assertEqual(plan_42["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertGreaterEqual(plan_42["max_seq_len"], 131072)

        hw_2x72 = _make_hw_via_smi("NVIDIA RTX PRO 5000 Blackwell, 73728, 12.0\n" * 2, host_ram_gib=256.0)
        plan_72 = _plan_on_hw(_flash_next_model(with_dflash=True), hw_2x72, env={"FLUX_CONCURRENCY": "8"})
        self.assertTrue(plan_72["vram_budget"]["fits"])
        self.assertEqual(plan_72["kv_cache_dtype"], "bf16")
        self.assertGreaterEqual(plan_72["vram_budget"]["recommended_kv_pool_tokens"], 2097152)
        self.assertEqual(len(plan_72["multi_gpu_launch"]["rank_commands"]), 2)

    def test_2x_and_4x_rtx_pro_6000_96gb_tp2_and_tp4_launch_plans(self):
        hw_2x96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 2, host_ram_gib=512.0)
        plan_2x = _plan_on_hw(_flash_next_model(with_dflash=True), hw_2x96, env={"FLUX_CONCURRENCY": "16"})
        self.assertEqual(plan_2x["world_size"], 2)
        self.assertEqual(plan_2x["tp_size"], 2)
        self.assertEqual(plan_2x["ep_size"], 2)
        self.assertIn("--dflash", plan_2x["command"])
        self.assertGreaterEqual(int(_arg_value(plan_2x["command"], "--kv-pool-tokens")), 3145728)
        self.assertEqual(len(plan_2x["multi_gpu_launch"]["rank_commands"]), 2)

        hw_4x96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 4, host_ram_gib=512.0)
        plan_4x = _plan_on_hw(_flash_next_model(with_dflash=True), hw_4x96, env={"FLUX_CONCURRENCY": "16"})
        self.assertEqual(plan_4x["world_size"], 4)
        self.assertEqual(plan_4x["tp_size"], 4)
        self.assertEqual(plan_4x["ep_size"], 4)
        self.assertIn("--dflash", plan_4x["command"])
        self.assertGreaterEqual(int(_arg_value(plan_4x["command"], "--kv-pool-tokens")), 4194304)
        mg4 = plan_4x["multi_gpu_launch"]
        self.assertEqual(len(mg4["rank_commands"]), 4)
        for r in range(4):
            self.assertEqual(_arg_value(mg4["rank_commands"][r], "--rank"), str(r))
            self.assertEqual(_arg_value(mg4["rank_commands"][r], "--gpu-ordinal"), str(r))
        self.assertIn("for r in $(seq 0 3); do", mg4["shell_snippet"])
        self.assertIn("--world-size 4", mg4["shell_snippet"])

    def test_explicit_single_gpu_and_ordinal_overrides_on_multi_gpu_host(self):
        hw_2x96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 2, host_ram_gib=256.0)
        plan_single = _plan_on_hw(
            _flash_next_model(with_dflash=True),
            hw_2x96,
            env={"FLUX_TP_SIZE": "1", "FLUX_WORLD_SIZE": "1", "FLUX_GPU_ORDINAL": "1"},
        )
        self.assertEqual(plan_single["world_size"], 1)
        self.assertEqual(_arg_value(plan_single["command"], "--gpu-ordinal"), "1")
        self.assertNotIn("multi_gpu_launch", plan_single)

        plan_rank1 = _plan_on_hw(
            _flash_next_model(with_dflash=True),
            hw_2x96,
            env={"FLUX_NODE_RANK": "1", "NCCL_P2P_LEVEL": "NVL"},
        )
        self.assertEqual(_arg_value(plan_rank1["command"], "--rank"), "1")
        self.assertEqual(_arg_value(plan_rank1["command"], "--gpu-ordinal"), "1")
        self.assertNotIn("NCCL_P2P_LEVEL", plan_rank1["env"])  # explicit os.environ override kept untouched


class ContainerTagAndDiagnosticsTests(TestCase):
    """Container image tag recommendation, diagnostics redaction, and Dockerfile hygiene."""

    def test_recommended_container_image_by_architecture(self):
        self.assertEqual(
            fa.recommended_container_image({"sm_family": "gb10-sm121a", "compute_capability": "12.1", "unified_memory_architecture": True}),
            "ghcr.io/theapathy/flux-engine:gb10",
        )
        self.assertEqual(
            fa.recommended_container_image({"sm_family": "blackwell-sm120a", "compute_capability": "12.0", "unified_memory_architecture": False}),
            "ghcr.io/theapathy/flux-engine:sm120",
        )
        self.assertEqual(
            fa.recommended_container_image({"sm_family": "blackwell-sm103a", "compute_capability": "10.3", "unified_memory_architecture": False}),
            "ghcr.io/theapathy/flux-engine:sm103",
        )
        self.assertEqual(
            fa.recommended_container_image({"sm_family": "hopper-sm90", "compute_capability": "9.0", "unified_memory_architecture": False}),
            "ghcr.io/theapathy/flux-engine:latest",
        )

    def test_agents_catalog_prints_sm120_image_on_sm120_host(self):
        hw_32 = _make_hw_via_smi("NVIDIA GeForce RTX 5090, 32607, 12.0\n", host_ram_gib=128.0)
        buf = io.StringIO()
        with (
            mock.patch.object(fa, "detect_hardware_topology", return_value=hw_32),
            mock.patch.object(fa, "scan_models_root", return_value=[]),
            redirect_stdout(buf),
        ):
            fa.print_agents_catalog()
        self.assertIn("ghcr.io/theapathy/flux-engine:sm120", buf.getvalue())

    def test_collect_diagnostics_handles_vram_refusal_and_redacts_gpu_names(self):
        hw_16 = _make_hw_via_smi("NVIDIA GeForce RTX 5060 Ti, 16311, 12.0\n", host_ram_gib=128.0)
        with (
            mock.patch.object(fa, "detect_hardware_topology", return_value=hw_16),
            mock.patch.object(fa, "scan_models_root", return_value=[_flash_next_model()]),
            mock.patch.object(fa.shutil, "which", return_value=None),
        ):
            diag = fa.collect_diagnostics({"telemetry": True, "host": True, "logs": True})
        self.assertIsNone(diag["launch_plan"])
        self.assertEqual(diag["recommended_container_image"], "ghcr.io/theapathy/flux-engine:sm120")
        self.assertEqual(diag["hardware_topology"]["gpu_names"], ["REDACTED-GPU"])
        self.assertFalse(diag["readiness"]["ready"])
        self.assertTrue(any("Insufficient discrete VRAM" in issue for issue in diag["readiness"]["issues"]))

    def test_sm120_dockerfile_hygiene_and_privacy(self):
        self.assertTrue(SM120_DOCKERFILE.is_file(), f"Missing {SM120_DOCKERFILE}")
        text = SM120_DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("--platform=linux/amd64 nvidia/cuda:13.0.0-devel-ubuntu24.04", text)
        self.assertIn("--platform=linux/amd64 nvidia/cuda:13.0.0-runtime-ubuntu24.04", text)
        self.assertIn("ARG FLUX_TARGET_HW=rtx6000", text)
        self.assertIn("libnccl2", text)
        self.assertIn('ge "2.28"', text)
        self.assertIn('FLUX_OPT_OUT="all"', text)
        self.assertIn('HF_HUB_OFFLINE="1"', text)
        self.assertIn('NCCL_P2P_LEVEL="SYS"', text)
        self.assertIn('ATLAS_QWEN4_PREFILL_ATTN_FI="0"', text)
        self.assertIn('/opt/flux/bin/flux-server-universal', text)
        self.assertIn('/opt/flux/bin/flux-server-glm53', text)
        # Never allow blanket COPY . . or copying local VCS/untracked workspace files
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("COPY "):
                self.assertNotIn(".git", stripped)
                self.assertFalse(stripped.startswith("COPY . "), f"Blanket COPY forbidden: {stripped}")
        self.assertNotIn("/home/", text)

    def test_detect_intranode_nccl_env_nvlink_vs_pcie(self):
        hw_2x = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 2, host_ram_gib=256.0)
        with (
            mock.patch.object(fa.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(fa.subprocess, "check_output", return_value="GPU0\tX\tNV12\nGPU1\tNV12\tX\n"),
        ):
            env_nv, trans_nv = fa.detect_intranode_nccl_env(hw_2x)
        self.assertEqual(env_nv["NCCL_P2P_LEVEL"], "SYS")
        self.assertEqual(trans_nv, "nvlink-p2p")

        with (
            mock.patch.object(fa.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(fa.subprocess, "check_output", return_value="GPU0\tX\tPHB\nGPU1\tPHB\tX\n"),
        ):
            env_pcie, trans_pcie = fa.detect_intranode_nccl_env(hw_2x)
        self.assertEqual(env_pcie["NCCL_P2P_LEVEL"], "SYS")
        self.assertEqual(trans_pcie, "pcie-p2p-shm")

    def test_explicit_oversized_seq_or_kv_pool_refused_on_discrete_gpu(self):
        hw_32 = _make_hw_via_smi("NVIDIA GeForce RTX 5090, 32607, 12.0\n", host_ram_gib=256.0)
        with self.assertRaisesRegex(ValueError, r"exceeds discrete VRAM KV capacity"):
            _plan_on_hw(
                _qwen27b_model(with_dflash=True),
                hw_32,
                env={"FLUX_KV_CACHE_DTYPE": "bf16", "FLUX_MAX_SEQ_LEN": "262144"},
            )
        with self.assertRaisesRegex(ValueError, r"exceeds discrete VRAM KV capacity"):
            _plan_on_hw(
                _qwen27b_model(with_dflash=True),
                hw_32,
                env={"FLUX_KV_POOL_TOKENS": "2097152"},
            )

    def test_startup_interstitial_reports_sm120a_and_sm103a_runtime_labels(self):
        hw_96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n", host_ram_gib=256.0)
        plan_120 = _plan_on_hw(_flash_next_model(with_dflash=True), hw_96)
        buf120 = io.StringIO()
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True), redirect_stdout(buf120):
            fa.render_startup_interstitial(_flash_next_model(with_dflash=True), plan_120, {})
        self.assertIn("NVIDIA Blackwell SM120a Native Runtime", buf120.getvalue())

        hw_b300 = _make_hw_via_smi("NVIDIA B300 SXM6 AC, 294912, 10.3\n", host_ram_gib=512.0)
        plan_103 = _plan_on_hw(_flash_next_model(with_dflash=True), hw_b300)
        buf103 = io.StringIO()
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True), redirect_stdout(buf103):
            fa.render_startup_interstitial(_flash_next_model(with_dflash=True), plan_103, {})
        self.assertIn("NVIDIA Blackwell Ultra SM103a Native Runtime", buf103.getvalue())

    def test_cli_inspect_dry_run_refusal_on_16gb_and_multigpu_json_on_2x96gb(self):
        hw_16 = _make_hw_via_smi("NVIDIA GeForce RTX 5060 Ti, 16311, 12.0\n", host_ram_gib=256.0)
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(fa.sys, "argv", ["flux", "inspect"]),
            mock.patch.object(fa, "resolve_model_query", return_value=_flash_next_model()),
            mock.patch.object(fa, "detect_hardware_topology", return_value=hw_16),
            mock.patch("sys.stderr", stderr),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            rc = fa.main()
        self.assertEqual(rc, 2)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertIn("Insufficient discrete VRAM", stderr.getvalue())

        hw_2x96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 2, host_ram_gib=512.0)
        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(fa.sys, "argv", ["flux", "inspect"]),
            mock.patch.object(fa, "resolve_model_query", return_value=_flash_next_model()),
            mock.patch.object(fa, "detect_hardware_topology", return_value=hw_2x96),
            mock.patch.object(
                fa,
                "detect_intranode_nccl_env",
                return_value=({"NCCL_P2P_DISABLE": "0", "NCCL_P2P_LEVEL": "SYS", "NCCL_SHM_DISABLE": "0", "NCCL_IB_DISABLE": "1"}, "pcie-p2p-shm"),
            ),
            redirect_stdout(stdout),
        ):
            rc2 = fa.main()
        self.assertEqual(rc2, 0)
        doc = json.loads(stdout.getvalue())
        self.assertEqual(doc["launch_plan"]["container_image"], "ghcr.io/theapathy/flux-engine:sm120")
        self.assertEqual(doc["launch_plan"]["multi_gpu_launch"]["world_size"], 2)
        self.assertIn("--gpu-ordinal", doc["launch_plan"]["multi_gpu_launch"]["shell_snippet"])

    def test_no_dflash_alias_disables_speculation_without_leaking_flag(self):
        hw_96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n", host_ram_gib=512.0)
        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(fa.sys, "argv", ["flux", "inspect", "--no-dflash"]),
            mock.patch.object(fa, "resolve_model_query", return_value=_flash_next_model(with_dflash=True)),
            mock.patch.object(fa, "detect_hardware_topology", return_value=hw_96),
            redirect_stdout(stdout),
        ):
            rc = fa.main()
        self.assertEqual(rc, 0)
        doc = json.loads(stdout.getvalue())
        cmd = doc["launch_plan"]["command"]
        self.assertNotIn("--no-dflash", cmd)
        self.assertNotIn("--dflash", cmd)
        self.assertNotIn("--speculative", cmd)
        self.assertEqual(doc["launch_plan"]["env"].get("FLUX_DISABLE_SPECULATION"), "1")

    def test_16k_prefill_chunk_accounts_for_extra_buffer_arena_on_96gb(self):
        hw_96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n", host_ram_gib=512.0)
        b8 = fa.calculate_discrete_vram_budget(
            _flash_next_model(with_dflash=True),
            hw_96,
            kv_cache_dtype="nvfp4",
            speculative_enabled=True,
            spec_mode="dflash",
            max_prefill_tokens=8192,
        )
        b16 = fa.calculate_discrete_vram_budget(
            _flash_next_model(with_dflash=True),
            hw_96,
            kv_cache_dtype="nvfp4",
            speculative_enabled=True,
            spec_mode="dflash",
            max_prefill_tokens=16384,
        )
        self.assertTrue(b16["fits"])
        self.assertLess(b16["free_vram_for_kv_per_gpu_gib"], b8["free_vram_for_kv_per_gpu_gib"])
        self.assertAlmostEqual(
            b16["scratch_overhead_gib"] - b8["scratch_overhead_gib"],
            1.69,
            places=2,
        )
        self.assertAlmostEqual(
            b8["free_vram_for_kv_per_gpu_gib"] - b16["free_vram_for_kv_per_gpu_gib"],
            1.69 * 0.90,
            places=1,
        )

    def test_1x_96gb_sm120_ple_ram_and_c64_keep_bf16_kv_and_verify_graphs(self):
        hw_96 = _make_hw_via_smi(
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 12.0\n",
            host_ram_gib=256.0,
        )
        plan_c64 = _plan_on_hw(
            _flash_next_model(with_dflash=False),
            hw_96,
            env={
                "FLUX_CONCURRENCY": "64",
                "FLUX_PLE_RAM_STAGE": "1",
                "FLUX_MAX_SEQ_LEN": "4096",
                "FLUX_MAX_PREFILL_TOKENS": "2048",
            },
        )
        self.assertEqual(plan_c64["kv_cache_dtype"], "bf16")
        self.assertEqual(plan_c64["env"]["ATLAS_QWEN4_VERIFY_GRAPH"], "1")
        self.assertEqual(plan_c64["env"]["ATLAS_MULTISEQ_GRAPHS"], "1")

    def test_1x_96gb_sm120_1m_nvfp4_hits_1p5m_plus_pool_and_fixes_all_blockers(self):
        hw_96 = _make_hw_via_smi(
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 12.0\n",
            host_ram_gib=256.0,
        )
        # 1. C=1 1m-nvfp4 with `flux-server` binary name (Blocker 3) and --ple-ram (Blocker 2)
        fn_model = _flash_next_model(with_dflash=True)
        fn_model["selected_binary"] = "/opt/flux/bin/flux-server"
        plan_1m = _plan_on_hw(
            fn_model,
            hw_96,
            mode="1m-nvfp4",
            env={"FLUX_PLE_RAM_STAGE": "1"},
        )
        self.assertEqual(plan_1m["max_seq_len"], 1048576)
        self.assertEqual(plan_1m["kv_cache_dtype"], "nvfp4")
        # Blocker 1: nvidia-blackwell-sm120.env `=qkpv` MUST NOT clobber NVFP4 KV!
        self.assertEqual(plan_1m["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertEqual(plan_1m["env"]["FLUX_QWEN4_PREFILL_QSA_FP8"], "0")
        # Blocker 4: lossless TS4E + C=1 SM120 PCUM_THRESH=0 + FAST_PROJ_FP8_WEIGHT_CACHE=0
        self.assertEqual(plan_1m["env"]["ATLAS_MOE_SCALE_TS4E"], "1")
        self.assertEqual(plan_1m["env"]["ATLAS_QWEN4_DFLASH_PCUM_THRESH"], "0")
        self.assertEqual(plan_1m["env"]["ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE"], "0")
        # Blocker 3: `flux-server` emits --rope-yarn-factor, --no-tui, --disable-simhash-watchdog
        self.assertEqual(_arg_value(plan_1m["command"], "--rope-yarn-factor"), "4")
        self.assertIn("--no-tui", plan_1m["command"])
        self.assertIn("--disable-simhash-watchdog", plan_1m["command"])
        # Blocker 6: calibrated 1.25M pool (<= real 1.46M cap with hp=auto), 0.92 util, 2048 MB headroom
        pool_1m = int(_arg_value(plan_1m["command"], "--kv-pool-tokens"))
        self.assertGreaterEqual(pool_1m, 1310720)
        self.assertLessEqual(pool_1m, 1458544)
        self.assertEqual(_arg_value(plan_1m["command"], "--gpu-memory-utilization"), "0.92")
        self.assertEqual(_arg_value(plan_1m["command"], "--speculative-cuda-headroom-mb"), "2048")

        # 2. C=1 & C=4 1m-nvfp4 with hp=0 hits 2M+ KV pool on 1x 96 GB!
        plan_1m_hp0 = _plan_on_hw(
            fn_model,
            hw_96,
            mode="1m-nvfp4",
            env={"FLUX_KV_HIGH_PRECISION_LAYERS": "0", "FLUX_CONCURRENCY": "4"},
        )
        self.assertEqual(_arg_value(plan_1m_hp0["command"], "--kv-high-precision-layers"), "0")
        self.assertGreaterEqual(int(_arg_value(plan_1m_hp0["command"], "--kv-pool-tokens")), 2097152)
        self.assertAlmostEqual(plan_1m_hp0["vram_budget"]["kv_kib_per_token_per_gpu"], 8.36, places=2)

        # 3. C=16 + 1m-nvfp4 shared KV pool on 1x 96 GB auto-selects hp=0 to preserve >=1.5M pool
        plan_1m_c16 = _plan_on_hw(
            fn_model,
            hw_96,
            mode="1m-nvfp4",
            env={"FLUX_CONCURRENCY": "16"},
        )
        self.assertEqual(plan_1m_c16["max_seq_len"], 1048576)
        self.assertEqual(plan_1m_c16["concurrency"], 16)
        self.assertEqual(_arg_value(plan_1m_c16["command"], "--kv-high-precision-layers"), "0")
        self.assertGreaterEqual(int(_arg_value(plan_1m_c16["command"], "--kv-pool-tokens")), 1572864)
        self.assertEqual(plan_1m_c16["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")

    def test_multi_gpu_2x_4x_8x_96gb_sm120_kv_pooling_scales_to_4m_16m(self):
        # 2x 96 GB TP=2+EP=2: 1m-nvfp4 pools >=6M tokens; auto BF16 C=16 pools >=3M tokens
        hw_2x96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 2, host_ram_gib=512.0)
        plan_2x_1m = _plan_on_hw(
            _flash_next_model(with_dflash=True),
            hw_2x96,
            mode="1m-nvfp4",
            env={"FLUX_CONCURRENCY": "16"},
        )
        self.assertEqual(plan_2x_1m["world_size"], 2)
        self.assertEqual(plan_2x_1m["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertEqual(plan_2x_1m["env"]["ATLAS_MOE_SCALE_TS4E"], "1")
        self.assertGreaterEqual(int(_arg_value(plan_2x_1m["command"], "--kv-pool-tokens")), 4194304)

        # 4x 96 GB TP=4+EP=4: 1m-nvfp4 pools >=8M tokens
        hw_4x96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 4, host_ram_gib=512.0)
        plan_4x_tp4 = _plan_on_hw(
            _flash_next_model(with_dflash=True),
            hw_4x96,
            mode="1m-nvfp4",
            env={"FLUX_CONCURRENCY": "16"},
        )
        self.assertEqual(plan_4x_tp4["world_size"], 4)
        self.assertEqual(plan_4x_tp4["tp_size"], 4)
        self.assertGreaterEqual(int(_arg_value(plan_4x_tp4["command"], "--kv-pool-tokens")), 8388608)

        # 4x 96 GB PP=4: 1m-nvfp4 pools 16M tokens (using extended 12M/16M tier!)
        plan_4x_pp4 = _plan_on_hw(
            _flash_next_model(with_dflash=False),
            hw_4x96,
            mode="1m-nvfp4",
            env={"FLUX_PP_SIZE": "4"},
        )
        self.assertEqual(plan_4x_pp4["pp_size"], 4)
        self.assertEqual(int(_arg_value(plan_4x_pp4["command"], "--kv-pool-tokens")), 16777216)

        # 8x 96 GB dense 27B TP=8: pools >=8M tokens across 8 GPUs
        hw_8x96 = _make_hw_via_smi("NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n" * 8, host_ram_gib=1024.0)
        plan_8x_27b = _plan_on_hw(
            _qwen27b_model(with_dflash=False),
            hw_8x96,
            env={"FLUX_CONCURRENCY": "16"},
        )
        self.assertEqual(plan_8x_27b["world_size"], 8)
        self.assertEqual(plan_8x_27b["tp_size"], 8)
        self.assertGreaterEqual(int(_arg_value(plan_8x_27b["command"], "--kv-pool-tokens")), 8388608)

        # 8x 96 GB Flash-Next PP=8 (via direct VRAM budget): hits 16M tier cleanly
        b_8x_pp8 = fa.calculate_discrete_vram_budget(
            _flash_next_model(with_dflash=False),
            hw_8x96,
            tp_size=1,
            ep_size=1,
            pp_size=8,
            gpu_mem_util=0.92,
            kv_cache_dtype="nvfp4",
            kv_high_precision_layers="auto",
            ts4e_enabled=True,
            release_unused_dflash=True,
        )
        self.assertTrue(b_8x_pp8["fits"])
        self.assertEqual(b_8x_pp8["recommended_kv_pool_tokens"], 16777216)

    def test_gb10_uma_is_untouched_by_discrete_pcum_headroom_and_util_changes(self):
        hw_gb10 = _make_hw_via_smi("NVIDIA GB10, 128000, 12.1\n", host_ram_gib=128.0)
        plan_gb10 = _plan_on_hw(_flash_next_model(with_dflash=True), hw_gb10)
        # GB10 UMA must NOT receive SM120 C=1 PCUM_THRESH=0, discrete 2048 MB headroom, or 0.92 util.
        self.assertNotIn("ATLAS_QWEN4_DFLASH_PCUM_THRESH", plan_gb10["env"])
        self.assertNotIn("ATLAS_QWEN4_DFLASH_PCUM", plan_gb10["env"])
        self.assertNotIn("--speculative-cuda-headroom-mb", plan_gb10["command"])
        self.assertEqual(_arg_value(plan_gb10["command"], "--gpu-memory-utilization"), "0.85")
        self.assertEqual(plan_gb10["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "qkpv")
        self.assertNotIn("vram_budget", plan_gb10)

        # GB10 UMA 1m-nvfp4 keeps 16384 prefill and 0.85 util, without discrete auto-pooling.
        plan_gb10_1m = _plan_on_hw(_flash_next_model(with_dflash=True), hw_gb10, mode="1m-nvfp4")
        self.assertEqual(_arg_value(plan_gb10_1m["command"], "--max-prefill-tokens"), "16384")
        self.assertEqual(_arg_value(plan_gb10_1m["command"], "--gpu-memory-utilization"), "0.85")
        self.assertNotIn("--speculative-cuda-headroom-mb", plan_gb10_1m["command"])
        self.assertNotIn("--kv-pool-tokens", plan_gb10_1m["command"])
        self.assertEqual(plan_gb10_1m["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")

    def test_detect_quantization_handles_modelopt_mixed_precision_in_config_json(self):
        cfg = {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
            }
        }
        self.assertEqual(
            fa.detect_quantization(Path("/models/custom-flash-next"), cfg, "custom-flash-next"),
            "NVFP4",
        )
        cfg_fp8 = {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
            }
        }
        self.assertEqual(
            fa.detect_quantization(Path("/models/custom-fp8-model"), cfg_fp8, "custom-fp8-model"),
            "FP8",
        )



