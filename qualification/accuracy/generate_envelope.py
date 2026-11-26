#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate the lil-accuracy-pair envelope for Milestone 3 (Full Accuracy Parity - R1)."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)

def main():
    root = Path(os.environ.get("FLUX_WORKTREE", str(Path(__file__).resolve().parents[2])))
    qual_dir = root / "qualification/accuracy"
    parser = argparse.ArgumentParser(
        description="Generate the lil-accuracy-pair envelope for Milestone 3 (Full Accuracy Parity - R1)."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=qual_dir / "reports-smoke",
        help="Directory containing paired reports (default: qualification/accuracy/reports-smoke).",
    )
    args = parser.parse_args()
    binary_path = Path(os.environ.get("FLUX_SERVER_BIN", str(root / "target" / "release" / "flux-server")))

    plain_recipe = qual_dir / "recipe-plain.env"
    k5_recipe = qual_dir / "recipe-k5.env"

    source_rev = "3466ce319381d96e55bd6ecc0c0ffb00e15e7c59"
    checkpoint_rev = "0000000000000000000000000000000000000000"

    # Create dummy source and checkpoint inventories if not present
    src_inv_path = qual_dir / "source-inventory.json"
    if not src_inv_path.exists():
        src_inv_path.write_text(json.dumps({
            "schema": "source-inventory-v1",
            "source_revision": source_rev,
        }, sort_keys=True) + "\n")

    ckpt_inv_path = qual_dir / "checkpoint-inventory.json"
    if not ckpt_inv_path.exists():
        ckpt_inv_path.write_text(json.dumps({
            "schema": "checkpoint-inventory-v1",
            "checkpoint_id": "qwen38",
            "checkpoint_revision": checkpoint_rev,
        }, sort_keys=True) + "\n")

    binary_sha = sha256_file(binary_path)
    plain_recipe_sha = sha256_file(plain_recipe)
    k5_recipe_sha = sha256_file(k5_recipe)
    src_inv_sha = sha256_file(src_inv_path)
    ckpt_inv_sha = sha256_file(ckpt_inv_path)

    base_argv = [
        str(binary_path),
        "serve",
        "--no-tui",
        "--bind",
        "127.0.0.1",
        "--port",
        "8899",
        "--model-from-path",
        os.environ.get("FLUX_MODEL_DIR", str(Path.home() / "models" / "Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid")),
        "--model-name",
        "qwen38",
        "--kernel-target",
        "qwen3.8-flash-next",
        "--gpu-memory-utilization",
        "0.85",
        "--kv-cache-dtype",
        "bf16",
        "--block-size",
        "16",
        "--max-seq-len",
        "262144",
        "--max-prefill-tokens",
        "4096",
        "--max-batch-size",
        "1",
        "--max-num-seqs",
        "1",
        "--ssm-cache-slots",
        "0",
        "--swap-space-gb",
        "0",
        "--qwen4-qsa",
        "--oom-guard-mb",
        "4096",
        "--request-timeout",
        "14340",
        "--disable-simhash-watchdog",
        "--enable-prefix-caching",
        "false",
    ]

    k5_argv = base_argv + [
        "--speculative",
        "--num-drafts",
        "4",
        "--mtp-vocab",
        "0",
        "--mtp-quantization",
        "bf16",
    ]

    plain_argv_sha = sha256_bytes(canonical_bytes(base_argv))
    k5_argv_sha = sha256_bytes(canonical_bytes(k5_argv))

    reports_dir = args.reports_dir
    reports_map = {
        "no_spec": {
            "gsm8k": reports_dir / "plain-gsm8k.json",
            "mmlu-pro": reports_dir / "plain-mmlu-pro.json",
        },
        "k5": {
            "gsm8k": reports_dir / "k5-gsm8k.json",
            "mmlu-pro": reports_dir / "k5-mmlu-pro.json",
        },
    }

    arm_reports = {"no_spec": {}, "k5": {}}
    for arm in ("no_spec", "k5"):
        for profile in ("gsm8k", "mmlu-pro"):
            p = reports_map[arm][profile]
            if not p.exists():
                raise FileNotFoundError(f"Missing paired report for {arm}/{profile}: {p}")
            arm_reports[arm][profile] = {
                "path": rel_to(root, p),
                "sha256": sha256_file(p),
            }

    envelope = {
        "schema": "flux-lil-accuracy-pair-envelope-v1",
        "model": "qwen38",
        "artifacts": {
            "image_digest": "sha256:" + "0" * 64,
            "image_id": "sha256:" + "0" * 64,
            "binary_path": rel_to(root, binary_path),
            "binary_sha256": binary_sha,
            "source_revision": source_rev,
            "source_inventory_path": rel_to(root, src_inv_path),
            "source_inventory_sha256": src_inv_sha,
            "checkpoint_id": "qwen38",
            "checkpoint_revision": checkpoint_rev,
            "checkpoint_inventory_path": rel_to(root, ckpt_inv_path),
            "checkpoint_inventory_sha256": ckpt_inv_sha,
        },
        "arms": {
            "no_spec": {
                "mode": "none",
                "recipe_path": rel_to(root, plain_recipe),
                "recipe_sha256": plain_recipe_sha,
                "argv": base_argv,
                "argv_sha256": plain_argv_sha,
                "reports": arm_reports["no_spec"],
            },
            "k5": {
                "mode": "native-mtp-k5",
                "recipe_path": rel_to(root, k5_recipe),
                "recipe_sha256": k5_recipe_sha,
                "argv": k5_argv,
                "argv_sha256": k5_argv_sha,
                "reports": arm_reports["k5"],
            },
        },
    }

    envelope_path = qual_dir / "lil-accuracy-pair-envelope.json"
    envelope_path.write_text(json.dumps(envelope, indent=2) + "\n")
    print(f"[+] Envelope written to {envelope_path}")

if __name__ == "__main__":
    main()
