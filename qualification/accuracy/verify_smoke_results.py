#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify paired smoke check reports for Milestone 3 (Full Accuracy Parity - R1)."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

def main():
    root = Path(os.environ.get("FLUX_ACCURACY_DIR", str(Path(__file__).resolve().parent)))
    parser = argparse.ArgumentParser(
        description="Verify paired smoke check reports for Milestone 3 (Full Accuracy Parity - R1)."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(os.environ.get("FLUX_ACCURACY_SMOKE_REPORTS_DIR", str(root / "reports-smoke"))),
        help="Directory containing paired smoke reports (default: qualification/accuracy/reports-smoke).",
    )
    parser.add_argument(
        "--memory-csv",
        type=Path,
        default=None,
        help="Path to smoke memory_samples.csv (default: <reports-dir>/memory_samples.csv).",
    )
    args = parser.parse_args()
    reports_dir = args.reports_dir
    mem_csv = args.memory_csv if args.memory_csv is not None else (reports_dir / "memory_samples.csv")

    profiles = ["gsm8k", "mmlu-pro"]
    results = {}
    total_items = 0
    mismatches = 0
    non_empty_interventions = 0

    print("=== Milestone 3 Smoke Check Verification ===")

    for profile in profiles:
        plain_file = reports_dir / f"plain-{profile}.json"
        k5_file = reports_dir / f"k5-{profile}.json"

        if not plain_file.exists():
            print(f"[-] ERROR: Missing {plain_file}")
            sys.exit(1)
        if not k5_file.exists():
            print(f"[-] ERROR: Missing {k5_file}")
            sys.exit(1)

        with open(plain_file, "r", encoding="utf-8") as f:
            plain_data = json.load(f)
        with open(k5_file, "r", encoding="utf-8") as f:
            k5_data = json.load(f)

        plain_runs = plain_data.get("runs", [])
        k5_runs = k5_data.get("runs", [])

        if len(plain_runs) != len(k5_runs):
            print(f"[-] ERROR: {profile} runs count differs: Plain={len(plain_runs)}, K5={len(k5_runs)}")
            sys.exit(1)

        count = len(plain_runs)
        print(f"[*] Validating {count} items for profile: {profile}...")
        profile_matches = 0
        profile_mismatches = 0

        for idx, (p_run, k_run) in enumerate(zip(plain_runs, k5_runs)):
            total_items += 1
            item_id = p_run.get("item_id")
            k_item_id = k_run.get("item_id")
            if item_id != k_item_id:
                print(f"[-] Item ID mismatch at {idx}: {item_id} vs {k_item_id}")
                mismatches += 1
                profile_mismatches += 1
                continue

            # Compare answers and tokens
            p_ans = p_run.get("parsed_answer")
            k_ans = k_run.get("parsed_answer")
            p_final = p_run.get("final_answer")
            k_final = k_run.get("final_answer")
            p_tokens = p_run.get("completion_tokens")
            k_tokens = k_run.get("completion_tokens")
            p_correct = p_run.get("correct")
            k_correct = k_run.get("correct")
            p_finish = p_run.get("finish_reason")
            k_finish = k_run.get("finish_reason")

            p_ledger = p_run.get("interventions", {})
            k_ledger = k_run.get("interventions", {})

            # Check interventions
            for name, ledger in [("Plain", p_ledger), ("K5", k_ledger)]:
                if ledger.get("schema") != "atlas.interventions.v1":
                    print(f"[-] {name} item {item_id} invalid ledger schema: {ledger.get('schema')}")
                    non_empty_interventions += 1
                if ledger.get("policy") != "report_only":
                    print(f"[-] {name} item {item_id} invalid ledger policy: {ledger.get('policy')}")
                    non_empty_interventions += 1
                events = ledger.get("events", [])
                if events:
                    print(f"[-] {name} item {item_id} non-empty intervention events: {events}")
                    non_empty_interventions += 1

            if p_ans == k_ans and p_final == k_final and p_tokens == k_tokens and p_correct == k_correct and p_finish == "stop" and k_finish == "stop":
                profile_matches += 1
                print(f"  [+] item {idx+1}/{count} ({item_id}): PARITY OK | tokens={p_tokens} | ans={p_ans!r} | correct={p_correct}")
            else:
                profile_mismatches += 1
                mismatches += 1
                print(f"  [-] item {idx+1}/{count} ({item_id}): PARITY DIFF: Plain(ans={p_ans!r}, tok={p_tokens}, corr={p_correct}, fin={p_finish}) vs K5(ans={k_ans!r}, tok={k_tokens}, corr={k_correct}, fin={k_finish})")

        results[profile] = {
            "total_items": count,
            "matched_items": profile_matches,
            "mismatches": profile_mismatches,
            "plain_score": plain_data.get("accuracy", {}).get("accuracy"),
            "k5_score": k5_data.get("accuracy", {}).get("accuracy"),
        }

    # Verify memory floor (fail closed when samples are missing or empty)
    min_mem_mib = None
    if mem_csv.exists():
        with open(mem_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            samples = [float(row["mem_available_mib"]) for row in reader if "mem_available_mib" in row]
            if samples:
                min_mem_mib = min(samples)
    print(f"[*] Memory floor: lowest measured MemAvailable = {min_mem_mib} MiB (required >= 6656 MiB)")

    memory_floor_satisfied = min_mem_mib is not None and min_mem_mib >= 6656
    summary = {
        "verdict": "PASS" if total_items > 0 and mismatches == 0 and non_empty_interventions == 0 and memory_floor_satisfied else "FAIL",
        "total_items_evaluated": total_items,
        "mismatches": mismatches,
        "non_empty_interventions": non_empty_interventions,
        "min_available_memory_mib": min_mem_mib,
        "memory_floor_satisfied": memory_floor_satisfied,
        "profiles": results,
    }

    summary_file = root / "smoke_parity_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Wrote smoke parity summary to {summary_file}")
    print(f"=== Verdict: {summary['verdict']} ===")
    if summary["verdict"] != "PASS":
        sys.exit(1)

if __name__ == "__main__":
    main()
