#!/usr/bin/env python3
"""
Verification of Prefill Budget Resolver and Qwen4 Prefill Geometry Validation Logic
Mirrors exact logic from:
crates/spark-server/src/main_modules/serve_phases/kv_cache.rs
"""
import sys

def resolve_prefill_budget(
    max_prefill_tokens: int = 8192,
    max_seq_len: int = 4096,
    ssm_prefill_chunk: int = 0,
    wide_override: str = None,
    enable_prefix_caching: bool = True,
    ssm_checkpoint_interval: int = 64,
    ssm_cache_slots: int = 64,
    block_size: int = 16,
    high_speed_swap: bool = False,
    high_speed_swap_cache_blocks_per_seq: int = 64,
    max_batch_size: int = 128,
):
    explicit_wide_map = {
        "2048": 2048,
        "4096": 4096,
        "8192": 8192,
        "16384": 16384,
        "32768": 32768,
    }
    explicit_wide = explicit_wide_map.get(wide_override)
    user_set_prefill = (max_prefill_tokens != 8192)

    if explicit_wide is not None:
        if not user_set_prefill or max_prefill_tokens == explicit_wide:
            prefill_budget_pre_hss = explicit_wide
        elif max_prefill_tokens > 0:
            prefill_budget_pre_hss = max_prefill_tokens
        else:
            prefill_budget_pre_hss = max_seq_len
    elif user_set_prefill and max_prefill_tokens > 0:
        prefill_budget_pre_hss = max_prefill_tokens
    elif ssm_prefill_chunk > 0:
        prefill_budget_pre_hss = ssm_prefill_chunk
    elif max_prefill_tokens > 0:
        prefill_budget_pre_hss = max_prefill_tokens
    else:
        prefill_budget_pre_hss = max_seq_len

    if (
        not user_set_prefill
        and explicit_wide is None
        and enable_prefix_caching
        and ssm_checkpoint_interval > 0
        and ssm_cache_slots > 0
    ):
        target = ssm_checkpoint_interval * block_size
        if prefill_budget_pre_hss > target and target > 0:
            prefill_budget_pre_hss = target

    if high_speed_swap:
        hss_cap_tokens = high_speed_swap_cache_blocks_per_seq * block_size
        hss_chunk_max = max(0, hss_cap_tokens - max_batch_size)
        prefill_budget = min(prefill_budget_pre_hss, hss_chunk_max)
    else:
        prefill_budget = prefill_budget_pre_hss

    return prefill_budget

def validate_qwen4_prefill_geometry(
    is_qwen4: bool,
    wide: str,
    multi_chunk: str,
    prefill_budget: int
):
    if not is_qwen4:
        return True, "OK (not qwen4)"
    if wide is None or wide == "0":
        return True, "OK (wide disabled)"
    valid_wides = {"2048": 2048, "4096": 4096, "8192": 8192, "16384": 16384, "32768": 32768}
    if wide not in valid_wides:
        return False, f"ATLAS_QWEN4_PREFILL_WIDE must be absent, 0, 2048, 4096, 8192, 16384, or 32768; got {wide!r}"
    wide_val = valid_wides[wide]
    if multi_chunk != "1":
        return False, f"ATLAS_QWEN4_PREFILL_WIDE={wide_val} requires ATLAS_QWEN4_PREFILL_MULTI_CHUNK=1"
    if wide_val != prefill_budget:
        return False, (
            f"ATLAS_QWEN4_PREFILL_WIDE={wide_val} requires the resolved scheduler prefill chunk to be {wide_val}, "
            f"but --max-prefill-tokens resolved to {prefill_budget}. This mismatch suppresses the selected "
            f"Qwen4 fast-prefill route. Set ATLAS_QWEN4_PREFILL_WIDE={prefill_budget} or --max-prefill-tokens {wide_val}."
        )
    return True, f"OK (matched {wide_val})"

def run_tests():
    print("=== RUNNING PREFILL BUDGET RESOLVER & GEOMETRY VALIDATION TESTS ===")
    failures = 0

    # Test 1: WIDE=4096 and --max-prefill-tokens omitted (default 8192)
    b1 = resolve_prefill_budget(max_prefill_tokens=8192, wide_override="4096")
    v1_ok, v1_msg = validate_qwen4_prefill_geometry(True, "4096", "1", b1)
    print(f"Test 1 [WIDE=4096, max_prefill_tokens omitted]: budget={b1}, valid={v1_ok} ({v1_msg})")
    if b1 != 4096 or not v1_ok:
        print("  FAIL: Expected budget 4096 and valid geometry")
        failures += 1
    else:
        print("  PASS")

    # Test 2: WIDE=16384 and --max-prefill-tokens omitted (default 8192)
    b2 = resolve_prefill_budget(max_prefill_tokens=8192, wide_override="16384")
    v2_ok, v2_msg = validate_qwen4_prefill_geometry(True, "16384", "1", b2)
    print(f"Test 2 [WIDE=16384, max_prefill_tokens omitted]: budget={b2}, valid={v2_ok} ({v2_msg})")
    if b2 != 16384 or not v2_ok:
        print("  FAIL: Expected budget 16384 and valid geometry")
        failures += 1
    else:
        print("  PASS")

    # Test 3: --max-prefill-tokens 2048 explicitly set with WIDE=4096
    b3 = resolve_prefill_budget(max_prefill_tokens=2048, wide_override="4096")
    v3_ok, v3_msg = validate_qwen4_prefill_geometry(True, "4096", "1", b3)
    print(f"Test 3 [--max-prefill-tokens 2048, WIDE=4096]: budget={b3}, valid={v3_ok} ({v3_msg})")
    if b3 != 2048 or v3_ok:
        print("  FAIL: Expected budget 2048 and rejected geometry")
        failures += 1
    else:
        print("  PASS (Budget resolved to 2048 and geometry appropriately rejected mismatch)")

    # Test 4: Default behavior without WIDE override (WIDE=None, max_prefill_tokens omitted)
    b4 = resolve_prefill_budget(max_prefill_tokens=8192, wide_override=None)
    print(f"Test 4 [WIDE=None, max_prefill_tokens omitted]: budget={b4} (auto-clamped to 1024)")
    if b4 != 1024:
        print("  FAIL: Expected budget 1024")
        failures += 1
    else:
        print("  PASS")

    # Test 5: WIDE=4096 with explicit --max-prefill-tokens 4096
    b5 = resolve_prefill_budget(max_prefill_tokens=4096, wide_override="4096")
    v5_ok, v5_msg = validate_qwen4_prefill_geometry(True, "4096", "1", b5)
    print(f"Test 5 [WIDE=4096, max_prefill_tokens=4096]: budget={b5}, valid={v5_ok} ({v5_msg})")
    if b5 != 4096 or not v5_ok:
        print("  FAIL: Expected budget 4096 and valid geometry")
        failures += 1
    else:
        print("  PASS")

    # Test 6: WIDE=8192 and --max-prefill-tokens omitted (default 8192)
    b6 = resolve_prefill_budget(max_prefill_tokens=8192, wide_override="8192")
    v6_ok, v6_msg = validate_qwen4_prefill_geometry(True, "8192", "1", b6)
    print(f"Test 6 [WIDE=8192, max_prefill_tokens omitted]: budget={b6}, valid={v6_ok} ({v6_msg})")
    if b6 != 8192 or not v6_ok:
        print("  FAIL: Expected budget 8192 and valid geometry")
        failures += 1
    else:
        print("  PASS")

    # Test 7: WIDE=32768 and --max-prefill-tokens omitted (default 8192)
    b7 = resolve_prefill_budget(max_prefill_tokens=8192, wide_override="32768")
    v7_ok, v7_msg = validate_qwen4_prefill_geometry(True, "32768", "1", b7)
    print(f"Test 7 [WIDE=32768, max_prefill_tokens omitted]: budget={b7}, valid={v7_ok} ({v7_msg})")
    if b7 != 32768 or not v7_ok:
        print("  FAIL: Expected budget 32768 and valid geometry")
        failures += 1
    else:
        print("  PASS")

    # Test 8: WIDE=0 (disabled fast prefill path)
    b8 = resolve_prefill_budget(max_prefill_tokens=4096, wide_override="0")
    v8_ok, v8_msg = validate_qwen4_prefill_geometry(True, "0", "1", b8)
    print(f"Test 8 [WIDE=0, max_prefill_tokens=4096]: budget={b8}, valid={v8_ok} ({v8_msg})")
    if b8 != 4096 or not v8_ok:
        print("  FAIL: Expected budget 4096 and valid (disabled) geometry")
        failures += 1
    else:
        print("  PASS")

    # Test 9: WIDE=2048 and --max-prefill-tokens omitted (default 8192)
    b9 = resolve_prefill_budget(max_prefill_tokens=8192, wide_override="2048")
    v9_ok, v9_msg = validate_qwen4_prefill_geometry(True, "2048", "1", b9)
    print(f"Test 9 [WIDE=2048, max_prefill_tokens omitted]: budget={b9}, valid={v9_ok} ({v9_msg})")
    if b9 != 2048 or not v9_ok:
        print("  FAIL: Expected budget 2048 and valid geometry")
        failures += 1
    else:
        print("  PASS")

    if failures == 0:
        print("\nALL 9 TESTS PASSED.")
        sys.exit(0)
    else:
        print(f"\n{failures} TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
