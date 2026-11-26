#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Inventory every ATLAS_*/FLUX_* env literal read under crates/ and classify it.

Classes (see docs/adr/recipe-contract.md):
  NUMERICS       changes served output, precision or kernel arithmetic
  ROUTE          changes the kernel/route but claims identical output (must be verified)
  RESOURCE       memory, threads, limits, paths, I/O shape
  OBSERVABILITY  logs, receipts, dumps, profiles
  DEBUG          debug or diagnostic production switches
  DEAD           literal present with no production reader (incl. test/bench/example-only)

Method: a literal is "read" when a non-test, non-comment production line holds
it quoted and is not a set_var/remove_var, option_env!, allowlist `rule(` entry,
prefix/meta exclusion table, or standalone tracing/log event tag. Dynamic
`FLUX_<suffix>` / `ATLAS_<suffix>` helpers (`env_setting`, `mem_watch`
`flux_value`/`parse_mib`/`parse_action`, and `think_nudge` `get`) are also
resolved so production NUMERICS and RESOURCE gates cannot hide behind helper
prefixing. Manual overrides in env_inventory_overrides.tsv win over the keyword
rules; the `basis` column says which one decided.

Usage: env_inventory.py [--check] <repo-root> <out-dir>
Writes env_inventory.csv and env_inventory_summary.md to <out-dir>, and the
runtime catalog (every non-DEAD name and class) to
crates/spark-server/src/recipe_contract/env_catalog.tsv.
With --check, exits non-zero if env_catalog.tsv would change instead of writing.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

LITERAL = re.compile(r'"((?:ATLAS|FLUX)_[A-Z0-9_]+)"')
CLASSES = ("NUMERICS", "ROUTE", "RESOURCE", "OBSERVABILITY", "DEBUG", "DEAD")

# Order matters: first match wins. Patterns match the name with the
# ATLAS_/FLUX_ prefix stripped.
KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    ("DEBUG", r"(^|_)(DEBUG|TEST|FAKE|MOCK|INJECT|FAULT|FORCE_FAIL|PANIC|SELFTEST|ASSERT|CHECK|VALIDATE|AUDIT|COMPARE|ORACLE|REF_CHECK|BISECT|PROBE|CANARY|SANITY)(_|$)"),
    ("OBSERVABILITY", r"(^|_)(LOG|LOGS|LOGGING|TRACE|TRACING|PROFILE|PROFILING|DUMP|DUMPS|RECEIPT|RECEIPTS|METRIC|METRICS|TIMING|TIMINGS|STATS|REPORT|PRINT|VERBOSE|TELEMETRY|NVTX|HIST|HISTOGRAM|CENSUS|FNV|HASH_LOG|TAP|TAPS|WATCHDOG|BENCH|SPAN|EVENTS|INSPECT|DIAG|DIAGNOSTIC|ENGAGED)(_|$)"),
    ("NUMERICS", r"(^|_)(FP8|FP4|NVFP4|MXFP8|BF16|F16|FP16|F32|FP32|E4M3|E5M2|W4A8|W4A4|W8A8|W8A16|INT8|INT4|QUANT|QUANTIZE|PRECISION|DTYPE|KV_CACHE_DTYPE|FAST_MATH|APPROX|FAST_PROJ|TEMPERATURE|TOP_P|TOP_K|MIN_P|PENALTY|PENALTIES|SAMPLING|SAMPLER|ROPE|YARN|THETA|AMAX|SCALE|ROUNDING|LOSSY|TRUNCATE|THINK|THINKING|BUDGET|NUDGE|REFLECTION|SUPPRESS|GRAMMAR|LOGIT|LOGITS|BIAS|SEED|TEMP|SOFTCAP|RELAX|TYPICAL|SKIP_LAYERS)(_|$)|FAKEQUANT"),
    ("RESOURCE", r"(^|_)(MB|GB|MIB|GIB|KB|BYTES|MEM|MEMORY|MEMAVAILABLE|POOL|THREADS|THREAD|WORKERS|RESERVE|FLOOR|CAP|LIMIT|TIMEOUT|TIMEOUT_MS|QUEUE|DEPTH|RINGS|RING|BLOCKS|PAGES|PORT|ADDR|DIR|PATH|HOME|FILE|ALLOC|SWAP|OFFLOAD|PREFETCH|IO|DIRECT_IO|AFFINITY|MAX_SEQ_LEN|MAX_BATCH|MAX_PREFILL_TOKENS|CONCURRENCY|NUMA|PINNED|MMAP|RELEASE|EVICT|HEADROOM|BUDGET_MB|GATHER_MS|INTERVAL|RETRY|RETRIES|TTL|URL|API_KEY|REPO|CACHE)(_|$)"),
    ("ROUTE", r"."),
)
CONTRACT_DIR = "crates/spark-server/src/recipe_contract/"
NON_READER_PREFIXES = (
    CONTRACT_DIR,
    "crates/spark-server/src/flux_config/",
)
NON_READER_FILES = {
    "crates/spark-server/src/flux_config.rs",
}

# Const string tables that enumerate excluded prefixes or launcher-only meta
# keys rather than reading env vars.
EXCLUSION_TABLE_CONSTS = {
    "crates/spark-server/src/main.rs": "META_KEYS",
    "crates/spark-model/src/model/seqstate/qwen4.rs": "NON_NUMERIC_ENV_PREFIXES",
}

# Dynamic FLUX_<suffix> / ATLAS_<suffix> readers that pass only the suffix
# literal to a helper function.
ENV_SETTING_RE = re.compile(r'\benv_setting\s*\(\s*"([A-Z0-9_]+)"\s*\)')
MEM_WATCH_HELPER_RE = re.compile(
    r'\b(?:flux_value|parse_mib|parse_action)\s*\([^,]+,\s*"([A-Z0-9_]+)"'
)
THINK_NUDGE_GET_RE = re.compile(r'\bget\s*\(\s*"(THINK_NUDGE[A-Z0-9_]*)"\s*\)')


def is_build_script(path: Path) -> bool:
    """build.rs and its sibling modules (build_*.rs) at a crate root run at compile time."""
    return path.name.startswith("build") and path.parent.parent.name == "crates"


TEST_PATH = re.compile(r"(/tests?/|_tests?\.rs$|/test_[^/]*\.rs$|/benches/|/examples/)")
NOT_A_READ = re.compile(r"(set_var|remove_var|option_env!|^\s*rule\(|^\s*//|^\s*\*(\s|/|$)|^\s*#\[doc)")
TEST_ONLY_REF = re.compile(r"(assert|contains\(|\.expect_err|error\.)")
RECEIPT_NEAR = re.compile(r"(ENGAGED|receipt|RECEIPT|engaged)")
APPROX_NAME = re.compile(r"(FP8|FP4|NVFP4|W4A8|W4A4|MXFP8|E4M3|_FI(_|$)|FAST_PROJ|FAST_MATH|APPROX|LOSSY|FUSED)")
FAITHFUL_NAME = re.compile(r"(_BF16(_|$)|_NATIVE(_|$)|DENSE_NATIVE|_EXACT(_|$)|IEEE)")
LOG_MACRO_RE = re.compile(r"(?:tracing|log)::(?:info|warn|error|debug|trace)!$")


@dataclass
class Var:
    name: str
    files: set[str] = field(default_factory=set)
    readers: list[str] = field(default_factory=list)
    test_refs: int = 0
    test_readers: int = 0
    receipt_near: bool = False
    default_on_hint: bool = False


def strip_rust_noncode(text: str) -> list[str]:
    """Strip Rust comments, strings, raw strings, and char literals, keeping line count 1:1."""
    out: list[str] = []
    cur: list[str] = []
    i, n = 0, len(text)
    block_comment = 0
    while i < n:
        ch = text[i]
        if block_comment > 0:
            if ch == "\n":
                out.append("".join(cur))
                cur = []
            elif ch == "/" and i + 1 < n and text[i + 1] == "*":
                block_comment += 1
                i += 1
            elif ch == "*" and i + 1 < n and text[i + 1] == "/":
                block_comment -= 1
                i += 1
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            block_comment = 1
            i += 2
            continue
        if ch == "r" or (ch == "b" and i + 1 < n and text[i + 1] == "r"):
            j = i + (2 if ch == "b" else 1)
            hashes = 0
            while j < n and text[j] == "#":
                hashes += 1
                j += 1
            if j < n and text[j] == '"':
                j += 1
                end_seq = '"' + "#" * hashes
                while j < n:
                    if text[j] == "\n":
                        out.append("".join(cur))
                        cur = []
                        j += 1
                    elif text.startswith(end_seq, j):
                        j += len(end_seq)
                        break
                    else:
                        j += 1
                i = j
                continue
        if ch == '"' or (ch == "b" and i + 1 < n and text[i + 1] == '"'):
            i += 2 if ch == "b" else 1
            while i < n:
                if text[i] == "\\":
                    if i + 1 < n and text[i + 1] == "\n":
                        out.append("".join(cur))
                        cur = []
                    i += 2
                    continue
                if text[i] == "\n":
                    out.append("".join(cur))
                    cur = []
                    i += 1
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "'" or (ch == "b" and i + 1 < n and text[i + 1] == "'"):
            j = i + (2 if ch == "b" else 1)
            if j < n and text[j] == "\\":
                k = j + 2
                while k < n and text[k] != "'" and text[k] != "\n":
                    k += 1
                if k < n and text[k] == "'":
                    i = k + 1
                    continue
            elif j + 1 < n and text[j + 1] == "'":
                i = j + 2
                continue
        if ch == "\n":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


def test_line_mask(text: str, raw_lines: list[str]) -> list[bool]:
    """Mark lines inside inline `#[cfg(test)] mod x { ... }` blocks (lexer-aware)."""
    if "#[cfg(test)]" not in text:
        return [False] * len(raw_lines)
    code_lines = strip_rust_noncode(text)
    if len(code_lines) < len(raw_lines):
        code_lines.extend([""] * (len(raw_lines) - len(code_lines)))
    mask = [False] * len(raw_lines)
    i = 0
    while i < len(raw_lines):
        if "#[cfg(test)]" in code_lines[i] and re.search(
            r"\bmod\s+\w+\s*\{", " ".join(code_lines[i : i + 3])
        ):
            depth, j, opened = 0, i, False
            while j < len(raw_lines):
                depth += code_lines[j].count("{") - code_lines[j].count("}")
                opened |= "{" in code_lines[j]
                mask[j] = True
                if opened and depth <= 0:
                    break
                j += 1
            i = j + 1
            continue
        i += 1
    return mask


def exclusion_table_mask(lines: list[str], const_name: str | None) -> list[bool]:
    """Mark lines inside a named `const <NAME>: &[&str] = &[ ... ];` table."""
    mask = [False] * len(lines)
    if not const_name:
        return mask
    header = re.compile(rf"\bconst\s+{re.escape(const_name)}\b")
    i = 0
    while i < len(lines):
        if header.search(lines[i]):
            j = i
            while j < len(lines):
                mask[j] = True
                if "];" in lines[j]:
                    break
                j += 1
            i = j + 1
            continue
        i += 1
    return mask


def is_log_event_tag(lines: list[str], idx: int, name: str) -> bool:
    """Return True if line `idx` is purely a standalone tracing/log event tag."""
    trimmed = lines[idx].strip().rstrip(",").strip()
    if trimmed != f'"{name}"':
        return False
    bal = 0
    for k in range(idx - 1, max(-1, idx - 20), -1):
        code = lines[k].split("//", 1)[0]
        for pos in range(len(code) - 1, -1, -1):
            ch = code[pos]
            if ch == ")":
                bal += 1
            elif ch == "(":
                if bal == 0:
                    head = code[:pos].strip()
                    if not head and k > 0:
                        head = lines[k - 1].split("//", 1)[0].strip()
                    return bool(LOG_MACRO_RE.search(head))
                bal -= 1
    return False


def dynamic_helper_vars(rel: str, line: str) -> list[str]:
    """Resolve suffix-based `FLUX_<suffix>` / `ATLAS_<suffix>` helper calls."""
    suffixes: list[str] = []
    suffixes.extend(ENV_SETTING_RE.findall(line))
    if rel == "crates/spark-server/src/main_modules/mem_watch.rs":
        suffixes.extend(MEM_WATCH_HELPER_RE.findall(line))
    elif rel == "crates/spark-server/src/scheduler/think_nudge.rs":
        suffixes.extend(THINK_NUDGE_GET_RE.findall(line))
    out: list[str] = []
    for suffix in suffixes:
        if suffix and not suffix.endswith("_"):
            out.append(f"ATLAS_{suffix}")
            out.append(f"FLUX_{suffix}")
    return out


def receipt_corpus(root: Path) -> str:
    """Text within 6 lines of every ENGAGED/receipt mention in production code."""
    chunks: list[str] = []
    for path in sorted((root / "crates").rglob("*.rs")):
        rel = str(path.relative_to(root))
        if (
            TEST_PATH.search(rel)
            or rel.startswith(NON_READER_PREFIXES)
            or rel in NON_READER_FILES
            or is_build_script(path)
        ):
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if RECEIPT_NEAR.search(line):
                chunks.append("\n".join(lines[max(0, i - 6) : i + 7]))
    return "\n".join(chunks)


def has_receipt(name: str, corpus: str) -> bool:
    stem = re.sub(r"^(ATLAS|FLUX)_", "", name)
    short = re.sub(r"^QWEN4_", "", stem)
    shorter = re.sub(r"^(QWEN4_)?PREFILL_", "", stem)
    return any(
        re.search(rf"(?<![A-Z0-9]){re.escape(s)}(?![A-Z0-9])", corpus)
        for s in {name, stem, short, shorter}
    )


def scan(root: Path) -> dict[str, Var]:
    found: dict[str, Var] = {}
    corpus = receipt_corpus(root)
    for path in sorted((root / "crates").rglob("*.rs")):
        rel = str(path.relative_to(root))
        if rel.startswith(NON_READER_PREFIXES) or rel in NON_READER_FILES:
            continue  # contract/config metadata tables name gates; not active readers
        if is_build_script(path):
            continue  # compile-time only; the runtime catalog must not hold these
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        whole_file_test = bool(TEST_PATH.search(rel))
        mask = test_line_mask(text, lines)
        excl_mask = exclusion_table_mask(lines, EXCLUSION_TABLE_CONSTS.get(rel))
        for i, line in enumerate(lines):
            # 1. Explicit quoted ATLAS_*/FLUX_* literals.
            for name in LITERAL.findall(line):
                if name.endswith("_"):
                    continue  # prefix fragments (e.g. ATLAS_GLM53_) are not env vars
                var = found.setdefault(name, Var(name))
                var.files.add(rel)
                if whole_file_test or mask[i]:
                    var.test_refs += 1
                    if not NOT_A_READ.search(line) and not TEST_ONLY_REF.search(line):
                        var.test_readers += 1
                    continue
                if excl_mask[i] or NOT_A_READ.search(line) or is_log_event_tag(lines, i, name):
                    continue
                var.readers.append(f"{rel}:{i + 1}")
                tail = "\n".join(lines[i : i + 4])
                var.default_on_hint |= bool(
                    re.search(r'unwrap_or\(true\)|!= "0"|unwrap_or\("1"\)|is_none_or', tail)
                )
            # 2. Dynamic prefix-helper production readers (env_setting, mem_watch, think_nudge).
            if not whole_file_test and not mask[i] and not NOT_A_READ.search(line):
                for name in dynamic_helper_vars(rel, line):
                    var = found.setdefault(name, Var(name))
                    var.files.add(rel)
                    var.readers.append(f"{rel}:{i + 1}")
    engaged_files = {
        str(p.relative_to(root))
        for p in (root / "crates").rglob("*.rs")
        if not is_build_script(p)
        and "_ENGAGED" in p.read_text(encoding="utf-8", errors="replace")
    }
    for var in found.values():
        reader_files = {r.rsplit(":", 1)[0] for r in var.readers}
        var.receipt_near = bool(var.readers) and (
            bool(reader_files & engaged_files) or has_receipt(var.name, corpus)
        )
    return found


def load_overrides(path: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        name, cls, reason = (raw.split("\t") + ["", ""])[:3]
        if cls not in CLASSES:
            raise SystemExit(f"bad class {cls!r} for {name} in {path}")
        out[name] = (cls, reason)
    return out


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def classify(var: Var, overrides: dict[str, tuple[str, str]]) -> tuple[str, str]:
    if var.name in overrides:
        cls, reason = overrides[var.name]
        return cls, f"manual: {reason}"
    if not var.readers and var.test_readers:
        return "DEAD", "rule: read only by tests/examples/benches (no production reader)"
    if not var.readers:
        return "DEAD", "rule: no reader"
    stem = re.sub(r"^(ATLAS|FLUX)_", "", var.name)
    for cls, pattern in KEYWORD_RULES:
        if re.search(pattern, stem):
            return (
                cls,
                f"rule: {cls.lower()} keyword"
                if cls != "ROUTE"
                else "rule: kernel/route selector (default)",
            )
    return "ROUTE", "rule: fallthrough"


def faithful_flags(var: Var, cls: str, prod: dict[str, str]) -> tuple[str, str]:
    if cls != "NUMERICS":
        return "", ""
    prod_value = prod.get(var.name, "")
    effective_on = (prod_value not in ("", "0")) if prod_value else var.default_on_hint
    if APPROX_NAME.search(var.name):
        return prod_value, "yes" if effective_on else "no"
    if FAITHFUL_NAME.search(var.name):
        return prod_value, "no" if effective_on else "yes"
    return prod_value, "unknown"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    check_only = False
    if "--check" in args:
        check_only = True
        args.remove("--check")
    if len(args) != 2:
        print(__doc__)
        return 2
    root, out = Path(args[0]), Path(args[1])
    out.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(Path(__file__).with_name("env_inventory_overrides.tsv"))
    prod = load_env_file(root / "recipes" / "flux-v9-max-prefill.env")
    found = scan(root)
    rows, counts, by_class = [], Counter(), defaultdict(list)
    for name in sorted(found):
        var = found[name]
        cls, basis = classify(var, overrides)
        prod_value, non_faithful = faithful_flags(var, cls, prod)
        counts[cls] += 1
        by_class[cls].append(name)
        rows.append(
            {
                "name": name,
                "class": cls,
                "basis": basis,
                "readers": len(var.readers),
                "first_reader": var.readers[0] if var.readers else "",
                "test_refs": var.test_refs,
                "receipt_near_reader": ("yes" if var.receipt_near else "no")
                if var.readers
                else "",
                "prod_recipe_value": prod_value or prod.get(name, ""),
                "non_faithful_effective": non_faithful,
            }
        )
    with (out / "env_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    catalog = "".join(f"{r['name']}\t{r['class']}\n" for r in rows if r["class"] != "DEAD")
    header = "# Generated by repro-kit/tools/env_inventory.py; regenerate, do not edit.\n"
    rendered = header + catalog
    catalog_path = root / CONTRACT_DIR / "env_catalog.tsv"
    if check_only:
        existing = catalog_path.read_text(encoding="utf-8") if catalog_path.exists() else ""
        if existing != rendered:
            print(
                f"ERROR: {catalog_path} is out of date; run repro-kit/tools/env_inventory.py",
                file=sys.stderr,
            )
            return 1
    else:
        catalog_path.write_text(rendered, encoding="utf-8")
    write_summary(out, rows, counts)
    print(" ".join(f"{c}={counts[c]}" for c in CLASSES), f"total={len(rows)}")
    return 0


def write_summary(out: Path, rows: list[dict], counts: Counter) -> None:
    numerics = [r for r in rows if r["class"] == "NUMERICS"]
    no_receipt = [r["name"] for r in numerics if r["receipt_near_reader"] == "no"]
    non_faithful = [r["name"] for r in numerics if r["non_faithful_effective"] == "yes"]
    manual = sum(1 for r in rows if r["basis"].startswith("manual"))
    active_total = sum(counts[c] for c in CLASSES if c != "DEAD")
    lines = [
        "# ATLAS_*/FLUX_* environment inventory",
        "",
        "Generated by `repro-kit/tools/env_inventory.py`. Do not edit by hand.",
        f"Distinct literals: {len(rows)} ({active_total} active runtime catalog entries, {counts['DEAD']} dead/test-only). Manually classified: {manual}.",
        "",
        "| class | count |",
        "|---|---|",
        *[f"| {c} | {counts[c]} |" for c in CLASSES],
        "",
        f"## NUMERICS gates with no receipt near any reader ({len(no_receipt)})",
        "",
        "Heuristic: the name (with or without ATLAS_/FLUX_/QWEN4_ prefix) appears within 6 lines of no `ENGAGED`/`receipt` mention in production code, and no reader file emits an `_ENGAGED` line.",
        "",
        *[f"- `{n}`" for n in no_receipt],
        "",
        f"## NUMERICS gates whose effective value is not checkpoint-faithful ({len(non_faithful)})",
        "",
        "Effective = the production `flux-v9-max-prefill.env` value when set, else the code default hint.",
        "An approximation gate (FP8/W4A8/FI/FAST/FUSED) that is on, or a faithful gate (BF16/NATIVE/EXACT) that is off.",
        "",
        *[f"- `{n}`" for n in non_faithful],
        "",
    ]
    (out / "env_inventory_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
