# SPDX-License-Identifier: AGPL-3.0-only
"""Strict flux evidence builder (used by make_evidence.py --flux).

Everything is recomputed from raw receipts; nothing is taken from a summary. Any missing,
foreign, unfinished or inconsistent input raises EvidenceError, so no record is written.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ts_identity as ts  # noqa: E402

P3_ARMS = ("Th", "S3h", "S2h", "Cleg", "Cext")
P3_SPEC = ("S3h", "S2h")
P3_CONTROLS = ("Cleg", "Cext")
STATE_LAYERS = "kda 0/17/33 conv+recurrent; dsa 0..10 latent/pools/tail + pool-ahead validity"


class EvidenceError(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise EvidenceError(msg)


def load_run(d: Path, pass_word: str) -> tuple[str, dict]:
    lines = (d / "status.txt").read_text().splitlines() if (d / "status.txt").exists() else []
    need(lines and lines[0] == pass_word, f"{d}: status is not {pass_word}")
    m = re.search(r"server_sha256=([0-9a-f]{64})", lines[1] if len(lines) > 1 else "")
    need(m, f"{d}: status binding has no server sha256")
    sha = m.group(1)
    arms = {}
    for f in sorted(d.glob("*/arm.json")):
        r = json.loads(f.read_text())
        need(r["arm"] == f.parent.name, f"{f}: arm name mismatch")
        need(not r["errors"], f"{d}/{r['arm']}: arm errors {r['errors'][:3]}")
        need(r["server_sha256"] == sha, f"{d}/{r['arm']}: binary differs from the status binding")
        for key, v in r["responses"].items():
            raw = f.parent / f"{key}.response.json"
            need(raw.exists(), f"{raw}: raw response missing")
            body = json.loads(raw.read_text())
            ch = (body.get("choices") or [{}])[0]
            text = ch.get("text")
            need(isinstance(text, str), f"{raw}: no text")
            need(hashlib.sha256(text.encode()).hexdigest() == v["sha"], f"{raw}: text does not match its arm record")
            ct = (body.get("usage") or {}).get("completion_tokens")
            need(ch.get("finish_reason") in ("stop", "length"), f"{raw}: finish_reason {ch.get('finish_reason')!r}")
            need(type(ct) is int and ct > 0, f"{raw}: completion_tokens {ct!r}")
            need(v.get("completion_tokens") == ct and v.get("finish") == ch.get("finish_reason"),
                 f"{raw}: arm record disagrees with the raw response")
        arms[r["arm"]] = r
    return sha, arms


def keys_for(r: dict) -> set:
    return {k for k in r["responses"] if not k.endswith(ts.CONTROL)}


# Exactly what Glm53Exl3TargetModel::state_hash_probe writes (target_model_exl3.rs).
KDA_KEYS = tuple(f"o{o}.{n}" for o in (0, 17, 33) for n in ("conv", "rec"))
DSA_KEYS = tuple(f"d{i}" for i in range(11))
AHEAD_KEYS = tuple(f"d{i}ahead" for i in range(11))
SOURCES = {"walk", "prefix", "full"}
HEX16 = re.compile(r"[0-9a-f]{16}")


def load_state(path: Path) -> list[dict]:
    """Strict parse into requests, grouped by the probe's req (sequence generation) field."""
    need(path.exists() and path.stat().st_size > 0, f"{path} missing or empty")
    reqs, cur, last, cur_req = [], {}, None, None
    for n, line in enumerate(path.read_text().splitlines(), 1):
        pairs = [kv.split("=", 1) for kv in line.split()]
        need(all(len(p) == 2 for p in pairs), f"{path}:{n}: malformed field")
        need(len({k for k, _ in pairs}) == len(pairs), f"{path}:{n}: duplicate state field")
        f = dict(pairs)
        need(set(f) - {"req", "pos", "src"} - set(KDA_KEYS) - set(DSA_KEYS) - set(AHEAD_KEYS) == set(),
             f"{path}:{n}: unknown state fields")
        need(all(k in f for k in ("req", "pos", "src", *KDA_KEYS, *DSA_KEYS)), f"{path}:{n}: missing KDA/DSA state fields")
        need(f["req"].isdigit() and f["pos"].isdigit() and f["src"] in SOURCES, f"{path}:{n}: bad req/pos/src")
        need(all(HEX16.fullmatch(f[k]) for k in (*KDA_KEYS, *DSA_KEYS)), f"{path}:{n}: state hash is not 16 hex")
        # The probe omits dNahead only once the pool buffer is full, which a <= 4096-position run never
        # reaches, so every DSA layer must report its next-pool validity byte.
        need(all(k in f and f[k].isdigit() and 0 <= int(f[k]) <= 255 for k in AHEAD_KEYS),
             f"{path}:{n}: missing or out-of-range ahead byte")
        req, pos = int(f.pop("req")), int(f.pop("pos"))
        if req != cur_req:
            need(cur_req is None or req > cur_req, f"{path}:{n}: request id went backwards")
            if cur:
                reqs.append(cur)
            cur, last, cur_req = {}, None, req
        need(last is None or pos > last, f"{path}:{n}: position did not increase within request {req}")
        cur[pos] = f
        last = pos
    if cur:
        reqs.append(cur)
    return reqs


def request_manifest(arm: dict, req_dir: Path) -> list[tuple[str, int, int]]:
    """(prompt, prompt length, completion tokens) for every request the arm sent, in order: the
    warm-up (short), then the recorded responses in run order (the control last, where it runs)."""
    warm = arm.get("warmup") or {}
    ct = warm.get("completion_tokens")
    need(type(ct) is int and ct > 0, f"{arm['arm']}: warm-up completion count missing or not a positive int ({ct!r})")
    need(warm.get("finish") in ("stop", "length"), f"{arm['arm']}: warm-up finish_reason {warm.get('finish')!r}")
    rows = [("short", warm["completion_tokens"])] + [(k.split("-", 2)[2], v["completion_tokens"])
                                                       for k, v in arm["responses"].items()]
    return [(p, len(json.loads((req_dir / f"request-{p}.json").read_text())["prompt"]), ct) for p, ct in rows]


def expected_traces(man: list) -> list[tuple[str, int]]:
    """(prompt, final committed position on the target walk) of every request that probes.

    From target_model_exl3.rs: prefill does not probe; each decode step feeds the token at
    `position` and probes `position + 1`; the last emitted token is never fed back. So on the walk
    a request with prompt P and C completion tokens ends at P + C - 1, and C == 1 leaves no trace.
    Speculative arms may end one later: verify_policy_transaction.rs select() counts a terminal
    token that matches its draft as accepted before breaking, and the commit takes
    accepted_drafts + 1 rows, so the terminal token itself is committed (see spec_end_ok)."""
    return [(p, plen + ct - 1) for p, plen, ct in man if ct > 1]


def spec_end_ok(end: int, walk_end: int) -> bool:
    """A speculative request ends where the walk would, or one later when its terminal token was a
    matching draft and got committed with the accepted prefix."""
    return end in (walk_end, walk_end + 1)


def trace_shape_ok(name: str, trace: list, arm: dict, req_dir: Path) -> tuple[bool, str]:
    exp = expected_traces(request_manifest(arm, req_dir))
    walk = arm.get("kind") == "T"
    if len(trace) != len(exp):
        return False, f"{name}: {len(trace)} state requests, expected {len(exp)} from the requests it sent"
    for i, ((prompt, end), r) in enumerate(zip(exp, trace)):
        if (max(r) != end) if walk else not spec_end_ok(max(r), end):
            allowed = f"{end}" if walk else f"{end} or {end + 1}"
            return False, f"{name} request {i} ({prompt}): ends at {max(r)}, its completion count implies {allowed}"
    return True, ""


def state_compare(d: Path, cand: str, arms: dict, req_dir: Path) -> tuple[bool, int, int]:
    """(equal on every compared state field, prefix positions compared, full positions compared).

    Both traces must first be well formed: one per committing request the arm sent, in order,
    each ending exactly where its own completion count says (so uniform truncation across all
    arms is caught). A malformed trace raises EvidenceError: it is never read as divergence,
    which a negative control needs to prove."""
    ref, got = load_state(d / "Th.hash"), load_state(d / f"{cand}.hash")
    need(all(v["src"] == "walk" for r in ref for v in r.values()), "Th.hash: target reference has non-walk commits")
    for name, trace in (("Th", ref), (cand, got)):
        ok, why = trace_shape_ok(name, trace, arms[name], req_dir)
        need(ok, f"malformed state trace: {why}")
    man_c = [p for p, _ in expected_traces(request_manifest(arms[cand], req_dir))]
    man_t = [p for p, _ in expected_traces(request_manifest(arms["Th"], req_dir))]
    need(man_c == man_t[:len(man_c)], f"{cand}: request order differs from Th")
    equal, counts = True, {"walk": 0, "prefix": 0, "full": 0}
    for a, b in zip(ref, got):
        for pos in set(a) & set(b):
            counts[b[pos]["src"]] += 1
            equal &= all(a[pos][k] == b[pos][k] for k in (*KDA_KEYS, *DSA_KEYS, *AHEAD_KEYS))
    return equal, counts["prefix"], counts["full"]


EXLLAMAV3_REV = "e648f1a131365aae15920073e761a3fa5a527654"  # pinned by atlas-kernels build.rs
HARNESS_KERNEL = "kernels/gb10/glm5.3-flash/exl3/glm53_exl3_k4_cb2.cu"


def check_harness(receipt: Path) -> None:
    """harness/run.sh receipt: compiled AND executed on a GB10 from this tree's sources."""
    need(receipt.suffix == ".json" and receipt.exists(), f"{receipt}: need harness/run.sh receipt.json")
    r = json.loads(receipt.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    need(r.get("schema") == "atlas.glm53.rowbatch_harness.v1", "harness receipt: unknown schema")
    need(r.get("build_rc") == 0 and r.get("executed") is True and r.get("run_rc") == 0,
         "harness receipt: not built and executed successfully")
    need(r.get("result") == "BIT-IDENTICAL" and r.get("failing_cases") == 0, "harness receipt: kernel not bit-identical")
    need(r.get("harness_source_sha256") == sha(HERE / "harness/rowbatch_equiv.cu"),
         "harness receipt: rowbatch_equiv.cu changed since it ran")
    need(r.get("kernel_source") == HARNESS_KERNEL and r.get("kernel_source_sha256") == sha(ts.REPO / HARNESS_KERNEL),
         "harness receipt: kernel source changed since it ran")
    need(r.get("spec_source_fnv1a64") == ts.spec_source_fnv1a64(), "harness receipt: ran on other speculation sources")
    need(r.get("exllamav3_rev") == EXLLAMAV3_REV and r.get("exllamav3_dirty") is False,
         "harness receipt: exllamav3 is not a clean checkout of the pinned revision")
    need(r.get("result_lines") == 1, "harness receipt: not exactly one RESULT summary")
    need(isinstance(r.get("binary_sha256"), str) and len(r["binary_sha256"]) == 64, "harness receipt: no binary hash")
    need("GB10" in (r.get("gpu") or ""), "harness receipt: not executed on a GB10")


def build(p2: Path, p3: Path, harness_receipt: Path, req_dir: Path = HERE / "req") -> dict:
    sha2, a2 = load_run(p2, "TS_PASS")
    sha3, a3 = load_run(p3, "P3_PASS")
    need(sha2 == sha3, "P2 and P3 ran different binaries")
    tree = ts.spec_source_fnv1a64()
    for run, arms in (("P2", a2), ("P3", a3)):
        for r in arms.values():
            if r["kind"] == "S":
                need(r.get("spec_source_fnv1a64") == tree,
                     f"{run}/{r['arm']}: binary logged sources {r.get('spec_source_fnv1a64')}, tree is {tree}")
    # P2: identity
    ts_arms = sorted((r for r in a2.values() if r["kind"] == "T"), key=lambda r: r["arm"])
    ss_arms = [r for r in a2.values() if r["kind"] == "S"]
    need(len(ts_arms) >= 2 and len(ss_arms) >= 2, "P2 needs at least 2 T and 2 S arms")
    ref = ts_arms[0]
    keys = keys_for(ref)
    prompts = sorted({k.split("-", 2)[2] for k in keys})
    need(set(prompts) >= set(ts.PROMPTS), f"P2 covers prompts {prompts}, needs {list(ts.PROMPTS)}")
    same = 0
    for r in ts_arms[1:] + ss_arms:
        need(keys_for(r) == keys, f"P2/{r['arm']}: response set differs from {ref['arm']}")
        for k in keys:
            need(r["responses"][k]["sha"] == ref["responses"][k]["sha"], f"P2/{r['arm']} {k}: differs from {ref['arm']}")
            same += r["kind"] == "S"
    p2_ctl = [r for r in a2.values() if f"trial-1-{ts.CONTROL}" in r["responses"]]
    need({r["kind"] for r in p2_ctl} == {"T", "S"}, "P2: perturbed control must run on a T and an S arm")
    for r in p2_ctl:
        need(r["responses"][f"trial-1-{ts.CONTROL}"]["sha"] != r["responses"]["trial-1-short"]["sha"],
             f"P2/{r['arm']}: perturbed control equals short output")
    # P3: state identity and controls
    need(set(a3) == set(P3_ARMS), f"P3 arms {sorted(a3)} != {list(P3_ARMS)}")
    th = a3["Th"]
    tkeys = keys_for(th)
    need(tkeys, "P3/Th has no responses")
    pre = full = 0
    for n in P3_SPEC:
        eq, p, f = state_compare(p3, n, a3, req_dir)
        need(eq, f"P3/{n}: state differs from the target walk")
        need(p > 0 and f > 0, f"P3/{n}: no prefix or full commits were compared")
        need(keys_for(a3[n]) == tkeys and all(a3[n]["responses"][k]["sha"] == th["responses"][k]["sha"] for k in tkeys),
             f"P3/{n}: output differs from Th")
        pre, full = pre + p, full + f
    for n in P3_CONTROLS:
        eq, p, f = state_compare(p3, n, a3, req_dir)
        need(p + f > 0, f"P3/{n}: no speculative commits to compare")
        need(not eq, f"P3/{n}: negative control did not fail the state gate")
        need(keys_for(a3[n]) == tkeys, f"P3/{n}: incomplete responses, a missing output is not a failed control")
        need(any(a3[n]["responses"][k]["sha"] != th["responses"][k]["sha"] for k in tkeys),
             f"P3/{n}: negative control changed no output")
    c, s = th["responses"].get(f"trial-1-{ts.CONTROL}"), th["responses"].get("trial-1-short")
    need(c and s and c["sha"] != s["sha"], "P3/Th: perturbed control missing or equal to short")
    check_harness(harness_receipt)
    return {
        "schema": "atlas.glm53.speculative_admission.v1", "engine": "flux-engine", "scope": "greedy",
        "identity_binary_sha256": sha2, "state_binary_sha256": sha3,
        "identity_prompts": prompts, "identity_trials_equal": same, "identity_trials_differ": 0,
        "perturbed_prompt_control_differs": True,
        "state_prefix_positions_equal": pre, "state_full_positions_equal": full, "state_gate_passed": True,
        "state_layers": STATE_LAYERS,
        "control_legacy_restage_fails_state": True, "control_legacy_restage_fails_identity": True,
        "control_extra_row_fails_state": True, "control_extra_row_fails_identity": True,
        "kernel_harness_bit_identical": True, "chunked_prefill_covered": False,
        "spec_source_fnv1a64": tree,
    }
