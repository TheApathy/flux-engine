#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""GLM-5.3 DFlash2 P3: per-layer state identity plus negative controls on flux-server.

This is the flux port of apathy window w5. Arms, each with a fresh server, all with
ATLAS_GLM53_STATE_HASH=<out>/<arm>.hash and --max-seq-len 4096 (long2k needs it):
  Th    target walk (T recipe); also runs the perturbed-prompt control
  S3h   speculative, --glm-dflash-max-drafts 3 (S recipe)
  S2h   speculative, --glm-dflash-max-drafts 2
  Cleg  S3 + ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE=1   (must FAIL state and identity)
  Cext  S3 + ATLAS_GLM53_SPEC_CONTROL_COMMIT_EXTRA_ROW=1          (must FAIL state and identity)
Control arms need the bring-up escape even after admission: admit() refuses while a control is set.

Gate (status.txt P3_PASS/P3_FAIL, exit 0 only on PASS):
  * S3h and S2h: statehash_compare.py exits 0 against Th, and every trial's text equals Th;
  * Cleg and Cext: statehash_compare.py exits 1 AND at least one trial's text differs from Th
    (a control that does not fail means the gate cannot);
  * Th's perturbed control differs from its short output.
make_evidence.py --flux then turns the P2 and P3 run directories into evidence.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ts_identity as ts  # noqa: E402

PROMPTS = ("short", "long", "prose", "code2", "think", "long2k")
ARMS = {
    "Th": ("T", None, {}),
    "S3h": ("S", 3, {}),
    "S2h": ("S", 2, {}),
    "Cleg": ("S", 3, {"ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE": "1"}),
    "Cext": ("S", 3, {"ATLAS_GLM53_SPEC_CONTROL_COMMIT_EXTRA_ROW": "1"}),
}
CONTROLS = ("Cleg", "Cext")


def plan(name: str, a, out: Path):
    kind, drafts, extra = ARMS[name]
    env, argv = ts.arm_plan(kind, a, a.port)
    i = argv.index("--max-seq-len")
    argv[i + 1] = str(a.max_seq_len)
    if drafts is not None:
        argv[argv.index("--glm-dflash-max-drafts") + 1] = str(drafts)
        if "ATLAS_GLM53_PREFIX_COMMIT_ROWS" in env:
            env["ATLAS_GLM53_PREFIX_COMMIT_ROWS"] = str(drafts)  # keep equal to the verify width
    env.update(extra)
    if name in CONTROLS:
        env["ATLAS_GLM53_UNVALIDATED_BRINGUP"] = "1"
    env["ATLAS_GLM53_STATE_HASH"] = str((out / f"{name}.hash").resolve())
    return env, argv


def compare(out: Path, cand: str, recs: dict, req_dir: Path) -> int:
    """0 equal, 1 differs, 2 malformed/unbound: the same strict, request-bound parser the evidence
    builder uses (flux_evidence.state_compare), so the P3 gate and the evidence cannot disagree."""
    import flux_evidence as fe
    try:
        eq, p, f = fe.state_compare(out, cand, recs, req_dir)
        rc, note = (0 if eq else 1), f"prefix={p} full={f} equal={eq}"
    except fe.EvidenceError as e:
        rc, note = 2, f"UNBOUND: {e}"
    (out / f"statehash-{cand}.txt").write_text(note + "\n")
    return rc


def score(out: Path, recs: dict, req_dir: Path) -> list[str]:
    fails = [f"{n}: {e}" for n, r in recs.items() for e in r["errors"]]
    ref = recs["Th"]["responses"]
    keys = [k for k in ref if not k.endswith(ts.CONTROL)]
    for name in ("S3h", "S2h"):
        rc = compare(out, name, recs, req_dir)
        if rc != 0:
            fails.append(f"{name}: state differs from the target walk (statehash rc={rc})")
        for k in keys:
            got = recs[name]["responses"].get(k)
            if got is None or got["sha"] != ref[k]["sha"]:
                fails.append(f"{name} {k}: output differs from Th")
    for name in CONTROLS:
        rc = compare(out, name, recs, req_dir)
        if rc == 2:
            fails.append(f"{name}: malformed control trace, which proves nothing (see statehash-{name}.txt)")
        elif rc != 1:
            fails.append(f"{name}: negative control did not fail the state gate")
        if all(recs[name]["responses"].get(k, {}).get("sha") == ref[k]["sha"] for k in keys):
            fails.append(f"{name}: negative control did not change any output")
    c, s = ref.get(f"trial-1-{ts.CONTROL}"), ref.get("trial-1-short")
    if not c or not s or c["sha"] == s["sha"]:
        fails.append("Th: perturbed control missing or equal to short, gate cannot fail")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="/opt/flux/bin/flux-server")
    ap.add_argument("--model", default="/models/GLM-5.3-Flash-exl3-2.05bpw")
    ap.add_argument("--draft-model")
    ap.add_argument("--out", type=Path, default=Path("runs/p3"))
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8893)
    ap.add_argument("--req-dir", type=Path, default=HERE / "req")
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--min-mem-gb", type=float, default=100)
    ap.add_argument("--watchdog-gb", type=float, default=6)
    ap.add_argument("--admitted", action="store_true")
    ap.add_argument("--pass-env", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true", help="re-score an existing --out directory (no servers)")
    a = ap.parse_args(argv)
    prompts = [p for p in a.prompts.split(",") if p]
    a.control_arms = {"Th"}
    a.out.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        for n in ARMS:
            env, argv_ = plan(n, a, a.out)
            print(f"== {n}\n" + "".join(f"  {k}={v}\n" for k, v in sorted(env.items())
                                        if not k.startswith("ATLAS_GLM53_EXL3") or "ROWBATCH" in k)
                  + "  " + " ".join(argv_))
        return 0
    if a.score_only:
        recs = {n: json.loads((a.out / n / "arm.json").read_text()) for n in ARMS}
        fails = score(a.out, recs, a.req_dir)
        binding = f"binding server_sha256={recs['Th']['server_sha256']} max_seq_len={a.max_seq_len} rescored"
        status = "P3_PASS" if not fails else "P3_FAIL " + "; ".join(fails[:20])
        (a.out / "status.txt").write_text(status + "\n" + binding + "\n")
        print(status)
        return 0 if not fails else 1
    recs = {}
    for n in ARMS:
        print(f"[{time.strftime('%H:%M:%S')}] arm {n}", flush=True)
        recs[n] = ts.run_arm(n, a, a.out, prompts, plan=plan(n, a, a.out))
    fails = score(a.out, recs, a.req_dir)
    binding = f"binding server_sha256={recs['Th']['server_sha256']} max_seq_len={a.max_seq_len}"
    if len({r["server_sha256"] for r in recs.values()}) > 1:
        fails.append("server binary changed between arms")
    status = "P3_PASS" if not fails else "P3_FAIL " + "; ".join(fails[:20])
    (a.out / "status.txt").write_text(status + "\n" + binding + "\n")
    (a.out / "summary.json").write_text(json.dumps({"fails": fails, "arms": list(ARMS)}, indent=1))
    print(status)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
