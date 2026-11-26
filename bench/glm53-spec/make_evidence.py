#!/usr/bin/env python3
"""make_evidence.py --flux <p2-dir> <p3-dir> <harness-receipt.json> <out.json>   (flux; admits on PASS)
   make_evidence.py <identity-window> <state-window> <out.json>                (apathy w4/w5, historical)

--flux reads ts_identity.py (P2) and p3_state.py (P3) run directories, both of which must hold a
TS_PASS / P3_PASS status.txt, plus harness/run.sh's receipt.json (compiled AND executed on a GB10).

Builds the Glm53SpeculativeAdmission evidence record from two measured windows:
  identity window (e.g. w4): interleaved target/spec arms; every spec trial must equal the
    first target arm's text, and the perturbed-prompt control must differ.
  state window (e.g. w5): Th (target walk, state hashes), S3h/S2h (spec, state hashes),
    Cleg/Cext (controls that must fail both output identity and the state gate).
Counts are recomputed from the receipts, never typed by hand.
"""
import glob, json, os, re, subprocess, sys

if sys.argv[1:2] == ["--flux"]:
    FLUX = sys.argv[2:6]
    if len(FLUX) != 4:
        sys.exit(__doc__)
    ident = state = out = None
else:
    FLUX = None
    ident, state, out = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[4] if not FLUX and len(sys.argv) > 4 else os.path.abspath(os.path.join(HERE, "..", ".."))


def spec_source_hash():
    # Uses the dynamically parsed spec_sources! manifest from speculative_admission.rs.
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from pathlib import Path
    import ts_identity
    return ts_identity.spec_source_fnv1a64(Path(REPO))


def texts(arm_dir):
    r = {}
    for f in glob.glob(os.path.join(arm_dir, "trial-*.response.json")):
        m = re.match(r"trial-(\d+)-(\w+)\.response\.json", os.path.basename(f))
        j = json.load(open(f))
        if "choices" in j:
            r[(int(m.group(1)), m.group(2))] = j["choices"][0]["text"]
    return r


def arm_sha(arm_dir):
    return open(os.path.join(arm_dir, "elf.sha256")).read().split()[0]


def identity(win, ref, spec_arms):
    rt = texts(os.path.join(win, ref))
    ref_by_prompt = {p: t for (i, p), t in rt.items() if i == 1}
    same = diff = 0
    prompts = set()
    for a in spec_arms:
        for (i, p), t in texts(os.path.join(win, a)).items():
            if p == "ctlperturb" or p not in ref_by_prompt:
                continue
            prompts.add(p)
            if t == ref_by_prompt[p]:
                same += 1
            else:
                diff += 1
    ctl = rt.get((1, "ctlperturb"))
    return same, diff, sorted(prompts), ctl is not None and ctl != ref_by_prompt["short"]


def state_gate(win, cand):
    r = subprocess.run([sys.executable, os.path.join(HERE, "statehash_compare.py"),
                        os.path.join(win, "Th.hash"), os.path.join(win, f"{cand}.hash")],
                       capture_output=True, text=True)
    m = re.search(r"\{([^}]*)\}", r.stdout)
    counts = {k.strip(" '"): int(v) for k, v in (kv.split(":") for kv in m.group(1).split(","))} if m and m.group(1) else {}
    return r.returncode, counts.get("prefix", 0), counts.get("full", 0)


if FLUX:
    # A failed build must not leave an older record behind to be promoted by mistake.
    if os.path.exists(FLUX[3]):
        os.remove(FLUX[3])
    sys.path.insert(0, HERE)
    import flux_evidence
    try:
        ev = flux_evidence.build(*(__import__("pathlib").Path(x) for x in FLUX[:3]))
    except (flux_evidence.EvidenceError, OSError, KeyError, ValueError) as e:
        print(f"make_evidence: REFUSED, no evidence written: {e}", file=sys.stderr)
        sys.exit(2)
    json.dump(ev, open(FLUX[3], "w"), indent=2, sort_keys=True)
    open(FLUX[3], "a").write("\n")
    print(json.dumps(ev, indent=2, sort_keys=True))
    sys.exit(0)

w4_arms = [os.path.basename(d) for d in glob.glob(os.path.join(ident, "G*")) if os.path.exists(os.path.join(d, "timing.txt"))]
same, diff, prompts, ctl_ok = identity(ident, "T1", w4_arms)
s_same, s_diff, s_prompts, s_ctl_ok = identity(state, "Th", ["S3h", "S2h"])
states = {a: state_gate(state, a) for a in ("S3h", "S2h", "Cleg", "Cext")}
ctl_ident = {a: identity(state, "Th", [a])[1] > 0 for a in ("Cleg", "Cext")}
ev = {
    "schema": "atlas.glm53.speculative_admission.v1",
    "scope": "greedy",
    "identity_binary_sha256": arm_sha(os.path.join(ident, "T1")),
    "state_binary_sha256": arm_sha(os.path.join(state, "Th")),
    "identity_prompts": prompts,
    "identity_trials_equal": same + s_same,
    "identity_trials_differ": diff + s_diff,
    "perturbed_prompt_control_differs": bool(ctl_ok and s_ctl_ok),
    "state_prefix_positions_equal": states["S3h"][1] + states["S2h"][1],
    "state_full_positions_equal": states["S3h"][2] + states["S2h"][2],
    "state_gate_passed": states["S3h"][0] == 0 and states["S2h"][0] == 0,
    "state_layers": "kda 0/17/33 conv+recurrent; dsa 0..10 latent/pools/tail + pool-ahead validity",
    "control_legacy_restage_fails_state": states["Cleg"][0] == 1,
    "control_legacy_restage_fails_identity": ctl_ident["Cleg"],
    "control_extra_row_fails_state": states["Cext"][0] == 1,
    "control_extra_row_fails_identity": ctl_ident["Cext"],
    "kernel_harness_bit_identical": "RESULT: BIT-IDENTICAL" in open(os.path.join(ident, "..", "w3", "pre_cmd.log")).read(),
    "chunked_prefill_covered": False,
    "spec_source_fnv1a64": spec_source_hash(),
}
json.dump(ev, open(out, "w"), indent=2, sort_keys=True)
open(out, "a").write("\n")
print(json.dumps(ev, indent=2, sort_keys=True))
