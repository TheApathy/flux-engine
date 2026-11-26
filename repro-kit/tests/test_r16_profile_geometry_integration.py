# SPDX-License-Identifier: AGPL-3.0-only
"""Plan /4 integration: per-profile geometry, the dual-recipe broker and
version pairing wired through the suite, the profile gate and the wrapper.

Synthetic CPU fixtures only (real recipe bytes for hashes). No process,
container, model or device execution; no 262K prompt is built here.
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

import profile_fixture
import profile_fixture_v2
import test_r15_exact_image_release as outer
import test_r16_live_process_broker as broker
from test_r16_prefill_decode_profiles_driver import D, S, HOST_PROVENANCE, fixture_usage

Q = outer.Q
ROOT = Path(__file__).resolve().parents[1]
BINARY = "/opt/flux/bin/server"
CHUNKS = profile_fixture_v2.CHUNKS


def recipe_sha(name):
    return hashlib.sha256((ROOT.parent / S.PROFILE_GEOMETRY_V2[name]["recipe_relpath"]).read_bytes()).hexdigest()


def contract_v2(binary=BINARY):
    profiles = {}
    for name, geometry in S.PROFILE_GEOMETRY_V2.items():
        overlay = profile_fixture_v2.overlay(geometry["chunk_tokens"])
        profiles[name] = {
            "recipe_relpath": geometry["recipe_relpath"],
            "recipe_sha256": recipe_sha(name),
            "in_container_recipe_path": "/opt/flux/" + geometry["recipe_relpath"],
            "chunk_tokens": geometry["chunk_tokens"],
            "argv_overlay": overlay,
            "launch_argv_sha256": {
                mode: S.profile_launch_template_sha256(binary, overlay, mode)
                for mode in ("none", "native-mtp-k5")
            },
        }
    return {"schema": "flux-performance-profile-contract/2", "profiles": profiles}


class SuiteContractV2(unittest.TestCase):
    def test_plan3_and_plan4_profile_subsystem_admits_contract2(self):
        S.validate_performance_profile_contract(contract_v2(), S.PLAN_SCHEMA_V3, BINARY)
        S.validate_performance_profile_contract(contract_v2(), S.PLAN_SCHEMA_V4, BINARY)
        self.assertEqual([S.profile_contract_chunk(contract_v2(), n) for n in ("interactive", "high-prefill")],
                         [4096, 16384])
        self.assertEqual(S.profile_contract_chunk({"schema": "flux-performance-profile-contract/1"}, "high-prefill"),
                         4096)

    def test_contract_plan_cross_pairs_are_refused(self):
        v1 = profile_fixture.contract(Q)
        S.validate_performance_profile_contract(v1)
        for schema in (S.PLAN_SCHEMA_V3, S.PLAN_SCHEMA_V4):
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, "schema differs"):
                S.validate_performance_profile_contract(v1, schema, BINARY)
        with self.assertRaisesRegex(ValueError, "schema differs"):
            S.validate_performance_profile_contract(contract_v2(), S.PLAN_SCHEMA, BINARY)
        with self.assertRaisesRegex(ValueError, "schema differs"):
            S.validate_performance_profile_contract(contract_v2(), "flux-r15-container-suite-plan/5", BINARY)

    def test_overlay_and_full_argv_refusals(self):
        def overlay(name):
            return lambda c: c["profiles"][name]["argv_overlay"]
        mutations = {
            "no-qsa": lambda c: overlay("high-prefill")(c).remove("--qwen4-qsa"),
            "no-policy": lambda c: overlay("interactive")(c).__setitem__(slice(7, 9), []),
            "foreign-flag": lambda c: overlay("interactive")(c).__setitem__(7, "--gpu-memory-utilization"),
            "chunk-4096-for-high-prefill": lambda c: overlay("high-prefill")(c).__setitem__(1, "4096"),
            "block-32": lambda c: overlay("interactive")(c).__setitem__(5, "32"),
            "policy-split": lambda c: overlay("high-prefill")(c).__setitem__(8, "slai"),
            "geometry-relabel": lambda c: c["profiles"]["high-prefill"].__setitem__("chunk_tokens", 4096),
            "same-recipe": lambda c: c["profiles"]["high-prefill"].__setitem__(
                "recipe_sha256", c["profiles"]["interactive"]["recipe_sha256"]),
            "foreign-path": lambda c: c["profiles"]["interactive"].__setitem__(
                "in_container_recipe_path", "/opt/flux/repro-kit/recipes/release.env"),
            "argv-hash": lambda c: c["profiles"]["interactive"]["launch_argv_sha256"].__setitem__("none", "ab" * 32),
            "third-profile": lambda c: c["profiles"].__setitem__("third", c["profiles"]["interactive"]),
        }
        for name, mutate in mutations.items():
            contract = contract_v2()
            mutate(contract)
            with self.subTest(name=name), self.assertRaises(ValueError):
                S.validate_performance_profile_contract(contract, S.PLAN_SCHEMA_V3, BINARY)
        # Policy and deadline are pinned identically; only the chunk may differ,
        # even when the split policy is re-hashed into a self-consistent argv.
        contract = contract_v2()
        row = contract["profiles"]["high-prefill"]
        row["argv_overlay"][8] = "slai"
        row["launch_argv_sha256"] = {
            mode: S.profile_launch_template_sha256(BINARY, row["argv_overlay"], mode)
            for mode in ("none", "native-mtp-k5")
        }
        # The sealed overlay now refuses the split before the pairwise check.
        with self.assertRaisesRegex(ValueError, "sealed profile recipe|outside the scheduler chunk"):
            S.validate_performance_profile_contract(contract, S.PLAN_SCHEMA_V3, BINARY)
        # A different base binary changes every bound full argv.
        with self.assertRaisesRegex(ValueError, "full launch argv"):
            S.validate_performance_profile_contract(contract_v2(), S.PLAN_SCHEMA_V3, "/opt/flux/bin/other")


class LiveBrokerV2(unittest.TestCase):
    """Broker /2 on the existing fake /proc fixture, with a real profile recipe."""

    def setUp(self):
        # Reuse the broker fixture's fake /proc and patches without
        # re-running its inherited test methods.
        broker.LiveProcessBrokerRegressions.setUp(self)
        S = broker.S
        recipes = self.root / "container-root/opt/flux/repro-kit/recipes"
        payload = (ROOT.parent / S.PROFILE_GEOMETRY_V2["high-prefill"]["recipe_relpath"]).read_bytes()
        (recipes / "flux-v9-max-prefill.env").write_bytes(payload)
        environment = b"".join(
            key + b"=" + value + b"\0" for key, value in S._recipe_environment(payload).items()
        )
        self.environment = environment
        (self.process / "environ").write_bytes(environment)
        self.argv = self.argv + b"--max-prefill-tokens\x0016384\0"
        (self.process / "cmdline").write_bytes(self.argv)
        self.plan.update(
            in_container_recipe_path="/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env",
            recipe_sha256=S.sha256_bytes(payload),
            profile="high-prefill",
            chunk_tokens=16384,
        )

    def verify2(self, profile="high-prefill", mode="native-mtp-k5"):
        S = broker.S
        return S.verify_live_server(
            {"schema": S.LIVE_REQUEST_V2, "profile": profile, "mode": mode, "host_pid": 123},
            self.plan, self.identities)

    def test_broker2_admits_the_profile_recipe_and_names_it(self):
        S = broker.S
        proof = self.verify2()
        self.assertEqual((proof["schema"], proof["profile"], proof["recipe_sha256"]),
                         (S.LIVE_PROOF_V2, "high-prefill", self.plan["recipe_sha256"]))

    def test_broker2_refuses_other_profile_wrong_recipe_and_geometry(self):
        with self.assertRaisesRegex(ValueError, "profile differs"):
            self.verify2("interactive")
        with self.assertRaisesRegex(ValueError, "profile differs"):
            self.verify2("third")
        self.plan["recipe_sha256"] = recipe_sha("interactive")
        with self.assertRaisesRegex(ValueError, "recipe hash differs"):
            self.verify2()

    def test_broker2_refuses_live_wide_chunk_and_arena_drift(self):
        for environ, argv, pattern in (
            (self.environment.replace(b"FLUX_QWEN4_PREFILL_WIDE=16384", b"FLUX_QWEN4_PREFILL_WIDE=4096"),
             self.argv, "differs from the release recipe"),
            (self.environment, self.argv.replace(b"16384", b"4096"), "scheduler chunk"),
            (self.environment + b"ATLAS_MAX_BATCH_TOKENS=65536\0", self.argv, "arena capacity"),
        ):
            (self.process / "environ").write_bytes(environ)
            (self.process / "cmdline").write_bytes(argv)
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                self.verify2()

    def test_broker1_request_cannot_name_a_profile(self):
        S = broker.S
        with self.assertRaisesRegex(ValueError, "keys differ"):
            S.verify_live_server({"schema": "flux-r15-live-process-request/1", "profile": "high-prefill",
                                  "mode": "none", "host_pid": 123}, self.plan, self.identities)


def host_v4():
    rows = [
        {
            "context_tokens": c,
            "decode_tokens": D.decode_tokens(c),
            "k5_decode_wall_seconds": D.decode_tokens(c) / (D.decode_tokens(c) / 5.0),
            "plain_decode_wall_seconds": D.decode_tokens(c) / (D.decode_tokens(c) / 7.5),
            "prefill_wall_seconds": 1.0,
            "prompt_sha256": "e" * 64,
            "k5_usage": fixture_usage(c, D.decode_budget(c), D.decode_tokens(c) / 5.0),
            "plain_usage": fixture_usage(c, D.decode_budget(c), D.decode_tokens(c) / 7.5),
            "prefill_client_monotonic": {"start": 0.0, "finish": 1.0},
        }
        for c in D.PROFILE_CONTEXTS
    ]
    contract = contract_v2()
    return {
        "schema": D.HOST_REFERENCE_SCHEMA_V4,
        "profile_chunk_tokens": dict(CHUNKS),
        "binary_sha256": "a" * 64,
        "decode_budget_tokens": 256,
        "measurement_contract": dict(D.MEASUREMENT_CONTRACT),
        "provenance": dict(HOST_PROVENANCE),
        "profiles": {
            name: {"argv_overlay": contract["profiles"][name]["argv_overlay"],
                   "host_recipe_sha256": recipe_sha(name),
                   "launch_argv_sha256": contract["profiles"][name]["launch_argv_sha256"],
                   "rows": copy.deepcopy(rows)}
            for name in D.PROFILE_NAMES
        },
    }


class GateV2(unittest.TestCase):
    def args(self, document, schema=S.PLAN_SCHEMA_V3):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "host.json"
        raw = S.canonical_bytes(document)
        path.write_bytes(raw)
        plan = {**HOST_PROVENANCE, "schema": schema, "source_root_path": str(ROOT.parent),
                "performance_profile_contract": contract_v2()}
        return types.SimpleNamespace(suite=S, launch_plan=plan, host_reference=str(path),
                                     host_reference_sha256=hashlib.sha256(raw).hexdigest())

    def image(self):
        return {"in_container_binary_path": BINARY, "in_container_binary_sha256": "a" * 64,
                "profiles": {name: {"in_container_recipe_path": "/opt/flux/" + g["recipe_relpath"],
                                    "recipe_sha256": recipe_sha(name), "environment": {}}
                             for name, g in S.PROFILE_GEOMETRY_V2.items()}}

    def test_host_reference4_and_contract2_bind_each_profile(self):
        args = self.args(host_v4())
        _document, table, _sha = D.load_host_reference(args)
        self.assertEqual(table["high-prefill"]["effective_overlay"]["--max-prefill-tokens"], 16384)
        contract = D.verify_profile_contract(args, table, self.image())
        self.assertEqual(contract["schema"], "flux-performance-profile-contract/2")
        plan = D.profile_server_plan(args, self.image(), "high-prefill")
        self.assertEqual((plan["profile"], plan["chunk_tokens"]), ("high-prefill", 16384))
        self.assertEqual(D.live_request(plan, "none", 7)["schema"], "flux-r16-live-process-request/2")
        self.assertEqual(D.live_request({"recipe_sha256": "c" * 64}, "none", 7)["schema"],
                         "flux-r15-live-process-request/1")

    def test_gate_cross_pairs_relabels_and_image_drift_are_refused(self):
        v3 = host_v4()
        v3["schema"] = D.HOST_REFERENCE_SCHEMA
        relabelled = host_v4()
        relabelled["profile_chunk_tokens"]["high-prefill"] = 4096
        swapped = host_v4()
        swapped["profiles"]["high-prefill"]["launch_argv_sha256"] = swapped["profiles"]["interactive"][
            "launch_argv_sha256"]
        for document in (v3, relabelled):
            with self.assertRaises(ValueError):
                D.load_host_reference(self.args(document))
        with self.assertRaises(ValueError):
            D.load_host_reference(self.args(host_v4(), S.PLAN_SCHEMA))
        args = self.args(swapped)
        _document, table, _sha = D.load_host_reference(args)
        with self.assertRaisesRegex(ValueError, "not bound"):
            D.verify_profile_contract(args, table, self.image())
        args = self.args(host_v4())
        _document, table, _sha = D.load_host_reference(args)
        image = self.image()
        image["profiles"]["high-prefill"]["recipe_sha256"] = recipe_sha("interactive")
        with self.assertRaisesRegex(ValueError, "image does not carry"):
            D.verify_profile_contract(args, table, image)


class WrapperV2(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path, _ = outer.make_fixture(temp.name)
        self.manifest = outer.R15ExactImageTests().load(path)
        (Path(self.manifest["runtime_dir"]) / "validators").mkdir()
        raw, evidence = profile_fixture.profile_raw(Q, self.manifest, outer.CONTAINER_ID)
        _plan, self.raw, self.evidence = profile_fixture_v2.upgrade(Q, self.manifest, raw, evidence)

    def validate(self, raw=None, evidence=None):
        raw = self.raw if raw is None else raw
        evidence = copy.deepcopy(self.evidence if evidence is None else evidence)
        target = Path(self.manifest["runtime_dir"]) / "validators/profiles.json"
        target.write_bytes(Q.canonical_bytes({
            "schema": Q.GATE_VALIDATOR_SCHEMA, "gate": "prefill-decode-profiles", "status": "PASS",
            "release_qualified": True, "container_id": outer.CONTAINER_ID,
            "run_id": self.manifest["launch"]["run_id"], "raw": raw}))
        evidence["validator_artifact"] = "validators/profiles.json"
        evidence["validator_receipt_sha256"] = Q.sha256_file(target)
        return Q._validate_profile_evidence(evidence, self.manifest, outer.CONTAINER_ID)

    def journal(self, change):
        path = Path(self.manifest["runtime_dir"]) / self.raw["response_journal"]["artifact"]
        records = [json.loads(line) for line in path.read_bytes().splitlines()]
        records = change(records)
        path.write_bytes(b"".join(Q.canonical_bytes(r) + b"\n" for r in records))
        raw = copy.deepcopy(self.raw)
        raw["response_journal"]["sha256"] = Q.sha256_file(path)
        return raw

    def test_dual_recipe_plan4_qualifies_without_mocking_the_runtime_binding(self):
        validated = self.validate()
        self.assertEqual(validated["profile_chunk_tokens"], CHUNKS)

    def test_version_cross_pairs_are_refused(self):
        raw = copy.deepcopy(self.raw)
        raw["measurement_schema"] = "flux-profile-metrics/3"
        with self.assertRaisesRegex(ValueError, "measurement schema"):
            self.validate(raw)
        raw = copy.deepcopy(self.raw)
        raw["host_reference"]["schema"] = "flux-prefill-decode-host-reference/3"
        with self.assertRaises(ValueError):
            self.validate(raw)
        evidence = copy.deepcopy(self.evidence)
        evidence["chunk_tokens"] = evidence.pop("profile_chunk_tokens")
        evidence["chunk_tokens"] = 4096
        with self.assertRaises(ValueError):
            self.validate(evidence=evidence)
        for schema in (S.PLAN_SCHEMA_V3, S.PLAN_SCHEMA_V4):
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, "cross pair"):
                Q._profile_contract(profile_fixture.contract(Q), {"schema": schema})

    def test_relabelled_rows_records_and_proofs_are_refused(self):
        changes = [
            lambda raw: raw["rows"][6].__setitem__("chunk_tokens", 4096),
            lambda raw: raw["rows"][0].__setitem__("recipe_sha256", recipe_sha("high-prefill")),
            lambda raw: raw["process_bindings"][2]["proof"].__setitem__("recipe_sha256", recipe_sha("interactive")),
            lambda raw: raw["process_bindings"][0]["proof"].__setitem__("profile", "high-prefill"),
            lambda raw: raw["profile_chunk_tokens"].__setitem__("high-prefill", 4096),
        ]
        for index, change in enumerate(changes):
            raw = copy.deepcopy(self.raw)
            change(raw)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(raw)
        def relabel(records):
            for record in records:
                if record["profile"] == "high-prefill" and record["kind"] == "ttft":
                    record["chunk_tokens"] = 4096
                    break
            return records
        with self.assertRaisesRegex(ValueError, "chunk/recipe binding"):
            self.validate(self.journal(relabel))

    def test_raw_72_record_census_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "record census"):
            self.validate(self.journal(lambda records: records[:-1]))
        with self.assertRaises(ValueError):
            self.validate(self.journal(lambda records: records[:-1] + records[:1]))
        raw = copy.deepcopy(self.raw)
        raw["response_journal"]["requests"] = 71
        with self.assertRaisesRegex(ValueError, "request census"):
            self.validate(raw)

    def test_observed_argv_must_equal_the_sealed_full_argv(self):
        raw = copy.deepcopy(self.raw)
        argv = raw["process_bindings"][1]["launch_argv"]
        argv[argv.index("--max-prefill-tokens") + 1] = "16384"
        raw["process_bindings"][1]["launch_argv_sha256"] = Q.sha256_bytes(Q.canonical_bytes(argv))
        with self.assertRaisesRegex(ValueError, "observed launch argv"):
            self.validate(raw)

    def test_image_must_carry_each_profile_recipe(self):
        self.manifest["image"]["performance_recipes"]["high-prefill"]["sha256"] = "ab" * 32
        with self.assertRaisesRegex(ValueError, "image does not carry"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
