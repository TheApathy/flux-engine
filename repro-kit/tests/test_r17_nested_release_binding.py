# SPDX-License-Identifier: AGPL-3.0-only
"""Source-only nested release preflight regressions. No model or Docker work."""

import ast
import copy
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import r9_source_to_binary as P
import qwen4_native_concurrency_probe as C
import qwen4_mtp_long_context_qualify as M
import qwen4_native_262k_qualify as N


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path, loader=SourceFileLoader(name, str(path)))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


S = module("r17_nested_suite", KIT / "r15-container-suite.py")
D = module("r17_nested_concurrency", KIT / "r15-gates/native-concurrency/native-concurrency-c2-c8")
L = module("r17_nested_plain_k5", KIT / "r15-gates/plain-k5-long-context")
VERSION = "v0.9.13-rc.r17.20261024"


class NestedReleaseTests(unittest.TestCase):
    def setUp(self):
        for target in ("subprocess.run", "subprocess.Popen", "os.fork",
                       "urllib.request.OpenerDirector.open", "socket.create_connection"):
            patch = mock.patch(target, side_effect=AssertionError("workload boundary denied"))
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(P.configure_release, P.DEFAULT_RELEASE_VERSION)

    def fixture(self, root, version=VERSION):
        p = json.loads((KIT / "examples/r15-exact-image-launch.example.json").read_bytes())["provenance"]
        p = json.loads(json.dumps(p).replace("0" * 64, "1" * 64).replace("0" * 40, "1" * 40))
        p["release_version"] = version
        p["checkpoint_id"] = "hf://JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"
        for key in p:
            if key.endswith("_path"):
                p[key] = str(root / key)
        Path(p["source_root_path"]).mkdir()
        Path(p["checkpoint_path"]).mkdir()
        for stem in ("source_bundle_manifest", "source_bundle", "build_receipt", "binary"):
            raw = b"{}\n" if stem != "binary" else b"synthetic ELF fixture\n"
            Path(p[stem + "_path"]).write_bytes(raw)
            p[stem + "_sha256"] = hashlib.sha256(raw).hexdigest()
        expected = {"release_version": version,
                    "source_bundle_manifest_sha256": p["source_bundle_manifest_sha256"],
                    "source_bundle_sha256": p["source_bundle_sha256"],
                    "source_file_count": 1,
                    "build": {"receipt_sha256": p["build_receipt_sha256"], "log_sha256": "2" * 64},
                    "binary": {"sha256": p["binary_sha256"]}}
        self.write_receipt(p, expected)
        return p, expected

    def write_receipt(self, provenance, receipt):
        raw = json.dumps(receipt, sort_keys=True).encode()
        Path(provenance["source_to_binary_receipt_path"]).write_bytes(raw)
        provenance["source_to_binary_receipt_sha256"] = hashlib.sha256(raw).hexdigest()

    def closure_boundaries(self, expected):
        # Keep real file/hash/receipt equality checks; replace archive/build work.
        def receipt(*_args):
            result = copy.deepcopy(expected)
            result["release_version"] = P.RELEASE_VERSION
            return result
        return (mock.patch.object(P, "validate_source_bundle", return_value={"file_count": 1}),
                mock.patch.object(P, "validate_build_receipt", return_value={"build": {"log_sha256": "2" * 64}}),
                mock.patch.object(P, "make_source_to_binary_receipt", side_effect=receipt))

    def test_dated_candidate_completes_concurrency_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, expected = self.fixture(root)
            launch = root / "launch.json"
            launch.write_bytes(S.canonical_bytes({"release_version": VERSION, "provenance": p}))
            args = types.SimpleNamespace(launch_manifest=str(launch),
                                         launch_manifest_sha256=hashlib.sha256(launch.read_bytes()).hexdigest())
            plan = {"in_container_binary_sha256": p["binary_sha256"],
                    "recipe_sha256": p["recipe_sha256"], "in_container_binary_path": "/opt/flux/bin/flux-server"}
            with self.closure_boundaries(expected)[0], self.closure_boundaries(expected)[1], \
                    self.closure_boundaries(expected)[2], \
                    mock.patch.object(C, "validate_checkpoint_inventory", return_value=({"files": [], "total_bytes": 0}, (), {})):
                proof = D.preflight_provenance(args, S, C, plan, {"/models": {"Source": p["checkpoint_path"]}})
            self.assertEqual(proof.public["release_version"], VERSION)
            self.assertEqual(proof.source_closure["release_version"], VERSION)
            self.assertEqual(P.RELEASE_VERSION, P.DEFAULT_RELEASE_VERSION)
            launch.write_bytes(S.canonical_bytes({"release_version": "v0.9.13-r15", "provenance": p}))
            args.launch_manifest_sha256 = hashlib.sha256(launch.read_bytes()).hexdigest()
            with mock.patch.object(C, "validate_r15_source_closure") as closure:
                with self.assertRaisesRegex(ValueError, "differs from the sealed launch manifest"):
                    D.preflight_provenance(args, S, C, plan, {"/models": {"Source": p["checkpoint_path"]}})
                closure.assert_not_called()

    def test_all_nested_source_closures_bind_exact_receipt_version(self):
        for version in (VERSION, "v0.9.13-r15"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                p, expected = self.fixture(Path(directory), version)
                functions = (lambda: C.validate_r15_source_closure(p),
                             lambda: M.validate_source_to_binary_receipt(Path(p["source_to_binary_receipt_path"]), p, True),
                             lambda: N.validate_source_to_binary_receipt(Path(p["source_to_binary_receipt_path"]), p, True))
                with self.closure_boundaries(expected)[0], self.closure_boundaries(expected)[1], self.closure_boundaries(expected)[2]:
                    for function in functions:
                        function()
                        self.assertEqual(P.RELEASE_VERSION, P.DEFAULT_RELEASE_VERSION)
                    other = copy.deepcopy(expected)
                    other["release_version"] = "v0.9.13-r16"
                    self.write_receipt(p, other)
                    for function in functions:
                        with self.assertRaisesRegex((ValueError, C.ProbeError), "receipt differs from verified inputs"):
                            function()
                        self.assertEqual(P.RELEASE_VERSION, P.DEFAULT_RELEASE_VERSION)

    def test_invalid_nested_versions_fail_before_source_work(self):
        for invalid in (None, True, VERSION + "\n", "v0.9.13-rc.r17.20260230"):
            with mock.patch.object(P, "validate_qualification_provenance") as validate:
                with self.assertRaises((ValueError, C.ProbeError)):
                    C.validate_r15_source_closure({"release_version": invalid})
                validate.assert_not_called()
            self.assertEqual(P.RELEASE_VERSION, P.DEFAULT_RELEASE_VERSION)

    def test_all_release_recipes_pin_fixed_k5_and_exact_runtime_logging(self):
        # flux-v9-max-prefill.env ships the MTP confidence gate (0.6 / early) since 9.15.3.1,
        # so it is no longer a fixed-K5 recipe; see test_max_prefill_recipe_ships_the_confidence_gate.
        names = ("flux-v9-w4096-interactive.env",
                 "flux-qwen4-native-262k-bf16.env",
                 "flux-qwen4-native-262k-w4096-bf16-owner16.env",
                 "flux-qwen4-native-concurrency-c8-owner16.env")
        for name in names:
            with self.subTest(recipe=name):
                recipe = {}
                for line in (KIT / "recipes" / name).read_text().splitlines():
                    if not line or line.startswith("#"):
                        continue
                    key, value = line.split("=", 1)
                    self.assertNotIn(key, recipe)
                    recipe[key] = value
                for key in L.FIXED_K5_ENV:
                    self.assertEqual(recipe[key], "0", key)
                self.assertFalse(any(key.endswith("MTP_CONF_FILE") for key in recipe))
                self.assertEqual(L.k5_draft_policy(recipe), L.FIXED_K5_POLICY)
                for mode in ("none", "native-mtp-k5"):
                    environment = L.receipt_environment(recipe, mode, {})
                    self.assertEqual(environment["RUST_LOG"], recipe["RUST_LOG"])
                    self.assertTrue(all(environment[key] == value for key, value in recipe.items()))
                directives = recipe["RUST_LOG"].split(",")
                for required in L.RUST_LOG_RECEIPT_DIRECTIVES.split(",")[1:]:
                    self.assertEqual(directives.count(required), 1)

    def test_max_prefill_recipe_ships_the_confidence_gate(self):
        recipe = {}
        for line in (KIT / "recipes" / "flux-v9-max-prefill.env").read_text().splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                recipe[key] = value
        for prefix in ("ATLAS_QWEN4_", "FLUX_QWEN4_"):
            self.assertEqual(recipe[prefix + "MTP_CONF"], "0.6")
            self.assertEqual(recipe[prefix + "MTP_CONF_EARLY"], "1")
            self.assertEqual(recipe[prefix + "MTP_VOCAB_PRUNE"], "0")
        self.assertFalse(any(key.endswith("MTP_CONF_FILE") for key in recipe))
        # The 9.15.3.1 recipe qualifies as confidence-gated K5: the native arm keeps its
        # gate selectors instead of being forced to fixed K5.
        self.assertEqual(L.k5_draft_policy(recipe), L.GATED_K5_POLICY)
        for mode in ("none", "native-mtp-k5"):
            environment = L.receipt_environment(recipe, mode, {})
            self.assertTrue(all(environment[key] == value for key, value in recipe.items()))
            self.assertEqual(environment["ATLAS_DISABLE_SPECULATION"], "1" if mode == "none" else "0")

    def test_gated_k5_refuses_malformed_or_unqualified_policy(self):
        gated = {prefix + suffix: value for prefix in L.K5_POLICY_PREFIXES
                 for suffix, value in (("MTP_CONF", "0.6"), ("MTP_CONF_EARLY", "1"),
                                       ("MTP_VOCAB_PRUNE", "0"))}
        self.assertEqual(L.k5_draft_policy(gated), L.GATED_K5_POLICY)
        self.assertEqual(L.k5_draft_policy(dict(gated, ATLAS_QWEN4_MTP_CONF_EARLY="0",
                                                FLUX_QWEN4_MTP_CONF_EARLY="0")), L.GATED_K5_POLICY)
        both = lambda suffix, value: {prefix + suffix: value for prefix in L.K5_POLICY_PREFIXES}
        refused = [
            *(both("MTP_CONF", value) for value in
              ("1.5", "", "abc", "1", "1.0", "0.0", "0.00", "-0.5", ".6", "0.6 ", " 0.6",
               "0.60001", "6e-1", "nan", "0,6")),
            *(both("MTP_CONF_EARLY", value) for value in ("2", "", "true", "01")),
            *(both("MTP_VOCAB_PRUNE", value) for value in ("1", "", "abc")),
            {"ATLAS_QWEN4_MTP_CONF": "0.5"},
            {"FLUX_QWEN4_MTP_CONF_EARLY": "0"},
            {"ATLAS_QWEN4_QSA_JOURNAL_ALL_K": "0"},
            {"FLUX_QWEN4_QSA_JOURNAL_ALL_K": ""},
            {"ATLAS_QWEN4_MTP_GATE_QSA_MIN_WIDTH": "5"},
        ]
        for change in refused:
            for mode in ("none", "native-mtp-k5"):
                with self.subTest(change=change, mode=mode), self.assertRaisesRegex(ValueError, "K5"):
                    L.receipt_environment(dict(gated, **change), mode, {})
        removed = dict(gated)
        del removed["FLUX_QWEN4_MTP_CONF"]
        with self.assertRaisesRegex(ValueError, "differ or are missing"):
            L.k5_draft_policy(removed)
        for key in ("ATLAS_QWEN4_MTP_CONF_FILE", "FLUX_QWEN4_MTP_CONF_FILE"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mutable"):
                L.receipt_environment(dict(gated, **{key: ""}), "native-mtp-k5", {})
        self.assertEqual(L.k5_draft_policy(dict(gated, ATLAS_QWEN4_QSA_JOURNAL_ALL_K="1")),
                         L.GATED_K5_POLICY)

    def test_fixed_k5_environment_refuses_shortening_pruning_and_mutable_policy(self):
        for key in L.FIXED_K5_ENV:
            for value in ("1", "0.6", "", "false"):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                        ValueError, "neither fixed nor confidence-gated"):
                    L.receipt_environment({key: value}, "native-mtp-k5", {})
        for key in ("ATLAS_QWEN4_MTP_CONF_FILE", "FLUX_QWEN4_MTP_CONF_FILE"):
            for mode in ("none", "native-mtp-k5"):
                with self.subTest(key=key, mode=mode), self.assertRaisesRegex(ValueError, "mutable"):
                    L.receipt_environment({key: ""}, mode, {})
        for recipe in ({}, dict(L.FIXED_K5_ENV), {"ATLAS_QWEN4_MTP_CONF": "0"}):
            with self.subTest(recipe=recipe):
                self.assertEqual(L.k5_draft_policy(recipe), L.FIXED_K5_POLICY)
                environment = L.receipt_environment(recipe, "native-mtp-k5", {})
                self.assertTrue(all(environment[key] == "0" for key in L.FIXED_K5_ENV))
        # Zero confidence with early stop set is neither policy.
        with self.assertRaisesRegex(ValueError, "neither fixed nor confidence-gated"):
            L.receipt_environment({**L.FIXED_K5_ENV, "ATLAS_QWEN4_MTP_CONF_EARLY": "1",
                                   "FLUX_QWEN4_MTP_CONF_EARLY": "1"}, "native-mtp-k5", {})

    def test_every_release_recipe_pins_identical_sixteen_mib_owner_policy(self):
        expected = {prefix + suffix: value
                    for prefix in ("ATLAS_QWEN4_", "FLUX_QWEN4_")
                    for suffix, value in (("BOUNDED_TRANSPOSE_OWNERS", "1"),
                                          ("BOUNDED_TRANSPOSE_OWNER_CAP_MIB", "16"))}
        names = ("flux-v9-w4096-interactive.env", "flux-v9-max-prefill.env",
                 "flux-qwen4-native-262k-bf16.env",
                 "flux-qwen4-native-262k-w4096-bf16-owner16.env",
                 "flux-qwen4-native-concurrency-c8-owner16.env")
        for name in names:
            with self.subTest(recipe=name):
                raw = (KIT / "recipes" / name).read_bytes()
                parsed = S._recipe_environment(raw)
                values = {key.decode(): value.decode() for key, value in parsed.items()}
                self.assertEqual({key: values[key] for key in expected}, expected)
                directives = values["RUST_LOG"].split(",")
                target = "atlas_bounded_transpose_allocation"
                self.assertEqual([item for item in directives if item.split("=", 1)[0] == target],
                                 [target + "=debug"])
                for mode in ("none", "native-mtp-k5"):
                    environment = L.receipt_environment(values, mode, {})
                    self.assertEqual({key: environment[key] for key in expected}, expected)
                    self.assertEqual(environment["RUST_LOG"], values["RUST_LOG"])

    def test_logging_requires_each_exact_runtime_target_once(self):
        complete = L.RUST_LOG_RECEIPT_DIRECTIVES
        for required in complete.split(",")[1:]:
            for altered in (complete.replace("," + required, ""),
                            complete + "," + required,
                            complete.replace(required, required.replace("=debug", "=info")),
                            complete + "," + required.replace("=debug", "=off")):
                with self.subTest(logging=altered), self.assertRaisesRegex(ValueError, "exact receipt target"):
                    L.receipt_environment({"RUST_LOG": altered}, "native-mtp-k5", {})
        extended = complete + ",atlas_bounded_transpose_allocation=debug"
        self.assertEqual(L.receipt_environment({"RUST_LOG": extended}, "native-mtp-k5", {})["RUST_LOG"], extended)

    def test_plain_k5_qualifier_pins_and_example_job_hashes_match(self):
        for name, expected in L.QUALIFIER_PINS:
            local = KIT / (name + ".py")
            self.assertTrue(local.is_file(), f"missing pinned qualifier {local}")
            self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(), expected)
        plan = json.loads((KIT / "examples/r15-container-suite-plan.example.json").read_bytes())
        for name in ("plain-k5-long-context", "native-concurrency-c2-c8"):
            row = plan["gate_jobs"][name]
            relative = Path(row["program_path"]).relative_to("/results/r15-source/clean-tree/repro-kit")
            self.assertEqual(hashlib.sha256((KIT / relative).read_bytes()).hexdigest(), row["program_sha256"])
        self.assertEqual(len(S.REQUIRED_GATES), 12)
        self.assertEqual(D.MODES, ("none", "native-mtp-k5"))
        self.assertEqual(D.LEVELS, (2, 4, 8))

    def test_all_example_source_arguments_and_recipe_contracts_match_actual_bytes(self):
        plan = json.loads((KIT / "examples/r15-container-suite-plan.example.json").read_bytes())
        source_prefix = "/results/r15-source/clean-tree/"
        internal_pins = {"--lifecycle-program": "LIFECYCLE_SHA256",
                         "--microgate-program": "MICROGATE_SHA256", "--tool-manifest": "MANIFEST_SHA256",
                         "--lil-program": "LIL_SHA256", "--endpoint-program": "ENDPOINT_SHA256",
                         "--engagement-program": "ENGAGEMENT_SHA256"}
        for name, row in plan["gate_jobs"].items():
            program = KIT.parent / Path(row["program_path"]).relative_to(source_prefix)
            self.assertEqual(hashlib.sha256(program.read_bytes()).hexdigest(), row["program_sha256"],
                             "stale example pin gate_jobs." + name + ".program_sha256")
            literals = {target.id: node.value.value for node in ast.parse(program.read_bytes()).body
                        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                        for target in node.targets if isinstance(target, ast.Name)}
            for index, value in enumerate(row["argv"]):
                if index == 0 or not value.startswith(source_prefix):
                    continue
                target = KIT.parent / Path(value).relative_to(source_prefix)
                if target.is_dir():
                    continue
                self.assertTrue(target.is_file(), value)
                flag = row["argv"][index - 1]
                choices = [flag + "-sha256"]
                if flag.endswith("-program"):
                    choices.append(flag[:-len("-program")] + "-sha256")
                matches = [choice for choice in choices if choice in row["argv"]]
                self.assertLessEqual(len(matches), 1, flag)
                if matches:
                    expected = row["argv"][row["argv"].index(matches[0]) + 1]
                else:
                    self.assertIn(flag, internal_pins)
                    expected = literals[internal_pins[flag]]
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), expected,
                                 "stale example pin gate_jobs." + name + " " + flag + " " + value)
        recipes = [*(("native_context_contract.gates." + name, row)
                     for name, row in plan["native_context_contract"]["gates"].items()),
                   *(("performance_profile_contract.profiles." + name, row)
                     for name, row in plan["performance_profile_contract"]["profiles"].items())]
        for pin, row in recipes:
            self.assertEqual(hashlib.sha256((KIT.parent / row["recipe_relpath"]).read_bytes()).hexdigest(),
                             row["recipe_sha256"],
                             "stale example pin " + pin + ".recipe_sha256 for " + row["recipe_relpath"])

    def test_matrix_accepts_candidate_but_rejects_mixed_release_arms(self):
        # Execute the actual pure validators without the sealer's Git preimport.
        source = ast.parse((KIT / "qwen4_native_concurrency_matrix.py").read_bytes())
        names = {"fail", "exact_object", "exact_int", "nonempty_string", "basename",
                 "canonical_sha256", "public_model_id", "validate_provenance", "validate_matrix"}
        namespace = {"probe": C, "Path": Path, "re": re}
        for node in source.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                            "PROVENANCE_KEYS", "COMMON_PROVENANCE_KEYS", "PROFILES", "WIDTHS"):
                        namespace[target.id] = ast.literal_eval(node.value)
        statements = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        statements += [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                     "actual-matrix-pure-validators", "exec"), namespace)
        p = {key: "1" * 64 for key in namespace["PROVENANCE_KEYS"]}
        p.update(manifest_schema=C.LAUNCH_SCHEMA, release_version=VERSION,
                 source_revision="1" * 40, source_tree="2" * 40,
                 checkpoint_id="hf://JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid",
                 checkpoint_revision="3" * 40, checkpoint_files_verified=1,
                 checkpoint_bytes_verified=1, selector_profile="native-reference",
                 configured_max_batch_size=8, configured_max_num_seqs=8,
                 server_process={"pid": 100, "boot_id": "12345678-1234-1234-1234-123456789abc",
                                 "start_ticks": 1, "log_device": 1, "log_inode": 1})
        namespace["validate_provenance"](p, "native-reference", 2, 100, "candidate")
        p["release_version"] = "v0.9.13-rc.r17.20260230"
        with self.assertRaisesRegex(C.ProbeError, "release version is invalid"):
            namespace["validate_provenance"](p, "native-reference", 2, 100, "invalid")
        p["release_version"] = VERSION
        arms = {}
        for index, key in enumerate((profile, width) for profile in namespace["PROFILES"] for width in namespace["WIDTHS"]):
            provenance = copy.deepcopy(p)
            provenance["server_process"].update(pid=100 + index, log_inode=1 + index)
            arms[key] = {"identity": key, "provenance": provenance, "model": "fixture",
                         "max_tokens": 16, "minimum_full_frames": 8, "memory_floor_mib": 8192,
                         "probe_name": "probe", "probe_sha256": "1" * 64}
        arms[("native-reference", 4)]["provenance"]["release_version"] = "v0.9.13-r15"
        namespace.update(reviewed_probe_source=lambda: types.SimpleNamespace(sha256="1" * 64),
                         validate_public=lambda *_args: None,
                         validate_arm=lambda arm, *_args, **_kwargs: arm["identity"])
        with self.assertRaisesRegex(C.ProbeError, "immutable R15 provenance differs"):
            namespace["validate_matrix"](arms, {})


if __name__ == "__main__":
    unittest.main()
