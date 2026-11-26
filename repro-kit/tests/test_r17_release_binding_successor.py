# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic release identity and recipe binding checks. No workload execution."""

import ast
import copy
import fcntl
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


KIT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path, loader=SourceFileLoader(name, str(path)))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


P = module("r9_source_to_binary", KIT / "r9_source_to_binary.py")
Q = module("r15_exact_image_release", KIT / "r15_exact_image_release.py")
S = module("r17_binding_suite", KIT / "r15-container-suite.py")
VERSION = "v0.9.13-rc.r17.20261024"
OCI_VERSION = "0.9.13-rc.r17.20261024"
MANIFEST = KIT / "examples/r15-exact-image-launch.example.json"
PLAN = KIT / "examples/r15-container-suite-plan.example.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe_values(raw):
    values = {}
    for line in raw.decode("ascii").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            if key in values:
                raise ValueError("duplicate recipe assignment")
            values[key] = value
    return values


class ReachedUnchangedValidation(Exception):
    pass


class ReleaseBindingTests(unittest.TestCase):
    def setUp(self):
        for target in ("subprocess.run", "subprocess.Popen", "os.fork",
                       "urllib.request.OpenerDirector.open", "socket.create_connection"):
            patch = mock.patch(target, side_effect=AssertionError("workload boundary denied"))
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(P.configure_release, P.DEFAULT_RELEASE_VERSION)

    def test_candidate_and_numbered_release_identities(self):
        for version, label, lane in ((VERSION, OCI_VERSION, "r17"),
                                     ("v0.9.15-r17", "0.9.15-r17", "r17"),
                                     ("v0.9.14-r17", "0.9.14-r17", "r17"),
                                     ("v0.9.13-r15", "0.9.13-r15", "r15")):
            expected = {"release_version": version, "oci_version": label, "source_lane": lane}
            self.assertEqual(P.release_identity(version), expected)
            self.assertEqual(S.release_identity(version), expected)
            P.configure_release(version)
            self.assertEqual(P.RELEASE_VERSION, version)
            self.assertEqual(P.SOURCE_MANIFEST_SCHEMA, "flux-" + lane + "-source-bundle-manifest/2")
            self.assertEqual(P.SOURCE_BUNDLE_NAME, "flux-engine-source-" + version + ".tar.gz")

    def test_invalid_versions_fail_without_reconfiguring_provenance(self):
        class Version(str):
            pass
        P.configure_release(VERSION)
        for invalid in (None, True, 17, Version(VERSION), "", OCI_VERSION,
                        VERSION + "\n", " " + VERSION, "v0.9.13-r0",
                        "v0.9.13-rc.r0.20261024", "v0.9.13-rc.r17.20260230",
                        "v0.9.13-rc.r17.00001024", "v0.9.13-rc.r17.20261024/extra",
                        "v0.9.13-rc.r17.20261024+arbitrary", "v" + "9" * 100 + ".1.2-r1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    P.release_identity(invalid)
                with self.assertRaises(ValueError):
                    S.release_identity(invalid)
                with self.assertRaises(P.ProvenanceError):
                    P.configure_release(invalid)
                self.assertEqual(P.RELEASE_VERSION, VERSION)

    def test_standalone_suite_uses_the_same_identity_parser(self):
        def definition(path, name):
            tree = ast.parse(path.read_bytes())
            return ast.dump(next(node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name == name),
                            include_attributes=False)
        self.assertEqual(definition(KIT / "r9_source_to_binary.py", "release_identity"),
                         definition(KIT / "r15-container-suite.py", "release_identity"))

    def test_candidate_example_is_nonqualified_and_keeps_all_required_gates(self):
        manifest = Q.load_manifest(MANIFEST, verify_files=False)
        plan = Q.dry_run_plan(manifest)
        self.assertEqual(manifest["release_version"], VERSION)
        self.assertEqual(manifest["provenance"]["release_version"], VERSION)
        self.assertEqual(plan["release_version"], VERSION)
        self.assertFalse(plan["release_qualified"])
        self.assertEqual(Q.REQUIRED_GATES, S.REQUIRED_GATES)
        self.assertEqual(len(Q.REQUIRED_GATES), 12)
        self.assertEqual(set(json.loads(PLAN.read_bytes())["gate_jobs"]), set(Q.REQUIRED_GATES))

    def test_manifest_rejects_valid_but_different_provenance_version(self):
        original = json.loads(MANIFEST.read_bytes())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            for changed in ("v0.9.13-r15", "v0.9.13-rc.r17.20261025"):
                value = copy.deepcopy(original)
                value["provenance"]["release_version"] = changed
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "provenance.release_version differs"):
                    Q.load_manifest(path, verify_files=False)

    def test_sealed_plan_accepts_declared_candidate_and_rejects_invalid_version(self):
        value = json.loads(PLAN.read_bytes())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source").mkdir()
            (root / "runtime").mkdir()
            before = value["source_root_path"]
            value = json.loads(json.dumps(value).replace(before, str(root / "source"))
                               .replace("0" * 64, "1" * 64).replace("0" * 40, "1" * 40))
            value["runtime_dir"] = str(root / "runtime")
            for version in (VERSION, "v0.9.13-rc.r17.20260230"):
                value["release_version"] = version
                fd = os.memfd_create("synthetic-release-plan", os.MFD_ALLOW_SEALING)
                try:
                    os.write(fd, S.canonical_bytes(value))
                    fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                                fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW |
                                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
                    if version == VERSION:
                        self.assertEqual(S.load_plan(fd)["release_version"], VERSION)
                    else:
                        with self.assertRaises(ValueError):
                            S.load_plan(fd)
                finally:
                    os.close(fd)

    def test_sealed_plan_version_must_match_provenance(self):
        manifest = Q.load_manifest(MANIFEST, verify_files=False)
        plan = json.loads(PLAN.read_bytes())
        for version in ("v0.9.13-r15", "v0.9.13-rc.r17.20261025"):
            plan["release_version"] = version
            with self.assertRaisesRegex(ValueError, "suite plan schema or release version differs"):
                Q._validate_program_closure(manifest["provenance"], manifest["suite"], plan, Path("/synthetic"))
        plan["release_version"] = VERSION
        with mock.patch.object(Q, "_normalized_absolute_path", side_effect=ReachedUnchangedValidation):
            with self.assertRaises(ReachedUnchangedValidation):
                Q._validate_program_closure(manifest["provenance"], manifest["suite"], plan, Path("/synthetic"))

    def test_image_label_must_equal_declared_candidate_version(self):
        manifest = Q.load_manifest(MANIFEST, verify_files=False)
        layers = ["sha256:" + "1" * 64]
        manifest["image"]["rootfs_layers_sha256"] = Q.sha256_bytes(Q.canonical_bytes(layers))
        labels = {"org.opencontainers.image.version": OCI_VERSION,
                  "org.opencontainers.image.revision": manifest["provenance"]["source_revision"],
                  "com.theapathy.flux.source-tree": manifest["provenance"]["source_tree"],
                  "com.theapathy.flux.binary-sha256": manifest["image"]["in_container_binary_sha256"],
                  "com.theapathy.flux.recipe-sha256": manifest["image"]["recipe_sha256"]}
        for name, binding in manifest["image"]["performance_recipes"].items():
            prefix = "com.theapathy.flux.performance." + name + ".recipe"
            labels[prefix + "-path"] = binding["path"]
            labels[prefix + "-sha256"] = binding["sha256"]
        document = {"Id": manifest["image"]["id"], "RepoDigests": [manifest["image"]["digest"]],
                    "RootFS": {"Layers": layers}, "Config": {"Labels": labels}}
        backend = types.SimpleNamespace(run=lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout=json.dumps([document])))
        self.assertEqual(Q.inspect_exact_image(manifest, backend)["id"], manifest["image"]["id"])
        for invalid in (None, "0.9.13-r15", "0.9.13-rc.r17.20261025", VERSION):
            labels["org.opencontainers.image.version"] = invalid
            with self.assertRaisesRegex(ValueError, "immutable image OCI labels differ"):
                Q.inspect_exact_image(manifest, backend)
        del labels["org.opencontainers.image.version"]
        with self.assertRaisesRegex(ValueError, "immutable image OCI labels differ"):
            Q.inspect_exact_image(manifest, backend)

    def test_suite_receipt_cannot_replay_another_release_version(self):
        manifest = Q.load_manifest(MANIFEST, verify_files=False)
        receipt = {"schema": Q.SUITE_SCHEMA, "status": "PASS", "release_qualified": True,
            "release_version": VERSION, "execution_scope": "immutable-container",
            "image_id": manifest["image"]["id"], "image_digest": manifest["image"]["digest"],
            "container_id": "a" * 64, "run_id": manifest["launch"]["run_id"],
            "source_revision": manifest["provenance"]["source_revision"],
            "source_tree": manifest["provenance"]["source_tree"],
            "source_bundle_manifest_sha256": manifest["provenance"]["source_bundle_manifest_sha256"],
            "in_container_binary_sha256": manifest["image"]["in_container_binary_sha256"],
            "recipe_sha256": manifest["image"]["recipe_sha256"],
            "checkpoint_id": manifest["provenance"]["checkpoint_id"],
            "checkpoint_revision": manifest["provenance"]["checkpoint_revision"],
            "checkpoint_manifest_sha256": manifest["provenance"]["checkpoint_manifest_sha256"],
            "suite_plan_sha256": manifest["suite"]["plan_sha256"],
            "runtime_dir_identity_sha256": "b" * 64,
            "gates": {}, "process_bindings": [], "raw_log_safety": {}}
        with mock.patch.object(Q, "_directory_identity_sha256", return_value="b" * 64), \
                mock.patch.object(Q, "_validate_process_bindings", side_effect=ReachedUnchangedValidation):
            for version in ("v0.9.13-r15", "v0.9.13-rc.r17.20261025"):
                receipt["release_version"] = version
                with self.assertRaisesRegex(ValueError, "suite receipt release_version differs"):
                    Q._validate_suite_receipt_bytes(Q.canonical_bytes(receipt), manifest, "a" * 64)
            receipt["release_version"] = VERSION
            with self.assertRaises(ReachedUnchangedValidation):
                Q._validate_suite_receipt_bytes(Q.canonical_bytes(receipt), manifest, "a" * 64)

    def test_recipe_and_program_hash_closure_matches_successor_bytes(self):
        recipe = KIT / "recipes/flux-qwen4-native-262k-w4096-bf16-owner16.env"
        gate = KIT / "r15-gates/native-262k-w4096-owner16"
        authority = Q._native_context_required_module().required_authority("native-262k-w4096-owner16")
        self.assertEqual(authority["recipe"][1], digest(recipe))
        self.assertEqual(authority["program"][1], digest(gate))
        self.assertEqual(Q.NATIVE_CONTEXT_RECIPES["native-262k-w4096-owner16"]["recipe_sha256"], digest(recipe))
        self.assertEqual(S.NATIVE_CONTEXT_RECIPES["native-262k-w4096-owner16"]["recipe_sha256"], digest(recipe))
        plan = json.loads(PLAN.read_bytes())
        job = plan["gate_jobs"]["native-262k-w4096-owner16"]
        self.assertEqual(job["program_sha256"], digest(gate))
        self.assertEqual(plan["native_context_contract"]["gates"]["native-262k-w4096-owner16"]["recipe_sha256"], digest(recipe))
        self.assertEqual(json.loads(MANIFEST.read_bytes())["suite"]["program_sha256"], digest(KIT / "r15-container-suite.py"))
        self.assertEqual(json.loads(MANIFEST.read_bytes())["suite"]["plan_sha256"], digest(PLAN))

    def test_owner16_recipes_enable_spans_and_preserve_accept_logging(self):
        for name in ("flux-qwen4-native-262k-w4096-bf16-owner16.env",
                     "flux-qwen4-native-concurrency-c8-owner16.env"):
            values = recipe_values((KIT / "recipes" / name).read_bytes())
            directives = values["RUST_LOG"].split(",")
            self.assertEqual(directives.count("atlas_bounded_transpose_allocation=debug"), 1)
            self.assertIn("spark::scheduler::spec_accept_log=debug", directives)
            self.assertEqual(values["ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB"], "16")
            if name.startswith("flux-qwen4-native-262k-"):
                self.assertEqual(values["FLUX_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB"], "16")

    def test_missing_span_logging_no_longer_satisfies_recipe_binding(self):
        recipe = KIT / "recipes/flux-qwen4-native-262k-w4096-bf16-owner16.env"
        raw = recipe.read_bytes()
        altered = raw.replace(b",atlas_bounded_transpose_allocation=debug", b"")
        self.assertNotEqual(altered, raw)
        authority = Q._native_context_required_module().required_authority("native-262k-w4096-owner16")
        self.assertNotEqual(hashlib.sha256(altered).hexdigest(), authority["recipe"][1])
        gate = module("r17_binding_native262_gate", KIT / "r15-gates/native-262k-w4096-owner16")
        self.assertEqual(gate.RECIPE_SHA256, authority["recipe"][1])
        suite = types.SimpleNamespace(validate_inherited_receipt_fd=mock.Mock(),
            load_native_context_recipe=mock.Mock(return_value=(
                {"recipe_sha256": hashlib.sha256(altered).hexdigest()}, recipe_values(altered))),
            normalized_absolute=mock.Mock(side_effect=AssertionError("advanced beyond recipe refusal")))
        lifecycle = types.SimpleNamespace(inspect_container=mock.Mock(return_value=({},)))
        args = types.SimpleNamespace(suite=suite, lifecycle=lifecycle, endpoint=object(),
                                     receipt_fd=5, raw_log_fd=6)
        with self.assertRaisesRegex(ValueError, "successor recipe hash differs"):
            gate.run(args)
        suite.normalized_absolute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
