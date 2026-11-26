# SPDX-License-Identifier: AGPL-3.0-only
"""Source/CPU mocks only. No Docker, model, runtime or quality qualification."""

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import profile_fixture
import test_r15_exact_image_release as fixture

Q = fixture.Q


class PerformanceImageBindingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path, self.source = fixture.make_fixture(temporary.name)
        self.loader = fixture.R15ExactImageTests()

    def write_manifest(self, document):
        self.path.write_bytes(Q.canonical_bytes(document))

    def test_legacy_manifest_schema_is_refused_even_in_dry_run(self):
        self.source["schema"] = "flux-r15-exact-image-launch/1"
        self.write_manifest(self.source)
        for verify in (False, True):
            with (
                self.subTest(verify=verify),
                self.assertRaisesRegex(ValueError, "schema"),
            ):
                self.loader.load(self.path, verify_files=verify)

    def test_map_is_required_closed_canonical_nonzero_and_nonaliased(self):
        valid = self.source["image"]["performance_recipes"]
        cases = [
            None,
            {},
            {"interactive": valid["interactive"]},
            {**valid, "max": valid["high-prefill"]},
        ]
        for profile, binding in valid.items():
            other = "high-prefill" if profile == "interactive" else "interactive"
            for changed in (
                {"path": binding["path"]},
                {**binding, "extra": True},
                {**binding, "path": valid[other]["path"]},
                {
                    **binding,
                    "path": binding["path"].replace(
                        "/recipes/", "/recipes/../recipes/"
                    ),
                },
                *(
                    {**binding, "sha256": value}
                    for value in (
                        None,
                        True,
                        "",
                        "0" * 64,
                        "A" * 64,
                        valid[other]["sha256"],
                    )
                ),
            ):
                cases.append({**valid, profile: changed})
        for index, bindings in enumerate(cases):
            candidate = copy.deepcopy(self.source)
            candidate["image"]["performance_recipes"] = bindings
            self.write_manifest(candidate)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.loader.load(self.path, verify_files=False)
        self.source["image"].pop("performance_recipes")
        self.write_manifest(self.source)
        with self.assertRaises(ValueError):
            self.loader.load(self.path, verify_files=False)

    def test_map_hash_cannot_be_repinned_away_from_proven_source(self):
        self.source["image"]["performance_recipes"]["interactive"]["sha256"] = "1" * 64
        self.write_manifest(self.source)
        with self.assertRaisesRegex(ValueError, "differs from proven source"):
            self.loader.load(self.path)

    def test_repinning_map_and_plan_cannot_bypass_source_inventory(self):
        digest = "1" * 64
        self.source["image"]["performance_recipes"]["interactive"]["sha256"] = digest
        path = Path(self.source["suite"]["plan_path"])
        plan = Q.strict_json_loads(path.read_bytes(), "fixture plan")
        plan["performance_profile_contract"]["profiles"]["interactive"][
            "recipe_sha256"
        ] = digest
        path.write_bytes(Q.canonical_bytes(plan))
        self.source["suite"]["plan_sha256"] = Q.sha256_file(path)
        self.write_manifest(self.source)
        with self.assertRaisesRegex(ValueError, "not in proven inventory"):
            self.loader.load(self.path)

    def test_every_profile_label_is_required_and_crossbound(self):
        manifest = self.loader.load(self.path)
        for profile in Q.PERFORMANCE_RECIPE_PATHS:
            for field in ("path", "sha256"):
                key = f"com.theapathy.flux.performance.{profile}.recipe-{field}"
                other = "high-prefill" if profile == "interactive" else "interactive"
                wrong = self.source["image"]["performance_recipes"][other][field]
                for remove in (True, False):

                    def mutate(document):
                        labels = document["Config"]["Labels"]
                        if remove:
                            labels.pop(key)
                        else:
                            labels[key] = wrong

                    with self.subTest(key=key, remove=remove):
                        with self.assertRaisesRegex(ValueError, "OCI labels differ"):
                            Q.inspect_exact_image(
                                manifest, fixture.DockerBackend(self.source, mutate)
                            )

    def test_unknown_performance_prefixed_labels_refuse(self):
        manifest = self.loader.load(self.path)
        for suffix in ("turbo.recipe-sha256", "interactive.recipe-alias"):

            def mutate(document):
                document["Config"]["Labels"][
                    "com.theapathy.flux.performance." + suffix
                ] = "x"

            with (
                self.subTest(suffix=suffix),
                self.assertRaisesRegex(ValueError, "unexpected performance"),
            ):
                Q.inspect_exact_image(
                    manifest, fixture.DockerBackend(self.source, mutate)
                )

    def test_correct_labels_do_not_substitute_for_actual_recipe_bytes(self):
        manifest = self.loader.load(self.path)
        for profile in Q.PERFORMANCE_RECIPE_PATHS:
            backend = fixture.DockerBackend(self.source)
            other = "high-prefill" if profile == "interactive" else "interactive"
            backend.performance_recipe_bytes[profile] = (
                backend.performance_recipe_bytes[other]
            )
            Q.inspect_exact_image(manifest, backend)
            with (
                self.subTest(profile=profile),
                self.assertRaisesRegex(
                    ValueError, "performance recipe SHA-256 mismatch"
                ),
            ):
                Q.measure_container_artifacts(manifest, fixture.CONTAINER_ID, backend)
            self.assertEqual(backend.calls[-1][:2], ("docker", "rm"))

    def test_missing_profile_artifact_fails_and_removes_standoff(self):
        manifest = self.loader.load(self.path)
        target = Q.PERFORMANCE_RECIPE_PATHS["interactive"]

        class MissingBackend(fixture.DockerBackend):
            def run(self, argv, *, timeout_s):
                if tuple(argv[:2]) == ("docker", "cp") and argv[-2].endswith(
                    ":" + target
                ):
                    self.calls.append(tuple(argv))
                    return Q.r12.CommandResult(1, "", "missing recipe")
                return super().run(argv, timeout_s=timeout_s)

        backend = MissingBackend(self.source)
        with self.assertRaisesRegex(
            RuntimeError, "copy exact in-container interactive"
        ):
            Q.measure_container_artifacts(manifest, fixture.CONTAINER_ID, backend)
        self.assertEqual(backend.calls[-1][:2], ("docker", "rm"))

    def test_extracted_recipe_symlinks_and_hardlinks_refuse(self):
        manifest = self.loader.load(self.path)
        target = Q.PERFORMANCE_RECIPE_PATHS["interactive"]
        for kind in ("symlink", "hardlink"):

            class AliasBackend(fixture.DockerBackend):
                def run(self, argv, *, timeout_s):
                    result = super().run(argv, timeout_s=timeout_s)
                    if tuple(argv[:2]) == ("docker", "cp") and argv[-2].endswith(
                        ":" + target
                    ):
                        destination = Path(argv[-1])
                        other = destination.with_name("aliased-recipe")
                        destination.rename(other)
                        if kind == "symlink":
                            destination.symlink_to(other.name)
                        else:
                            os.link(other, destination)
                    return result

            backend = AliasBackend(self.source)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                Q.measure_container_artifacts(manifest, fixture.CONTAINER_ID, backend)
            self.assertEqual(backend.calls[-1][:2], ("docker", "rm"))

    def test_extracted_recipe_fifos_refuse_with_a_child_process_deadline(self):
        script = r"""
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, sys.argv[1])
import test_r15_exact_image_release as fixture
q = fixture.Q
target = q.PERFORMANCE_RECIPE_PATHS[sys.argv[2]]
class FifoBackend(fixture.DockerBackend):
    def run(self, argv, *, timeout_s):
        result = super().run(argv, timeout_s=timeout_s)
        if tuple(argv[:2]) == ("docker", "cp") and argv[-2].endswith(":" + target):
            destination = Path(argv[-1])
            destination.unlink()
            os.mkfifo(destination)
        return result
with tempfile.TemporaryDirectory() as temp:
    path, source = fixture.make_fixture(temp)
    manifest = fixture.R15ExactImageTests().load(path)
    backend = FifoBackend(source)
    try:
        q.measure_container_artifacts(manifest, fixture.CONTAINER_ID, backend)
    except ValueError as error:
        assert "bounded regular file" in str(error), str(error)
    else:
        raise AssertionError("extracted FIFO was accepted")
    assert backend.calls[-1][:2] == ("docker", "rm")
"""
        for profile in Q.PERFORMANCE_RECIPE_PATHS:
            with self.subTest(profile=profile):
                completed = subprocess.run(
                    [sys.executable, "-c", script, str(Path(__file__).parent), profile],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_each_recipe_drift_before_or_after_suite_prevents_seal(self):
        for profile in Q.PERFORMANCE_RECIPE_PATHS:
            for phase in ("before", "after"):
                with (
                    self.subTest(profile=profile, phase=phase),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    path, source = fixture.make_fixture(temp)
                    manifest = self.loader.load(path)
                    backend = fixture.DockerBackend(source)

                    def mutate():
                        backend.performance_recipe_bytes[profile] = b"changed recipe\n"

                    if phase == "before":
                        mutate()
                    suite = fixture.SuiteBackend(
                        fixture.suite_receipt(source),
                        hook=mutate if phase == "after" else None,
                    )
                    outer = Path(temp) / "outer.json"
                    with (
                        mock.patch.object(Q.provenance_gate, "configure_release"),
                        mock.patch.object(
                            Q.provenance_gate, "validate_qualification_provenance"
                        ),
                        mock.patch.object(Q, "_profile_runtime_recipe_binding"),
                        self.assertRaisesRegex(
                            ValueError, "performance recipe SHA-256 mismatch"
                        ),
                    ):
                        Q.run_live(
                            manifest,
                            path,
                            outer,
                            docker_backend=backend,
                            suite_backend=suite,
                            runner_factory=lambda **kw: fixture.Runner(kw),
                        )
                    self.assertFalse(outer.exists())
                    self.assertEqual(bool(suite.calls), phase == "after")
                    self.assertEqual(backend.calls[-1][:2], ("docker", "rm"))

    def test_source_recipe_drift_is_reopened_and_refused(self):
        manifest = self.loader.load(self.path)
        path = (
            Path(manifest["provenance"]["source_root_path"])
            / "repro-kit/recipes/flux-v9-w4096-interactive.env"
        )
        path.write_bytes(path.read_bytes() + b"# modified\n")
        with (
            mock.patch.object(Q.provenance_gate, "configure_release"),
            mock.patch.object(Q.provenance_gate, "validate_qualification_provenance"),
            self.assertRaisesRegex(ValueError, "differs from proven inventory"),
        ):
            Q.revalidate_manifest(manifest, self.path, "hostile source recipe drift")

    def test_new_image_provenance_does_not_relax_single_recipe_broker_hold(self):
        manifest = self.loader.load(self.path)
        plan = Q.strict_json_loads(
            Path(manifest["suite"]["plan_path"]).read_bytes(), "fixture plan"
        )
        # The suite plan's contract is /2 (911ec7cef): each profile binds its
        # own image recipe, so the plan itself passes. A /1 contract still
        # meets the single-recipe broker HOLD whatever the image provenance.
        Q._profile_runtime_recipe_binding(
            plan["performance_profile_contract"], manifest["image"]
        )
        with self.assertRaisesRegex(ValueError, "HOLD.*both profile recipes"):
            Q._profile_runtime_recipe_binding(profile_fixture.contract(Q), manifest["image"])


if __name__ == "__main__":
    unittest.main()
