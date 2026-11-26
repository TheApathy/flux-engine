# SPDX-License-Identifier: AGPL-3.0-only
"""CPU checks for the files and OCI labels consumed by the release wrapper."""

import ast
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_image_provenance", KIT / "validate-image-provenance.py"
)
assert SPEC is not None and SPEC.loader is not None
PROVENANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVENANCE)

LABEL_ARGS = {
    "org.opencontainers.image.version": "FLUX_ENGINE_VERSION",
    "org.opencontainers.image.revision": "FLUX_SOURCE_REVISION",
    "com.theapathy.flux.source-tree": "FLUX_SOURCE_TREE",
    "com.theapathy.flux.binary-sha256": "FLUX_BINARY_SHA256",
    "com.theapathy.flux.recipe-sha256": "FLUX_RECIPE_SHA256",
}
PERFORMANCE_LABEL_ARGS = {
    "com.theapathy.flux.performance.interactive.recipe-sha256": "FLUX_INTERACTIVE_RECIPE_SHA256",
    "com.theapathy.flux.performance.high-prefill.recipe-sha256": "FLUX_HIGH_PREFILL_RECIPE_SHA256",
}
PERFORMANCE_PATHS = {
    "interactive": "/opt/flux/repro-kit/recipes/flux-v9-w4096-interactive.env",
    "high-prefill": "/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env",
}


class ImageProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "bin" / "flux-server-real"
        self.binary.parent.mkdir()
        self.binary.write_bytes(b"fixture executable")
        self.binary.chmod(0o755)
        self.recipe = self.root / "repro-kit" / "recipes" / "release.env"
        self.recipe.parent.mkdir(parents=True)
        self.recipe.write_bytes(b"FLUX_MODE=interactive\n")
        self.performance_files = {}
        performance_recipes = {}
        for profile, path in PERFORMANCE_PATHS.items():
            target = self.root / Path(path).relative_to("/opt/flux")
            target.write_bytes(f"PROFILE={profile}\n".encode())
            self.performance_files[profile] = target
            performance_recipes[profile] = {
                "path": path,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        self.arguments = {
            "version": "0.9.13-r15",
            "source_revision": "a" * 40,
            "source_tree": "b" * 40,
            "binary_name": self.binary.name,
            "binary_sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            "recipe_name": self.recipe.name,
            "recipe_sha256": hashlib.sha256(self.recipe.read_bytes()).hexdigest(),
            "performance_recipes": performance_recipes,
        }

    def validate(self, **changes):
        return PROVENANCE.validate_image_inputs(
            self.root, **{**self.arguments, **changes}
        )

    def test_real_copied_files_supply_the_exact_release_label_schema(self):
        # The fake primary executable is never run; no auxiliary build is needed.
        self.assertEqual(
            [path.name for path in self.binary.parent.iterdir()],
            ["flux-server-real"],
        )
        labels = self.validate()
        expected = set(LABEL_ARGS) | set(PERFORMANCE_LABEL_ARGS)
        expected.update(
            f"com.theapathy.flux.performance.{profile}.recipe-path"
            for profile in PERFORMANCE_PATHS
        )
        self.assertEqual(set(labels), expected)
        self.assertEqual(labels["org.opencontainers.image.version"], "0.9.13-r15")
        self.assertEqual(
            labels["com.theapathy.flux.binary-sha256"],
            self.arguments["binary_sha256"],
        )

    def test_unset_malformed_and_uppercase_provenance_is_rejected(self):
        for name, width in (
            ("source_revision", 40),
            ("source_tree", 40),
            ("binary_sha256", 64),
            ("recipe_sha256", 64),
        ):
            for value in ("", "a" * (width - 1), "A" * width, "g" * width):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        self.validate(**{name: value})

    def test_selected_file_digest_must_match_the_actual_image_bytes(self):
        for name in ("binary_sha256", "recipe_sha256"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    self.validate(**{name: "0" * 64})

    def test_copying_a_different_binary_or_recipe_invalidates_provenance(self):
        self.binary.write_bytes(b"different executable")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate()
        self.arguments["binary_sha256"] = hashlib.sha256(
            self.binary.read_bytes()
        ).hexdigest()
        self.recipe.write_bytes(b"FLUX_MODE=fast\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate()

    def test_nonexecutable_or_empty_engine_is_rejected(self):
        self.binary.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "executable"):
            self.validate()
        self.binary.chmod(0o755)
        self.binary.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "nonempty regular file"):
            self.validate(binary_sha256=hashlib.sha256(b"").hexdigest())

    def test_symlinked_engine_and_recipe_directory_are_rejected(self):
        real = self.binary.with_name("other-file")
        self.binary.rename(real)
        self.binary.symlink_to(real.name)
        with self.assertRaisesRegex(ValueError, "symlink component"):
            self.validate()
        self.binary.unlink()
        real.rename(self.binary)
        recipes = self.recipe.parent
        real_recipes = recipes.with_name("actual-recipes")
        recipes.rename(real_recipes)
        recipes.symlink_to(real_recipes.name, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink component"):
            self.validate()

    def test_unsafe_or_unbundled_names_and_invalid_versions_are_rejected(self):
        for binary in ("../bin/flux-server-real", "arbitrary-server", ""):
            with self.subTest(binary=binary), self.assertRaises(ValueError):
                self.validate(binary_name=binary)
        for recipe in ("../release.env", "/release.env", "release.sh", ""):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                self.validate(recipe_name=recipe)
        for version in ("", "v0.9.13-r15", "0.9.13-r15\n", "0.9.13 r15"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.validate(version=version)

    def test_profiles_bind_both_actual_files_and_fixed_paths(self):
        self.assertEqual(PROVENANCE.PERFORMANCE_RECIPE_PATHS, PERFORMANCE_PATHS)
        labels = self.validate()
        for profile, binding in self.arguments["performance_recipes"].items():
            prefix = f"com.theapathy.flux.performance.{profile}.recipe"
            self.assertEqual(labels[f"{prefix}-path"], binding["path"])
            self.assertEqual(labels[f"{prefix}-sha256"], binding["sha256"])
        self.performance_files["high-prefill"].write_bytes(b"different recipe\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate()

    def test_startup_recipe_can_be_a_bound_performance_recipe(self):
        profile = self.arguments["performance_recipes"]["high-prefill"]
        labels = self.validate(
            recipe_name=Path(profile["path"]).name,
            recipe_sha256=profile["sha256"],
        )
        self.assertEqual(labels["com.theapathy.flux.recipe-sha256"], profile["sha256"])

    def test_missing_extra_and_malformed_profile_maps_refuse(self):
        valid = self.arguments["performance_recipes"]
        cases = [
            None,
            [],
            {},
            {"interactive": valid["interactive"]},
            {**valid, "maximum": valid["high-prefill"]},
            {"interactive": [], "high-prefill": valid["high-prefill"]},
        ]
        for bindings in cases:
            with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                self.validate(performance_recipes=bindings)
        missing = {
            key: value
            for key, value in self.arguments.items()
            if key != "performance_recipes"
        }
        with self.assertRaises(TypeError):
            PROVENANCE.validate_image_inputs(self.root, **missing)

    def test_profile_fields_cannot_be_extra_missing_swapped_or_path_aliases(self):
        valid = self.arguments["performance_recipes"]
        for profile, binding in valid.items():
            cases = [
                {"path": binding["path"]},
                {**binding, "name": profile},
                {
                    **binding,
                    "path": valid[
                        "high-prefill" if profile == "interactive" else "interactive"
                    ]["path"],
                },
                {
                    **binding,
                    "path": binding["path"].replace(
                        "/recipes/", "/recipes/../recipes/"
                    ),
                },
                {**binding, "path": Path(binding["path"]).name},
                {**binding, "path": None},
            ]
            for changed in cases:
                with self.subTest(profile=profile, binding=changed):
                    with self.assertRaises(ValueError):
                        self.validate(performance_recipes={**valid, profile: changed})

    def test_malformed_profile_hashes_refuse(self):
        valid = self.arguments["performance_recipes"]
        for profile, binding in valid.items():
            for digest in (None, 1, "", "a" * 63, "A" * 64, "g" * 64):
                with self.subTest(profile=profile, digest=digest):
                    with self.assertRaises(ValueError):
                        self.validate(
                            performance_recipes={
                                **valid,
                                profile: {**binding, "sha256": digest},
                            }
                        )

    def test_cross_profile_hash_replay_refuses(self):
        valid = self.arguments["performance_recipes"]
        swapped = {
            "interactive": {
                **valid["interactive"],
                "sha256": valid["high-prefill"]["sha256"],
            },
            "high-prefill": {
                **valid["high-prefill"],
                "sha256": valid["interactive"]["sha256"],
            },
        }
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate(performance_recipes=swapped)

    def test_missing_empty_directory_and_fifo_profile_files_refuse(self):
        target = self.performance_files["interactive"]
        target.unlink()
        with self.assertRaises(OSError):
            self.validate()
        target.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "nonempty regular file"):
            self.validate()
        target.unlink()
        target.mkdir()
        with self.assertRaisesRegex(ValueError, "nonempty regular file"):
            self.validate()
        target.rmdir()
        os.mkfifo(target)
        with self.assertRaisesRegex(ValueError, "nonempty regular file"):
            self.validate()

    def test_profile_hardlinks_and_duplicate_bytes_do_not_fake_two_recipes(self):
        first = self.performance_files["interactive"]
        second = self.performance_files["high-prefill"]
        second.unlink()
        os.link(first, second)
        bindings = self.arguments["performance_recipes"]
        bindings["high-prefill"]["sha256"] = bindings["interactive"]["sha256"]
        with self.assertRaisesRegex(ValueError, "must not alias"):
            self.validate()
        second.unlink()
        second.write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError, "must not alias"):
            self.validate()

    def test_profile_leaf_and_parent_symlink_substitutions_refuse(self):
        original_open = PROVENANCE.os.open
        for component in (self.performance_files["interactive"].name, "recipes"):
            swapped = False

            def substitute(path, flags, **kwargs):
                nonlocal swapped
                if path == component and not swapped:
                    swapped = True
                    target = (
                        self.performance_files["interactive"]
                        if component != "recipes"
                        else self.recipe.parent
                    )
                    actual = target.with_name(target.name + ".actual")
                    target.rename(actual)
                    target.symlink_to(actual.name)
                return original_open(path, flags, **kwargs)

            with self.subTest(component=component):
                with mock.patch.object(PROVENANCE.os, "open", side_effect=substitute):
                    with self.assertRaises(OSError):
                        self.validate()
                target = (
                    self.performance_files["interactive"]
                    if component != "recipes"
                    else self.recipe.parent
                )
                target.unlink()
                target.with_name(target.name + ".actual").rename(target)

    def test_regular_file_replacement_after_hashing_refuses(self):
        target = self.performance_files["interactive"]
        original_open = PROVENANCE._open_without_symlinks
        calls = 0

        def replace_on_reopen(path):
            nonlocal calls
            if path == target:
                calls += 1
                if calls == 2:
                    replacement = target.with_name("replacement.env")
                    replacement.write_bytes(target.read_bytes())
                    replacement.replace(target)
            return original_open(path)

        with mock.patch.object(
            PROVENANCE, "_open_without_symlinks", side_effect=replace_on_reopen
        ):
            with self.assertRaisesRegex(ValueError, "path changed"):
                self.validate()

    def test_in_place_file_mutation_during_hashing_refuses(self):
        target = self.performance_files["interactive"]
        original_read = PROVENANCE.os.read
        mutated = False

        def mutate_after_read(descriptor, size):
            nonlocal mutated
            block = original_read(descriptor, size)
            if block and not mutated:
                mutated = True
                target.write_bytes(b"changed while hashing\n")
            return block

        with mock.patch.object(PROVENANCE.os, "read", side_effect=mutate_after_read):
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                self.validate()

    def test_real_canonical_source_recipe_bytes_validate_without_qualification(self):
        for profile, target in self.performance_files.items():
            target.write_bytes((KIT / "recipes" / target.name).read_bytes())
            self.arguments["performance_recipes"][profile]["sha256"] = hashlib.sha256(
                target.read_bytes()
            ).hexdigest()
        labels = self.validate()
        self.assertFalse(any("qualified" in key for key in labels))

    def test_cli_requires_both_profile_hashes_and_has_no_path_selector(self):
        arguments = []
        for name, value in self.arguments.items():
            if name != "performance_recipes":
                arguments.extend([f"--{name.replace('_', '-')}", value])
        command = [
            sys.executable,
            str(KIT / "validate-image-provenance.py"),
            "--root",
            str(self.root),
            *arguments,
        ]
        profiles = self.arguments["performance_recipes"]
        for profile in profiles:
            command.extend([f"--{profile}-recipe-sha256", profiles[profile]["sha256"]])
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for changed in (
            command[:-2],
            command[:-4] + command[-2:],
            command + ["--interactive-recipe-path", "/tmp/x"],
        ):
            result = subprocess.run(changed, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)


class DockerMetadataContractTests(unittest.TestCase):
    def test_docker_labels_use_the_validated_build_args(self):
        dockerfile = (KIT / "Dockerfile").read_text()
        for label, argument in {**LABEL_ARGS, **PERFORMANCE_LABEL_ARGS}.items():
            with self.subTest(label=label):
                self.assertIn(f'{label}="${{{argument}}}"', dockerfile)
                self.assertRegex(dockerfile, rf"(?m)^ARG {argument}(?:=|$)")
                self.assertIn(f'"${argument}"', dockerfile)
        self.assertIn(
            "RUN python3 /opt/flux/repro-kit/validate-image-provenance.py",
            dockerfile,
        )
        wrapper = KIT / "r15_exact_image_release.py"
        if wrapper.exists():
            contracts = []
            for node in ast.walk(ast.parse(wrapper.read_text())):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "expected_labels"
                        for target in node.targets
                    )
                    and isinstance(node.value, ast.Dict)
                ):
                    contracts.append(
                        {
                            key.value
                            for key in node.value.keys
                            if isinstance(key, ast.Constant)
                            and isinstance(key.value, str)
                        }
                    )
            expected = set(LABEL_ARGS) | set(PERFORMANCE_LABEL_ARGS)
            expected.update(
                f"com.theapathy.flux.performance.{profile}.recipe-path"
                for profile in PERFORMANCE_PATHS
            )
            self.assertEqual(contracts, [expected])

    def test_profile_labels_are_fixed_and_startup_defaults_remain_unchanged(self):
        dockerfile = (KIT / "Dockerfile").read_text()
        for profile, path in PERFORMANCE_PATHS.items():
            self.assertIn(
                f'com.theapathy.flux.performance.{profile}.recipe-path="{path}"',
                dockerfile,
            )
        for argument in PERFORMANCE_LABEL_ARGS.values():
            self.assertIn(f"ARG {argument}", dockerfile.splitlines())
            self.assertNotIn(f"ARG {argument}=", dockerfile)
        self.assertIn("ARG FLUX_RECIPE_NAME=flux-nextgen-max-prefill-decode.env", dockerfile)
        self.assertIn("ARG FLUX_BINARY_NAME=flux-server-real", dockerfile)
        self.assertNotIn("--interactive-recipe-name", dockerfile)

    def test_only_primary_engine_and_recipe_inputs_are_required(self):
        dockerfile = (KIT / "Dockerfile").read_text()
        copies = re.findall(r"(?m)^COPY (bin/\S+) (\S+)$", dockerfile)
        self.assertEqual(
            copies,
            [("bin/flux-server-real", "/opt/flux/bin/flux-server-real")],
        )
        self.assertIn("COPY recipes /opt/flux/repro-kit/recipes", dockerfile)
        self.assertIn('ENTRYPOINT ["/usr/local/bin/flux"]', dockerfile)
        self.assertIn('CMD ["serve"]', dockerfile)
        for argument in ("FLUX_BINARY_NAME", "FLUX_RECIPE_NAME"):
            match = re.search(rf"(?m)^ARG {argument}=(.+)$", dockerfile)
            self.assertIsNotNone(match)
            self.assertIn(f'"${argument}"', dockerfile)

    def test_license_and_source_match_the_engine_and_not_diagnostics(self):
        dockerfile = (KIT / "Dockerfile").read_text()
        self.assertIn('org.opencontainers.image.licenses="AGPL-3.0-only"', dockerfile)
        self.assertIn(
            'org.opencontainers.image.source="https://github.com/TheApathy/flux-engine"',
            dockerfile,
        )
        copy_lines = "\n".join(
            line for line in dockerfile.splitlines() if line.startswith("COPY ")
        )
        self.assertIn("SOURCE_AND_LICENSES.md", copy_lines)
        self.assertNotIn("SOURCE_AND_LICENSES.md", (KIT / ".dockerignore").read_text())


if __name__ == "__main__":
    unittest.main()
