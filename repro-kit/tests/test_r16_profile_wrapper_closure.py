# SPDX-License-Identifier: AGPL-3.0-only
"""Independent profile arithmetic and provenance tests using synthetic CPU data only
(the /1 fixture upgraded to contract /2, as the release suite plan /4 requires)."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import profile_fixture
import profile_fixture_v2
import test_r15_exact_image_release as fixture

Q = fixture.Q


class ProfileWrapperClosureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path, _ = fixture.make_fixture(temp.name)
        self.manifest = fixture.R15ExactImageTests().load(path)
        (Path(self.manifest["runtime_dir"]) / "validators").mkdir()
        self.raw, self.evidence = profile_fixture.profile_raw(
            Q, self.manifest, fixture.CONTAINER_ID
        )
        # The suite plan is /4 since 911ec7cef, so the validator pairs it with
        # profile contract /2; upgrade the /1 synthetic data the same way the
        # exact-image fixture does.
        _plan, self.raw, self.evidence = profile_fixture_v2.upgrade(
            Q, self.manifest, self.raw, self.evidence
        )

    def validate(self, *, runtime_policy_mock=True):
        target = Path(self.manifest["runtime_dir"]) / "validators/profiles.json"
        document = {
            "schema": Q.GATE_VALIDATOR_SCHEMA,
            "gate": "prefill-decode-profiles",
            "status": "PASS",
            "release_qualified": True,
            "container_id": fixture.CONTAINER_ID,
            "run_id": self.manifest["launch"]["run_id"],
            "raw": self.raw,
        }
        target.write_bytes(Q.canonical_bytes(document))
        self.evidence["validator_artifact"] = "validators/profiles.json"
        self.evidence["validator_receipt_sha256"] = Q.sha256_file(target)
        if runtime_policy_mock:
            with mock.patch.object(Q, "_profile_runtime_recipe_binding"):
                return Q._validate_profile_evidence(
                    self.evidence, self.manifest, fixture.CONTAINER_ID
                )
        return Q._validate_profile_evidence(
            self.evidence, self.manifest, fixture.CONTAINER_ID
        )

    def repin_host(self):
        digest = Q.sha256_bytes(Q.canonical_bytes(self.raw["host_reference"]))
        self.raw["host_reference_sha256"] = digest
        path = Path(self.manifest["suite"]["plan_path"])
        plan = Q.strict_json_loads(path.read_bytes(), "synthetic plan")
        plan["gate_jobs"]["prefill-decode-profiles"]["argv"][-1] = digest
        path.write_bytes(Q.canonical_bytes(plan))
        self.manifest["suite"]["plan_sha256"] = Q.sha256_file(path)

    def change_journal(self, change):
        path = (
            Path(self.manifest["runtime_dir"])
            / self.raw["response_journal"]["artifact"]
        )
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        change(rows)
        path.write_bytes(b"".join(Q.canonical_bytes(row) + b"\n" for row in rows))
        self.raw["response_journal"]["sha256"] = Q.sha256_file(path)

    def test_synthetic_arithmetic_fixture_is_recomputed_end_to_end(self):
        self.assertEqual(self.validate(), self.evidence)
        boundary = self.raw["rows"][5]
        self.assertEqual(
            (boundary["context_tokens"], boundary["k5_decode_tokens"]), (262080, 63)
        )

    def test_real_dual_recipe_policy_binds_each_profile_to_its_image_recipe(self):
        # Contract /2 (911ec7cef) serves each profile from its own image recipe,
        # so the /1 single-recipe HOLD no longer applies; the binding is exact.
        self.assertEqual(self.validate(runtime_policy_mock=False), self.evidence)
        image = self.manifest["image"]["performance_recipes"]
        for name in image:
            with self.subTest(profile=name):
                saved = copy.deepcopy(image[name])
                image[name]["sha256"] = "1" * 64
                with self.assertRaisesRegex(ValueError, "image does not carry this profile's recipe"):
                    self.validate(runtime_policy_mock=False)
                image[name] = saved

    def test_healthy_n_text_events_with_identical_arrival_stamps_do_not_set_decode_clock(
        self,
    ):
        def change(rows):
            for row in rows:
                if row["kind"] != "stream":
                    continue
                first = json.loads(row["events"][0][1])
                text = first["choices"][0]["text"]
                count = row["request"]["max_tokens"]
                pieces = [text[index] for index in range(count - 1)] + [
                    text[count - 1 :]
                ]
                terminal = row["events"][-1]
                row["events"] = []
                for piece in pieces:
                    value = copy.deepcopy(first)
                    value["choices"][0]["text"] = piece
                    row["events"].append([0.25, Q.canonical_bytes(value).decode()])
                row["events"].append(terminal)

        self.change_journal(change)
        for observation in self.raw["observations"]:
            budget = min(256, 262144 - observation["context_tokens"])
            for key in ("plain", "native_k5"):
                observation[key]["stream"]["stream_text_events"] = budget
        self.assertEqual(self.validate(), self.evidence)

    def test_missing_legacy_and_wrong_clock_contracts_fail(self):
        initial = copy.deepcopy(self.raw)
        changes = [
            lambda raw: raw.pop("measurement_schema"),
            lambda raw: raw.update(measurement_schema="flux-profile-metrics/2"),
            lambda raw: raw["measurement_contract"].update(decode_clock="sse-arrival"),
            lambda raw: raw["measurement_contract"].update(prefill_clock="engine-ttft"),
            lambda raw: raw["measurement_contract"].update(
                completions_source_sha256="a" * 64
            ),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.raw = copy.deepcopy(initial)
                change(self.raw)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_host_mutation_and_rehash_cannot_escape_pinned_plan(self):
        self.raw["host_reference"]["binary_sha256"] = "a" * 64
        self.raw["host_reference_sha256"] = Q.sha256_bytes(
            Q.canonical_bytes(self.raw["host_reference"])
        )
        with self.assertRaises(ValueError):
            self.validate()

    def test_host_provenance_is_bound_even_when_reference_is_repinned(self):
        initial = copy.deepcopy(self.raw)
        for key in Q.PROFILE_PROVENANCE_KEYS:
            with self.subTest(key=key):
                self.raw = copy.deepcopy(initial)
                self.raw["host_reference"]["provenance"][key] = "unrelated"
                self.repin_host()
                with self.assertRaisesRegex(ValueError, "source/checkpoint provenance"):
                    self.validate()

    def test_host_legacy_and_redundant_clock_operands_fail(self):
        initial = copy.deepcopy(self.raw)
        for field, value in (
            ("schema", "flux-prefill-decode-host-reference/2"),
            ("decode_budget_tokens", 64),
            ("chunk_tokens", True),
        ):
            with self.subTest(field=field):
                self.raw = copy.deepcopy(initial)
                self.raw["host_reference"][field] = value
                self.repin_host()
                with self.assertRaises(ValueError):
                    self.validate()
        self.raw = copy.deepcopy(initial)
        self.raw["host_reference"]["profiles"]["interactive"]["rows"][0][
            "k5_decode_wall_seconds"
        ] += 1.0
        self.repin_host()
        with self.assertRaisesRegex(ValueError, "source-derived decode wall"):
            self.validate()

    def test_host_plain_or_prefill_regression_fails_with_matching_k5(self):
        initial = copy.deepcopy(self.raw)
        for kind in ("plain", "prefill"):
            with self.subTest(kind=kind):
                self.raw = copy.deepcopy(initial)
                row = self.raw["host_reference"]["profiles"]["interactive"]["rows"][0]
                if kind == "plain":
                    row["plain_usage"]["response_token/s"] = 80.0
                    row["plain_decode_wall_seconds"] = 255 / 80.0
                else:
                    row["prefill_client_monotonic"]["finish"] = 101.0
                    row["prefill_wall_seconds"] = 1.0
                self.repin_host()
                with self.assertRaisesRegex(ValueError, "tolerance exceeded"):
                    self.validate()

    def test_context_and_decode_budget_metadata_cannot_be_fabricated(self):
        initial = copy.deepcopy(self.raw)
        for index, key, value in (
            (0, "prefill_prompt_tokens", 10200),
            (0, "plain_decode_tokens", 17),
            (5, "k5_decode_tokens", 255),
            (0, "profile_decode_wall_seconds", 100.0),
        ):
            with self.subTest(index=index, key=key):
                self.raw = copy.deepcopy(initial)
                self.raw["rows"][index][key] = value
                with self.assertRaises(ValueError):
                    self.validate()

    def test_observation_and_host_band_metadata_cannot_be_rewritten(self):
        initial = copy.deepcopy(self.raw)
        for branch, field in (
            ("observations", "prompt_sha256"),
            ("host_bands", "prefill_delta_percent"),
        ):
            with self.subTest(branch=branch):
                self.raw = copy.deepcopy(initial)
                self.raw[branch][0][field] = (
                    "a" * 64 if branch == "observations" else 0.1
                )
                with self.assertRaises(ValueError):
                    self.validate()

    def test_journal_artifact_mutation_is_detected_without_rehash(self):
        path = (
            Path(self.manifest["runtime_dir"])
            / self.raw["response_journal"]["artifact"]
        )
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "journal hash differs"):
            self.validate()

    def test_journal_missing_duplicate_and_swapped_roles_fail_after_rehash(self):
        initial = copy.deepcopy(self.raw)
        path = (
            Path(self.manifest["runtime_dir"])
            / self.raw["response_journal"]["artifact"]
        )
        original = path.read_bytes()
        for change in (
            lambda rows: rows.pop(),
            lambda rows: rows.__setitem__(1, rows[0]),
            lambda rows: rows[0].update(profile="high-prefill"),
            lambda rows: rows[0]["request"].update(ignore_eos=False),
        ):
            with self.subTest(change=change):
                self.raw = copy.deepcopy(initial)
                path.write_bytes(original)
                self.change_journal(change)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_terminal_usage_census_and_nonpositive_rate_fail(self):
        initial = copy.deepcopy(self.raw)
        path = (
            Path(self.manifest["runtime_dir"])
            / self.raw["response_journal"]["artifact"]
        )
        original = path.read_bytes()
        for key, value in (
            ("completion_tokens", 64),
            ("response_token/s", 0),
            ("response_token/s", True),
            ("prompt_tokens", 999),
            ("total_tokens", 8448.0),
        ):
            with self.subTest(key=key, value=value):
                self.raw = copy.deepcopy(initial)
                path.write_bytes(original)

                def change(rows):
                    event = json.loads(rows[2]["events"][-1][1])
                    event["usage"][key] = value
                    rows[2]["events"][-1][1] = Q.canonical_bytes(event).decode()

                self.change_journal(change)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_direct_rate_validator_rejects_nan_inf_bool(self):
        for value in (float("nan"), float("inf"), True, 0.0, -1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Q._profile_usage(
                    profile_fixture.usage(8192, 256, value), 8192, 256, decode=True
                )

    def test_plain_k5_token_identity_and_first_token_mismatch_fail(self):
        self.change_journal(
            lambda rows: rows[18]["response"]["choices"][0]["token_ids"].__setitem__(
                0, 43
            )
        )
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.validate()

    def test_process_role_argv_proof_and_verifier_metrics_tamper_fail(self):
        initial = copy.deepcopy(self.raw)
        for change in (
            lambda raw: raw["process_bindings"][1].update(mode="none"),
            lambda raw: raw["process_bindings"][0]["launch_argv"].__setitem__(
                16, "nvfp4"
            ),
            lambda raw: raw["profile_identity"]["profiles"]["interactive"][
                "process_proof_sha256"
            ].update(none="a" * 64),
            lambda raw: raw["verification_metrics"][0].update(k5_events=7),
            lambda raw: raw["verification_metrics"][0].update(metrics="modified"),
        ):
            with self.subTest(change=change):
                self.raw = copy.deepcopy(initial)
                change(self.raw)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_effective_default_reordering_and_noop_deadline_are_not_profiles(self):
        policy = profile_fixture.contract(Q)
        first = policy["profiles"]["interactive"]["argv_overlay"]
        for overlay in (
            first[:],
            ["--max-seq-len", "262144", "--max-prefill-tokens", "4096"],
            first + ["--tbt-deadline-ms", "9"],
        ):
            with self.subTest(overlay=overlay):
                policy["profiles"]["high-prefill"]["argv_overlay"] = overlay
                with self.assertRaisesRegex(ValueError, "share one effective argv"):
                    Q._profile_contract(policy)

    def test_missing_policy_wrong_recipe_and_rewritten_inventory_fail(self):
        path = Path(self.manifest["suite"]["plan_path"])
        original = path.read_bytes()
        for change in (
            lambda plan: plan.pop("performance_profile_contract"),
            lambda plan: plan["performance_profile_contract"]["profiles"][
                "interactive"
            ].update(recipe_relpath="other.env"),
            lambda plan: plan["performance_profile_contract"]["profiles"][
                "interactive"
            ].update(recipe_sha256="a" * 64),
        ):
            with self.subTest(change=change):
                plan = Q.strict_json_loads(original, "synthetic plan")
                change(plan)
                path.write_bytes(Q.canonical_bytes(plan))
                self.manifest["suite"]["plan_sha256"] = Q.sha256_file(path)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_source_clock_and_recipe_bytes_must_still_match_inventory(self):
        source = Path(self.manifest["provenance"]["source_root_path"])
        target = source / "crates/spark-server/src/api/completions.rs"
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "differs from proven inventory"):
            self.validate()

    def test_clock_usage_and_request_wall_proof_disagree_even_after_rehash(self):
        self.change_journal(
            lambda rows: rows[1]["client_monotonic"].update(finish=103.0)
        )
        with self.assertRaisesRegex(ValueError, "client wall"):
            self.validate()

    def test_blocking_index_and_ttft_finish_geometry_are_not_optional(self):
        initial = copy.deepcopy(self.raw)
        path = (
            Path(self.manifest["runtime_dir"])
            / self.raw["response_journal"]["artifact"]
        )
        original = path.read_bytes()
        cases = [
            (kind, "index", value)
            for kind in (0, 1)
            for value in (-1, 1, True, "missing")
        ]
        cases += [
            (1, "finish_reason", value) for value in ("stop", "error", None, "missing")
        ]
        for index, key, value in cases:
            with self.subTest(index=index, key=key, value=value):
                self.raw = copy.deepcopy(initial)
                path.write_bytes(original)

                def change(rows):
                    choice = rows[index]["response"]["choices"][0]
                    if value == "missing":
                        choice.pop(key)
                    else:
                        choice[key] = value

                self.change_journal(change)
                with self.assertRaisesRegex(
                    ValueError, "blocking choice index|blocking finish reason"
                ):
                    self.validate()

    def test_dependency_artifact_global_uniqueness_policy_is_independent(self):
        # Inject only the normalized dependency locator after real gate/journal
        # verification, to test the independent cross-gate uniqueness policy.
        receipt = fixture.suite_receipt(self.manifest)
        validate_gate = Q._validate_gate_artifact
        collisions = [
            receipt["gates"]["startup-readiness"]["receipt_relpath"],
            receipt["gates"]["memory-floor"]["receipt_relpath"],
            self.manifest["suite"]["_stdout_relpath"],
            receipt["raw_log_safety"]["raw_log_relpath"],
        ]
        for locator in collisions:
            with self.subTest(locator=locator):

                def intercept(name, *args):
                    result = validate_gate(name, *args)
                    if name == "prefill-decode-profiles":
                        result["dependent_artifacts"][0]["artifact"] = locator
                    return result

                with (
                    mock.patch.object(Q, "_profile_runtime_recipe_binding"),
                    mock.patch.object(
                        Q, "_validate_gate_artifact", side_effect=intercept
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "unique release artifacts|globally unique|distinct release artifact",
                    ),
                ):
                    Q._validate_suite_receipt_bytes(
                        Q.canonical_bytes(receipt), self.manifest, fixture.CONTAINER_ID
                    )


if __name__ == "__main__":
    unittest.main()
