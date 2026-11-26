# SPDX-License-Identifier: AGPL-3.0-only
"""Offline-only tests for r15-gates/prefill-decode-profiles. All Docker, server and HTTP
execution is mocked; nothing here measures an engine or qualifies an image."""

import copy
import base64
import contextlib
from contextlib import ExitStack, contextmanager
import functools
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


S = module(ROOT / "r15-container-suite.py", "r16_profile_suite_fixture")
Q = module(ROOT / "r15_exact_image_release.py", "r16_profile_outer_fixture")
D = module(ROOT / "r15-gates" / "prefill-decode-profiles", "r16_profile_driver")

BINARY_SHA = "a" * 64
RECIPE_SHA = "c" * 64
HOST_PROVENANCE = {
    "source_revision": "1" * 40, "source_tree": "2" * 40,
    "source_bundle_manifest_sha256": "3" * 64,
    "checkpoint_id": "offline/fixture", "checkpoint_revision": "4" * 40,
    "checkpoint_manifest_sha256": "5" * 64,
}
LAUNCH_PLAN = {**HOST_PROVENANCE, "source_root_path": str(ROOT.parent),
               "in_container_binary_sha256": BINARY_SHA, "recipe_sha256": RECIPE_SHA}
# Mocked end-to-end runs assemble framed prompts for every context in two
# profiles x two arms; at the release grid (up to 262080 ids) that exceeds the
# 90 MB address-space cap these suites run under. The rules they check do not
# depend on prompt length, so those runs use this grid (same order, same
# 4096-token chunk crossing). Release sizes stay covered by
# test_assemble_prompt_is_exact_with_one_needle_each_at_every_context and
# test_decode_budget_respects_the_native_context.
EXERCISE_CONTEXTS = (1024, 2048, 4096, 8192, 12288, 16320)


def scaled_contexts(test):
    @functools.wraps(test)
    def wrapper(self, *args, **kwargs):
        # The whole body runs on the scaled grid, so host references built in
        # it match the measured workload and refusals fail for their own reason.
        with mock.patch.object(D, "PROFILE_CONTEXTS", EXERCISE_CONTEXTS):
            return test(self, *args, **kwargs)
    return wrapper


PASSKEY_TEXT = "ALPHA: alpha-0731826\nBETA: beta-0492175\nGAMMA: gamma-0863204\n" + "filler " * 40


def expected_wall(text, step=0.02):
    """The stream fixture emits one event per space-separated piece, 20 ms apart; empty pieces carry no text."""
    return step * (len([p for p in text.split(" ") if p]) - 1)


def fake_tokens(text):
    return [1000 + (hashlib.blake2b(w.encode(), digest_size=2).digest()[0] % 200) for w in text.split()] or [7]


@functools.cache
def host_prompt_hash(context):
    components = {key: fake_tokens(text) for key, text in D.workload_texts().items()}
    return D.prompt_sha256(D.assemble_prompt(context, components)[0])


def fixture_usage(context, requested, rate):
    return {"prompt_tokens": context, "completion_tokens": requested,
            "total_tokens": context + requested, "prompt_tokens_details": {"cached_tokens": 0},
            "response_token/s": rate}


def host_reference(k5_wall=5.0, plain_wall=7.5, prefill_wall=1.0, **overrides):
    def rows():
        return [
            {
                "context_tokens": c,
                "decode_tokens": D.decode_tokens(c),
                "k5_decode_wall_seconds": D.decode_tokens(c) / (D.decode_tokens(c) / k5_wall),
                "plain_decode_wall_seconds": D.decode_tokens(c) / (D.decode_tokens(c) / plain_wall),
                "prefill_wall_seconds": prefill_wall,
                "prompt_sha256": host_prompt_hash(c),
                "k5_usage": fixture_usage(c, D.decode_budget(c), D.decode_tokens(c) / k5_wall),
                "plain_usage": fixture_usage(c, D.decode_budget(c), D.decode_tokens(c) / plain_wall),
                "prefill_client_monotonic": {"start": 0.0, "finish": prefill_wall},
            }
            for c in D.PROFILE_CONTEXTS
        ]

    overlay = ["--max-prefill-tokens", "4096", "--max-seq-len", "262144"]
    # Distinct source-defined configs in a mock are not validated production recipes.
    overlays = {
        "interactive": list(overlay),
        "high-prefill": list(overlay) + ["--scheduling-policy", "slai"],
    }
    document = {
        "schema": D.HOST_REFERENCE_SCHEMA,
        "chunk_tokens": 4096,
        "binary_sha256": BINARY_SHA,
        "decode_budget_tokens": 256,
        "measurement_contract": dict(D.MEASUREMENT_CONTRACT),
        "provenance": dict(HOST_PROVENANCE),
        "profiles": {
            name: {"argv_overlay": overlays[name], "host_recipe_sha256": RECIPE_SHA, "rows": rows()}
            for name in D.PROFILE_NAMES
        },
    }
    document.update(overrides)
    return document


class PrefillDecodeProfilesDriver(unittest.TestCase):
    # ------------------------------------------------------------------ static contracts
    def test_checked_in_plan_example_satisfies_every_required_driver_argument(self):
        plan = S.strict_json_loads(
            (ROOT / "examples" / "r15-container-suite-plan.example.json").read_bytes(), "plan"
        )
        argv = plan["gate_jobs"]["prefill-decode-profiles"]["argv"][1:]
        substituted = S.substitute_argv(
            argv,
            {
                "container_id": "1" * 64,
                "image_id": "sha256:" + "2" * 64,
                "image_digest": "sha256:" + "3" * 64,
                "base_url": "http://127.0.0.1:8888",
                "run_id": "example",
            },
            5,
            6,
        )
        args = D.build_parser().parse_args(substituted)
        self.assertEqual(args.runtime_dir, "/results/r15-runtime")
        self.assertTrue(args.suite_program.endswith("r15-container-suite.py"))
        self.assertEqual(args.host_reference, "/results/prefill-decode-host-reference.json")
        self.assertEqual(args.container_port, 8888)
        example = S.strict_json_loads(
            (ROOT / "examples" / "prefill-decode-host-reference.example.json").read_bytes(),
            "host reference example",
        )
        self.assertNotEqual(example["schema"], D.HOST_REFERENCE_SCHEMA)
        with self.assertRaises(ValueError):
            self.load_reference(example)
        self.assertEqual(set(example["profiles"]), set(D.PROFILE_NAMES))

    def test_contract_constants_match_the_outer_wrapper(self):
        self.assertEqual(D.PROFILE_CONTEXTS, Q.PROFILE_CONTEXTS)
        self.assertEqual(D.PROFILE_NAMES, Q.PROFILE_NAMES)
        self.assertEqual(D.CHUNK_TOKENS, 4096)
        self.assertIn("prefill-decode-profiles", S.REQUIRED_GATES)
        self.assertEqual(
            D.MEASUREMENT_CONTRACT["completions_source_sha256"],
            D.digest((ROOT.parent / "crates/spark-server/src/api/completions.rs").read_bytes()),
        )
        self.assertEqual(
            D.MEASUREMENT_CONTRACT["cli_source_sha256"],
            D.digest((ROOT.parent / "crates/spark-server/src/cli.rs").read_bytes()),
        )

    def test_decode_budget_respects_the_native_context(self):
        self.assertEqual(D.decode_budget(8192), 256)
        self.assertEqual(D.decode_budget(262_080), 64)
        self.assertEqual(D.decode_tokens(262_080), 63)
        self.assertEqual(D.decode_tokens(131_072), 255)

    def test_assemble_prompt_is_exact_with_one_needle_each_at_every_context(self):
        components = {
            "head": [1, 2, 3, 4, 5, 6, 7, 8],
            "filler": [20 + i for i in range(23)],
            "tail": [90 + i for i in range(17)],
            "ALPHA": [501, 502, 503],
            "BETA": [601, 602, 603, 604],
            "GAMMA": [701, 702],
        }
        for context in D.PROFILE_CONTEXTS:
            prompt, positions = D.assemble_prompt(context, components)
            self.assertEqual(len(prompt), context)
            self.assertEqual(prompt[:8], components["head"])
            self.assertEqual(prompt[-17:], components["tail"])
            self.assertEqual(sorted(positions), ["ALPHA", "BETA", "GAMMA"])
            self.assertLess(positions["ALPHA"], positions["BETA"])
            self.assertLess(positions["BETA"], positions["GAMMA"])
        with self.assertRaises(ValueError):
            D.assemble_prompt(4096, components)
        bad = dict(components)
        bad["ALPHA"] = []
        with self.assertRaises(ValueError):
            D.assemble_prompt(8192, bad)

    # ------------------------------------------------------------------ host reference
    def load_reference(self, document):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "host.json"
            raw = S.canonical_bytes(document)
            path.write_bytes(raw)
            args = types.SimpleNamespace(
                suite=S, launch_plan=LAUNCH_PLAN,
                host_reference=str(path), host_reference_sha256=hashlib.sha256(raw).hexdigest()
            )
            return D.load_host_reference(args)

    def test_host_reference_fail_closed_mutations(self):
        document, table, _sha = self.load_reference(host_reference())
        self.assertEqual(set(table), set(D.PROFILE_NAMES))
        self.assertEqual(table["interactive"]["rows"][8192]["decode_tokens"], 255)
        bad = [
            lambda d: d.update(schema="other"),
            lambda d: d.update(chunk_tokens=16384),
            lambda d: d.update(decode_budget_tokens=512),
            lambda d: d.update(binary_sha256="z" * 64),
            lambda d: d["profiles"].pop("high-prefill"),
            lambda d: d["profiles"]["interactive"]["rows"].pop(),
            lambda d: d["profiles"]["interactive"]["rows"][0].update(decode_tokens=256),
            lambda d: d["profiles"]["interactive"]["rows"][0].update(k5_decode_wall_seconds=0.0),
            lambda d: d["profiles"]["interactive"]["rows"][0].update(k5_decode_wall_seconds=5),
            lambda d: d["profiles"]["interactive"]["rows"][0].pop("prefill_wall_seconds"),
            lambda d: d["profiles"]["interactive"]["rows"][0].update(prefill_wall_seconds=0.0),
            lambda d: d["profiles"]["interactive"]["rows"][0].update(prefill_wall_seconds=2),
            lambda d: d["profiles"]["interactive"]["argv_overlay"].append("--speculative"),
            lambda d: d["profiles"]["interactive"]["argv_overlay"].__setitem__(1, "16384"),
            lambda d: d["profiles"]["interactive"]["argv_overlay"].__setitem__(3, "135168"),
            lambda d: d["profiles"]["interactive"].update(extra=1),
        ]
        for change in bad:
            value = host_reference()
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load_reference(value)
        with self.assertRaisesRegex(ValueError, "hash differs"):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "host.json"
                path.write_bytes(S.canonical_bytes(host_reference()))
                D.load_host_reference(types.SimpleNamespace(
                    suite=S, launch_plan=LAUNCH_PLAN,
                    host_reference=str(path), host_reference_sha256="1" * 64))

    def test_host_reference_rejects_wrong_clock_source_checkpoint_and_raw_proofs(self):
        mutations = (
            lambda d: d["measurement_contract"].update(decode_clock="client-text-arrivals"),
            lambda d: d["measurement_contract"].update(prefill_clock="engine-ttft"),
            lambda d: d["measurement_contract"].update(completions_source_sha256="f" * 64),
            lambda d: d["provenance"].update(checkpoint_manifest_sha256="f" * 64),
            lambda d: d["provenance"].update(checkpoint_revision="f" * 40),
            lambda d: d["provenance"].update(source_tree="f" * 40),
            lambda d: d["profiles"]["interactive"]["rows"][0]["k5_usage"].update(**{"response_token/s": 1.0}),
            lambda d: d["profiles"]["interactive"]["rows"][0]["plain_usage"].update(completion_tokens=255),
            lambda d: d["profiles"]["interactive"]["rows"][0]["prefill_client_monotonic"].update(finish=100.0),
        )
        for mutation in mutations:
            document = copy.deepcopy(host_reference())
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load_reference(document)

    def test_host_reference_requires_canonical_bytes_for_replayable_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "host.json"
            raw = json.dumps(host_reference(), indent=2).encode()
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "not canonical"):
                D.load_host_reference(types.SimpleNamespace(
                    suite=S, launch_plan=LAUNCH_PLAN, host_reference=str(path),
                    host_reference_sha256=hashlib.sha256(raw).hexdigest()))

    def test_effective_profiles_normalize_order_defaults_and_unused_deadline(self):
        base = ["--max-prefill-tokens", "4096", "--max-seq-len", "262144"]
        expected = D.effective_overlay(base)
        for overlay in (
            base[2:] + base[:2], base + ["--scheduling-policy", "fifo"],
            base + ["--tbt-deadline-ms", "200"],
            ["--max-prefill-tokens", "04096", "--max-seq-len", "0262144"],
        ):
            self.assertEqual(D.effective_overlay(overlay), expected)
        for overlay in (
            base + ["--scheduling-policy", "fifo", "--scheduling-policy", "slai"],
            base + ["--unknown-selector", "1"], base + ["--temperature", "2"],
            base + ["--scheduling-policy=fifo"], base + ["--tbt-deadline-ms", "-1"],
        ):
            with self.subTest(overlay=overlay), self.assertRaises(ValueError):
                D.effective_overlay(overlay)

    def test_real_plan_holds_without_reviewed_source_profile_contract(self):
        _document, table, _sha = self.load_reference(host_reference())
        args = types.SimpleNamespace(suite=S, launch_plan=LAUNCH_PLAN)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            D.verify_profile_contract(args, table, {"recipe_sha256": RECIPE_SHA})

    def test_profile_contract_cannot_replace_recipe_semantics_with_policy_names(self):
        _document, table, _sha = self.load_reference(host_reference())
        contract = {"schema": "flux-performance-profile-contract/1", "profiles": {
            name: {"recipe_relpath": "invented/" + name, "recipe_sha256": RECIPE_SHA,
                   "argv_overlay": table[name]["argv_overlay"]} for name in D.PROFILE_NAMES
        }}
        args = types.SimpleNamespace(suite=S, launch_plan={**LAUNCH_PLAN, "performance_profile_contract": contract})
        with self.assertRaisesRegex(ValueError, "source-defined recipe"):
            D.verify_profile_contract(args, table, {"recipe_sha256": RECIPE_SHA})

    # ------------------------------------------------------------------ measurements
    def response(self, prompt_tokens, completion, text=PASSKEY_TEXT, ids=None):
        return {
            "id": "fixture-id",
            "choices": [{"index": 0, "text": text, "finish_reason": "length",
                         "token_ids": list(range(completion)) if ids is None else ids}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                      "total_tokens": prompt_tokens + completion,
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "time_to_first_token_ms": 1234.5},
        }

    def test_exact_and_ttft_measurements_fail_closed(self):
        good = D.exact_measurement(self.response(8192, 256), 8192, 256)
        self.assertTrue(good["accuracy_pass"])
        self.assertEqual(good["completion_tokens"], 256)
        self.assertFalse(D.exact_measurement(self.response(8192, 256, text="nothing"), 8192, 256)["accuracy_pass"])
        changes = (
            lambda r: r["usage"].update(prompt_tokens=8191),
            lambda r: r["usage"].update(completion_tokens=255),
            lambda r: r["usage"].update(total_tokens=1),
            lambda r: r["usage"]["prompt_tokens_details"].update(cached_tokens=3),
            lambda r: r["usage"].pop("prompt_tokens_details"),
            lambda r: r["choices"][0].pop("token_ids"),
            lambda r: r["choices"][0].update(token_ids=list(range(255))),
            lambda r: r["choices"][0].update(token_ids=[True] * 256),
            lambda r: r["choices"].append({}),
            lambda r: r["choices"][0].update(text=None),
        )
        for change in changes:
            value = self.response(8192, 256)
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                D.exact_measurement(value, 8192, 256)
        ttft = D.ttft_measurement(self.response(16384, 1), 16384, 2.5,
                                  client_monotonic={"start": 0.0, "finish": 2.5})
        self.assertEqual(ttft["prefill_prompt_tokens"], 16384)
        self.assertEqual(ttft["first_token_id"], 0)
        with self.assertRaises(ValueError):
            D.ttft_measurement(self.response(16384, 2), 16384, 2.5,
                               client_monotonic={"start": 0.0, "finish": 2.5})
        with self.assertRaises(ValueError):
            D.ttft_measurement(self.response(16384, 1), 16384, 0.0,
                               client_monotonic={"start": 0.0, "finish": 0.0})
        with self.assertRaises(ValueError):
            D.ttft_measurement(self.response(16384, 1), 16384, 2,
                               client_monotonic={"start": 0.0, "finish": 2.0})
        stopped = self.response(16384, 1)
        stopped["choices"][0]["finish_reason"] = "stop"
        with self.assertRaisesRegex(ValueError, "first-token finish reason"):
            D.ttft_measurement(stopped, 16384, 2.5,
                               client_monotonic={"start": 0.0, "finish": 2.5})

    def test_ttft_requires_exact_finite_monotonic_wall_proof(self):
        for clock in (
            {"start": 0.0, "finish": 3.0}, {"start": 2.5, "finish": 0.0},
            {"start": 0, "finish": 2.5}, {"start": 0.0, "finish": float("nan")},
            {"start": 0.0, "finish": 2.5, "extra": True}, {},
        ):
            with self.subTest(clock=clock), self.assertRaisesRegex(ValueError, "monotonic proof"):
                D.ttft_measurement(self.response(16384, 1), 16384, 2.5,
                                   client_monotonic=clock)

    def events(self, text, start=1.0, step=0.02, usage_tokens=256, prompt_tokens=8192, engine_rate=None):
        pieces = text.split(" ")
        events = []
        for index, piece in enumerate(pieces):
            body = {"id": "s", "choices": [{"index": 0, "text": piece + (" " if index + 1 < len(pieces) else "")}]}
            events.append((start + index * step, S.canonical_bytes(body)))
        wall = expected_wall(text)
        rate = ((usage_tokens - 1) / wall if wall > 0.0 else 1.0) if engine_rate is None else engine_rate
        done = {"id": "s", "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": usage_tokens,
                          "total_tokens": prompt_tokens + usage_tokens,
                          "prompt_tokens_details": {"cached_tokens": 0},
                          "response_token/s": rate}}
        events.append((start + len(pieces) * step, S.canonical_bytes(done)))
        return events

    def test_stream_measurement_excludes_ttft_and_fails_closed(self):
        measured = D.stream_measurement(self.events(PASSKEY_TEXT), 8192, 256, S)
        self.assertEqual(measured["decode_tokens"], 255)
        self.assertEqual(measured["streamed_text_sha256"], hashlib.sha256(PASSKEY_TEXT.encode()).hexdigest())
        self.assertAlmostEqual(measured["decode_wall_seconds"], expected_wall(PASSKEY_TEXT))
        with self.assertRaises(ValueError):
            D.stream_measurement(self.events(PASSKEY_TEXT, usage_tokens=200), 8192, 256, S)
        with self.assertRaises(ValueError):
            D.stream_measurement(self.events(PASSKEY_TEXT)[-1:], 8192, 256, S)
        with self.assertRaises(ValueError):
            D.stream_measurement([], 8192, 256, S)
        with self.assertRaises(ValueError):
            D.stream_measurement(self.events(PASSKEY_TEXT)[:-1], 8192, 256, S)

    def test_k5_metric_census(self):
        self.assertEqual(D.k5_events('atlas_spec_decode_verify_total{k="5",outcome="accepted"} 12\n'), 12)
        self.assertEqual(D.k5_events('atlas_spec_decode_verify_total{k="dflash"} 12\n'), 0)
        with self.assertRaises(ValueError):
            D.k5_events('atlas_spec_decode_verify_total{k="5"} 1.5\n')

    # ------------------------------------------------------------------ mocked end to end
    def exercise_mocked_run(self, mutation=None, *, host=None, missing_k5=False,
                            plan_binary=BINARY_SHA, plan_recipe=RECIPE_SHA, client_wall=1.0):
        return self._exercise_mocked_run(
            mutation, host=host, missing_k5=missing_k5, plan_binary=plan_binary,
            plan_recipe=plan_recipe, client_wall=client_wall)

    def _exercise_mocked_run(self, mutation=None, *, host=None, missing_k5=False,
                             plan_binary=BINARY_SHA, plan_recipe=RECIPE_SHA, client_wall=1.0):
        """Explicit mocked fixture: four fresh "servers", 72 journal entries, never qualification."""
        self.assertEqual(D.PROFILE_CONTEXTS, EXERCISE_CONTEXTS, "decorate the test with @scaled_contexts")
        for document in [host] if host is not None else []:
            for profile in document["profiles"].values():
                self.assertEqual([row["context_tokens"] for row in profile["rows"]], list(EXERCISE_CONTEXTS))
        mode = [None]
        read_fd, write_fd = os.pipe()
        try:
            with tempfile.TemporaryDirectory() as temp:
                runtime = Path(temp) / "runtime"
                runtime.mkdir()
                reference = Path(temp) / "host-reference.json"
                raw = S.canonical_bytes(host or host_reference())
                reference.write_bytes(raw)
                receipt = Path(temp) / "receipt.json"
                receipt_fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                args = types.SimpleNamespace(
                    suite=S, runtime_dir=str(runtime), receipt_fd=receipt_fd, container_port=8888,
                    launch_plan=LAUNCH_PLAN,
                    container_id="1" * 64, image_id="sha256:" + "2" * 64, image_digest="sha256:" + "3" * 64,
                    run_id="offline-fixture", host_reference=str(reference),
                    host_reference_sha256=hashlib.sha256(raw).hexdigest(), base_url="http://127.0.0.1:8888",
                )
                counter = [0]

                @contextmanager
                def fixture_server(_args, arm_mode, _plan, _recipe, overlay):
                    self.assertEqual(overlay[:2], ["--max-prefill-tokens", "4096"])
                    mode[0] = arm_mode
                    yield "fixture-model-" + arm_mode, {"mode": arm_mode, "offline_test_only": True}, read_fd

                def fixture_request(_args, method, route, body=None, **_kwargs):
                    if route == "/metrics":
                        if mode[0] == "none" or missing_k5:
                            return b"# no K5 verification event\n"
                        return b'atlas_spec_decode_verify_total{k="5",outcome="accepted"} 3\n'
                    if route == "/tokenize":
                        return S.canonical_bytes({"tokens": fake_tokens(body["prompt"]), "count": 0})
                    self.assertEqual(route, "/v1/completions")
                    counter[0] += 1
                    prompt_tokens = len(body["prompt_token_ids"])
                    completion = body["max_tokens"]
                    row = self.response(prompt_tokens, completion)
                    row["id"] = f"{mode[0]}-{counter[0]}"
                    # offline fixture: a deterministic engine prefill duration (0.1 ms)
                    row["usage"]["time_to_first_token_ms"] = 0.1
                    if mutation:
                        mutation(row, mode[0], prompt_tokens, "blocking")
                    return S.canonical_bytes(row)

                def fixture_stream(_args, route, body, **_kwargs):
                    counter[0] += 1
                    prompt_tokens = len(body["prompt_token_ids"])
                    budget = body["max_tokens"]
                    text = PASSKEY_TEXT
                    row = {"text": text}
                    if mutation:
                        mutation(row, mode[0], prompt_tokens, "stream")
                    # 20 ms per piece: wall = (pieces-1)*0.02 s; host reference uses the same wall
                    events = self.events(row["text"], usage_tokens=budget, prompt_tokens=prompt_tokens)
                    events = [(t, d.replace(b'"id":"s"', f'"id":"{mode[0]}-{counter[0]}"'.encode())) for t, d in events]
                    return events

                wall = expected_wall(PASSKEY_TEXT)
                if host is None:
                    raw = S.canonical_bytes(host_reference(k5_wall=wall, plain_wall=wall))
                    reference.write_bytes(raw)
                    args.host_reference_sha256 = hashlib.sha256(raw).hexdigest()
                patches = (
                    mock.patch.object(D, "inspect_container", return_value=(
                        {"in_container_binary_sha256": plan_binary, "in_container_binary_path": "/opt/flux/bin/x",
                         "in_container_recipe_path": "/opt/flux/repro-kit/recipes/r.env", "recipe_sha256": plan_recipe}, {})),
                    mock.patch.object(D, "server", side_effect=fixture_server),
                    mock.patch.object(D, "request", side_effect=fixture_request),
                    mock.patch.object(D, "stream_request", side_effect=fixture_stream),
                    # Legacy contract /1 fixture exercises driver arithmetic only
                    # (dual-recipe plan /4 policy is covered by verify_profile_contract_v2 tests).
                    mock.patch.object(D, "verify_profile_contract", side_effect=lambda _args, table, _plan: {
                        "schema": "flux-performance-profile-contract/1", "profiles": {
                            name: {"recipe_relpath": D.PROFILE_RECIPES[name][0],
                                   "recipe_sha256": RECIPE_SHA,
                                   "argv_overlay": table[name]["argv_overlay"]}
                            for name in D.PROFILE_NAMES
                        }}),
                    mock.patch.object(D.time, "monotonic", side_effect=iter(float(i) * client_wall for i in range(10000))),
                )
                try:
                    with ExitStack() as stack:
                        for patch in patches:
                            stack.enter_context(patch)
                        if (mutation or missing_k5 or host is not None or plan_binary != BINARY_SHA
                                or plan_recipe != RECIPE_SHA or client_wall != 1.0):
                            with self.assertRaises(ValueError):
                                D.run(args)
                            self.assertEqual(receipt.read_bytes(), b"")
                            self.assertFalse((runtime / "performance" / D.GATE / "validator.json").exists())
                            return None
                        result = D.run(args)
                        evidence = result["evidence"]
                        validator_raw = (runtime / evidence["validator_artifact"]).read_bytes()
                        self.assertEqual(hashlib.sha256(validator_raw).hexdigest(), evidence["validator_receipt_sha256"])
                        document = json.loads(validator_raw)
                        # This synthetic fixture checks arithmetic, not the
                        # independent production provenance/release closure.
                        self.assertEqual(len(document["raw"]["rows"]), len(evidence["rows"]))
                        for row, measured in zip(document["raw"]["rows"], evidence["rows"], strict=True):
                            self.assertEqual((row["profile"], row["context_tokens"]),
                                             (measured["profile"], measured["context_tokens"]))
                            self.assertEqual(row["prefill_prompt_tokens"], row["context_tokens"])
                            self.assertEqual(row["plain_decode_tokens"], D.decode_tokens(row["context_tokens"]))
                            self.assertEqual(row["k5_decode_tokens"], D.decode_tokens(row["context_tokens"]))
                            for key, expected in {
                                "prefill_tokens_per_second": row["prefill_prompt_tokens"] / row["prefill_ttft_seconds"],
                                "plain_decode_tokens_per_second": row["plain_decode_tokens"] / row["plain_decode_wall_seconds"],
                                "k5_decode_tokens_per_second": row["k5_decode_tokens"] / row["k5_decode_wall_seconds"],
                                "host_delta_percent": 100.0 * (row["host_decode_wall_seconds"]
                                                              - row["profile_decode_wall_seconds"])
                                                      / row["profile_decode_wall_seconds"],
                            }.items():
                                self.assertEqual(measured[key], expected)
                            self.assertIs(row["k5_lossless"], True)
                            self.assertIs(row["accuracy_pass"], True)
                            self.assertIs(measured["k5_lossless"], True)
                        for key in ("all_host_tolerances_met", "all_accuracy_checks_pass"):
                            self.assertIs(document["raw"][key], True)
                            self.assertIs(evidence[key], True)
                        journal = document["raw"]["response_journal"]
                        journal_raw = (runtime / journal["artifact"]).read_bytes()
                        self.assertEqual(hashlib.sha256(journal_raw).hexdigest(), journal["sha256"])
                        self.assertEqual(len(journal_raw.splitlines()), 2 * 2 * 6 * 3)
                        entries = [S.strict_json_loads(line, "journal entry") for line in journal_raw.splitlines()]
                        self.assertEqual({(entry["profile"], entry["mode"]) for entry in entries},
                                         {(profile, mode) for profile in D.PROFILE_NAMES for mode in D.MODES})
                        self.assertEqual({entry["profile"] for entry in entries[:36]}, {"interactive"})
                        self.assertEqual(journal["requests"], 72)
                        self.assertEqual(oct(os.stat(runtime / journal["artifact"]).st_mode & 0o777), "0o400")
                        self.assertEqual(json.loads(receipt.read_bytes()), result)
                        self.assertEqual(len(evidence["rows"]), 12)
                        self.assertEqual([row["context_tokens"] for row in evidence["rows"]][:6], list(D.PROFILE_CONTEXTS))
                        # the fixture host wall equals the fixture stream wall up to float roundoff
                        self.assertTrue(all(abs(row["host_delta_percent"]) < 1e-6 for row in evidence["rows"]))
                        self.assertEqual(document["raw"]["rows"][5]["k5_decode_tokens"],
                                         D.decode_tokens(EXERCISE_CONTEXTS[5]))
                        self.assertEqual(document["raw"]["host_reference_sha256"], args.host_reference_sha256)
                        self.assertEqual(len(document["raw"]["host_bands"]), 12)
                        identity = document["raw"]["profile_identity"]
                        self.assertEqual(identity["image_recipe_sha256"], RECIPE_SHA)
                        self.assertFalse(identity["profiles_share_argv"])
                        for name in D.PROFILE_NAMES:
                            self.assertEqual(identity["profiles"][name]["host_recipe_sha256"], RECIPE_SHA)
                            self.assertEqual(identity["profiles"][name]["source_tree"], HOST_PROVENANCE["source_tree"])
                        for binding in document["raw"]["process_bindings"]:
                            self.assertEqual(set(binding), {"profile", "mode", "proof", "launch_argv",
                                                            "launch_argv_sha256"})
                            self.assertEqual(binding["launch_argv_sha256"],
                                             D.digest(S.canonical_bytes(binding["launch_argv"])))
                            self.assertEqual(identity["profiles"][binding["profile"]]["process_proof_sha256"][binding["mode"]],
                                             D.digest(S.canonical_bytes(binding["proof"])))
                        self.assertEqual({(metric["profile"], metric["mode"])
                                          for metric in document["raw"]["verification_metrics"]},
                                         {(profile, mode) for profile in D.PROFILE_NAMES for mode in D.MODES})
                        return result
                finally:
                    os.close(receipt_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    @scaled_contexts
    def test_mocked_end_to_end_passes_driver_arithmetic_recomputation(self):
        result = self.exercise_mocked_run()
        self.assertEqual(result["gate"], D.GATE)
        self.assertEqual(result["evidence"]["chunk_tokens"], 4096)

    @scaled_contexts
    def test_mocked_end_to_end_refuses_token_identity_drift(self):
        def different_tokens(row, mode, prompt_tokens, kind):
            if mode == "native-mtp-k5" and kind == "blocking" and prompt_tokens == EXERCISE_CONTEXTS[2] and len(row["choices"][0]["token_ids"]) > 1:
                row["choices"][0]["token_ids"][3] = 999_999
        self.exercise_mocked_run(different_tokens)

    @scaled_contexts
    def test_mocked_end_to_end_refuses_missing_passkeys(self):
        def no_passkeys(row, mode, prompt_tokens, kind):
            if mode == "none" and prompt_tokens == EXERCISE_CONTEXTS[3]:
                if kind == "blocking":
                    row["choices"][0]["text"] = "filler " * 40
                else:
                    row["text"] = "filler " * 40
        self.exercise_mocked_run(no_passkeys)

    @scaled_contexts
    def test_mocked_end_to_end_refuses_stream_text_drift(self):
        def stream_drift(row, mode, prompt_tokens, kind):
            if kind == "stream" and prompt_tokens == EXERCISE_CONTEXTS[0]:
                row["text"] = PASSKEY_TEXT + "extra"
        self.exercise_mocked_run(stream_drift)

    @scaled_contexts
    def test_mocked_end_to_end_refuses_host_band_breach(self):
        wall = expected_wall(PASSKEY_TEXT)
        self.exercise_mocked_run(host=host_reference(k5_wall=wall * 1.06, plain_wall=wall))

    @scaled_contexts
    def test_mocked_end_to_end_refuses_missing_native_k5_verifier_events(self):
        self.exercise_mocked_run(missing_k5=True)

    @scaled_contexts
    def test_mocked_end_to_end_refuses_host_reference_from_another_binary(self):
        self.exercise_mocked_run(plan_binary="d" * 64)

    @scaled_contexts
    def test_mocked_end_to_end_refuses_reused_response_ids(self):
        def reused(row, mode, prompt_tokens, kind):
            if kind == "blocking":
                row["id"] = "reused"
        self.exercise_mocked_run(reused)

    @scaled_contexts
    def test_mocked_end_to_end_refuses_plain_host_band_breach(self):
        wall = expected_wall(PASSKEY_TEXT)
        self.exercise_mocked_run(host=host_reference(k5_wall=wall, plain_wall=wall * 1.06))

    @scaled_contexts
    def test_mocked_end_to_end_refuses_prefill_host_band_breach(self):
        wall = expected_wall(PASSKEY_TEXT)
        self.exercise_mocked_run(
            host=host_reference(k5_wall=wall, plain_wall=wall, prefill_wall=0.0002)
        )

    @scaled_contexts
    def test_request_wall_prefill_band_cannot_be_hidden_by_fast_engine_ttft(self):
        self.exercise_mocked_run(client_wall=100.0)

    @scaled_contexts
    def test_mocked_end_to_end_refuses_host_recipe_from_another_recipe(self):
        self.exercise_mocked_run(plan_recipe="d" * 64)

    def test_stream_measurement_uses_engine_rate_over_text_wall_and_requires_rate(self):
        measured = D.stream_measurement(self.events(PASSKEY_TEXT, engine_rate=1.0), 8192, 256, S)
        self.assertEqual(measured["decode_wall_seconds"], 255.0)
        self.assertNotEqual(measured["decode_wall_seconds"], measured["observed_text_event_wall_seconds"])
        broken = self.events(PASSKEY_TEXT)
        body = S.strict_json_loads(broken[-1][1], "fixture")
        body["usage"].pop("response_token/s")
        broken[-1] = (broken[-1][0], S.canonical_bytes(body))
        with self.assertRaisesRegex(ValueError, "engine decode rate is missing"):
            D.stream_measurement(broken, 8192, 256, S)

    def test_stream_measurement_accepts_coalesced_transport_using_engine_rate(self):
        usage = {"prompt_tokens": 8192, "completion_tokens": 256, "total_tokens": 8448,
                 "prompt_tokens_details": {"cached_tokens": 0},
                 "response_token/s": 255.0}
        events = [
            (1.0, S.canonical_bytes({"id": "s", "choices": [{"index": 0, "text": "a"}]})),
            (1.02, S.canonical_bytes({"id": "s", "choices": [{"index": 0, "text": "b"}]})),
            (1.03, S.canonical_bytes({"id": "s", "choices": [{"index": 0, "text": "",
                                                               "finish_reason": "length"}],
                                      "usage": usage})),
        ]
        measured = D.stream_measurement(events, 8192, 256, S)
        self.assertEqual(measured["decode_wall_seconds"], 1.0)
        self.assertAlmostEqual(measured["observed_text_event_wall_seconds"], 0.02)

    def test_mocked_failed_claim_never_signals_an_unproven_pid(self):
        self.exercise_mocked_server("none", claim_error="exe hash differs")

    @scaled_contexts
    def test_mocked_run_holds_when_profiles_share_one_argv(self):
        shared = host_reference()
        shared["profiles"]["high-prefill"]["argv_overlay"] = list(
            shared["profiles"]["interactive"]["argv_overlay"]
        )
        self.exercise_mocked_run(host=shared)

    @scaled_contexts
    def test_mocked_run_holds_reordered_or_explicit_default_fifo_profiles(self):
        for extra in ([], ["--scheduling-policy", "fifo"], ["--tbt-deadline-ms", "1000"]):
            shared = host_reference()
            base = shared["profiles"]["interactive"]["argv_overlay"]
            shared["profiles"]["high-prefill"]["argv_overlay"] = base[2:] + base[:2] + extra
            with self.subTest(extra=extra):
                self.exercise_mocked_run(host=shared)

    def test_healthy_one_text_event_per_output_token_is_accepted(self):
        events = self.events(" ".join("x" for _ in range(256)), engine_rate=100.0)
        measured = D.stream_measurement(events, 8192, 256, S)
        self.assertEqual(measured["stream_text_events"], 256)
        self.assertEqual(measured["decode_tokens"], 255)
        self.assertEqual(measured["decode_wall_seconds"], 2.55)

    def test_first_coalesced_sse_cannot_hide_15_percent_engine_regression(self):
        pieces = [PASSKEY_TEXT[:40], *PASSKEY_TEXT[40:255], PASSKEY_TEXT[255:]]
        self.assertEqual(len(pieces), 217)
        events = [(float(i) / 100.0, S.canonical_bytes({"id": "s", "choices": [
            {"index": 0, "text": piece}]})) for i, piece in enumerate(pieces)]
        events.append((2.17, S.canonical_bytes({"id": "s", "choices": [
            {"index": 0, "text": "", "finish_reason": "length"}],
            "usage": fixture_usage(8192, 256, 100.0)})))
        stream = D.stream_measurement(events, 8192, 256, S)
        self.assertEqual(stream["decode_wall_seconds"], 2.55)
        self.assertEqual(stream["observed_text_event_wall_seconds"], 2.16)
        arm = {"exact": D.exact_measurement(self.response(8192, 256), 8192, 256),
               "ttft": D.ttft_measurement(self.response(8192, 1), 8192, 2.0,
                                          client_monotonic={"start": 0.0, "finish": 2.0}),
               "first_token_id_exact": 0, "stream": stream}
        row = host_reference(k5_wall=2.16, plain_wall=2.55, prefill_wall=2.0)["profiles"]["interactive"]["rows"][0]
        with self.assertRaisesRegex(ValueError, "host decode band exceeded"):
            D.build_row("interactive", 8192, arm, arm, row)

    def test_stream_clock_finish_and_identity_fail_closed(self):
        for kind in ("clock", "finish", "identity", "duplicate-usage"):
            events = self.events(PASSKEY_TEXT)
            if kind == "clock":
                events[1] = (-1.0, events[1][1])
            elif kind == "duplicate-usage":
                events.append(events[-1])
            else:
                at = -1 if kind == "finish" else 1
                value = S.strict_json_loads(events[at][1], "fixture")
                if kind == "finish":
                    value["choices"][0]["finish_reason"] = "stop"
                else:
                    value["id"] = "other"
                events[at] = (events[at][0], S.canonical_bytes(value))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                D.stream_measurement(events, 8192, 256, S)

    def test_live_argv_matches_exact_launch_on_the_same_live_pidfd(self):
        argv = ["/opt/flux/bin/engine", "serve", "--max-prefill-tokens", "4096"]
        encoded = b"\0".join(item.encode() for item in argv) + b"\0"
        with mock.patch.object(D.Path, "read_bytes", return_value=encoded), \
             mock.patch.object(D.select, "select", return_value=([], [], [])) as poll:
            D.verify_live_argv(123, 77, argv)
            self.assertEqual(poll.call_count, 2)
        for bad in (encoded.replace(b"4096", b"8192"), encoded + b"--unknown\0", encoded[:-1]):
            with mock.patch.object(D.Path, "read_bytes", return_value=bad), \
                 mock.patch.object(D.select, "select", return_value=([], [], [])), \
                 self.assertRaisesRegex(ValueError, "live server argv differs"):
                D.verify_live_argv(123, 77, argv)
        for polls in (([77], [77]), ([], [77])):
            with mock.patch.object(D.Path, "read_bytes", return_value=encoded), \
                 mock.patch.object(D.select, "select", side_effect=[(ready, [], []) for ready in polls]), \
                 self.assertRaisesRegex(ValueError, "server exited"):
                D.verify_live_argv(123, 77, argv)

    def test_launcher_exit_before_readiness_fails_closed(self):
        self.exercise_mocked_server("none", launcher_exited=True)

    # ------------------------------------------------------------------ lifecycle
    def exercise_mocked_server(self, mode, *, force_cleanup=False, verifier_failure=False,
                               claim_error=None, launcher_exited=False, missing_ack=False,
                               popen_failure=False, signal_error=False):
        read_fd, write_fd = os.pipe()
        dead = [False]
        process = mock.Mock()
        process.poll.return_value = 1 if launcher_exited else None
        token = []

        class Control(io.BytesIO):
            def close(self):
                if not self.closed:
                    process.stdout = io.BytesIO(b"" if missing_ack else (
                        "R16_EXEC_CLEANUP token=" + token[0]
                        + " child_started=true retired=true forced=false\n"
                    ).encode("ascii"))
                super().close()

        process.stdin = Control()

        def popen(command, **_kwargs):
            if popen_failure:
                raise OSError("launch failed")
            token.append(command[-1])
            return process

        def wait(**_kwargs):
            process.poll.return_value = 0
            return 0

        process.wait.side_effect = wait
        proof = {"offline_test_only": True, "mode": mode}
        verifier = mock.Mock(side_effect=ValueError("live binding failed") if verifier_failure else None,
                             return_value=proof)
        args = types.SimpleNamespace(
            container_id="1" * 64, container_port=8888,
            suite=types.SimpleNamespace(verify_live_server=verifier),
            identities={"base_url": "http://127.0.0.1:8888"},
        )

        def sent(_descriptor, signum):
            if signal_error:
                raise PermissionError("pidfd signal denied")
            if signum == signal.SIGKILL or not force_cleanup:
                dead[0] = True

        def claim(*_args):
            if claim_error:
                raise ValueError(claim_error)
            return os.dup(read_fd)

        overlay = ["--max-prefill-tokens", "4096", "--max-seq-len", "262144"]
        plan = {"in_container_binary_path": "/opt/flux/bin/flux-server-real"}
        try:
            with mock.patch.object(D.subprocess, "Popen", side_effect=popen) as launch, \
                 mock.patch.object(D, "find_pid", side_effect=[123, None]) as locate, \
                 mock.patch.object(D, "claim_pid", side_effect=claim) as claim_pid, \
                 mock.patch.object(D, "verify_live_argv") as check_argv, \
                 mock.patch.object(D, "request", return_value=b"{}") as endpoint, \
                 mock.patch.object(D.select, "select",
                                   side_effect=lambda descriptors, _w, _e, _t: (descriptors if dead[0] else [], [], [])), \
                 mock.patch.object(D.signal, "pidfd_send_signal", side_effect=sent) as kill:
                if popen_failure:
                    with self.assertRaisesRegex(OSError, "launch failed"):
                        with D.server(args, mode, plan, {"FLUX_MODE": "x"}, overlay):
                            pass
                    return
                if claim_error:
                    with self.assertRaisesRegex(ValueError, claim_error):
                        with D.server(args, mode, plan, {"FLUX_MODE": "x"}, overlay):
                            pass
                    self.assertEqual(kill.call_args_list, [])
                elif force_cleanup or verifier_failure or launcher_exited or missing_ack or signal_error:
                    pattern = ("forced cleanup" if force_cleanup else "live binding" if verifier_failure
                               else "launcher exited" if launcher_exited else "cleanup")
                    with self.assertRaisesRegex(ValueError, pattern):
                        with D.server(args, mode, plan, {"FLUX_MODE": "x"}, overlay):
                            pass
                else:
                    with D.server(args, mode, plan, {"FLUX_MODE": "x"}, overlay) as (_model, actual, _pidfd):
                        self.assertEqual(actual, proof)
                    self.assertEqual(verifier.call_count, 2)
                    self.assertEqual(check_argv.call_count, 2)
                if not force_cleanup and not claim_error and not launcher_exited:
                    self.assertEqual(endpoint.call_count, 2)
                if claim_error:
                    self.assertEqual(endpoint.call_count, 0)
                command = launch.call_args.args[0]
                self.assertEqual(command[:4], ["docker", "exec", "-i", "1" * 64])
                self.assertEqual(command[4], "/usr/bin/env")
                self.assertIn("ATLAS_DISABLE_SPECULATION=" + ("1" if mode == "none" else "0"), command)
                self.assertIn("FLUX_MODE=x", command)
                self.assertEqual(command[-3], D.EXEC_SUPERVISOR)
                argv = json.loads(base64.b64decode(command[-2], validate=True))
                for flag, value in (("--model-from-path", "/models"), ("--max-prefill-tokens", "4096"),
                                    ("--max-seq-len", "262144"), ("--enable-prefix-caching", "false")):
                    self.assertEqual(argv[argv.index(flag) + 1], value)
                self.assertIn("--no-tui", argv)
                self.assertNotIn("--mtp-gate", argv)
                self.assertNotIn("--mtp-from-path", argv)
                self.assertEqual("--speculative" in argv, mode == "native-mtp-k5")
                if mode == "native-mtp-k5":
                    self.assertEqual(argv[argv.index("--num-drafts") + 1], "4")
                self.assertTrue(process.stdin.closed)
                self.assertTrue(process.stdout.closed)
                if claim_error:
                    self.assertEqual(locate.call_count, 1)
                    self.assertEqual(claim_pid.call_count, 1)
                else:
                    self.assertEqual([call.args[1] for call in kill.call_args_list],
                                     [] if launcher_exited else [signal.SIGTERM, signal.SIGKILL]
                                     if force_cleanup else [signal.SIGTERM])
                process.wait.assert_called()
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_mocked_lifecycle_uses_canonical_argv_double_health_and_exact_cleanup(self):
        for mode in D.MODES:
            self.exercise_mocked_server(mode)

    def test_mocked_forced_cleanup_fails_the_gate(self):
        self.exercise_mocked_server("none", force_cleanup=True)

    def test_mocked_binding_failure_still_terminates_the_claimed_server(self):
        self.exercise_mocked_server("none", verifier_failure=True)

    def test_mocked_missing_cleanup_ack_fails_closed(self):
        self.exercise_mocked_server("none", missing_ack=True)

    def test_mocked_popen_failure_does_not_enter_partial_process_cleanup(self):
        self.exercise_mocked_server("none", popen_failure=True)

    def test_mocked_host_pidfd_error_still_closes_control_and_stdout(self):
        self.exercise_mocked_server("none", signal_error=True)

    def exercise_supervisor(self, *, before_go=False, exited=False, denied=None,
                            forced=False, survived=False, pidfd_failure=False):
        namespace = {"__name__": "offline_profile_supervisor"}
        exec(compile(D.EXEC_SUPERVISOR, "<sealed-profile-supervisor>", "exec"), namespace)
        child = mock.Mock(pid=4321)
        child.wait.side_effect = (
            [D.subprocess.TimeoutExpired("owned-child", 10),
             D.subprocess.TimeoutExpired("owned-child", 5) if survived else 0]
            if forced or survived else [0]
        )
        def send(descriptor, signum):
            self.assertEqual(descriptor, 77)
            if signum == denied:
                raise PermissionError("pidfd signal denied")
        captured = io.StringIO()
        with (
            mock.patch.object(D.os, "read", side_effect=[b"" if before_go else b"G", b""]),
            mock.patch.object(D.os, "pidfd_open", return_value=77,
                              side_effect=OSError("pidfd setup failed") if pidfd_failure else None),
            mock.patch.object(D.os, "close") as close,
            mock.patch.object(D.select, "select", side_effect=[([77] if exited else [0], [], []),
                                                               ([77] if exited else [], [], [])]),
            mock.patch.object(D.subprocess, "Popen", return_value=child) as popen,
            mock.patch.object(D.signal, "pidfd_send_signal", side_effect=send) as signal_child,
            contextlib.redirect_stdout(captured),
        ):
            error = (D.subprocess.TimeoutExpired if survived else PermissionError if denied
                     else OSError if pidfd_failure else None)
            if error:
                with self.assertRaises(error):
                    namespace["supervise"](["/sealed/engine", "serve"], "a" * 64)
            else:
                self.assertEqual(namespace["supervise"](["/sealed/engine", "serve"], "a" * 64), int(forced))
            if before_go:
                popen.assert_not_called()
                child.wait.assert_not_called()
                close.assert_not_called()
            else:
                popen.assert_called_once_with(["/sealed/engine", "serve"], stdin=D.subprocess.DEVNULL, stdout=2)
                child.wait.assert_any_call(timeout=10)
                if forced or survived:
                    child.wait.assert_any_call(timeout=5)
                if not pidfd_failure:
                    close.assert_called_once_with(77)
                if denied == signal.SIGTERM or pidfd_failure:
                    child.terminate.assert_called_once_with()
                if denied == signal.SIGKILL:
                    child.kill.assert_called_once_with()
                if exited:
                    signal_child.assert_not_called()
                    child.terminate.assert_not_called()
                    child.kill.assert_not_called()
            if denied or survived:
                self.assertEqual(captured.getvalue(), "")
            else:
                self.assertIn("retired=true forced=" + str(forced).lower(), captured.getvalue())

    def test_supervisor_eof_before_go_does_not_launch_a_child(self):
        self.exercise_supervisor(before_go=True)

    def test_supervisor_eof_reaps_only_its_pidfd_owned_child(self):
        self.exercise_supervisor()

    def test_supervisor_already_exited_child_gets_no_stale_numeric_pid_signal(self):
        self.exercise_supervisor(exited=True)

    def test_supervisor_term_permission_error_falls_back_and_still_reaps(self):
        self.exercise_supervisor(denied=signal.SIGTERM)

    def test_supervisor_kill_permission_error_falls_back_and_still_reaps(self):
        self.exercise_supervisor(denied=signal.SIGKILL, forced=True)

    def test_supervisor_pidfd_setup_failure_retires_its_direct_owned_child(self):
        self.exercise_supervisor(pidfd_failure=True)

    def test_supervisor_never_acknowledges_an_unreaped_child(self):
        self.exercise_supervisor(survived=True, denied=signal.SIGKILL)

    def test_symlink_runtime_artifact_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "outside").mkdir()
            (root / "performance").symlink_to(root / "outside", target_is_directory=True)
            with self.assertRaises(OSError):
                D.make_directory(S, root, ["performance", D.GATE])

    def test_parent_verifier_pin_requires_sealed_actual_launch_bytes(self):
        descriptor = S.anonymous_writable_file("offline-parent-suite")
        S.write_all(descriptor, b"offline sealed suite fixture")
        expected = hashlib.sha256(b"offline sealed suite fixture").hexdigest()
        try:
            with mock.patch.object(D.os, "getppid", return_value=123), \
                 mock.patch.object(D.Path, "read_bytes", return_value=b"python3\0/proc/self/fd/99\0"), \
                 mock.patch.object(D.os, "open", side_effect=lambda *_a, **_k: os.dup(descriptor)):
                with self.assertRaisesRegex(ValueError, "not sealed"):
                    D.verify_parent_suite(expected)
                S.seal_written_file(descriptor, "offline parent", 1000)
                D.verify_parent_suite(expected)
                with self.assertRaisesRegex(ValueError, "differs"):
                    D.verify_parent_suite("1" * 64)
        finally:
            os.close(descriptor)

    def test_source_dependency_hash_mismatch_fails_before_module_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "untrusted.py"
            path.write_text("raise AssertionError('must not execute')\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                D.load_module(path, "1" * 64, "untrusted_profile_dependency")

    def test_parent_plan_uses_the_actual_parent_descriptor_and_closes_it(self):
        suite = types.SimpleNamespace(load_plan=mock.Mock(return_value={"sealed": True}))
        with mock.patch.object(D.os, "getppid", return_value=123), \
             mock.patch.object(D.Path, "read_bytes", return_value=b"python3\0/proc/self/fd/99\0--plan-fd\0" b"75\0"), \
             mock.patch.object(D.os, "open", return_value=77) as opened, \
             mock.patch.object(D.os, "close") as closed:
            self.assertEqual(D.load_parent_plan(suite), {"sealed": True})
            opened.assert_called_once_with("/proc/123/fd/75", D.os.O_RDONLY | D.os.O_CLOEXEC)
            suite.load_plan.assert_called_once_with(77)
            closed.assert_called_once_with(77)

    def test_parent_plan_missing_ambiguous_unsealed_or_changed_parent_fails_closed(self):
        for command in (b"python3\0", b"python3\0--plan-fd\0", b"python3\0--plan-fd\0x\0",
                        b"python3\0--plan-fd\0" b"75\0--plan-fd\0" b"76\0"):
            with mock.patch.object(D.os, "getppid", return_value=123), \
                 mock.patch.object(D.Path, "read_bytes", return_value=command), \
                 mock.patch.object(D.os, "open") as opened, self.assertRaises(ValueError):
                D.load_parent_plan(S)
            opened.assert_not_called()
        descriptor = S.anonymous_writable_file("offline-parent-plan")
        try:
            with mock.patch.object(D.os, "getppid", return_value=123), \
                 mock.patch.object(D.Path, "read_bytes", return_value=b"python3\0--plan-fd\0" b"75\0"), \
                 mock.patch.object(D.os, "open", side_effect=lambda *_a, **_k: os.dup(descriptor)), \
                 self.assertRaisesRegex(ValueError, "not fully sealed"):
                D.load_parent_plan(S)
        finally:
            os.close(descriptor)
        suite = types.SimpleNamespace(load_plan=mock.Mock(return_value={"sealed": True}))
        with mock.patch.object(D.os, "getppid", side_effect=[123, 124]), \
             mock.patch.object(D.Path, "read_bytes", return_value=b"python3\0--plan-fd\0" b"75\0"), \
             mock.patch.object(D.os, "open", return_value=77), \
             mock.patch.object(D.os, "close") as closed, \
             self.assertRaisesRegex(ValueError, "launching suite changed"):
            D.load_parent_plan(suite)
        closed.assert_called_once_with(77)


if __name__ == "__main__":
    unittest.main()
