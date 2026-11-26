# SPDX-License-Identifier: AGPL-3.0-only
"""Offline checks for the profile v4 packet: sealed dual recipes, live startup
proofs, the K5 tail capacity proof and the host producer.

Synthetic logs, argv, /proc trees and HTTP responses only. No process,
container, model or device execution, and no measured number.
"""

import copy
import hashlib
import importlib.machinery
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
ROOT = KIT.parent


def module(path, name):
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


P = module(KIT / "r16_profile_host_producer.py", "r16_profile_host_producer")
G, S, D = P.G, P.S, P.D
Q = module(KIT / "r15_exact_image_release.py", "r16_packet_release_tool")
EXE = G.REVIEWED_EXECUTABLE["path"]
ALIAS = "flux-profile-prefill-decode-profiles-" + "0123456789abcdef" * 2
ARMS = [(profile, mode) for profile in G.PROFILE_NAMES for mode in G.MODES]
PASSKEY_TEXT = "ALPHA: alpha-0731826\nBETA: beta-0492175\nGAMMA: gamma-0863204\n" + "filler " * 40
OWN_PIDFD = os.pidfd_open
PROVENANCE = {
    "source_revision": "1" * 40, "source_tree": "2" * 40, "source_bundle_manifest_sha256": "3" * 64,
    "checkpoint_id": "offline/fixture", "checkpoint_revision": "4" * 40, "checkpoint_manifest_sha256": "5" * 64,
}


def argv_file(path):
    return [line for line in (ROOT / path).read_text().splitlines() if line]


def flag_map(argv):
    values, index = {}, 0
    while index < len(argv):
        item = argv[index]
        if item.startswith("--"):
            bare = index + 1 >= len(argv) or argv[index + 1].startswith("--")
            values[item] = None if bare else argv[index + 1]
            index += 1 if bare else 2
        else:
            index += 1
    return values


def startup_lines(arms, *, drift=None):
    lines = []
    for profile, mode in arms:
        config = G.expected_prefill_config(profile, mode)
        pool = G.expected_kv_pool(profile)
        if drift:
            config, pool = drift(profile, mode, dict(config), dict(pool))
        lines.append("\x1b[2m2026-10-23T00:00:00Z\x1b[0m \x1b[32m INFO\x1b[0m spark_server::kv_cache: Prefill config: "
                     + ", ".join(f"{key}={value}" for key, value in config.items()))
        lines.append(f"2026-10-23T00:00:01Z  INFO spark_model::factory::build: KV cache (post-construction): "
                     f"84.2 GB free, 80.1 GB allocatable, {pool['blocks']} blocks × {pool['block_size']} "
                     f"tok/block = {pool['max_tokens']} max tokens")
    return ("\n".join(lines) + "\n").encode()


class SealedRecipes(unittest.TestCase):
    def test_checked_in_recipes_are_the_source_derivation(self):
        for profile, relpath in G.SEALED_RECIPE_RELPATHS.items():
            raw = (ROOT / relpath).read_bytes()
            self.assertEqual(raw, G.sealed_recipe_bytes(profile))
            recipe = json.loads(raw)
            self.assertEqual(recipe["schema"], "flux-profile-recipe/1")
            self.assertEqual(recipe["geometry_sha256"], G.REVIEWED_GEOMETRY_SHA256)
            self.assertEqual(recipe["environment_recipe"]["sha256"], hashlib.sha256(
                (ROOT / recipe["environment_recipe"]["relpath"]).read_bytes()).hexdigest())
            self.assertEqual(recipe["versions"]["plan"], "flux-r15-container-suite-plan/3")

    def test_engine_argv_is_the_host_recipe_argv(self):
        """The only differences from recipes/flux-serve*.argv are transport,
        identity, the native 262144 window and the explicit FIFO pins."""
        for profile, host in (("high-prefill", "recipes/flux-serve.argv"),
                              ("interactive", "recipes/flux-serve-w4096.argv")):
            self.assertEqual(argv_file(host), argv_file("repro-kit/" + host))
            host_flags = flag_map(argv_file(host))
            sealed = flag_map(G.reviewed_effective_argv(profile, "native-mtp-k5"))
            self.assertEqual(set(host_flags) - set(sealed), set())
            self.assertEqual(set(sealed) - set(host_flags), {"--scheduling-policy", "--tbt-deadline-ms"})
            differ = {flag for flag in host_flags if host_flags[flag] != sealed[flag]}
            self.assertEqual(differ, {"--bind", "--port", "--model-from-path", "--model-name", "--max-seq-len"})
            self.assertEqual((host_flags["--max-seq-len"], sealed["--max-seq-len"]), ("135168", "262144"))
            self.assertEqual(sealed["--max-prefill-tokens"], str(G.REVIEWED_GEOMETRY[profile]["chunk_tokens"]))

    def test_every_tool_builds_the_same_full_argv(self):
        for profile, mode in ARMS:
            chunk = G.REVIEWED_GEOMETRY[profile]["chunk_tokens"]
            overlay = S.reviewed_profile_overlay_v2(chunk)
            self.assertEqual(overlay, G.reviewed_overlay(profile))
            expected = G.reviewed_effective_argv(profile, mode)
            model, port = S.PROFILE_LAUNCH_MODEL, S.PROFILE_LAUNCH_PORT
            self.assertEqual(S.profile_launch_argv(EXE, model, port, overlay, mode), expected)
            self.assertEqual(Q._profile_launch_argv(EXE, model, port, overlay, mode), expected)
            self.assertEqual(D.server_argv({"in_container_binary_path": EXE}, mode, model, port, overlay), expected)
            self.assertEqual(S.profile_launch_template_sha256(EXE, overlay, mode),
                             P.sealed_recipe(profile)["launch_argv_sha256"][mode])
        contract = P.contract_v2()
        Q._profile_contract_v2(copy.deepcopy(contract), EXE)

    def test_one_policy_qsa_and_chunk_only_difference(self):
        for mode in G.MODES:
            argv = {profile: G.reviewed_effective_argv(profile, mode) for profile in G.PROFILE_NAMES}
            for item in argv.values():
                self.assertEqual(item.count("--qwen4-qsa"), 1)
                self.assertEqual(flag_map(item)["--scheduling-policy"], "fifo")
                self.assertEqual(flag_map(item)["--tbt-deadline-ms"], "100")
            self.assertEqual(G._without_chunk(argv["interactive"]), G._without_chunk(argv["high-prefill"]))

    def test_consistent_slai_or_cli_default_engine_is_refused_everywhere(self):
        for name, mutate in (
            ("slai-both", lambda overlay: overlay.__setitem__(8, "slai")),
            ("container-defaults", lambda overlay: overlay.__delitem__(slice(11, None))),
            ("gpu-memory-0.90", lambda overlay: overlay.__setitem__(overlay.index("--gpu-memory-utilization") + 1, "0.90")),
            ("mtp-vocab-default", lambda overlay: overlay.__setitem__(overlay.index("--mtp-vocab") + 1, "100000")),
        ):
            contract = P.contract_v2()
            for row in contract["profiles"].values():
                mutate(row["argv_overlay"])
                row["launch_argv_sha256"] = {
                    mode: S.profile_launch_template_sha256(EXE, row["argv_overlay"], mode) for mode in G.MODES}
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    S.validate_performance_profile_contract(copy.deepcopy(contract), S.PLAN_SCHEMA_V3, EXE)
                with self.assertRaises(ValueError):
                    Q._profile_contract_v2(copy.deepcopy(contract), EXE)
                document = G.reviewed_geometry_v2()
                for row in document["profiles"].values():
                    for mode in G.MODES:
                        argv = row["effective_argv"][mode]
                        start = argv.index("--max-prefill-tokens")
                        tail = argv[start:start + 26]
                        mutate(tail)
                        argv[start:start + 26] = tail
                with mock.patch.object(G, "REVIEWED_GEOMETRY_SHA256", G._document_sha256(document)), \
                        self.assertRaises(G.GeometryError):
                    G.validate_geometry_v2(document)


class LiveStartupProof(unittest.TestCase):
    def test_expected_lines_follow_the_engine_resolver(self):
        self.assertEqual(G.expected_prefill_config("interactive", "native-mtp-k5"),
                         {"ssm_prefill_chunk": 8192, "args.max_prefill_tokens": 4096,
                          "prefill_budget": 4096, "max_batch_tokens": 4097})
        self.assertEqual(G.expected_prefill_config("high-prefill", "none")["max_batch_tokens"], 16385)
        self.assertEqual(G.expected_kv_pool("interactive"), {"blocks": 16386, "block_size": 16, "max_tokens": 262176})
        census = G.startup_census(startup_lines(ARMS), ARMS)
        self.assertEqual([(row["profile"], row["mode"]) for row in census], ARMS)

    def test_missing_extra_reordered_and_drifted_lines_are_refused(self):
        cases = {
            "missing": (startup_lines(ARMS[:3]), ARMS),
            "extra": (startup_lines(ARMS + ARMS[:1]), ARMS),
            "reordered": (startup_lines(ARMS[2:] + ARMS[:2]), ARMS),
            "arena-override": (startup_lines(ARMS, drift=lambda p, m, c, k: (
                dict(c, max_batch_tokens=c["max_batch_tokens"] + 4096), k)), ARMS),
            "budget-clamped": (startup_lines(ARMS, drift=lambda p, m, c, k: (
                dict(c, prefill_budget=2048), k)), ARMS),
            "pool-without-spare": (startup_lines(ARMS, drift=lambda p, m, c, k: (
                c, dict(k, blocks=16385, max_tokens=16385 * 16))), ARMS),
            "malformed": (startup_lines(ARMS).replace(b"prefill_budget=", b"prefill_budget= "), ARMS),
            "not-bytes": (startup_lines(ARMS).decode(), ARMS),
        }
        for name, (log, arms) in cases.items():
            with self.subTest(name=name), self.assertRaises(G.GeometryError):
                G.startup_census(log, arms)


class K5TailProof(unittest.TestCase):
    def test_roles_chunks_and_tail_headroom(self):
        document = G.reviewed_geometry_v2()
        last = G.PROFILE_CONTEXTS[-1]
        interactive = G.k5_tail_proof(document, "interactive", last)
        high = G.k5_tail_proof(document, "high-prefill", last)
        self.assertEqual((interactive["chunks"], interactive["last_chunk_rows"]), (64, 4032))
        self.assertEqual((high["chunks"], high["last_chunk_rows"]), (16, 16320))
        for proof in (interactive, high):
            self.assertEqual(proof["capacity_headroom"], 0)
            self.assertEqual(proof["k5_rows_needed"], 262080 + 64 + 6)
            self.assertEqual(proof["addressable_rows"], 16385 * 16)
            self.assertEqual(proof["k5_tail_headroom_rows"], 10)
        small = G.k5_tail_proof(document, "high-prefill", 8192)
        self.assertEqual((small["chunks"], small["last_chunk_rows"]), (1, 8192))

    def test_pool_without_its_spare_block_cannot_hold_the_tail(self):
        document = G.reviewed_geometry_v2()
        with mock.patch.object(G, "expected_kv_pool",
                               return_value={"blocks": 16385, "block_size": 16, "max_tokens": 262160}):
            with self.assertRaisesRegex(G.GeometryError, "K5 tail"):
                G.k5_tail_proof(document, "interactive", 262_080)
            G.k5_tail_proof(document, "interactive", 131_072)


class HostBinding(unittest.TestCase):
    def host_argv(self, profile, mode):
        argv = G.reviewed_effective_argv(profile, mode)
        argv[0], argv[4] = "/opt/host/flux/target/release/flux-server-real", "/srv/models/Qwen3.8-Flash-Next"
        argv[6], argv[8], argv[10] = ALIAS, "127.0.0.1", "18080"
        return argv

    def test_host_argv_maps_to_the_container_template_only_by_substitution(self):
        for profile, mode in ARMS:
            template = G.normalize_host_argv(self.host_argv(profile, mode), profile, mode)
            self.assertEqual(G.launch_argv_sha256(template), P.sealed_recipe(profile)["launch_argv_sha256"][mode])
        for name, mutate in (
            ("public-bind", lambda a: a.__setitem__(8, "0.0.0.0")),
            ("gpu-memory", lambda a: a.__setitem__(a.index("--gpu-memory-utilization") + 1, "0.90")),
            ("no-qsa", lambda a: a.remove("--qwen4-qsa")),
            ("extra-flag", lambda a: a.extend(["--swap-space-gb", "0"])),
            ("relative-binary", lambda a: a.__setitem__(0, "flux-server-real")),
            ("other-profile-chunk", lambda a: a.__setitem__(a.index("--max-prefill-tokens") + 1, "16384")),
        ):
            argv = self.host_argv("interactive", "none")
            mutate(argv)
            with self.subTest(name=name), self.assertRaises(G.GeometryError):
                G.normalize_host_argv(argv, "interactive", "none")

    def test_environment_is_image_env_plus_recipe_plus_kill_switch(self):
        recipe = P.environment_recipe("interactive")
        plain = G.profile_environment("interactive", "none", recipe)
        self.assertEqual(plain["ATLAS_DISABLE_SPECULATION"], "1")
        self.assertEqual(plain["FLUX_MODE"], "interactive-w4096")
        self.assertEqual(plain["CUDA_MODULE_LOADING"], "LAZY")
        self.assertEqual(G.profile_environment("interactive", "native-mtp-k5", recipe)["ATLAS_DISABLE_SPECULATION"], "0")
        self.assertNotIn("ATLAS_MAX_BATCH_TOKENS", plain)
        drifted = dict(plain, ATLAS_MAX_BATCH_TOKENS="8192")
        self.assertNotEqual(G.engine_environment(drifted), G.engine_environment(plain))
        self.assertEqual(G.engine_environment(dict(plain, PATH="/x")), G.engine_environment(plain))

    def test_image_environment_is_the_dockerfile_env_block(self):
        text = (KIT / "Dockerfile").read_text()
        block = text[text.index("ENV DEBIAN_FRONTEND"):].split("\n\n", 1)[0]
        values = {}
        for line in block.replace("ENV ", "", 1).splitlines():
            key, value = line.strip().rstrip("\\").strip().split("=", 1)
            values[key] = value.strip('"')
        values["FLUX_ENGINE_VERSION"] = "1.0.2"
        values.pop("LD_LIBRARY_PATH")
        self.assertIn('ARG FLUX_ENGINE_VERSION="1.0.2"', text)
        self.assertEqual(values, G.IMAGE_ENVIRONMENT)


def fake_tokens(text):
    return [1000 + (hashlib.blake2b(w.encode(), digest_size=2).digest()[0] % 200) for w in text.split()] or [7]


class HostProducer(unittest.TestCase):
    """The producer runs the gate's own arm measurement on a fake transport."""

    def fake_proc(self, root, pid, argv, environment):
        directory = root / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
        (directory / "environ").write_bytes(b"".join(f"{k}={v}".encode() + b"\0" for k, v in environment.items()))
        (directory / "exe").write_bytes(b"\x7fELF offline fixture")
        (directory / "stat").write_text(f"{pid} (flux-server) S " + " ".join(["0"] * 18 + ["4242"] + ["0"] * 10))

    def transport(self, mode, counter):
        def blocking(_args, _method, route, body=None, **_kwargs):
            if route == "/metrics":
                return (b'atlas_spec_decode_verify_total{k="5",outcome="accepted"} 3\n'
                        if mode == "native-mtp-k5" else b"# none\n")
            if route == "/tokenize":
                return S.canonical_bytes({"tokens": fake_tokens(body["prompt"]), "count": 0})
            counter[0] += 1
            prompt, completion = len(body["prompt_token_ids"]), body["max_tokens"]
            return S.canonical_bytes({
                "id": f"{mode}-{counter[0]}",
                "choices": [{"index": 0, "text": PASSKEY_TEXT, "finish_reason": "length",
                             "token_ids": list(range(completion))}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                          "total_tokens": prompt + completion, "prompt_tokens_details": {"cached_tokens": 0},
                          "time_to_first_token_ms": 0.1}})

        def stream(_args, _route, body, **_kwargs):
            counter[0] += 1
            prompt, budget = len(body["prompt_token_ids"]), body["max_tokens"]
            pieces = PASSKEY_TEXT.split(" ")
            events = [(1.0 + index * 0.02, S.canonical_bytes(
                {"id": f"s{counter[0]}", "choices": [{"index": 0, "text": piece + (" " if index + 1 < len(pieces) else "")}]}))
                for index, piece in enumerate(pieces)]
            rate = (budget - 1) / (0.02 * (len([p for p in pieces if p]) - 1))
            events.append((1.0 + len(pieces) * 0.02, S.canonical_bytes(
                {"id": f"s{counter[0]}", "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": budget, "total_tokens": prompt + budget,
                           "prompt_tokens_details": {"cached_tokens": 0}, "response_token/s": rate}})))
            return events
        return blocking, stream

    def produce(self, temp, mutate_env=None):
        arms, counter = [], [0]
        proc = temp / "proc"
        proc.mkdir()
        for index, (profile, mode) in enumerate(ARMS):
            pid = 1000 + index
            launch = P.render(profile, mode, "/opt/host/flux-server-real", "/srv/models/q38", 18080, ALIAS,
                              {"PATH": "/usr/bin"})
            environment = mutate_env(launch["environment"]) if mutate_env else launch["environment"]
            self.fake_proc(proc, pid, launch["argv"], environment)
            log = temp / f"server-{index}.log"
            log.write_bytes(startup_lines([(profile, mode)]))
            blocking, stream = self.transport(mode, counter)
            with mock.patch.object(D, "request", side_effect=blocking), \
                    mock.patch.object(D, "stream_request", side_effect=stream), \
                    mock.patch.object(D, "PROFILE_CONTEXTS", (8192,)), \
                    mock.patch.object(P.os, "pidfd_open", side_effect=lambda _pid: OWN_PIDFD(os.getpid())):
                arms.append(P.measure(profile, mode, "http://127.0.0.1:18080", pid, log,
                                      temp / f"journal-{index}.jsonl", proc))
        return arms

    def test_four_arms_assemble_into_a_self_validated_host_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            arms = self.produce(temp)
            with mock.patch.object(D, "PROFILE_CONTEXTS", (8192,)):
                document, summary = P.assemble(json.loads(json.dumps(arms)), PROVENANCE)
                path = temp / "host-reference.json"
                path.write_bytes(S.canonical_bytes(document))
                P.self_validate(document, path)
            self.assertEqual(document["schema"], "flux-prefill-decode-host-reference/4")
            self.assertEqual(document["decode_budget_tokens"], 256)
            self.assertEqual(document["profile_chunk_tokens"], {"interactive": 4096, "high-prefill": 16384})
            for name in G.PROFILE_NAMES:
                self.assertEqual(document["profiles"][name]["launch_argv_sha256"],
                                 P.sealed_recipe(name)["launch_argv_sha256"])
                self.assertEqual(document["profiles"][name]["rows"][0]["decode_tokens"], 255)
            self.assertEqual([(row["profile"], row["context_tokens"]) for row in summary["rows"]],
                             [("interactive", 8192), ("high-prefill", 8192)])
            self.assertEqual(summary["concurrency"], 1)

    def test_drifted_host_environment_or_arm_census_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(P.ProducerError, "engine environment"):
                self.produce(Path(directory), lambda env: dict(env, ATLAS_MAX_BATCH_TOKENS="8192"))
        with tempfile.TemporaryDirectory() as directory:
            arms = self.produce(Path(directory))
        with mock.patch.object(D, "PROFILE_CONTEXTS", (8192,)):
            for name, mutate in (
                ("missing-arm", lambda a: a.pop()),
                ("duplicate-arm", lambda a: a.__setitem__(3, copy.deepcopy(a[2]))),
                ("other-binary", lambda a: a[1].__setitem__("binary_sha256", "f" * 64)),
                ("unsealed-argv", lambda a: a[0].__setitem__("launch_argv_sha256", "0" * 64)),
                ("no-k5-events", lambda a: a[1].__setitem__("k5_events", 0)),
                ("census-drift", lambda a: a[2]["startup_census"][0]["kv_pool"].__setitem__("blocks", 16385)),
            ):
                candidate = copy.deepcopy(arms)
                mutate(candidate)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    P.assemble(candidate, PROVENANCE)


if __name__ == "__main__":
    unittest.main()
