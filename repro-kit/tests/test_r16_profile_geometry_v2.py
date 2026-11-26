# SPDX-License-Identifier: AGPL-3.0-only
"""Offline checks for profile geometry /2 and the dual-recipe broker /2.

Synthetic argv/journal fixtures; recipe hashes are the real recipe bytes.
No process, container, model or device execution.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("r16_profile_geometry", ROOT / "repro-kit/r16_profile_geometry.py")
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


def recipe_sha(name):
    return hashlib.sha256((ROOT / G.REVIEWED_GEOMETRY[name]["recipe_relpath"]).read_bytes()).hexdigest()


ALIAS = "flux-profile-prefill-decode-profiles-" + "0123456789abcdef" * 2
PORT = "8888"
TRUSTED_MODEL = {"config_sha256": "c0" * 32}


def geometry():
    return copy.deepcopy(G.reviewed_geometry_v2())


def observation(document, profile, mode):
    row = document["profiles"][profile]
    argv = list(row["effective_argv"][mode])
    argv[G.MODEL_INDEX], argv[G.PORT_INDEX] = ALIAS, PORT
    return {
        "executable": dict(G.REVIEWED_EXECUTABLE),
        "model": {"path": G.MODEL_PATH, **TRUSTED_MODEL},
        "recipe_path": G.IN_CONTAINER_ROOT + row["recipe_relpath"],
        "recipe_sha256": row["recipe_sha256"],
        "argv": argv,
        "environment": {"FLUX_QWEN4_PREFILL_WIDE": str(row["wide"]),
                        "ATLAS_QWEN4_PREFILL_WIDE": str(row["wide"]),
                        "ATLAS_DISABLE_SPECULATION": "1" if mode == "none" else "0"},
    }


def pinned(document):
    """Trust exactly this document, as a re-reviewed pin would (tests only)."""
    return mock.patch.object(G, "REVIEWED_GEOMETRY_SHA256", G._document_sha256(document))


def admit(document, broker, live_request, observed, trusted=TRUSTED_MODEL):
    return G.admit_live_recipe(document, broker, live_request, observed, trusted_model_identity=trusted)


def request(profile, mode="none"):
    return {"schema": G.REQUEST_V2, "profile": profile, "mode": mode, "host_pid": 4242}


def journal(document, mutate=None):
    lines = []
    for profile in G.PROFILE_NAMES:
        for mode in G.MODES:
            for context in G.PROFILE_CONTEXTS:
                for kind in G.REQUEST_KINDS:
                    record = {"profile": profile, "mode": mode, "context_tokens": context, "kind": kind,
                              "geometry": G.bind_row(document, profile, mode, context, kind)}
                    if mutate:
                        record = mutate(record)
                    if record is not None:
                        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return b"".join(lines)


class ProfileGeometryV2(unittest.TestCase):
    def setUp(self):
        self.document = G.validate_geometry_v2(geometry())
        self.broker = G.broker_contract(self.document)

    def test_reviewed_geometry_matches_the_real_recipes(self):
        for name, reviewed in G.REVIEWED_GEOMETRY.items():
            environment = G.recipe_environment((ROOT / reviewed["recipe_relpath"]).read_bytes())
            self.assertEqual(environment["FLUX_MODE"], reviewed["flux_mode"])
            for selector in G.WIDE_SELECTORS:
                self.assertEqual(environment[selector], str(reviewed["wide"]))
            self.assertNotIn(G.MAX_BATCH_TOKENS, environment)
        self.assertEqual(G.CENSUS_RECORDS, 72)
        self.assertNotEqual(recipe_sha("interactive"), recipe_sha("high-prefill"))

    def test_dual_recipe_broker_admits_each_reviewed_recipe(self):
        for profile in G.PROFILE_NAMES:
            for mode in G.MODES:
                admitted = admit(self.document, self.broker, request(profile, mode),
                                               observation(self.document, profile, mode))
                self.assertEqual(admitted, {"profile": profile, "mode": mode, "recipe_sha256": recipe_sha(profile),
                                            "model_alias": ALIAS, "port": 8888})
        self.assertEqual(set(self.broker["recipes"]), set(G.PROFILE_NAMES))

    def test_single_recipe_broker_is_refused(self):
        single = {"schema": G.BROKER_V1, "recipe_sha256": recipe_sha("interactive")}
        with self.assertRaisesRegex(G.GeometryError, "single-recipe"):
            admit(self.document, single, request("interactive"), observation(self.document, "interactive", "none"))
        narrowed = copy.deepcopy(self.broker)
        del narrowed["recipes"]["high-prefill"]
        with self.assertRaises(G.GeometryError):
            admit(self.document, narrowed, request("interactive"), observation(self.document, "interactive", "none"))

    def test_wrong_and_third_recipes_are_refused(self):
        swapped = observation(self.document, "interactive", "none")
        with self.assertRaisesRegex(G.GeometryError, "another profile"):
            admit(self.document, self.broker, request("high-prefill"), swapped)
        third = dict(swapped, recipe_sha256="ab" * 32)
        with self.assertRaisesRegex(G.GeometryError, "two reviewed recipes"):
            admit(self.document, self.broker, request("interactive"), third)
        extra = copy.deepcopy(self.broker)
        extra["recipes"]["third"] = dict(extra["recipes"]["interactive"], recipe_sha256="cd" * 32)
        with self.assertRaisesRegex(G.GeometryError, "reviewed pair"):
            admit(self.document, extra, request("interactive"), swapped)

    def test_live_argv_wide_and_arena_must_match_the_binding(self):
        for change in (
            lambda o: o["argv"].__setitem__(o["argv"].index("--max-prefill-tokens") + 1, "16384"),
            lambda o: o["environment"].__setitem__("FLUX_QWEN4_PREFILL_WIDE", "16384"),
            lambda o: o["environment"].__setitem__("ATLAS_MAX_BATCH_TOKENS", "65536"),
            lambda o: o["argv"].append("--mtp-vocab"),
        ):
            live = observation(self.document, "interactive", "none")
            change(live)
            with self.assertRaises(G.GeometryError):
                admit(self.document, self.broker, request("interactive"), live)

    def test_4096_and_16384_rows_never_satisfy_each_other_and_relabels_are_refused(self):
        for profile, other in (("interactive", "high-prefill"), ("high-prefill", "interactive")):
            row = G.bind_row(self.document, profile, "none", 16_384, "exact")
            G.require_row(self.document, row, profile)
            with self.assertRaises(G.GeometryError):
                G.require_row(self.document, row, other)
            relabelled = dict(row, profile=other)
            with self.assertRaises(G.GeometryError):
                G.require_row(self.document, relabelled, other)
            for key, value in (("chunk_tokens", G.REVIEWED_GEOMETRY[other]["chunk_tokens"]),
                               ("recipe_sha256", recipe_sha(other))):
                with self.assertRaises(G.GeometryError):
                    G.require_row(self.document, dict(row, **{key: value}), profile)

    def test_v1_is_readable_but_refused_for_dual_profiles(self):
        v1 = {"schema": G.GEOMETRY_V1, "chunk_tokens": 4096}
        self.assertEqual(G.read_geometry(v1), v1)
        self.assertEqual(G.chunk_tokens(v1, "high-prefill"), 4096)
        with self.assertRaisesRegex(G.GeometryError, "universal 4096"):
            G.require_dual_profile(v1)
        with self.assertRaises(G.GeometryError):
            G.broker_contract(v1)
        self.assertEqual(G.chunk_tokens(self.document, "interactive"), 4096)
        self.assertEqual(G.chunk_tokens(self.document, "high-prefill"), 16384)

    def test_v1_v2_cross_pairs_are_refused(self):
        for version, pairing in G.VERSION_PAIRINGS.items():
            self.assertEqual(G.check_version_pairing(dict(pairing, geometry=version)), version)
        v1, v2 = G.VERSION_PAIRINGS[G.GEOMETRY_V1], G.VERSION_PAIRINGS[G.GEOMETRY_V2]
        for key in v1:
            with self.subTest(key=key), self.assertRaisesRegex(G.GeometryError, "cross pair"):
                G.check_version_pairing(dict(v2, geometry=G.GEOMETRY_V2, **{key: v1[key]}))
            with self.assertRaisesRegex(G.GeometryError, "cross pair"):
                G.check_version_pairing(dict(v1, geometry=G.GEOMETRY_V1, **{key: v2[key]}))
        with self.assertRaises(G.GeometryError):
            G.check_version_pairing(dict(v2, geometry=G.GEOMETRY_V1))

    def test_raw_72_record_census_is_preserved(self):
        roles = G.recensus(self.document, journal(self.document))
        self.assertEqual(len(roles), 72)
        lines = journal(self.document).splitlines(keepends=True)
        for mutant in (
            b"".join(lines[:-1]),
            b"".join(lines + lines[:1]),
            b"".join(lines[:-1] + lines[:1]),
            journal(self.document, lambda r: dict(r, profile="high-prefill") if r["profile"] == "interactive" and r["context_tokens"] == 8192 and r["kind"] == "exact" and r["mode"] == "none" else r),
            journal(self.document, lambda r: dict(r, geometry=dict(r["geometry"], chunk_tokens=4096)) if r["profile"] == "high-prefill" and r["context_tokens"] == 65536 and r["kind"] == "ttft" and r["mode"] == "none" else r),
            journal(self.document, lambda r: dict(r, kind="stream") if r["kind"] == "ttft" and r["profile"] == "interactive" and r["context_tokens"] == 8192 and r["mode"] == "none" else r),
            lines[0].replace(b"\n", b" \n") + b"".join(lines[1:]),
        ):
            with self.assertRaises(G.GeometryError):
                G.recensus(self.document, mutant)

    def test_role_geometry_binds_chunk_count_partial_last_chunk_and_capacity(self):
        self.assertEqual(G.role_geometry(self.document, "interactive", 262_080),
                         {"chunk_tokens": 4096, "chunks": 64, "last_chunk_rows": 4032, "capacity_headroom": 0})
        self.assertEqual(G.role_geometry(self.document, "high-prefill", 262_080),
                         {"chunk_tokens": 16384, "chunks": 16, "last_chunk_rows": 16320, "capacity_headroom": 0})
        self.assertEqual(G.role_geometry(self.document, "high-prefill", 8_192)["chunks"], 1)
        with self.assertRaises(G.GeometryError):
            G.role_geometry(self.document, "interactive", 4096)

    def test_geometry_refuses_unpinned_base_argv_and_policy(self):
        for name, mutate in (
            ("policy", lambda d: d["profiles"]["interactive"]["effective_argv"]["none"].__setitem__(
                slice(d["profiles"]["interactive"]["effective_argv"]["none"].index("--scheduling-policy"),
                      d["profiles"]["interactive"]["effective_argv"]["none"].index("--scheduling-policy") + 2),
                [])),
            ("qsa", lambda d: d["profiles"]["high-prefill"]["effective_argv"]["none"].remove("--qwen4-qsa")),
            ("block", lambda d: d["profiles"]["interactive"]["effective_argv"]["none"].__setitem__(
                d["profiles"]["interactive"]["effective_argv"]["none"].index("--block-size") + 1, "32")),
            ("base", lambda d: d["profiles"]["high-prefill"]["effective_argv"]["native-mtp-k5"].__setitem__(
                d["profiles"]["high-prefill"]["effective_argv"]["native-mtp-k5"].index("--max-num-seqs") + 1, "2")),
            ("policy-split", lambda d: d["profiles"]["high-prefill"]["effective_argv"]["none"].__setitem__(
                d["profiles"]["high-prefill"]["effective_argv"]["none"].index("--scheduling-policy") + 1, "slai")),
            ("chunk", lambda d: d["profiles"]["high-prefill"].__setitem__("chunk_tokens", 4096)),
            ("same-recipe", lambda d: d["profiles"]["high-prefill"].__setitem__(
                "recipe_sha256", d["profiles"]["interactive"]["recipe_sha256"])),
        ):
            document = geometry()
            mutate(document)
            with self.subTest(name=name), self.assertRaises(G.GeometryError):
                G.validate_geometry_v2(document)

    def test_host_reference_keeps_timer_and_host_identity(self):
        header = {"schema": "flux-prefill-decode-host-reference/4", "decode_clock": G.DECODE_CLOCK,
                  "prefill_clock": G.PREFILL_CLOCK, "geometry_sha256": G.geometry_sha256(self.document),
                  **{key: "x" for key in G.HOST_IDENTITY_KEYS}}
        G.validate_host_reference_header(self.document, header)
        for change in ({"schema": "flux-prefill-decode-host-reference/3"}, {"decode_clock": "sse-arrival/v1"},
                       {"source_tree": ""}, {"geometry_sha256": "0" * 64}):
            with self.assertRaises(G.GeometryError):
                G.validate_host_reference_header(self.document, dict(header, **change))


def resealed(document):
    """Re-derive every hash the attacker controls after a consistent edit."""
    return G.validate_geometry_v2(document), G.broker_contract(document)


class AdversarialFalsePasses(unittest.TestCase):
    """Each test reproduces one false pass of 1053a6e6."""

    # Finding 1: mode semantics.
    def test_f1_k5_argv_rebound_to_plain_and_resealed_is_refused(self):
        document = geometry()
        for row in document["profiles"].values():
            row["effective_argv"]["native-mtp-k5"] = list(row["effective_argv"]["none"])
        with self.assertRaisesRegex(G.GeometryError, "selects another mode"):
            resealed(document)

    def test_f1_speculative_argv_declared_plain_is_refused(self):
        document = geometry()
        for row in document["profiles"].values():
            row["effective_argv"]["none"] = list(row["effective_argv"]["native-mtp-k5"])
        with self.assertRaisesRegex(G.GeometryError, "selects another mode"):
            resealed(document)

    def test_f1_live_mode_must_match_argv_and_kill_switch(self):
        document = G.validate_geometry_v2(geometry())
        broker = G.broker_contract(document)
        k5 = observation(document, "interactive", "native-mtp-k5")
        with self.assertRaisesRegex(G.GeometryError, "another mode"):
            admit(document, broker, request("interactive", "none"), k5)
        plain = observation(document, "interactive", "none")
        del plain["environment"]["ATLAS_DISABLE_SPECULATION"]
        with self.assertRaisesRegex(G.GeometryError, "kill switch"):
            admit(document, broker, request("interactive", "none"), plain)
        k5_alias = observation(document, "interactive", "native-mtp-k5")
        del k5_alias["environment"]["ATLAS_DISABLE_SPECULATION"]
        k5_alias["environment"]["FLUX_DISABLE_SPECULATION"] = "1"
        with self.assertRaisesRegex(G.GeometryError, "kill switch"):
            admit(document, broker, request("interactive", "native-mtp-k5"), k5_alias)
        for argv in (["x", "--speculative"], ["x", "--speculative", "--num-drafts", "5"],
                     ["x", "--dflash"], ["x", "--speculative", "--num-drafts", "4", "--mtp-gate", "auto"]):
            with self.subTest(argv=argv), self.assertRaises(G.GeometryError):
                G.argv_mode(argv)
        self.assertEqual(G.argv_mode(["x", "--speculative", "--num-drafts", "4"]), "native-mtp-k5")

    def test_f1_mode_relabelled_journal_record_is_refused(self):
        document = G.validate_geometry_v2(geometry())
        def relabel(record):
            if record["mode"] == "native-mtp-k5" and record["kind"] == "exact" and record["context_tokens"] == 8192:
                record = dict(record, mode="none", geometry=dict(record["geometry"], mode="none"))
            return record
        with self.assertRaises(G.GeometryError):
            G.recensus(document, journal(document, relabel))

    # Finding 2: duplicate pinned flags.
    def test_f2_trailing_or_repeated_pinned_flag_is_refused(self):
        # The exact 1053 false pass: a trailing bare duplicate in every argv.
        document = geometry()
        for row in document["profiles"].values():
            for mode in G.MODES:
                row["effective_argv"][mode] += ["--max-prefill-tokens"]
        with self.assertRaisesRegex(G.GeometryError, "repeats an option"):
            resealed(document)
        for suffix in (["--max-prefill-tokens"], ["--max-prefill-tokens", "4096"],
                       ["--max-prefill-tokens", "8192"], ["--max-prefill-tokens=4096"],
                       ["--block-size=16"], ["--scheduling-policy", "fifo"]):
            document = geometry()
            document["profiles"]["interactive"]["effective_argv"]["none"] += suffix
            with self.subTest(suffix=suffix), self.assertRaises(G.GeometryError):
                resealed(document)
        document = G.validate_geometry_v2(geometry())
        live = observation(document, "interactive", "none")
        live["argv"] = live["argv"] + ["--max-prefill-tokens"]
        with self.assertRaisesRegex(G.GeometryError, "repeats an option"):
            admit(document, G.broker_contract(document), request("interactive"), live)

    # Finding 3: per-profile recipe binding.
    def test_f3_consistent_recipe_swap_is_refused(self):
        document = geometry()
        first, second = (document["profiles"][name] for name in G.PROFILE_NAMES)
        first["recipe_sha256"], second["recipe_sha256"] = second["recipe_sha256"], first["recipe_sha256"]
        with self.assertRaisesRegex(G.GeometryError, "own reviewed recipe"):
            resealed(document)
        document = geometry()
        document["profiles"]["interactive"]["recipe_sha256"] = "ab" * 32
        with self.assertRaisesRegex(G.GeometryError, "own reviewed recipe"):
            resealed(document)

    def test_f3_recipe_bytes_must_be_the_profiles_own_with_matching_wide(self):
        for name in G.PROFILE_NAMES:
            payload = (ROOT / G.REVIEWED_GEOMETRY[name]["recipe_relpath"]).read_bytes()
            self.assertEqual(G.verify_recipe_bytes(name, payload)["FLUX_QWEN4_PREFILL_WIDE"],
                             str(G.REVIEWED_GEOMETRY[name]["wide"]))
            other = [n for n in G.PROFILE_NAMES if n != name][0]
            with self.assertRaisesRegex(G.GeometryError, "not its reviewed recipe"):
                G.verify_recipe_bytes(other, payload)
        self.assertEqual(G.REVIEWED_RECIPE_SHA256, {n: recipe_sha(n) for n in G.PROFILE_NAMES})

    # Finding 4: strict integers for arena and context.
    def test_f4_arena_bound_rejects_negative_float_bool_and_tiny(self):
        for value in (-1, 0, 4095, 1.5, 65536.0, True, "65536", 1 << 40):
            document = geometry()
            document["profiles"]["interactive"]["max_batch_tokens"] = value
            with self.subTest(value=value), self.assertRaises(G.GeometryError):
                resealed(document)
        document = geometry()
        document["profiles"]["interactive"]["max_batch_tokens"] = 65536
        with pinned(document):
            resealed(document)
        # Without a re-reviewed pin the bound arena is not the trusted geometry.
        with self.assertRaisesRegex(G.GeometryError, "trusted reviewed geometry"):
            resealed(document)

    def test_f4_live_arena_is_strict_decimal_and_null_means_absent(self):
        document = G.validate_geometry_v2(geometry())
        broker = G.broker_contract(document)
        for value in ("", "-1", "+65536", "65536.0", "6.5e4", " 65536", "065536", "65536"):
            live = observation(document, "interactive", "none")
            live["environment"]["ATLAS_MAX_BATCH_TOKENS"] = value
            with self.subTest(value=value), self.assertRaises(G.GeometryError):
                admit(document, broker, request("interactive"), live)
        bound = geometry()
        bound["profiles"]["interactive"]["max_batch_tokens"] = 65536
        with pinned(bound):
            bound, bound_broker = resealed(bound)
            for value, ok in (("65536", True), ("+65536", False), ("65536.0", False), (None, False)):
                live = observation(bound, "interactive", "none")
                if value is not None:
                    live["environment"]["ATLAS_MAX_BATCH_TOKENS"] = value
                if ok:
                    admit(bound, bound_broker, request("interactive"), live)
                else:
                    with self.subTest(value=value), self.assertRaises(G.GeometryError):
                        admit(bound, bound_broker, request("interactive"), live)

    def test_f4_float_and_bool_contexts_and_pids_are_refused(self):
        document = G.validate_geometry_v2(geometry())
        for context in (8192.0, 262080.0, True):
            with self.subTest(context=context), self.assertRaises(G.GeometryError):
                G.role_geometry(document, "interactive", context)
            with self.assertRaises(G.GeometryError):
                G.bind_row(document, "interactive", "none", context, "exact")
        row = G.bind_row(document, "interactive", "none", 8192, "exact")
        with self.assertRaises(G.GeometryError):
            G.require_row(document, dict(row, context_tokens=8192.0), "interactive")
        for pid in (0, -1, True, 1.0):
            live = observation(document, "interactive", "none")
            with self.subTest(pid=pid), self.assertRaises(G.GeometryError):
                admit(document, G.broker_contract(document),
                                    dict(request("interactive"), host_pid=pid), live)
        for text in ("4096.0", "4e3", "+4096", "-4096", " 4096", "4096 ", "", "04096", "\u0664\u0660\u0669\u0666"):
            with self.subTest(text=text), self.assertRaises(G.GeometryError):
                G.strict_decimal(text, "fixture")
        self.assertEqual(G.strict_decimal("4096", "fixture"), 4096)


def every_argv(document, change):
    """Apply one change to every argv, as a consistent attacker would."""
    for row in document["profiles"].values():
        for mode in G.MODES:
            change(row["effective_argv"][mode])
    return document


class ReleaseBindingHold(unittest.TestCase):
    """A consistently resealed pair must still be refused."""

    def setUp(self):
        self.document = G.validate_geometry_v2(geometry())
        self.broker = G.broker_contract(self.document)

    def test_trusted_geometry_pin_is_the_reviewed_document(self):
        self.assertEqual(G._document_sha256(G.reviewed_geometry_v2()), G.REVIEWED_GEOMETRY_SHA256)
        self.assertEqual(G.geometry_sha256(self.document), G.REVIEWED_GEOMETRY_SHA256)
        self.assertEqual(self.broker["executable"], G.REVIEWED_EXECUTABLE)
        self.assertEqual(G.REVIEWED_EXECUTABLE["size"], 92_082_232)
        self.assertTrue(G.REVIEWED_EXECUTABLE["sha256"].startswith("3460c766"))

    def test_resealed_arbitrary_executable_is_refused(self):
        document = every_argv(geometry(), lambda argv: argv.__setitem__(0, "/opt/flux/bin/other"))
        with self.assertRaisesRegex(G.GeometryError, "not the reviewed release binary"):
            resealed(document)
        for change in ({"sha256": "ab" * 32}, {"size": 92_082_233}, {"path": "/opt/flux/bin/other"},
                       {"size": 92_082_232.0}):
            live = observation(self.document, "interactive", "none")
            live["executable"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(G.GeometryError, "release binary"):
                admit(self.document, self.broker, request("interactive"), live)

    def test_model_swap_and_missing_trusted_identity_are_refused(self):
        live = observation(self.document, "interactive", "none")
        live["model"]["config_sha256"] = "d1" * 32
        with self.assertRaisesRegex(G.GeometryError, "trusted checkpoint"):
            admit(self.document, self.broker, request("interactive"), live)
        live = observation(self.document, "interactive", "none")
        for trusted in (None, {}, {"config_sha256": "0" * 64}, {"dev": 1}, {"dev": 1.0, "ino": 2}):
            with self.subTest(trusted=trusted), self.assertRaisesRegex(G.GeometryError, "model"):
                admit(self.document, self.broker, request("interactive"), live, trusted)
        with self.assertRaises(TypeError):
            G.admit_live_recipe(self.document, self.broker, request("interactive"), live)
        by_inode = {"dev": 7, "ino": 11}
        live["model"] = {"path": G.MODEL_PATH, "dev": 7, "ino": 11}
        admit(self.document, self.broker, request("interactive"), live, by_inode)
        live["model"]["ino"] = 12
        with self.assertRaisesRegex(G.GeometryError, "trusted checkpoint"):
            admit(self.document, self.broker, request("interactive"), live, by_inode)
        live["model"] = {"path": "/models-other", **TRUSTED_MODEL}
        with self.assertRaisesRegex(G.GeometryError, "trusted checkpoint"):
            admit(self.document, self.broker, request("interactive"), live)

    def test_resealed_missing_serve_or_model_is_refused(self):
        for change, pattern in (
            (lambda argv: argv.__setitem__(1, "bench"), "'serve'"),
            (lambda argv: argv.__setitem__(slice(3, 5), []), "pinned /models"),
            (lambda argv: argv.__setitem__(4, "/elsewhere"), "pinned /models"),
            (lambda argv: argv.__setitem__(3, "--model"), "pinned /models"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(G.GeometryError, pattern):
                resealed(every_argv(geometry(), change))
            live = observation(self.document, "interactive", "none")
            change(live["argv"])
            with self.assertRaises(G.GeometryError):
                admit(self.document, self.broker, request("interactive"), live)

    def test_resealed_non_bf16_kv_is_refused(self):
        for dtype in ("nvfp4", "fp8", "fp8_e4m3", "turboquant"):
            index = geometry()["profiles"]["interactive"]["effective_argv"]["none"].index("--kv-cache-dtype") + 1
            document = every_argv(geometry(), lambda argv: argv.__setitem__(index, dtype))
            with self.subTest(dtype=dtype), self.assertRaisesRegex(G.GeometryError, "KV cache dtype"):
                resealed(document)
            live = observation(self.document, "interactive", "none")
            live["argv"][index] = dtype
            with self.assertRaisesRegex(G.GeometryError, "KV cache dtype"):
                admit(self.document, self.broker, request("interactive"), live)

    def test_consistent_geometry_not_equal_to_the_trusted_pin_is_refused(self):
        # A self-consistent deadline change in every argv: refused by the one
        # pinned scheduling policy before the pin is even consulted.
        index = geometry()["profiles"]["interactive"]["effective_argv"]["none"].index("--tbt-deadline-ms") + 1
        document = every_argv(geometry(), lambda argv: argv.__setitem__(index, "200"))
        with self.assertRaisesRegex(G.GeometryError, "one reviewed policy"):
            resealed(document)
        # Every structural rule satisfied, but not the trusted document.
        document = geometry()
        document["profiles"]["high-prefill"]["max_batch_tokens"] = 16_385
        with self.assertRaisesRegex(G.GeometryError, "trusted reviewed geometry"):
            resealed(document)
        with self.assertRaisesRegex(G.GeometryError, "trusted reviewed geometry"):
            G.broker_contract(document)
        with self.assertRaisesRegex(G.GeometryError, "trusted reviewed geometry"):
            G.bind_row(document, "high-prefill", "none", 8192, "exact")

    def test_arena_minimum_is_chunk_plus_slots_and_k5_rows(self):
        for name, reviewed in G.REVIEWED_GEOMETRY.items():
            chunk = reviewed["chunk_tokens"]
            self.assertEqual(G.arena_bounds(chunk, 262_144), (chunk + 1, 262_144))
            for arena, ok in ((chunk, False), (chunk + 1, True), (262_144, True), (262_145, False)):
                document = geometry()
                document["profiles"][name]["max_batch_tokens"] = arena
                with pinned(document):
                    if ok:
                        bound, broker = resealed(document)
                        live = observation(bound, name, "native-mtp-k5")
                        live["environment"]["ATLAS_MAX_BATCH_TOKENS"] = str(arena)
                        admit(bound, broker, request(name, "native-mtp-k5"), live)
                    else:
                        with self.subTest(name=name, arena=arena), self.assertRaises(G.GeometryError):
                            resealed(document)
        # kv_cache.rs speculative_rows_remain_part_of_the_override_minimum:
        # chunk 1, seq 1, batch 1, K5 drafts 4 resolves to exactly 6.
        self.assertEqual(G.arena_bounds(1, 1), (6, 6))
        self.assertEqual(G.arena_bounds(1, 1, num_drafts=0), (2, 2))
        self.assertEqual(G.arena_bounds(4096, 262_144, max_batch_size=3), (4099, 786_432))
        self.assertEqual(G.padded_decode_rows(3), 4)


if __name__ == "__main__":
    unittest.main()
