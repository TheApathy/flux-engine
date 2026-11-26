#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Host producer for profile host reference /4 (sustained 256-token rows).

The host measures with the gate's own workload and clocks: prompts, request
bodies, `measure_arm`, `build_row` and `load_host_reference` are imported from
r15-gates/prefill-decode-profiles, so a host row and a container row are the
same quantity by construction. The sealed profile recipe pins the argv and
environment; only the executable path, model path, alias, bind and port may
differ on the host. Nothing here builds, seals or qualifies a release.

  render    print the host argv and environment of one (profile, mode) arm
  launch    exec that server (stdout/stderr to the operator's log file)
  measure   bind the live host process, then run the gate's arm measurement
  assemble  build and self-validate host reference /4 from the four arms
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ARM_SCHEMA = "flux-profile-host-arm/1"
SUMMARY_SCHEMA = "flux-profile-host-summary/1"
# Transport keys a host may keep from the operator's own environment.
HOST_TRANSPORT_KEYS = ("PATH", "HOME", "LD_LIBRARY_PATH", "LANG", "TERM")


def _module(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


G = _module(HERE / "r16_profile_geometry.py", "r16_profile_geometry")
S = _module(HERE / "r15-container-suite.py", "r16_host_producer_suite")
D = _module(HERE / "r15-gates" / "prefill-decode-profiles", "r16_host_producer_gate")


class ProducerError(ValueError):
    """A host arm or host reference cannot be bound to the sealed recipe."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProducerError(message)


def source_root() -> Path:
    return HERE.parent


def sealed_recipe(profile: str) -> dict:
    """The checked-in sealed recipe, which must equal its source derivation."""
    raw = (source_root() / G.SEALED_RECIPE_RELPATHS[profile]).read_bytes()
    require(raw == G.sealed_recipe_bytes(profile), f"{profile} sealed recipe differs from its source derivation")
    return json.loads(raw)


def environment_recipe(profile: str) -> bytes:
    return (source_root() / G.REVIEWED_GEOMETRY[profile]["recipe_relpath"]).read_bytes()


def render(profile: str, mode: str, binary: str, model_path: str, port: int, alias: str,
           transport: dict[str, str]) -> dict:
    argv = G.reviewed_effective_argv(profile, mode)
    argv[G.EXECUTABLE_INDEX], argv[G.MODEL_PATH_INDEX] = binary, model_path
    argv[G.MODEL_INDEX], argv[G.BIND_INDEX], argv[G.PORT_INDEX] = alias, G.HOST_BIND, str(port)
    template = G.normalize_host_argv(argv, profile, mode)
    environment = G.profile_environment(profile, mode, environment_recipe(profile))
    require(not set(transport) - set(HOST_TRANSPORT_KEYS), "host transport environment has engine keys")
    return {"profile": profile, "mode": mode, "argv": argv,
            "environment": {**transport, **environment},
            "launch_argv_sha256": G.launch_argv_sha256(template)}


def observe_process(pid: int, proc: Path = Path("/proc")) -> dict:
    """Read one live host server's cmdline, environment and executable."""
    root = proc / str(pid)
    cmdline = (root / "cmdline").read_bytes()
    require(cmdline.endswith(b"\0"), "host cmdline is truncated")
    environ = (root / "environ").read_bytes()
    environment = {}
    for item in environ.split(b"\0"):
        if not item:
            continue
        key, separator, value = item.partition(b"=")
        require(bool(separator) and key.decode() not in environment, "host environment is ambiguous")
        environment[key.decode()] = value.decode()
    digest, size = hashlib.sha256(), 0
    with open(root / "exe", "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    start = (root / "stat").read_text().rsplit(")", 1)[1].split()[19]
    return {"argv": [item.decode() for item in cmdline[:-1].split(b"\0")], "environment": environment,
            "executable_sha256": digest.hexdigest(), "executable_size": size, "start_ticks": int(start)}


def bind_process(observed: dict, profile: str, mode: str) -> dict:
    template = G.normalize_host_argv(observed["argv"], profile, mode)
    expected = G.profile_environment(profile, mode, environment_recipe(profile))
    require(G.engine_environment(observed["environment"]) == G.engine_environment(expected),
            "host engine environment differs from the sealed recipe environment")
    return {"launch_argv_sha256": G.launch_argv_sha256(template), "template": template}


def measure(profile: str, mode: str, base_url: str, pid: int, server_log: Path, journal_path: Path,
            proc: Path = Path("/proc")) -> dict:
    """One arm on a fresh host server: the gate's measure_arm, then the log census."""
    before = observe_process(pid, proc)
    binding = bind_process(before, profile, mode)
    args = types.SimpleNamespace(suite=S, base_url=base_url)
    model = before["argv"][G.MODEL_INDEX]
    components = D.tokenize_components(args, model)
    prompts = {}
    for context in D.PROFILE_CONTEXTS:
        ids, positions = D.assemble_prompt(context, components)
        prompts[context] = {"ids": ids, "sha256": D.prompt_sha256(ids), "positions": positions}
    pidfd = os.pidfd_open(pid)
    journal = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    journal_hash = hashlib.sha256()
    try:
        recipe = G.REVIEWED_RECIPE_SHA256[profile]
        arm, metrics = D.measure_arm(args, profile, mode, model, pidfd, prompts, journal, journal_hash, set(),
                                     {"chunk_tokens": G.REVIEWED_GEOMETRY[profile]["chunk_tokens"],
                                      "recipe_sha256": recipe})
        os.fsync(journal)
    finally:
        os.close(journal)
        os.close(pidfd)
    after = observe_process(pid, proc)
    require(after == before, "host server identity changed during the arm")
    census = G.startup_census(server_log.read_bytes(), [(profile, mode)])
    return {
        "schema": ARM_SCHEMA, "profile": profile, "mode": mode,
        "binary_sha256": before["executable_sha256"], "binary_size": before["executable_size"],
        "launch_argv_sha256": binding["launch_argv_sha256"], "observed_argv": before["argv"],
        "engine_environment": G.engine_environment(before["environment"]),
        "startup_census": census, "journal_sha256": journal_hash.hexdigest(),
        "prompt_sha256": {str(context): prompts[context]["sha256"] for context in D.PROFILE_CONTEXTS},
        "k5_events": metrics["k5_events"],
        "arm": {str(context): value for context, value in arm.items()},
    }


def _arm_table(arms: list[dict]) -> dict:
    table = {}
    for arm in arms:
        require(type(arm) is dict and arm.get("schema") == ARM_SCHEMA, "host arm schema differs")
        key = (arm["profile"], arm["mode"])
        require(key[0] in G.PROFILE_NAMES and key[1] in G.MODES and key not in table, "host arm census differs")
        recipe = sealed_recipe(key[0])
        require(arm["launch_argv_sha256"] == recipe["launch_argv_sha256"][key[1]],
                "host arm did not run the sealed launch argv")
        require(arm["startup_census"] == [{"profile": key[0], "mode": key[1],
                                           "prefill_config": G.expected_prefill_config(*key),
                                           "kv_pool": G.expected_kv_pool(key[0])}],
                "host arm startup census differs from the sealed live proof")
        require((arm["k5_events"] > 0) == (key[1] == "native-mtp-k5"), "host K5 verifier engagement differs")
        table[key] = arm
    require(set(table) == {(p, m) for p in G.PROFILE_NAMES for m in G.MODES}, "host arm census differs")
    require(len({arm["binary_sha256"] for arm in arms}) == 1, "host arms ran different binaries")
    return table


def assemble(arms: list[dict], provenance: dict) -> tuple[dict, dict]:
    """Host reference /4 plus a per-context throughput summary."""
    table = _arm_table(arms)
    require(type(provenance) is dict and set(provenance) == set(D.PROVENANCE_KEYS), "host provenance keys differ")
    profiles, summary = {}, []
    for profile in G.PROFILE_NAMES:
        plain, native = table[(profile, "none")], table[(profile, "native-mtp-k5")]
        require(plain["prompt_sha256"] == native["prompt_sha256"], "tokenized prompt differs between host arms")
        rows = []
        for context in D.PROFILE_CONTEXTS:
            p, n = plain["arm"][str(context)], native["arm"][str(context)]
            prefill = n["ttft"]
            row = {
                "context_tokens": context,
                "decode_tokens": D.decode_tokens(context),
                "k5_decode_wall_seconds": n["stream"]["decode_wall_seconds"],
                "plain_decode_wall_seconds": p["stream"]["decode_wall_seconds"],
                "prefill_wall_seconds": prefill["prefill_ttft_seconds"],
                "prompt_sha256": native["prompt_sha256"][str(context)],
                "k5_usage": n["stream"]["terminal_usage"],
                "plain_usage": p["stream"]["terminal_usage"],
                "prefill_client_monotonic": prefill["client_monotonic"],
            }
            # The gate's own identity, passkey and stream checks, against itself.
            _raw, evidence, _bands = D.build_row(profile, context, p, n, row)
            rows.append(row)
            summary.append({"profile": profile, "context_tokens": context,
                            "decode_tokens": row["decode_tokens"],
                            "prefill_tokens_per_second": evidence["prefill_tokens_per_second"],
                            "k5_decode_tokens_per_second": evidence["k5_decode_tokens_per_second"],
                            "plain_decode_tokens_per_second": evidence["plain_decode_tokens_per_second"],
                            "server_ttft_ms": prefill["server_time_to_first_token_ms"]})
        recipe = sealed_recipe(profile)
        profiles[profile] = {"argv_overlay": recipe["argv_overlay"],
                             "host_recipe_sha256": recipe["environment_recipe"]["sha256"],
                             "launch_argv_sha256": recipe["launch_argv_sha256"], "rows": rows}
    document = {
        "schema": G.VERSION_PAIRINGS[G.GEOMETRY_V2]["host_reference"],
        "profile_chunk_tokens": {name: G.REVIEWED_GEOMETRY[name]["chunk_tokens"] for name in G.PROFILE_NAMES},
        "binary_sha256": arms[0]["binary_sha256"],
        "decode_budget_tokens": D.DECODE_BUDGET,
        "measurement_contract": dict(D.MEASUREMENT_CONTRACT),
        "provenance": dict(provenance),
        "profiles": profiles,
    }
    return document, {"schema": SUMMARY_SCHEMA, "binary_sha256": document["binary_sha256"],
                      "concurrency": 1, "rows": summary}


def contract_v2() -> dict:
    """Profile contract /2 exactly as plan /3 must carry it for these recipes."""
    profiles = {}
    for name in G.PROFILE_NAMES:
        recipe = sealed_recipe(name)
        profiles[name] = {
            "recipe_relpath": recipe["environment_recipe"]["relpath"],
            "recipe_sha256": recipe["environment_recipe"]["sha256"],
            "in_container_recipe_path": recipe["environment_recipe"]["in_container_path"],
            "chunk_tokens": recipe["geometry"]["chunk_tokens"],
            "argv_overlay": recipe["argv_overlay"],
            "launch_argv_sha256": recipe["launch_argv_sha256"],
        }
    contract = {"schema": G.VERSION_PAIRINGS[G.GEOMETRY_V2]["contract"], "profiles": profiles}
    S.validate_performance_profile_contract(contract, S.PLAN_SCHEMA_V3, G.REVIEWED_EXECUTABLE["path"])
    return contract


def self_validate(document: dict, path: Path) -> None:
    """Round-trip through the gate's own plan /3 host-reference loader."""
    raw = path.read_bytes()
    require(raw == S.canonical_bytes(document), "host reference bytes are not canonical")
    args = types.SimpleNamespace(
        suite=S, host_reference=str(path), host_reference_sha256=hashlib.sha256(raw).hexdigest(),
        launch_plan={"schema": S.PLAN_SCHEMA_V3, "performance_profile_contract": contract_v2(),
                     **document["provenance"]})
    D.load_host_reference(args)


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=G.PROFILE_NAMES, required=True)
        command.add_argument("--mode", choices=G.MODES, required=True)
        command.add_argument("--binary", required=True)
        command.add_argument("--model-path", required=True)
        command.add_argument("--port", type=int, required=True)
        command.add_argument("--alias", required=True)
    command = commands.add_parser("measure")
    command.add_argument("--profile", choices=G.PROFILE_NAMES, required=True)
    command.add_argument("--mode", choices=G.MODES, required=True)
    command.add_argument("--port", type=int, required=True)
    command.add_argument("--pid", type=int, required=True)
    command.add_argument("--server-log", type=Path, required=True)
    command.add_argument("--journal", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser("assemble")
    command.add_argument("--arm", type=Path, action="append", required=True)
    command.add_argument("--provenance", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("render", "launch"):
        transport = {key: os.environ[key] for key in HOST_TRANSPORT_KEYS if key in os.environ}
        launch = render(args.profile, args.mode, args.binary, args.model_path, args.port, args.alias, transport)
        if args.command == "render":
            print(json.dumps(launch, indent=2, sort_keys=True))
            return 0
        os.execve(launch["argv"][0], launch["argv"], launch["environment"])
    if args.command == "measure":
        arm = measure(args.profile, args.mode, f"http://127.0.0.1:{args.port}", args.pid, args.server_log,
                      args.journal)
        _write_new(args.out, S.canonical_bytes(arm) + b"\n")
        return 0
    arms = [json.loads(path.read_bytes()) for path in args.arm]
    document, summary = assemble(arms, json.loads(args.provenance.read_bytes()))
    _write_new(args.out, S.canonical_bytes(document))
    self_validate(document, args.out)
    _write_new(args.summary, S.canonical_bytes(summary) + b"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
