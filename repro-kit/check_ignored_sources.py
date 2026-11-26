#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Check committed source presence without relaxing credential ignore rules.

This is a source-presence snapshot, not a compiler, implementation, artifact
TOCTOU, or runtime qualification. Source bundles need their own manifest gate;
missing Git is an error here, never an implicit bundle-verification mode.

Inventory is limited to crates/ and kernels/, excluding the generated/cache
trees named in INVENTORY_PATHS. Opaque directory records fail closed rather
than recursively inspecting nested repositories. This is not include closure.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

REQUIRED_SOURCES = (
    "crates/atlas-embed/src/token.rs",
    "kernels/gb10/glm5.3-flash/exl3/ggml_q6_token_embedding.cu",
    "kernels/gb10/glm5.3-flash/iq3/ggml_q6_token_embedding.cu",
)
CREDENTIAL_PROBES = (
    "api_token",
    "secret.txt",
    "credentials.txt",
    "github_credentials.json",
    "access_token.yaml",
    "token_backup",
    ".env",
    "private.key",
    "crates/_source_gate_probe/src/access_token.rs",
    "kernels/_source_gate_probe/private_token.cu",
    "crates/_source_gate_probe/tokens.json",
    "crates/_source_gate_probe/secret.toml",
    "GH_TOKEN",
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "api_key.txt",
    "service-account.json",
    "sa-key.json",
    ".aws/config",
    "kubeconfig",
    ".kube/config",
)
SOURCE_SUFFIXES = frozenset(
    (".rs", ".cu", ".cuh", ".c", ".cpp", ".h", ".hpp", ".s", ".py", ".sh")
)
SOURCE_MANIFESTS = frozenset(("Cargo.toml", "KERNEL.toml", "HARDWARE.toml"))
INVENTORY_PATHS = (
    "crates",
    "kernels",
    ":(exclude,glob)**/target/**",
    ":(exclude,glob)**/*-target/**",
    ":(exclude,glob)**/.cargo-target/**",
    ":(exclude,glob)**/.aux-spec-target/**",
    ":(exclude,glob)**/__pycache__/**",
)


class GateError(Exception):
    """An inspection failure or a violated source-presence contract."""


def quoted(path: str | list[str]) -> str:
    return json.dumps(path, ensure_ascii=True)


class Git:
    def __init__(self, repo: Path):
        self.repo = repo
        self.env = os.environ.copy()
        # A caller's unrelated Git session must not redirect this inspection.
        for name in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CONFIG_PARAMETERS",
            "GIT_CONFIG_COUNT",
        ):
            self.env.pop(name, None)
        self.env["GIT_CONFIG_NOSYSTEM"] = "1"
        self.env["GIT_CONFIG_GLOBAL"] = os.devnull

    def run(
        self, *args: str, allowed: tuple[int, ...] = (0,), input: bytes | None = None
    ) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    f"core.excludesFile={os.devnull}",
                    "-c",
                    "core.ignoreCase=false",
                    "-C",
                    str(self.repo),
                    *args,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                input=input,
                check=False,
            )
        except OSError as error:
            raise GateError("Git inspection could not start") from error
        if result.returncode not in allowed:
            # Git stderr can contain local paths. Report status, not its contents.
            raise GateError(
                f"Git inspection {args[0]} failed with status {result.returncode}"
            )
        return result


def nul_records(data: bytes) -> list[bytes]:
    if data and not data.endswith(b"\0"):
        raise GateError("Git returned an unterminated filename inventory")
    return data[:-1].split(b"\0") if data else []


def require_committed_files(git: Git, commit: str, paths: tuple[str, ...]) -> None:
    tree = {}
    for record in nul_records(git.run("ls-tree", "-z", commit, "--", *paths).stdout):
        try:
            metadata, path = record.split(b"\t", 1)
            mode, kind, blob = metadata.split()
        except ValueError as error:
            raise GateError("Git returned an invalid source-tree record") from error
        tree[os.fsdecode(path)] = (mode, kind, blob)
    index = {}
    for record in nul_records(
        git.run("ls-files", "--stage", "-z", "--", *paths).stdout
    ):
        try:
            metadata, path = record.split(b"\t", 1)
            mode, blob, stage = metadata.split()
        except ValueError as error:
            raise GateError("Git returned an invalid source-index record") from error
        if stage != b"0" or os.fsdecode(path) in index:
            raise GateError("required source has an unresolved index entry")
        index[os.fsdecode(path)] = (mode, blob)
    for path in paths:
        entry = tree.get(path)
        if entry is None:
            raise GateError(
                f"required file is absent from the committed tree: {quoted(path)}"
            )
        mode, kind, blob = entry
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise GateError(
                f"required file is not a regular committed source: {quoted(path)}"
            )
        if index.get(path) != (mode, blob):
            raise GateError(
                f"required file index differs from the committed tree: {quoted(path)}"
            )
        try:
            regular = stat.S_ISREG((git.repo / path).lstat().st_mode)
        except OSError as error:
            raise GateError(
                f"required working file is unavailable: {quoted(path)}"
            ) from error
        if not regular:
            raise GateError(f"required working file is not regular: {quoted(path)}")
        actual = git.run("hash-object", "--no-filters", "--", path).stdout.strip()
        if actual != blob:
            raise GateError(
                f"required working file differs from the committed tree: {quoted(path)}"
            )


def is_source(path: str) -> bool:
    name = PurePosixPath(path)
    return name.suffix.lower() in SOURCE_SUFFIXES or name.name in SOURCE_MANIFESTS


def require_credential_policy(git: Git, commit: str) -> None:
    """Require root-rule provenance and exact rule text, allowing CRLF lines.

    Other Git-normalized spellings, such as unescaped trailing rule spaces,
    deliberately fail closed rather than guessing their committed spelling.
    """
    policy = git.run("show", f"{commit}:.gitignore").stdout.split(b"\n")
    for path in CREDENTIAL_PROBES:
        encoded = os.fsencode(path)
        result = git.run(
            "check-ignore",
            "--no-index",
            "--verbose",
            "-z",
            "--stdin",
            allowed=(0, 1),
            input=encoded + b"\0",
        )
        fields = nul_records(result.stdout)
        if result.returncode != 0 or not fields:
            raise GateError(f"credential filename is no longer ignored: {quoted(path)}")
        if len(fields) != 4 or fields[3] != encoded:
            raise GateError("Git returned invalid credential-rule provenance")
        origin, number, pattern, _ = fields
        # Verbose check-ignore can report an unignore rule with status zero.
        if pattern.startswith(b"!"):
            raise GateError(f"credential filename is no longer ignored: {quoted(path)}")
        line = 0
        if number.isdigit() and len(number) <= len(str(len(policy))):
            line = int(number)
        if (
            origin != b".gitignore"
            or not 1 <= line <= len(policy)
            or not pattern
            or policy[line - 1].removesuffix(b"\r") != pattern
        ):
            raise GateError(
                f"credential ignore rule is not committed root policy: {quoted(path)}"
            )


def check_repository(repo: Path) -> str:
    repo = repo.resolve()
    git = Git(repo)
    root = os.fsdecode(git.run("rev-parse", "--show-toplevel").stdout).rstrip("\n")
    if Path(root).resolve() != repo:
        raise GateError("--repo must name the repository root")
    commit = (
        git.run("rev-parse", "--verify", "HEAD^{commit}").stdout.decode("ascii").strip()
    )
    require_committed_files(git, commit, (".gitignore", *REQUIRED_SOURCES))
    for path in REQUIRED_SOURCES:
        if (
            git.run(
                "check-ignore", "--no-index", "-q", "--", path, allowed=(0, 1)
            ).returncode
            == 0
        ):
            raise GateError(f"required source is ignored: {quoted(path)}")
    require_credential_policy(git, commit)
    for ignored in (False, True):
        args = ["ls-files", "--others", "--exclude-standard", "-z"]
        if ignored:
            args.append("--ignored")
        args.extend(("--", *INVENTORY_PATHS))
        inventory = [
            os.fsdecode(record) for record in nul_records(git.run(*args).stdout)
        ]
        category = "ignored" if ignored else "untracked"
        opaque = sorted(path for path in inventory if path.endswith("/"))
        if opaque:
            raise GateError(
                f"{category} opaque directories require review: {quoted(opaque)}"
            )
        offenders = sorted(path for path in inventory if is_source(path))
        if offenders:
            raise GateError(
                f"{category} source files require review: {quoted(offenders)}"
            )
    if (
        git.run("rev-parse", "--verify", "HEAD^{commit}").stdout.decode("ascii").strip()
        != commit
    ):
        raise GateError("committed tree changed during source inspection")
    return commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    try:
        commit = check_repository(args.repo)
    except (GateError, UnicodeError) as error:
        print(f"check_ignored_sources: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"check_ignored_sources: OK (committed-tree {commit[:12]}, {len(REQUIRED_SOURCES)} required sources)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
