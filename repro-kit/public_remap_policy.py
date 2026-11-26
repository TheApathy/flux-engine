#!/usr/bin/env python3
"""Pure source-only environment recipe. No subprocess, scan or build entry point.

The controller remains responsible for admission, source and dependency seals,
fresh outputs, workload locks and cleanup. Public qualification stays HOLD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import tomllib


SCHEMA = "flux.release-path-remap-policy/1"
PUBLIC_ROOT = "/flux-build"
FROM = "ATLAS_RELEASE_PATH_REMAP_FROM"
TO = "ATLAS_RELEASE_PATH_REMAP_TO"
POLICY = "ATLAS_RELEASE_PATH_REMAP_POLICY"
SOURCE_CONFIG_SHA256 = "409c89b9f6e43aee3e5675be36262bca2a21dc7beef27742e7d6866dbcc988c3"
CPU_FLAGS = ["-C", "target-cpu=native"]
MAP_OPTIONS = ("-ffile-prefix-map", "-fmacro-prefix-map", "-fdebug-prefix-map")
ROOT_FIELDS = ("source", "target", "evidence", "cargo_home", "home", "flashinfer_home")
FLAG_OVERRIDES = frozenset({"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CFLAGS", "CXXFLAGS",
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", FROM, TO, POLICY})
# Optional locally built FlashInfer GDN AOT objects (excluded from git; default
# open-source builds use the native CUDA GDN fallback — see crates/spark-model/csrc/fi_gdn/NOTICE).
AOT_OBJECTS = (
    "crates/spark-model/csrc/fi_gdn/atlas_fi_gdn_sm121a_prefill.o",
    "crates/spark-model/csrc/fi_gdn/atlas_fi_gdn_sm121a_prefill_f16.o",
    "crates/spark-model/csrc/fi_gdn/atlas_fi_gdn_sm121a_prefill_f16_nf8.o",
)
HOLDS = ("PREEXPORTED_AOT_PROVENANCE_NOT_VERIFIED",
         "CUDA_DEVICE_AND_FINAL_ARTIFACT_PRIVACY_NOT_VERIFIED",
         "SUCCESSOR_SOURCE_BUILD_AND_RUNTIME_NOT_QUALIFIED")


class Hold(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Hold(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii"))


def canonical_root(value):
    require(type(value) is str and 1 < len(value) <= 4096
            and re.fullmatch(r"/[A-Za-z0-9_./-]+", value) is not None
            and not value.endswith("/") and "//" not in value
            and all(part not in ("", ".", "..") for part in value.split("/")[1:]),
            "canonical remap root required")
    return Path(value)


def paired_roots(remap_from, remap_to):
    if remap_from is None and remap_to is None:
        return None
    require(remap_from is not None and remap_to is not None,
            "path-remap roots must be configured together")
    source, destination = canonical_root(remap_from), canonical_root(remap_to)
    require(str(destination) == PUBLIC_ROOT, "fixed public remap destination required")
    require(source != destination and source not in destination.parents
            and destination not in source.parents, "disjoint remap roots required")
    require(source.is_dir() and source.resolve(strict=True) == source,
            "existing canonical remap source directory required")
    return source, destination


def canonical_directory_chain(path):
    require(path.resolve(strict=False) == path, "symlinked build root refused")
    current = Path("/")
    for component in path.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                "directory-only build root chain required")
    return path


def native_argv(tool, arguments, remap_from=None, remap_to=None, *, nvcc=False):
    """Reference argv builder for offline checks; Rust supplies production calls."""
    require(type(tool) is str and bool(tool) and type(arguments) is list
            and all(type(item) is str for item in arguments) and type(nvcc) is bool,
            "explicit compiler argv required")
    roots = paired_roots(remap_from, remap_to)
    flags = []
    if roots:
        for option in MAP_OPTIONS:
            if nvcc:
                flags.append("-Xcompiler")
            flags.append(f"{option}={roots[0]}={roots[1]}")
    return [tool, *flags, *arguments]


def checked_cpu_flags(source_config):
    require(type(source_config) is bytes and digest(source_config) == SOURCE_CONFIG_SHA256,
            "reviewed source Cargo config bytes required")
    config = tomllib.loads(source_config.decode("utf-8", "strict"))
    require(config == {"build": {"rustflags": CPU_FLAGS}},
            "reviewed CPU flag policy differs")
    return list(config["build"]["rustflags"])


def build_inputs(base_environment, manifest, *, remap_from=None, remap_to=None,
                 source_config=None):
    """Return private environment plus opaque public recipe, never launch authority.

    The base must come from the unchanged, already verified guarded environment
    constructor. The private environment must remain in local sealed evidence.
    """
    require(type(base_environment) is dict and all(type(key) is str and type(value) is str
            for key, value in base_environment.items()), "plain base environment required")
    environment = dict(base_environment)
    roots = paired_roots(remap_from, remap_to)
    if roots is None:
        require(not ({FROM, TO, POLICY} & environment.keys()),
                "unrequested remap environment refused")
        return {"environment": environment, "public_policy": {
            "schema": SCHEMA, "enabled": False,
            "base_environment_sha256": json_digest(base_environment),
            "environment_sha256": json_digest(environment),
            "runtime_qualified": False, "release_qualified": False,
            "public_path_qualified": False, "execution": "NOT_AUTHORIZED"}}
    source, destination = roots
    require(type(manifest) is dict, "reviewed build manifest required")
    require(not (FLAG_OVERRIDES & environment.keys()), "conflicting build flag override refused")
    require(not any(key.endswith(("_RUSTFLAGS", "_CFLAGS", "_CXXFLAGS"))
                    or key.startswith(("CFLAGS_", "CXXFLAGS_")) for key in environment),
            "target-specific build flag override refused")
    paths = {key: canonical_directory_chain(canonical_root(manifest.get(key)))
             for key in ROOT_FIELDS}
    require(all(source in path.parents for path in paths.values()),
            "all mutable and private input roots must be under remap source")
    require(all(a != b and a not in b.parents and b not in a.parents
                for index, a in enumerate(paths.values())
                for b in list(paths.values())[index + 1:]), "build roots must be disjoint")
    for name in ("commit", "tree"):
        require(type(manifest.get(name)) is str
                and re.fullmatch("[0-9a-f]{40}", manifest[name]) is not None,
                "successor source identity required")
    require(environment.get("ATLAS_BUILD_COMMIT") == manifest["commit"],
            "base environment source identity differs")
    expected_environment_roots = {"CARGO_HOME": "cargo_home", "HOME": "home",
        "FLASHINFER_HOME": "flashinfer_home", "CARGO_TARGET_DIR": "target"}
    require(all(environment.get(key) == str(paths[field])
                for key, field in expected_environment_roots.items()),
            "base environment input/output roots differ")
    cpu_flags = checked_cpu_flags(source_config)
    rust_flags = [*cpu_flags, f"--remap-path-prefix={source}={destination}"]
    host_flags = [f"{option}={source}={destination}" for option in MAP_OPTIONS]
    environment.update({FROM: str(source), TO: str(destination), POLICY: SCHEMA,
                        "CARGO_ENCODED_RUSTFLAGS": "\x1f".join(rust_flags),
                        "CFLAGS": " ".join(host_flags), "CXXFLAGS": " ".join(host_flags)})
    policy = {
        "schema": SCHEMA, "enabled": True, "from_root_sha256": digest(str(source).encode("ascii")),
        "public_root": str(destination), "source": {name: manifest[name] for name in ("commit", "tree")},
        "source_config_sha256": digest(source_config), "cpu_flags": cpu_flags,
        "rust_override": "CARGO_ENCODED_RUSTFLAGS_WITH_EXACT_REVIEWED_CONFIG_BASE",
        "host_map_options": list(MAP_OPTIONS), "nvcc_scope": "HOST_COMPILER_ONLY",
        "preexported_aot_inputs": list(AOT_OBJECTS), "holds": list(HOLDS),
        "base_environment_sha256": json_digest(base_environment),
        "environment_sha256": json_digest(environment),
        "runtime_qualified": False, "release_qualified": False,
        "public_path_qualified": False, "execution": "HOLD_AOT_AND_DEVICE_PRIVACY_REVIEW",
    }
    return {"environment": environment, "public_policy": policy}


def environment_factory(parent, *, remap_from, remap_to, source_config, record_policy):
    """Prepare an inherited environment hook without installing or running it.

    The caller must separately integrate policy recording and exact expected
    environment verification into a fresh reviewed guarded-build successor.
    """
    original = parent.environment
    reject = parent.reject_build_environment_markers
    require(callable(original) and callable(reject) and callable(record_policy),
            "reviewed inherited environment and recording hooks required")
    def environment(manifest):
        recipe = build_inputs(original(manifest), manifest, remap_from=remap_from,
                              remap_to=remap_to, source_config=source_config)
        reject(recipe["environment"])
        record_policy(recipe["public_policy"])
        if recipe["public_policy"]["enabled"]:
            require_public_eligibility(recipe)
        return recipe["environment"]
    return environment


def require_public_eligibility(recipe):
    """Fail closed until source-pinned AOT/device audit replay is implemented.

    No approval flag, saved PASS, or edited qualification boolean can satisfy
    this boundary. Its future positive path requires independently reviewed
    source and audit-verifier pins, not a new caller-supplied assertion.
    """
    require(type(recipe) is dict and set(recipe) == {"environment", "public_policy"},
            "exact public-remap recipe required")
    environment, public = recipe["environment"], recipe["public_policy"]
    require(type(environment) is dict and type(public) is dict
            and type(public.get("schema")) is str and public["schema"] == SCHEMA,
            "public-remap recipe schema differs")
    require(all(type(key) is str and type(value) is str for key, value in environment.items())
            and public.get("environment_sha256") == json_digest(environment),
            "public-remap environment binding differs")
    raise Hold("public eligibility HOLD: no source-pinned AOT/device audit verifier is registered")


def main():
    print(json.dumps({"mode": "PLAN_ONLY", "execution": "NOT_AUTHORIZED",
                      "public_path_qualified": False, "holds": list(HOLDS)}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
