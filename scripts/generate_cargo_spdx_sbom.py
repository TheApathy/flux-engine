#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate a deterministic SPDX 2.3 JSON SBOM from Cargo.toml + Cargo.lock.

Zero third-party dependencies (Python 3.11+ standard library only). Produces
byte-for-byte reproducible output for a given repository commit, release tag,
and creation timestamp.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_URL = "https://github.com/TheApathy/flux-engine"
SUPPLIER = "Organization: TheApathy"
CREATOR_TOOL = "Tool: flux-engine-cargo-spdx-sbom-1.0"
SPDX_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9.-]+")


def _sanitize_spdx_id(raw: str) -> str:
    cleaned = SPDX_ID_SAFE_RE.sub("-", raw).strip("-")
    return cleaned or "pkg"


def _normalize_created(raw: str | None) -> str:
    if raw:
        candidate = raw.strip()
        if candidate.isdigit():
            ts = dt.datetime.fromtimestamp(int(candidate), tz=dt.timezone.utc)
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    epoch = os.environ.get("SOURCE_DATE_EPOCH", "").strip()
    if epoch.isdigit():
        ts = dt.datetime.fromtimestamp(int(epoch), tz=dt.timezone.utc)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    return "1970-01-01T00:00:00Z"


def _load_workspace_metadata(repo_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root_toml = tomllib.loads((repo_root / "Cargo.toml").read_text(encoding="utf-8"))
    workspace = root_toml.get("workspace", {})
    ws_pkg = workspace.get("package", {})
    members_meta: dict[str, dict[str, Any]] = {}

    for member_rel in workspace.get("members", []):
        member_path = repo_root / member_rel / "Cargo.toml"
        if not member_path.is_file():
            continue
        member_data = tomllib.loads(member_path.read_text(encoding="utf-8"))
        pkg = member_data.get("package", {})
        name = pkg.get("name")
        if not name:
            continue

        version = pkg.get("version")
        if isinstance(version, dict) and version.get("workspace"):
            version = ws_pkg.get("version", "0.0.0")
        elif not isinstance(version, str):
            version = ws_pkg.get("version", "0.0.0")

        license_expr = pkg.get("license")
        if isinstance(license_expr, dict) and license_expr.get("workspace"):
            license_expr = ws_pkg.get("license", "NOASSERTION")
        elif not isinstance(license_expr, str):
            license_expr = "NOASSERTION"

        description = pkg.get("description")
        if isinstance(description, dict) and description.get("workspace"):
            description = ws_pkg.get("description")

        members_meta[name] = {
            "name": name,
            "version": version,
            "license": license_expr,
            "description": description if isinstance(description, str) else None,
            "path": member_rel,
        }
    return ws_pkg, members_meta


def _parse_dep_spec(spec: str) -> tuple[str, str | None, str | None]:
    """Parse Cargo.lock dependency string: 'name', 'name ver', or 'name ver (source)'."""
    parts = spec.split(" ", 2)
    name = parts[0]
    version = parts[1] if len(parts) >= 2 else None
    source = None
    if len(parts) == 3 and parts[2].startswith("(") and parts[2].endswith(")"):
        source = parts[2][1:-1]
    return name, version, source


def _download_location(name: str, version: str, source: str | None) -> str:
    if not source:
        return "NONE"
    if source == "registry+https://github.com/rust-lang/crates.io-index":
        return f"https://crates.io/api/v1/crates/{quote(name, safe='')}/{quote(version, safe='')}/download"
    if source.startswith("git+"):
        return source[4:]
    return "NOASSERTION"


def _cargo_purl(name: str, version: str, source: str | None) -> str:
    base = f"pkg:cargo/{quote(name, safe='')}@{quote(version, safe='')}"
    if source and source.startswith("git+"):
        return f"{base}?vcs_url={quote(source, safe='')}"
    return base


def build_spdx_document(
    repo_root: Path,
    release_tag: str | None = None,
    commit: str | None = None,
    created: str | None = None,
) -> dict[str, Any]:
    ws_pkg, members_meta = _load_workspace_metadata(repo_root)
    lock_bytes = (repo_root / "Cargo.lock").read_bytes()
    lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    lock_packages: list[dict[str, Any]] = list(lock_data.get("package", []))

    ws_version = str(ws_pkg.get("version", "0.0.0"))
    ws_license = str(ws_pkg.get("license", "AGPL-3.0-only"))
    effective_tag = release_tag or f"v{ws_version}"
    created_iso = _normalize_created(created)

    ns_suffix = commit.strip() if commit and commit.strip() else hashlib.sha256(lock_bytes).hexdigest()[:16]
    document_namespace = (
        f"{REPO_URL}/releases/tag/{quote(effective_tag, safe='')}/spdx/{quote(ns_suffix, safe='')}"
    )

    # Sort Cargo.lock packages deterministically before assigning SPDXIDs.
    lock_packages.sort(
        key=lambda p: (
            str(p.get("name", "")),
            str(p.get("version", "")),
            str(p.get("source", "")),
            str(p.get("checksum", "")),
        )
    )

    # Assign unique, valid SPDXIDs for every package entry in Cargo.lock.
    used_ids: set[str] = {"SPDXRef-DOCUMENT", "SPDXRef-Package-flux-engine"}
    pkg_spdx_id: dict[tuple[str, str, str], str] = {}
    by_name: dict[str, list[tuple[str, str, str]]] = {}

    for pkg in lock_packages:
        name = str(pkg["name"])
        version = str(pkg["version"])
        source = str(pkg.get("source", ""))
        key = (name, version, source)
        base_id = f"SPDXRef-Package-crate-{_sanitize_spdx_id(name)}-{_sanitize_spdx_id(version)}"
        candidate = base_id
        if candidate in used_ids:
            digest = hashlib.sha256(f"{name}\0{version}\0{source}".encode("utf-8")).hexdigest()[:8]
            candidate = f"{base_id}-{digest}"
        counter = 2
        while candidate in used_ids:
            candidate = f"{base_id}-{counter}"
            counter += 1
        used_ids.add(candidate)
        pkg_spdx_id[key] = candidate
        by_name.setdefault(name, []).append(key)

    root_spdx_id = "SPDXRef-Package-flux-engine"
    root_package: dict[str, Any] = {
        "SPDXID": root_spdx_id,
        "name": "flux-engine",
        "versionInfo": effective_tag.lstrip("v") if effective_tag.startswith("v") else effective_tag,
        "downloadLocation": f"{REPO_URL}/archive/refs/tags/{quote(effective_tag, safe='')}.tar.gz",
        "filesAnalyzed": False,
        "supplier": SUPPLIER,
        "homepage": str(ws_pkg.get("homepage", REPO_URL)),
        "licenseConcluded": ws_license,
        "licenseDeclared": ws_license,
        "copyrightText": "NOASSERTION",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:github/TheApathy/flux-engine@{quote(effective_tag, safe='')}",
            }
        ],
    }

    packages: list[dict[str, Any]] = [root_package]
    relationships: list[dict[str, str]] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_spdx_id,
        }
    ]

    for pkg in lock_packages:
        name = str(pkg["name"])
        version = str(pkg["version"])
        source = str(pkg.get("source", ""))
        key = (name, version, source)
        spdx_id = pkg_spdx_id[key]
        is_workspace_member = (not source) and (name in members_meta)

        if is_workspace_member:
            meta = members_meta[name]
            lic = meta["license"]
            entry: dict[str, Any] = {
                "SPDXID": spdx_id,
                "name": name,
                "versionInfo": version,
                "downloadLocation": "NONE",
                "filesAnalyzed": False,
                "supplier": SUPPLIER,
                "licenseConcluded": lic,
                "licenseDeclared": lic,
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": _cargo_purl(name, version, None),
                    }
                ],
            }
            if meta.get("description"):
                entry["description"] = meta["description"]
            relationships.append(
                {
                    "spdxElementId": root_spdx_id,
                    "relationshipType": "CONTAINS",
                    "relatedSpdxElement": spdx_id,
                }
            )
        else:
            entry = {
                "SPDXID": spdx_id,
                "name": name,
                "versionInfo": version,
                "downloadLocation": _download_location(name, version, source or None),
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": _cargo_purl(name, version, source or None),
                    }
                ],
            }
            checksum = pkg.get("checksum")
            if isinstance(checksum, str) and checksum:
                entry["checksums"] = [{"algorithm": "SHA256", "checksumValue": checksum}]

        packages.append(entry)

    # Emit deterministic DEPENDS_ON relationships from Cargo.lock.
    dep_edges: set[tuple[str, str]] = set()
    for pkg in lock_packages:
        src_key = (str(pkg["name"]), str(pkg["version"]), str(pkg.get("source", "")))
        src_id = pkg_spdx_id[src_key]
        for dep_spec in pkg.get("dependencies", []):
            dep_name, dep_ver, dep_src = _parse_dep_spec(str(dep_spec))
            candidates = by_name.get(dep_name, [])
            matched_keys: list[tuple[str, str, str]] = []
            for cand in candidates:
                c_name, c_ver, c_src = cand
                if dep_ver is not None and c_ver != dep_ver:
                    continue
                if dep_src is not None and c_src != dep_src:
                    continue
                matched_keys.append(cand)
            if len(matched_keys) == 1:
                dep_edges.add((src_id, pkg_spdx_id[matched_keys[0]]))
            elif matched_keys:
                # Deterministic tie-break if a spec omits source/version.
                dep_edges.add((src_id, pkg_spdx_id[sorted(matched_keys)[0]]))

    for src_id, dst_id in sorted(dep_edges):
        relationships.append(
            {
                "spdxElementId": src_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": dst_id,
            }
        )

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"flux-engine-{effective_tag}",
        "documentNamespace": document_namespace,
        "creationInfo": {
            "created": created_iso,
            "creators": [SUPPLIER, CREATOR_TOOL],
        },
        "packages": packages,
        "relationships": relationships,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic SPDX 2.3 JSON SBOM from Cargo.toml + Cargo.lock."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the Flux Engine repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--release-tag",
        type=str,
        default=None,
        help="Release tag (e.g., v1.0.3). Defaults to v<workspace.package.version>.",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Git commit SHA embedded into the deterministic documentNamespace.",
    )
    parser.add_argument(
        "--created",
        type=str,
        default=None,
        help="Deterministic UTC creation timestamp (ISO-8601 or Unix epoch seconds).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for the generated .spdx.json file.",
    )
    args = parser.parse_args(argv)

    doc = build_spdx_document(
        repo_root=args.repo_root.resolve(),
        release_tag=args.release_tag,
        commit=args.commit,
        created=args.created,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(doc, indent=2, sort_keys=False) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"Wrote SPDX 2.3 SBOM to {args.output} "
        f"({len(doc['packages'])} packages, {len(doc['relationships'])} relationships)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
