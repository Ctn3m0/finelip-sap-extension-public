#!/usr/bin/env python3
"""Install extension-owned modules into a separately obtained FineLIP checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


REQUIRED_UPSTREAM_FILES = (
    "model/model_finelip.py",
    "train/train.py",
    "train/arguments.py",
)


def _git_head(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def verify_upstream_checkout(root: Path | str, provenance: dict) -> list[str]:
    root = Path(root)
    violations = []
    expected = provenance.get("finelip_upstream", {}).get("main_commit")
    actual = _git_head(root)
    if actual is None:
        violations.append("FineLIP checkout must retain Git metadata for revision verification")
    elif actual != expected:
        violations.append(f"FineLIP revision mismatch: expected {expected}, got {actual}")
    for relative in REQUIRED_UPSTREAM_FILES:
        path = root / relative
        if not path.is_file():
            violations.append(f"missing expected upstream file: {relative}")
    expected_hashes = provenance.get("finelip_upstream", {}).get(
        "required_files_sha256", {}
    )
    for relative, expected_hash in expected_hashes.items():
        path = root / relative
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            violations.append(f"upstream file hash mismatch: {relative}")
    return violations


def _tree_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return manifest


def install_extension(
    root: Path | str,
    extension_root: Path | str,
    *,
    provenance: dict | None = None,
) -> None:
    root = Path(root).resolve()
    extension_root = Path(extension_root).resolve()
    if provenance is None:
        provenance = json.loads(
            (extension_root / "patches-or-installer" / "upstream_provenance.json").read_text(
                encoding="utf-8"
            )
        )
    violations = verify_upstream_checkout(root, provenance)
    if violations:
        raise ValueError("; ".join(violations))
    source = extension_root / "src" / "finelip_sap_ext"
    if not source.is_dir():
        raise FileNotFoundError(f"extension package not found: {source}")
    destination = root / "finelip_sap_ext"
    source_manifest = _tree_manifest(source)
    if destination.exists():
        if _tree_manifest(destination) != source_manifest:
            raise FileExistsError(
                f"refusing to replace a different extension package: {destination}"
            )
    else:
        shutil.copytree(source, destination, symlinks=False)
    marker = {
        "upstream_commit": provenance["finelip_upstream"]["main_commit"],
        "installed_package": "finelip_sap_ext",
        "files": source_manifest,
        "note": (
            "Component-only install: extension-owned modules were copied; upstream "
            "source was not redistributed or modified. This does not install the "
            "private training integration."
        ),
    }
    marker_path = root / "FINELIP_SAP_EXTENSION.json"
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if existing != marker:
            raise FileExistsError(f"refusing to replace different install marker: {marker_path}")
    else:
        marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("finelip_checkout")
    parser.add_argument(
        "--extension-root",
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    install_extension(args.finelip_checkout, args.extension_root)
    print(
        "Installed extension-owned components only; no upstream source file was "
        "modified and no runnable training integration was installed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

