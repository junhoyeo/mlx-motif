#!/usr/bin/env python3
"""Validate a packaged MotifChatApp bundle without launching the GUI."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import sys
import zipfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 1:
        fail(f"unsupported metadata schema: {metadata.get('schema_version')!r}")

    app_dir = Path(metadata["app_dir"])
    zip_path = Path(metadata["zip_path"])
    app_name = metadata["app_name"]
    bundle_id = metadata["bundle_id"]
    binary = app_dir / "Contents" / "MacOS" / app_name
    plist_path = app_dir / "Contents" / "Info.plist"

    for path in [app_dir, zip_path, binary, plist_path]:
        if not path.exists():
            fail(f"missing package artifact: {path}")

    plist = plistlib.loads(plist_path.read_bytes())
    if plist.get("CFBundleExecutable") != app_name:
        fail("Info.plist CFBundleExecutable does not match app name")
    if plist.get("CFBundleIdentifier") != bundle_id:
        fail("Info.plist CFBundleIdentifier does not match metadata")

    if sha256(binary) != metadata.get("binary_sha256"):
        fail("binary sha256 mismatch")
    if sha256(zip_path) != metadata.get("zip_sha256"):
        fail("zip sha256 mismatch")

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    expected_binary = f"{app_name}.app/Contents/MacOS/{app_name}"
    expected_plist = f"{app_name}.app/Contents/Info.plist"
    missing = {expected_binary, expected_plist} - names
    if missing:
        fail(f"zip missing entries: {sorted(missing)}")

    return {
        "metadata": str(metadata_path),
        "app_dir": str(app_dir),
        "zip_path": str(zip_path),
        "zip_sha256": metadata["zip_sha256"],
        "binary_sha256": metadata["binary_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata", type=Path, help="Path to *.metadata.json from package_swift_chat_app.sh"
    )
    args = parser.parse_args()
    result = validate(args.metadata)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
