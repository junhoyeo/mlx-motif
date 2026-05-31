from __future__ import annotations

import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest


def _load_verify_package():
    spec = importlib.util.spec_from_file_location(
        "verify_swift_chat_app_package", Path("scripts/verify_swift_chat_app_package.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_swift_chat_app_package"] = module
    spec.loader.exec_module(module)
    return module


def _make_valid_package(tmp_path: Path, verifier):
    """Build a minimal valid package under tmp_path and return (metadata, metadata_path)."""
    app_name = "MotifChatApp"
    app_dir = tmp_path / f"{app_name}.app"
    binary = app_dir / "Contents" / "MacOS" / app_name
    plist = app_dir / "Contents" / "Info.plist"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"#!/bin/sh\necho motif\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    plist.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>MotifChatApp</string>
  <key>CFBundleIdentifier</key>
  <string>io.test.motif</string>
</dict>
</plist>
"""
    )
    zip_path = tmp_path / "MotifChatApp.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(binary, f"{app_name}.app/Contents/MacOS/{app_name}")
        zf.write(plist, f"{app_name}.app/Contents/Info.plist")
    metadata = {
        "schema_version": 1,
        "app_name": app_name,
        "bundle_id": "io.test.motif",
        "app_dir": str(app_dir),
        "zip_path": str(zip_path),
        "binary_sha256": verifier.sha256(binary),
        "zip_sha256": verifier.sha256(zip_path),
    }
    metadata_path = tmp_path / "MotifChatApp.metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    return metadata, metadata_path


def test_validate_packaged_app_metadata(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    result = verifier.validate(metadata_path)
    assert result["zip_sha256"] == metadata["zip_sha256"]


def test_validate_fails_on_wrong_schema_version(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    bad = json.loads(metadata_path.read_text())
    bad["schema_version"] = 99
    metadata_path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        verifier.validate(metadata_path)


def test_validate_fails_on_missing_binary(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    app_name = metadata["app_name"]
    binary = Path(metadata["app_dir"]) / "Contents" / "MacOS" / app_name
    binary.unlink()
    with pytest.raises(SystemExit):
        verifier.validate(metadata_path)


def test_validate_fails_on_missing_zip(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    Path(metadata["zip_path"]).unlink()
    with pytest.raises(SystemExit):
        verifier.validate(metadata_path)


def test_validate_fails_on_binary_sha256_mismatch(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    bad = json.loads(metadata_path.read_text())
    bad["binary_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        verifier.validate(metadata_path)


def test_validate_fails_on_zip_sha256_mismatch(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    bad = json.loads(metadata_path.read_text())
    bad["zip_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        verifier.validate(metadata_path)


def test_validate_fails_on_plist_bundle_id_mismatch(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    bad = json.loads(metadata_path.read_text())
    bad["bundle_id"] = "io.wrong.bundle"
    metadata_path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        verifier.validate(metadata_path)


def test_validate_fails_on_zip_missing_entry(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    app_name = metadata["app_name"]
    app_dir = Path(metadata["app_dir"])
    zip_path = Path(metadata["zip_path"])
    binary = app_dir / "Contents" / "MacOS" / app_name
    # Rewrite zip without the plist entry
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(binary, f"{app_name}.app/Contents/MacOS/{app_name}")
    bad = json.loads(metadata_path.read_text())
    bad["zip_sha256"] = verifier.sha256(zip_path)
    metadata_path.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        verifier.validate(metadata_path)


def test_validate_fails_on_non_executable_binary(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    metadata, metadata_path = _make_valid_package(tmp_path, verifier)
    app_name = metadata["app_name"]
    binary = Path(metadata["app_dir"]) / "Contents" / "MacOS" / app_name
    binary.chmod(0o644)
    with pytest.raises(SystemExit):
        verifier.validate(metadata_path)
    # restore so tmp_path cleanup works on all platforms
    binary.chmod(0o755)
