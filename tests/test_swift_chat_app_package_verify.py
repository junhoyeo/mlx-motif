from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


def _load_verify_package():
    spec = importlib.util.spec_from_file_location(
        "verify_swift_chat_app_package", Path("scripts/verify_swift_chat_app_package.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_swift_chat_app_package"] = module
    spec.loader.exec_module(module)
    return module


def test_validate_packaged_app_metadata(tmp_path: Path) -> None:
    verifier = _load_verify_package()
    app_name = "MotifChatApp"
    app_dir = tmp_path / f"{app_name}.app"
    binary = app_dir / "Contents" / "MacOS" / app_name
    plist = app_dir / "Contents" / "Info.plist"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"#!/bin/sh\necho motif\n")
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

    result = verifier.validate(metadata_path)

    assert result["zip_sha256"] == metadata["zip_sha256"]
