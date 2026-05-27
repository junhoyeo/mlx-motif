#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

configuration="${CONFIGURATION:-release}"
app_name="${APP_NAME:-MotifChatApp}"
bundle_id="${BUNDLE_ID:-io.junho.motif.chat}"
output_dir="${OUTPUT_DIR:-dist}"
app_dir="${output_dir}/${app_name}.app"
zip_path="${output_dir}/${app_name}.zip"
metadata_path="${METADATA_PATH:-${output_dir}/${app_name}.metadata.json}"
triple="$(swift -print-target-info | python3 -c 'import json,sys; info=json.load(sys.stdin)["target"]; print(info.get("unversionedTriple") or info["triple"])')"
binary_path="swift/.build/${triple}/${configuration}/${app_name}"

swift build --package-path swift -c "${configuration}" --product "${app_name}"

rm -rf "${app_dir}"
mkdir -p "${app_dir}/Contents/MacOS" "${app_dir}/Contents/Resources"
cp "${binary_path}" "${app_dir}/Contents/MacOS/${app_name}"

cat > "${app_dir}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>${app_name}</string>
  <key>CFBundleIdentifier</key>
  <string>${bundle_id}</string>
  <key>CFBundleName</key>
  <string>Motif</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION:-0.0.1}</string>
  <key>CFBundleVersion</key>
  <string>${BUILD_NUMBER:-1}</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --sign "${CODESIGN_IDENTITY}" "${app_dir}"
else
  codesign --force --deep --sign - "${app_dir}"
fi

ditto -c -k --keepParent "${app_dir}" "${zip_path}"

app_size="$(du -sk "${app_dir}" | awk '{print $1}')"
binary_sha256="$(shasum -a 256 "${app_dir}/Contents/MacOS/${app_name}" | awk '{print $1}')"
zip_sha256="$(shasum -a 256 "${zip_path}" | awk '{print $1}')"
git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
git_dirty="false"
if ! git diff --quiet --ignore-submodules -- 2>/dev/null || ! git diff --cached --quiet --ignore-submodules -- 2>/dev/null; then
  git_dirty="true"
fi
swift_version="$(swift --version | head -n 1)"

APP_NAME="${app_name}" \
BUNDLE_ID="${bundle_id}" \
CONFIGURATION="${configuration}" \
APP_DIR="${app_dir}" \
ZIP_PATH="${zip_path}" \
METADATA_PATH="${metadata_path}" \
TRIPLE="${triple}" \
BINARY_SHA256="${binary_sha256}" \
ZIP_SHA256="${zip_sha256}" \
APP_SIZE_KB="${app_size}" \
GIT_COMMIT="${git_commit}" \
GIT_DIRTY="${git_dirty}" \
SWIFT_VERSION="${swift_version}" \
python3 - <<'PY'
import json
import os
from pathlib import Path

metadata = {
    "schema_version": 1,
    "app_name": os.environ["APP_NAME"],
    "bundle_id": os.environ["BUNDLE_ID"],
    "configuration": os.environ["CONFIGURATION"],
    "app_dir": os.environ["APP_DIR"],
    "zip_path": os.environ["ZIP_PATH"],
    "target_triple": os.environ["TRIPLE"],
    "binary_sha256": os.environ["BINARY_SHA256"],
    "zip_sha256": os.environ["ZIP_SHA256"],
    "app_size_kb": int(os.environ["APP_SIZE_KB"]),
    "git": {
        "commit": os.environ["GIT_COMMIT"],
        "dirty": os.environ["GIT_DIRTY"] == "true",
    },
    "toolchain": {
        "swift": os.environ["SWIFT_VERSION"],
    },
}
path = Path(os.environ["METADATA_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

if [[ -n "${NOTARYTOOL_PROFILE:-}" ]]; then
  xcrun notarytool submit "${zip_path}" \
    --keychain-profile "${NOTARYTOOL_PROFILE}" \
    --wait
  xcrun stapler staple "${app_dir}"
fi

printf '%s\n' "${app_dir}"
printf '%s\n' "${zip_path}"
printf '%s\n' "${metadata_path}"
