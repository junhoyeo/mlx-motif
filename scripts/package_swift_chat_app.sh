#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

configuration="${CONFIGURATION:-release}"
app_name="${APP_NAME:-MotifChatApp}"
bundle_id="${BUNDLE_ID:-io.junho.motif.chat}"
output_dir="${OUTPUT_DIR:-dist}"
app_dir="${output_dir}/${app_name}.app"
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

ditto -c -k --keepParent "${app_dir}" "${output_dir}/${app_name}.zip"

if [[ -n "${NOTARYTOOL_PROFILE:-}" ]]; then
  xcrun notarytool submit "${output_dir}/${app_name}.zip" \
    --keychain-profile "${NOTARYTOOL_PROFILE}" \
    --wait
  xcrun stapler staple "${app_dir}"
fi

printf '%s\n' "${app_dir}"
printf '%s\n' "${output_dir}/${app_name}.zip"
