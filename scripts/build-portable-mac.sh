#!/usr/bin/env bash
# Assembles the final macOS portable folder and zips it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST_NAME="DouyinVietnamizer-0.1.0-portable"
DST="$REPO_ROOT/dist-portable/$DST_NAME"
STAGING_RUNTIME="$REPO_ROOT/dist-portable/macos-staging/portable-runtime"
APP_PATH="$REPO_ROOT/src-tauri/target/aarch64-apple-darwin/release/bundle/macos/DouyinVietnamizer.app"

if [ ! -d "$APP_PATH" ]; then
  echo "ERROR: $APP_PATH not found. Run 'pnpm tauri build --target aarch64-apple-darwin' first." >&2
  exit 1
fi
if [ ! -d "$STAGING_RUNTIME" ]; then
  echo "ERROR: $STAGING_RUNTIME not found. Run scripts/build-portable-runtime-mac.sh first." >&2
  exit 1
fi

echo ">>> Assembling $DST/"
rm -rf "$DST"
mkdir -p "$DST"
cp -R "$APP_PATH" "$DST/DouyinVietnamizer.app"
cp -R "$STAGING_RUNTIME" "$DST/portable-runtime"

# Re-sync backend Python sources in case they changed after the staging build.
rsync -a --delete "$REPO_ROOT/backend/dv_backend/" "$DST/portable-runtime/backend/dv_backend/"
rsync -a "$REPO_ROOT/backend/scripts/" "$DST/portable-runtime/backend/scripts/"
cp "$REPO_ROOT/backend/pyproject.toml" "$DST/portable-runtime/backend/pyproject.toml"

cd "$REPO_ROOT/dist-portable"
ditto -c -k --sequesterRsrc --keepParent "$DST_NAME" "${DST_NAME}-macos.zip"
echo ">>> Built: dist-portable/${DST_NAME}-macos.zip"
ls -la "${DST_NAME}-macos.zip"
