#!/bin/bash
# Build Dictation.app — a double-clickable wrapper around dictate.py
set -e
cd "$(dirname "$0")"

APP="Dictation.app"
CONTENTS="$APP/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"

echo "=== Building Dictation.app ==="

# Generate icon
if [ ! -f AppIcon.icns ]; then
    echo "Generating app icon..."
    /usr/bin/python3 make_icon.py
fi

# Clean old build
rm -rf "$APP"

# Create .app structure
mkdir -p "$MACOS" "$RESOURCES"

# Copy resources
cp dictate.py "$RESOURCES/"
cp AppIcon.icns "$RESOURCES/"

# Create launcher script
cat > "$MACOS/Dictation" << 'LAUNCHER'
#!/bin/bash
# Launch dictation — bootstraps its own venv on first run
DIR="$(cd "$(dirname "$0")" && pwd)"
RESOURCES="$DIR/../Resources"
exec /usr/bin/python3 "$RESOURCES/dictate.py"
LAUNCHER
chmod +x "$MACOS/Dictation"

# Create Info.plist
cat > "$CONTENTS/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Dictation</string>
    <key>CFBundleDisplayName</key>
    <string>Dictation</string>
    <key>CFBundleIdentifier</key>
    <string>com.treyhoffman.dictation</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>Dictation</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Dictation needs microphone access to transcribe your speech.</string>
</dict>
</plist>
PLIST

echo ""
echo "Built $APP"
echo "  Double-click to launch, or move to /Applications."
echo "  First launch installs dependencies automatically."
