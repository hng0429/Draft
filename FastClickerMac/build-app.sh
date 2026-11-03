#!/usr/bin/env bash
# FastClicker.app 번들 생성 (Apple Silicon + Intel 유니버설 바이너리)
# 사용법: ./build-app.sh   →  build/FastClicker.app, build/FastClicker-macOS.zip
set -euo pipefail
cd "$(dirname "$0")"

swift build -c release --arch arm64 --arch x86_64
BIN="$(swift build -c release --arch arm64 --arch x86_64 --show-bin-path)/FastClicker"

APP=build/FastClicker.app
rm -rf build
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/FastClicker"
cp Info.plist "$APP/Contents/Info.plist"

# ad-hoc 서명: 손쉬운 사용 권한이 앱에 안정적으로 연결되도록
codesign --force --deep --sign - "$APP"

ditto -c -k --keepParent "$APP" build/FastClicker-macOS.zip
echo "완료: $APP"
