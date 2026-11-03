# FastClicker — 순차 고속 클릭 (Windows / macOS)

화면의 원하는 위치들을 **등록한 순서대로 매우 빠르게** 클릭하는 Windows 앱입니다.

## 사용법

1. `FastClicker.exe` 실행 (설치 불필요, .NET 런타임 내장 단일 파일)
2. 클릭할 위치에 마우스를 올리고 **F2** → 지점이 목록에 추가됩니다. 원하는 만큼 반복.
3. 각 지점의 버튼(왼쪽/오른쪽/가운데), 클릭 횟수(2 = 더블클릭), 클릭 후 대기(ms)를 설정
   - 목록에서 항목 선택 → 값 변경 → **선택 항목 수정**
   - **대기 일괄 적용**: 현재 대기 값을 모든 지점에 적용
4. 반복 횟수(0 = 무한), 한 바퀴 후 대기, 시작 지연을 설정
5. **F6** 또는 **시작** 버튼 → 실행. 다시 **F6** 또는 **Esc** 로 즉시 정지

| 단축키 | 동작 |
|---|---|
| F2 | 현재 마우스 위치를 지점으로 추가 (다른 창이 활성 상태여도 동작) |
| F6 | 시작 / 정지 |
| Esc | 실행 중 비상 정지 |

지점 목록과 설정은 **저장... / 불러오기...** 로 JSON 파일에 보관할 수 있습니다.

## 속도

- `SendInput`으로 이동+누름+뗌을 한 번에 전송하고, 전용 고우선순위 스레드에서 실행합니다.
- 대기 0 ms로 설정하면 대기 없이 연속 클릭합니다 (초당 수천 회 이상, 대상 프로그램이 처리할 수 있는 속도가 실제 한계).
- 짧은 대기는 스핀 대기로 1 ms 단위까지 정확하게 맞춥니다.
- 완료 후 상태 표시줄에 총 클릭 수와 초당 클릭 수가 표시됩니다.

## 참고

- 다중 모니터와 화면 배율(DPI)을 지원합니다. 좌표는 실제 픽셀 기준입니다.
- 관리자 권한으로 실행된 프로그램은 클릭이 전달되지 않을 수 있습니다. 그런 경우 FastClicker도 관리자 권한으로 실행하세요.
- 게임/서비스 이용 약관에서 자동 입력을 금지하는 경우가 있으니 주의하세요.

## macOS 버전 (`FastClickerMac/`)

기능은 Windows 버전과 같고, 프로필(JSON) 파일 형식도 같습니다. macOS 13 이상, Apple Silicon / Intel 모두 지원.

| 단축키 | 동작 |
|---|---|
| ⌃⌥A 또는 F2 | 현재 마우스 위치를 지점으로 추가 |
| ⌃⌥S 또는 F6 | 시작 / 정지 |
| Esc | 실행 중 비상 정지 |

처음 실행 시:

1. `FastClicker-macOS.zip`을 풀고 `FastClicker.app`을 **우클릭 → 열기** (서명되지 않은 앱이라 Gatekeeper 경고가 뜹니다)
2. 앱 상단의 **권한 요청** → 시스템 설정 → 개인정보 보호 및 보안 → **손쉬운 사용**에서 FastClicker를 켭니다. 권한이 없으면 클릭이 전달되지 않습니다.
3. 앱을 새 버전으로 교체한 뒤 클릭이 안 되면 손쉬운 사용 목록에서 FastClicker를 지웠다가 다시 추가하세요.

좌표는 macOS 포인트 단위(주 모니터 왼쪽 위가 원점)라 Windows에서 저장한 프로필의 좌표는 화면에 맞게 다시 잡아야 할 수 있습니다.

빌드 (Xcode 또는 Command Line Tools 필요):

```
cd FastClickerMac && ./build-app.sh     # → build/FastClicker.app, build/FastClicker-macOS.zip
```

## 빌드 (Windows)

Windows + .NET 8 SDK:

```
dotnet publish FastClicker/FastClicker.csproj -c Release -r win-x64 --self-contained -p:PublishSingleFile=true
```

GitHub에 push하면 Actions가 자동으로 빌드해 아티팩트로 올립니다.
- `Build FastClicker` → `FastClicker-win-x64` (FastClicker.exe)
- `Build FastClicker (macOS)` → `FastClicker-macOS` (FastClicker.app zip)
