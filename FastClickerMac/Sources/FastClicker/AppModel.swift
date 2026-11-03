import AppKit
import Carbon.HIToolbox
import Foundation
import UniformTypeIdentifiers

final class AppModel: ObservableObject {
    // 목록
    @Published var points: [ClickPoint] = []
    @Published var selection: ClickPoint.ID?

    // 지점 편집기
    @Published var editX = 0
    @Published var editY = 0
    @Published var editButton: ClickButton = .left
    @Published var editClicks = 1
    @Published var editDelay = 10

    // 실행 설정
    @Published var repeatCount = 1
    @Published var loopDelayMs = 0
    @Published var startDelayMs = 300
    @Published var restoreCursor = true
    @Published var alwaysOnTop = false {
        didSet { NSApp.windows.forEach { $0.level = alwaysOnTop ? .floating : .normal } }
    }

    // 상태
    @Published var isRunning = false
    @Published var status = "준비"
    @Published var cursorText = "커서: -"
    @Published var trusted = ClickEngine.isTrusted(prompt: false)

    private let engine = ClickEngine()
    private var cursorTimer: Timer?

    static let addShortcut = "⌃⌥A 또는 F2"
    static let toggleShortcut = "⌃⌥S 또는 F6"

    init() {
        registerHotKeys()
        cursorTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self else { return }
            let p = ClickEngine.cursorPosition()
            let text = "커서: \(Int(p.x.rounded())), \(Int(p.y.rounded()))"
            if text != self.cursorText { self.cursorText = text }
        }
        if !trusted {
            status = "손쉬운 사용 권한이 필요합니다. 위의 '권한 요청' 버튼을 누르세요."
        }
    }

    private func registerHotKeys() {
        let ctrlOpt = controlKey | optionKey
        let hk = HotKeyCenter.shared
        let okAdd = hk.register(keyCode: kVK_ANSI_A, modifiers: ctrlOpt) { [weak self] in self?.addCursorPoint() }
        let okToggle = hk.register(keyCode: kVK_ANSI_S, modifiers: ctrlOpt) { [weak self] in self?.toggleRun() }
        // 외장 키보드 등 F키를 바로 누를 수 있는 경우용
        hk.register(keyCode: kVK_F2, modifiers: 0) { [weak self] in self?.addCursorPoint() }
        hk.register(keyCode: kVK_F6, modifiers: 0) { [weak self] in self?.toggleRun() }
        if !(okAdd && okToggle) {
            status = "경고: 단축키 등록 실패 — 다른 앱이 ⌃⌥A / ⌃⌥S를 사용 중일 수 있습니다."
        }
    }

    // MARK: - 권한

    func requestAccessibility() {
        trusted = ClickEngine.isTrusted(prompt: true)
        if !trusted {
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
            status = "시스템 설정 → 개인정보 보호 및 보안 → 손쉬운 사용에서 FastClicker를 켜 주세요."
        }
    }

    func recheckAccessibility() {
        trusted = ClickEngine.isTrusted(prompt: false)
    }

    // MARK: - 실행

    func toggleRun() {
        if engine.isRunning {
            engine.stop()
            return
        }
        recheckAccessibility()
        guard trusted else {
            requestAccessibility()
            return
        }
        guard !points.isEmpty else {
            status = "클릭 지점이 없습니다. 원하는 위치에 마우스를 올리고 \(Self.addShortcut)를 누르세요."
            return
        }

        isRunning = true
        status = "실행 중... (Esc 또는 \(Self.toggleShortcut)로 정지)"
        engine.start(points: points,
                     repeatCount: repeatCount,
                     loopDelayMs: max(0, loopDelayMs),
                     restoreCursor: restoreCursor,
                     startDelayMs: max(0, startDelayMs),
                     progress: { [weak self] loop, clicks in
                         DispatchQueue.main.async {
                             self?.status = "실행 중... \(loop)회차 완료, 총 \(clicks.formatted()) 클릭"
                         }
                     },
                     finished: { [weak self] clicks, seconds, escaped in
                         DispatchQueue.main.async {
                             guard let self else { return }
                             let cps = seconds > 0 ? Double(clicks) / seconds : 0
                             let ms = Int((seconds * 1000).rounded())
                             self.status = "\(escaped ? "Esc로 중단됨" : "완료") — 총 \(clicks.formatted()) 클릭, \(ms.formatted()) ms (\(Int(cps).formatted()) 클릭/초)"
                             self.isRunning = false
                         }
                     })
    }

    // MARK: - 지점 편집

    private var editorPoint: ClickPoint {
        ClickPoint(x: editX, y: editY, button: editButton,
                   clicks: max(1, editClicks), delayMs: max(0, editDelay))
    }

    private var selectedIndex: Int? {
        guard let selection else { return nil }
        return points.firstIndex { $0.id == selection }
    }

    func index(of p: ClickPoint) -> Int {
        (points.firstIndex { $0.id == p.id } ?? 0) + 1
    }

    func addCursorPoint() {
        guard !isRunning else { return }
        let c = ClickEngine.cursorPosition()
        var p = editorPoint
        p.x = Int(c.x.rounded())
        p.y = Int(c.y.rounded())
        add(p)
    }

    func addFromEditor() { add(editorPoint) }

    private func add(_ p: ClickPoint) {
        points.append(p)
        selection = p.id
        status = "지점 #\(points.count) 추가: (\(p.x), \(p.y))"
    }

    func loadSelectedIntoEditor() {
        guard let i = selectedIndex else { return }
        let p = points[i]
        editX = p.x
        editY = p.y
        editButton = p.button
        editClicks = p.clicks
        editDelay = p.delayMs
    }

    func updateSelected() {
        guard let i = selectedIndex else { return }
        var p = editorPoint
        p.id = points[i].id
        points[i] = p
    }

    func deleteSelected() {
        guard let i = selectedIndex else { return }
        points.remove(at: i)
        selection = points.isEmpty ? nil : points[min(i, points.count - 1)].id
    }

    func moveSelected(_ dir: Int) {
        guard let i = selectedIndex else { return }
        let j = i + dir
        guard j >= 0, j < points.count else { return }
        points.swapAt(i, j)
    }

    func clearAll() {
        guard !points.isEmpty else { return }
        let alert = NSAlert()
        alert.messageText = "모든 지점을 지울까요?"
        alert.addButton(withTitle: "지우기")
        alert.addButton(withTitle: "취소")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        points.removeAll()
        selection = nil
    }

    func applyDelayToAll() {
        let d = max(0, editDelay)
        for i in points.indices { points[i].delayMs = d }
        status = "모든 지점의 대기 시간을 \(d) ms로 설정했습니다."
    }

    // MARK: - 저장 / 불러오기

    func save() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "clicks.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let profile = Profile(points: points, repeatCount: repeatCount,
                              loopDelayMs: loopDelayMs, restoreCursor: restoreCursor)
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted]
            try encoder.encode(profile).write(to: url)
            status = "저장됨: \(url.path)"
        } catch {
            showError("저장 실패: \(error.localizedDescription)")
        }
    }

    func load() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let profile = try JSONDecoder().decode(Profile.self, from: Data(contentsOf: url))
            points = profile.points
            repeatCount = profile.repeatCount
            loopDelayMs = profile.loopDelayMs
            restoreCursor = profile.restoreCursor
            selection = points.first?.id
            status = "불러옴: \(url.lastPathComponent) (\(points.count)개 지점)"
        } catch {
            showError("불러오기 실패: \(error.localizedDescription)")
        }
    }

    private func showError(_ message: String) {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = message
        alert.runModal()
    }
}
