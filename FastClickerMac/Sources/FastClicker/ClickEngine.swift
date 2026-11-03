import ApplicationServices
import Carbon.HIToolbox
import CoreGraphics
import Foundation

/// CGEvent 기반 고속 클릭 엔진. 전용 스레드에서 실행되며,
/// 짧은 대기는 스핀으로 처리해 1ms 단위까지 정확하게 맞춘다.
final class ClickEngine {
    private let lock = NSLock()
    private var stopRequested = false
    private var running = false

    var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return running
    }

    func stop() {
        lock.lock(); stopRequested = true; lock.unlock()
    }

    private var shouldStop: Bool {
        lock.lock(); defer { lock.unlock() }
        return stopRequested
    }

    /// - progress: (완료한 회차, 총 클릭 수) — 엔진 스레드에서 호출
    /// - finished: (총 클릭 수, 경과 초, Esc로 중단 여부) — 엔진 스레드에서 호출
    func start(points: [ClickPoint],
               repeatCount: Int,
               loopDelayMs: Int,
               restoreCursor: Bool,
               startDelayMs: Int,
               progress: @escaping (Int, Int) -> Void,
               finished: @escaping (Int, Double, Bool) -> Void) {
        lock.lock()
        if running || points.isEmpty { lock.unlock(); return }
        running = true
        stopRequested = false
        lock.unlock()

        let thread = Thread { [self] in
            run(points: points, repeatCount: repeatCount, loopDelayMs: loopDelayMs,
                restoreCursor: restoreCursor, startDelayMs: startDelayMs,
                progress: progress, finished: finished)
        }
        thread.name = "ClickEngine"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    private func run(points: [ClickPoint],
                     repeatCount: Int,
                     loopDelayMs: Int,
                     restoreCursor: Bool,
                     startDelayMs: Int,
                     progress: (Int, Int) -> Void,
                     finished: (Int, Double, Bool) -> Void) {
        let original = CGEvent(source: nil)?.location ?? .zero
        var clicks = 0
        var escaped = false
        var startTime = DispatchTime.now().uptimeNanoseconds

        defer {
            if restoreCursor {
                CGWarpMouseCursorPosition(original)
                CGAssociateMouseAndMouseCursorPosition(1)
            }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - startTime) / 1_000_000_000
            lock.lock(); running = false; lock.unlock()
            finished(clicks, elapsed, escaped)
        }

        if startDelayMs > 0 && wait(ms: startDelayMs, escaped: &escaped) { return }

        guard let source = CGEventSource(stateID: .hidSystemState) else { return }
        // 합성 이벤트 직후 실제 마우스 입력이 잠시 막히는 현상 제거
        source.localEventsSuppressionInterval = 0

        // 지점마다 이벤트를 미리 만들어 둔다 (이동 + 누름/뗌 × 클릭 수)
        let prepared: [[CGEvent]] = points.map { buildEvents(for: $0, source: source) }
        let safePoints = points.map { p -> ClickPoint in
            var q = p
            q.clicks = max(1, p.clicks)
            q.delayMs = max(0, p.delayMs)
            return q
        }

        startTime = DispatchTime.now().uptimeNanoseconds
        var loop = 1
        while repeatCount <= 0 || loop <= repeatCount {
            for (i, events) in prepared.enumerated() {
                if checkStop(escaped: &escaped) { return }
                for e in events { e.post(tap: .cghidEventTap) }
                clicks += safePoints[i].clicks
                let d = safePoints[i].delayMs
                if d > 0 && wait(ms: d, escaped: &escaped) { return }
            }
            progress(loop, clicks)

            let last = repeatCount > 0 && loop >= repeatCount
            if !last && loopDelayMs > 0 && wait(ms: loopDelayMs, escaped: &escaped) { return }
            loop += 1
        }
    }

    private func checkStop(escaped: inout Bool) -> Bool {
        if shouldStop { return true }
        // 비상 정지: Esc 키
        if CGEventSource.keyState(.combinedSessionState, key: CGKeyCode(kVK_Escape)) {
            escaped = true
            stop()
            return true
        }
        return false
    }

    /// 지정 시간 대기. 중단되면 true.
    private func wait(ms: Int, escaped: inout Bool) -> Bool {
        let target = DispatchTime.now().uptimeNanoseconds + UInt64(ms) * 1_000_000
        while true {
            if checkStop(escaped: &escaped) { return true }
            let now = DispatchTime.now().uptimeNanoseconds
            if now >= target { return false }
            if target - now > 2_000_000 {
                usleep(1000)        // 긴 대기는 CPU를 쉬게 함
            }
            // 마지막 2ms는 스핀으로 정밀하게
        }
    }

    private func buildEvents(for p: ClickPoint, source: CGEventSource) -> [CGEvent] {
        let pos = CGPoint(x: p.x, y: p.y)
        let (downType, upType, button): (CGEventType, CGEventType, CGMouseButton) = {
            switch p.button {
            case .left: return (.leftMouseDown, .leftMouseUp, .left)
            case .right: return (.rightMouseDown, .rightMouseUp, .right)
            case .middle: return (.otherMouseDown, .otherMouseUp, .center)
            }
        }()

        var events: [CGEvent] = []
        if let move = CGEvent(mouseEventSource: source, mouseType: .mouseMoved,
                              mouseCursorPosition: pos, mouseButton: button) {
            events.append(move)
        }
        for n in 1...max(1, p.clicks) {
            guard let down = CGEvent(mouseEventSource: source, mouseType: downType,
                                     mouseCursorPosition: pos, mouseButton: button),
                  let up = CGEvent(mouseEventSource: source, mouseType: upType,
                                   mouseCursorPosition: pos, mouseButton: button) else { continue }
            // 클릭 상태 값(1, 2, ...)을 넣어야 더블클릭으로 인식된다
            down.setIntegerValueField(.mouseEventClickState, value: Int64(n))
            up.setIntegerValueField(.mouseEventClickState, value: Int64(n))
            events.append(down)
            events.append(up)
        }
        return events
    }

    /// 현재 커서 위치 (CG 전역 좌표: 주 모니터 왼쪽 위가 원점, 단위는 포인트)
    static func cursorPosition() -> CGPoint {
        CGEvent(source: nil)?.location ?? .zero
    }

    /// 손쉬운 사용 권한 확인. prompt가 true면 시스템 권한 요청 창을 띄운다.
    static func isTrusted(prompt: Bool) -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        return AXIsProcessTrustedWithOptions([key: prompt] as CFDictionary)
    }
}
