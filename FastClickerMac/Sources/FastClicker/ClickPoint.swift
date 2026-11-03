import Foundation

enum ClickButton: String, Codable, CaseIterable, Identifiable {
    case left = "Left"
    case right = "Right"
    case middle = "Middle"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .left: return "왼쪽"
        case .right: return "오른쪽"
        case .middle: return "가운데"
        }
    }
}

/// 클릭 지점. JSON 키 이름은 Windows 버전과 같아서 프로필 파일을 서로 열 수 있다.
struct ClickPoint: Codable, Identifiable, Equatable {
    var id = UUID()
    var x: Int
    var y: Int
    var button: ClickButton = .left
    /// 같은 위치에서 연속으로 누를 횟수 (1 = 단일 클릭, 2 = 더블클릭)
    var clicks: Int = 1
    /// 클릭 후 다음 지점으로 넘어가기 전 대기(ms). 0이면 즉시.
    var delayMs: Int = 10

    enum CodingKeys: String, CodingKey {
        case x = "X", y = "Y", button = "Button", clicks = "Clicks", delayMs = "DelayMs"
    }

    init(x: Int, y: Int, button: ClickButton = .left, clicks: Int = 1, delayMs: Int = 10) {
        self.x = x
        self.y = y
        self.button = button
        self.clicks = clicks
        self.delayMs = delayMs
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        x = try c.decode(Int.self, forKey: .x)
        y = try c.decode(Int.self, forKey: .y)
        button = try c.decodeIfPresent(ClickButton.self, forKey: .button) ?? .left
        clicks = try c.decodeIfPresent(Int.self, forKey: .clicks) ?? 1
        delayMs = try c.decodeIfPresent(Int.self, forKey: .delayMs) ?? 10
    }
}

struct Profile: Codable {
    var points: [ClickPoint] = []
    var repeatCount: Int = 1
    var loopDelayMs: Int = 0
    var restoreCursor: Bool = true

    enum CodingKeys: String, CodingKey {
        case points = "Points", repeatCount = "Repeat", loopDelayMs = "LoopDelayMs", restoreCursor = "RestoreCursor"
    }

    init(points: [ClickPoint], repeatCount: Int, loopDelayMs: Int, restoreCursor: Bool) {
        self.points = points
        self.repeatCount = repeatCount
        self.loopDelayMs = loopDelayMs
        self.restoreCursor = restoreCursor
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        points = try c.decodeIfPresent([ClickPoint].self, forKey: .points) ?? []
        repeatCount = try c.decodeIfPresent(Int.self, forKey: .repeatCount) ?? 1
        loopDelayMs = try c.decodeIfPresent(Int.self, forKey: .loopDelayMs) ?? 0
        restoreCursor = try c.decodeIfPresent(Bool.self, forKey: .restoreCursor) ?? true
    }
}
