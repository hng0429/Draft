import SwiftUI

private let intFormat = IntegerFormatStyle<Int>().grouping(.never)

struct ContentView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !model.trusted {
                permissionBanner
            }

            Text("\(AppModel.addShortcut): 현재 마우스 위치 추가   ·   \(AppModel.toggleShortcut): 시작/정지   ·   Esc: 비상 정지")
                .font(.callout)
                .foregroundStyle(.secondary)

            Table(model.points, selection: $model.selection) {
                TableColumn("#") { p in Text("\(model.index(of: p))") }.width(36)
                TableColumn("X") { p in Text(String(p.x)) }.width(70)
                TableColumn("Y") { p in Text(String(p.y)) }.width(70)
                TableColumn("버튼") { p in Text(p.button.label) }.width(60)
                TableColumn("횟수") { p in Text(String(p.clicks)) }.width(45)
                TableColumn("대기(ms)") { p in Text(String(p.delayMs)) }
            }
            .frame(minHeight: 180)
            .disabled(model.isRunning)
            .onDeleteCommand { model.deleteSelected() }
            .onChange(of: model.selection) { _ in model.loadSelectedIntoEditor() }

            GroupBox("클릭 지점") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        field("X", $model.editX)
                        field("Y", $model.editY)
                        Picker("버튼", selection: $model.editButton) {
                            ForEach(ClickButton.allCases) { Text($0.label).tag($0) }
                        }
                        .frame(width: 140)
                        field("횟수", $model.editClicks, width: 50)
                        field("후 대기(ms)", $model.editDelay)
                    }
                    HStack {
                        Button("추가") { model.addFromEditor() }
                        Button("선택 항목 수정") { model.updateSelected() }
                        Button("삭제") { model.deleteSelected() }
                        Button("▲") { model.moveSelected(-1) }
                        Button("▼") { model.moveSelected(1) }
                        Button("모두 지우기") { model.clearAll() }
                        Button("대기 일괄 적용") { model.applyDelayToAll() }
                    }
                }
                .padding(4)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .disabled(model.isRunning)

            GroupBox("실행 설정") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        field("반복 횟수 (0=무한)", $model.repeatCount)
                        field("한 바퀴 후 대기(ms)", $model.loopDelayMs)
                        field("시작 지연(ms)", $model.startDelayMs)
                    }
                    HStack {
                        Toggle("끝나면 커서 원위치", isOn: $model.restoreCursor)
                        Toggle("항상 위", isOn: $model.alwaysOnTop)
                        Spacer().frame(width: 16)
                        Button("저장...") { model.save() }
                        Button("불러오기...") { model.load() }
                    }
                }
                .padding(4)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .disabled(model.isRunning)

            HStack(spacing: 16) {
                Button {
                    model.toggleRun()
                } label: {
                    Text(model.isRunning ? "■ 정지 (Esc)" : "▶ 시작")
                        .font(.headline)
                        .frame(minWidth: 120)
                }
                .controlSize(.large)
                .keyboardShortcut(.defaultAction)

                Text(model.cursorText)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            Divider()
            Text(model.status)
                .font(.callout)
                .lineLimit(2)
        }
        .padding(14)
        .frame(minWidth: 680, minHeight: 560)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            model.recheckAccessibility()
        }
    }

    private var permissionBanner: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.yellow)
            Text("클릭하려면 손쉬운 사용 권한이 필요합니다.")
            Spacer()
            Button("권한 요청") { model.requestAccessibility() }
        }
        .padding(8)
        .background(Color.yellow.opacity(0.15), in: RoundedRectangle(cornerRadius: 6))
    }

    private func field(_ title: String, _ value: Binding<Int>, width: CGFloat = 70) -> some View {
        HStack(spacing: 4) {
            Text(title)
            TextField(title, value: value, format: intFormat)
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .frame(width: width)
        }
    }
}
