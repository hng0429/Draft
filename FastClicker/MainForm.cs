using System.Text.Json;
using System.Text.Json.Serialization;
using static FastClicker.NativeMethods;

namespace FastClicker;

public sealed class MainForm : Form
{
    private const int HotkeyAdd = 1;
    private const int HotkeyToggle = 2;
    private const Keys AddKey = Keys.F2;
    private const Keys ToggleKey = Keys.F6;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() },
    };

    private readonly List<ClickPoint> _points = new();
    private readonly ClickEngine _engine = new();

    // 목록
    private readonly ListView _list = new()
    {
        View = View.Details,
        FullRowSelect = true,
        GridLines = true,
        HideSelection = false,
        MultiSelect = false,
        Dock = DockStyle.Fill,
    };

    // 지점 편집
    private readonly NumericUpDown _x = Num(-100000, 100000, 0);
    private readonly NumericUpDown _y = Num(-100000, 100000, 0);
    private readonly ComboBox _button = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 90 };
    private readonly NumericUpDown _clicks = Num(1, 100, 1);
    private readonly NumericUpDown _delay = Num(0, 3_600_000, 10);

    // 실행 설정
    private readonly NumericUpDown _repeat = Num(0, 100_000_000, 1);
    private readonly NumericUpDown _loopDelay = Num(0, 3_600_000, 0);
    private readonly NumericUpDown _startDelay = Num(0, 60_000, 300);
    private readonly CheckBox _restore = new() { Text = "끝나면 커서 원위치", Checked = true, AutoSize = true };
    private readonly CheckBox _topMost = new() { Text = "항상 위", AutoSize = true };

    private readonly Button _startBtn = new() { Text = "▶ 시작 (F6)", AutoSize = true, Height = 36 };
    private readonly Label _cursorLabel = new() { AutoSize = true, Text = "커서: -" };
    private readonly ToolStripStatusLabel _status = new() { Text = "준비", Spring = true, TextAlign = ContentAlignment.MiddleLeft };
    private readonly System.Windows.Forms.Timer _cursorTimer = new() { Interval = 50 };

    public MainForm()
    {
        Text = "FastClicker - 순차 고속 클릭";
        StartPosition = FormStartPosition.CenterScreen;
        MinimumSize = new Size(640, 520);
        Size = new Size(720, 600);
        Font = new Font("Malgun Gothic", 9F);

        _button.Items.AddRange(new object[] { "왼쪽", "오른쪽", "가운데" });
        _button.SelectedIndex = 0;

        BuildLayout();
        WireEvents();
        RefreshList();
    }

    // ───────────────────────────── UI 구성 ─────────────────────────────

    private void BuildLayout()
    {
        _list.Columns.Add("#", 40);
        _list.Columns.Add("X", 80);
        _list.Columns.Add("Y", 80);
        _list.Columns.Add("버튼", 70);
        _list.Columns.Add("횟수", 55);
        _list.Columns.Add("대기(ms)", 80);

        var help = new Label
        {
            AutoSize = true,
            ForeColor = SystemColors.GrayText,
            Text = $"{AddKey}: 현재 마우스 위치를 지점으로 추가   ·   {ToggleKey}: 시작/정지   ·   Esc: 실행 중 비상 정지",
            Padding = new Padding(0, 4, 0, 4),
        };

        // 지점 편집 그룹
        var editFlow = Flow();
        editFlow.Controls.AddRange(new Control[]
        {
            Caption("X"), _x, Caption("Y"), _y, Caption("버튼"), _button,
            Caption("횟수"), _clicks, Caption("후 대기(ms)"), _delay,
        });
        var editBtns = Flow();
        editBtns.Controls.AddRange(new Control[]
        {
            Btn("추가", (_, _) => AddPoint(ReadEditor())),
            Btn("선택 항목 수정", (_, _) => UpdateSelected()),
            Btn("삭제", (_, _) => DeleteSelected()),
            Btn("▲", (_, _) => MoveSelected(-1)),
            Btn("▼", (_, _) => MoveSelected(+1)),
            Btn("모두 지우기", (_, _) => ClearAll()),
            Btn("대기 일괄 적용", (_, _) => ApplyDelayToAll()),
        });
        var editGroup = Group("클릭 지점", editFlow, editBtns);

        // 실행 설정 그룹
        var runFlow = Flow();
        runFlow.Controls.AddRange(new Control[]
        {
            Caption("반복 횟수 (0=무한)"), _repeat,
            Caption("한 바퀴 후 대기(ms)"), _loopDelay,
            Caption("시작 지연(ms)"), _startDelay,
        });
        var runFlow2 = Flow();
        runFlow2.Controls.AddRange(new Control[]
        {
            _restore, _topMost,
            Btn("저장...", (_, _) => SaveProfile()),
            Btn("불러오기...", (_, _) => LoadProfile()),
        });
        var runGroup = Group("실행 설정", runFlow, runFlow2);

        var bottom = Flow();
        _startBtn.Font = new Font(Font.FontFamily, 10F, FontStyle.Bold);
        _startBtn.Padding = new Padding(12, 2, 12, 2);
        bottom.Controls.AddRange(new Control[] { _startBtn, _cursorLabel });
        _cursorLabel.Margin = new Padding(16, 12, 0, 0);

        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            Padding = new Padding(10),
        };
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.Controls.Add(help, 0, 0);
        root.Controls.Add(_list, 0, 1);
        root.Controls.Add(editGroup, 0, 2);
        root.Controls.Add(runGroup, 0, 3);
        root.Controls.Add(bottom, 0, 4);

        var statusStrip = new StatusStrip();
        statusStrip.Items.Add(_status);

        Controls.Add(root);
        Controls.Add(statusStrip);
    }

    private void WireEvents()
    {
        _startBtn.Click += (_, _) => ToggleRun();
        _topMost.CheckedChanged += (_, _) => TopMost = _topMost.Checked;
        _list.SelectedIndexChanged += (_, _) => LoadSelectedIntoEditor();
        _list.KeyDown += (_, e) =>
        {
            if (e.KeyCode == Keys.Delete) DeleteSelected();
        };

        _cursorTimer.Tick += (_, _) =>
        {
            if (GetCursorPos(out var p)) _cursorLabel.Text = $"커서: {p.X}, {p.Y}";
        };
        _cursorTimer.Start();

        _engine.Progress += (loop, clicks) =>
            BeginInvokeSafe(() => _status.Text = $"실행 중... {loop}회차 완료, 총 {clicks:N0} 클릭");
        _engine.Finished += (clicks, elapsed, escaped) => BeginInvokeSafe(() =>
        {
            double cps = elapsed.TotalSeconds > 0 ? clicks / elapsed.TotalSeconds : 0;
            _status.Text = $"{(escaped ? "Esc로 중단됨" : "완료")} — 총 {clicks:N0} 클릭, {elapsed.TotalMilliseconds:N0} ms ({cps:N0} 클릭/초)";
            SetRunningUi(false);
        });
    }

    // ───────────────────────────── 전역 단축키 ─────────────────────────────

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        bool ok = RegisterHotKey(Handle, HotkeyAdd, MOD_NOREPEAT, (uint)AddKey)
                & RegisterHotKey(Handle, HotkeyToggle, MOD_NOREPEAT, (uint)ToggleKey);
        if (!ok)
            _status.Text = $"경고: 단축키({AddKey}/{ToggleKey}) 등록 실패 — 다른 프로그램이 사용 중일 수 있습니다.";
    }

    protected override void OnHandleDestroyed(EventArgs e)
    {
        UnregisterHotKey(Handle, HotkeyAdd);
        UnregisterHotKey(Handle, HotkeyToggle);
        base.OnHandleDestroyed(e);
    }

    protected override void WndProc(ref Message m)
    {
        if (m.Msg == WM_HOTKEY)
        {
            switch (m.WParam.ToInt32())
            {
                case HotkeyAdd:
                    if (!_engine.IsRunning && GetCursorPos(out var p))
                    {
                        var point = ReadEditor();
                        point.X = p.X;
                        point.Y = p.Y;
                        AddPoint(point);
                    }
                    break;
                case HotkeyToggle:
                    ToggleRun();
                    break;
            }
            return;
        }
        base.WndProc(ref m);
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        _engine.Stop();
        _cursorTimer.Stop();
        base.OnFormClosing(e);
    }

    // ───────────────────────────── 실행 ─────────────────────────────

    private void ToggleRun()
    {
        if (_engine.IsRunning)
        {
            _engine.Stop();
            return;
        }
        if (_points.Count == 0)
        {
            _status.Text = $"클릭 지점이 없습니다. 원하는 위치에 마우스를 올리고 {AddKey}를 누르세요.";
            return;
        }

        SetRunningUi(true);
        _status.Text = "실행 중... (Esc 또는 F6으로 정지)";
        _engine.Start(_points, (int)_repeat.Value, (int)_loopDelay.Value, _restore.Checked, (int)_startDelay.Value);
    }

    private void SetRunningUi(bool running)
    {
        _startBtn.Text = running ? "■ 정지 (F6 / Esc)" : "▶ 시작 (F6)";
        _list.Enabled = !running;
    }

    // ───────────────────────────── 지점 편집 ─────────────────────────────

    private ClickPoint ReadEditor() => new()
    {
        X = (int)_x.Value,
        Y = (int)_y.Value,
        Button = (ClickButton)Math.Max(0, _button.SelectedIndex),
        Clicks = (int)_clicks.Value,
        DelayMs = (int)_delay.Value,
    };

    private void LoadSelectedIntoEditor()
    {
        int i = SelectedIndex;
        if (i < 0) return;
        var p = _points[i];
        _x.Value = Clamp(_x, p.X);
        _y.Value = Clamp(_y, p.Y);
        _button.SelectedIndex = (int)p.Button;
        _clicks.Value = Clamp(_clicks, p.Clicks);
        _delay.Value = Clamp(_delay, p.DelayMs);
    }

    private void AddPoint(ClickPoint p)
    {
        _points.Add(p);
        RefreshList(_points.Count - 1);
        _status.Text = $"지점 #{_points.Count} 추가: ({p.X}, {p.Y})";
    }

    private void UpdateSelected()
    {
        int i = SelectedIndex;
        if (i < 0) return;
        _points[i] = ReadEditor();
        RefreshList(i);
    }

    private void DeleteSelected()
    {
        int i = SelectedIndex;
        if (i < 0) return;
        _points.RemoveAt(i);
        RefreshList(Math.Min(i, _points.Count - 1));
    }

    private void MoveSelected(int dir)
    {
        int i = SelectedIndex;
        int j = i + dir;
        if (i < 0 || j < 0 || j >= _points.Count) return;
        (_points[i], _points[j]) = (_points[j], _points[i]);
        RefreshList(j);
    }

    private void ClearAll()
    {
        if (_points.Count == 0) return;
        if (MessageBox.Show(this, "모든 지점을 지울까요?", Text, MessageBoxButtons.YesNo) != DialogResult.Yes) return;
        _points.Clear();
        RefreshList();
    }

    private void ApplyDelayToAll()
    {
        int d = (int)_delay.Value;
        foreach (var p in _points) p.DelayMs = d;
        RefreshList(SelectedIndex);
        _status.Text = $"모든 지점의 대기 시간을 {d} ms로 설정했습니다.";
    }

    private int SelectedIndex => _list.SelectedIndices.Count > 0 ? _list.SelectedIndices[0] : -1;

    private void RefreshList(int select = -1)
    {
        _list.BeginUpdate();
        _list.Items.Clear();
        for (int i = 0; i < _points.Count; i++)
        {
            var p = _points[i];
            _list.Items.Add(new ListViewItem(new[]
            {
                (i + 1).ToString(), p.X.ToString(), p.Y.ToString(),
                ButtonName(p.Button), p.Clicks.ToString(), p.DelayMs.ToString(),
            }));
        }
        if (select >= 0 && select < _list.Items.Count)
        {
            _list.Items[select].Selected = true;
            _list.EnsureVisible(select);
        }
        _list.EndUpdate();
    }

    // ───────────────────────────── 저장 / 불러오기 ─────────────────────────────

    private void SaveProfile()
    {
        using var dlg = new SaveFileDialog { Filter = "클릭 프로필 (*.json)|*.json", FileName = "clicks.json" };
        if (dlg.ShowDialog(this) != DialogResult.OK) return;
        var profile = new Profile
        {
            Points = _points,
            Repeat = (int)_repeat.Value,
            LoopDelayMs = (int)_loopDelay.Value,
            RestoreCursor = _restore.Checked,
        };
        File.WriteAllText(dlg.FileName, JsonSerializer.Serialize(profile, JsonOptions));
        _status.Text = $"저장됨: {dlg.FileName}";
    }

    private void LoadProfile()
    {
        using var dlg = new OpenFileDialog { Filter = "클릭 프로필 (*.json)|*.json|모든 파일|*.*" };
        if (dlg.ShowDialog(this) != DialogResult.OK) return;
        try
        {
            var profile = JsonSerializer.Deserialize<Profile>(File.ReadAllText(dlg.FileName), JsonOptions)
                          ?? throw new InvalidDataException("빈 파일");
            _points.Clear();
            _points.AddRange(profile.Points);
            _repeat.Value = Clamp(_repeat, profile.Repeat);
            _loopDelay.Value = Clamp(_loopDelay, profile.LoopDelayMs);
            _restore.Checked = profile.RestoreCursor;
            RefreshList(_points.Count > 0 ? 0 : -1);
            _status.Text = $"불러옴: {dlg.FileName} ({_points.Count}개 지점)";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"불러오기 실패: {ex.Message}", Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    // ───────────────────────────── 헬퍼 ─────────────────────────────

    private void BeginInvokeSafe(Action a)
    {
        if (IsDisposed || !IsHandleCreated) return;
        try { BeginInvoke(a); } catch (InvalidOperationException) { }
    }

    private static string ButtonName(ClickButton b) => b switch
    {
        ClickButton.Right => "오른쪽",
        ClickButton.Middle => "가운데",
        _ => "왼쪽",
    };

    private static decimal Clamp(NumericUpDown n, int v) => Math.Min(n.Maximum, Math.Max(n.Minimum, v));

    private static NumericUpDown Num(int min, int max, int value) => new()
    {
        Minimum = min,
        Maximum = max,
        Value = value,
        Width = 80,
        ThousandsSeparator = false,
    };

    private static Label Caption(string text) => new()
    {
        Text = text,
        AutoSize = true,
        Margin = new Padding(8, 7, 2, 0),
    };

    private static Button Btn(string text, EventHandler onClick)
    {
        var b = new Button { Text = text, AutoSize = true };
        b.Click += onClick;
        return b;
    }

    private static FlowLayoutPanel Flow() => new()
    {
        AutoSize = true,
        AutoSizeMode = AutoSizeMode.GrowAndShrink,
        Dock = DockStyle.Fill,
        WrapContents = true,
        Margin = new Padding(0),
    };

    private static GroupBox Group(string title, params Control[] rows)
    {
        var table = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoSize = true,
            ColumnCount = 1,
        };
        foreach (var r in rows) table.Controls.Add(r);

        var g = new GroupBox
        {
            Text = title,
            Dock = DockStyle.Fill,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Padding = new Padding(6),
        };
        g.Controls.Add(table);
        return g;
    }
}
