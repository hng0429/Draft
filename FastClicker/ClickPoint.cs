namespace FastClicker;

public enum ClickButton
{
    Left,
    Right,
    Middle,
}

public sealed class ClickPoint
{
    public int X { get; set; }
    public int Y { get; set; }
    public ClickButton Button { get; set; } = ClickButton.Left;

    /// <summary>같은 위치에서 연속으로 누를 횟수 (1 = 단일 클릭, 2 = 더블클릭)</summary>
    public int Clicks { get; set; } = 1;

    /// <summary>이 지점을 클릭한 뒤 다음 지점으로 넘어가기 전 대기 시간(ms). 0이면 즉시.</summary>
    public int DelayMs { get; set; } = 10;
}

public sealed class Profile
{
    public List<ClickPoint> Points { get; set; } = new();
    public int Repeat { get; set; } = 1;
    public int LoopDelayMs { get; set; } = 0;
    public bool RestoreCursor { get; set; } = true;
}
