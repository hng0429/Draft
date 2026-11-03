using System.Diagnostics;
using System.Runtime.InteropServices;
using static FastClicker.NativeMethods;

namespace FastClicker;

/// <summary>
/// SendInput 기반 고속 클릭 엔진. 전용 스레드에서 실행되며,
/// 짧은 대기는 Stopwatch 스핀으로 처리해 1ms 미만 단위까지 정확하게 맞춘다.
/// </summary>
internal sealed class ClickEngine
{
    private static readonly int InputSize = Marshal.SizeOf<INPUT>();

    private volatile bool _stop;
    private Thread? _thread;

    public bool IsRunning => _thread is { IsAlive: true };

    /// <summary>진행 상황 (반복 회차, 총 클릭 수). 엔진 스레드에서 호출됨.</summary>
    public event Action<int, long>? Progress;

    /// <summary>종료 시 호출 (총 클릭 수, 경과 시간, Esc로 중단되었는지). 엔진 스레드에서 호출됨.</summary>
    public event Action<long, TimeSpan, bool>? Finished;

    public void Start(IReadOnlyList<ClickPoint> points, int repeat, int loopDelayMs, bool restoreCursor, int startDelayMs)
    {
        if (IsRunning || points.Count == 0) return;

        // 실행 도중 UI에서 목록을 수정해도 안전하도록 복사본 사용
        var snapshot = points.Select(p => new ClickPoint
        {
            X = p.X, Y = p.Y, Button = p.Button,
            Clicks = Math.Max(1, p.Clicks), DelayMs = Math.Max(0, p.DelayMs),
        }).ToArray();

        _stop = false;
        _thread = new Thread(() => Run(snapshot, repeat, loopDelayMs, restoreCursor, startDelayMs))
        {
            IsBackground = true,
            Priority = ThreadPriority.Highest,
            Name = "ClickEngine",
        };
        _thread.Start();
    }

    public void Stop() => _stop = true;

    private void Run(ClickPoint[] points, int repeat, int loopDelayMs, bool restoreCursor, int startDelayMs)
    {
        GetCursorPos(out var original);
        timeBeginPeriod(1);
        var sw = Stopwatch.StartNew();
        long clicks = 0;
        bool escaped = false;

        try
        {
            if (startDelayMs > 0 && Wait(startDelayMs, ref escaped)) return;

            // 가상 데스크톱(다중 모니터 전체) 좌표계
            int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
            int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
            int vw = Math.Max(2, GetSystemMetrics(SM_CXVIRTUALSCREEN));
            int vh = Math.Max(2, GetSystemMetrics(SM_CYVIRTUALSCREEN));

            // 지점마다 SendInput 배열을 미리 만들어 둔다 (이동 + 다운/업 × 클릭 수)
            var prepared = new INPUT[points.Length][];
            for (int i = 0; i < points.Length; i++)
                prepared[i] = BuildInputs(points[i], vx, vy, vw, vh);

            sw.Restart();
            for (int loop = 1; repeat <= 0 || loop <= repeat; loop++)
            {
                for (int i = 0; i < points.Length; i++)
                {
                    if (CheckStop(ref escaped)) return;

                    var inputs = prepared[i];
                    SendInput((uint)inputs.Length, inputs, InputSize);
                    clicks += points[i].Clicks;

                    if (points[i].DelayMs > 0 && Wait(points[i].DelayMs, ref escaped)) return;
                }

                Progress?.Invoke(loop, clicks);

                bool last = repeat > 0 && loop >= repeat;
                if (!last && loopDelayMs > 0 && Wait(loopDelayMs, ref escaped)) return;
            }
        }
        finally
        {
            timeEndPeriod(1);
            sw.Stop();
            if (restoreCursor) SetCursorPos(original.X, original.Y);
            Finished?.Invoke(clicks, sw.Elapsed, escaped);
        }
    }

    private bool CheckStop(ref bool escaped)
    {
        if (_stop) return true;
        // 비상 정지: Esc 키
        if ((GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0)
        {
            escaped = true;
            _stop = true;
            return true;
        }
        return false;
    }

    /// <summary>지정 시간 대기. 중단되면 true.</summary>
    private bool Wait(int ms, ref bool escaped)
    {
        long target = Stopwatch.GetTimestamp() + (long)(ms * (Stopwatch.Frequency / 1000.0));
        while (true)
        {
            if (CheckStop(ref escaped)) return true;
            long remaining = target - Stopwatch.GetTimestamp();
            if (remaining <= 0) return false;

            double remainingMs = remaining * 1000.0 / Stopwatch.Frequency;
            if (remainingMs > 2.0)
                Thread.Sleep(1);       // 긴 대기는 CPU를 쉬게 함
            else
                Thread.SpinWait(50);   // 마지막 2ms는 스핀으로 정밀하게
        }
    }

    private static INPUT[] BuildInputs(ClickPoint p, int vx, int vy, int vw, int vh)
    {
        // 절대 좌표를 0..65535 범위로 정규화
        int nx = (int)Math.Round((p.X - vx) * 65535.0 / (vw - 1));
        int ny = (int)Math.Round((p.Y - vy) * 65535.0 / (vh - 1));

        (uint down, uint up) = p.Button switch
        {
            ClickButton.Right => (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
            ClickButton.Middle => (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
            _ => (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        };

        const uint abs = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
        int clicks = Math.Max(1, p.Clicks);
        var list = new INPUT[1 + clicks * 2];

        list[0] = Mouse(nx, ny, MOUSEEVENTF_MOVE | abs);
        for (int c = 0; c < clicks; c++)
        {
            list[1 + c * 2] = Mouse(nx, ny, down | abs);
            list[2 + c * 2] = Mouse(nx, ny, up | abs);
        }
        return list;
    }

    private static INPUT Mouse(int dx, int dy, uint flags) => new()
    {
        type = INPUT_MOUSE,
        mi = new MOUSEINPUT { dx = dx, dy = dy, dwFlags = flags },
    };
}
