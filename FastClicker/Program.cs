namespace FastClicker;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        // 모니터별 DPI 인식: 화면 배율(125%, 150% 등)과 무관하게 실제 픽셀 좌표로 클릭하기 위해 필요
        Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new MainForm());
    }
}
