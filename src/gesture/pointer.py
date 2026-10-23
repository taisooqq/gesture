"""인식된 제스처를 마우스와 브라우저 단축키로 보냅니다."""

import os
import subprocess
import sys
import threading
import time

_KEYS = {"back": "[", "forward": "]", "refresh": "r"}
_left_held = False
_browser_lock = threading.Lock()
_drive_lock = threading.Lock()
_drive_enabled = False


if sys.platform == "darwin":
    from ctypes import CDLL, c_bool, c_double, c_int32, c_int64, c_uint32, c_void_p, Structure

    _core = CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    _access = CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
    _foundation = CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

    _LEFT_DOWN = 1
    _LEFT_UP = 2
    _RIGHT_DOWN = 3
    _RIGHT_UP = 4
    _LEFT = 0
    _RIGHT = 1
    _HID = 0
    _CLICK_STATE = 1

    class _Point(Structure):
        _fields_ = [("x", c_double), ("y", c_double)]

    class _Rect(Structure):
        _fields_ = [("x", c_double), ("y", c_double), ("width", c_double), ("height", c_double)]

    _core.CGEventCreate.restype = c_void_p
    _core.CGEventCreate.argtypes = [c_void_p]
    _core.CGEventGetLocation.restype = _Point
    _core.CGEventGetLocation.argtypes = [c_void_p]
    _core.CGEventCreateMouseEvent.restype = c_void_p
    _core.CGEventCreateMouseEvent.argtypes = [c_void_p, c_uint32, _Point, c_uint32]
    _core.CGEventSetIntegerValueField.argtypes = [c_void_p, c_uint32, c_int64]
    _core.CGEventPost.argtypes = [c_uint32, c_void_p]
    _foundation.CFRelease.argtypes = [c_void_p]
    _access.AXIsProcessTrusted.restype = c_bool
    _core.CGMainDisplayID.restype = c_uint32
    _core.CGDisplayBounds.restype = _Rect
    _core.CGDisplayBounds.argtypes = [c_uint32]
    _core.CGWarpMouseCursorPosition.argtypes = [_Point]
    _core.CGWarpMouseCursorPosition.restype = c_int32
    _core.CGAssociateMouseAndMouseCursorPosition.argtypes = [c_bool]
    _core.CGAssociateMouseAndMouseCursorPosition.restype = c_int32

    def _trusted():
        return bool(_access.AXIsProcessTrusted())

    def _cursor():
        event = _core.CGEventCreate(None)
        point = _core.CGEventGetLocation(event)
        if event:
            _foundation.CFRelease(event)
        return point.x, point.y

    def _warp(rel_x, rel_y):
        bounds = _core.CGDisplayBounds(_core.CGMainDisplayID())
        point = _Point(
            bounds.x + float(rel_x) * bounds.width,
            bounds.y + float(rel_y) * bounds.height,
        )
        error = _core.CGWarpMouseCursorPosition(point)
        return error == 0

    def _associate(enabled):
        _core.CGAssociateMouseAndMouseCursorPosition(bool(enabled))

    # 맥은 몇 번째 클릭인지(clicks)를 이벤트에 직접 적어야 더블클릭으로 본다.
    def _post(point, event_type, button, clicks=1):
        event = _core.CGEventCreateMouseEvent(None, event_type, point, button)
        if not event:
            return False
        _core.CGEventSetIntegerValueField(event, _CLICK_STATE, int(clicks))
        _core.CGEventPost(_HID, event)
        _foundation.CFRelease(event)
        return True

    def _click(kind):
        point = _core.CGEventGetLocation(_core.CGEventCreate(None))
        if kind == "left_down":
            return _post(point, _LEFT_DOWN, _LEFT)
        if kind == "left_up":
            return _post(point, _LEFT_UP, _LEFT)
        if kind == "double":
            return all(
                (
                    _post(point, _LEFT_DOWN, _LEFT, 1),
                    _post(point, _LEFT_UP, _LEFT, 1),
                    _post(point, _LEFT_DOWN, _LEFT, 2),
                    _post(point, _LEFT_UP, _LEFT, 2),
                )
            )
        return _post(point, _RIGHT_DOWN, _RIGHT) and _post(point, _RIGHT_UP, _RIGHT)

    _core.CGEventCreateScrollWheelEvent.restype = c_void_p
    _core.CGEventCreateScrollWheelEvent.argtypes = [c_void_p, c_uint32, c_uint32, c_int32]

    def _scroll(steps):
        # 양수는 위로, 음수는 아래로. 단위는 휠 한 칸이다.
        event = _core.CGEventCreateScrollWheelEvent(None, 1, 1, int(steps))
        if not event:
            return False
        _core.CGEventPost(_HID, event)
        _foundation.CFRelease(event)
        return True

    def _send_browser(action):
        subprocess.run(
            ["osascript", "-e", _BROWSER_SCRIPT, action],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )

    def _send_screenshot():
        # 단축키는 영역 선택 창이 뜰 수 있다. 메인 화면 전체를 바탕화면에 바로 저장한다.
        folder = os.path.expanduser("~/Desktop")
        stamp = time.strftime("%Y-%m-%d %H.%M.%S")
        path = os.path.join(folder, f"Screen Shot {stamp}.png")
        subprocess.run(["screencapture", "-m", path], check=False, timeout=8)

    _BROWSER_SCRIPT = r'''
on run argv
  set actionName to item 1 of argv
  set browserNames to {"Google Chrome", "Safari", "Microsoft Edge", "Arc", "Whale", "Naver Whale", "Brave Browser", "Firefox", "Opera", "Chromium"}
  tell application "System Events"
    set frontName to name of first application process whose frontmost is true
    set targetName to ""
    if browserNames contains frontName then
      set targetName to frontName
    else
      repeat with browserName in browserNames
        if exists process (browserName as text) then
          set targetName to browserName as text
          exit repeat
        end if
      end repeat
    end if
    if targetName is "" then
      return "no-browser"
    end if
    tell process targetName
      set frontmost to true
    end tell
    if actionName is "back" then
      keystroke "[" using command down
    else if actionName is "forward" then
      keystroke "]" using command down
    else if actionName is "refresh" then
      -- 글자 r 은 Option 이 붙어 읽기 모드(Command+Option+R)가 열린다. R 키 코드만 보낸다.
      key code 15 using {command down}
    end if
  end tell
  return "ok"
end run
'''

elif sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _INPUT_MOUSE = 0
    _INPUT_KEYBOARD = 1
    _MOUSE_LEFT_DOWN = 0x0002
    _MOUSE_LEFT_UP = 0x0004
    _MOUSE_RIGHT_DOWN = 0x0008
    _MOUSE_RIGHT_UP = 0x0010
    _MOUSE_WHEEL = 0x0800
    _WHEEL_DELTA = 120
    _KEY_UP = 0x0002
    _MONITOR_PRIMARY = 1
    _SW_RESTORE = 9
    _PROCESS_QUERY = 0x1000
    _ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
    _BROWSERS = {
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
        "brave.exe",
        "whale.exe",
        "opera.exe",
        "chromium.exe",
    }
    # 뒤로·앞으로는 브라우저 전용 키(VK_BROWSER_BACK/FORWARD). Alt+화살표는 유튜브 5초 이동과 겹친다.
    # 새로고침은 Ctrl+R.
    _WIN_BROWSER = {"back": 0xA6, "forward": 0xA7}
    _WIN_REFRESH = (0x11, 0x52)

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        ]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        ]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUT(ctypes.Structure):
        class _U(ctypes.Union):
            _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]

        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    class _RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class _MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", _RECT),
            ("rcWork", _RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    _user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    _user32.SetCursorPos.restype = wintypes.BOOL
    _user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
    _user32.SendInput.restype = wintypes.UINT
    _user32.GetDesktopWindow.restype = wintypes.HWND
    _user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.MonitorFromWindow.restype = wintypes.HMONITOR
    _user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MONITORINFO)]
    _user32.GetMonitorInfoW.restype = wintypes.BOOL
    _user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    _user32.GetSystemMetrics.restype = ctypes.c_int
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.restype = wintypes.BOOL
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.ShowWindow.restype = wintypes.BOOL
    _user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    _user32.SetForegroundWindow.restype = wintypes.BOOL
    _user32.BringWindowToTop.argtypes = [wintypes.HWND]
    _user32.BringWindowToTop.restype = wintypes.BOOL
    _user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    _user32.AttachThreadInput.restype = wintypes.BOOL
    _kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL

    def _trusted():
        # 윈도우는 손쉬운 사용 권한 없이 마우스 입력을 보낼 수 있다.
        return True

    def _cursor():
        class _Point(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        point = _Point()
        _user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y

    def _primary_bounds():
        monitor = _user32.MonitorFromWindow(_user32.GetDesktopWindow(), _MONITOR_PRIMARY)
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if _user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            rect = info.rcMonitor
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        return 0, 0, _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)

    def _warp(rel_x, rel_y):
        left, top, width, height = _primary_bounds()
        if width <= 0 or height <= 0:
            return False
        x = int(left + float(rel_x) * width)
        y = int(top + float(rel_y) * height)
        return bool(_user32.SetCursorPos(x, y))

    def _associate(_enabled):
        return None

    def _send_input(entry):
        return _user32.SendInput(1, ctypes.byref(entry), ctypes.sizeof(entry)) == 1

    def _click(kind):
        flags = {
            "left_down": _MOUSE_LEFT_DOWN,
            "left_up": _MOUSE_LEFT_UP,
            "right_down": _MOUSE_RIGHT_DOWN,
            "right_up": _MOUSE_RIGHT_UP,
        }
        if kind == "right":
            return _click("right_down") and _click("right_up")
        # 윈도우는 같은 자리에서 0.5초 안에 두 번 누르면 스스로 더블클릭으로 본다.
        if kind == "double":
            return all(_click(step) for step in ("left_down", "left_up", "left_down", "left_up"))
        entry = _INPUT(type=_INPUT_MOUSE)
        entry.mi.dwFlags = flags[kind]
        return _send_input(entry)

    def _scroll(steps):
        entry = _INPUT(type=_INPUT_MOUSE)
        entry.mi.dwFlags = _MOUSE_WHEEL
        entry.mi.mouseData = (int(steps) * _WHEEL_DELTA) & 0xFFFFFFFF
        return _send_input(entry)

    def _exe_name(hwnd):
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = _kernel32.OpenProcess(_PROCESS_QUERY, False, pid.value)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(size.value)
            if not _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return ""
            return os.path.basename(buf.value).lower()
        finally:
            _kernel32.CloseHandle(handle)

    def _browser_window():
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _lparam):
            if not _user32.IsWindowVisible(hwnd) or _user32.GetWindowTextLengthW(hwnd) <= 0:
                return True
            if _exe_name(hwnd) in _BROWSERS:
                found.append(hwnd)
                return False
            return True

        _user32.EnumWindows(visit, 0)
        return found[0] if found else None

    def _focus(hwnd):
        foreground = _user32.GetForegroundWindow()
        current = _kernel32.GetCurrentThreadId()
        front_thread = _user32.GetWindowThreadProcessId(foreground, None)
        _user32.AttachThreadInput(current, front_thread, True)
        _user32.ShowWindow(hwnd, _SW_RESTORE)
        _user32.BringWindowToTop(hwnd)
        _user32.SetForegroundWindow(hwnd)
        _user32.AttachThreadInput(current, front_thread, False)

    def _tap(vk, up):
        entry = _INPUT(type=_INPUT_KEYBOARD)
        entry.ki.wVk = vk
        entry.ki.dwFlags = _KEY_UP if up else 0
        _send_input(entry)

    def _send_browser(action):
        hwnd = _browser_window()
        if hwnd is None:
            return
        _focus(hwnd)
        time.sleep(0.08)
        if action in _WIN_BROWSER:
            key = _WIN_BROWSER[action]
            _tap(key, False)
            time.sleep(0.02)
            _tap(key, True)
            return
        modifier, key = _WIN_REFRESH
        _tap(modifier, False)
        _tap(key, False)
        time.sleep(0.02)
        _tap(key, True)
        _tap(modifier, True)

    def _send_screenshot():
        # Win+PrintScreen. 전체 화면을 그림 파일로 저장한다.
        _tap(0x5B, False)
        _tap(0x2C, False)
        time.sleep(0.02)
        _tap(0x2C, True)
        _tap(0x5B, True)

else:

    def _trusted():
        return False

    def _cursor():
        return 0.0, 0.0

    def _warp(_rel_x, _rel_y):
        return False

    def _associate(_enabled):
        return None

    def _click(_kind):
        return False

    def _scroll(_steps):
        return False

    def _send_browser(_action):
        return None

    def _send_screenshot():
        return None


def trusted():
    """다른 앱에 클릭을 보낼 수 있는지 확인합니다. 맥은 손쉬운 사용 권한이 필요합니다."""
    return _trusted()


def cursor():
    return _cursor()


def move_ratio(rel_x, rel_y):
    """박스 안 비율을 메인 모니터의 같은 위치로 옮깁니다. 0, 0 이 왼쪽 위입니다."""
    if not trusted():
        return False
    with _drive_lock:
        if not _drive_enabled:
            return False
        ok = _warp(rel_x, rel_y)
        # 옮긴 뒤에도 트랙패드는 커서를 움직인다. 떼 두면 실제 동작 동안 트랙패드가 멈춘다.
        _associate(True)
        return ok


def arm_cursor():
    """실제 동작이 커서를 따라가게 할 때 켭니다."""
    global _drive_enabled
    with _drive_lock:
        _drive_enabled = True


def release_cursor():
    """검지가 커서를 놓지 않게 하고, 트랙패드가 커서를 움직이게 둡니다."""
    global _drive_enabled
    with _drive_lock:
        _drive_enabled = False
        _associate(True)


def scroll(steps):
    """휠을 보냅니다. 양수는 위, 음수는 아래입니다."""
    if not trusted() or not steps:
        return False
    return _scroll(int(steps))


def right_click():
    """포인터 위치에 우클릭을 한 번 보냅니다. 그 자리의 사이트가 메뉴를 엽니다."""
    if not trusted():
        return False
    return _click("right")


def left_down():
    """좌클릭을 누른 채로 둡니다. 드래그는 손가락을 떼기 전까지 유지됩니다."""
    global _left_held
    if _left_held:
        return True
    if not trusted():
        return False
    if not _click("left_down"):
        return False
    _left_held = True
    return True


def left_up():
    global _left_held
    if not _left_held:
        return True
    _click("left_up")
    _left_held = False
    return True


def double_click():
    """지금 커서 자리에 누르기·떼기를 두 번 곧바로 보냅니다. 버튼을 누른 채로 두지 않습니다."""
    global _left_held
    if not trusted():
        return False
    if _left_held:
        _click("left_up")
        _left_held = False
    return _click("double")


def screenshot():
    """전체 화면을 한 장 찍습니다. 맥은 메인 화면을 바탕화면에 저장하고, 윈도우는 Win+PrintScreen입니다."""
    if not trusted():
        return False
    if not _browser_lock.acquire(blocking=False):
        return True

    def run():
        try:
            _send_screenshot()
        finally:
            _browser_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return True


def browser(action):
    """뒤로, 앞으로, 새로고침을 열려 있는 브라우저에 한 번 보냅니다."""
    if action not in _KEYS or not trusted():
        return False
    if not _browser_lock.acquire(blocking=False):
        return True

    def run():
        try:
            _send_browser(action)
        finally:
            _browser_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return True
