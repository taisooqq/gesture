"""인식된 제스처를 마우스와 브라우저 단축키로 보냅니다."""

import subprocess
import threading
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

_KEYS = {"back": "[", "forward": "]", "refresh": "r"}
_left_held = False
_browser_lock = threading.Lock()
_drive_lock = threading.Lock()
_drive_enabled = False


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


def trusted():
    """손쉬운 사용 권한이 있어야 다른 앱에 클릭을 보낼 수 있습니다."""
    return bool(_access.AXIsProcessTrusted())


def cursor():
    point = _core.CGEventGetLocation(_core.CGEventCreate(None))
    return point.x, point.y


def move_ratio(rel_x, rel_y):
    """박스 안 비율을 메인 모니터의 같은 위치로 옮깁니다. 0, 0 이 왼쪽 위입니다."""
    if not trusted():
        return False
    with _drive_lock:
        if not _drive_enabled:
            return False
        ok = _warp(rel_x, rel_y)
        # 옮긴 뒤에도 트랙패드는 커서를 움직인다. 떼 두면 실제 동작 동안 트랙패드가 멈춘다.
        _core.CGAssociateMouseAndMouseCursorPosition(True)
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
        _core.CGAssociateMouseAndMouseCursorPosition(True)


def _warp(rel_x, rel_y):
    bounds = _core.CGDisplayBounds(_core.CGMainDisplayID())
    point = _Point(
        bounds.x + float(rel_x) * bounds.width,
        bounds.y + float(rel_y) * bounds.height,
    )
    error = _core.CGWarpMouseCursorPosition(point)
    return error == 0


def right_click():
    """포인터 위치에 우클릭을 한 번 보냅니다. 그 자리의 사이트가 메뉴를 엽니다."""
    if not trusted():
        return False
    point = _core.CGEventGetLocation(_core.CGEventCreate(None))
    return _post(point, _RIGHT_DOWN, _RIGHT) and _post(point, _RIGHT_UP, _RIGHT)


def left_down():
    """좌클릭을 누른 채로 둡니다. 드래그는 손가락을 떼기 전까지 유지됩니다."""
    global _left_held
    if _left_held:
        return True
    if not trusted():
        return False
    point = _core.CGEventGetLocation(_core.CGEventCreate(None))
    if not _post(point, _LEFT_DOWN, _LEFT):
        return False
    _left_held = True
    return True


def left_up():
    global _left_held
    if not _left_held:
        return True
    point = _core.CGEventGetLocation(_core.CGEventCreate(None))
    _post(point, _LEFT_UP, _LEFT)
    _left_held = False
    return True


def browser(action):
    """뒤로, 앞으로, 새로고침을 열려 있는 브라우저에 한 번 보냅니다."""
    if action not in _KEYS or not trusted():
        return False
    if not _browser_lock.acquire(blocking=False):
        return True

    def run():
        try:
            subprocess.run(
                ["osascript", "-e", _BROWSER_SCRIPT, action],
                check=False,
                capture_output=True,
                text=True,
                timeout=4,
            )
        finally:
            _browser_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return True


def _post(point, event_type, button):
    event = _core.CGEventCreateMouseEvent(None, event_type, point, button)
    if not event:
        return False
    _core.CGEventSetIntegerValueField(event, _CLICK_STATE, 1)
    _core.CGEventPost(_HID, event)
    _foundation.CFRelease(event)
    return True


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
      keystroke "r" using command down
    end if
  end tell
  return "ok"
end run
'''
