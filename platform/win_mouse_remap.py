"""
platform/win_mouse_remap.py

Windows implementation of AbstractMouseRemapper.

Installs a Win32 WH_MOUSE_LL (low-level mouse) hook in a dedicated background
thread.  For every physical button-press event the hook:

  1. Reads the cursor position (cx, cy) from the MSLLHOOKSTRUCT payload.
  2. Applies the inverse of each split-panel's cv2.resize scale via
     AbstractMouseRemapper._map_coords() (shared base-class math).
  3. Moves the system cursor to the mapped position via SetCursorPos.
  4. Synthesises an identical button event at the new position via SendInput.
  5. Restores the cursor to (cx, cy) so the user sees no visual jump.
  6. Returns 1 to suppress the original (wrong-position) event.

Events injected by step 4 carry the LLMHF_INJECTED flag in
MSLLHOOKSTRUCT.flags.  The hook detects this and lets injected events pass
through unchanged, preventing the recursion loop.

Thread model
────────────
  Main thread  — calls start() / stop().
  Pump thread  — owns the HHOOK handle; runs GetMessageW loop.
                 The hook callback fires synchronously in this thread.
  No Qt calls are made inside the callback — pure Win32 only.

CRITICAL — lParam type safety
─────────────────────────────
The WINFUNCTYPE lParam argument MUST be c_void_p, not ctypes.wintypes.LPARAM.
LPARAM is aliased to c_long, which is 32-bit on Windows (LLP64 model: long
= 32 bits even on 64-bit Windows).  For WH_MOUSE_LL, lParam is a pointer to
MSLLHOOKSTRUCT — a 64-bit address on 64-bit Windows.  Receiving a 64-bit
pointer in a 32-bit c_long produces OverflowError on every mouse event.
c_void_p is always pointer-sized (64-bit on 64-bit Windows).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import threading
from typing import Optional

from platform.base import AbstractMouseRemapper


# ── Win32 hook & message constants ─────────────────────────────────────────────

_WH_MOUSE_LL    = 14       # SetWindowsHookExW hook type for low-level mouse events
_WM_LBUTTONDOWN = 0x0201   # left button pressed
_WM_LBUTTONUP   = 0x0202   # left button released
_WM_RBUTTONDOWN = 0x0204   # right button pressed
_WM_RBUTTONUP   = 0x0205   # right button released
_WM_MBUTTONDOWN = 0x0207   # middle button pressed
_WM_MBUTTONUP   = 0x0208   # middle button released
_WM_QUIT        = 0x0012   # posted by PostThreadMessageW to break the GetMessageW loop

# MOUSEEVENTF_* flags — passed to MOUSEINPUT.dwFlags to specify which button
# action SendInput synthesises.
_MOUSEEVENTF_LEFTDOWN   = 0x0002
_MOUSEEVENTF_LEFTUP     = 0x0004
_MOUSEEVENTF_RIGHTDOWN  = 0x0008
_MOUSEEVENTF_RIGHTUP    = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP   = 0x0040

# Map each WM_* message to the MOUSEEVENTF_* flag that re-creates the same action.
_BUTTON_FLAG: dict[int, int] = {
    _WM_LBUTTONDOWN: _MOUSEEVENTF_LEFTDOWN,
    _WM_LBUTTONUP:   _MOUSEEVENTF_LEFTUP,
    _WM_RBUTTONDOWN: _MOUSEEVENTF_RIGHTDOWN,
    _WM_RBUTTONUP:   _MOUSEEVENTF_RIGHTUP,
    _WM_MBUTTONDOWN: _MOUSEEVENTF_MIDDLEDOWN,
    _WM_MBUTTONUP:   _MOUSEEVENTF_MIDDLEUP,
}

# Frozen set for O(1) membership test in the hot callback path.
_BUTTON_MESSAGES: frozenset = frozenset(_BUTTON_FLAG)

# Bit 0 of MSLLHOOKSTRUCT.flags — set by Windows on events injected via SendInput.
# We check this to skip our own synthetic events and break the recursion loop.
_LLMHF_INJECTED = 0x00000001

# INPUT.type value that selects the MOUSEINPUT union member.
_INPUT_MOUSE = 0


# ── ctypes Win32 structure definitions ─────────────────────────────────────────

class _POINT(ctypes.Structure):
    """Win32 POINT — two 32-bit signed integers for a screen coordinate."""
    _fields_ = [
        ("x", ctypes.c_long),   # horizontal position in screen pixels
        ("y", ctypes.c_long),   # vertical   position in screen pixels
    ]


class _MSLLHOOKSTRUCT(ctypes.Structure):
    """
    Win32 MSLLHOOKSTRUCT — data block delivered to a WH_MOUSE_LL callback.

    lParam in the hook callback is a C pointer to this structure.  We cast
    lParam (c_void_p) → POINTER(_MSLLHOOKSTRUCT) and read .contents to access
    the cursor position and injection flag without any Python-level copying.
    """
    _fields_ = [
        ("pt",          _POINT),                 # cursor position at event time
        ("mouseData",   ctypes.wintypes.DWORD),  # scroll delta / X-button ID (unused)
        ("flags",       ctypes.wintypes.DWORD),  # bit 0 = LLMHF_INJECTED
        ("time",        ctypes.wintypes.DWORD),  # OS timestamp in milliseconds
        ("dwExtraInfo", ctypes.c_size_t),        # ULONG_PTR app-defined extra data
    ]


class _MOUSEINPUT(ctypes.Structure):
    """
    Win32 MOUSEINPUT — payload for a single synthetic mouse event via SendInput.

    dx, dy, and mouseData are 0 because we pre-move the cursor via SetCursorPos
    before calling SendInput; the OS delivers the click at the current cursor
    position automatically.  time=0 lets the OS fill the current timestamp.
    dwExtraInfo=0 causes Windows to set LLMHF_INJECTED in the delivered flags.
    """
    _fields_ = [
        ("dx",          ctypes.c_long),
        ("dy",          ctypes.c_long),
        ("mouseData",   ctypes.wintypes.DWORD),
        ("dwFlags",     ctypes.wintypes.DWORD),  # MOUSEEVENTF_* button action
        ("time",        ctypes.wintypes.DWORD),  # 0 → OS assigns current timestamp
        ("dwExtraInfo", ctypes.c_size_t),        # 0 → OS sets LLMHF_INJECTED
    ]


class _InputUnion(ctypes.Union):
    """Anonymous union inside Win32 INPUT.  Only the MOUSEINPUT member is used."""
    _fields_ = [("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    """
    Win32 INPUT — top-level SendInput payload container.

    type = 0 (INPUT_MOUSE) selects the union.mi (_MOUSEINPUT) member.
    ctypes inserts 4 bytes of padding between type (DWORD, align 4) and
    union (align 8 on 64-bit) — matching the real Win32 INPUT layout exactly.
    """
    _fields_ = [
        ("type",  ctypes.wintypes.DWORD),
        ("union", _InputUnion),
    ]


# ── Hook callback function-pointer type ────────────────────────────────────────

# WINFUNCTYPE creates a stdcall-compatible function-pointer type.
# Arguments match LowLevelMouseProc:
#   LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam)
# lParam MUST be c_void_p (pointer-sized), NOT c_long (32-bit) — see module docstring.
_HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,           # return: LRESULT (0 = pass through, 1 = suppress)
    ctypes.c_int,            # nCode  — hook code; < 0 means must call CallNextHookEx
    ctypes.wintypes.WPARAM,  # wParam — WM_LBUTTONDOWN / etc.
    ctypes.c_void_p,         # lParam — 64-bit pointer to _MSLLHOOKSTRUCT
)


# ── Explicit Win32 function signatures ─────────────────────────────────────────
# Setting argtypes + restype prevents ctypes from guessing types (its default
# is c_int for all args, which is 32-bit and overflows pointer args on 64-bit).

_user32 = ctypes.windll.user32   # cached handle — avoids repeated attribute lookup

_user32.SetWindowsHookExW.restype  = ctypes.c_void_p    # HHOOK — pointer-sized
_user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int,            # idHook  — WH_MOUSE_LL = 14
    ctypes.c_void_p,         # lpfn    — HOOKPROC function pointer
    ctypes.c_void_p,         # hMod    — NULL for low-level hooks
    ctypes.wintypes.DWORD,   # dwThreadId — 0 = system-wide
]

_user32.CallNextHookEx.restype  = ctypes.c_long          # LRESULT
_user32.CallNextHookEx.argtypes = [
    ctypes.c_void_p,         # hhk    — HHOOK (pointer-sized)
    ctypes.c_int,            # nCode
    ctypes.wintypes.WPARAM,  # wParam
    ctypes.c_void_p,         # lParam — pointer-sized
]

_user32.UnhookWindowsHookEx.restype  = ctypes.wintypes.BOOL
_user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]   # HHOOK


# ── WindowsMouseRemapper ────────────────────────────────────────────────────────

class WindowsMouseRemapper(AbstractMouseRemapper):
    """
    Installs a Win32 WH_MOUSE_LL hook that transparently remaps mouse clicks
    during split-screen mode so that visual panel positions correspond 1:1 with
    the underlying real desktop positions.
    """

    def __init__(self, split_screen, screen_w: int, screen_h: int) -> None:
        """
        Args:
            split_screen: Shared SplitScreenManager (read-only).
            screen_w:     Primary screen width  in physical pixels.
            screen_h:     Primary screen height in physical pixels.
        """
        super().__init__(split_screen, screen_w, screen_h)

        self._hook_id: Optional[int] = None           # HHOOK handle — valid between start()/stop()
        self._thread:  Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # CRITICAL: keep the WINFUNCTYPE-wrapped callback alive as an instance
        # attribute.  If Python garbage-collects this wrapper, Win32 holds a
        # dangling pointer and the next mouse event causes an access violation.
        self._callback = _HOOKPROC(self._hook_callback)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawns the message-pump thread and installs the WH_MOUSE_LL hook."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._pump_thread,
            name="MouseRemapPump",
            daemon=True,   # killed automatically when the main thread exits
        )
        self._thread.start()
        print("[WindowsMouseRemapper] Hook thread started — click remapping active")

    def stop(self) -> None:
        """
        Signals the hook thread to exit and waits up to 1 s for clean-up.

        Posts WM_QUIT to the thread's message queue so the blocked GetMessageW
        call returns immediately rather than waiting for the next mouse event.
        """
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            # Wake the blocked GetMessageW by posting WM_QUIT to the thread queue.
            _user32.PostThreadMessageW(
                self._thread.ident,   # OS-level thread ID (not Python's id())
                _WM_QUIT,
                0,
                0,
            )
            self._thread.join(timeout=1.0)

        print("[WindowsMouseRemapper] Hook thread stopped")

    # ── Background thread body ─────────────────────────────────────────────

    def _pump_thread(self) -> None:
        """
        Installs the hook system-wide, then drives a Win32 message loop.

        Win32 requires the WH_MOUSE_LL callback to be serviced via the
        message queue of the thread that called SetWindowsHookExW.  If the
        queue is not drained within ~300 ms the OS assumes the hook has hung
        and stops delivering events to it.
        """
        # Install the hook.  hMod=0 (NULL) is valid for WH_MOUSE_LL — it does
        # not need to live inside a DLL.  dwThreadId=0 hooks the entire desktop.
        self._hook_id = _user32.SetWindowsHookExW(
            _WH_MOUSE_LL,
            self._callback,
            0,    # hMod: NULL acceptable for low-level hooks
            0,    # dwThreadId: 0 = hook all threads system-wide
        )

        if not self._hook_id:
            print("[WindowsMouseRemapper] SetWindowsHookExW failed — click remapping disabled")
            return

        # Message pump — drains the queue so the OS keeps the hook alive.
        msg = ctypes.wintypes.MSG()   # reusable MSG; stack-allocated once

        while not self._stop_event.is_set():
            result = _user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            # > 0 → message retrieved; 0 → WM_QUIT; -1 → OS error
            if result == 0 or result == -1:
                break
            # These two calls are no-ops for mouse hook messages but are
            # required by the Win32 message-loop contract.
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))

        # Uninstall the hook before the thread exits.
        if self._hook_id:
            _user32.UnhookWindowsHookEx(self._hook_id)
            self._hook_id = None

    # ── Hook callback (pump thread) ────────────────────────────────────────

    def _hook_callback(self, nCode: int, wParam: int, lParam: int) -> int:
        """
        LowLevelMouseProc — called synchronously for every system mouse event.

        Must complete in under ~200–300 ms.  This implementation performs only
        integer comparisons, a pointer cast, arithmetic, and two Win32 calls.

        Return values:
          1                  → event suppressed (no application receives it).
          CallNextHookEx()   → event passed to the next hook in the chain.
        """
        # nCode < 0 → MSDN requires forwarding without inspection.
        if nCode < 0:
            return _user32.CallNextHookEx(self._hook_id, nCode, wParam, lParam)

        # Only intercept button press/release; pass everything else through.
        if wParam not in _BUTTON_MESSAGES:
            return _user32.CallNextHookEx(self._hook_id, nCode, wParam, lParam)

        # Dereference lParam → MSLLHOOKSTRUCT (no data copied; reads OS memory).
        info = ctypes.cast(lParam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents

        # Skip events we injected ourselves (LLMHF_INJECTED set) to break the
        # recursion loop: our SendInput call triggers this hook again.
        if info.flags & _LLMHF_INJECTED:
            return _user32.CallNextHookEx(self._hook_id, nCode, wParam, lParam)

        # Only remap when split-screen is active (GIL-atomic attribute read).
        if not self._split_screen.is_active:
            return _user32.CallNextHookEx(self._hook_id, nCode, wParam, lParam)

        # ── Coordinate remapping ───────────────────────────────────────────
        cx: int = info.pt.x    # cursor x at event time (physical pixels)
        cy: int = info.pt.y    # cursor y at event time (physical pixels)

        # Apply inverse panel scale to find the real desktop target.
        mapped_x, mapped_y = self._map_coords(cx, cy)

        # Move cursor to the real desktop position.
        _user32.SetCursorPos(mapped_x, mapped_y)

        # Build and inject a synthetic button event at the new cursor position.
        # No MOUSEEVENTF_MOVE needed — SetCursorPos already positioned the cursor.
        inp = _INPUT()
        inp.type               = _INPUT_MOUSE
        inp.union.mi.dx        = 0
        inp.union.mi.dy        = 0
        inp.union.mi.mouseData = 0
        inp.union.mi.dwFlags   = _BUTTON_FLAG[wParam]   # replay the same button action
        inp.union.mi.time      = 0                       # OS fills current timestamp
        inp.union.mi.dwExtraInfo = 0                     # triggers LLMHF_INJECTED on re-entry

        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

        # Restore the cursor to the original panel pixel so the user sees no jump.
        _user32.SetCursorPos(cx, cy)

        # Suppress the original event; our synthetic event replaces it.
        return 1
