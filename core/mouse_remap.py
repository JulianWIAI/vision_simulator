"""
Split-Screen Mouse Coordinate Remapper
core/mouse_remap.py

Solves the click-through targeting misalignment that occurs when split-screen
mode is active.

Root cause
──────────
In split-screen, SplitScreenManager.compose() calls cv2.resize() to shrink the
full desktop frame (W × H pixels) into each panel slot.  A real desktop element
at position (real_x, real_y) appears inside the overlay at the downscaled panel
position (panel_x, panel_y).  Because the overlay is WS_EX_TRANSPARENT, every
physical click falls through at its actual screen coordinate — meaning the click
lands at (panel_x, panel_y) on the real desktop, NOT at (real_x, real_y).  The
further into a panel the element appears, the larger the offset grows.

Fix strategy
────────────
Install a Win32 WH_MOUSE_LL (low-level mouse) hook in a dedicated background
thread.  For every physical button-press event the hook:

  1. Reads the current cursor position (cx, cy) in screen pixels.
  2. Applies the INVERSE of each panel's cv2.resize scale to compute
     (mapped_x, mapped_y) — the real desktop position that visually corresponds
     to (cx, cy) in the composed overlay frame.
  3. Moves the system cursor to (mapped_x, mapped_y) via SetCursorPos.
  4. Synthesises an identical button event at the new cursor position via SendInput.
  5. Restores the cursor to (cx, cy) via SetCursorPos so the user sees no jump.
  6. Returns 1 from the hook to SUPPRESS the original (wrong-position) event.

Events injected by step 4 carry the LLMHF_INJECTED flag (bit 0 of
MSLLHOOKSTRUCT.flags).  The hook detects this flag and lets injected events pass
through unchanged, breaking the potential recursion loop.

Inverse coordinate math
───────────────────────
SplitScreenManager uses these dimension formulas (identical here for consistency):

  h_top   = H − H // 2   (ceiling half-height, e.g. 540 for H=1080, 540 for H=1079)
  h_bot   = H // 2       (floor   half-height, e.g. 540 for H=1080, 539 for H=1079)
  w_left  = W − W // 2   (ceiling half-width)
  w_right = W // 2       (floor   half-width)

Compose scale  (forward):  panel_y = real_y × (h_top / H)
Remap  scale   (inverse):  real_y  = panel_y × (H / h_top)

Each layout variant repeats this inversion for the applicable axis/quadrant.

Thread model
────────────
  Main thread   — creates SplitScreenMouseRemapper, calls start() / stop().
  Pump thread   — owns the Win32 HHOOK; runs a GetMessageW loop.
                  The hook callback fires synchronously in this thread's context.
  No Qt calls are made inside the callback or the pump thread — pure Win32 only.

Strict modularisation
─────────────────────
This file has NO imports of any project module at module load time.  The only
project reference is the SplitScreenManager instance injected via __init__().
This prevents any circular import issues and keeps the file portable.
"""

from __future__ import annotations

import ctypes                          # Python → Win32 C-library bridge
import ctypes.wintypes                 # pre-defined Win32 type aliases (DWORD, WPARAM, etc.)
import threading                       # daemon pump thread; Event for stop signalling
from typing import Optional, Tuple, TYPE_CHECKING

# TYPE_CHECKING is False at runtime — keeps the SplitScreenManager import
# type-annotation-only so no project module is loaded at import time.
if TYPE_CHECKING:
    from core.split_screen_manager import SplitScreenManager


# ── Win32 hook & message constants ─────────────────────────────────────────────

_WH_MOUSE_LL    = 14          # SetWindowsHookExW hook-type for low-level mouse events
_WM_LBUTTONDOWN = 0x0201      # left button pressed
_WM_LBUTTONUP   = 0x0202      # left button released
_WM_RBUTTONDOWN = 0x0204      # right button pressed
_WM_RBUTTONUP   = 0x0205      # right button released
_WM_MBUTTONDOWN = 0x0207      # middle (wheel) button pressed
_WM_MBUTTONUP   = 0x0208      # middle button released
_WM_QUIT        = 0x0012      # posted by PostThreadMessageW to break the GetMessageW loop

# ── MOUSEEVENTF synthesis flags ────────────────────────────────────────────────
# Passed in MOUSEINPUT.dwFlags to specify which button action SendInput synthesises.

_MOUSEEVENTF_LEFTDOWN   = 0x0002   # synthesise left-button press at current cursor pos
_MOUSEEVENTF_LEFTUP     = 0x0004   # synthesise left-button release
_MOUSEEVENTF_RIGHTDOWN  = 0x0008   # synthesise right-button press
_MOUSEEVENTF_RIGHTUP    = 0x0010   # synthesise right-button release
_MOUSEEVENTF_MIDDLEDOWN = 0x0020   # synthesise middle-button press
_MOUSEEVENTF_MIDDLEUP   = 0x0040   # synthesise middle-button release

# Map each WM_* message to the MOUSEEVENTF_* flag that re-creates the same action.
# Used in the callback to construct the synthetic SendInput payload.
_BUTTON_FLAG: dict[int, int] = {
    _WM_LBUTTONDOWN: _MOUSEEVENTF_LEFTDOWN,
    _WM_LBUTTONUP:   _MOUSEEVENTF_LEFTUP,
    _WM_RBUTTONDOWN: _MOUSEEVENTF_RIGHTDOWN,
    _WM_RBUTTONUP:   _MOUSEEVENTF_RIGHTUP,
    _WM_MBUTTONDOWN: _MOUSEEVENTF_MIDDLEDOWN,
    _WM_MBUTTONUP:   _MOUSEEVENTF_MIDDLEUP,
}

# Frozen set for fast O(1) membership check in the hot callback path.
_BUTTON_MESSAGES: frozenset = frozenset(_BUTTON_FLAG)

# ── Injection-detection flag ───────────────────────────────────────────────────

# Bit 0 of MSLLHOOKSTRUCT.flags is set by Windows for every event injected via
# SendInput or the older mouse_event() API.  We check this to skip our own
# synthetic events and break the recursion loop.
_LLMHF_INJECTED = 0x00000001   # event was injected (not from a physical device)

# ── SendInput type constant ────────────────────────────────────────────────────

_INPUT_MOUSE = 0               # INPUT.type value that selects the MOUSEINPUT union member

# ── GetSystemMetrics indices ───────────────────────────────────────────────────

_SM_CXSCREEN = 0               # index → primary screen width  in physical pixels
_SM_CYSCREEN = 1               # index → primary screen height in physical pixels


# ── ctypes Win32 structure definitions ─────────────────────────────────────────

class _POINT(ctypes.Structure):
    """Win32 POINT: two 32-bit signed integers representing a screen coordinate."""
    _fields_ = [
        ("x", ctypes.c_long),   # horizontal position in screen pixels
        ("y", ctypes.c_long),   # vertical   position in screen pixels
    ]


class _MSLLHOOKSTRUCT(ctypes.Structure):
    """
    Win32 MSLLHOOKSTRUCT — data block delivered to a WH_MOUSE_LL callback.

    The hook callback receives lParam as a C integer pointer to this structure.
    We cast lParam → POINTER(_MSLLHOOKSTRUCT) and read .contents to access the
    cursor position and injection flag without any Python-level copying.
    """
    _fields_ = [
        ("pt",          _POINT),                 # cursor position at event time (physical px)
        ("mouseData",   ctypes.wintypes.DWORD),  # scroll delta / X-button ID (unused here)
        ("flags",       ctypes.wintypes.DWORD),  # bit 0 = LLMHF_INJECTED
        ("time",        ctypes.wintypes.DWORD),  # OS timestamp in milliseconds
        ("dwExtraInfo", ctypes.c_size_t),        # ULONG_PTR — app-defined extra data
    ]


class _MOUSEINPUT(ctypes.Structure):
    """
    Win32 MOUSEINPUT — payload for a single synthetic mouse event via SendInput.

    We use it exclusively to inject button-press/release events at the current
    cursor position.  dx, dy, and mouseData are therefore always 0.
    time=0 lets the OS fill in the current timestamp automatically.
    dwExtraInfo=0 causes Windows to set the LLMHF_INJECTED flag in the delivered
    MSLLHOOKSTRUCT.flags so our hook can recognise its own injected events.
    """
    _fields_ = [
        ("dx",          ctypes.c_long),          # x displacement — 0 (cursor pre-moved)
        ("dy",          ctypes.c_long),          # y displacement — 0 (cursor pre-moved)
        ("mouseData",   ctypes.wintypes.DWORD),  # extra data — 0 for standard button events
        ("dwFlags",     ctypes.wintypes.DWORD),  # MOUSEEVENTF_* flags describing the action
        ("time",        ctypes.wintypes.DWORD),  # 0 → OS assigns current timestamp
        ("dwExtraInfo", ctypes.c_size_t),        # 0 → OS will set LLMHF_INJECTED in flags
    ]


class _InputUnion(ctypes.Union):
    """
    Anonymous union inside Win32 INPUT.

    We define only the MOUSEINPUT member (mi) because we never synthesise
    keyboard or hardware input.  ctypes sizes the union to sizeof(_MOUSEINPUT),
    which matches the real Win32 union size (MOUSEINPUT is the largest member).
    """
    _fields_ = [("mi", _MOUSEINPUT)]   # MOUSEINPUT variant of the INPUT union


class _INPUT(ctypes.Structure):
    """
    Win32 INPUT — top-level SendInput payload container.

    `type` selects the active union member:
        0 = INPUT_MOUSE    → union.mi (_MOUSEINPUT)
        1 = INPUT_KEYBOARD (not used here)
        2 = INPUT_HARDWARE (not used here)

    ctypes automatically inserts 4 bytes of padding between `type` (DWORD,
    4 bytes, alignment 4) and `union` (_InputUnion, alignment 8 on 64-bit) so
    the union starts at offset 8 — exactly matching the real Win32 INPUT layout.
    This padding is invisible to the caller; ctypes handles it silently.
    """
    _fields_ = [
        ("type",  ctypes.wintypes.DWORD),   # INPUT_MOUSE = 0
        ("union", _InputUnion),             # overlapping mouse / keyboard / hardware data
    ]


# ── Hook callback function-pointer type ────────────────────────────────────────

# WINFUNCTYPE creates a stdcall-compatible function-pointer type that ctypes can
# pass to Win32 APIs.  The three arguments match LowLevelMouseProc's signature:
#   LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam)
#
# CRITICAL: lParam MUST be declared as c_void_p, NOT ctypes.wintypes.LPARAM.
# ctypes.wintypes.LPARAM is aliased to c_long, which is 32-bit on Windows
# (Windows uses the LLP64 data model: long = 32 bits even on 64-bit).
# For WH_MOUSE_LL, lParam is a pointer to MSLLHOOKSTRUCT — a 64-bit address on
# 64-bit Windows.  Receiving a 64-bit address into a 32-bit c_long overflows,
# producing "OverflowError: int too long to convert" on every mouse event and
# also making it impossible to forward the lParam to CallNextHookEx.
# c_void_p is always pointer-sized (64-bit on 64-bit Windows) and correctly
# carries the full address without truncation.
_HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,           # return type: LRESULT (0 = pass through, 1 = suppress)
    ctypes.c_int,            # nCode  — hook code; < 0 means must call CallNextHookEx
    ctypes.wintypes.WPARAM,  # wParam — WM_LBUTTONDOWN / WM_LBUTTONUP / etc.
    ctypes.c_void_p,         # lParam — 64-bit pointer to _MSLLHOOKSTRUCT (must NOT be c_long)
)

# ── Explicit Win32 function signatures ─────────────────────────────────────────
# Setting argtypes + restype on every Win32 function we call prevents ctypes
# from guessing types (its default is c_int for all args, which is 32-bit and
# would overflow pointer arguments on 64-bit Windows).

_user32 = ctypes.windll.user32   # cached module handle — avoids repeated attribute lookup

# SetWindowsHookExW returns HHOOK which is a pointer-sized handle.
_user32.SetWindowsHookExW.restype  = ctypes.c_void_p    # HHOOK — pointer-sized on 64-bit
_user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int,            # idHook  — hook type (WH_MOUSE_LL = 14)
    ctypes.c_void_p,         # lpfn    — HOOKPROC function pointer
    ctypes.c_void_p,         # hMod    — module handle (NULL for low-level hooks)
    ctypes.wintypes.DWORD,   # dwThreadId — 0 = system-wide
]

# CallNextHookEx forwards the event to the next hook; lParam must be pointer-sized.
_user32.CallNextHookEx.restype  = ctypes.c_long          # LRESULT
_user32.CallNextHookEx.argtypes = [
    ctypes.c_void_p,         # hhk    — HHOOK handle (pointer-sized)
    ctypes.c_int,            # nCode  — passed through unchanged
    ctypes.wintypes.WPARAM,  # wParam — passed through unchanged
    ctypes.c_void_p,         # lParam — 64-bit pointer; must be pointer-sized
]

# UnhookWindowsHookEx takes the HHOOK pointer-sized handle.
_user32.UnhookWindowsHookEx.restype  = ctypes.wintypes.BOOL
_user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]   # HHOOK

# PostThreadMessageW — wParam/lParam are UINT_PTR/LONG_PTR; use defaults (fine here
# since we only post WM_QUIT with wParam=lParam=0, well within any integer type).

# SetCursorPos / SendInput use standard integer types — no override needed.


# ── Public class ───────────────────────────────────────────────────────────────

class SplitScreenMouseRemapper:
    """
    Manages a Win32 WH_MOUSE_LL hook that transparently remaps mouse clicks so
    that visual positions in split-screen panels correspond 1:1 with the
    underlying real desktop positions.

    Lifecycle
    ─────────
        remapper = SplitScreenMouseRemapper(manager.split_screen)
        remapper.start()     # spawns background thread, installs hook
        ...
        remapper.stop()      # removes hook, joins thread (called on app quit)
    """

    def __init__(self, split_screen: "SplitScreenManager") -> None:
        """
        Args:
            split_screen: Shared SplitScreenManager instance owned by OverlayManager.
                          The remapper reads .is_active and .layout_mode from it;
                          it never writes to it.
        """
        self._split_screen = split_screen           # shared read-only reference

        # Query the primary screen's physical pixel dimensions once.
        # GetSystemMetrics returns physical (not logical/DPI-scaled) pixels,
        # which matches mss's capture dimensions and the overlay's geometry
        # (because main.py sets PassThrough DPI rounding via Qt).
        self._screen_w: int = ctypes.windll.user32.GetSystemMetrics(_SM_CXSCREEN)  # physical width
        self._screen_h: int = ctypes.windll.user32.GetSystemMetrics(_SM_CYSCREEN)  # physical height

        self._hook_id: Optional[int] = None          # HHOOK handle — valid between start()/stop()
        self._thread: Optional[threading.Thread] = None   # background pump thread
        self._stop_event = threading.Event()          # set by stop() to signal loop exit

        # Store the WINFUNCTYPE-wrapped callback as an instance attribute.
        # CRITICAL: if Python garbage-collects this wrapper, the C function pointer
        # held by Win32 becomes a dangling pointer → access violation crash.
        # Keeping a reference here prevents collection for the lifetime of the object.
        self._callback = _HOOKPROC(self._hook_callback)   # stable function-pointer reference

    # ── Public lifecycle API ────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Spawns the message-pump background thread and installs the mouse hook.

        Returns immediately.  The hook is active once the thread's message loop
        starts (typically within a few milliseconds).  Safe to call from any thread.
        """
        self._stop_event.clear()             # clear any leftover stop signal from a prior run
        self._thread = threading.Thread(
            target=self._pump_thread,        # function that owns the hook and message loop
            name="MouseRemapPump",           # human-readable name in debuggers / task managers
            daemon=True,                     # killed automatically when the main thread exits
        )
        self._thread.start()                 # launch the background thread
        print("[MouseRemap] Hook thread started — click remapping active during split-screen")

    def stop(self) -> None:
        """
        Signals the hook thread to exit and waits up to 1 second for it to clean up.

        Posts WM_QUIT to the thread's message queue so the blocked GetMessageW
        call returns immediately rather than waiting for the next mouse event.
        Safe to call from the main (GUI) thread during app shutdown.
        """
        self._stop_event.set()               # signal the pump thread to exit its loop

        if self._thread is not None and self._thread.is_alive():
            # PostThreadMessageW wakes a thread blocked inside GetMessageW.
            # Without this, the thread would stay blocked until the next mouse
            # event, causing a hang of up to several seconds on shutdown.
            ctypes.windll.user32.PostThreadMessageW(
                self._thread.ident,          # OS-level thread ID (not the Python id())
                _WM_QUIT,                    # WM_QUIT — conventional "stop the loop" message
                0,                           # wParam — not used by WM_QUIT
                0,                           # lParam — not used by WM_QUIT
            )
            self._thread.join(timeout=1.0)   # wait at most 1 second for clean exit

        print("[MouseRemap] Hook thread stopped")

    # ── Background thread body ──────────────────────────────────────────────────

    def _pump_thread(self) -> None:
        """
        Installs the WH_MOUSE_LL hook, then runs a Win32 message loop.

        Win32 requirement: the hook callback for WH_MOUSE_LL is invoked in the
        context of the thread that called SetWindowsHookExW, via that thread's
        message queue.  The thread MUST continuously drain its queue via
        GetMessageW; if the queue is not drained within roughly 300 ms the OS
        assumes the hook has hung and stops delivering events to it.

        ctypes releases the Python GIL for each blocking Win32 call (GetMessageW,
        SetWindowsHookExW), so this thread does not starve other Python threads.
        The GIL is re-acquired when a C call returns and Python code resumes.
        """
        # Install the hook system-wide (dwThreadId=0 hooks all desktop threads).
        # hMod=0 (NULL) is valid for WH_MOUSE_LL — it does not require a DLL.
        self._hook_id = ctypes.windll.user32.SetWindowsHookExW(
            _WH_MOUSE_LL,       # hook type: low-level mouse — fires before event delivery
            self._callback,     # WINFUNCTYPE-wrapped Python callable
            0,                  # hMod: NULL is acceptable for low-level global hooks
            0,                  # dwThreadId: 0 = hook the entire desktop
        )

        if not self._hook_id:            # SetWindowsHookExW returns NULL on failure
            print("[MouseRemap] SetWindowsHookExW failed — click remapping disabled")
            return                       # exit without the loop; app continues without remap

        # ── Win32 message pump ──────────────────────────────────────────────
        msg = ctypes.wintypes.MSG()      # reusable MSG structure; stack-allocated once

        while not self._stop_event.is_set():              # loop until stop() signals exit
            result = ctypes.windll.user32.GetMessageW(
                ctypes.byref(msg),       # output buffer: receives next message from the queue
                None,                    # hWnd: None = retrieve messages for all windows
                0,                       # wMsgFilterMin: 0 = no lower filter
                0,                       # wMsgFilterMax: 0 = no upper filter
            )
            # GetMessageW return values:
            #   > 0  → message retrieved and processed; continue loop
            #     0  → WM_QUIT received; exit gracefully
            #    -1  → OS error; abort

            if result == 0:              # WM_QUIT: stop() posted it via PostThreadMessageW
                break
            if result == -1:             # Win32 error inside GetMessageW
                break

            # Standard message dispatch — necessary for timers, etc.
            # Mouse hook callbacks are NOT dispatched through DispatchMessageW;
            # they are called directly by the OS during GetMessageW.  These calls
            # are therefore no-ops for our purposes but are required for correctness.
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))   # WM_KEYDOWN → WM_CHAR (no-op here)
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))   # route to window proc (no-op here)

        # ── Cleanup: uninstall the hook before the thread exits ─────────────
        if self._hook_id:
            ctypes.windll.user32.UnhookWindowsHookEx(self._hook_id)   # release the HHOOK handle
            self._hook_id = None         # prevent double-unhook if stop() is called again

    # ── Hook callback (runs in the pump thread, NOT the GUI thread) ─────────────

    def _hook_callback(self, nCode: int, wParam: int, lParam: int) -> int:
        """
        LowLevelMouseProc — called synchronously for every system mouse event.

        Per MSDN contract:
          • nCode >= 0 → we MAY inspect and optionally suppress the event.
          • nCode <  0 → we MUST call CallNextHookEx immediately and return.

        Return values:
          • 1 (any non-zero) → event suppressed; no application receives it.
          • CallNextHookEx() → event is passed to the next hook in the chain.

        This method must complete in under ~200–300 ms.  If it takes longer,
        Windows assumes the hook is unresponsive and bypasses it for that event.
        Our implementation does only integer comparisons, a pointer cast, integer
        arithmetic, and two Win32 calls — well within any timeout.
        """
        # ── Rule 1: mandatory pass-through when nCode < 0 ──────────────────
        # MSDN requires calling CallNextHookEx when nCode is negative.
        if nCode < 0:
            return ctypes.windll.user32.CallNextHookEx(
                self._hook_id, nCode, wParam, lParam  # forward to the next hook in the chain
            )

        # ── Rule 2: only intercept button-press/release events ──────────────
        # Mouse-move, scroll-wheel, and X-button messages are left unchanged.
        # _BUTTON_MESSAGES is a frozenset → O(1) lookup with no GIL pressure.
        if wParam not in _BUTTON_MESSAGES:
            return ctypes.windll.user32.CallNextHookEx(
                self._hook_id, nCode, wParam, lParam  # pass through non-button events
            )

        # ── Rule 3: decode the hook data structure ──────────────────────────
        # lParam is a Win32 LPARAM (a C integer holding a pointer address).
        # ctypes.cast reinterprets it as a pointer to _MSLLHOOKSTRUCT.
        # .contents dereferences the pointer, producing a Python struct view
        # (no data is copied; this reads directly from the OS-supplied memory).
        info = ctypes.cast(lParam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents

        # ── Rule 4: skip events we injected ourselves ───────────────────────
        # Our SendInput call in step 5 below causes the OS to call this hook
        # again with LLMHF_INJECTED set.  Passing it through here breaks the
        # recursion loop — the injected event reaches the application normally.
        if info.flags & _LLMHF_INJECTED:
            return ctypes.windll.user32.CallNextHookEx(
                self._hook_id, nCode, wParam, lParam  # let our own injected events pass
            )

        # ── Rule 5: only remap when split-screen is active ──────────────────
        # .is_active reads a single Python str attribute — GIL-atomic read.
        # No lock needed (same contract as OverlayManager.distribute()).
        if not self._split_screen.is_active:
            return ctypes.windll.user32.CallNextHookEx(
                self._hook_id, nCode, wParam, lParam  # split-screen off → pass through
            )

        # ── Step 1: read cursor position from the hook structure ────────────
        cx: int = info.pt.x    # cursor x in physical screen pixels at event time
        cy: int = info.pt.y    # cursor y in physical screen pixels at event time

        # ── Step 2: compute the remapped desktop position ───────────────────
        # _map_coords applies the inverse of each panel's cv2.resize scale.
        mapped_x, mapped_y = self._map_coords(cx, cy)   # actual desktop target position

        # ── Step 3: move the system cursor to the mapped position ───────────
        # SetCursorPos is thread-safe (documented Win32 guarantee).
        # After this call, GetCursorPos / MSLLHOOKSTRUCT.pt returns (mapped_x, mapped_y).
        ctypes.windll.user32.SetCursorPos(mapped_x, mapped_y)   # reposition cursor

        # ── Step 4: build the synthetic button event ─────────────────────────
        # We send a button-only event (no MOUSEEVENTF_MOVE).  Because the cursor
        # was already moved by SetCursorPos, the OS delivers the click at the
        # moved position automatically — no absolute coordinates needed here.
        inp = _INPUT()                                  # zero-initialised INPUT structure
        inp.type               = _INPUT_MOUSE           # mark payload as MOUSEINPUT
        inp.union.mi.dx        = 0                      # no horizontal movement
        inp.union.mi.dy        = 0                      # no vertical movement
        inp.union.mi.mouseData = 0                      # no extra data (not X-button)
        inp.union.mi.dwFlags   = _BUTTON_FLAG[wParam]   # the exact button action to replay
        inp.union.mi.time      = 0                      # 0 → OS fills current timestamp
        inp.union.mi.dwExtraInfo = 0                    # 0 → OS sets LLMHF_INJECTED in flags

        # ── Step 5: inject the synthetic event ──────────────────────────────
        # SendInput processes all inputs atomically as a group (nInputs=1 here).
        # cbSize=sizeof(_INPUT) lets Win32 validate the structure version.
        ctypes.windll.user32.SendInput(
            1,                       # nInputs: one INPUT structure in the array
            ctypes.byref(inp),       # pInputs: pointer to the INPUT array
            ctypes.sizeof(_INPUT),   # cbSize:  must equal sizeof(INPUT) — Win32 version check
        )

        # ── Step 6: restore cursor to original panel position ───────────────
        # SetCursorPos in step 3 moved the physical cursor to (mapped_x, mapped_y),
        # which is the real desktop target — typically 2× deeper into the screen
        # than where the user clicked inside the panel.  Without restoring, the
        # cursor remains at the mapped position after the click.  This causes:
        #   • A visible "jump" on every button event (the cursor teleports and stays).
        #   • Corrupted spatial reference for subsequent clicks — the user's next
        #     click starts from the wrong position, making all panels appear offset.
        # Restoring to (cx, cy) makes the cursor snap back to the exact panel pixel
        # where the physical device fired, so the user's visual position is preserved.
        ctypes.windll.user32.SetCursorPos(cx, cy)   # return cursor to original panel position

        # ── Step 7: suppress the original event ─────────────────────────────
        # Returning a non-zero value from a WH_MOUSE_LL callback tells Win32 to
        # discard the original event.  No application or window receives it.
        # Our synthetic event (step 5) replaces it at the corrected position.
        return 1     # suppress original — synthetic event replaces it at mapped position

    # ── Inverse coordinate transform ────────────────────────────────────────────

    def _map_coords(self, cx: int, cy: int) -> Tuple[int, int]:
        """
        Converts a cursor position in the composed overlay frame (panel space)
        to the corresponding position in the real desktop space.

        This is the exact inverse of SplitScreenManager._process_panel()'s
        cv2.resize call.  The dimension formulas (h_top, h_bot, w_left, w_right)
        are byte-for-byte identical to the ones in split_screen_manager.py to
        guarantee pixel-perfect consistency — any drift would re-introduce offset.

        Args:
            cx: Cursor x in overlay / composed frame pixels.
            cy: Cursor y in overlay / composed frame pixels.

        Returns:
            (mapped_x, mapped_y) in real desktop pixels, clamped to screen bounds.
            Returns (cx, cy) unchanged when the layout is "none" or unrecognised.
        """
        W: int = self._screen_w      # primary screen width  (physical pixels, set in __init__)
        H: int = self._screen_h      # primary screen height (physical pixels, set in __init__)

        # Single GIL-atomic str attribute read — no lock required.
        layout: str = self._split_screen.layout_mode    # "2x_horizontal" | "2x_vertical" | "4x_grid"

        # ── Layout: 2x_horizontal (Top / Bottom) ───────────────────────────
        if layout == "2x_horizontal":
            # Compose formula (split_screen_manager.py):
            #   top panel:    cv2.resize(raw, (W, h_top))   → h_top rows show all H desktop rows
            #   bottom panel: cv2.resize(raw, (W, h_bot))   → h_bot rows show all H desktop rows
            #
            # Panel pixel y corresponds to desktop pixel y via:
            #   panel_y = real_y × (h_panel / H)      ← forward (compose)
            #   real_y  = panel_y × (H / h_panel)     ← inverse  (remap)   ← we apply this

            h_top = H - H // 2                    # ceiling half — MUST match compose() exactly
            h_bot = H // 2                         # floor   half — MUST match compose() exactly

            mapped_x = cx                          # x is not scaled in horizontal split

            if cy < h_top:
                # ── Top panel ────────────────────────────────────────────
                # panel_y ∈ [0, h_top).  Real desktop row = panel_y × H / h_top.
                # Integer multiply first to avoid intermediate float truncation loss.
                mapped_y = int(cy * H / h_top)     # scale panel y back to full desktop y

            else:
                # ── Bottom panel ──────────────────────────────────────────
                # local_y = position within the bottom panel (subtract top panel height).
                # Real desktop row = local_y × H / h_bot.
                local_y  = cy - h_top              # y relative to bottom panel origin
                mapped_y = int(local_y * H / h_bot)  # inverse scale to full desktop y

        # ── Layout: 2x_vertical (Left / Right) ─────────────────────────────
        elif layout == "2x_vertical":
            # Compose formula:
            #   left  panel: cv2.resize(raw, (w_left,  H)) → w_left  cols show all W desktop cols
            #   right panel: cv2.resize(raw, (w_right, H)) → w_right cols show all W desktop cols
            #
            # Inverse:  real_x = panel_x × (W / w_panel)

            w_left  = W - W // 2                   # ceiling half — MUST match compose() exactly
            w_right = W // 2                        # floor   half — MUST match compose() exactly

            mapped_y = cy                           # y is not scaled in vertical split

            if cx < w_left:
                # ── Left panel ────────────────────────────────────────────
                # panel_x ∈ [0, w_left).  Real desktop col = panel_x × W / w_left.
                mapped_x = int(cx * W / w_left)     # inverse scale to full desktop x

            else:
                # ── Right panel ───────────────────────────────────────────
                local_x  = cx - w_left             # x relative to right panel origin
                mapped_x = int(local_x * W / w_right)  # inverse scale to full desktop x

        # ── Layout: 4x_grid (2×2 quad) ─────────────────────────────────────
        elif layout == "4x_grid":
            # Compose formula:
            #   TL: cv2.resize(raw, (w_left,  h_top)) → covers [0,w_left) × [0,h_top)
            #   TR: cv2.resize(raw, (w_right, h_top)) → covers [w_left,W) × [0,h_top)
            #   BL: cv2.resize(raw, (w_left,  h_bot)) → covers [0,w_left) × [h_top,H)
            #   BR: cv2.resize(raw, (w_right, h_bot)) → covers [w_left,W) × [h_top,H)
            #
            # Each quadrant independently scales both x and y.
            # Inverse:  real_x = local_x × (W / w_panel)
            #           real_y = local_y × (H / h_panel)

            h_top   = H - H // 2                   # ceiling half height
            h_bot   = H // 2                        # floor   half height
            w_left  = W - W // 2                    # ceiling half width
            w_right = W // 2                        # floor   half width

            if cy < h_top:
                # ── Upper row (TL or TR) ──────────────────────────────────
                # panel_y ∈ [0, h_top).  Vertical inverse: real_y = cy × H / h_top.
                mapped_y = int(cy * H / h_top)      # undo the vertical downscale

                if cx < w_left:
                    # Top-Left quadrant: panel_x ∈ [0, w_left)
                    mapped_x = int(cx * W / w_left)          # undo horizontal downscale
                else:
                    # Top-Right quadrant: local_x ∈ [0, w_right)
                    local_x  = cx - w_left                   # x within right panel
                    mapped_x = int(local_x * W / w_right)    # undo horizontal downscale

            else:
                # ── Lower row (BL or BR) ──────────────────────────────────
                # local_y ∈ [0, h_bot).  Vertical inverse: real_y = local_y × H / h_bot.
                local_y  = cy - h_top                        # y within bottom half
                mapped_y = int(local_y * H / h_bot)          # undo the vertical downscale

                if cx < w_left:
                    # Bottom-Left quadrant: panel_x ∈ [0, w_left)
                    mapped_x = int(cx * W / w_left)          # undo horizontal downscale
                else:
                    # Bottom-Right quadrant: local_x ∈ [0, w_right)
                    local_x  = cx - w_left                   # x within right panel
                    mapped_x = int(local_x * W / w_right)    # undo horizontal downscale

        else:
            # Layout is "none" or an unrecognised value — no remapping needed.
            return cx, cy                          # return original coordinates unchanged

        # ── Clamp to valid screen bounds ────────────────────────────────────
        # Floating-point division can produce a value of exactly W or H at edge
        # pixels (e.g., int(539 × 1080 / 539) = int(1080.0) = 1080 ≥ H).
        # Clamping guarantees SetCursorPos always receives a valid coordinate.
        mapped_x = max(0, min(mapped_x, W - 1))    # clamp x to [0, W-1]
        mapped_y = max(0, min(mapped_y, H - 1))    # clamp y to [0, H-1]

        return mapped_x, mapped_y                  # remapped real desktop coordinates
