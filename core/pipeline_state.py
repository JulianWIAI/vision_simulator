"""
Pipeline State Persistence Utilities
core/pipeline_state.py

Provides three stateless helper functions used by OverlayWindow to detect
and preserve active VisionPipeline effect chains when the user presses hotkeys
(M / 1-9) to cycle or directly select a new base vision mode.

Problem solved
──────────────
Without this module, OverlayWindow.cycle_mode() and set_mode() replace the
entire _mode attribute with a plain BaseVisionMode, silently discarding all
post-processing effects (e.g., the HD Matrix Analyzer overlay).  The user
selects "Dog Vision + HD Matrix Analyzer", presses M, and the matrix markers
vanish — the pipeline is gone.

Solution contract
─────────────────
  is_pipeline(mode)                — O(1) type guard; callable from any thread.
  resolve_base_index(idx, mode, n) — Converts the -1 pipeline sentinel to a
                                     safe numerical cycle origin.
  apply_base_mode_change(cur, new, idx) — Single authoritative patch point;
                                     returns (mode_to_store, index_to_store).

Thread safety
─────────────
All three functions are stateless (no shared mutable state).  The only write
they perform is a single attribute assignment on a VisionPipeline object under
CPython's GIL, which is atomic.  Safe to call from the FrameWorker thread or
the GUI thread.

Why a separate module
─────────────────────
The logic is imported by OverlayWindow (which must not import the full UI stack)
and must not create a circular dependency.  Isolating it here lets both
overlay_window.py and overlay_manager.py import it without cross-coupling.
"""

from __future__ import annotations

# TYPE_CHECKING is False at runtime — imports inside this block only exist for
# static analysis tools (mypy / pyright).  This pattern breaks circular imports:
#   pipeline_state → VisionPipeline → BaseVisionMode ← OverlayWindow → pipeline_state
# At runtime, each function performs a local import instead.
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    # These are ONLY used by the type checker; never executed at runtime.
    from modes.base_mode import BaseVisionMode   # for type annotations only
    from core.vision_pipeline import VisionPipeline  # for type annotations only


# ── Public API ─────────────────────────────────────────────────────────────────

def is_pipeline(mode: object) -> bool:
    """
    Returns True when *mode* is an active VisionPipeline composition.

    Uses a runtime-local import inside the function body to avoid the
    circular import that would occur if we imported VisionPipeline at the
    top of this module (pipeline_state ← overlay_window ← VisionPipeline
    ← pipeline_state).  Python caches sys.modules so repeated local imports
    cost only a single dictionary lookup — no performance penalty.

    Args:
        mode: Any object — typically a BaseVisionMode subclass instance.

    Returns:
        True  → mode is a VisionPipeline; the caller should preserve its
                effects and only patch the base_mode reference.
        False → mode is a plain base mode; direct replacement is safe.
    """
    # Deferred import: executed at function call time, not at module load time.
    # This breaks the circular dependency chain without affecting speed.
    from core.vision_pipeline import VisionPipeline  # local import — avoids circular dep

    # isinstance is the idiomatic, O(1) way to check class membership in Python.
    # It correctly returns True for any subclass of VisionPipeline as well.
    return isinstance(mode, VisionPipeline)          # True → pipeline active, preserve effects


def resolve_base_index(mode_index: int, mode: object, total_modes: int) -> int:
    """
    Returns a valid mode-list index to use as the cycling origin.

    Why this is needed
    ──────────────────
    When a VisionPipeline is active, OverlayWindow stores _mode_index = -1
    to signal 'custom / pipeline mode' to the HUD renderer.  If we feed -1
    directly into the modulo cycling formula:

        next_idx = (-1 + 1) % n  →  0   (always!)

    …every single press of M would restart from index 0 instead of advancing.
    This function converts the -1 sentinel to a safe integer in [0, n-1] so
    that cycling continues from where the user left off.

    The caller (OverlayWindow) stores the computed next_idx into its own
    _base_mode_index attribute, so each subsequent M press advances correctly
    without revisiting index 0 again.

    Args:
        mode_index:  Current _mode_index on the OverlayWindow (-1 or ≥ 0).
        mode:        Current mode object (may be VisionPipeline or a plain mode).
        total_modes: Length of the mode list (used only for bounds-clamping).

    Returns:
        Integer in [0, total_modes - 1] to use as the cycle-from origin.
    """
    # Fast path: if mode_index is already a valid position, return it directly.
    # This handles the common case (no pipeline, normal mode is active).
    if 0 <= mode_index < total_modes:
        return mode_index                   # already valid — return as-is, O(1)

    # mode_index is -1 (pipeline sentinel) or otherwise out of range.
    # We have no stored record of which base mode the pipeline was wrapping
    # (VisionPipeline does not track an index — it only holds a mode reference).
    # Safest fallback: return 0 so the next cycle call produces index 1.
    # The caller updates _base_mode_index after every cycle, so subsequent
    # presses advance normally (1 → 2 → 3 … not 0 → 1 → 0 → 1).
    return 0                                # sentinel fallback to start of mode list


def apply_base_mode_change(
    current_mode: "BaseVisionMode",
    new_base:     "BaseVisionMode",
    new_index:    int,
) -> Tuple["BaseVisionMode", int]:
    """
    Applies a base-mode change while preserving any active VisionPipeline.

    This is the single authoritative decision point for all mode transitions
    inside OverlayWindow.  Both cycle_mode() and set_mode() call this function
    so the preservation logic lives in exactly one place and can be tested
    independently of the Qt widget layer.

    Behaviour
    ─────────
    Pipeline active (current_mode is VisionPipeline):
        1. Writes new_base into pipeline.base_mode  (in-place, O(1)).
           → The effect chain (e.g., HD Matrix Analyzer) is untouched.
        2. Returns (current_mode, -1)
           → Caller stores the SAME pipeline object; HUD keeps showing '≡'.

    No pipeline (current_mode is a plain BaseVisionMode):
        1. Returns (new_base, new_index)
           → Caller replaces _mode and stores the numeric index for the HUD.

    Thread safety
    ─────────────
    The assignment `current_mode.base_mode = new_base` is a single attribute
    write.  Under CPython's GIL this is atomic: the worker thread calling
    current_mode.apply() at the same instant sees either the old base_mode or
    the new one — never a partially-written object reference (no torn read).

    Args:
        current_mode: The mode currently assigned to the OverlayWindow._mode.
        new_base:     The new plain BaseVisionMode selected by the user.
        new_index:    Registry index of new_base (stored when no pipeline).

    Returns:
        Tuple (mode_to_assign, index_to_store).  Unpack directly into the
        OverlayWindow's _mode and _mode_index attributes:
            self._mode, self._mode_index = apply_base_mode_change(...)
    """
    # Runtime-local import — avoids module-level circular dependency.
    # Python's module cache makes this effectively free after the first call.
    from core.vision_pipeline import VisionPipeline  # local import — avoids circular dep

    # ── Case A: a VisionPipeline is currently active ───────────────────────
    if isinstance(current_mode, VisionPipeline):
        # Patch the pipeline's base_mode attribute in-place.
        # `base_mode` has a property setter in VisionPipeline; we are writing
        # through it, which is a single GIL-atomic attribute assignment.
        # The effect list (_effects) is never touched — it stays intact.
        current_mode.base_mode = new_base   # swap only the base; effects unchanged

        # Return the SAME pipeline object so the caller stores it back into
        # _mode (no allocation, no new object).  The index -1 tells the HUD
        # to display the '≡' pipeline indicator instead of a numeric counter.
        return current_mode, -1             # pipeline preserved, sentinel index

    # ── Case B: no pipeline active — straightforward replacement ──────────
    # Return the new plain mode and its registry index.  The HUD will render
    # "Mode [n/N]: <Name>" using new_index.
    return new_base, new_index              # full replacement, numeric index
