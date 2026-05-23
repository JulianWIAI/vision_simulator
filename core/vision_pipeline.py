"""
Vision Pipeline — Phase 6

A BaseVisionMode subclass that chains a base vision mode with an ordered
list of post-processing effects.  Frame processing flows sequentially:

    raw frame → base_mode.apply() → effect_0.apply() → … → output

Thread model
────────────
apply() runs on the worker thread.  The base_mode and effects list are
written once from the GUI thread before the pipeline is active; no
mutation occurs during apply(), so no explicit lock is required
(identical GIL-atomic-write pattern used throughout the codebase).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from modes.base_mode import BaseVisionMode


class VisionPipeline(BaseVisionMode):
    """
    Sequential composition of a base vision mode and zero or more effects.

    Each effect is itself a BaseVisionMode, so any mode can act as a
    post-processing step.  When no effects are attached the pipeline is a
    transparent pass-through wrapper around the base mode.
    """

    def __init__(
        self,
        base_mode: BaseVisionMode,
        effects: Optional[List[BaseVisionMode]] = None,
    ) -> None:
        self._base_mode = base_mode
        self._effects: List[BaseVisionMode] = list(effects) if effects else []

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        if not self._effects:
            return self._base_mode.name
        effect_part = " + ".join(e.name for e in self._effects)
        return f"{self._base_mode.name} + {effect_part}"

    @property
    def base_mode(self) -> BaseVisionMode:
        return self._base_mode

    @base_mode.setter
    def base_mode(self, mode: BaseVisionMode) -> None:
        self._base_mode = mode

    @property
    def effects(self) -> List[BaseVisionMode]:
        """Read-only view — use set_effects() to replace the list."""
        return list(self._effects)

    # ── Mutation (GUI thread only) ─────────────────────────────────────────

    def set_effects(self, effects: List[BaseVisionMode]) -> None:
        """
        Replaces the effect list atomically.

        Must be called from the GUI thread.  The list re-assignment is a
        single attribute write — atomic under CPython's GIL — so
        worker-thread reads of self._effects see either the old or the
        new list, never a partially-updated one.

        Args:
            effects: Ordered list of effect instances (empty = no effects).
        """
        self._effects = list(effects)

    # ── Processing (worker thread) ─────────────────────────────────────────

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies base_mode then each effect in order.

        Args:
            frame: BGR uint8 (H, W, 3).

        Returns:
            BGR uint8 (H, W, 3) after all stages.
        """
        result = self._base_mode.apply(frame)
        for effect in self._effects:        # single read — atomic under GIL
            result = effect.apply(result)
        return result
