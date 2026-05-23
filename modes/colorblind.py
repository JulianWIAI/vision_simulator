"""
Color Blindness Simulation Mode

Simulates the three most common forms of colour vision deficiency (CVD):

Protanopia   — absence of the L-cone (red-sensitive).
               Red appears very dark; ~1 % of males.
Deuteranopia — absence of the M-cone (green-sensitive).
               Red and green are confused; ~1 % of males.
Tritanopia   — absence of the S-cone (blue-sensitive).
               Blue and yellow are confused; very rare (~0.003 %).

These simulations are widely used in UI/UX accessibility testing to
verify that interfaces remain usable for people with CVD.

Simulation matrices sourced from:
    Viénot, Brettel & Mollon (1999); Brettel, Viénot & Mollon (1997).
"""

import numpy as np
from modes.base_mode import BaseVisionMode
from effects.color_filters import (
    simulate_protanopia,
    simulate_deuteranopia,
    simulate_tritanopia,
)


class ColorBlindVision(BaseVisionMode):
    """
    Applies one of three CVD simulation matrices to the frame.

    The active subtype is fixed at construction time so the mode can
    be instantiated multiple times (one per subtype) and registered
    individually in the mode list.
    """

    # Maps subtype names to their simulation functions.
    _FILTERS = {
        "deuteranopia": simulate_deuteranopia,
        "protanopia":   simulate_protanopia,
        "tritanopia":   simulate_tritanopia,
    }

    def __init__(self, subtype: str = "deuteranopia") -> None:
        """
        Args:
            subtype: Which CVD to simulate.
                     One of 'deuteranopia', 'protanopia', 'tritanopia'.

        Raises:
            ValueError: If subtype is not a recognised CVD type.
        """
        if subtype not in self._FILTERS:
            raise ValueError(
                f"Unknown CVD subtype '{subtype}'. "
                f"Valid options: {list(self._FILTERS.keys())}"
            )
        self.subtype = subtype
        self._filter = self._FILTERS[subtype]

    @property
    def name(self) -> str:
        return f"Color Blind ({self.subtype.capitalize()})"

    @property
    def description(self) -> str:
        return f"Simulates {self.subtype} colour vision deficiency."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the CVD simulation matrix.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            CVD-simulated BGR frame.
        """
        return self._filter(frame)
