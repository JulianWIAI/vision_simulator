"""
Base Vision Mode

Defines the abstract contract that every vision mode must satisfy.
Using an abstract base class (ABC) enforces the interface at
definition time and enables the engine to treat all modes identically
through polymorphism — it never needs to know which mode is active.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseVisionMode(ABC):
    """
    Abstract base class for all vision simulation modes.

    Contract:
    - Every subclass must provide a `name` property (shown in the HUD).
    - Every subclass must implement `apply(frame)` to transform a frame.

    The single apply() method is intentionally the only required
    integration point.  This keeps each mode fully encapsulated and
    independently unit-testable by simply passing a synthetic frame.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the mode, displayed in the HUD.

        Keep it short (≤ 25 characters) so it fits the bottom bar.
        """
        ...

    @property
    def description(self) -> str:
        """
        Optional one-sentence description of the biological or scientific
        basis for this mode.

        Subclasses may override this for richer in-app documentation
        or future help-screen displays.
        """
        return ""

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Transforms a raw BGR screen capture into the simulated vision.

        Args:
            frame: Input frame in BGR format, shape (H, W, 3), dtype uint8.
                   This is the raw output from ScreenCapture — no
                   pre-processing has been applied.

        Returns:
            Processed frame in BGR format with the same (H, W, 3) shape.
            The output dtype MUST be uint8 so OpenCV can display it.
            The spatial dimensions must remain unchanged so the HUD
            overlay can be composited at a predictable position.
        """
        ...

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
