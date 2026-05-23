"""
Emotional Vision Mode

Cognitive neuroscience research shows that emotional state measurably
alters visual processing:

- Happiness (Fredrickson 2001, "broaden-and-build"): widens attentional
  scope, increases perceived colour saturation and warmth.
- Fear (LeDoux 1994, amygdala hijack): tunnels attention toward threats,
  desaturates peripheral colour, raises contrast globally.
- Anger (Harmon-Jones 2003): biases red-channel processing, narrows
  field of view, raises contrast.
- Sadness (Gilboa 2002, mood-congruent encoding): reduces engagement,
  desaturates and darkens the entire scene.

This mode applies colour, contrast, and spatial transforms that proxy
the known perceptual effects of each emotional state.
"""

import numpy as np
from modes.base_mode import BaseVisionMode
from effects.color_filters import adjust_saturation, apply_channel_weights
from effects.contrast import adjust_contrast_brightness, gamma_correction
from effects.overlays import add_vignette


class EmotionalVision(BaseVisionMode):
    """
    Simulates emotionally modulated visual perception.

    The emotion is fixed at construction time.  Multiple instances
    (one per emotion) can be registered simultaneously in the mode list.
    """

    VALID_EMOTIONS = ("happy", "fearful", "angry", "sad")

    def __init__(self, emotion: str = "happy") -> None:
        """
        Args:
            emotion: One of 'happy', 'fearful', 'angry', 'sad'.

        Raises:
            ValueError: If the emotion is not supported.
        """
        if emotion not in self.VALID_EMOTIONS:
            raise ValueError(
                f"emotion must be one of {self.VALID_EMOTIONS}, got '{emotion}'"
            )
        self.emotion = emotion
        self._filter = getattr(self, f"_{emotion}")

    @property
    def name(self) -> str:
        return f"Emotional: {self.emotion.capitalize()}"

    @property
    def description(self) -> str:
        return f"Perceptual bias of the '{self.emotion}' emotional state."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Dispatches to the appropriate emotional filter.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Emotionally filtered BGR frame.
        """
        return self._filter(frame)

    # ── Emotional filters ─────────────────────────────────────────────

    def _happy(self, frame: np.ndarray) -> np.ndarray:
        """
        Warm, bright, highly saturated — expanded attentional breadth.

        Red/orange boost raises perceived warmth; saturation increase
        reflects the broader colour gamut of positive affect.
        """
        warm = apply_channel_weights(frame, b=0.75, g=1.0, r=1.25)
        return adjust_saturation(warm, factor=1.5)

    def _fearful(self, frame: np.ndarray) -> np.ndarray:
        """
        High contrast, desaturated, tight vignette — hypervigilant tunnel.

        The amygdala response suppresses peripheral colour processing while
        boosting central luminance contrast for threat detection.
        """
        high_contrast = adjust_contrast_brightness(frame, alpha=1.55, beta=-25)
        desaturated   = adjust_saturation(high_contrast, factor=0.50)
        return add_vignette(desaturated, strength=0.55)

    def _angry(self, frame: np.ndarray) -> np.ndarray:
        """
        Red-dominant, high contrast, moderate vignette.

        Anger biases attention toward red stimuli and narrows the
        visual field via increased arousal-driven vasoconstriction.
        """
        red_biased = apply_channel_weights(frame, b=0.45, g=0.70, r=1.60)
        contrasted  = adjust_contrast_brightness(red_biased, alpha=1.45, beta=0)
        return add_vignette(contrasted, strength=0.40)

    def _sad(self, frame: np.ndarray) -> np.ndarray:
        """
        Cool, heavily desaturated, dark — reduced perceptual engagement.

        Sadness is associated with hypo-arousal; the world appears
        drained of colour and slightly dim.
        """
        cold        = apply_channel_weights(frame, b=1.15, g=0.85, r=0.65)
        desaturated = adjust_saturation(cold, factor=0.25)
        return gamma_correction(desaturated, gamma=1.50)
