"""
Dog Vision Mode

Dogs are dichromats — they possess only two types of cone cells: one
sensitive to blue-violet (~430 nm) and one to yellow-green (~555 nm).
They cannot distinguish red from green; both appear as the same
muddy yellow-brown tone.

Key characteristics simulated here:
- Dichromatic colour space (deuteranopia-like matrix)
- Slightly lower overall colour saturation
- Mild luminance contrast boost (dogs have superior motion/contrast detection)
"""

import numpy as np
from modes.base_mode import BaseVisionMode
from effects.color_filters import simulate_deuteranopia, adjust_saturation
from effects.contrast import adjust_contrast_brightness


class DogVision(BaseVisionMode):
    """
    Simulates canine dichromatic vision.

    The deuteranopia simulation matrix was chosen because it best
    approximates the known spectral sensitivity of the canine L-cone
    (peak ~555 nm) and S-cone (peak ~429 nm).
    """

    @property
    def name(self) -> str:
        return "Dog Vision"

    @property
    def description(self) -> str:
        return "Dichromatic blue-yellow vision with reduced saturation."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies canine dichromatic filtering.

        Steps:
        1. Deuteranopia matrix — removes green cone contribution.
        2. Saturation reduction — dogs perceive less vivid colours.
        3. Mild contrast boost — better low-light / motion discrimination.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Processed BGR frame simulating canine vision.
        """
        # Remove green-cone sensitivity via colour matrix
        filtered = simulate_deuteranopia(frame)

        # Dogs experience lower colour saturation than humans
        desaturated = adjust_saturation(filtered, factor=0.75)

        # Mild contrast boost simulates the dog's motion-detection advantage
        return adjust_contrast_brightness(desaturated, alpha=1.1, beta=5)
