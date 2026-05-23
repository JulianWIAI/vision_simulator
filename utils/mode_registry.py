"""
Mode Registry

The single location where all vision modes are listed and instantiated.

To add a new mode:
  1. Create modes/<your_mode>.py implementing BaseVisionMode.
  2. Import it below and add an instance to the list in get_all_modes().

The engine reads this list at startup.  No engine code ever needs to
change when a new mode is added — only this file.
"""

from typing import List
from modes.base_mode import BaseVisionMode

from modes.dog_vision     import DogVision
from modes.cat_vision     import CatVision
from modes.bird_vision    import BirdVision
from modes.insect_vision  import InsectVision
from modes.snake_thermal  import SnakeThermalVision
from modes.pit_viper      import PitViperVision
from modes.shark_vision   import SharkVision
from modes.frog_vision    import FrogVision
from modes.uv_vision      import UVVision
from modes.depth_map      import DepthMapVision
from modes.ai_edge        import AIEdgeVision
from modes.colorblind     import ColorBlindVision
from modes.tunnel_vision    import TunnelVision
from modes.emotional        import EmotionalVision
from modes.polarized_vision import PolarizedVision


def get_all_modes() -> List[BaseVisionMode]:
    """
    Returns the ordered list of all registered vision modes.

    Position in this list determines the hotkey mapping:
      index 0  → key '1'
      index 1  → key '2'
      …
      index 8  → key '9'
      index 9  → key '0'
    Modes beyond index 9 are reachable via N (next) / P (previous) keys.

    Returns:
        List of instantiated BaseVisionMode subclasses.
    """
    return [
        DogVision(),                            # Key 1
        CatVision(),                            # Key 2
        BirdVision(),                           # Key 3
        InsectVision(),                         # Key 4
        SnakeThermalVision(),                   # Key 5
        PitViperVision(),                       # Key 6  ★ HIGH PRIORITY
        SharkVision(),                          # Key 7
        FrogVision(),                           # Key 8
        UVVision(),                             # Key 9
        DepthMapVision(),                       # Key 0
        AIEdgeVision(),                         # N/P navigation
        ColorBlindVision("deuteranopia"),       # N/P navigation
        ColorBlindVision("protanopia"),         # N/P navigation
        ColorBlindVision("tritanopia"),         # N/P navigation
        TunnelVision(),                         # N/P navigation
        EmotionalVision("happy"),               # N/P navigation
        EmotionalVision("sad"),                 # N/P navigation
        EmotionalVision("angry"),               # N/P navigation
        EmotionalVision("fearful"),             # N/P navigation
        PolarizedVision(),                      # N/P navigation
    ]
