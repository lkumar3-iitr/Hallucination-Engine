"""Production Hallucination Engine sprite renderer."""

from .compositor import HESpriteRendererCompositor
from .renderer import HESpriteRenderer

__all__ = ["HESpriteRenderer", "HESpriteRendererCompositor", "HECalibratedCompositor"]
__version__ = "he_calibrated_renderer_v2"


def __getattr__(name):
    if name == "HECalibratedCompositor":
        from .calibrated.compositor import HECalibratedCompositor
        return HECalibratedCompositor
    raise AttributeError(name)
