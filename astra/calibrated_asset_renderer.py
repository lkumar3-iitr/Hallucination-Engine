"""Asset-neutral entry point to the validated calibrated-hull algorithm.

The inherited historical bus name remains untouched to preserve its checkpoint.
All active geometry comes from each asset's native metadata and capture bank.
"""
import json
from pathlib import Path

from .calibrated_bus_renderer import CalibratedBusRenderer


class CalibratedAssetRenderer(CalibratedBusRenderer):
    def __init__(self, native_bank, close_bank):
        native = json.loads((Path(native_bank)/"asset_metadata.json").read_text())
        close = json.loads((Path(close_bank)/"config.json").read_text())
        if native["carla_blueprint"] != close["blueprint"]:
            raise ValueError("Native and close banks belong to different actor blueprints")
        super().__init__(native_bank, close_bank)
