"""Route-triggered clock for deterministic scenario actor trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RouteTriggeredEventClock:
    """Map execution frames to authored actor frames around a route trigger.

    Before the ego reaches ``trigger_route_progress_m``, the authored actor
    trajectory advances only as far as its event staging frame. At the first
    execution frame after the threshold is observed, the event starts from
    that exact staging frame. CARLA and HE therefore use the same authored
    state for every trigger-relative frame even when their egos approach at
    different speeds.
    """

    trigger_route_progress_m: Optional[float]
    event_source_frame: int
    pre_trigger_source_frame: Optional[int] = None
    trigger_frame: Optional[int] = None
    trigger_observed_progress_m: Optional[float] = None

    def source_frame(self, execution_frame: int) -> int:
        execution_frame = int(execution_frame)
        if self.trigger_route_progress_m is None:
            return execution_frame
        if self.trigger_frame is None:
            staging_frame = (
                int(self.event_source_frame)
                if self.pre_trigger_source_frame is None
                else int(self.pre_trigger_source_frame)
            )
            return min(execution_frame, staging_frame)
        return int(self.event_source_frame) + max(
            0, execution_frame - int(self.trigger_frame)
        )

    def observe_for_next_frame(
        self,
        next_execution_frame: int,
        ego_route_progress_m: float,
    ) -> int:
        if (
            self.trigger_route_progress_m is not None
            and self.trigger_frame is None
            and float(ego_route_progress_m)
            >= float(self.trigger_route_progress_m)
        ):
            self.trigger_frame = int(next_execution_frame)
            self.trigger_observed_progress_m = float(ego_route_progress_m)
        return self.source_frame(next_execution_frame)

    def relative_frame(self, execution_frame: int) -> Optional[int]:
        if self.trigger_frame is None:
            return None
        return int(execution_frame) - int(self.trigger_frame)
