"""
route_progress_metrics_v1.py

Shared turn-aware route metrics for Hallucination Engine experiments.

Purpose
-------
Short straight scenarios can measure longitudinal separation in the
ego-initial frame. Long scenarios with turns cannot.

This module projects arbitrary CARLA world positions onto a traced
route polyline and computes:

    route progress s [m]
    lateral route deviation [m]
    route tangent
    route-aligned velocity [m/s]
    center separation along route [m]
    bumper gap along route [m]
    closing speed [m/s]
    TTC [s]

It contains no TCP/NEAT/model-specific logic.

Expected route CSV columns
--------------------------
route_idx
route_progress_m
x
y

The route CSV produced by inspect_carla_test_route_v1.py has these
columns.

Sign convention
---------------
For projection lateral_m:
    positive = left of the local route tangent
    negative = right of the local route tangent

For bumper gap:
    positive = actor is ahead with physical clearance
    zero     = longitudinal bumper contact
    negative = longitudinal overlap

TTC is only reported when:
    actor is ahead
    bumper gap > 0
    closing speed > 0
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


EPS = 1e-9


# ============================================================
# Data
# ============================================================

@dataclass(frozen=True)
class RoutePoint:
    route_idx: int
    s_m: float
    x: float
    y: float


@dataclass(frozen=True)
class RouteProjection:
    segment_idx: int

    s_m: float
    lateral_m: float
    distance_to_route_m: float

    projected_x: float
    projected_y: float

    tangent_x: float
    tangent_y: float


@dataclass(frozen=True)
class RoutePairMetrics:
    ego: RouteProjection
    actor: RouteProjection

    center_gap_m: float
    bumper_gap_m: float
    lateral_separation_m: float

    ego_route_speed_mps: float
    actor_route_speed_mps: float

    closing_speed_mps: float
    ttc_s: float


# ============================================================
# Route loading
# ============================================================

def load_route_csv(
    path,
) -> list[RoutePoint]:

    path = Path(path)

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as fp:

        reader = csv.DictReader(fp)

        required = {
            "route_idx",
            "route_progress_m",
            "x",
            "y",
        }

        missing = (
            required
            - set(
                reader.fieldnames
                or []
            )
        )

        if missing:
            raise ValueError(
                "Route CSV is missing columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        for row in reader:

            rows.append(
                RoutePoint(
                    route_idx=int(
                        row["route_idx"]
                    ),
                    s_m=float(
                        row[
                            "route_progress_m"
                        ]
                    ),
                    x=float(
                        row["x"]
                    ),
                    y=float(
                        row["y"]
                    ),
                )
            )

    rows.sort(
        key=lambda p: (
            p.s_m,
            p.route_idx,
        )
    )

    if len(rows) < 2:
        raise ValueError(
            "Route requires at least "
            "two points."
        )

    # --------------------------------------------------------
    # Collapse duplicate route-progress samples.
    #
    # CARLA GlobalRoutePlanner can emit two route entries at
    # the same physical position/progress around junction
    # transitions, e.g.:
    #
    #     LANEFOLLOW -> RIGHT
    #
    # These are useful semantically but represent the same
    # geometric point. Route projection requires strictly
    # increasing arc-length samples, so retain one sample.
    # --------------------------------------------------------

    deduplicated = []

    for point in rows:

        if not deduplicated:

            deduplicated.append(
                point
            )

            continue

        previous = (
            deduplicated[-1]
        )

        if abs(
            point.s_m
            - previous.s_m
        ) <= 1e-6:

            spatial_difference = (
                math.hypot(
                    point.x
                    - previous.x,

                    point.y
                    - previous.y,
                )
            )

            # Same route progress should also represent
            # essentially the same physical location.
            if spatial_difference > 0.50:

                raise ValueError(
                    "Duplicate route_progress_m values "
                    "refer to different physical locations: "
                    f"s={point.s_m:.6f} m, "
                    f"distance={spatial_difference:.3f} m"
                )

            # Keep the first geometric sample.
            continue

        if (
            point.s_m
            < previous.s_m
        ):

            raise ValueError(
                "route_progress_m decreases "
                "after sorting."
            )

        deduplicated.append(
            point
        )

    if len(deduplicated) < 2:

        raise ValueError(
            "Route contains fewer than two "
            "unique progress samples."
        )

    return deduplicated


# ============================================================
# Polyline projection
# ============================================================

class RouteProjector:
    """
    Continuous XY -> route-progress projector.

    previous_segment_idx can be supplied to restrict the search
    around the previous projection. This avoids snapping to a
    nearby but topologically different road section at junctions.
    """

    def __init__(
        self,
        route_points: list[RoutePoint],
    ):
        if len(route_points) < 2:
            raise ValueError(
                "At least two route points "
                "are required."
            )

        self.points = list(
            route_points
        )

    @classmethod
    def from_csv(
        cls,
        path,
    ):
        return cls(
            load_route_csv(path)
        )

    @property
    def total_length_m(
        self,
    ) -> float:
        return float(
            self.points[-1].s_m
        )

    def _search_bounds(
        self,
        previous_segment_idx:
            Optional[int],
        search_back_segments: int,
        search_forward_segments: int,
    ):
        n_segments = (
            len(self.points)
            - 1
        )

        if previous_segment_idx is None:
            return (
                0,
                n_segments,
            )

        previous_segment_idx = int(
            previous_segment_idx
        )

        start = max(
            0,
            previous_segment_idx
            - int(
                search_back_segments
            ),
        )

        end = min(
            n_segments,
            previous_segment_idx
            + int(
                search_forward_segments
            )
            + 1,
        )

        return (
            start,
            end,
        )

    def project(
        self,
        x: float,
        y: float,
        previous_segment_idx:
            Optional[int] = None,
        search_back_segments: int = 8,
        search_forward_segments: int = 40,
    ) -> RouteProjection:

        x = float(x)
        y = float(y)

        (
            start,
            end,
        ) = self._search_bounds(
            previous_segment_idx,
            search_back_segments,
            search_forward_segments,
        )

        best = None

        for segment_idx in range(
            start,
            end,
        ):
            a = self.points[
                segment_idx
            ]

            b = self.points[
                segment_idx + 1
            ]

            dx = (
                b.x - a.x
            )

            dy = (
                b.y - a.y
            )

            length_sq = (
                dx * dx
                + dy * dy
            )

            if length_sq <= EPS:
                continue

            segment_length = (
                math.sqrt(
                    length_sq
                )
            )

            u = (
                (
                    x - a.x
                ) * dx
                +
                (
                    y - a.y
                ) * dy
            ) / length_sq

            # ------------------------------------------------
            # Endpoint extrapolation
            #
            # GlobalRoutePlanner's first route waypoint may
            # start slightly ahead of the actual CARLA spawn
            # transform.
            #
            # Example in Town10HD_Opt spawn 10:
            #
            #     actual ego start
            #          |
            #          | ~0.5 m
            #          v
            #     route.csv s=0
            #
            # Clamping the first segment to u=0 incorrectly
            # reports ego route progress as exactly 0 and
            # shortens actor-to-ego separation by ~0.5 m.
            #
            # Allow a small extrapolation before the first
            # segment and after the final segment.
            # ------------------------------------------------

            max_endpoint_extrapolation_m = (
                5.0
            )

            if (
                segment_idx == 0
                and
                u < 0.0
            ):

                max_back_u = (
                    max_endpoint_extrapolation_m
                    /
                    segment_length
                )

                u_clamped = max(
                    -max_back_u,
                    u,
                )

            elif (
                segment_idx
                == len(self.points) - 2
                and
                u > 1.0
            ):

                max_forward_u = (
                    max_endpoint_extrapolation_m
                    /
                    segment_length
                )

                u_clamped = min(
                    1.0
                    + max_forward_u,
                    u,
                )

            else:

                u_clamped = max(
                    0.0,
                    min(
                        1.0,
                        u,
                    ),
                )

            px = (
                a.x
                + u_clamped * dx
            )

            py = (
                a.y
                + u_clamped * dy
            )

            ex = (
                x - px
            )

            ey = (
                y - py
            )

            distance_sq = (
                ex * ex
                + ey * ey
            )

            if (
                best is not None
                and
                distance_sq
                >= best[0]
            ):
                continue

            tangent_x = (
                dx
                / segment_length
            )

            tangent_y = (
                dy
                / segment_length
            )

            # Signed left/right deviation:
            #
            # tangent cross error
            # = tx*ey - ty*ex
            lateral = (
                tangent_x * ey
                - tangent_y * ex
            )

            # Prefer the route CSV's cumulative s values
            # rather than assuming Euclidean segment length
            # exactly equals delta route_progress_m.
            s_m = (
                a.s_m
                + u_clamped
                * (
                    b.s_m
                    - a.s_m
                )
            )

            best = (
                distance_sq,
                RouteProjection(
                    segment_idx=(
                        segment_idx
                    ),
                    s_m=float(s_m),
                    lateral_m=float(
                        lateral
                    ),
                    distance_to_route_m=(
                        math.sqrt(
                            distance_sq
                        )
                    ),
                    projected_x=float(
                        px
                    ),
                    projected_y=float(
                        py
                    ),
                    tangent_x=float(
                        tangent_x
                    ),
                    tangent_y=float(
                        tangent_y
                    ),
                ),
            )

        if best is None:

            # Defensive full-route retry in case a pathological
            # short/zero segment prevented the local search.
            if (
                previous_segment_idx
                is not None
            ):
                return self.project(
                    x=x,
                    y=y,
                    previous_segment_idx=None,
                )

            raise RuntimeError(
                "Could not project point "
                "onto route."
            )

        projection = best[1]

        # If a local-window projection is implausibly far from
        # the route, retry globally. This is useful after a reset
        # or large teleport.
        if (
            previous_segment_idx
            is not None
            and
            projection
            .distance_to_route_m
            > 15.0
        ):
            return self.project(
                x=x,
                y=y,
                previous_segment_idx=None,
            )

        return projection


# ============================================================
# Velocity
# ============================================================

def route_aligned_speed_mps(
    velocity_x: float,
    velocity_y: float,
    projection: RouteProjection,
) -> float:

    return float(
        float(velocity_x)
        * projection.tangent_x
        +
        float(velocity_y)
        * projection.tangent_y
    )


# ============================================================
# Pair metrics
# ============================================================

def compute_route_pair_metrics(
    projector: RouteProjector,

    ego_x: float,
    ego_y: float,
    ego_vx: float,
    ego_vy: float,

    actor_x: float,
    actor_y: float,

    ego_length_m: float,
    actor_length_m: float,

    # Preferred for resolved ScenarioGenerator actors:
    # pass their scalar resolved speed directly.
    actor_route_speed_mps:
        Optional[float] = None,

    # Optional alternative if a physical actor world velocity
    # is available.
    actor_vx: Optional[float] = None,
    actor_vy: Optional[float] = None,

    previous_ego_segment_idx:
        Optional[int] = None,

    previous_actor_segment_idx:
        Optional[int] = None,
) -> RoutePairMetrics:

    ego_projection = (
        projector.project(
            x=ego_x,
            y=ego_y,
            previous_segment_idx=(
                previous_ego_segment_idx
            ),
        )
    )

    actor_projection = (
        projector.project(
            x=actor_x,
            y=actor_y,
            previous_segment_idx=(
                previous_actor_segment_idx
            ),
        )
    )

    center_gap_m = (
        actor_projection.s_m
        - ego_projection.s_m
    )

    half_lengths_m = (
        0.5
        * (
            float(
                ego_length_m
            )
            +
            float(
                actor_length_m
            )
        )
    )

    bumper_gap_m = (
        center_gap_m
        - half_lengths_m
    )

    lateral_separation_m = abs(
        actor_projection.lateral_m
        - ego_projection.lateral_m
    )

    ego_route_speed = (
        route_aligned_speed_mps(
            ego_vx,
            ego_vy,
            ego_projection,
        )
    )

    if actor_route_speed_mps is not None:

        actor_route_speed = float(
            actor_route_speed_mps
        )

    elif (
        actor_vx is not None
        and
        actor_vy is not None
    ):

        actor_route_speed = (
            route_aligned_speed_mps(
                actor_vx,
                actor_vy,
                actor_projection,
            )
        )

    else:
        raise ValueError(
            "Provide actor_route_speed_mps "
            "or actor_vx/actor_vy."
        )

    closing_speed = (
        ego_route_speed
        - actor_route_speed
    )

    if (
        center_gap_m > 0.0
        and
        bumper_gap_m > 0.0
        and
        closing_speed > EPS
    ):
        ttc_s = (
            bumper_gap_m
            / closing_speed
        )
    else:
        ttc_s = float("inf")

    return RoutePairMetrics(
        ego=ego_projection,
        actor=actor_projection,

        center_gap_m=float(
            center_gap_m
        ),

        bumper_gap_m=float(
            bumper_gap_m
        ),

        lateral_separation_m=float(
            lateral_separation_m
        ),

        ego_route_speed_mps=float(
            ego_route_speed
        ),

        actor_route_speed_mps=float(
            actor_route_speed
        ),

        closing_speed_mps=float(
            closing_speed
        ),

        ttc_s=float(
            ttc_s
        ),
    )


def route_virtual_collision(
    metrics: RoutePairMetrics,

    ego_length_m: float,
    actor_length_m: float,

    ego_width_m: float,
    actor_width_m: float,

    longitudinal_margin_m: float = 0.0,
    lateral_margin_m: float = 0.0,
) -> bool:
    """
    Route-coordinate overlap test for vehicles
    following approximately the same route/lane.

    This is intended for the long route-following
    benchmark, not arbitrary crossing-vehicle OBB
    collision testing.
    """

    half_length_sum = (
        0.5
        * (
            float(ego_length_m)
            +
            float(actor_length_m)
        )
        +
        float(longitudinal_margin_m)
    )

    longitudinal_overlap = (
        abs(
            metrics.center_gap_m
        )
        <= half_length_sum
    )

    half_width_sum = (
        0.5
        * (
            float(ego_width_m)
            +
            float(actor_width_m)
        )
        +
        float(lateral_margin_m)
    )

    lateral_overlap = (
        metrics.lateral_separation_m
        <= half_width_sum
    )

    return bool(
        longitudinal_overlap
        and
        lateral_overlap
    )
