"""
sprite_native_geometry_v1.py

Generic sprite-native visual geometry for Hallucination Engine.

Purpose
-------
Predict the apparent visible width / height of a sprite in an arbitrary
target camera without using hand-written physical/render dimensions.

The geometry is derived entirely from the sprite bank:

    viewpoint angle
    elevation
    depth
        ->
    target visible width / height

Interpolation
-------------
Angle:
    circular linear interpolation.

Elevation:
    linear interpolation.

Depth:
    inverse-depth interpolation inside the measured range.

Outside the measured distance range:
    effective-depth model fitted automatically from the bank:

        q(Z) = A / (Z - delta)

where:

        q = visible_pixels / focal_length

This keeps the runtime geometry file self-contained. No separate fitted
CSV is required.

The geometry_alpha_threshold stored in sprite_geometry_v1.csv defines
the visible silhouette convention and should match the compositor when
sprite-native geometry is used.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


# ============================================================
# Camera geometry
# ============================================================

def focal_length_px(
    image_width_px,
    horizontal_fov_deg,
):
    return (
        float(image_width_px)
        /
        (
            2.0
            *
            math.tan(
                math.radians(
                    float(horizontal_fov_deg)
                )
                /
                2.0
            )
        )
    )


def _lerp(
    a,
    b,
    t,
):
    return (
        float(a)
        +
        float(t)
        *
        (
            float(b)
            -
            float(a)
        )
    )


# ============================================================
# Effective-depth fit
# ============================================================

def _fit_effective_depth(
    distances,
    q_values,
):
    """
    Fit:

        q(Z) = A / (Z - delta)

    Rearranged:

        1 / q = (1/A) Z - delta/A

    Returns:

        A
        delta
    """

    x = np.asarray(
        distances,
        dtype=np.float64,
    )

    q = np.asarray(
        q_values,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(q)
        &
        (
            q > 1e-12
        )
    )

    x = x[
        valid
    ]

    q = q[
        valid
    ]

    if len(x) < 2:
        return None

    y = (
        1.0
        /
        q
    )

    design = np.column_stack(
        [
            x,
            np.ones_like(
                x
            ),
        ]
    )

    solution, _, _, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    slope = float(
        solution[
            0
        ]
    )

    intercept = float(
        solution[
            1
        ]
    )

    if (
        not math.isfinite(
            slope
        )
        or
        abs(
            slope
        )
        < 1e-12
    ):
        return None

    apparent_extent = (
        1.0
        /
        slope
    )

    depth_offset = (
        -intercept
        /
        slope
    )

    if (
        not math.isfinite(
            apparent_extent
        )
        or
        not math.isfinite(
            depth_offset
        )
    ):
        return None

    return {
        "apparent_extent":
            float(
                apparent_extent
            ),

        "depth_offset":
            float(
                depth_offset
            ),
    }


# ============================================================
# Main geometry class
# ============================================================

class SpriteNativeGeometryV1:

    def __init__(
        self,
        geometry_csv,
    ):
        self.geometry_csv = (
            Path(
                geometry_csv
            )
            .expanduser()
            .resolve()
        )

        if not self.geometry_csv.exists():
            raise FileNotFoundError(
                f"Sprite geometry CSV not found: "
                f"{self.geometry_csv}"
            )

        self.table = {}

        self.angles = set()
        self.elevations = set()
        self.distances = set()

        self.geometry_version = None
        self.geometry_alpha_threshold = None

        self._fits = {}

        self._load()

    # ========================================================
    # Loading
    # ========================================================

    def _load(
        self,
    ):
        with self.geometry_csv.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as fp:

            reader = csv.DictReader(
                fp
            )

            rows = list(
                reader
            )

        if not rows:
            raise RuntimeError(
                f"Empty sprite geometry CSV: "
                f"{self.geometry_csv}"
            )

        versions = {
            str(
                row[
                    "geometry_version"
                ]
            )
            for row in rows
        }

        thresholds = {
            int(
                float(
                    row[
                        "geometry_alpha_threshold"
                    ]
                )
            )
            for row in rows
        }

        if len(
            versions
        ) != 1:
            raise RuntimeError(
                "Geometry CSV contains multiple "
                f"geometry versions: {versions}"
            )

        if len(
            thresholds
        ) != 1:
            raise RuntimeError(
                "Geometry CSV contains multiple "
                f"alpha thresholds: {thresholds}"
            )

        self.geometry_version = (
            next(
                iter(
                    versions
                )
            )
        )

        self.geometry_alpha_threshold = int(
            next(
                iter(
                    thresholds
                )
            )
        )

        for row in rows:

            angle = int(
                round(
                    float(
                        row[
                            "angle_deg"
                        ]
                    )
                )
            )

            angle = (
                angle
                %
                360
            )

            elevation = float(
                row[
                    "elevation_deg"
                ]
            )

            distance = float(
                row[
                    "source_depth_m"
                ]
            )

            alpha_width = float(
                row[
                    "alpha_width_px"
                ]
            )

            alpha_height = float(
                row[
                    "alpha_height_px"
                ]
            )

            source_fx = float(
                row[
                    "fx_px"
                ]
            )

            source_fy = float(
                row[
                    "fy_px"
                ]
            )

            if (
                source_fx <= 0.0
                or
                source_fy <= 0.0
            ):
                raise RuntimeError(
                    "Invalid focal length in "
                    f"{self.geometry_csv}"
                )

            q_width = (
                alpha_width
                /
                source_fx
            )

            q_height = (
                alpha_height
                /
                source_fy
            )

            key = (
                angle,
                elevation,
                distance,
            )

            if key in self.table:
                raise RuntimeError(
                    "Duplicate sprite geometry key: "
                    f"{key}"
                )

            self.table[
                key
            ] = {
                "q_width":
                    float(
                        q_width
                    ),

                "q_height":
                    float(
                        q_height
                    ),

                "alpha_width_px":
                    float(
                        alpha_width
                    ),

                "alpha_height_px":
                    float(
                        alpha_height
                    ),
            }

            self.angles.add(
                angle
            )

            self.elevations.add(
                elevation
            )

            self.distances.add(
                distance
            )

        self.angles = sorted(
            self.angles
        )

        self.elevations = sorted(
            self.elevations
        )

        self.distances = sorted(
            self.distances
        )

        if not self.angles:
            raise RuntimeError(
                "No angles in sprite geometry table."
            )

        if not self.elevations:
            raise RuntimeError(
                "No elevations in sprite geometry table."
            )

        if not self.distances:
            raise RuntimeError(
                "No distances in sprite geometry table."
            )

        self._build_effective_depth_fits()

    # ========================================================
    # Fit extrapolation models
    # ========================================================

    def _build_effective_depth_fits(
        self,
    ):
        for angle in self.angles:

            for elevation in self.elevations:

                distances = []
                q_widths = []
                q_heights = []

                for distance in self.distances:

                    key = (
                        angle,
                        elevation,
                        distance,
                    )

                    item = self.table.get(
                        key
                    )

                    if item is None:
                        continue

                    distances.append(
                        float(
                            distance
                        )
                    )

                    q_widths.append(
                        float(
                            item[
                                "q_width"
                            ]
                        )
                    )

                    q_heights.append(
                        float(
                            item[
                                "q_height"
                            ]
                        )
                    )

                width_fit = (
                    _fit_effective_depth(
                        distances,
                        q_widths,
                    )
                )

                height_fit = (
                    _fit_effective_depth(
                        distances,
                        q_heights,
                    )
                )

                self._fits[
                    (
                        angle,
                        elevation,
                    )
                ] = {
                    "width":
                        width_fit,

                    "height":
                        height_fit,
                }

    # ========================================================
    # Bracketing
    # ========================================================

    def _elevation_bracket(
        self,
        elevation_deg,
    ):
        value = float(
            elevation_deg
        )

        values = self.elevations

        if value <= values[0]:
            return (
                values[0],
                values[0],
                0.0,
            )

        if value >= values[-1]:
            return (
                values[-1],
                values[-1],
                0.0,
            )

        for index in range(
            len(values) - 1
        ):

            low = values[
                index
            ]

            high = values[
                index + 1
            ]

            if (
                low
                <=
                value
                <=
                high
            ):

                weight = (
                    value - low
                ) / (
                    high - low
                )

                return (
                    low,
                    high,
                    float(
                        weight
                    ),
                )

        raise RuntimeError(
            "Could not bracket elevation."
        )

    def _angle_bracket(
        self,
        angle_deg,
    ):
        """
        Circular bracketing that also works if a future sprite bank
        uses angle increments larger than one degree.
        """

        query = (
            float(
                angle_deg
            )
            %
            360.0
        )

        values = self.angles

        if len(
            values
        ) == 1:
            return (
                values[0],
                values[0],
                0.0,
            )

        for index in range(
            len(values)
        ):

            low = float(
                values[
                    index
                ]
            )

            high = float(
                values[
                    (
                        index + 1
                    )
                    %
                    len(values)
                ]
            )

            if index == (
                len(values) - 1
            ):
                high += 360.0

            q = query

            if (
                index
                ==
                len(values) - 1
                and
                q
                <
                low
            ):
                q += 360.0

            if (
                low
                <=
                q
                <=
                high
            ):

                if abs(
                    high - low
                ) < 1e-12:

                    weight = 0.0

                else:

                    weight = (
                        q - low
                    ) / (
                        high - low
                    )

                return (
                    int(
                        low
                    )
                    %
                    360,

                    int(
                        high
                    )
                    %
                    360,

                    float(
                        weight
                    ),
                )

        raise RuntimeError(
            "Could not bracket viewpoint angle."
        )

    # ========================================================
    # Depth prediction at one angle/elevation corner
    # ========================================================

    def _predict_corner(
        self,
        angle_deg,
        elevation_deg,
        depth_m,
    ):
        z = float(
            depth_m
        )

        if (
            not math.isfinite(
                z
            )
            or
            z <= 0.0
        ):
            return None

        distances = self.distances

        # ----------------------------------------------------
        # Exact measured anchor
        # ----------------------------------------------------

        for distance in distances:

            if abs(
                z - distance
            ) < 1e-9:

                item = self.table[
                    (
                        angle_deg,
                        elevation_deg,
                        distance,
                    )
                ]

                return {
                    "q_width":
                        float(
                            item[
                                "q_width"
                            ]
                        ),

                    "q_height":
                        float(
                            item[
                                "q_height"
                            ]
                        ),

                    "depth_mode":
                        "measured",

                    "distance_low_m":
                        float(
                            distance
                        ),

                    "distance_high_m":
                        float(
                            distance
                        ),

                    "distance_weight":
                        0.0,
                }

        # ----------------------------------------------------
        # Interpolate inside measured distance range.
        # ----------------------------------------------------

        if (
            distances[0]
            <
            z
            <
            distances[-1]
        ):

            low = None
            high = None

            for index in range(
                len(distances) - 1
            ):

                a = distances[
                    index
                ]

                b = distances[
                    index + 1
                ]

                if (
                    a
                    <=
                    z
                    <=
                    b
                ):
                    low = a
                    high = b
                    break

            if (
                low is None
                or
                high is None
            ):
                raise RuntimeError(
                    "Could not bracket depth."
                )

            g0 = self.table[
                (
                    angle_deg,
                    elevation_deg,
                    low,
                )
            ]

            g1 = self.table[
                (
                    angle_deg,
                    elevation_deg,
                    high,
                )
            ]

            u = (
                1.0
                /
                z
            )

            u0 = (
                1.0
                /
                low
            )

            u1 = (
                1.0
                /
                high
            )

            weight = (
                u - u0
            ) / (
                u1 - u0
            )

            return {
                "q_width":
                    _lerp(
                        g0[
                            "q_width"
                        ],
                        g1[
                            "q_width"
                        ],
                        weight,
                    ),

                "q_height":
                    _lerp(
                        g0[
                            "q_height"
                        ],
                        g1[
                            "q_height"
                        ],
                        weight,
                    ),

                "depth_mode":
                    "inverse_depth_interpolation",

                "distance_low_m":
                    float(
                        low
                    ),

                "distance_high_m":
                    float(
                        high
                    ),

                "distance_weight":
                    float(
                        weight
                    ),
            }

        # ----------------------------------------------------
        # Extrapolate outside [minimum, maximum] using the
        # effective-depth model fitted from this sprite bank.
        # ----------------------------------------------------

        fits = self._fits.get(
            (
                angle_deg,
                elevation_deg,
            )
        )

        if fits is None:
            return None

        width_fit = fits.get(
            "width"
        )

        height_fit = fits.get(
            "height"
        )

        if (
            width_fit is None
            or
            height_fit is None
        ):
            return None

        width_denominator = (
            z
            -
            float(
                width_fit[
                    "depth_offset"
                ]
            )
        )

        height_denominator = (
            z
            -
            float(
                height_fit[
                    "depth_offset"
                ]
            )
        )

        if (
            width_denominator <= 1e-6
            or
            height_denominator <= 1e-6
        ):
            return None

        q_width = (
            float(
                width_fit[
                    "apparent_extent"
                ]
            )
            /
            width_denominator
        )

        q_height = (
            float(
                height_fit[
                    "apparent_extent"
                ]
            )
            /
            height_denominator
        )

        return {
            "q_width":
                float(
                    q_width
                ),

            "q_height":
                float(
                    q_height
                ),

            "depth_mode":
                (
                    "effective_depth_extrapolation_near"
                    if z < distances[0]
                    else
                    "effective_depth_extrapolation_far"
                ),

            "distance_low_m":
                float(
                    distances[
                        0
                    ]
                    if z < distances[0]
                    else
                    distances[
                        -1
                    ]
                ),

            "distance_high_m":
                float(
                    distances[
                        0
                    ]
                    if z < distances[0]
                    else
                    distances[
                        -1
                    ]
                ),

            "distance_weight":
                0.0,
        }

    # ========================================================
    # Public prediction
    # ========================================================

    def predict(
        self,
        viewpoint_angle_deg,
        elevation_deg,
        depth_m,
        target_width,
        target_height,
        target_fov,
    ):
        """
        Predict continuous apparent visual geometry.

        target_height is retained explicitly because camera geometry
        should be fully described by the caller, although CARLA's
        horizontal FOV convention gives fx == fy from target_width.
        """

        target_width = int(
            target_width
        )

        target_height = int(
            target_height
        )

        if (
            target_width <= 0
            or
            target_height <= 0
        ):
            raise ValueError(
                "Target camera dimensions must be positive."
            )

        target_fx = focal_length_px(
            target_width,
            target_fov,
        )

        # CARLA uses square pixels.
        target_fy = (
            target_fx
        )

        (
            angle_low,
            angle_high,
            angle_weight,
        ) = self._angle_bracket(
            viewpoint_angle_deg
        )

        (
            elevation_low,
            elevation_high,
            elevation_weight,
        ) = self._elevation_bracket(
            elevation_deg
        )

        g00 = self._predict_corner(
            angle_low,
            elevation_low,
            depth_m,
        )

        g10 = self._predict_corner(
            angle_high,
            elevation_low,
            depth_m,
        )

        g01 = self._predict_corner(
            angle_low,
            elevation_high,
            depth_m,
        )

        g11 = self._predict_corner(
            angle_high,
            elevation_high,
            depth_m,
        )

        corners = [
            g00,
            g10,
            g01,
            g11,
        ]

        if any(
            item is None
            for item in corners
        ):
            return {
                "valid":
                    False,

                "reason":
                    "geometry_corner_prediction_failed",
            }

        # ----------------------------------------------------
        # Angle interpolation at lower elevation.
        # ----------------------------------------------------

        q_width_e0 = _lerp(
            g00[
                "q_width"
            ],
            g10[
                "q_width"
            ],
            angle_weight,
        )

        q_height_e0 = _lerp(
            g00[
                "q_height"
            ],
            g10[
                "q_height"
            ],
            angle_weight,
        )

        # ----------------------------------------------------
        # Angle interpolation at upper elevation.
        # ----------------------------------------------------

        q_width_e1 = _lerp(
            g01[
                "q_width"
            ],
            g11[
                "q_width"
            ],
            angle_weight,
        )

        q_height_e1 = _lerp(
            g01[
                "q_height"
            ],
            g11[
                "q_height"
            ],
            angle_weight,
        )

        # ----------------------------------------------------
        # Elevation interpolation.
        # ----------------------------------------------------

        q_width = _lerp(
            q_width_e0,
            q_width_e1,
            elevation_weight,
        )

        q_height = _lerp(
            q_height_e0,
            q_height_e1,
            elevation_weight,
        )

        box_width_px = (
            target_fx
            *
            q_width
        )

        box_height_px = (
            target_fy
            *
            q_height
        )

        if (
            not math.isfinite(
                box_width_px
            )
            or
            not math.isfinite(
                box_height_px
            )
            or
            box_width_px <= 0.0
            or
            box_height_px <= 0.0
        ):
            return {
                "valid":
                    False,

                "reason":
                    "invalid_predicted_geometry",
            }

        depth_modes = sorted(
            {
                str(
                    item[
                        "depth_mode"
                    ]
                )
                for item in corners
            }
        )

        return {
            "valid":
                True,

            "geometry_version":
                self.geometry_version,

            "geometry_alpha_threshold":
                int(
                    self.geometry_alpha_threshold
                ),

            "box_width_px":
                float(
                    box_width_px
                ),

            "box_height_px":
                float(
                    box_height_px
                ),

            "q_width":
                float(
                    q_width
                ),

            "q_height":
                float(
                    q_height
                ),

            "target_fx_px":
                float(
                    target_fx
                ),

            "target_fy_px":
                float(
                    target_fy
                ),

            "query_viewpoint_angle_deg":
                float(
                    viewpoint_angle_deg
                )
                %
                360.0,

            "angle_low_deg":
                int(
                    angle_low
                ),

            "angle_high_deg":
                int(
                    angle_high
                ),

            "angle_weight":
                float(
                    angle_weight
                ),

            "query_elevation_deg":
                float(
                    elevation_deg
                ),

            "elevation_low_deg":
                float(
                    elevation_low
                ),

            "elevation_high_deg":
                float(
                    elevation_high
                ),

            "elevation_weight":
                float(
                    elevation_weight
                ),

            "query_depth_m":
                float(
                    depth_m
                ),

            "depth_modes":
                depth_modes,
        }


# ============================================================
# Public loader
# ============================================================

def load_sprite_native_geometry(
    geometry_csv,
):
    return SpriteNativeGeometryV1(
        geometry_csv
    )