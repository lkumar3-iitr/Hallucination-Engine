"""
he_geometry_calibration_v1.py

Runtime geometry-calibration surface for Hallucination Engine.

Architecture
------------
HEPlacement
    ↓
HEGeometryCalibrationV1
    ↓
sprite selection / frozen compositor

Calibration coordinates:
    distance_m
    viewpoint_deg

Correction:
    width'    = width    * width_scale
    height'   = height   * height_scale
    bottom_y' = bottom_y + bottom_offset_px
    cx'       = cx       + cx_offset_px

Interpolation
-------------
Distance:
    linear interpolation with boundary clamping.

Viewpoint:
    circular linear interpolation on [0, 360).

No extrapolation is performed outside the measured distance range.
"""

from pathlib import Path

import numpy as np


def normalize_angle_360(angle_deg):
    value = float(angle_deg) % 360.0

    if (
        abs(value) < 1e-12
        or
        abs(value - 360.0) < 1e-12
    ):
        return 0.0

    return value


class HEGeometryCalibrationV1:

    def __init__(self, npz_path):

        self.npz_path = str(
            npz_path
        )

        path = Path(
            self.npz_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Geometry calibration not found: "
                f"{path}"
            )

        data = np.load(
            path,
            allow_pickle=False,
        )

        required = [
            "distance_values",
            "angle_values",
            "width_scale",
            "height_scale",
            "bottom_offset_px",
            "cx_offset_px",
        ]

        for key in required:

            if key not in data:
                raise RuntimeError(
                    f"Calibration NPZ missing array: "
                    f"{key}"
                )

        self.distance_values = np.asarray(
            data["distance_values"],
            dtype=np.float64,
        )

        self.angle_values = np.asarray(
            data["angle_values"],
            dtype=np.float64,
        )

        self.width_scale = np.asarray(
            data["width_scale"],
            dtype=np.float64,
        )

        self.height_scale = np.asarray(
            data["height_scale"],
            dtype=np.float64,
        )

        self.bottom_offset_px = np.asarray(
            data["bottom_offset_px"],
            dtype=np.float64,
        )

        self.cx_offset_px = np.asarray(
            data["cx_offset_px"],
            dtype=np.float64,
        )

        self.schema_version = None

        if "schema_version" in data:

            self.schema_version = str(
                data[
                    "schema_version"
                ].item()
            )

        # ========================================================
        # Validation
        # ========================================================

        if self.distance_values.ndim != 1:
            raise RuntimeError(
                "distance_values must be 1-D"
            )

        if self.angle_values.ndim != 1:
            raise RuntimeError(
                "angle_values must be 1-D"
            )

        if len(
            self.distance_values
        ) < 2:

            raise RuntimeError(
                "Need at least two distance knots."
            )

        if len(
            self.angle_values
        ) < 2:

            raise RuntimeError(
                "Need at least two angle knots."
            )

        if not np.all(
            np.diff(
                self.distance_values
            )
            >
            0.0
        ):

            raise RuntimeError(
                "distance_values must be strictly increasing."
            )

        if not np.all(
            np.diff(
                self.angle_values
            )
            >
            0.0
        ):

            raise RuntimeError(
                "angle_values must be strictly increasing."
            )

        if (
            float(
                self.angle_values[0]
            )
            <
            0.0
            or
            float(
                self.angle_values[-1]
            )
            >=
            360.0
        ):

            raise RuntimeError(
                "angle_values must lie in [0, 360)."
            )

        expected_shape = (
            len(
                self.distance_values
            ),
            len(
                self.angle_values
            ),
        )

        surfaces = {
            "width_scale":
                self.width_scale,

            "height_scale":
                self.height_scale,

            "bottom_offset_px":
                self.bottom_offset_px,

            "cx_offset_px":
                self.cx_offset_px,
        }

        for name, surface in surfaces.items():

            if surface.shape != expected_shape:

                raise RuntimeError(
                    f"{name} shape "
                    f"{surface.shape} != "
                    f"{expected_shape}"
                )

            if not np.all(
                np.isfinite(
                    surface
                )
            ):

                raise RuntimeError(
                    f"{name} contains "
                    "non-finite values."
                )

        self.distance_min = float(
            self.distance_values[0]
        )

        self.distance_max = float(
            self.distance_values[-1]
        )

        print(
            "[HEGeometryCalibrationV1] Loaded:",
            self.npz_path,
        )

        print(
            "[HEGeometryCalibrationV1] distance:",
            self.distance_values.tolist(),
        )

        print(
            "[HEGeometryCalibrationV1] angles:",
            self.angle_values.tolist(),
        )

        print(
            "[HEGeometryCalibrationV1] shape:",
            expected_shape,
        )

    # ============================================================
    # Distance interpolation
    # ============================================================

    def _distance_interval(
        self,
        distance_m,
    ):

        query = float(
            distance_m
        )

        clamped = min(
            max(
                query,
                self.distance_min,
            ),
            self.distance_max,
        )

        if clamped <= self.distance_min:

            return (
                0,
                0,
                0.0,
                clamped,
            )

        if clamped >= self.distance_max:

            last = (
                len(
                    self.distance_values
                )
                -
                1
            )

            return (
                last,
                last,
                0.0,
                clamped,
            )

        i1 = int(
            np.searchsorted(
                self.distance_values,
                clamped,
                side="right",
            )
        )

        i0 = i1 - 1

        d0 = float(
            self.distance_values[i0]
        )

        d1 = float(
            self.distance_values[i1]
        )

        weight = (
            clamped - d0
        ) / (
            d1 - d0
        )

        return (
            i0,
            i1,
            float(
                weight
            ),
            float(
                clamped
            ),
        )

    # ============================================================
    # Circular viewpoint interpolation
    # ============================================================

    def _angle_interval(
        self,
        viewpoint_deg,
    ):

        angle = normalize_angle_360(
            viewpoint_deg
        )

        values = self.angle_values

        n = len(
            values
        )

        index = int(
            np.searchsorted(
                values,
                angle,
                side="right",
            )
        )

        # --------------------------------------------------------
        # Before first knot:
        # interpolate circularly from last-360 -> first.
        # --------------------------------------------------------

        if index == 0:

            i0 = n - 1
            i1 = 0

            a0 = float(
                values[i0]
            ) - 360.0

            a1 = float(
                values[i1]
            )

            weight = (
                angle - a0
            ) / (
                a1 - a0
            )

            return (
                i0,
                i1,
                float(
                    weight
                ),
                float(
                    angle
                ),
            )

        # --------------------------------------------------------
        # At/after last knot:
        # interpolate last -> first+360.
        # --------------------------------------------------------

        if index >= n:

            i0 = n - 1
            i1 = 0

            a0 = float(
                values[i0]
            )

            a1 = (
                float(
                    values[i1]
                )
                +
                360.0
            )

            weight = (
                angle - a0
            ) / (
                a1 - a0
            )

            return (
                i0,
                i1,
                float(
                    weight
                ),
                float(
                    angle
                ),
            )

        # --------------------------------------------------------
        # Ordinary interval.
        # --------------------------------------------------------

        i0 = index - 1
        i1 = index

        a0 = float(
            values[i0]
        )

        a1 = float(
            values[i1]
        )

        weight = (
            angle - a0
        ) / (
            a1 - a0
        )

        return (
            i0,
            i1,
            float(
                weight
            ),
            float(
                angle
            ),
        )

    # ============================================================
    # Bilinear interpolation
    # ============================================================

    def _interpolate_surface(
        self,
        surface,
        distance_m,
        viewpoint_deg,
    ):

        (
            di0,
            di1,
            wd,
            clamped_distance,
        ) = self._distance_interval(
            distance_m
        )

        (
            ai0,
            ai1,
            wa,
            normalized_angle,
        ) = self._angle_interval(
            viewpoint_deg
        )

        # Angle interpolation at lower distance.
        v_d0_a0 = float(
            surface[
                di0,
                ai0,
            ]
        )

        v_d0_a1 = float(
            surface[
                di0,
                ai1,
            ]
        )

        lower = (
            v_d0_a0
            *
            (
                1.0 - wa
            )
            +
            v_d0_a1
            *
            wa
        )

        # Same distance knot when clamped/exact.
        if di0 == di1:

            value = lower

        else:

            v_d1_a0 = float(
                surface[
                    di1,
                    ai0,
                ]
            )

            v_d1_a1 = float(
                surface[
                    di1,
                    ai1,
                ]
            )

            upper = (
                v_d1_a0
                *
                (
                    1.0 - wa
                )
                +
                v_d1_a1
                *
                wa
            )

            value = (
                lower
                *
                (
                    1.0 - wd
                )
                +
                upper
                *
                wd
            )

        debug = {
            "distance_query_m":
                float(
                    distance_m
                ),

            "distance_used_m":
                float(
                    clamped_distance
                ),

            "distance_i0":
                int(
                    di0
                ),

            "distance_i1":
                int(
                    di1
                ),

            "distance_weight":
                float(
                    wd
                ),

            "viewpoint_query_deg":
                float(
                    viewpoint_deg
                ),

            "viewpoint_used_deg":
                float(
                    normalized_angle
                ),

            "angle_i0":
                int(
                    ai0
                ),

            "angle_i1":
                int(
                    ai1
                ),

            "angle_weight":
                float(
                    wa
                ),
        }

        return (
            float(
                value
            ),
            debug,
        )

    # ============================================================
    # Public correction query
    # ============================================================

    def sample(
        self,
        distance_m,
        viewpoint_deg,
    ):

        width_scale, debug = (
            self._interpolate_surface(
                self.width_scale,
                distance_m,
                viewpoint_deg,
            )
        )

        height_scale, _ = (
            self._interpolate_surface(
                self.height_scale,
                distance_m,
                viewpoint_deg,
            )
        )

        bottom_offset, _ = (
            self._interpolate_surface(
                self.bottom_offset_px,
                distance_m,
                viewpoint_deg,
            )
        )

        cx_offset, _ = (
            self._interpolate_surface(
                self.cx_offset_px,
                distance_m,
                viewpoint_deg,
            )
        )

        return {
            "width_scale":
                float(
                    width_scale
                ),

            "height_scale":
                float(
                    height_scale
                ),

            "bottom_offset_px":
                float(
                    bottom_offset
                ),

            "cx_offset_px":
                float(
                    cx_offset
                ),

            "interpolation":
                debug,
        }

    # ============================================================
    # Apply correction to a placement box
    # ============================================================

    def apply_box(
        self,
        box,
        distance_m,
        viewpoint_deg,
    ):

        if box is None:
            return None

        if not box.get(
            "visible",
            False,
        ):
            return dict(
                box
            )

        correction = self.sample(
            distance_m=
                distance_m,
            viewpoint_deg=
                viewpoint_deg,
        )

        out = dict(
            box
        )

        original_cx = float(
            box["cx"]
        )

        original_bottom = float(
            box["bottom_y"]
        )

        original_width = float(
            box["box_width"]
        )

        original_height = float(
            box["box_height"]
        )

        corrected_cx = (
            original_cx
            +
            correction[
                "cx_offset_px"
            ]
        )

        corrected_bottom = (
            original_bottom
            +
            correction[
                "bottom_offset_px"
            ]
        )

        corrected_width = (
            original_width
            *
            correction[
                "width_scale"
            ]
        )

        corrected_height = (
            original_height
            *
            correction[
                "height_scale"
            ]
        )

        out["cx"] = float(
            corrected_cx
        )

        out["bottom_y"] = float(
            corrected_bottom
        )

        out["box_width"] = float(
            corrected_width
        )

        out["box_height"] = float(
            corrected_height
        )

        out["x1"] = float(
            corrected_cx
            -
            corrected_width / 2.0
        )

        out["x2"] = float(
            corrected_cx
            +
            corrected_width / 2.0
        )

        out["y2"] = float(
            corrected_bottom
        )

        out["y1"] = float(
            corrected_bottom
            -
            corrected_height
        )

        out["source_before_geometry_calibration"] = (
            box.get(
                "source",
                "unknown",
            )
        )

        out["source"] = (
            "he_geometry_calibration_v1"
        )

        out["geometry_calibration"] = {
            "distance_m":
                float(
                    distance_m
                ),

            "viewpoint_deg":
                float(
                    viewpoint_deg
                ),

            "width_scale":
                float(
                    correction[
                        "width_scale"
                    ]
                ),

            "height_scale":
                float(
                    correction[
                        "height_scale"
                    ]
                ),

            "bottom_offset_px":
                float(
                    correction[
                        "bottom_offset_px"
                    ]
                ),

            "cx_offset_px":
                float(
                    correction[
                        "cx_offset_px"
                    ]
                ),

            "interpolation":
                correction[
                    "interpolation"
                ],
        }

        return out