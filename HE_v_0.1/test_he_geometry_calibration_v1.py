"""
test_he_geometry_calibration_v1.py

Deterministic validation of HEGeometryCalibrationV1.

Tests:
1. Every measured knot is reproduced exactly.
2. 0 and 360 degree queries are identical.
3. Circular interpolation across 345 -> 0 is correct.
4. Distance interpolation is correct.
5. Distance is clamped outside 10..40 m.
6. apply_box() obeys the correction equations.
"""

import argparse

import numpy as np

from he_geometry_calibration_v1 import (
    HEGeometryCalibrationV1,
)


def assert_close(
    name,
    actual,
    expected,
    tol=1e-10,
):

    error = abs(
        float(actual)
        -
        float(expected)
    )

    if error > tol:

        raise AssertionError(
            f"{name}: "
            f"actual={actual}, "
            f"expected={expected}, "
            f"error={error}, "
            f"tol={tol}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--calibration",
        required=True,
    )

    args = parser.parse_args()

    calibration = (
        HEGeometryCalibrationV1(
            args.calibration
        )
    )

    data = np.load(
        args.calibration,
        allow_pickle=False,
    )

    distances = np.asarray(
        data["distance_values"],
        dtype=np.float64,
    )

    angles = np.asarray(
        data["angle_values"],
        dtype=np.float64,
    )

    surfaces = {
        "width_scale":
            np.asarray(
                data["width_scale"],
                dtype=np.float64,
            ),

        "height_scale":
            np.asarray(
                data["height_scale"],
                dtype=np.float64,
            ),

        "bottom_offset_px":
            np.asarray(
                data["bottom_offset_px"],
                dtype=np.float64,
            ),

        "cx_offset_px":
            np.asarray(
                data["cx_offset_px"],
                dtype=np.float64,
            ),
    }

    print()
    print("=" * 92)
    print(
        "HE GEOMETRY CALIBRATION V1 TEST"
    )
    print("=" * 92)

    # ========================================================
    # Test 1: exact reproduction of all 80 knots
    # ========================================================

    max_errors = {
        key:
            0.0
        for key in surfaces.keys()
    }

    knot_count = 0

    for di, distance in enumerate(
        distances
    ):

        for ai, angle in enumerate(
            angles
        ):

            result = calibration.sample(
                distance_m=
                    float(
                        distance
                    ),
                viewpoint_deg=
                    float(
                        angle
                    ),
            )

            for key, surface in surfaces.items():

                actual = float(
                    result[
                        key
                    ]
                )

                expected = float(
                    surface[
                        di,
                        ai,
                    ]
                )

                error = abs(
                    actual
                    -
                    expected
                )

                max_errors[
                    key
                ] = max(
                    max_errors[
                        key
                    ],
                    error,
                )

                assert_close(
                    f"knot d={distance} "
                    f"a={angle} {key}",
                    actual,
                    expected,
                )

            knot_count += 1

    print(
        "[PASS] exact knot reproduction:",
        knot_count,
        "knots",
    )

    for key, error in max_errors.items():

        print(
            f"       {key:20s} "
            f"max_error={error:.3e}"
        )

    # ========================================================
    # Test 2: 0 == 360 == -360
    # ========================================================

    for distance in distances:

        q0 = calibration.sample(
            distance_m=
                float(distance),
            viewpoint_deg=0.0,
        )

        q360 = calibration.sample(
            distance_m=
                float(distance),
            viewpoint_deg=360.0,
        )

        qneg = calibration.sample(
            distance_m=
                float(distance),
            viewpoint_deg=-360.0,
        )

        for key in surfaces.keys():

            assert_close(
                f"0/360 {key}",
                q0[key],
                q360[key],
            )

            assert_close(
                f"0/-360 {key}",
                q0[key],
                qneg[key],
            )

    print(
        "[PASS] circular identity: "
        "0 == 360 == -360"
    )

    # ========================================================
    # Test 3: circular midpoint 345 -> 360/0
    #
    # 352.5 degrees should be exactly halfway between
    # the 345-degree and 0-degree calibration knots.
    # ========================================================

    test_distance = 20.0

    q345 = calibration.sample(
        distance_m=
            test_distance,
        viewpoint_deg=
            345.0,
    )

    q0 = calibration.sample(
        distance_m=
            test_distance,
        viewpoint_deg=
            0.0,
    )

    qmid = calibration.sample(
        distance_m=
            test_distance,
        viewpoint_deg=
            352.5,
    )

    for key in surfaces.keys():

        expected = (
            float(
                q345[key]
            )
            +
            float(
                q0[key]
            )
        ) / 2.0

        assert_close(
            f"circular midpoint {key}",
            qmid[key],
            expected,
        )

    print(
        "[PASS] circular interpolation "
        "345 -> 0"
    )

    # ========================================================
    # Test 4: distance midpoint
    #
    # At an exact angle knot, d=12.5 should be halfway
    # between 10 and 15 m.
    # ========================================================

    test_angle = 90.0

    q10 = calibration.sample(
        distance_m=10.0,
        viewpoint_deg=
            test_angle,
    )

    q15 = calibration.sample(
        distance_m=15.0,
        viewpoint_deg=
            test_angle,
    )

    q125 = calibration.sample(
        distance_m=12.5,
        viewpoint_deg=
            test_angle,
    )

    for key in surfaces.keys():

        expected = (
            float(
                q10[key]
            )
            +
            float(
                q15[key]
            )
        ) / 2.0

        assert_close(
            f"distance midpoint {key}",
            q125[key],
            expected,
        )

    print(
        "[PASS] linear distance interpolation"
    )

    # ========================================================
    # Test 5: distance clamping
    # ========================================================

    for angle in [
        0.0,
        45.0,
        150.0,
        270.0,
        345.0,
    ]:

        below = calibration.sample(
            distance_m=1.0,
            viewpoint_deg=angle,
        )

        minimum = calibration.sample(
            distance_m=10.0,
            viewpoint_deg=angle,
        )

        above = calibration.sample(
            distance_m=100.0,
            viewpoint_deg=angle,
        )

        maximum = calibration.sample(
            distance_m=40.0,
            viewpoint_deg=angle,
        )

        for key in surfaces.keys():

            assert_close(
                f"lower clamp {angle} {key}",
                below[key],
                minimum[key],
            )

            assert_close(
                f"upper clamp {angle} {key}",
                above[key],
                maximum[key],
            )

    print(
        "[PASS] distance clamping"
    )

    # ========================================================
    # Test 6: apply_box equations
    # ========================================================

    input_box = {
        "visible":
            True,

        "cx":
            640.0,

        "bottom_y":
            400.0,

        "box_width":
            100.0,

        "box_height":
            50.0,

        "x1":
            590.0,

        "x2":
            690.0,

        "y1":
            350.0,

        "y2":
            400.0,

        "source":
            "unit_test",
    }

    correction = calibration.sample(
        distance_m=20.0,
        viewpoint_deg=90.0,
    )

    output_box = calibration.apply_box(
        box=input_box,
        distance_m=20.0,
        viewpoint_deg=90.0,
    )

    expected_width = (
        100.0
        *
        correction[
            "width_scale"
        ]
    )

    expected_height = (
        50.0
        *
        correction[
            "height_scale"
        ]
    )

    expected_bottom = (
        400.0
        +
        correction[
            "bottom_offset_px"
        ]
    )

    expected_cx = (
        640.0
        +
        correction[
            "cx_offset_px"
        ]
    )

    assert_close(
        "apply width",
        output_box[
            "box_width"
        ],
        expected_width,
    )

    assert_close(
        "apply height",
        output_box[
            "box_height"
        ],
        expected_height,
    )

    assert_close(
        "apply bottom",
        output_box[
            "bottom_y"
        ],
        expected_bottom,
    )

    assert_close(
        "apply cx",
        output_box[
            "cx"
        ],
        expected_cx,
    )

    assert_close(
        "apply x1",
        output_box[
            "x1"
        ],
        expected_cx
        -
        expected_width / 2.0,
    )

    assert_close(
        "apply x2",
        output_box[
            "x2"
        ],
        expected_cx
        +
        expected_width / 2.0,
    )

    assert_close(
        "apply y1",
        output_box[
            "y1"
        ],
        expected_bottom
        -
        expected_height,
    )

    assert_close(
        "apply y2",
        output_box[
            "y2"
        ],
        expected_bottom,
    )

    print(
        "[PASS] box correction equations"
    )

    print()
    print(
        "ALL HE GEOMETRY CALIBRATION V1 "
        "TESTS PASSED"
    )

    print("=" * 92)


if __name__ == "__main__":
    main()