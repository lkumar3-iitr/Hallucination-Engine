import numpy as np
import pytest
import carla

from he_camera_renderer import (
    _circular_interpolation_bracket,
    continuous_view_matrix_samples,
    warp_view_matrix_sprite_camera_rotation,
)


def test_camera_rotation_reprojection_is_identity_for_matching_cameras():
    sprite = np.zeros((100, 100, 4), dtype=np.uint8)
    sprite[40:71, 30:71, :3] = (10, 20, 200)
    sprite[40:71, 30:71, 3] = 255

    result = warp_view_matrix_sprite_camera_rotation(
        sprite_rgba=sprite,
        actor_tf=carla.Transform(location=carla.Location(x=10.0)),
        camera_tf=carla.Transform(),
        physical_bbox={
            "local_center_x_m": 0.0,
            "local_center_y_m": 0.0,
            "local_center_z_m": 0.0,
        },
        sprite_info={
            "selected_angle": 180,
            "selected_elevation_deg": 0.0,
            "crop_x1_px": 0.0,
            "crop_y1_px": 0.0,
            "source_ground_anchor_x": 50.0,
            "source_ground_anchor_y": 55.0,
        },
        capture_metadata={
            "image_width_px": 100,
            "image_height_px": 100,
            "fov_deg": 90.0,
            "camera_cx_px": 50.0,
            "camera_cy_px": 50.0,
        },
        target_support_anchor={"u": 50.0, "v": 55.0},
        width=100,
        height=100,
        fov=90.0,
    )

    assert result is not None
    warped, unused_resize, metadata = result
    assert metadata["similarity_scale"] == pytest.approx(1.0)
    assert metadata["similarity_rotation_deg"] == pytest.approx(0.0)
    assert np.array_equal(warped, sprite)


def test_circular_interpolation_bracket_wraps_at_360_degrees():
    lower, upper, weight = _circular_interpolation_bracket(
        [0.0, 10.0, 350.0], 355.0
    )

    assert lower == 350.0
    assert upper == 0.0
    assert weight == pytest.approx(0.5)


def test_continuous_view_matrix_samples_are_trilinear_and_normalized():
    view_matrix = {
        "angles": [0.0, 10.0],
        "distances": [5.0, 10.0],
        "elevations": [0.0, 10.0],
        "index": {},
    }
    for angle in view_matrix["angles"]:
        for distance in view_matrix["distances"]:
            for elevation in view_matrix["elevations"]:
                view_matrix["index"][(int(angle), distance, elevation)] = {
                    "rgba_path": f"{angle}_{distance}_{elevation}.png",
                    "crop_x1_px": 0.0,
                    "crop_y1_px": 0.0,
                    "source_ground_anchor_x": 1.0,
                    "source_ground_anchor_y": 2.0,
                    "bbox_center_distance_m": distance,
                }

    samples = continuous_view_matrix_samples(
        {
            "relative_angle_deg": 5.0,
            "query_distance_m": 7.5,
            "query_elevation_deg": 5.0,
            "selected_angle": 0,
            "selected_distance_m": 5.0,
            "selected_elevation_deg": 0.0,
        },
        view_matrix,
    )

    assert len(samples) == 8
    assert sum(weight for weight, unused in samples) == pytest.approx(1.0)
    assert all(weight == pytest.approx(0.125) for weight, unused in samples)
