import math

import pytest

from he_multi_actor_compositor_v1 import HEMultiActorCompositorV1
from he_multi_actor_compositor_v2 import HEMultiActorCompositorV2


def test_circular_yaw_bracket_wraps_without_a_discontinuity():
    lower, upper, weight = HEMultiActorCompositorV2._circular_yaw_bracket(
        list(range(360)), 359.75
    )

    assert lower == 359.0
    assert upper == 0.0
    assert weight == pytest.approx(0.75)


def test_continuous_rows_normalize_available_neighbors():
    compositor = HEMultiActorCompositorV2()
    bank = {
        "forward_values": [1.0, 2.0],
        "right_values": [3.0, 4.0],
        "yaw_values": list(range(360)),
        "index": {},
    }
    for forward in bank["forward_values"]:
        for right in bank["right_values"]:
            for yaw in bank["yaw_values"]:
                row = {
                    "close_forward_m": str(forward),
                    "close_right_m": str(right),
                    "close_relative_yaw_deg": str(yaw),
                }
                bank["index"][(forward, right, yaw)] = row

    rows = compositor._continuous_rows({
        "bank": bank,
        "query": {
            "forward_m": 1.5,
            "right_m": 3.5,
            "relative_yaw_deg": 0.5,
        },
    })

    assert len(rows) == 8
    assert sum(weight for weight, unused in rows) == pytest.approx(1.0)
    assert all(weight > 0.0 for weight, unused in rows)


def test_v2_close_handoff_is_continuous_at_bank_edges():
    compositor = HEMultiActorCompositorV2()
    selection = {
        "query": {"forward_m": 7.0, "right_m": 3.5},
        "bank": {
            "forward_values": [0.25, 7.0],
            "right_values": [3.0, 3.5, 4.0],
        },
        "eligibility": {
            "forward_min_m": 0.25,
            "forward_max_m": 7.0,
            "right_min_m": 2.75,
            "right_max_m": 5.25,
        },
    }

    assert compositor._close_handoff_weight(selection) == pytest.approx(0.0)

    selection["query"] = {"forward_m": 5.0, "right_m": 3.5}
    assert compositor._close_handoff_weight(selection) == pytest.approx(1.0)


def test_continuous_rows_preserve_viewpoint_when_lateral_position_is_extrapolated():
    compositor = HEMultiActorCompositorV2()
    bank = {
        "forward_values": [3.0, 3.25, 6.5, 6.75, 7.0],
        "right_values": [-4.0, -3.0],
        "yaw_values": list(range(360)),
        "index": {},
    }
    for forward in bank["forward_values"]:
        for right in bank["right_values"]:
            for yaw in bank["yaw_values"]:
                bank["index"][(forward, right, float(yaw))] = {
                    "close_forward_m": str(forward),
                    "close_right_m": str(right),
                    "close_relative_yaw_deg": str(yaw),
                }

    query = {
        "forward_m": 3.196,
        "right_m": -1.465,
        "relative_yaw_deg": 60.0,
    }
    rows = compositor._continuous_rows({"bank": bank, "query": query})
    requested_view = (
        math.degrees(math.atan2(
            query["right_m"], query["forward_m"]
        ))
        - query["relative_yaw_deg"]
        + 180.0
    ) % 360.0

    for unused_weight, row in rows:
        selected_view = (
            math.degrees(math.atan2(
                float(row["close_right_m"]), float(row["close_forward_m"])
            ))
            - float(row["close_relative_yaw_deg"])
            + 180.0
        ) % 360.0
        error = abs((selected_view - requested_view + 180.0) % 360.0 - 180.0)
        assert error <= 1.0

    dominant_row = max(rows, key=lambda item: item[0])[1]
    assert float(dominant_row["close_forward_m"]) == pytest.approx(6.5)
    assert float(dominant_row["close_relative_yaw_deg"]) == pytest.approx(60.0)


def test_close_sampling_query_uses_forward_limit_before_yaw_compensation():
    bank = {
        "forward_values": [value / 4.0 for value in range(1, 29)],
        "right_values": [3.0, 3.5, 4.0],
    }
    query = {
        "forward_m": 4.9957,
        "right_m": 1.6528,
    }

    sample = HEMultiActorCompositorV2._close_sampling_query(bank, query)

    assert sample["right_m"] == pytest.approx(3.0)
    assert sample["forward_m"] == pytest.approx(7.0)
    assert sample["ray_preserved"] is False
    assert abs(
        sample["sample_bearing_deg"] - sample["runtime_bearing_deg"]
    ) < 5.0


def test_v2_edge_box_matches_far_bank_before_converging_to_close_bank():
    close_box = {
        "x1": 20.0,
        "x2": 80.0,
        "y1": 30.0,
        "y2": 90.0,
        "scale": 2.0,
    }
    fallback_box = {"x1": 25, "x2": 75, "y1": 35, "y2": 85}

    at_edge = HEMultiActorCompositorV2._blend_box_with_fallback(
        close_box, fallback_box, 0.0
    )
    in_interior = HEMultiActorCompositorV2._blend_box_with_fallback(
        close_box, fallback_box, 1.0
    )

    assert {key: at_edge[key] for key in fallback_box} == fallback_box
    assert {key: in_interior[key] for key in fallback_box} == {
        key: close_box[key] for key in fallback_box
    }


def test_v2_geometry_policy_does_not_change_v1_defaults():
    assert HEMultiActorCompositorV1.fallback_far_geometry_mode == "proxy"
    assert HEMultiActorCompositorV1.centered_close_geometry_mode == "proxy"
    assert HEMultiActorCompositorV1.center_depth_alpha_bbox_anchor is False
    assert HEMultiActorCompositorV2.fallback_far_geometry_mode == "sprite_alpha_metric"
    assert HEMultiActorCompositorV1.close_inner_min_bbox_clearance_m is None
    assert HEMultiActorCompositorV2.close_inner_min_bbox_clearance_m == pytest.approx(0.15)
