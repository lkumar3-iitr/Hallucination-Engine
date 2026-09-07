from scripts.analyze_partial_observability_v1 import intersection_fraction


def test_missing_projected_box_has_zero_overlap():
    box = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}

    assert intersection_fraction(None, box) == 0.0
    assert intersection_fraction(box, None) == 0.0


def test_projected_box_overlap_is_normalized_by_target_area():
    target = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}
    occluder = {"x1": 5.0, "y1": 0.0, "x2": 15.0, "y2": 10.0}

    assert intersection_fraction(target, occluder) == 0.5
