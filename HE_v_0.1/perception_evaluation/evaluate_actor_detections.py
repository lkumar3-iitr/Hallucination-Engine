#!/usr/bin/env python3
"""
evaluate_actor_detections.py

Actor-aware CARLA-vs-HE perception evaluator.

Inputs:
  1. CARLA ground truth JSONL
  2. HE metadata.json
  3. YOLO detections on CARLA video
  4. YOLO detections on HE video

The evaluator:

  - identifies scenario actors by actor ID
  - uses CARLA GT boxes for CARLA detections
  - uses HE placement boxes for HE detections
  - performs one-to-one YOLO detection matching per frame
  - prevents one YOLO detection from matching multiple overlapping actors
  - computes actor-level CARLA-vs-HE perception agreement

Example:

python perception_evaluation\\evaluate_actor_detections.py ^
  --carla-gt recordings\\he_pairs\\v2_multi_actor_001\\real_ground_truth_v2.jsonl ^
  --he-metadata he_outputs\\v2_multi_actor_001_he\\metadata.json ^
  --carla-detections perception_evaluation\\outputs\\v2_multi_actor_001_carla_yolo11s.jsonl ^
  --he-detections perception_evaluation\\outputs\\v2_multi_actor_001_he_yolo11s.jsonl ^
  --match-iou 0.10 ^
  --output-json perception_evaluation\\outputs\\v2_multi_actor_001_yolo11s_actor_eval.json ^
  --output-csv perception_evaluation\\outputs\\v2_multi_actor_001_yolo11s_actor_eval.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--carla-gt",
        required=True,
        help="CARLA real_ground_truth_v2.jsonl",
    )

    parser.add_argument(
        "--he-metadata",
        required=True,
        help="HE metadata.json",
    )

    parser.add_argument(
        "--carla-detections",
        required=True,
        help="YOLO JSONL generated from CARLA video",
    )

    parser.add_argument(
        "--he-detections",
        required=True,
        help="YOLO JSONL generated from HE video",
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.10,
        help="Minimum IoU for assigning detection to actor.",
    )

    parser.add_argument(
        "--vehicle-classes",
        default="car,truck,bus",
        help=(
            "Comma-separated detector classes allowed to match "
            "vehicle actors."
        ),
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        required=True,
    )

    return parser.parse_args()


# ============================================================
# File utilities
# ============================================================

def load_jsonl(path):
    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def ensure_parent(path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return path


# ============================================================
# Bounding boxes
# ============================================================

def normalize_box(box):
    """
    Convert a box-like dictionary into common xyxy format.
    """

    if box is None:
        return None

    required = [
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    if not all(
        key in box
        for key in required
    ):
        return None

    return {
        "x1": float(box["x1"]),
        "y1": float(box["y1"]),
        "x2": float(box["x2"]),
        "y2": float(box["y2"]),
    }


def box_area(box):
    if box is None:
        return 0.0

    w = max(
        0.0,
        box["x2"] - box["x1"]
    )

    h = max(
        0.0,
        box["y2"] - box["y1"]
    )

    return w * h


def box_iou(a, b):
    if a is None or b is None:
        return 0.0

    ix1 = max(
        a["x1"],
        b["x1"]
    )

    iy1 = max(
        a["y1"],
        b["y1"]
    )

    ix2 = min(
        a["x2"],
        b["x2"]
    )

    iy2 = min(
        a["y2"],
        b["y2"]
    )

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = iw * ih

    union = (
        box_area(a)
        + box_area(b)
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union

def intersection_box(a, b):
    if a is None or b is None:
        return None

    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def rectangle_union_area(rectangles):
    """
    Exact union area for a small set of axis-aligned rectangles.
    """

    rectangles = [
        r for r in rectangles
        if r is not None and box_area(r) > 0.0
    ]

    if not rectangles:
        return 0.0

    xs = sorted(
        set(
            [r["x1"] for r in rectangles]
            + [r["x2"] for r in rectangles]
        )
    )

    total_area = 0.0

    for i in range(len(xs) - 1):

        left = xs[i]
        right = xs[i + 1]

        if right <= left:
            continue

        intervals = []

        for r in rectangles:
            if r["x1"] < right and r["x2"] > left:
                intervals.append(
                    (r["y1"], r["y2"])
                )

        if not intervals:
            continue

        intervals.sort()

        covered_y = 0.0

        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:

            if start <= current_end:
                current_end = max(
                    current_end,
                    end
                )
            else:
                covered_y += (
                    current_end
                    - current_start
                )

                current_start = start
                current_end = end

        covered_y += (
            current_end
            - current_start
        )

        total_area += (
            right - left
        ) * covered_y

    return total_area


def compute_occlusion_fraction(
    actor_id,
    actors,
):
    """
    Estimate what fraction of an actor's GT rectangle is covered
    by nearer scenario actors.

    depth_m smaller => nearer camera.
    """

    target = actors.get(actor_id)

    if target is None:
        return None

    if not target.get("visible", False):
        return None

    target_box = target.get("box")
    target_depth = target.get("depth_m")

    if target_box is None or target_depth is None:
        return None

    target_area = box_area(target_box)

    if target_area <= 0.0:
        return None

    occluding_regions = []

    for other_id, other in actors.items():

        if other_id == actor_id:
            continue

        if not other.get("visible", False):
            continue

        other_box = other.get("box")
        other_depth = other.get("depth_m")

        if other_box is None or other_depth is None:
            continue

        # Smaller depth means actor is closer to camera.
        if other_depth >= target_depth:
            continue

        overlap = intersection_box(
            target_box,
            other_box
        )

        if overlap is not None:
            occluding_regions.append(
                overlap
            )

    covered_area = rectangle_union_area(
        occluding_regions
    )

    fraction = (
        covered_area
        / target_area
    )

    return min(
        max(
            fraction,
            0.0
        ),
        1.0
    )


def occlusion_category(value):
    if value is None:
        return None

    if value < 0.25:
        return "low"

    if value < 0.50:
        return "partial"

    return "heavy"

# ============================================================
# CARLA GT conversion
# ============================================================

def build_carla_frames(records):
    """
    Returns:

    {
      frame_idx: {
        actor_id: {
          actor_id,
          actor_type,
          role,
          visible,
          box,
          depth_m
        }
      }
    }
    """

    frames = {}

    for record in records:

        frame_idx = int(
            record["recorded_frame_idx"]
        )

        actor_map = {}

        for actor in record.get(
            "actors",
            []
        ):

            actor_id = actor["actor_id"]

            bbox = actor.get(
                "bbox"
            )

            visible = bool(
                bbox
                and bbox.get(
                    "visible",
                    False
                )
            )

            actor_map[actor_id] = {
                "actor_id": actor_id,

                "actor_type": actor.get(
                    "actor_type"
                ),

                "role": actor.get(
                    "role"
                ),

                "visible": visible,

                "box": (
                    normalize_box(bbox)
                    if visible
                    else None
                ),

                "depth_m": (
                    float(
                        bbox.get(
                            "depth_m"
                        )
                    )
                    if bbox
                    and bbox.get(
                        "depth_m"
                    ) is not None
                    else None
                ),
            }

        frames[frame_idx] = actor_map

    return frames


# ============================================================
# HE metadata conversion
# ============================================================

def build_he_frames(metadata):
    frames = {}

    for frame in metadata.get(
        "frames",
        []
    ):

        frame_idx = int(
            frame["frame_idx"]
        )

        actor_map = {}

        for actor in frame.get(
            "adversaries",
            []
        ):

            actor_id = actor["id"]

            box = actor.get(
                "box"
            )

            paste = actor.get(
                "paste"
            )

            visible = bool(
                box
                and box.get(
                    "visible",
                    False
                )
            )

            state = actor.get(
                "state",
                {}
            )

            actor_map[actor_id] = {
                "actor_id": actor_id,

                "actor_type": state.get(
                    "type",
                    "vehicle"
                ),

                "visible": visible,

                "rendered": bool(
                    actor.get(
                        "rendered",
                        False
                    )
                ),

                # YOLO sees the actually pasted sprite, not the
                # placement-model reference rectangle.
                #
                # Therefore use the paste rectangle for detector
                # matching whenever it is available.
                "box": (
                    normalize_box(paste)
                    if visible and paste is not None
                    else (
                        normalize_box(box)
                        if visible
                        else None
                    )
                ),

                "depth_m": (
                    float(
                        state["z_m"]
                    )
                    if state.get(
                        "z_m"
                    ) is not None
                    else None
                ),
            }

        frames[frame_idx] = actor_map

    return frames


# ============================================================
# YOLO conversion
# ============================================================

def build_detection_frames(records):
    frames = {}

    for record in records:

        frame_idx = int(
            record["frame_idx"]
        )

        frames[frame_idx] = (
            record.get(
                "detections",
                []
            )
        )

    return frames


# ============================================================
# One-to-one matching
# ============================================================

def match_actors_to_detections(
    actors,
    detections,
    minimum_iou,
    vehicle_classes,
):
    """
    Greedy global IoU matching.

    Important:

    All valid actor/detection pairs are created first.
    Pairs are sorted by IoU descending.

    Once a detection is assigned to one actor, it cannot be
    assigned to another actor.

    This is critical for overlapping actors.
    """

    actor_items = []

    for actor_id, actor in actors.items():

        if not actor.get(
            "visible",
            False
        ):
            continue

        if actor.get(
            "box"
        ) is None:
            continue

        actor_items.append(
            (
                actor_id,
                actor
            )
        )

    candidate_detections = []

    for det_index, detection in enumerate(
        detections
    ):

        class_name = str(
            detection.get(
                "class_name",
                ""
            )
        ).lower()

        if class_name not in vehicle_classes:
            continue

        candidate_detections.append(
            (
                det_index,
                detection
            )
        )

    candidate_pairs = []

    for actor_id, actor in actor_items:

        gt_box = actor["box"]

        for det_index, detection in candidate_detections:

            det_box = normalize_box(
                detection
            )

            iou = box_iou(
                gt_box,
                det_box
            )

            if iou >= minimum_iou:

                candidate_pairs.append(
                    {
                        "actor_id": actor_id,
                        "det_index": det_index,
                        "iou": float(iou),
                        "detection": detection,
                    }
                )

    # Highest IoU gets first claim.
    candidate_pairs.sort(
        key=lambda item: item["iou"],
        reverse=True,
    )

    used_actors = set()
    used_detections = set()

    matches = {}

    for pair in candidate_pairs:

        actor_id = pair["actor_id"]
        det_index = pair["det_index"]

        if actor_id in used_actors:
            continue

        if det_index in used_detections:
            continue

        matches[actor_id] = {
            "detection": pair["detection"],
            "iou": pair["iou"],
        }

        used_actors.add(
            actor_id
        )

        used_detections.add(
            det_index
        )

    return matches

def find_supported_detection(
    actor,
    detections,
    minimum_iou,
    vehicle_classes,
):
    """
    Relaxed detector-support test.

    Unlike one-to-one matching, this does NOT reserve a detection
    for only one actor.

    It asks:

        Is there any valid vehicle detection that spatially
        overlaps this actor sufficiently?

    This is useful in overlapping / occluded multi-actor scenes,
    where a single YOLO vehicle box may support more than one
    scenario actor.
    """

    if actor is None:
        return None

    if not actor.get(
        "visible",
        False
    ):
        return None

    actor_box = actor.get(
        "box"
    )

    if actor_box is None:
        return None

    best = None
    best_iou = 0.0

    for detection in detections:

        class_name = str(
            detection.get(
                "class_name",
                ""
            )
        ).lower()

        if class_name not in vehicle_classes:
            continue

        det_box = normalize_box(
            detection
        )

        value = box_iou(
            actor_box,
            det_box
        )

        if value > best_iou:

            best_iou = value

            best = {
                "detection": detection,
                "iou": float(value),
            }

    if (
        best is None
        or best_iou < minimum_iou
    ):
        return None

    return best

# ============================================================
# Metrics
# ============================================================

def mean_or_none(values):
    values = [
        float(v)
        for v in values
        if v is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def longest_false_streak(flags):
    longest = 0
    current = 0

    for flag in flags:

        if flag:
            current = 0
        else:
            current += 1

            longest = max(
                longest,
                current
            )

    return longest


def first_true_frame(rows, key):
    for row in rows:

        if row.get(
            key,
            False
        ):
            return int(
                row["frame_idx"]
            )

    return None


# ============================================================
# Main evaluation
# ============================================================

def main():
    args = parse_args()

    vehicle_classes = {
        x.strip().lower()
        for x in args.vehicle_classes.split(",")
        if x.strip()
    }

    print("=" * 76)
    print("CARLA vs HE Actor-Aware Perception Evaluation")
    print("=" * 76)

    print(
        "match IoU:",
        args.match_iou
    )

    print(
        "vehicle classes:",
        sorted(vehicle_classes)
    )

    # ------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------

    carla_records = load_jsonl(
        args.carla_gt
    )

    carla_detection_records = load_jsonl(
        args.carla_detections
    )

    he_detection_records = load_jsonl(
        args.he_detections
    )

    with open(
        args.he_metadata,
        "r",
        encoding="utf-8"
    ) as f:
        he_metadata = json.load(f)

    # ------------------------------------------------------------
    # Convert schemas
    # ------------------------------------------------------------

    carla_frames = build_carla_frames(
        carla_records
    )

    he_frames = build_he_frames(
        he_metadata
    )

    carla_detection_frames = (
        build_detection_frames(
            carla_detection_records
        )
    )

    he_detection_frames = (
        build_detection_frames(
            he_detection_records
        )
    )

    common_frames = sorted(
        set(carla_frames)
        & set(he_frames)
        & set(carla_detection_frames)
        & set(he_detection_frames)
    )

    print(
        "CARLA GT frames:",
        len(carla_frames)
    )

    print(
        "HE frames:",
        len(he_frames)
    )

    print(
        "CARLA detector frames:",
        len(carla_detection_frames)
    )

    print(
        "HE detector frames:",
        len(he_detection_frames)
    )

    print(
        "common frames:",
        len(common_frames)
    )

    # ------------------------------------------------------------
    # Actor IDs
    # ------------------------------------------------------------

    actor_ids = set()

    for frame_idx in common_frames:

        actor_ids.update(
            carla_frames[
                frame_idx
            ].keys()
        )

        actor_ids.update(
            he_frames[
                frame_idx
            ].keys()
        )

    actor_ids = sorted(
        actor_ids
    )

    print(
        "actors:",
        ", ".join(actor_ids)
    )

    # ------------------------------------------------------------
    # Frame-by-frame matching
    # ------------------------------------------------------------

    rows = []

    for frame_idx in common_frames:

        carla_actor_map = (
            carla_frames[
                frame_idx
            ]
        )

        he_actor_map = (
            he_frames[
                frame_idx
            ]
        )

        carla_matches = match_actors_to_detections(
            actors=carla_actor_map,
            detections=carla_detection_frames[
                frame_idx
            ],
            minimum_iou=args.match_iou,
            vehicle_classes=vehicle_classes,
        )

        he_matches = match_actors_to_detections(
            actors=he_actor_map,
            detections=he_detection_frames[
                frame_idx
            ],
            minimum_iou=args.match_iou,
            vehicle_classes=vehicle_classes,
        )

        for actor_id in actor_ids:

            carla_actor = carla_actor_map.get(
                actor_id
            )

            he_actor = he_actor_map.get(
                actor_id
            )

            carla_present = (
                carla_actor is not None
            )

            he_present = (
                he_actor is not None
            )

            carla_visible = bool(
                carla_actor
                and carla_actor.get(
                    "visible",
                    False
                )
            )

            carla_occlusion = compute_occlusion_fraction(
                actor_id,
                carla_actor_map,
            )

            he_occlusion = compute_occlusion_fraction(
                actor_id,
                he_actor_map,
            )

            he_visible = bool(
                he_actor
                and he_actor.get(
                    "visible",
                    False
                )
            )

            carla_match = (
                carla_matches.get(
                    actor_id
                )
            )

            carla_support = find_supported_detection(
                actor=carla_actor,
                detections=carla_detection_frames[
                    frame_idx
                ],
                minimum_iou=args.match_iou,
                vehicle_classes=vehicle_classes,
            )

            he_support = find_supported_detection(
                actor=he_actor,
                detections=he_detection_frames[
                    frame_idx
                ],
                minimum_iou=args.match_iou,
                vehicle_classes=vehicle_classes,
            )

            he_match = (
                he_matches.get(
                    actor_id
                )
            )

            carla_detected = (
                carla_match is not None
            )

            he_detected = (
                he_match is not None
            )

            carla_supported_detected = (
                carla_support is not None
            )

            he_supported_detected = (
                he_support is not None
            )

            supported_detection_agreement = (
                carla_supported_detected
                == he_supported_detected
            )

            supported_both_detected = (
                carla_supported_detected
                and he_supported_detected
            )

            carla_detection = (
                carla_match[
                    "detection"
                ]
                if carla_match
                else None
            )

            he_detection = (
                he_match[
                    "detection"
                ]
                if he_match
                else None
            )

            carla_supported_detection = (
                carla_support[
                    "detection"
                ]
                if carla_support
                else None
            )

            he_supported_detection = (
                he_support[
                    "detection"
                ]
                if he_support
                else None
            )

            carla_supported_confidence = (
                float(
                    carla_supported_detection[
                        "confidence"
                    ]
                )
                if carla_supported_detection
                else None
            )

            he_supported_confidence = (
                float(
                    he_supported_detection[
                        "confidence"
                    ]
                )
                if he_supported_detection
                else None
            )

            carla_supported_class = (
                carla_supported_detection.get(
                    "class_name"
                )
                if carla_supported_detection
                else None
            )

            he_supported_class = (
                he_supported_detection.get(
                    "class_name"
                )
                if he_supported_detection
                else None
            )

            supported_class_agreement = (
                supported_both_detected
                and carla_supported_class
                == he_supported_class
            )

            supported_confidence_delta = (
                he_supported_confidence
                - carla_supported_confidence
                if supported_both_detected
                else None
            )

            carla_confidence = (
                float(
                    carla_detection[
                        "confidence"
                    ]
                )
                if carla_detection
                else None
            )

            he_confidence = (
                float(
                    he_detection[
                        "confidence"
                    ]
                )
                if he_detection
                else None
            )

            carla_class = (
                carla_detection.get(
                    "class_name"
                )
                if carla_detection
                else None
            )

            he_class = (
                he_detection.get(
                    "class_name"
                )
                if he_detection
                else None
            )

            both_detected = (
                carla_detected
                and he_detected
            )

            detection_agreement = (
                carla_detected
                == he_detected
            )

            class_agreement = (
                both_detected
                and carla_class
                == he_class
            )

            confidence_delta = (
                he_confidence
                - carla_confidence
                if both_detected
                else None
            )

            row = {
                "frame_idx": int(
                    frame_idx
                ),
                "carla_occlusion_fraction": carla_occlusion,

                "he_occlusion_fraction": he_occlusion,

                "carla_occlusion_category": occlusion_category(
                    carla_occlusion
                ),

                "he_occlusion_category": occlusion_category(
                    he_occlusion
                ),
                "actor_id": actor_id,

                "carla_present": carla_present,
                "he_present": he_present,

                "carla_visible": carla_visible,
                "he_visible": he_visible,

                "carla_detected": carla_detected,
                "he_detected": he_detected,

                "detection_agreement": detection_agreement,

                "both_detected": both_detected,

                # Relaxed detector-support metrics.
                #
                # These deliberately allow the same detector box to support
                # multiple overlapping scenario actors.
                "carla_supported_detected": carla_supported_detected,

                "he_supported_detected": he_supported_detected,

                "supported_detection_agreement": (
                    supported_detection_agreement
                ),

                "supported_both_detected": (
                    supported_both_detected
                ),

                "carla_supported_class": (
                    carla_supported_class
                ),

                "he_supported_class": (
                    he_supported_class
                ),

                "supported_class_agreement": (
                    supported_class_agreement
                ),

                "carla_supported_confidence": (
                    carla_supported_confidence
                ),

                "he_supported_confidence": (
                    he_supported_confidence
                ),

                "supported_confidence_delta_he_minus_carla": (
                    supported_confidence_delta
                ),

                "carla_supported_gt_iou": (
                    float(
                        carla_support["iou"]
                    )
                    if carla_support
                    else None
                ),

                "he_supported_gt_iou": (
                    float(
                        he_support["iou"]
                    )
                    if he_support
                    else None
                ),

                "carla_class": carla_class,
                "he_class": he_class,

                "class_agreement": class_agreement,

                "carla_confidence": carla_confidence,
                "he_confidence": he_confidence,

                "confidence_delta_he_minus_carla": (
                    confidence_delta
                ),

                "carla_detector_gt_iou": (
                    float(
                        carla_match[
                            "iou"
                        ]
                    )
                    if carla_match
                    else None
                ),

                "he_detector_gt_iou": (
                    float(
                        he_match[
                            "iou"
                        ]
                    )
                    if he_match
                    else None
                ),

                "carla_depth_m": (
                    carla_actor.get(
                        "depth_m"
                    )
                    if carla_actor
                    else None
                ),

                "he_depth_m": (
                    he_actor.get(
                        "depth_m"
                    )
                    if he_actor
                    else None
                ),
            }

            rows.append(
                row
            )

    # ------------------------------------------------------------
    # Per-actor summaries
    # ------------------------------------------------------------

    actor_summaries = {}

    for actor_id in actor_ids:

        actor_rows = [
            row
            for row in rows
            if row[
                "actor_id"
            ] == actor_id
        ]

        # Compare only frames where actor is present and visible
        # in BOTH conditions.
        comparable = [
            row
            for row in actor_rows
            if row[
                "carla_present"
            ]
            and row[
                "he_present"
            ]
            and row[
                "carla_visible"
            ]
            and row[
                "he_visible"
            ]
        ]

        low_occlusion = [
            row
            for row in comparable
            if row["carla_occlusion_fraction"] is not None
            and row["he_occlusion_fraction"] is not None
            and row["carla_occlusion_fraction"] < 0.25
            and row["he_occlusion_fraction"] < 0.25
        ]

        low_occ_agreement = [
            row
            for row in low_occlusion
            if row["detection_agreement"]
        ]

        low_occ_both_detected = [
            row
            for row in low_occlusion
            if row["both_detected"]
        ]

        carla_detected_rows = [
            row
            for row in comparable
            if row[
                "carla_detected"
            ]
        ]

        he_detected_rows = [
            row
            for row in comparable
            if row[
                "he_detected"
            ]
        ]

        both_detected_rows = [
            row
            for row in comparable
            if row[
                "both_detected"
            ]
        ]

        supported_carla_rows = [
            row
            for row in comparable
            if row[
                "carla_supported_detected"
            ]
        ]

        supported_he_rows = [
            row
            for row in comparable
            if row[
                "he_supported_detected"
            ]
        ]

        supported_both_rows = [
            row
            for row in comparable
            if row[
                "supported_both_detected"
            ]
        ]

        supported_agreement_rows = [
            row
            for row in comparable
            if row[
                "supported_detection_agreement"
            ]
        ]

        supported_class_agreement_rows = [
            row
            for row in supported_both_rows
            if row[
                "supported_class_agreement"
            ]
        ]

        agreement_rows = [
            row
            for row in comparable
            if row[
                "detection_agreement"
            ]
        ]

        class_agreement_rows = [
            row
            for row in both_detected_rows
            if row[
                "class_agreement"
            ]
        ]

        carla_flags = [
            row[
                "carla_detected"
            ]
            for row in comparable
        ]

        he_flags = [
            row[
                "he_detected"
            ]
            for row in comparable
        ]

        summary = {
            "actor_id": actor_id,

            "comparable_frames": len(
                comparable
            ),

            "carla_detected_frames": len(
                carla_detected_rows
            ),

            "he_detected_frames": len(
                he_detected_rows
            ),

            "both_detected_frames": len(
                both_detected_rows
            ),

            "carla_detection_rate": (
                len(carla_detected_rows)
                / len(comparable)
                if comparable
                else None
            ),

            "he_detection_rate": (
                len(he_detected_rows)
                / len(comparable)
                if comparable
                else None
            ),

            "detection_agreement_rate": (
                len(agreement_rows)
                / len(comparable)
                if comparable
                else None
            ),

            "both_detected_rate": (
                len(both_detected_rows)
                / len(comparable)
                if comparable
                else None
            ),

            "class_agreement_rate_when_both_detected": (
                len(class_agreement_rows)
                / len(both_detected_rows)
                if both_detected_rows
                else None
            ),

            "mean_carla_confidence": mean_or_none(
                [
                    row[
                        "carla_confidence"
                    ]
                    for row
                    in carla_detected_rows
                ]
            ),

            "mean_he_confidence": mean_or_none(
                [
                    row[
                        "he_confidence"
                    ]
                    for row
                    in he_detected_rows
                ]
            ),

            "mean_confidence_delta_he_minus_carla": mean_or_none(
                [
                    row[
                        "confidence_delta_he_minus_carla"
                    ]
                    for row
                    in both_detected_rows
                ]
            ),

            "mean_abs_confidence_delta": mean_or_none(
                [
                    abs(
                        row[
                            "confidence_delta_he_minus_carla"
                        ]
                    )
                    for row
                    in both_detected_rows
                    if row[
                        "confidence_delta_he_minus_carla"
                    ] is not None
                ]
            ),

            "mean_carla_detector_gt_iou": mean_or_none(
                [
                    row[
                        "carla_detector_gt_iou"
                    ]
                    for row
                    in carla_detected_rows
                ]
            ),

            "mean_he_detector_gt_iou": mean_or_none(
                [
                    row[
                        "he_detector_gt_iou"
                    ]
                    for row
                    in he_detected_rows
                ]
            ),

            "first_carla_detection_frame": first_true_frame(
                comparable,
                "carla_detected"
            ),

            "first_he_detection_frame": first_true_frame(
                comparable,
                "he_detected"
            ),

            "longest_carla_miss_streak": longest_false_streak(
                carla_flags
            ),

            "longest_he_miss_streak": longest_false_streak(
                he_flags
            ),

            "low_occlusion_frames": len(
                low_occlusion
            ),

            "low_occlusion_detection_agreement_rate": (
                len(low_occ_agreement)
                / len(low_occlusion)
                if low_occlusion
                else None
            ),

            "low_occlusion_carla_detection_rate": (
                sum(
                    row["carla_detected"]
                    for row in low_occlusion
                )
                / len(low_occlusion)
                if low_occlusion
                else None
            ),

            "low_occlusion_he_detection_rate": (
                sum(
                    row["he_detected"]
                    for row in low_occlusion
                )
                / len(low_occlusion)
                if low_occlusion
                else None
            ),

            "low_occlusion_both_detected_rate": (
                len(low_occ_both_detected)
                / len(low_occlusion)
                if low_occlusion
                else None
            ),
            "supported_carla_detection_rate": (
                len(supported_carla_rows)
                / len(comparable)
                if comparable
                else None
            ),

            "supported_he_detection_rate": (
                len(supported_he_rows)
                / len(comparable)
                if comparable
                else None
            ),

            "supported_detection_agreement_rate": (
                len(supported_agreement_rows)
                / len(comparable)
                if comparable
                else None
            ),

            "supported_both_detected_rate": (
                len(supported_both_rows)
                / len(comparable)
                if comparable
                else None
            ),

            "supported_class_agreement_rate_when_both_detected": (
                len(supported_class_agreement_rows)
                / len(supported_both_rows)
                if supported_both_rows
                else None
            ),

            "supported_mean_carla_confidence": mean_or_none(
                [
                    row[
                        "carla_supported_confidence"
                    ]
                    for row in supported_carla_rows
                ]
            ),

            "supported_mean_he_confidence": mean_or_none(
                [
                    row[
                        "he_supported_confidence"
                    ]
                    for row in supported_he_rows
                ]
            ),

            "supported_mean_abs_confidence_delta": mean_or_none(
                [
                    abs(
                        row[
                            "supported_confidence_delta_he_minus_carla"
                        ]
                    )
                    for row in supported_both_rows
                    if row[
                        "supported_confidence_delta_he_minus_carla"
                    ] is not None
                ]
            ),

            "supported_mean_carla_gt_iou": mean_or_none(
                [
                    row[
                        "carla_supported_gt_iou"
                    ]
                    for row in supported_carla_rows
                ]
            ),

            "supported_mean_he_gt_iou": mean_or_none(
                [
                    row[
                        "he_supported_gt_iou"
                    ]
                    for row in supported_he_rows
                ]
            ),
        }

        actor_summaries[
            actor_id
        ] = summary

    # ------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------

    comparable_rows = [
        row
        for row in rows
        if row[
            "carla_present"
        ]
        and row[
            "he_present"
        ]
        and row[
            "carla_visible"
        ]
        and row[
            "he_visible"
        ]
    ]

    both_detected_rows = [
        row
        for row in comparable_rows
        if row[
            "both_detected"
        ]
    ]

    supported_both_overall = [
        row
        for row in comparable_rows
        if row[
            "supported_both_detected"
        ]
    ]
    overall = {
        "common_frames": len(
            common_frames
        ),

        "actors": actor_ids,

        "comparable_actor_frames": len(
            comparable_rows
        ),

        "carla_detection_rate": (
            sum(
                row[
                    "carla_detected"
                ]
                for row
                in comparable_rows
            )
            / len(
                comparable_rows
            )
            if comparable_rows
            else None
        ),

        "he_detection_rate": (
            sum(
                row[
                    "he_detected"
                ]
                for row
                in comparable_rows
            )
            / len(
                comparable_rows
            )
            if comparable_rows
            else None
        ),

        "detection_agreement_rate": (
            sum(
                row[
                    "detection_agreement"
                ]
                for row
                in comparable_rows
            )
            / len(
                comparable_rows
            )
            if comparable_rows
            else None
        ),

        "both_detected_rate": (
            len(
                both_detected_rows
            )
            / len(
                comparable_rows
            )
            if comparable_rows
            else None
        ),

        "class_agreement_rate_when_both_detected": (
            sum(
                row[
                    "class_agreement"
                ]
                for row
                in both_detected_rows
            )
            / len(
                both_detected_rows
            )
            if both_detected_rows
            else None
        ),

        "mean_abs_confidence_delta": mean_or_none(
            [
                abs(
                    row[
                        "confidence_delta_he_minus_carla"
                    ]
                )
                for row
                in both_detected_rows
                if row[
                    "confidence_delta_he_minus_carla"
                ] is not None
            ]
        ),

        "mean_carla_detector_gt_iou": mean_or_none(
            [
                row[
                    "carla_detector_gt_iou"
                ]
                for row
                in comparable_rows
                if row[
                    "carla_detector_gt_iou"
                ] is not None
            ]
        ),

        "mean_he_detector_gt_iou": mean_or_none(
            [
                row[
                    "he_detector_gt_iou"
                ]
                for row
                in comparable_rows
                if row[
                    "he_detector_gt_iou"
                ] is not None
            ]
        ),

        "supported_carla_detection_rate": (
            sum(
                row[
                    "carla_supported_detected"
                ]
                for row in comparable_rows
            )
            / len(comparable_rows)
            if comparable_rows
            else None
        ),

        "supported_he_detection_rate": (
            sum(
                row[
                    "he_supported_detected"
                ]
                for row in comparable_rows
            )
            / len(comparable_rows)
            if comparable_rows
            else None
        ),

        "supported_detection_agreement_rate": (
            sum(
                row[
                    "supported_detection_agreement"
                ]
                for row in comparable_rows
            )
            / len(comparable_rows)
            if comparable_rows
            else None
        ),

        "supported_both_detected_rate": (
            len(
                supported_both_overall
            )
            / len(comparable_rows)
            if comparable_rows
            else None
        ),

        "supported_class_agreement_rate_when_both_detected": (
            sum(
                row[
                    "supported_class_agreement"
                ]
                for row in supported_both_overall
            )
            / len(supported_both_overall)
            if supported_both_overall
            else None
        ),

        "supported_mean_abs_confidence_delta": mean_or_none(
            [
                abs(
                    row[
                        "supported_confidence_delta_he_minus_carla"
                    ]
                )
                for row in supported_both_overall
                if row[
                    "supported_confidence_delta_he_minus_carla"
                ] is not None
            ]
        ),
    }

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    csv_path = ensure_parent(
        args.output_csv
    )

    if rows:

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                )
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    # ------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------

    json_path = ensure_parent(
        args.output_json
    )

    result = {
        "configuration": {
            "match_iou": float(
                args.match_iou
            ),

            "vehicle_classes": sorted(
                vehicle_classes
            ),
        },

        "overall": overall,

        "actors": actor_summaries,

        "rows": rows,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    # ------------------------------------------------------------
    # Print summaries
    # ------------------------------------------------------------

    print()
    print("=" * 76)
    print("PER-ACTOR RESULTS")
    print("=" * 76)

    for actor_id in actor_ids:

        s = actor_summaries[
            actor_id
        ]

        print()
        print(
            actor_id
        )

        print(
            "  comparable frames        :",
            s[
                "comparable_frames"
            ]
        )

        print(
            "  CARLA detection rate     :",
            (
                f"{s['carla_detection_rate']:.3f}"
                if s[
                    "carla_detection_rate"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  HE detection rate        :",
            (
                f"{s['he_detection_rate']:.3f}"
                if s[
                    "he_detection_rate"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  detection agreement      :",
            (
                f"{s['detection_agreement_rate']:.3f}"
                if s[
                    "detection_agreement_rate"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  both detected rate       :",
            (
                f"{s['both_detected_rate']:.3f}"
                if s[
                    "both_detected_rate"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  class agreement          :",
            (
                f"{s['class_agreement_rate_when_both_detected']:.3f}"
                if s[
                    "class_agreement_rate_when_both_detected"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  mean CARLA confidence    :",
            (
                f"{s['mean_carla_confidence']:.3f}"
                if s[
                    "mean_carla_confidence"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  mean HE confidence       :",
            (
                f"{s['mean_he_confidence']:.3f}"
                if s[
                    "mean_he_confidence"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  mean |confidence delta|  :",
            (
                f"{s['mean_abs_confidence_delta']:.3f}"
                if s[
                    "mean_abs_confidence_delta"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  mean CARLA detector IoU  :",
            (
                f"{s['mean_carla_detector_gt_iou']:.3f}"
                if s[
                    "mean_carla_detector_gt_iou"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  mean HE detector IoU     :",
            (
                f"{s['mean_he_detector_gt_iou']:.3f}"
                if s[
                    "mean_he_detector_gt_iou"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  first CARLA detection    :",
            s[
                "first_carla_detection_frame"
            ]
        )

        print(
            "  first HE detection       :",
            s[
                "first_he_detection_frame"
            ]
        )

        print(
            "  longest CARLA miss streak:",
            s[
                "longest_carla_miss_streak"
            ]
        )

        print(
            "  longest HE miss streak   :",
            s[
                "longest_he_miss_streak"
            ]
        )

        print(
            "  low-occlusion frames     :",
            s["low_occlusion_frames"]
        )

        print(
            "  low-occ CARLA det. rate  :",
            (
                f"{s['low_occlusion_carla_detection_rate']:.3f}"
                if s["low_occlusion_carla_detection_rate"] is not None
                else "N/A"
            )
        )

        print(
            "  low-occ HE det. rate     :",
            (
                f"{s['low_occlusion_he_detection_rate']:.3f}"
                if s["low_occlusion_he_detection_rate"] is not None
                else "N/A"
            )
        )

        print(
            "  low-occ agreement        :",
            (
                f"{s['low_occlusion_detection_agreement_rate']:.3f}"
                if s["low_occlusion_detection_agreement_rate"] is not None
                else "N/A"
            )
        )

        print()
        print("  --- SUPPORTED / RELAXED ---")

        print(
            "  supported CARLA det. rate:",
            (
                f"{s['supported_carla_detection_rate']:.3f}"
                if s["supported_carla_detection_rate"] is not None
                else "N/A"
            )
        )

        print(
            "  supported HE det. rate   :",
            (
                f"{s['supported_he_detection_rate']:.3f}"
                if s["supported_he_detection_rate"] is not None
                else "N/A"
            )
        )

        print(
            "  supported agreement      :",
            (
                f"{s['supported_detection_agreement_rate']:.3f}"
                if s["supported_detection_agreement_rate"] is not None
                else "N/A"
            )
        )

        print(
            "  supported both detected  :",
            (
                f"{s['supported_both_detected_rate']:.3f}"
                if s["supported_both_detected_rate"] is not None
                else "N/A"
            )
        )

        print(
            "  supported class agreement:",
            (
                f"{s['supported_class_agreement_rate_when_both_detected']:.3f}"
                if s[
                    "supported_class_agreement_rate_when_both_detected"
                ] is not None
                else "N/A"
            )
        )

        print(
            "  supported CARLA conf.    :",
            (
                f"{s['supported_mean_carla_confidence']:.3f}"
                if s["supported_mean_carla_confidence"] is not None
                else "N/A"
            )
        )

        print(
            "  supported HE conf.       :",
            (
                f"{s['supported_mean_he_confidence']:.3f}"
                if s["supported_mean_he_confidence"] is not None
                else "N/A"
            )
        )

        print(
            "  supported |conf delta|   :",
            (
                f"{s['supported_mean_abs_confidence_delta']:.3f}"
                if s["supported_mean_abs_confidence_delta"] is not None
                else "N/A"
            )
        )

    print()
    print("=" * 76)
    print("OVERALL")
    print("=" * 76)

    for key, value in overall.items():
        print(
            f"{key:40s}: {value}"
        )

    print()
    print(
        "[SAVED]",
        json_path
    )

    print(
        "[SAVED]",
        csv_path
    )

    print("=" * 76)


if __name__ == "__main__":
    main()