#!/usr/bin/env python3
"""
compare_scenario_carla_he_pair_v1.py

Frame-accurate apples-to-apples evaluator for recordings produced by:

    record_scenario_carla_he_pair_v1.py

The comparison is intentionally based on the actual visible pixels:

    CARLA mask = physical target instance silhouette
    HE mask    = final HE rendered alpha silhouette

NOT compared as primary metrics:
    - CARLA 3-D bounding box vs HE alpha box
    - HE analytical target box vs CARLA mask
    - support-plane proxy box vs visible CARLA pixels

Primary metrics:
    visibility agreement
    delta center x/y
    delta bottom y
    delta visible width/height
    visible bbox IoU
    visible mask IoU
    visible mask Dice
    visible area ratio

Pair validity checks happen BEFORE interpreting those errors:
    same resolved scenario hash
    same scenario frame
    same camera world pose
    same actor world pose
    same independent CARLA geometric viewpoint angle

Angle diagnostics:
    CARLA geometric viewpoint
    HE independently reconstructed geometric viewpoint
    HE requested viewpoint angle
    HE selected integer sprite angle

The script aborts if the camera or actor trajectories are not paired within
the configured tolerances.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--carla-dir",
        required=True,
    )

    parser.add_argument(
        "--he-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--camera-position-tolerance-m",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--camera-angle-tolerance-deg",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--actor-position-tolerance-m",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--actor-angle-tolerance-deg",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
    )

    return parser


# ============================================================
# I/O
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()

            if text:
                row = json.loads(text)

                rows.append(row)

    return rows


def load_mask(root, relative_path):
    path = (
        Path(root)
        /
        relative_path
    )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read mask: {path}"
        )

    return image > 0


# ============================================================
# Math
# ============================================================

def circular_signed_delta_deg(
    a,
    b,
):
    return (
        (
            float(a)
            -
            float(b)
            +
            180.0
        )
        %
        360.0
        -
        180.0
    )


def circular_abs_delta_deg(
    a,
    b,
):
    return abs(
        circular_signed_delta_deg(
            a,
            b,
        )
    )


def transform_position_distance(
    a,
    b,
):
    return math.sqrt(
        (
            float(a["x"])
            -
            float(b["x"])
        ) ** 2
        +
        (
            float(a["y"])
            -
            float(b["y"])
        ) ** 2
        +
        (
            float(a["z"])
            -
            float(b["z"])
        ) ** 2
    )


def max_transform_angle_error(
    a,
    b,
):
    return max(
        circular_abs_delta_deg(
            a.get("pitch", 0.0),
            b.get("pitch", 0.0),
        ),
        circular_abs_delta_deg(
            a.get("yaw", 0.0),
            b.get("yaw", 0.0),
        ),
        circular_abs_delta_deg(
            a.get("roll", 0.0),
            b.get("roll", 0.0),
        ),
    )


# ============================================================
# Mask / bbox metrics
# ============================================================

def measurement_from_mask(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width_px": float(x2 - x1 + 1),
        "height_px": float(y2 - y1 + 1),
        "center_x_px": float((x1 + x2) / 2.0),
        "center_y_px": float((y1 + y2) / 2.0),
        "bottom_y_px": float(y2),
        "area_px": int(np.count_nonzero(mask)),
    }


def bbox_iou(a, b):
    if a is None or b is None:
        return float("nan")

    ix1 = max(
        float(a["x1"]),
        float(b["x1"]),
    )

    iy1 = max(
        float(a["y1"]),
        float(b["y1"]),
    )

    ix2 = min(
        float(a["x2"]),
        float(b["x2"]),
    )

    iy2 = min(
        float(a["y2"]),
        float(b["y2"]),
    )

    iw = max(
        0.0,
        ix2 - ix1 + 1.0,
    )

    ih = max(
        0.0,
        iy2 - iy1 + 1.0,
    )

    inter = iw * ih

    area_a = (
        float(a["width_px"])
        *
        float(a["height_px"])
    )

    area_b = (
        float(b["width_px"])
        *
        float(b["height_px"])
    )

    union = (
        area_a
        +
        area_b
        -
        inter
    )

    if union <= 0.0:
        return float("nan")

    return inter / union


def mask_iou(a, b):
    inter = int(
        np.count_nonzero(
            a & b
        )
    )

    union = int(
        np.count_nonzero(
            a | b
        )
    )

    if union == 0:
        return float("nan")

    return (
        float(inter)
        /
        float(union)
    )


def mask_dice(a, b):
    inter = int(
        np.count_nonzero(
            a & b
        )
    )

    total = (
        int(
            np.count_nonzero(a)
        )
        +
        int(
            np.count_nonzero(b)
        )
    )

    if total == 0:
        return float("nan")

    return (
        2.0
        *
        float(inter)
        /
        float(total)
    )


def safe_ratio(a, b):
    if (
        a is None
        or
        b is None
        or
        float(b) == 0.0
    ):
        return float("nan")

    return (
        float(a)
        /
        float(b)
    )


def mean_finite(values):
    values = [
        float(v)
        for v in values
        if v is not None
        and math.isfinite(
            float(v)
        )
    ]

    if not values:
        return float("nan")

    return float(
        np.mean(values)
    )


# ============================================================
# Video
# ============================================================

def draw_box(
    bgr,
    measurement,
    color,
    thickness=2,
):
    if measurement is None:
        return

    cv2.rectangle(
        bgr,
        (
            int(measurement["x1"]),
            int(measurement["y1"]),
        ),
        (
            int(measurement["x2"]),
            int(measurement["y2"]),
        ),
        color,
        thickness,
    )


def put_lines(
    bgr,
    lines,
    x=8,
    y=18,
    line_height=17,
):
    for index, text in enumerate(lines):
        cv2.putText(
            bgr,
            str(text),
            (
                int(x),
                int(
                    y
                    +
                    index
                    *
                    line_height
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def make_pair_video(
    carla_dir,
    he_dir,
    output_path,
    rows,
    fps,
    width,
    height,
):
    carla_cap = cv2.VideoCapture(
        str(
            Path(carla_dir)
            /
            "carla_reference.mp4"
        )
    )

    he_cap = cv2.VideoCapture(
        str(
            Path(he_dir)
            /
            "he_replay.mp4"
        )
    )

    if (
        not carla_cap.isOpened()
        or
        not he_cap.isOpened()
    ):
        raise RuntimeError(
            "Could not open input videos."
        )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        float(fps),
        (
            int(width) * 3,
            int(height),
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open pair video: "
            f"{output_path}"
        )

    try:
        for row in rows:
            ok_c, frame_c = (
                carla_cap.read()
            )

            ok_h, frame_h = (
                he_cap.read()
            )

            if not ok_c or not ok_h:
                raise RuntimeError(
                    "Input video ended before metadata."
                )

            carla_mask = load_mask(
                carla_dir,
                row["carla_mask_path"],
            )

            he_mask = load_mask(
                he_dir,
                row["he_mask_path"],
            )

            carla_measurement = (
                measurement_from_mask(
                    carla_mask
                )
            )

            he_measurement = (
                measurement_from_mask(
                    he_mask
                )
            )

            left = frame_c.copy()
            middle = frame_h.copy()

            # Green = CARLA visible pixels.
            draw_box(
                left,
                carla_measurement,
                (0, 255, 0),
            )

            # Red = HE actual final-alpha pixels.
            draw_box(
                middle,
                he_measurement,
                (0, 0, 255),
            )

            overlay = cv2.addWeighted(
                frame_c,
                0.5,
                frame_h,
                0.5,
                0.0,
            )

            overlay[
                carla_mask
            ] = (
                0.35
                *
                overlay[
                    carla_mask
                ].astype(
                    np.float32
                )
                +
                0.65
                *
                np.array(
                    [0, 255, 0],
                    dtype=np.float32,
                )
            ).astype(
                np.uint8
            )

            overlay[
                he_mask
            ] = (
                0.35
                *
                overlay[
                    he_mask
                ].astype(
                    np.float32
                )
                +
                0.65
                *
                np.array(
                    [0, 0, 255],
                    dtype=np.float32,
                )
            ).astype(
                np.uint8
            )

            lines = [
                (
                    f"frame={row['scenario_frame']} "
                    f"t={float(row['t_s']):.2f}s "
                    f"d={float(row['distance_m']):.2f}m"
                    if math.isfinite(
                        float(row["distance_m"])
                    )
                    else
                    (
                        f"frame={row['scenario_frame']} "
                        f"t={float(row['t_s']):.2f}s"
                    )
                ),
                (
                    f"dCx={row['delta_cx_px']} "
                    f"dB={row['delta_bottom_y_px']} "
                    f"dW={row['delta_width_px']} "
                    f"dH={row['delta_height_px']}"
                ),
                (
                    f"bboxIoU={row['bbox_iou']} "
                    f"maskIoU={row['mask_iou']}"
                ),
                (
                    f"CARLA angle={row['carla_geom_angle_deg']} "
                    f"HE req={row['he_requested_angle_deg']} "
                    f"HE sel={row['he_selected_angle_deg']}"
                ),
            ]

            put_lines(
                overlay,
                lines,
            )

            title_left = left.copy()
            title_middle = middle.copy()

            put_lines(
                title_left,
                ["CARLA physical"],
            )

            put_lines(
                title_middle,
                ["HE replay"],
            )

            combined = np.concatenate(
                [
                    title_left,
                    title_middle,
                    overlay,
                ],
                axis=1,
            )

            writer.write(
                combined
            )

    finally:
        carla_cap.release()
        he_cap.release()
        writer.release()


# ============================================================
# Main
# ============================================================

def main():
    args = build_parser().parse_args()

    carla_dir = Path(
        args.carla_dir
    ).resolve()

    he_dir = Path(
        args.he_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    carla_setup = load_json(
        carla_dir / "setup.json"
    )

    he_setup = load_json(
        he_dir / "setup.json"
    )

    if (
        carla_setup.get(
            "condition"
        )
        !=
        "carla"
    ):
        raise RuntimeError(
            "carla-dir is not a CARLA reference run."
        )

    if (
        he_setup.get(
            "condition"
        )
        !=
        "he"
    ):
        raise RuntimeError(
            "he-dir is not an HE replay run."
        )

    if (
        carla_setup[
            "resolved_sha256"
        ]
        !=
        he_setup[
            "resolved_sha256"
        ]
    ):
        raise RuntimeError(
            "Resolved scenario hashes differ."
        )

    if (
        carla_setup[
            "camera"
        ]
        !=
        he_setup[
            "camera"
        ]
    ):
        raise RuntimeError(
            "Camera configurations differ."
        )

    if (
        carla_setup[
            "execution_origin"
        ]
        !=
        he_setup[
            "execution_origin"
        ]
    ):
        raise RuntimeError(
            "Execution origins differ."
        )

    if (
        carla_setup[
            "actor_id"
        ]
        !=
        he_setup[
            "actor_id"
        ]
    ):
        raise RuntimeError(
            "Target actor IDs differ."
        )

    carla_rows = load_jsonl(
        carla_dir
        /
        "frames.jsonl"
    )

    he_rows = load_jsonl(
        he_dir
        /
        "frames.jsonl"
    )

    if (
        len(carla_rows)
        !=
        len(he_rows)
    ):
        raise RuntimeError(
            "CARLA and HE frame counts differ: "
            f"{len(carla_rows)} vs "
            f"{len(he_rows)}"
        )

    results = []

    max_camera_position_error = 0.0
    max_camera_angle_error = 0.0
    max_actor_position_error = 0.0
    max_actor_angle_error = 0.0
    max_geom_angle_pair_error = 0.0

    for carla_row, he_row in zip(
        carla_rows,
        he_rows,
    ):
        frame_c = int(
            carla_row[
                "scenario_frame"
            ]
        )

        frame_h = int(
            he_row[
                "scenario_frame"
            ]
        )

        if frame_c != frame_h:
            raise RuntimeError(
                "Scenario-frame mismatch: "
                f"{frame_c} vs {frame_h}"
            )

        camera_pos_error = (
            transform_position_distance(
                carla_row[
                    "camera_transform"
                ],
                he_row[
                    "camera_transform"
                ],
            )
        )

        camera_angle_error = (
            max_transform_angle_error(
                carla_row[
                    "camera_transform"
                ],
                he_row[
                    "camera_transform"
                ],
            )
        )

        max_camera_position_error = max(
            max_camera_position_error,
            camera_pos_error,
        )

        max_camera_angle_error = max(
            max_camera_angle_error,
            camera_angle_error,
        )

        if (
            camera_pos_error
            >
            float(
                args.camera_position_tolerance_m
            )
            or
            camera_angle_error
            >
            float(
                args.camera_angle_tolerance_deg
            )
        ):
            raise RuntimeError(
                f"Frame {frame_c}: camera pair mismatch. "
                f"position={camera_pos_error:.6f}m, "
                f"angle={camera_angle_error:.6f}deg"
            )

        actor_c = (
            carla_row.get(
                "actor_transform"
            )
        )

        actor_h = (
            he_row.get(
                "actor_transform"
            )
        )

        actor_pos_error = float("nan")
        actor_angle_error = float("nan")

        if (
            actor_c is None
            or
            actor_h is None
        ):
            if (
                actor_c is None
            ) != (
                actor_h is None
            ):
                raise RuntimeError(
                    f"Frame {frame_c}: actor-active mismatch."
                )
        else:
            actor_pos_error = (
                transform_position_distance(
                    actor_c,
                    actor_h,
                )
            )

            actor_angle_error = (
                max_transform_angle_error(
                    actor_c,
                    actor_h,
                )
            )

            max_actor_position_error = max(
                max_actor_position_error,
                actor_pos_error,
            )

            max_actor_angle_error = max(
                max_actor_angle_error,
                actor_angle_error,
            )

            if (
                actor_pos_error
                >
                float(
                    args.actor_position_tolerance_m
                )
                or
                actor_angle_error
                >
                float(
                    args.actor_angle_tolerance_deg
                )
            ):
                raise RuntimeError(
                    f"Frame {frame_c}: actor pair mismatch. "
                    f"position={actor_pos_error:.6f}m, "
                    f"angle={actor_angle_error:.6f}deg"
                )

        carla_angle = (
            carla_row.get(
                "carla_geometric_view_angle_deg"
            )
        )

        he_geom_angle = (
            he_row.get(
                "carla_geometric_view_angle_deg"
            )
        )

        geom_pair_error = float("nan")

        if (
            carla_angle is not None
            and
            he_geom_angle is not None
        ):
            geom_pair_error = (
                circular_abs_delta_deg(
                    carla_angle,
                    he_geom_angle,
                )
            )

            max_geom_angle_pair_error = max(
                max_geom_angle_pair_error,
                geom_pair_error,
            )

        he_requested = (
            he_row.get(
                "he_requested_viewpoint_angle_deg"
            )
        )

        he_selected = (
            he_row.get(
                "he_selected_angle_deg"
            )
        )

        requested_vs_carla = float(
            "nan"
        )

        selected_vs_carla = float(
            "nan"
        )

        if (
            carla_angle is not None
            and
            he_requested is not None
        ):
            requested_vs_carla = (
                circular_signed_delta_deg(
                    he_requested,
                    carla_angle,
                )
            )

        if (
            carla_angle is not None
            and
            he_selected is not None
        ):
            selected_vs_carla = (
                circular_signed_delta_deg(
                    he_selected,
                    carla_angle,
                )
            )

        carla_mask = load_mask(
            carla_dir,
            carla_row[
                "mask_path"
            ],
        )

        he_mask = load_mask(
            he_dir,
            he_row[
                "mask_path"
            ],
        )

        if (
            carla_mask.shape
            !=
            he_mask.shape
        ):
            raise RuntimeError(
                f"Frame {frame_c}: mask shapes differ: "
                f"{carla_mask.shape} vs {he_mask.shape}"
            )

        carla_measurement = (
            measurement_from_mask(
                carla_mask
            )
        )

        he_measurement = (
            measurement_from_mask(
                he_mask
            )
        )

        carla_visible = bool(
            carla_measurement
            is not None
        )

        he_visible = bool(
            he_measurement
            is not None
        )

        both_visible = (
            carla_visible
            and
            he_visible
        )

        delta_cx = float("nan")
        delta_cy = float("nan")
        delta_bottom = float("nan")
        delta_width = float("nan")
        delta_height = float("nan")
        width_ratio = float("nan")
        height_ratio = float("nan")
        area_ratio = float("nan")
        bbox_overlap = float("nan")
        mask_overlap = float("nan")
        dice = float("nan")

        if both_visible:
            delta_cx = (
                float(
                    he_measurement[
                        "center_x_px"
                    ]
                )
                -
                float(
                    carla_measurement[
                        "center_x_px"
                    ]
                )
            )

            delta_cy = (
                float(
                    he_measurement[
                        "center_y_px"
                    ]
                )
                -
                float(
                    carla_measurement[
                        "center_y_px"
                    ]
                )
            )

            delta_bottom = (
                float(
                    he_measurement[
                        "bottom_y_px"
                    ]
                )
                -
                float(
                    carla_measurement[
                        "bottom_y_px"
                    ]
                )
            )

            delta_width = (
                float(
                    he_measurement[
                        "width_px"
                    ]
                )
                -
                float(
                    carla_measurement[
                        "width_px"
                    ]
                )
            )

            delta_height = (
                float(
                    he_measurement[
                        "height_px"
                    ]
                )
                -
                float(
                    carla_measurement[
                        "height_px"
                    ]
                )
            )

            width_ratio = safe_ratio(
                he_measurement[
                    "width_px"
                ],
                carla_measurement[
                    "width_px"
                ],
            )

            height_ratio = safe_ratio(
                he_measurement[
                    "height_px"
                ],
                carla_measurement[
                    "height_px"
                ],
            )

            area_ratio = safe_ratio(
                he_measurement[
                    "area_px"
                ],
                carla_measurement[
                    "area_px"
                ],
            )

            bbox_overlap = bbox_iou(
                carla_measurement,
                he_measurement,
            )

            mask_overlap = mask_iou(
                carla_mask,
                he_mask,
            )

            dice = mask_dice(
                carla_mask,
                he_mask,
            )

        distance_m = float("nan")
        forward_depth_m = float("nan")

        rel = carla_row.get(
            "camera_relative"
        )

        if rel is not None:
            distance_m = float(
                rel[
                    "camera_distance_m"
                ]
            )

            forward_depth_m = float(
                rel[
                    "camera_forward_m"
                ]
            )

        result = {
            "scenario_frame":
                frame_c,
            "t_s":
                float(
                    carla_row[
                        "t_s"
                    ]
                ),
            "distance_m":
                distance_m,
            "forward_depth_m":
                forward_depth_m,

            # Pair validity.
            "camera_position_pair_error_m":
                camera_pos_error,
            "camera_angle_pair_error_deg":
                camera_angle_error,
            "actor_position_pair_error_m":
                actor_pos_error,
            "actor_angle_pair_error_deg":
                actor_angle_error,
            "geometric_view_pair_error_deg":
                geom_pair_error,

            # Independent angle chain.
            "carla_geom_angle_deg":
                (
                    float(carla_angle)
                    if carla_angle is not None
                    else float("nan")
                ),
            "he_geom_angle_deg":
                (
                    float(he_geom_angle)
                    if he_geom_angle is not None
                    else float("nan")
                ),
            "he_requested_angle_deg":
                (
                    float(he_requested)
                    if he_requested is not None
                    else float("nan")
                ),
            "he_selected_angle_deg":
                (
                    float(he_selected)
                    if he_selected is not None
                    else float("nan")
                ),
            "requested_minus_carla_angle_deg":
                requested_vs_carla,
            "selected_minus_carla_angle_deg":
                selected_vs_carla,

            "he_selected_distance_m":
                he_row.get(
                    "he_selected_distance_m"
                ),
            "he_selected_elevation_deg":
                he_row.get(
                    "he_selected_elevation_deg"
                ),
            "he_reason":
                he_row.get(
                    "he_reason",
                    "",
                ),

            # Actual visible-pixel comparison.
            "carla_visible":
                int(carla_visible),
            "he_visible":
                int(he_visible),
            "visibility_agree":
                int(
                    carla_visible
                    ==
                    he_visible
                ),
            "both_visible":
                int(both_visible),

            "carla_mask_path":
                carla_row[
                    "mask_path"
                ],
            "he_mask_path":
                he_row[
                    "mask_path"
                ],

            "carla_cx_px":
                (
                    carla_measurement[
                        "center_x_px"
                    ]
                    if carla_measurement
                    is not None
                    else float("nan")
                ),
            "carla_cy_px":
                (
                    carla_measurement[
                        "center_y_px"
                    ]
                    if carla_measurement
                    is not None
                    else float("nan")
                ),
            "carla_bottom_y_px":
                (
                    carla_measurement[
                        "bottom_y_px"
                    ]
                    if carla_measurement
                    is not None
                    else float("nan")
                ),
            "carla_width_px":
                (
                    carla_measurement[
                        "width_px"
                    ]
                    if carla_measurement
                    is not None
                    else float("nan")
                ),
            "carla_height_px":
                (
                    carla_measurement[
                        "height_px"
                    ]
                    if carla_measurement
                    is not None
                    else float("nan")
                ),
            "carla_area_px":
                (
                    carla_measurement[
                        "area_px"
                    ]
                    if carla_measurement
                    is not None
                    else 0
                ),

            "he_cx_px":
                (
                    he_measurement[
                        "center_x_px"
                    ]
                    if he_measurement
                    is not None
                    else float("nan")
                ),
            "he_cy_px":
                (
                    he_measurement[
                        "center_y_px"
                    ]
                    if he_measurement
                    is not None
                    else float("nan")
                ),
            "he_bottom_y_px":
                (
                    he_measurement[
                        "bottom_y_px"
                    ]
                    if he_measurement
                    is not None
                    else float("nan")
                ),
            "he_width_px":
                (
                    he_measurement[
                        "width_px"
                    ]
                    if he_measurement
                    is not None
                    else float("nan")
                ),
            "he_height_px":
                (
                    he_measurement[
                        "height_px"
                    ]
                    if he_measurement
                    is not None
                    else float("nan")
                ),
            "he_area_px":
                (
                    he_measurement[
                        "area_px"
                    ]
                    if he_measurement
                    is not None
                    else 0
                ),

            "delta_cx_px":
                delta_cx,
            "delta_cy_px":
                delta_cy,
            "delta_bottom_y_px":
                delta_bottom,
            "delta_width_px":
                delta_width,
            "delta_height_px":
                delta_height,
            "width_ratio_he_over_carla":
                width_ratio,
            "height_ratio_he_over_carla":
                height_ratio,
            "area_ratio_he_over_carla":
                area_ratio,
            "bbox_iou":
                bbox_overlap,
            "mask_iou":
                mask_overlap,
            "mask_dice":
                dice,
        }

        results.append(
            result
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------
    csv_path = (
        output_dir
        /
        "comparison.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    both = [
        row
        for row in results
        if int(
            row["both_visible"]
        )
        == 1
    ]

    visibility_agreement = (
        sum(
            int(
                row[
                    "visibility_agree"
                ]
            )
            for row in results
        )
        /
        max(
            1,
            len(results),
        )
    )

    summary = {
        "frames":
            len(results),
        "both_visible_frames":
            len(both),
        "visibility_agreement":
            visibility_agreement,
        "max_camera_position_pair_error_m":
            max_camera_position_error,
        "max_camera_angle_pair_error_deg":
            max_camera_angle_error,
        "max_actor_position_pair_error_m":
            max_actor_position_error,
        "max_actor_angle_pair_error_deg":
            max_actor_angle_error,
        "max_geometric_view_pair_error_deg":
            max_geom_angle_pair_error,

        "mae_cx_px":
            mean_finite(
                abs(
                    float(
                        row[
                            "delta_cx_px"
                        ]
                    )
                )
                for row in both
            ),
        "mae_cy_px":
            mean_finite(
                abs(
                    float(
                        row[
                            "delta_cy_px"
                        ]
                    )
                )
                for row in both
            ),
        "mae_bottom_y_px":
            mean_finite(
                abs(
                    float(
                        row[
                            "delta_bottom_y_px"
                        ]
                    )
                )
                for row in both
            ),
        "mae_width_px":
            mean_finite(
                abs(
                    float(
                        row[
                            "delta_width_px"
                        ]
                    )
                )
                for row in both
            ),
        "mae_height_px":
            mean_finite(
                abs(
                    float(
                        row[
                            "delta_height_px"
                        ]
                    )
                )
                for row in both
            ),
        "mean_bbox_iou":
            mean_finite(
                row[
                    "bbox_iou"
                ]
                for row in both
            ),
        "mean_mask_iou":
            mean_finite(
                row[
                    "mask_iou"
                ]
                for row in both
            ),
        "mean_mask_dice":
            mean_finite(
                row[
                    "mask_dice"
                ]
                for row in both
            ),
        "mae_requested_vs_carla_angle_deg":
            mean_finite(
                abs(
                    float(
                        row[
                            "requested_minus_carla_angle_deg"
                        ]
                    )
                )
                for row in results
            ),
        "mae_selected_vs_carla_angle_deg":
            mean_finite(
                abs(
                    float(
                        row[
                            "selected_minus_carla_angle_deg"
                        ]
                    )
                )
                for row in results
            ),
    }

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            summary,
            fp,
            indent=2,
        )

    # --------------------------------------------------------
    # Worst frames
    # --------------------------------------------------------
    worst = sorted(
        both,
        key=lambda row: (
            float(
                row[
                    "mask_iou"
                ]
            )
            if math.isfinite(
                float(
                    row[
                        "mask_iou"
                    ]
                )
            )
            else 1.0
        ),
    )[:max(1, int(args.top))]

    print()
    print("=" * 112)
    print("CARLA <-> HE PAIR COMPARISON V1")
    print("=" * 112)
    print("[pair validity]")
    print(
        "  max camera position error :",
        f"{max_camera_position_error:.8f} m",
    )
    print(
        "  max camera angle error    :",
        f"{max_camera_angle_error:.8f} deg",
    )
    print(
        "  max actor position error  :",
        f"{max_actor_position_error:.8f} m",
    )
    print(
        "  max actor angle error     :",
        f"{max_actor_angle_error:.8f} deg",
    )
    print(
        "  max geometric-view error  :",
        f"{max_geom_angle_pair_error:.8f} deg",
    )
    print()
    print("[visible-pixel metrics]")
    print(
        "  frames                    :",
        len(results),
    )
    print(
        "  both visible              :",
        len(both),
    )
    print(
        "  visibility agreement      :",
        f"{visibility_agreement:.4f}",
    )
    print(
        "  MAE cx                    :",
        f"{summary['mae_cx_px']:.3f} px",
    )
    print(
        "  MAE cy                    :",
        f"{summary['mae_cy_px']:.3f} px",
    )
    print(
        "  MAE bottom_y              :",
        f"{summary['mae_bottom_y_px']:.3f} px",
    )
    print(
        "  MAE width                 :",
        f"{summary['mae_width_px']:.3f} px",
    )
    print(
        "  MAE height                :",
        f"{summary['mae_height_px']:.3f} px",
    )
    print(
        "  mean bbox IoU             :",
        f"{summary['mean_bbox_iou']:.4f}",
    )
    print(
        "  mean mask IoU             :",
        f"{summary['mean_mask_iou']:.4f}",
    )
    print(
        "  mean mask Dice            :",
        f"{summary['mean_mask_dice']:.4f}",
    )
    print()
    print("[angle chain]")
    print(
        "  MAE HE requested vs CARLA :",
        f"{summary['mae_requested_vs_carla_angle_deg']:.4f} deg",
    )
    print(
        "  MAE HE selected vs CARLA  :",
        f"{summary['mae_selected_vs_carla_angle_deg']:.4f} deg",
    )
    print()
    print(
        "frame   t(s)   dist   cAng   req    sel   "
        "dCx   dBot    dW    dH   bIoU   mIoU   reason"
    )

    for row in worst:
        print(
            f"{int(row['scenario_frame']):5d} "
            f"{float(row['t_s']):6.2f} "
            f"{float(row['distance_m']):6.2f} "
            f"{float(row['carla_geom_angle_deg']):6.1f} "
            f"{float(row['he_requested_angle_deg']):6.1f} "
            f"{float(row['he_selected_angle_deg']):6.1f} "
            f"{float(row['delta_cx_px']):6.1f} "
            f"{float(row['delta_bottom_y_px']):6.1f} "
            f"{float(row['delta_width_px']):6.1f} "
            f"{float(row['delta_height_px']):6.1f} "
            f"{float(row['bbox_iou']):6.3f} "
            f"{float(row['mask_iou']):6.3f} "
            f"{row['he_reason']}"
        )

    print()
    print("[comparison csv]", csv_path)
    print("[summary json]  ", summary_path)

    if args.save_video:
        video_path = (
            output_dir
            /
            "pair_comparison.mp4"
        )

        camera = carla_setup[
            "camera"
        ]

        make_pair_video(
            carla_dir=carla_dir,
            he_dir=he_dir,
            output_path=video_path,
            rows=results,
            fps=float(
                carla_setup["fps"]
            ),
            width=int(
                camera["width"]
            ),
            height=int(
                camera["height"]
            ),
        )

        print("[pair video]    ", video_path)

    print("=" * 112)


if __name__ == "__main__":
    main()
