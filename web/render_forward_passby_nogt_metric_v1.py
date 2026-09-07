#!/usr/bin/env python3
"""
render_forward_passby_nogt_metric_v1.py

No-GT forward-pass visualization.

View selection:
    CURRENT production HE view-matrix selector.

Placement/scale:
    analytic sprite-alpha metric geometry from the copied renderer.

GT mask is used only after rendering for evaluation and inpainting the original
CARLA Tesla out of the background. It does not influence selection, scale,
placement, or clipping.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import carla

import gtmask_oracle_helpers as oracle


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--carla-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--bank",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--renderer-copy",
        type=Path,
        default=Path(
            r"D:\HallucinationEngine\web\he_camera_renderer_gtmask_oracle_v1.py"
        ),
    )

    p.add_argument(
        "--frame-start",
        type=int,
        default=0,
    )

    p.add_argument(
        "--frame-end",
        type=int,
        default=60,
    )

    p.add_argument(
        "--frame-step",
        type=int,
        default=1,
    )

    p.add_argument(
        "--mask-dilate-px",
        type=int,
        default=7,
    )

    p.add_argument(
        "--inpaint-radius",
        type=float,
        default=4.0,
    )

    p.add_argument(
        "--background-mode",
        choices=[
            "inpaint",
            "original",
            "solid",
        ],
        default="inpaint",
        help=(
            "Preview background. 'solid' is a clean no-inpainting "
            "diagnostic background; GT is still used only for scoring."
        ),
    )

    p.add_argument(
        "--source-distance-m",
        type=float,
        default=5.0,
        help=(
            "Restrict the experimental view matrix to this source distance. "
            "Use a negative value to keep all source distances."
        ),
    )

    p.add_argument(
        "--save-frame-step",
        type=int,
        default=5,
    )

    p.add_argument(
        "--distance-selection-mode",
        choices=[
            "linear",
            "log",
            "inverse_depth",
        ],
        default="linear",
    )

    p.add_argument(
        "--viewpoint-lateral-sign",
        type=float,
        choices=[
            -1.0,
            1.0,
        ],
        default=1.0,
    )

    return p


def restrict_view_matrix_distance(
    view_matrix,
    source_distance_m,
):
    if float(source_distance_m) <= 0.0:
        return view_matrix

    target = float(source_distance_m)
    filtered_index = {}

    for key, record in view_matrix["index"].items():
        angle, distance, elevation = key
        if abs(float(distance) - target) <= 1e-6:
            filtered_index[(angle, float(distance), elevation)] = record

    if not filtered_index:
        raise RuntimeError(
            f"No sprites found at source distance {target} m."
        )

    filtered = dict(view_matrix)
    filtered["index"] = filtered_index
    filtered["distances"] = [target]
    return filtered


def dimensions_from_physical_bbox(view_matrix):
    physical_bbox = view_matrix.get("physical_bbox")
    if not physical_bbox:
        raise RuntimeError(
            "View matrix is missing physical_bbox metadata."
        )

    return {
        "length_m": 2.0 * float(physical_bbox["extent_x_m"]),
        "width_m": 2.0 * float(physical_bbox["extent_y_m"]),
        "height_m": 2.0 * float(physical_bbox["extent_z_m"]),
    }


def mask_from_meta(meta, shape):
    rendered_bbox = meta.get("rendered_alpha_bbox")
    if not rendered_bbox:
        return np.zeros(shape, dtype=np.uint8)

    mask = np.zeros(shape, dtype=np.uint8)
    x1 = max(0, int(rendered_bbox["x1"]))
    y1 = max(0, int(rendered_bbox["y1"]))
    x2 = min(shape[1] - 1, int(rendered_bbox["x2"]))
    y2 = min(shape[0] - 1, int(rendered_bbox["y2"]))

    if x2 >= x1 and y2 >= y1:
        mask[y1:y2 + 1, x1:x2 + 1] = 1

    return mask


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = (
                line.strip()
            )

            if line:
                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def import_renderer(path):
    # The production renderer normally lives inside
    # driving_models/common and imports sibling modules directly,
    # e.g. he_projection_geometry_v1.
    #
    # Our experimental renderer copy lives in web/, so explicitly
    # expose the original common directory before importing it.
    repo_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    common_dir = (
        repo_root
        / "driving_models"
        / "common"
    )

    if str(common_dir) not in sys.path:
        sys.path.insert(
            0,
            str(common_dir),
        )
    spec = (
        importlib.util
        .spec_from_file_location(
            "he_camera_renderer_gtmask_oracle_v1",
            str(
                path.resolve()
            ),
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            f"Could not import renderer copy: {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def transform_from_dict(d):
    return carla.Transform(
        carla.Location(
            x=float(d["x"]),
            y=float(d["y"]),
            z=float(d["z"]),
        ),
        carla.Rotation(
            pitch=float(
                d.get(
                    "pitch",
                    0.0,
                )
            ),
            yaw=float(
                d.get(
                    "yaw",
                    0.0,
                )
            ),
            roll=float(
                d.get(
                    "roll",
                    0.0,
                )
            ),
        ),
    )


def inpaint_target(
    frame_bgr,
    target_mask,
    dilate_px,
    radius,
):
    kernel_size = max(
        1,
        int(
            dilate_px
        ),
    )

    kernel = np.ones(
        (
            kernel_size,
            kernel_size,
        ),
        dtype=np.uint8,
    )

    inpaint_mask = cv2.dilate(
        (
            target_mask
            > 0
        ).astype(
            np.uint8
        )
        * 255,
        kernel,
        iterations=1,
    )

    return cv2.inpaint(
        frame_bgr,
        inpaint_mask,
        float(radius),
        cv2.INPAINT_TELEA,
    )


def add_text(
    canvas,
    lines,
    x=10,
    y=22,
):
    yy = int(y)

    for line in lines:
        cv2.putText(
            canvas,
            str(line),
            (
                int(x),
                yy,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        yy += 20


def make_diagnostic(
    carla_bgr,
    preview_bgr,
    gt_mask,
    fitted_mask,
    lines,
):
    h, w = (
        carla_bgr.shape[:2]
    )

    header = 90

    gt_rgb = np.repeat(
        (
            gt_mask
            > 0
        )[
            :,
            :,
            None,
        ].astype(
            np.uint8
        )
        * 255,
        3,
        axis=2,
    )

    fitted_rgb = np.repeat(
        (
            fitted_mask
            > 0
        )[
            :,
            :,
            None,
        ].astype(
            np.uint8
        )
        * 255,
        3,
        axis=2,
    )

    canvas = np.zeros(
        (
            header
            + 2 * h,
            2 * w,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        header:header+h,
        :w,
    ] = carla_bgr

    canvas[
        header:header+h,
        w:2*w,
    ] = preview_bgr

    canvas[
        header+h:header+2*h,
        :w,
    ] = gt_rgb

    canvas[
        header+h:header+2*h,
        w:2*w,
    ] = fitted_rgb

    add_text(
        canvas,
        lines,
        x=10,
        y=20,
    )

    return canvas


def main():
    args = (
        build_parser()
        .parse_args()
    )

    carla_dir = (
        args.carla_dir
        .resolve()
    )

    bank_dir = (
        args.bank
        .resolve()
    )

    output_dir = (
        args.output_dir
        .resolve()
    )

    renderer_path = (
        args.renderer_copy
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames_dir = (
        output_dir
        / "frames"
    )

    frames_dir.mkdir(
        exist_ok=True
    )

    renderer = (
        import_renderer(
            renderer_path
        )
    )

    setup = load_json(
        carla_dir
        / "setup.json"
    )

    frames = load_jsonl(
        carla_dir
        / "frames.jsonl"
    )

    camera = setup[
        "camera"
    ]

    width = int(
        camera["width"]
    )

    height = int(
        camera["height"]
    )

    fps = float(
        setup["fps"]
    )

    view_matrix_csv = (
        bank_dir
        / "view_matrix.csv"
    )

    if not view_matrix_csv.is_file():
        raise FileNotFoundError(
            f"Missing {view_matrix_csv}"
        )

    # This is exactly the configuration contract expected by the
    # production load_view_matrix_sprite_bank/select_view_matrix_sprite.
    sprite_bank = {
        "mode":
            "view_matrix",

        "view_matrix_csvs":
            [
                str(
                    view_matrix_csv
                )
            ],

        # Asset metadata will override this with resolved_target_height_m
        # when present.
        "target_height_m":
            0.75,

        "vertical_mode":
            "state_y",

        "camera_height_m":
            float(
                camera.get(
                    "z_m",
                    1.6,
                )
            ),

        "distance_selection_mode":
            str(
                args.distance_selection_mode
            ),
    }

    view_matrix = (
        renderer
        .load_view_matrix_sprite_bank(
            sprite_bank
        )
    )
    view_matrix = restrict_view_matrix_distance(
        view_matrix=view_matrix,
        source_distance_m=float(args.source_distance_m),
    )

    dimensions = dimensions_from_physical_bbox(
        view_matrix
    )

    sprite_cache = (
        renderer.SpriteCache()
    )

    video_path = (
        carla_dir
        / "carla_reference.mp4"
    )

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open {video_path}"
        )

    last_frame = (
        len(frames)
        - 1
    )

    end_frame = (
        last_frame
        if int(
            args.frame_end
        ) < 0
        else min(
            int(
                args.frame_end
            ),
            last_frame,
        )
    )

    wanted = list(
        range(
            max(
                0,
                int(
                    args.frame_start
                ),
            ),
            end_frame
            + 1,
            max(
                1,
                int(
                    args.frame_step
                ),
            ),
        )
    )

    output_fps = (
        fps
        / max(
            1,
            int(
                args.frame_step
            ),
        )
    )

    preview_writer = (
        cv2.VideoWriter(
            str(
                output_dir
                / "forward_passby_nogt_metric_preview.mp4"
            ),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            output_fps,
            (
                width,
                height,
            ),
        )
    )

    diagnostic_writer = (
        cv2.VideoWriter(
            str(
                output_dir
                / "forward_passby_nogt_metric_side_by_side.mp4"
            ),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            output_fps,
            (
                2 * width,
                90
                + 2 * height,
            ),
        )
    )

    if (
        not preview_writer.isOpened()
        or
        not diagnostic_writer.isOpened()
    ):
        raise RuntimeError(
            "Could not open output video writer."
        )

    rows = []

    print("=" * 96)
    print(
        "FORWARD PASS-BY NO-GT METRIC V1: "
        "PRODUCTION SELECTOR + ANALYTIC SPRITE-ALPHA GEOMETRY"
    )
    print("=" * 96)
    print("[renderer]   ", renderer_path)
    print("[CARLA]      ", carla_dir)
    print("[bank]       ", bank_dir)
    print("[frames]     ", f"{wanted[0]}..{wanted[-1]}")
    print("[source d]   ", args.source_distance_m)
    print("[background] ", args.background_mode)
    print(
        "[GT use]     ",
        "post-render IoU/Dice only; plus inpaint only when background-mode=inpaint",
    )
    print("=" * 96)

    for scenario_frame in wanted:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(
                scenario_frame
            ),
        )

        ok, carla_bgr = (
            cap.read()
        )

        if not ok:
            raise RuntimeError(
                "Could not read CARLA video frame "
                f"{scenario_frame}"
            )

        frame_row = (
            frames[
                scenario_frame
            ]
        )

        gt_mask_path = (
            carla_dir
            / str(
                frame_row[
                    "mask_path"
                ]
            )
        )

        gt_gray = cv2.imread(
            str(
                gt_mask_path
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        if gt_gray is None:
            raise RuntimeError(
                f"Could not read {gt_mask_path}"
            )

        gt_mask = (
            gt_gray
            > 0
        ).astype(
            np.uint8
        )

        visible = bool(
            np.any(
                gt_mask
            )
        )

        actor_dict = (
            frame_row.get(
                "actor_transform"
            )
        )

        if (
            not visible
            or
            actor_dict is None
        ):
            preview_bgr = (
                carla_bgr.copy()
            )

            rendered_mask = (
                np.zeros_like(
                    gt_mask
                )
            )

            meta = None
            fit_iou = None
            fit_dice = None

        else:
            actor_tf = (
                transform_from_dict(
                    actor_dict
                )
            )

            camera_tf = (
                transform_from_dict(
                    frame_row[
                        "camera_transform"
                    ]
                )
            )

            if args.background_mode == "inpaint":
                background_bgr = (
                    inpaint_target(
                        frame_bgr=carla_bgr,
                        target_mask=gt_mask,
                        dilate_px=int(
                            args.mask_dilate_px
                        ),
                        radius=float(
                            args.inpaint_radius
                        ),
                    )
                )
            elif args.background_mode == "original":
                background_bgr = carla_bgr.copy()
            else:
                background_bgr = np.full_like(
                    carla_bgr,
                    210,
                    dtype=np.uint8,
                )

            # Current HE SpriteCache returns RGBA and HE rendering is RGB.
            background_rgb = (
                cv2.cvtColor(
                    background_bgr,
                    cv2.COLOR_BGR2RGB,
                )
            )

            preview_rgb, meta = renderer.render_he_actor_view_matrix(
                base_rgb=background_rgb,
                actor_tf=actor_tf,
                camera_tf=camera_tf,
                dimensions=dimensions,
                sprite_bank=sprite_bank,
                view_matrix=view_matrix,
                sprite_cache=sprite_cache,
                width=width,
                height=height,
                fov=float(camera["fov_deg"]),
                geometry_mode="sprite_alpha_metric",
                projection_mode="oriented_2p5d_support",
                warp_scale_mode="independent",
                viewpoint_lateral_sign=float(args.viewpoint_lateral_sign),
            )

            preview_bgr = (
                cv2.cvtColor(
                    preview_rgb,
                    cv2.COLOR_RGB2BGR,
                )
            )

            diff = np.max(
                np.abs(
                    preview_rgb.astype(np.int16)
                    - background_rgb.astype(np.int16)
                ),
                axis=2,
            )
            rendered_mask = (
                diff > 2
            ).astype(
                np.uint8
            )
            fit_iou = oracle.mask_iou(
                gt_mask,
                rendered_mask,
            )
            fit_dice = oracle.mask_dice(
                gt_mask,
                rendered_mask,
            )

        if meta is None:
            selected_angle = None
            selected_distance = None
            selected_elevation = None
            query_angle = None
            query_distance = None
            query_elevation = None

        else:
            selected_angle = (
                meta[
                    "selected_angle"
                ]
            )

            selected_distance = (
                meta[
                    "selected_distance_m"
                ]
            )

            selected_elevation = (
                meta[
                    "selected_elevation_deg"
                ]
            )

            query_angle = (
                meta[
                    "viewpoint_angle_deg"
                ]
            )

            query_distance = (
                meta[
                    "query_distance_m"
                ]
            )

            query_elevation = (
                meta[
                    "query_elevation_deg"
                ]
            )

        lines = [
            (
                f"frame={scenario_frame} "
                f"CARLA_geom={frame_row.get('carla_geometric_view_angle_deg')} "
                f"production_query={query_angle} "
                f"selected={selected_angle}"
            ),
            (
                f"query_d={query_distance} "
                f"source_d={selected_distance} "
                f"query_e={query_elevation} "
                f"source_e={selected_elevation}"
            ),
            (
                f"postIoU={fit_iou} "
                f"postDice={fit_dice} "
                f"anchor={None if meta is None else meta.get('anchor_mode')} "
                f"noGT=True"
            ),
        ]

        diagnostic = (
            make_diagnostic(
                carla_bgr=carla_bgr,
                preview_bgr=preview_bgr,
                gt_mask=gt_mask,
                fitted_mask=rendered_mask,
                lines=lines,
            )
        )

        preview_writer.write(
            preview_bgr
        )

        diagnostic_writer.write(
            diagnostic
        )

        if (
            scenario_frame
            % max(
                1,
                int(
                    args.save_frame_step
                ),
            )
            == 0
        ):
            cv2.imwrite(
                str(
                    frames_dir
                    / (
                        f"frame_"
                        f"{scenario_frame:06d}.png"
                    )
                ),
                diagnostic,
            )

        csv_row = {
            "scenario_frame":
                int(
                    scenario_frame
                ),

            "visible":
                int(
                    visible
                ),

            "carla_geometric_view_angle_deg":
                frame_row.get(
                    "carla_geometric_view_angle_deg"
                ),

            "production_query_angle_deg":
                query_angle,

            "selected_angle_deg":
                selected_angle,

            "query_distance_m":
                query_distance,

            "selected_distance_m":
                selected_distance,

            "query_elevation_deg":
                query_elevation,

            "selected_elevation_deg":
                selected_elevation,

            "fit_iou":
                fit_iou,

            "fit_dice":
                fit_dice,

            "runtime_depth_z_m":
                (
                    None
                    if meta is None
                    else (
                        meta.get("box") or {}
                    ).get(
                        "depth_m"
                    )
                ),

            "camera_right_m":
                (
                    None
                    if meta is None
                    else (
                        meta.get("box") or {}
                    ).get(
                        "camera_right_m"
                    )
                ),

            "camera_up_m":
                (
                    None
                    if meta is None
                    else (
                        meta.get("box") or {}
                    ).get(
                        "camera_up_m"
                    )
                ),

            "target_width_px":
                (
                    None
                    if meta is None
                    else meta.get(
                        "target_box_width_px"
                    )
                ),

            "target_height_px":
                (
                    None
                    if meta is None
                    else meta.get(
                        "target_box_height_px"
                    )
                ),

            "source_alpha_width_px":
                None if meta is None else meta.get("source_alpha_width_px"),

            "source_alpha_height_px":
                None if meta is None else meta.get("source_alpha_height_px"),

            "metric_scale":
                None if meta is None else meta.get("sprite_alpha_metric_scale"),

            "target_support_anchor":
                None if meta is None else json.dumps(meta.get("target_physical_support")),

            "rendered_visible_bbox":
                None if meta is None else json.dumps(meta.get("rendered_alpha_bbox")),

            "unclipped_visible_bbox":
                None if meta is None else json.dumps(meta.get("target_visible_bbox_unclipped")),

            "anchor_mode":
                None if meta is None else meta.get("anchor_mode"),

            "render_reason":
                None if meta is None else meta.get("reason"),
        }

        rows.append(
            csv_row
        )

        print(
            f"[frame {scenario_frame:04d}] "
            f"visible={int(visible)} "
            f"query={query_angle} "
            f"selected={selected_angle} "
            f"d={selected_distance} "
            f"e={selected_elevation} "
            f"postIoU={csv_row['fit_iou']}"
        )

    cap.release()
    preview_writer.release()
    diagnostic_writer.release()

    csv_path = (
        output_dir
        / "forward_passby_nogt_metric_rows.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = (
            csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 96)
    print("NO-GT METRIC PASS-BY COMPLETE")
    print("=" * 96)
    print(
        "[preview] ",
        output_dir
        / "forward_passby_nogt_metric_preview.mp4",
    )
    print(
        "[diagnostic] ",
        output_dir
        / "forward_passby_nogt_metric_side_by_side.mp4",
    )
    print(
        "[rows] ",
        csv_path,
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
