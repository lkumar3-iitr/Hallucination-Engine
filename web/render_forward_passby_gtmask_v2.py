#!/usr/bin/env python3
"""
render_forward_passby_gtmask_v2.py

Stable forward-pass visualization.

View selection:
    CURRENT production HE view-matrix selector.

GT mask:
    used ONLY to fit x/y/width/height for the already-selected sprite.

Default final alpha:
    exact CARLA GT target mask.

The default exact-alpha mode intentionally asks the strongest qualitative
question:

    "If box/silhouette geometry were perfect, does the current production
     4320-selected appearance look right through the pass-by?"

This is not a benchmark renderer.
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
        "--continuity-weight",
        type=float,
        default=0.015,
    )

    p.add_argument(
        "--natural-alpha",
        action="store_true",
        help=(
            "Use the selected sprite's own fitted alpha. "
            "Default forces exact CARLA GT alpha."
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
                / "forward_passby_gtmask_preview.mp4"
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
                / "forward_passby_gtmask_side_by_side.mp4"
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

    exact_alpha = (
        not bool(
            args.natural_alpha
        )
    )

    rows = []
    previous_fit = None

    print("=" * 96)
    print(
        "FORWARD PASS-BY V3: "
        "CURRENT PRODUCTION SELECTOR + GT-MASK PLACEMENT"
    )
    print("=" * 96)
    print("[renderer]   ", renderer_path)
    print("[CARLA]      ", carla_dir)
    print("[bank]       ", bank_dir)
    print("[frames]     ", f"{wanted[0]}..{wanted[-1]}")
    print("[exact alpha]", exact_alpha)
    print(
        "[selection]  ",
        "production select_view_matrix_sprite; GT mask NOT used",
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

            fitted_mask = (
                np.zeros_like(
                    gt_mask
                )
            )

            selection = None
            fit = None
            previous_fit = None

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

            # ---------------------------------------------
            # 1. Select sprite from CURRENT production logic
            #    WITHOUT GT.
            # ---------------------------------------------
            (
                selector_state,
                selection,
            ) = (
                oracle
                .select_sprite_with_production_logic(
                    renderer=renderer,
                    actor_tf=actor_tf,
                    camera_tf=camera_tf,
                    sprite_bank=sprite_bank,
                    view_matrix=view_matrix,
                    viewpoint_lateral_sign=float(
                        args.viewpoint_lateral_sign
                    ),
                )
            )

            if not selection.get(
                "exists",
                False,
            ):
                raise RuntimeError(
                    "Production selector returned missing sprite "
                    f"at frame {scenario_frame}: {selection}"
                )

            selected_rgba = (
                sprite_cache
                .load_rgba(
                    selection[
                        "sprite_path"
                    ]
                )
            )

            # ---------------------------------------------
            # 2. Fit ONLY the selected sprite's 2-D box.
            # ---------------------------------------------
            fit = (
                oracle
                .fit_fixed_sprite_to_gt_mask(
                    selected_rgba=selected_rgba,
                    target_mask=gt_mask,
                    previous_fit=previous_fit,
                    continuity_weight=float(
                        args.continuity_weight
                    ),
                )
            )

            if not fit.get(
                "success",
                False,
            ):
                raise RuntimeError(
                    "GT-mask box fit failed at frame "
                    f"{scenario_frame}: {fit}"
                )

            previous_fit = {
                "center_x":
                    float(
                        fit[
                            "center_x"
                        ]
                    ),

                "bottom_y":
                    float(
                        fit[
                            "bottom_y"
                        ]
                    ),

                "target_width":
                    float(
                        fit[
                            "target_width"
                        ]
                    ),

                "target_height":
                    float(
                        fit[
                            "target_height"
                        ]
                    ),
            }

            fitted_mask = (
                fit[
                    "fitted_mask"
                ]
            )

            fitted_rgba = (
                fit[
                    "warped_rgba"
                ]
            )

            if exact_alpha:
                final_rgba = (
                    oracle
                    .force_exact_gt_alpha(
                        fitted_rgba=fitted_rgba,
                        target_mask=gt_mask,
                    )
                )

            else:
                final_rgba = (
                    fitted_rgba
                )

            # Remove the physical Tesla approximately.
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

            # Current HE SpriteCache returns RGBA and HE rendering is RGB.
            background_rgb = (
                cv2.cvtColor(
                    background_bgr,
                    cv2.COLOR_BGR2RGB,
                )
            )

            preview_rgb = (
                oracle
                .alpha_composite_rgb(
                    background_rgb=background_rgb,
                    foreground_rgba=final_rgba,
                )
            )

            preview_bgr = (
                cv2.cvtColor(
                    preview_rgb,
                    cv2.COLOR_RGB2BGR,
                )
            )

        if selection is None:
            selected_angle = None
            selected_distance = None
            selected_elevation = None
            query_angle = None
            query_distance = None
            query_elevation = None

        else:
            selected_angle = (
                selection[
                    "selected_angle"
                ]
            )

            selected_distance = (
                selection[
                    "selected_distance_m"
                ]
            )

            selected_elevation = (
                selection[
                    "selected_elevation_deg"
                ]
            )

            query_angle = (
                selection[
                    "relative_angle_deg"
                ]
            )

            query_distance = (
                selection[
                    "query_distance_m"
                ]
            )

            query_elevation = (
                selection[
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
                f"fitIoU={None if fit is None else fit.get('fit_iou')} "
                f"fitDice={None if fit is None else fit.get('fit_dice')} "
                f"border={None if fit is None else fit.get('border')} "
                f"exactAlpha={exact_alpha}"
            ),
        ]

        diagnostic = (
            make_diagnostic(
                carla_bgr=carla_bgr,
                preview_bgr=preview_bgr,
                gt_mask=gt_mask,
                fitted_mask=fitted_mask,
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
                (
                    None
                    if fit is None
                    else fit.get(
                        "fit_iou"
                    )
                ),

            "fit_dice":
                (
                    None
                    if fit is None
                    else fit.get(
                        "fit_dice"
                    )
                ),

            "fit_center_x":
                (
                    None
                    if fit is None
                    else fit.get(
                        "center_x"
                    )
                ),

            "fit_bottom_y":
                (
                    None
                    if fit is None
                    else fit.get(
                        "bottom_y"
                    )
                ),

            "fit_width":
                (
                    None
                    if fit is None
                    else fit.get(
                        "target_width"
                    )
                ),

            "fit_height":
                (
                    None
                    if fit is None
                    else fit.get(
                        "target_height"
                    )
                ),

            "border":
                (
                    None
                    if fit is None
                    else json.dumps(
                        fit.get(
                            "border"
                        )
                    )
                ),
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
            f"fitIoU={csv_row['fit_iou']}"
        )

    cap.release()
    preview_writer.release()
    diagnostic_writer.release()

    csv_path = (
        output_dir
        / "forward_passby_gtmask_rows.csv"
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
    print("GT-MASK PASS-BY COMPLETE")
    print("=" * 96)
    print(
        "[preview] ",
        output_dir
        / "forward_passby_gtmask_preview.mp4",
    )
    print(
        "[diagnostic] ",
        output_dir
        / "forward_passby_gtmask_side_by_side.mp4",
    )
    print(
        "[rows] ",
        csv_path,
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
