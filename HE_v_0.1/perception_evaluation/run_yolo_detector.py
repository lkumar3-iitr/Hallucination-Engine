#!/usr/bin/env python3
"""
run_yolo_detector.py

Run the same YOLO detector on a video and save per-frame detections
in a backend-independent JSONL format.

Designed for CARLA-vs-HE perception equivalence experiments.

Example:

python perception_evaluation\run_yolo_detector.py ^
  --video recordings\he_pairs\v2_multi_actor_001\real_adversary.mp4 ^
  --model yolo11s.pt ^
  --device 0 ^
  --conf 0.10 ^
  --output perception_evaluation\outputs\v2_multi_actor_001_carla_yolo11s.jsonl ^
  --annotated-video perception_evaluation\outputs\v2_multi_actor_001_carla_yolo11s.mp4
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Ultralytics detection model/checkpoint.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device, e.g. 0 or cpu.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU threshold.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--annotated-video",
        default=None,
        help="Optional annotated output MP4.",
    )

    return parser.parse_args()


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def main():
    args = parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_path = ensure_parent(
        args.output
    )

    annotated_path = None

    if args.annotated_video:
        annotated_path = ensure_parent(
            args.annotated_video
        )

    print("=" * 72)
    print("HE Perception Evaluation - YOLO Detector")
    print("=" * 72)

    print("video :", video_path)
    print("model :", args.model)
    print("device:", args.device)
    print("conf  :", args.conf)
    print("iou   :", args.iou)
    print("imgsz :", args.imgsz)

    print()
    print("torch :", torch.__version__)
    print("cuda  :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "gpu   :",
            torch.cuda.get_device_name(0)
        )

    print("=" * 72)

    # ------------------------------------------------------------
    # Load YOLO
    # ------------------------------------------------------------

    model = YOLO(
        args.model
    )

    # ------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        f"video size : {width}x{height}"
    )

    print(
        f"video fps  : {fps:.3f}"
    )

    print(
        f"frames     : {total_frames}"
    )

    # ------------------------------------------------------------
    # Optional annotated video writer
    # ------------------------------------------------------------

    writer = None

    if annotated_path is not None:

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(annotated_path),
            fourcc,
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not create annotated video: "
                f"{annotated_path}"
            )

    # ------------------------------------------------------------
    # Detection loop
    # ------------------------------------------------------------

    frame_idx = 0

    total_detections = 0
    total_inference_ms = 0.0

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:

        while True:

            ret, frame_bgr = cap.read()

            if not ret:
                break

            start = time.perf_counter()

            results = model.predict(
                source=frame_bgr,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )

            elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            total_inference_ms += elapsed_ms

            if len(results) != 1:
                raise RuntimeError(
                    "Expected exactly one YOLO "
                    "result for one input frame."
                )

            result = results[0]

            detections = []

            boxes = result.boxes

            if boxes is not None:

                xyxy = (
                    boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                confidences = (
                    boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                classes = (
                    boxes.cls
                    .detach()
                    .cpu()
                    .numpy()
                )

                for box_coords, conf, cls_id in zip(
                    xyxy,
                    confidences,
                    classes,
                ):

                    x1 = float(
                        box_coords[0]
                    )

                    y1 = float(
                        box_coords[1]
                    )

                    x2 = float(
                        box_coords[2]
                    )

                    y2 = float(
                        box_coords[3]
                    )

                    cls_id = int(
                        cls_id
                    )

                    class_name = str(
                        result.names[
                            cls_id
                        ]
                    )

                    box_width = (
                        x2 - x1
                    )

                    box_height = (
                        y2 - y1
                    )

                    center_x = (
                        x1 + x2
                    ) / 2.0

                    center_y = (
                        y1 + y2
                    ) / 2.0

                    detection = {
                        "class_id": cls_id,
                        "class_name": class_name,

                        "confidence": float(
                            conf
                        ),

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,

                        "cx": center_x,
                        "cy": center_y,

                        "width": box_width,
                        "height": box_height,
                    }

                    detections.append(
                        detection
                    )

            total_detections += len(
                detections
            )

            record = {
                "frame_idx": int(
                    frame_idx
                ),

                "t_s": (
                    float(frame_idx) / fps
                    if fps > 0.0
                    else None
                ),

                "image_width": int(
                    width
                ),

                "image_height": int(
                    height
                ),

                "model": args.model,

                "device": str(
                    args.device
                ),

                "confidence_threshold": float(
                    args.conf
                ),

                "nms_iou_threshold": float(
                    args.iou
                ),

                "inference_ms": float(
                    elapsed_ms
                ),

                "detections": detections,
            }

            fout.write(
                json.dumps(
                    record
                )
                + "\n"
            )

            # --------------------------------------------------------
            # Optional annotated output
            # --------------------------------------------------------

            if writer is not None:

                annotated_bgr = (
                    result.plot()
                )

                writer.write(
                    annotated_bgr
                )

            if (
                frame_idx % 25 == 0
                or frame_idx
                == total_frames - 1
            ):

                print(
                    f"frame {frame_idx:4d}"
                    f" / {total_frames - 1:4d}"
                    f" | detections={len(detections):2d}"
                    f" | {elapsed_ms:7.2f} ms"
                )

            frame_idx += 1

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    cap.release()

    if writer is not None:
        writer.release()

    avg_inference_ms = (
        total_inference_ms
        / max(
            frame_idx,
            1
        )
    )

    detector_fps = (
        1000.0
        / avg_inference_ms
        if avg_inference_ms > 0.0
        else 0.0
    )

    print()
    print("=" * 72)
    print("Detection complete")
    print("=" * 72)

    print(
        "frames processed :",
        frame_idx
    )

    print(
        "total detections :",
        total_detections
    )

    print(
        "avg inference ms :",
        f"{avg_inference_ms:.3f}"
    )

    print(
        "approx detector FPS:",
        f"{detector_fps:.2f}"
    )

    print(
        "JSONL:",
        output_path
    )

    if annotated_path is not None:
        print(
            "video:",
            annotated_path
        )

    print("=" * 72)


if __name__ == "__main__":
    main()