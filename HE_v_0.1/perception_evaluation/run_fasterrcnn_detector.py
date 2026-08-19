import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)


def parse_device(value):
    value = str(value)

    if value.isdigit():
        return torch.device(
            f"cuda:{value}"
        )

    return torch.device(value)


def draw_detection(
    frame,
    detection,
):
    x1 = int(round(detection["x1"]))
    y1 = int(round(detection["y1"]))
    x2 = int(round(detection["x2"]))
    y2 = int(round(detection["y2"]))

    label = (
        f'{detection["class_name"]} '
        f'{detection["confidence"]:.2f}'
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--annotated-video",
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = parse_device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    print(
        "[INFO] device:",
        device,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    weights = (
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )

    categories = (
        weights.meta["categories"]
    )

    print(
        "[INFO] loading Faster R-CNN ResNet50-FPN V2..."
    )

    model = (
        fasterrcnn_resnet50_fpn_v2(
            weights=weights
        )
    )

    model.eval()
    model.to(device)

    print(
        "[INFO] model ready"
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.video
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.video}"
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

    print(
        "[INFO] video:",
        args.video,
    )

    print(
        "[INFO] frames:",
        total_frames,
    )

    print(
        "[INFO] size:",
        width,
        "x",
        height,
    )

    print(
        "[INFO] fps:",
        fps,
    )

    # --------------------------------------------------------
    # Output JSONL
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_file = open(
        output_path,
        "w",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Annotated video
    # --------------------------------------------------------

    writer = None

    if args.annotated_video:

        annotated_path = Path(
            args.annotated_video
        )

        annotated_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not open annotated video writer."
            )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    frame_idx = 0

    total_inference_ms = 0.0

    try:

        while True:

            ret, frame_bgr = (
                cap.read()
            )

            if not ret:
                break

            # OpenCV BGR -> RGB
            frame_rgb = (
                cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )
            )

            # H,W,C uint8 -> C,H,W float32 [0,1]
            image_tensor = (
                torch
                .from_numpy(
                    frame_rgb
                )
                .permute(
                    2,
                    0,
                    1,
                )
                .float()
                / 255.0
            )

            image_tensor = (
                image_tensor.to(
                    device
                )
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            with torch.inference_mode():

                prediction = (
                    model(
                        [
                            image_tensor
                        ]
                    )[0]
                )

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_ms = (
                (
                    time.perf_counter()
                    - t0
                )
                * 1000.0
            )

            total_inference_ms += (
                inference_ms
            )

            boxes = (
                prediction["boxes"]
                .detach()
                .cpu()
            )

            labels = (
                prediction["labels"]
                .detach()
                .cpu()
            )

            scores = (
                prediction["scores"]
                .detach()
                .cpu()
            )

            detections = []

            for (
                box,
                label_id,
                score,
            ) in zip(
                boxes,
                labels,
                scores,
            ):

                confidence = float(
                    score.item()
                )

                if confidence < args.conf:
                    continue

                class_id = int(
                    label_id.item()
                )

                if (
                    class_id < 0
                    or class_id >= len(categories)
                ):
                    class_name = (
                        str(class_id)
                    )
                else:
                    class_name = (
                        categories[
                            class_id
                        ]
                    )

                x1 = float(
                    box[0].item()
                )

                y1 = float(
                    box[1].item()
                )

                x2 = float(
                    box[2].item()
                )

                y2 = float(
                    box[3].item()
                )

                detection = {
                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        confidence,

                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,

                    "cx":
                        (x1 + x2) / 2.0,

                    "cy":
                        (y1 + y2) / 2.0,

                    "width":
                        x2 - x1,

                    "height":
                        y2 - y1,
                }

                detections.append(
                    detection
                )

            record = {
                "frame_idx":
                    frame_idx,

                "model":
                    "fasterrcnn_resnet50_fpn_v2",

                "device":
                    str(device),

                "confidence_threshold":
                    float(args.conf),

                "inference_ms":
                    float(inference_ms),

                "detections":
                    detections,
            }

            out_file.write(
                json.dumps(
                    record
                )
                + "\n"
            )

            if writer is not None:

                annotated = (
                    frame_bgr.copy()
                )

                for detection in detections:
                    draw_detection(
                        annotated,
                        detection,
                    )

                writer.write(
                    annotated
                )

            if (
                frame_idx % 30
                == 0
            ):
                print(
                    f"[INFO] frame "
                    f"{frame_idx}/"
                    f"{total_frames - 1} "
                    f"detections="
                    f"{len(detections)} "
                    f"time="
                    f"{inference_ms:.1f} ms"
                )

            frame_idx += 1

    finally:

        cap.release()

        out_file.close()

        if writer is not None:
            writer.release()

    mean_ms = (
        total_inference_ms
        / frame_idx
        if frame_idx
        else 0.0
    )

    print()
    print("=" * 72)
    print("Faster R-CNN inference complete")
    print("=" * 72)

    print(
        "frames:",
        frame_idx,
    )

    print(
        "mean inference ms:",
        f"{mean_ms:.3f}",
    )

    if mean_ms > 0:
        print(
            "model inference FPS:",
            f"{1000.0 / mean_ms:.2f}",
        )

    print(
        "output:",
        output_path,
    )

    if args.annotated_video:
        print(
            "annotated:",
            args.annotated_video,
        )

    print("=" * 72)


if __name__ == "__main__":
    main()