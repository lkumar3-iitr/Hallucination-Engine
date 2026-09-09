"""Create a synchronized labeled comparison from CARLA and HE mosaic videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla", type=Path, required=True)
    parser.add_argument("--he", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    carla = cv2.VideoCapture(str(args.carla))
    he = cv2.VideoCapture(str(args.he))
    if not carla.isOpened() or not he.isOpened():
        raise RuntimeError("Could not open both input videos")

    width = int(carla.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(carla.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(carla.get(cv2.CAP_PROP_FPS))
    if (
        width != int(he.get(cv2.CAP_PROP_FRAME_WIDTH))
        or height != int(he.get(cv2.CAP_PROP_FRAME_HEIGHT))
        or abs(fps - float(he.get(cv2.CAP_PROP_FPS))) > 1e-6
    ):
        raise RuntimeError("Input videos do not share dimensions and FPS")

    header = 34
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, 2 * (height + header)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.output}")

    frame_index = 0
    while True:
        ok_carla, carla_frame = carla.read()
        ok_he, he_frame = he.read()
        if ok_carla != ok_he:
            raise RuntimeError(f"Input frame-count mismatch at frame {frame_index}")
        if not ok_carla:
            break

        canvas = np.zeros((2 * (height + header), width, 3), np.uint8)
        canvas[header:header + height] = carla_frame
        he_top = height + 2 * header
        canvas[he_top:he_top + height] = he_frame
        cv2.putText(canvas, "CARLA physical - native camera view(s)", (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 255, 80), 1, cv2.LINE_AA)
        cv2.putText(canvas, "HE he_sprite_renderer_v1 - native camera view(s)", (12, height + header + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 180, 255), 1, cv2.LINE_AA)
        stamp = f"frame {frame_index:04d}   t={frame_index / fps:05.2f}s"
        (text_width, _), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        cv2.putText(canvas, stamp, (width - text_width - 12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        writer.write(canvas)
        frame_index += 1

    carla.release()
    he.release()
    writer.release()
    print(f"frames={frame_index} fps={fps} size={width}x{2 * (height + header)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
