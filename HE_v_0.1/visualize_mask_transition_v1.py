import cv2
import numpy as np
from pathlib import Path


REAL_DIR = Path(
    r"recordings\he_pairs"
    r"\pair_static_straight_tesla_depth55_001"
    r"\real_instance_masks"
)

HE_DIR = Path(
    r"he_outputs"
    r"\pair_static_straight_tesla_depth55_001_prod4320"
    r"\masks"
)

OUT_DIR = Path(
    r"he_outputs"
    r"\pair_static_straight_tesla_depth55_001_prod4320"
    r"\mask_transition_debug"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


for frame_idx in range(274, 281):

    real_path = (
        REAL_DIR
        / f"frame_{frame_idx:06d}_mask.png"
    )

    he_path = (
        HE_DIR
        / f"frame_{frame_idx:06d}_mask.png"
    )

    real = cv2.imread(
        str(real_path),
        cv2.IMREAD_GRAYSCALE,
    )

    he = cv2.imread(
        str(he_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if real is None or he is None:
        raise RuntimeError(
            f"Missing mask at frame {frame_idx}"
        )

    real = real > 127
    he = he > 10

    h, w = real.shape

    vis = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    # CARLA only = red
    vis[
        real & ~he
    ] = (
        0,
        0,
        255,
    )

    # HE only = blue
    vis[
        he & ~real
    ] = (
        255,
        0,
        0,
    )

    # Intersection = white
    vis[
        real & he
    ] = (
        255,
        255,
        255,
    )

    ys, xs = np.where(
        real | he
    )

    if len(xs) > 0:

        margin = 30

        x1 = max(
            0,
            int(xs.min()) - margin,
        )

        y1 = max(
            0,
            int(ys.min()) - margin,
        )

        x2 = min(
            w,
            int(xs.max()) + margin + 1,
        )

        y2 = min(
            h,
            int(ys.max()) + margin + 1,
        )

        crop = vis[
            y1:y2,
            x1:x2,
        ]

        # Enlarge just for visual inspection.
        crop = cv2.resize(
            crop,
            None,
            fx=4.0,
            fy=4.0,
            interpolation=cv2.INTER_NEAREST,
        )

    else:
        crop = vis

    cv2.putText(
        crop,
        f"frame {frame_idx}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path = (
        OUT_DIR
        / f"frame_{frame_idx:06d}_comparison.png"
    )

    cv2.imwrite(
        str(output_path),
        crop,
    )

    print(
        "saved:",
        output_path,
    )