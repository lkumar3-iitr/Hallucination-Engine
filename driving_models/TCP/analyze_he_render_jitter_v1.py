import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT
    / "driving_models"
    / "TCP"
    / "outputs"
    / "tcp_he_pair_v1"
    / "tcp_lead_brake_001"
    / "tcp_lead_brake_001_he.csv"
)

START_FRAME = 60
END_FRAME = 180


def f(row, key):
    value = str(row.get(key, "")).strip()

    if not value:
        return float("nan")

    try:
        return float(value)
    except ValueError:
        return float("nan")


def circular_diff_deg(a, b):
    return (
        (float(a) - float(b) + 180.0)
        % 360.0
        - 180.0
    )


with CSV_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as fp:
    rows = list(
        csv.DictReader(fp)
    )


rows = [
    row
    for row in rows
    if (
        START_FRAME
        <= int(row["probe_idx"])
        <= END_FRAME
    )
]


keys = [
    "he_cx",
    "he_bottom_y",
    "he_box_width",
    "he_box_height",
    "he_sprite_width",
    "he_sprite_height",
]


arrays = {}

for key in keys:
    arrays[key] = np.array(
        [
            f(row, key)
            for row in rows
        ],
        dtype=np.float64,
    )


# ------------------------------------------------------------
# Derived paste position actually used by compositor
# ------------------------------------------------------------

cx = arrays["he_cx"]
bottom_y = arrays["he_bottom_y"]

sprite_w = arrays[
    "he_sprite_width"
]

sprite_h = arrays[
    "he_sprite_height"
]


paste_x = np.round(
    cx - sprite_w / 2.0
)

paste_y = np.round(
    bottom_y - sprite_h
)


arrays["derived_paste_x"] = paste_x
arrays["derived_paste_y"] = paste_y


print()
print("=" * 88)
print("HE RENDER JITTER DIAGNOSTIC")
print("=" * 88)

print(
    "CSV:",
    CSV_PATH,
)

print(
    "frame range:",
    START_FRAME,
    "to",
    END_FRAME,
)

print()


# ------------------------------------------------------------
# First and second differences
# ------------------------------------------------------------

print("FRAME-TO-FRAME MOTION")
print("-" * 88)

for key, values in arrays.items():

    valid = np.isfinite(
        values
    )

    values = values[
        valid
    ]

    if len(values) < 3:
        continue

    d1 = np.diff(
        values
    )

    d2 = np.diff(
        values,
        n=2,
    )

    print(
        f"{key:22s} "
        f"| mean |d1|={np.mean(np.abs(d1)):7.3f} "
        f"| p95 |d1|={np.percentile(np.abs(d1),95):7.3f} "
        f"| max |d1|={np.max(np.abs(d1)):7.3f} "
        f"| p95 |d2|={np.percentile(np.abs(d2),95):7.3f} "
        f"| max |d2|={np.max(np.abs(d2)):7.3f}"
    )


# ------------------------------------------------------------
# Sprite-angle switching
# ------------------------------------------------------------

view_angles = np.array(
    [
        f(
            row,
            "he_viewpoint_angle_deg",
        )
        for row in rows
    ],
    dtype=np.float64,
)

selected_angles = np.array(
    [
        f(
            row,
            "he_selected_angle",
        )
        for row in rows
    ],
    dtype=np.float64,
)


view_deltas = []

selected_deltas = []

angle_switches = 0


for i in range(
    1,
    len(rows),
):

    if (
        np.isfinite(
            view_angles[i]
        )
        and
        np.isfinite(
            view_angles[i - 1]
        )
    ):
        view_deltas.append(
            circular_diff_deg(
                view_angles[i],
                view_angles[i - 1],
            )
        )

    if (
        np.isfinite(
            selected_angles[i]
        )
        and
        np.isfinite(
            selected_angles[i - 1]
        )
    ):

        delta = (
            circular_diff_deg(
                selected_angles[i],
                selected_angles[i - 1],
            )
        )

        selected_deltas.append(
            delta
        )

        if abs(delta) > 0.01:
            angle_switches += 1


print()
print("SPRITE ANGLE")
print("-" * 88)

if view_deltas:

    view_deltas = np.array(
        view_deltas
    )

    print(
        "viewpoint mean |delta| :",
        f"{np.mean(np.abs(view_deltas)):.4f} deg/frame",
    )

    print(
        "viewpoint max  |delta| :",
        f"{np.max(np.abs(view_deltas)):.4f} deg/frame",
    )


if selected_deltas:

    selected_deltas = np.array(
        selected_deltas
    )

    print(
        "selected angle changes :",
        angle_switches,
    )

    print(
        "selected max |delta|   :",
        f"{np.max(np.abs(selected_deltas)):.2f} deg",
    )


# ------------------------------------------------------------
# Largest visual jumps
# ------------------------------------------------------------

scores = []


for i in range(
    2,
    len(rows),
):

    # Second difference is useful because smooth ego/actor
    # motion naturally creates first differences.
    #
    # Sudden direction reversal / oscillation creates a
    # comparatively large second difference.

    d2_cx = (
        cx[i]
        - 2.0 * cx[i - 1]
        + cx[i - 2]
    )

    d2_by = (
        bottom_y[i]
        - 2.0 * bottom_y[i - 1]
        + bottom_y[i - 2]
    )

    d2_w = (
        sprite_w[i]
        - 2.0 * sprite_w[i - 1]
        + sprite_w[i - 2]
    )

    d2_h = (
        sprite_h[i]
        - 2.0 * sprite_h[i - 1]
        + sprite_h[i - 2]
    )

    score = (
        abs(d2_cx)
        +
        abs(d2_by)
        +
        0.5 * abs(d2_w)
        +
        0.5 * abs(d2_h)
    )

    scores.append(
        (
            score,
            i,
        )
    )


scores.sort(
    reverse=True
)


print()
print("TOP POSSIBLE JITTER FRAMES")
print("-" * 88)

print(
    "frame   time    cx      bottom   boxW   sprW  "
    "pasteX pasteY viewAng selAng"
)


for score, i in scores[:20]:

    row = rows[i]

    print(
        f"{int(row['probe_idx']):5d} "
        f"{f(row,'t_s'):6.2f} "
        f"{cx[i]:7.2f} "
        f"{bottom_y[i]:7.2f} "
        f"{arrays['he_box_width'][i]:6.2f} "
        f"{sprite_w[i]:6.1f} "
        f"{paste_x[i]:6.0f} "
        f"{paste_y[i]:6.0f} "
        f"{view_angles[i]:7.2f} "
        f"{selected_angles[i]:6.0f}"
    )


# ------------------------------------------------------------
# Integer paste oscillations
# ------------------------------------------------------------

paste_x_reverse = 0
paste_y_reverse = 0


for i in range(
    2,
    len(rows),
):

    dx0 = (
        paste_x[i - 1]
        - paste_x[i - 2]
    )

    dx1 = (
        paste_x[i]
        - paste_x[i - 1]
    )

    dy0 = (
        paste_y[i - 1]
        - paste_y[i - 2]
    )

    dy1 = (
        paste_y[i]
        - paste_y[i - 1]
    )

    # Example:
    # 450 -> 451 -> 450

    if (
        dx0 != 0
        and
        dx1 != 0
        and
        np.sign(dx0)
        != np.sign(dx1)
    ):
        paste_x_reverse += 1

    if (
        dy0 != 0
        and
        dy1 != 0
        and
        np.sign(dy0)
        != np.sign(dy1)
    ):
        paste_y_reverse += 1


print()
print("INTEGER PASTE OSCILLATION")
print("-" * 88)

print(
    "horizontal reversals:",
    paste_x_reverse,
)

print(
    "vertical reversals  :",
    paste_y_reverse,
)

print("=" * 88)