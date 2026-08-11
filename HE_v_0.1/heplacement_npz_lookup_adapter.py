import math
from pathlib import Path

import numpy as np


def normalize_yaw_0_360(yaw_deg):
    yaw = float(yaw_deg) % 360.0
    if yaw < 0:
        yaw += 360.0
    return yaw


def find_interval(values, x):
    """
    Return i0, i1, weight for sorted 1D axis.
    Clamps outside range.
    """
    values = np.asarray(values, dtype=np.float32)
    x = float(x)

    if x <= float(values[0]):
        return 0, 0, 0.0

    if x >= float(values[-1]):
        last = len(values) - 1
        return last, last, 0.0

    i1 = int(np.searchsorted(values, x, side="right"))
    i0 = i1 - 1

    v0 = float(values[i0])
    v1 = float(values[i1])

    if abs(v1 - v0) < 1e-9:
        return i0, i1, 0.0

    w = (x - v0) / (v1 - v0)
    return i0, i1, float(w)


class HEPlacementNPZLookupAdapter:
    """
    Runtime placement adapter for HEPlacementModel v2 NPZ lookup table.

    Expected NPZ layout:
      arrays shape [rel_z_index, rel_x_index, yaw_index]
      rel_x_values
      rel_z_values
      yaw_values
      visible
      center_x
      bottom_y
      box_width
      box_height
    """

    def __init__(self, npz_path):
        self.npz_path = str(npz_path)

        if not Path(self.npz_path).exists():
            raise FileNotFoundError(f"NPZ lookup not found: {self.npz_path}")

        data = np.load(self.npz_path)

        self.rel_x_values = data["rel_x_values"].astype(np.float32)
        self.rel_z_values = data["rel_z_values"].astype(np.float32)
        self.yaw_values = data["yaw_values"].astype(np.float32)

        self.visible = data["visible"].astype(np.float32)
        self.center_x = data["center_x"].astype(np.float32)
        self.bottom_y = data["bottom_y"].astype(np.float32)
        self.box_width = data["box_width"].astype(np.float32)
        self.box_height = data["box_height"].astype(np.float32)

        self.x_min_axis = float(self.rel_x_values[0])
        self.x_max_axis = float(self.rel_x_values[-1])
        self.z_min_axis = float(self.rel_z_values[0])
        self.z_max_axis = float(self.rel_z_values[-1])

        print("[HEPlacementNPZ] Loaded:", self.npz_path)
        print(
            "[HEPlacementNPZ] x:",
            self.x_min_axis,
            "to",
            self.x_max_axis,
            "count",
            len(self.rel_x_values),
        )
        print(
            "[HEPlacementNPZ] z:",
            self.z_min_axis,
            "to",
            self.z_max_axis,
            "count",
            len(self.rel_z_values),
        )
        print(
            "[HEPlacementNPZ] yaw:",
            float(self.yaw_values[0]),
            "to",
            float(self.yaw_values[-1]),
            "count",
            len(self.yaw_values),
        )

    def _sample_yaw_indices(self, yaw_deg):
        """
        Circular yaw interpolation for 0..359 degree table.
        """
        yaw = normalize_yaw_0_360(yaw_deg)

        # For integer yaw table 0..359, interpolate between floor and next.
        y0 = int(math.floor(yaw)) % 360
        y1 = (y0 + 1) % 360
        wy = yaw - math.floor(yaw)

        return y0, y1, float(wy)

    def _trilinear(self, arr, rel_x, rel_z, yaw_deg):
        iz0, iz1, wz = find_interval(self.rel_z_values, rel_z)
        ix0, ix1, wx = find_interval(self.rel_x_values, rel_x)
        iy0, iy1, wy = self._sample_yaw_indices(yaw_deg)

        # Corners: z, x, yaw.
        c000 = float(arr[iz0, ix0, iy0])
        c001 = float(arr[iz0, ix0, iy1])
        c010 = float(arr[iz0, ix1, iy0])
        c011 = float(arr[iz0, ix1, iy1])
        c100 = float(arr[iz1, ix0, iy0])
        c101 = float(arr[iz1, ix0, iy1])
        c110 = float(arr[iz1, ix1, iy0])
        c111 = float(arr[iz1, ix1, iy1])

        c00 = c000 * (1.0 - wy) + c001 * wy
        c01 = c010 * (1.0 - wy) + c011 * wy
        c10 = c100 * (1.0 - wy) + c101 * wy
        c11 = c110 * (1.0 - wy) + c111 * wy

        c0 = c00 * (1.0 - wx) + c01 * wx
        c1 = c10 * (1.0 - wx) + c11 * wx

        return c0 * (1.0 - wz) + c1 * wz

    def predict_box(self, state, image_width=1280, image_height=720):
        """
        state convention:
          x_m = lateral, positive right
          z_m = forward depth
          yaw_deg = relative yaw / camera-relative yaw
        """
        rel_x = float(state.get("x_m", 0.0))
        rel_z = float(state.get("z_m", 0.0))
        yaw_deg = normalize_yaw_0_360(float(state.get("yaw_deg", 0.0)))

        clamped_x = min(max(rel_x, self.x_min_axis), self.x_max_axis)
        clamped_z = min(max(rel_z, self.z_min_axis), self.z_max_axis)

        clamped = (
            abs(clamped_x - rel_x) > 1e-6
            or abs(clamped_z - rel_z) > 1e-6
        )

        visible_prob = self._trilinear(self.visible, clamped_x, clamped_z, yaw_deg)

        cx = self._trilinear(self.center_x, clamped_x, clamped_z, yaw_deg)
        bottom_y = self._trilinear(self.bottom_y, clamped_x, clamped_z, yaw_deg)
        box_w = self._trilinear(self.box_width, clamped_x, clamped_z, yaw_deg)
        box_h = self._trilinear(self.box_height, clamped_x, clamped_z, yaw_deg)

        visible = (
            visible_prob >= 0.5
            and box_w >= 1.0
            and box_h >= 1.0
            and rel_z >= 0.0
        )

        if not visible:
            return {
                "visible": False,
                "visible_prob": float(visible_prob),
                "reason": "npz_lookup_not_visible",
                "source": "heplacement_v2_npz_lookup",
                "z_m": float(rel_z),
                "x_m": float(rel_x),
                "yaw_deg": float(yaw_deg),
                "clamped": bool(clamped),
                "clamped_x_m": float(clamped_x),
                "clamped_z_m": float(clamped_z),
            }

        x1 = float(cx - box_w / 2.0)
        x2 = float(cx + box_w / 2.0)
        y2 = float(bottom_y)
        y1 = float(bottom_y - box_h)

        # Clip to image boundaries for paste safety.
        x1_clip = max(0.0, min(float(image_width - 1), x1))
        x2_clip = max(0.0, min(float(image_width - 1), x2))
        y1_clip = max(0.0, min(float(image_height - 1), y1))
        y2_clip = max(0.0, min(float(image_height - 1), y2))

        if x2_clip <= x1_clip or y2_clip <= y1_clip:
            return {
                "visible": False,
                "visible_prob": float(visible_prob),
                "reason": "npz_lookup_clipped_invalid",
                "source": "heplacement_v2_npz_lookup",
                "z_m": float(rel_z),
                "x_m": float(rel_x),
                "yaw_deg": float(yaw_deg),
                "clamped": bool(clamped),
                "clamped_x_m": float(clamped_x),
                "clamped_z_m": float(clamped_z),
            }

        return {
            "visible": True,
            "visible_prob": float(visible_prob),
            "cx": float(cx),
            "bottom_y": float(bottom_y),
            "x1": float(x1_clip),
            "y1": float(y1_clip),
            "x2": float(x2_clip),
            "y2": float(y2_clip),
            "box_width": float(x2_clip - x1_clip),
            "box_height": float(y2_clip - y1_clip),
            "unclipped_x1": float(x1),
            "unclipped_y1": float(y1),
            "unclipped_x2": float(x2),
            "unclipped_y2": float(y2),
            "z_m": float(rel_z),
            "x_m": float(rel_x),
            "yaw_deg": float(yaw_deg),
            "clamped": bool(clamped),
            "clamped_x_m": float(clamped_x),
            "clamped_z_m": float(clamped_z),
            "source": "heplacement_v2_npz_lookup",
        }