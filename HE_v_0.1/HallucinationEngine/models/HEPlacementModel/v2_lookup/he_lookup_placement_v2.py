import argparse
import json
import math
from pathlib import Path

import numpy as np


def normalize_yaw_deg(yaw):
    """
    Normalize yaw to [-180, 180].
    """
    yaw = float(yaw)
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return yaw


class HELookupPlacementV2:
    """
    Deterministic HEPlacementModel v2.

    Runtime input:
        rel_x, rel_z, rel_yaw

    Output:
        center_x, bottom_y, box_width, box_height
        plus compositor rectangle x, y, w, h

    Current version:
        nearest-neighbor lookup

    Next version:
        trilinear interpolation
    """

    def __init__(
        self,
        labels_path,
        metadata_path=None,
        safe_distance_margin_m=5.0,
        default_mode="clamp",
    ):
        """
        default_mode:
            "clamp"      : clamp rel_z to safe_min_rel_z
            "edge"       : allow close-range lookup
            "invisible"  : return visible=0 if rel_z < safe_min_rel_z
        """
        self.labels_path = Path(labels_path)

        if metadata_path is None:
            metadata_path = self.labels_path.parent / "metadata.json"

        self.metadata_path = Path(metadata_path)

        self.safe_distance_margin_m = float(safe_distance_margin_m)
        self.default_mode = default_mode

        self.metadata = None
        self.rel_x_values = None
        self.rel_z_values = None
        self.yaw_values = None

        self.rel_x_to_idx = {}
        self.rel_z_to_idx = {}
        self.yaw_to_idx = {}

        self.center_x_grid = None
        self.bottom_y_grid = None
        self.box_width_grid = None
        self.box_height_grid = None
        self.visible_grid = None

        self.image_width = None
        self.image_height = None

        self.dataset_min_rel_z = None
        self.safe_min_rel_z = None

        self._load_metadata()
        self._load_labels_into_grid()

    def _load_metadata(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        grid = self.metadata["grid"]
        camera = self.metadata["camera"]

        self.rel_x_values = np.array(grid["rel_x_values"], dtype=np.float32)
        self.rel_z_values = np.array(grid["rel_z_values"], dtype=np.float32)
        self.yaw_values = np.array(grid["yaw_values"], dtype=np.float32)

        self.image_width = int(camera["width"])
        self.image_height = int(camera["height"])

        self.dataset_min_rel_z = float(np.min(self.rel_z_values))
        self.safe_min_rel_z = self.dataset_min_rel_z + self.safe_distance_margin_m

        for i, v in enumerate(self.rel_x_values):
            self.rel_x_to_idx[round(float(v), 6)] = i

        for i, v in enumerate(self.rel_z_values):
            self.rel_z_to_idx[round(float(v), 6)] = i

        for i, v in enumerate(self.yaw_values):
            self.yaw_to_idx[round(float(v), 6)] = i

        shape = (
            len(self.rel_z_values),
            len(self.rel_x_values),
            len(self.yaw_values),
        )

        self.center_x_grid = np.zeros(shape, dtype=np.float32)
        self.bottom_y_grid = np.zeros(shape, dtype=np.float32)
        self.box_width_grid = np.zeros(shape, dtype=np.float32)
        self.box_height_grid = np.zeros(shape, dtype=np.float32)
        self.visible_grid = np.zeros(shape, dtype=np.float32)

    def _load_labels_into_grid(self):
        if not self.labels_path.exists():
            raise FileNotFoundError(f"labels.jsonl not found: {self.labels_path}")

        loaded = 0

        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)

                rel = row["relative_state"]
                target = row["target"]

                rel_x = round(float(rel["rel_x"]), 6)
                rel_z = round(float(rel["rel_z"]), 6)
                rel_yaw = round(float(rel["rel_yaw"]), 6)

                zi = self.rel_z_to_idx[rel_z]
                xi = self.rel_x_to_idx[rel_x]
                yi = self.yaw_to_idx[rel_yaw]

                self.center_x_grid[zi, xi, yi] = float(target["center_x"])
                self.bottom_y_grid[zi, xi, yi] = float(target["bottom_y"])
                self.box_width_grid[zi, xi, yi] = float(target["box_width"])
                self.box_height_grid[zi, xi, yi] = float(target["box_height"])
                self.visible_grid[zi, xi, yi] = float(target["visible"])

                loaded += 1

        expected = (
            len(self.rel_z_values)
            * len(self.rel_x_values)
            * len(self.yaw_values)
        )

        if loaded != expected:
            print(
                f"[WARN] Loaded {loaded} rows, but expected {expected}. "
                f"Lookup may still work, but grid may be incomplete."
            )

    def _nearest_index(self, values, value):
        value = float(value)
        return int(np.argmin(np.abs(values - value)))

    def _prepare_query(self, rel_x, rel_z, rel_yaw, mode=None):
        if mode is None:
            mode = self.default_mode

        rel_x = float(rel_x)
        rel_z_original = float(rel_z)
        rel_z = float(rel_z)
        rel_yaw = normalize_yaw_deg(rel_yaw)

        # Clamp lateral offset to table range.
        rel_x_clamped = float(np.clip(rel_x, self.rel_x_values[0], self.rel_x_values[-1]))

        close_range = rel_z < self.safe_min_rel_z

        if close_range:
            if mode == "clamp":
                rel_z = self.safe_min_rel_z
            elif mode == "edge":
                rel_z = rel_z
            elif mode == "invisible":
                return {
                    "valid": False,
                    "reason": "below_safe_min_rel_z",
                    "rel_x": rel_x,
                    "rel_x_used": rel_x_clamped,
                    "rel_z": rel_z_original,
                    "rel_z_used": rel_z_original,
                    "rel_yaw": rel_yaw,
                    "close_range": True,
                }
            else:
                raise ValueError(f"Unknown mode: {mode}")

        rel_z_clamped = float(np.clip(rel_z, self.rel_z_values[0], self.rel_z_values[-1]))

        return {
            "valid": True,
            "reason": "ok",
            "rel_x": rel_x,
            "rel_x_used": rel_x_clamped,
            "rel_z": rel_z_original,
            "rel_z_used": rel_z_clamped,
            "rel_yaw": rel_yaw,
            "close_range": close_range,
        }

    def predict_nearest(self, rel_x, rel_z, rel_yaw, mode=None):
        query = self._prepare_query(rel_x, rel_z, rel_yaw, mode=mode)

        if not query["valid"]:
            return self._invisible_result(query)

        xi = self._nearest_index(self.rel_x_values, query["rel_x_used"])
        zi = self._nearest_index(self.rel_z_values, query["rel_z_used"])
        yi = self._nearest_index(self.yaw_values, query["rel_yaw"])

        center_x = float(self.center_x_grid[zi, xi, yi])
        bottom_y = float(self.bottom_y_grid[zi, xi, yi])
        box_width = float(self.box_width_grid[zi, xi, yi])
        box_height = float(self.box_height_grid[zi, xi, yi])
        visible = int(self.visible_grid[zi, xi, yi] >= 0.5)

        return self._format_result(
            center_x=center_x,
            bottom_y=bottom_y,
            box_width=box_width,
            box_height=box_height,
            visible=visible,
            query=query,
            lookup={
                "method": "nearest",
                "rel_x_grid": float(self.rel_x_values[xi]),
                "rel_z_grid": float(self.rel_z_values[zi]),
                "rel_yaw_grid": float(self.yaw_values[yi]),
                "rel_x_index": int(xi),
                "rel_z_index": int(zi),
                "rel_yaw_index": int(yi),
            },
        )
    def _bounds_and_weight(self, values, value):
        """
        Return lower index, upper index, interpolation weight.

        If value is exactly on grid or outside range after clamping:
            lower == upper, weight = 0
        """
        value = float(value)

        if value <= float(values[0]):
            return 0, 0, 0.0

        if value >= float(values[-1]):
            last = len(values) - 1
            return last, last, 0.0

        upper = int(np.searchsorted(values, value, side="right"))
        lower = upper - 1

        v0 = float(values[lower])
        v1 = float(values[upper])

        if abs(v1 - v0) < 1e-9:
            return lower, upper, 0.0

        weight = (value - v0) / (v1 - v0)
        return lower, upper, float(weight)

    def _yaw_bounds_and_weight(self, yaw):
        """
        Yaw-aware bounds.

        Current grid has -180 ... 180.
        Since -180 and +180 are duplicate orientations, normal bounded
        interpolation is okay for most queries. The yaw is already normalized.
        """
        yaw = normalize_yaw_deg(yaw)
        return self._bounds_and_weight(self.yaw_values, yaw)

    def _trilinear_grid_value(self, grid, zi0, zi1, wz, xi0, xi1, wx, yi0, yi1, wy):
        """
        Trilinear interpolation over:
            z axis: rel_z
            x axis: rel_x
            y axis: rel_yaw
        Grid shape:
            [rel_z, rel_x, rel_yaw]
        """

        c000 = float(grid[zi0, xi0, yi0])
        c001 = float(grid[zi0, xi0, yi1])
        c010 = float(grid[zi0, xi1, yi0])
        c011 = float(grid[zi0, xi1, yi1])

        c100 = float(grid[zi1, xi0, yi0])
        c101 = float(grid[zi1, xi0, yi1])
        c110 = float(grid[zi1, xi1, yi0])
        c111 = float(grid[zi1, xi1, yi1])

        c00 = c000 * (1.0 - wy) + c001 * wy
        c01 = c010 * (1.0 - wy) + c011 * wy
        c10 = c100 * (1.0 - wy) + c101 * wy
        c11 = c110 * (1.0 - wy) + c111 * wy

        c0 = c00 * (1.0 - wx) + c01 * wx
        c1 = c10 * (1.0 - wx) + c11 * wx

        c = c0 * (1.0 - wz) + c1 * wz

        return float(c)

    def predict_interpolated(self, rel_x, rel_z, rel_yaw, mode=None):
        query = self._prepare_query(rel_x, rel_z, rel_yaw, mode=mode)

        if not query["valid"]:
            return self._invisible_result(query)

        xi0, xi1, wx = self._bounds_and_weight(
            self.rel_x_values,
            query["rel_x_used"],
        )

        zi0, zi1, wz = self._bounds_and_weight(
            self.rel_z_values,
            query["rel_z_used"],
        )

        yi0, yi1, wy = self._yaw_bounds_and_weight(
            query["rel_yaw"],
        )

        center_x = self._trilinear_grid_value(
            self.center_x_grid,
            zi0, zi1, wz,
            xi0, xi1, wx,
            yi0, yi1, wy,
        )

        bottom_y = self._trilinear_grid_value(
            self.bottom_y_grid,
            zi0, zi1, wz,
            xi0, xi1, wx,
            yi0, yi1, wy,
        )

        box_width = self._trilinear_grid_value(
            self.box_width_grid,
            zi0, zi1, wz,
            xi0, xi1, wx,
            yi0, yi1, wy,
        )

        box_height = self._trilinear_grid_value(
            self.box_height_grid,
            zi0, zi1, wz,
            xi0, xi1, wx,
            yi0, yi1, wy,
        )

        visible_value = self._trilinear_grid_value(
            self.visible_grid,
            zi0, zi1, wz,
            xi0, xi1, wx,
            yi0, yi1, wy,
        )

        visible = int(visible_value >= 0.5)

        return self._format_result(
            center_x=center_x,
            bottom_y=bottom_y,
            box_width=box_width,
            box_height=box_height,
            visible=visible,
            query=query,
            lookup={
                "method": "trilinear",
                "rel_x_bounds": [
                    float(self.rel_x_values[xi0]),
                    float(self.rel_x_values[xi1]),
                ],
                "rel_z_bounds": [
                    float(self.rel_z_values[zi0]),
                    float(self.rel_z_values[zi1]),
                ],
                "rel_yaw_bounds": [
                    float(self.yaw_values[yi0]),
                    float(self.yaw_values[yi1]),
                ],
                "weights": {
                    "wx": float(wx),
                    "wz": float(wz),
                    "wy": float(wy),
                },
                "indices": {
                    "xi0": int(xi0),
                    "xi1": int(xi1),
                    "zi0": int(zi0),
                    "zi1": int(zi1),
                    "yi0": int(yi0),
                    "yi1": int(yi1),
                },
            },
        )

    def predict(self, rel_x, rel_z, rel_yaw, mode=None, method="interpolated"):
        if method == "nearest":
            return self.predict_nearest(
                rel_x=rel_x,
                rel_z=rel_z,
                rel_yaw=rel_yaw,
                mode=mode,
            )

        if method in ["interpolated", "trilinear"]:
            return self.predict_interpolated(
                rel_x=rel_x,
                rel_z=rel_z,
                rel_yaw=rel_yaw,
                mode=mode,
            )

        raise ValueError(f"Unknown prediction method: {method}")
        
    def _format_result(
        self,
        center_x,
        bottom_y,
        box_width,
        box_height,
        visible,
        query,
        lookup,
    ):
        x = center_x - box_width / 2.0
        y = bottom_y - box_height

        return {
            "center_x": center_x,
            "bottom_y": bottom_y,
            "box_width": box_width,
            "box_height": box_height,
            "visible": int(visible),
            "x": x,
            "y": y,
            "w": box_width,
            "h": box_height,
            "query": query,
            "lookup": lookup,
        }

    def _invisible_result(self, query):
        return {
            "center_x": 0.0,
            "bottom_y": 0.0,
            "box_width": 0.0,
            "box_height": 0.0,
            "visible": 0,
            "x": 0.0,
            "y": 0.0,
            "w": 0.0,
            "h": 0.0,
            "query": query,
            "lookup": {
                "method": "none",
            },
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="dataset/v2_lookup_full/labels.jsonl")
    parser.add_argument("--metadata", type=str, default="dataset/v2_lookup_full/metadata.json")
    parser.add_argument("--rel-x", type=float, default=0.0)
    parser.add_argument("--rel-z", type=float, default=35.0)
    parser.add_argument("--rel-yaw", type=float, default=180.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="clamp",
        choices=["clamp", "edge", "invisible"],
    )
    parser.add_argument(
        "--method",
        type=str,
        default="interpolated",
        choices=["nearest", "interpolated", "trilinear"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading v2 lookup placement model...")
    model = HELookupPlacementV2(
        labels_path=args.labels,
        metadata_path=args.metadata,
        safe_distance_margin_m=5.0,
        default_mode=args.mode,
    )

    print("[INFO] Loaded.")
    print(f"[INFO] dataset_min_rel_z = {model.dataset_min_rel_z}")
    print(f"[INFO] safe_min_rel_z    = {model.safe_min_rel_z}")

    pred = model.predict(
        rel_x=args.rel_x,
        rel_z=args.rel_z,
        rel_yaw=args.rel_yaw,
        mode=args.mode,
        method=args.method,
    )

    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()