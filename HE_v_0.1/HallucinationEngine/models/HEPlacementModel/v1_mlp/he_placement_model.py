import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PlacementMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

        self.visible_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.backbone(x)
        bbox = self.bbox_head(h)
        visible_logit = self.visible_head(h)
        return bbox, visible_logit


class HEPlacementModel:
    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = str(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.input_dim = int(checkpoint["input_dim"])
        self.input_order = checkpoint["input_order"]
        self.target_order = checkpoint["target_order"]
        self.normalizer = checkpoint["normalizer"]
        self.default_rel_y = float(
            self.normalizer["mean"][self.input_order.index("rel_y")]
        )
        self.model = PlacementMLP(input_dim=self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.mean = torch.tensor(
            self.normalizer["mean"],
            dtype=torch.float32,
            device=self.device,
        )

        self.std = torch.tensor(
            self.normalizer["std"],
            dtype=torch.float32,
            device=self.device,
        )

    def build_input_vector(self, relative_state, camera):
        """
        Expected relative_state:
            rel_x, rel_z, rel_y, rel_yaw

        Expected camera:
            width, height, fov,
            fx, fy, cx, cy,
            mount_x, mount_y, mount_z,
            pitch, yaw, roll
        """

        width = float(camera["width"])
        height = float(camera["height"])

        x = np.array(
            [
                float(relative_state["rel_x"]),
                float(relative_state["rel_z"]),
                float(relative_state.get("rel_y", self.default_rel_y)),
                float(relative_state["rel_yaw"]),

                float(camera["fx"]),
                float(camera["fy"]),
                float(camera["cx"]),
                float(camera["cy"]),

                width,
                height,
                float(camera["fov"]),

                float(camera.get("mount_x", 1.5)),
                float(camera.get("mount_y", 0.0)),
                float(camera.get("mount_z", 1.6)),

                float(camera.get("pitch", 0.0)),
                float(camera.get("yaw", 0.0)),
                float(camera.get("roll", 0.0)),
            ],
            dtype=np.float32,
        )

        return x

    def predict(self, relative_state, camera, visible_threshold=0.5):
        x = self.build_input_vector(relative_state, camera)

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_t = (x_t - self.mean) / self.std

        with torch.no_grad():
            bbox_norm, visible_logit = self.model(x_t)

        bbox_norm = bbox_norm.squeeze(0).cpu().numpy()
        visible_prob = float(torch.sigmoid(visible_logit).item())

        width = float(camera["width"])
        height = float(camera["height"])

        center_x = float(bbox_norm[0]) * width
        bottom_y = float(bbox_norm[1]) * height
        box_width = float(bbox_norm[2]) * width
        box_height = float(bbox_norm[3]) * height

        visible = int(visible_prob >= visible_threshold)

        return {
            "center_x": center_x,
            "bottom_y": bottom_y,
            "box_width": box_width,
            "box_height": box_height,
            "visible": visible,
            "visible_prob": visible_prob,
            "bbox_norm": {
                "center_x": float(bbox_norm[0]),
                "bottom_y": float(bbox_norm[1]),
                "box_width": float(bbox_norm[2]),
                "box_height": float(bbox_norm[3]),
            },
        }

    def predict_from_flat_values(
        self,
        rel_x,
        rel_z,
        rel_yaw,
        camera,
        rel_y=None,
        visible_threshold=0.5,
    ):
        relative_state = {
            "rel_x": rel_x,
            "rel_z": rel_z,
            "rel_yaw": rel_yaw,
        }

        if rel_y is not None:
            relative_state["rel_y"] = rel_y

        return self.predict(
            relative_state=relative_state,
            camera=camera,
            visible_threshold=visible_threshold,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/v1_mlp_combined/heplacement_v1_mlp_best.pt",
    )
    parser.add_argument("--rel-y", type=float, default=None)
    parser.add_argument("--rel-x", type=float, default=0.0)
    parser.add_argument("--rel-z", type=float, default=35.0)
    parser.add_argument("--rel-yaw", type=float, default=180.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    return parser.parse_args()


def build_camera_from_fov(width, height, fov):
    """
    Runtime helper for the same pinhole intrinsics used during training.
    CARLA RGB camera FOV is horizontal FOV.
    """
    fov_rad = np.deg2rad(float(fov))
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    return {
        "width": width,
        "height": height,
        "fov": fov,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "mount_x": 1.5,
        "mount_y": 0.0,
        "mount_z": 1.6,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }


def main():
    args = parse_args()

    print("[INFO] Starting HEPlacementModel runtime test")
    print(f"[INFO] Checkpoint: {args.checkpoint}")

    print("[INFO] Loading model...")
    model = HEPlacementModel(args.checkpoint, device="cpu")
    print("[INFO] Model loaded successfully")

    print("[INFO] Building camera...")
    camera = build_camera_from_fov(
        width=args.width,
        height=args.height,
        fov=args.fov,
    )
    print("[INFO] Camera built")

    print("[INFO] Running prediction...")
    pred = model.predict_from_flat_values(
        rel_x=args.rel_x,
        rel_z=args.rel_z,
        rel_y=args.rel_y,
        rel_yaw=args.rel_yaw,
        camera=camera,
    )
    print("[INFO] Prediction complete")

    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()