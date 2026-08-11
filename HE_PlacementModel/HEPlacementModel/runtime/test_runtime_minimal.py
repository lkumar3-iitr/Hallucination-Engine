print("[TOP] test_runtime_minimal.py started", flush=True)

import json
import numpy as np
import torch
import torch.nn as nn

print("[TOP] imports completed", flush=True)


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


def build_camera_from_fov(width=1280, height=720, fov=90.0):
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
    print("[MAIN] entered main()", flush=True)

    checkpoint_path = "outputs/v1_mlp_combined/heplacement_v1_mlp_best.pt"
    print(f"[MAIN] loading checkpoint: {checkpoint_path}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("[MAIN] checkpoint loaded", flush=True)
    print(f"[MAIN] keys: {list(checkpoint.keys())}", flush=True)

    input_dim = int(checkpoint["input_dim"])
    print(f"[MAIN] input_dim={input_dim}", flush=True)

    model = PlacementMLP(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[MAIN] model loaded", flush=True)

    normalizer = checkpoint["normalizer"]
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
    std = torch.tensor(normalizer["std"], dtype=torch.float32)

    camera = build_camera_from_fov()

    rel_x = 0.0
    rel_z = 35.0
    rel_y = 0.0
    rel_yaw = 180.0

    x = np.array(
        [
            rel_x,
            rel_z,
            rel_y,
            rel_yaw,
            float(camera["fx"]),
            float(camera["fy"]),
            float(camera["cx"]),
            float(camera["cy"]),
            float(camera["width"]),
            float(camera["height"]),
            float(camera["fov"]),
            float(camera["mount_x"]),
            float(camera["mount_y"]),
            float(camera["mount_z"]),
            float(camera["pitch"]),
            float(camera["yaw"]),
            float(camera["roll"]),
        ],
        dtype=np.float32,
    )

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    x_t = (x_t - mean) / std

    print("[MAIN] running inference", flush=True)

    with torch.no_grad():
        bbox_norm, visible_logit = model(x_t)

    bbox_norm = bbox_norm.squeeze(0).numpy()
    visible_prob = float(torch.sigmoid(visible_logit).item())

    pred = {
        "center_x": float(bbox_norm[0]) * camera["width"],
        "bottom_y": float(bbox_norm[1]) * camera["height"],
        "box_width": float(bbox_norm[2]) * camera["width"],
        "box_height": float(bbox_norm[3]) * camera["height"],
        "visible": int(visible_prob >= 0.5),
        "visible_prob": visible_prob,
        "bbox_norm": {
            "center_x": float(bbox_norm[0]),
            "bottom_y": float(bbox_norm[1]),
            "box_width": float(bbox_norm[2]),
            "box_height": float(bbox_norm[3]),
        },
    }

    print("[MAIN] prediction:", flush=True)
    print(json.dumps(pred, indent=2), flush=True)


if __name__ == "__main__":
    print("[BOTTOM] __main__ reached", flush=True)
    main()