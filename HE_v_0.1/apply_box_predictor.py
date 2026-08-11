import os
import cv2
import json
import glob
import argparse
import numpy as np

import torch
import torch.nn as nn

from HallucinationEngine.insertion import ObjectInserter


# ============================================================
# Settings
# ============================================================

PAIRED_BASE_DIR = "paired_data"
CHECKPOINT_BASE_DIR = "box_predictor_checkpoints"

SPRITE_PATH = "assets/sprites/carla_front_vehicle.png"

IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALAR_DIM = 12


# ============================================================
# Utility
# ============================================================

def find_latest_pair_run(base_dir=PAIRED_BASE_DIR):
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Paired-data base not found: {base_dir}")

    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("pair_run_")
        and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not runs:
        raise RuntimeError(f"No pair_run folders found in: {base_dir}")

    runs = sorted(runs, key=os.path.getmtime, reverse=True)
    return runs[0]


def find_latest_box_checkpoint(base_dir=CHECKPOINT_BASE_DIR):
    """
    Prefer latest box_predictor_best.pth.
    If unavailable, use latest .pth checkpoint.
    """

    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Box checkpoint base not found: {base_dir}")

    best_ckpts = glob.glob(
        os.path.join(base_dir, "box_predictor*", "checkpoints", "box_predictor_best.pth")
    )

    if best_ckpts:
        best_ckpts = sorted(best_ckpts, key=os.path.getmtime, reverse=True)
        return best_ckpts[0]

    all_ckpts = glob.glob(
        os.path.join(base_dir, "box_predictor*", "checkpoints", "*.pth")
    )

    if not all_ckpts:
        raise RuntimeError(f"No box predictor checkpoints found in: {base_dir}")

    all_ckpts = sorted(all_ckpts, key=os.path.getmtime, reverse=True)
    return all_ckpts[0]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def read_bgr(path):
    img = cv2.imread(path)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    return img


def bgr_to_tensor_resized(img_bgr, size=IMAGE_SIZE):
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)

    return tensor


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def get_frame_id_from_name(path):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]

    if not stem.startswith("frame_"):
        return None

    return int(stem.replace("frame_", ""))


# ============================================================
# Model definition
# Must match train_box_predictor.py v2
# ============================================================

class SmallCNNBackbone(nn.Module):
    """
    Lightweight CNN encoder.
    """

    def __init__(self, out_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x


class BoxPredictor(nn.Module):
    """
    Predict normalized [cx, cy, w, h].

    Inputs:
        image feature
        distance-aware scalar scenario features
    """

    def __init__(self, scalar_dim=SCALAR_DIM, hidden_dim=256):
        super().__init__()

        self.backbone = SmallCNNBackbone(out_dim=hidden_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 4),
        )

    def forward(self, image, scalars):
        img_feat = self.backbone(image)
        scalar_feat = self.scalar_mlp(scalars)

        feat = torch.cat([img_feat, scalar_feat], dim=1)
        raw = self.head(feat)

        out = torch.sigmoid(raw)
        return out


# ============================================================
# Box utilities
# ============================================================

def normalized_box_to_xyxy(box_norm, image_w, image_h):
    """
    Convert normalized [cx,cy,w,h] to pixel [x1,y1,x2,y2].
    """

    cx, cy, bw, bh = [float(v) for v in box_norm]

    cx_px = cx * image_w
    cy_px = cy * image_h
    bw_px = bw * image_w
    bh_px = bh * image_h

    x1 = int(round(cx_px - bw_px / 2.0))
    y1 = int(round(cy_px - bh_px / 2.0))
    x2 = int(round(cx_px + bw_px / 2.0))
    y2 = int(round(cy_px + bh_px / 2.0))

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def is_valid_box_xyxy(box):
    if box is None:
        return False

    if len(box) != 4:
        return False

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return False

    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return False

    return True


def draw_box_bgr(img_bgr, box_xyxy, color, label):
    out = img_bgr.copy()

    if box_xyxy is None:
        return out

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

    cv2.putText(
        out,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    return out


# ============================================================
# Sequence metadata helpers
# ============================================================

def find_real_meta_by_frame(real_sequence, frame_id):
    for item in real_sequence:
        if int(item.get("frame", -1)) == int(frame_id):
            return item
    return None


def find_clean_meta_by_frame(clean_sequence, frame_id):
    for item in clean_sequence:
        if int(item.get("frame", -1)) == int(frame_id):
            return item
    return None


def estimate_distance_progress(distance_to_adversary_m, initial_distance_m, frame_progress):
    if initial_distance_m is None or initial_distance_m <= 1e-6:
        return frame_progress

    if distance_to_adversary_m is None:
        return frame_progress

    progress = 1.0 - (float(distance_to_adversary_m) / float(initial_distance_m))
    progress = max(0.0, min(1.5, progress))

    return progress


def get_initial_distance(sequence_metadata):
    """
    Prefer measured initial distance from metadata.
    Fallback to scenario adversary_forward_m.
    """

    adversary_meta = sequence_metadata.get("adversary", {})
    scenario_config = sequence_metadata.get("scenario_config", {})

    initial_distance = safe_float(
        adversary_meta.get("initial_distance_to_adversary_m", None),
        None,
    )

    if initial_distance is None:
        initial_distance = safe_float(
            scenario_config.get("adversary_forward_m", 40.0),
            40.0,
        )

    return initial_distance


def build_scalar_features_from_metadata(sequence_metadata, frame_id):
    """
    Build the same 12 scalar features used in train_box_predictor.py.

    During evaluation on generated pair data, we can use the metadata to compute
    distance-aware features. During future real HE use, these should come from
    HE internal state:
        initial distance from scenario planner
        ego speed / odometry
        estimated current distance
        cumulative forward motion
    """

    scenario_config = sequence_metadata.get("scenario_config", {})
    clean_sequence = sequence_metadata.get("clean_sequence", [])
    real_sequence = sequence_metadata.get("real_sequence", [])

    num_frames = safe_int(sequence_metadata.get("num_frames", 1), 1)
    fixed_delta_seconds = safe_float(sequence_metadata.get("fixed_delta_seconds", 0.05), 0.05)

    frame_progress = 0.0
    if num_frames > 1:
        frame_progress = float(frame_id) / float(num_frames - 1)

    adversary_forward_m = safe_float(
        scenario_config.get("adversary_forward_m", 40.0),
        40.0,
    )

    adversary_lateral_m = safe_float(
        scenario_config.get("adversary_lateral_m", 0.0),
        0.0,
    )

    ego_throttle = safe_float(
        scenario_config.get("ego_throttle", 0.0),
        0.0,
    )

    ego_spawn_index = safe_float(
        scenario_config.get("ego_spawn_index", 0.0),
        0.0,
    )

    time_seconds = float(frame_id) * fixed_delta_seconds

    initial_distance_to_adversary_m = get_initial_distance(sequence_metadata)

    # Prefer real_sequence metadata because it has the real stationary adversary distance.
    # For final real HE, replace this with estimated distance from HE state.
    real_meta = find_real_meta_by_frame(real_sequence, frame_id)
    clean_meta = find_clean_meta_by_frame(clean_sequence, frame_id)

    distance_to_adversary_m = None
    distance_progress = None

    if real_meta is not None:
        distance_to_adversary_m = safe_float(
            real_meta.get("distance_to_adversary_m", None),
            None,
        )

        distance_progress = safe_float(
            real_meta.get("distance_progress", None),
            None,
        )

    # Ego speed and displacement should be available in both clean and real sequences.
    # Prefer clean sequence because prediction operates on clean frames.
    ego_speed_mps = 0.0
    ego_forward_displacement_m = 0.0

    if clean_meta is not None:
        ego_speed_mps = safe_float(
            clean_meta.get("ego_speed_mps", 0.0),
            0.0,
        )
        ego_forward_displacement_m = safe_float(
            clean_meta.get("ego_forward_displacement_m", 0.0),
            0.0,
        )
    elif real_meta is not None:
        ego_speed_mps = safe_float(
            real_meta.get("ego_speed_mps", 0.0),
            0.0,
        )
        ego_forward_displacement_m = safe_float(
            real_meta.get("ego_forward_displacement_m", 0.0),
            0.0,
        )

    estimated_distance_from_displacement_m = (
        initial_distance_to_adversary_m - ego_forward_displacement_m
    )

    if estimated_distance_from_displacement_m < 0.0:
        estimated_distance_from_displacement_m = 0.0

    if distance_to_adversary_m is None:
        distance_to_adversary_m = estimated_distance_from_displacement_m

    if distance_progress is None:
        distance_progress = estimate_distance_progress(
            distance_to_adversary_m=distance_to_adversary_m,
            initial_distance_m=initial_distance_to_adversary_m,
            frame_progress=frame_progress,
        )

    scalar_features = [
        frame_progress,
        adversary_forward_m / 60.0,
        adversary_lateral_m / 3.0,
        ego_throttle / 1.0,
        ego_spawn_index / 300.0,
        time_seconds / 20.0,
        initial_distance_to_adversary_m / 60.0,
        distance_to_adversary_m / 60.0,
        estimated_distance_from_displacement_m / 60.0,
        distance_progress,
        ego_speed_mps / 20.0,
        ego_forward_displacement_m / 60.0,
    ]

    if len(scalar_features) != SCALAR_DIM:
        raise RuntimeError(f"Expected {SCALAR_DIM} scalar features, got {len(scalar_features)}")

    return torch.tensor(scalar_features, dtype=torch.float32), {
        "frame_progress": frame_progress,
        "adversary_forward_m": adversary_forward_m,
        "adversary_lateral_m": adversary_lateral_m,
        "ego_throttle": ego_throttle,
        "ego_spawn_index": ego_spawn_index,
        "time_seconds": time_seconds,
        "initial_distance_to_adversary_m": initial_distance_to_adversary_m,
        "distance_to_adversary_m": distance_to_adversary_m,
        "estimated_distance_from_displacement_m": estimated_distance_from_displacement_m,
        "distance_progress": distance_progress,
        "ego_speed_mps": ego_speed_mps,
        "ego_forward_displacement_m": ego_forward_displacement_m,
    }


# ============================================================
# Load model
# ============================================================

def load_box_predictor(checkpoint_path):
    print("[ApplyBoxV2] Loading checkpoint:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    train_meta = checkpoint.get("train_meta", {})
    hidden_dim = int(train_meta.get("hidden_dim", 256))
    scalar_dim = int(train_meta.get("scalar_dim", SCALAR_DIM))

    if scalar_dim != SCALAR_DIM:
        raise RuntimeError(
            f"Checkpoint scalar_dim={scalar_dim}, but this script expects {SCALAR_DIM}. "
            f"Make sure you are using the v2 distance-aware checkpoint."
        )

    model = BoxPredictor(
        scalar_dim=SCALAR_DIM,
        hidden_dim=hidden_dim,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[ApplyBoxV2] Loaded box predictor.")
    print("[ApplyBoxV2] Hidden dim:", hidden_dim)
    print("[ApplyBoxV2] Scalar dim:", scalar_dim)
    print("[ApplyBoxV2] Device:", DEVICE)

    return model, train_meta


# ============================================================
# Main apply function
# ============================================================

def apply_box_predictor(pair_run_dir, checkpoint_path):
    model, train_meta = load_box_predictor(checkpoint_path)

    metadata_path = os.path.join(
        pair_run_dir,
        "metadata",
        "sequence_metadata.json",
    )

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Missing sequence metadata: {metadata_path}")

    sequence_metadata = load_json(metadata_path)
    real_sequence = sequence_metadata.get("real_sequence", [])

    clean_dir = os.path.join(pair_run_dir, "clean")
    output_dir = os.path.join(pair_run_dir, "he_box_predicted")
    preview_dir = os.path.join(pair_run_dir, "he_box_predicted_preview")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    if not clean_paths:
        raise RuntimeError(f"No clean frames found in: {clean_dir}")

    inserter = ObjectInserter()

    predicted_metadata = []

    print("[ApplyBoxV2] Pair run:", pair_run_dir)
    print("[ApplyBoxV2] Frames:", len(clean_paths))
    print("[ApplyBoxV2] Output:", output_dir)

    with torch.no_grad():
        for idx, clean_path in enumerate(clean_paths):
            frame_name = os.path.basename(clean_path)
            frame_id = get_frame_id_from_name(clean_path)

            if frame_id is None:
                continue

            clean_bgr = read_bgr(clean_path)
            image_h, image_w = clean_bgr.shape[:2]

            image_tensor = bgr_to_tensor_resized(clean_bgr, IMAGE_SIZE)

            scalar_tensor, scalar_debug = build_scalar_features_from_metadata(
                sequence_metadata=sequence_metadata,
                frame_id=frame_id,
            )

            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
            scalar_tensor = scalar_tensor.unsqueeze(0).to(DEVICE)

            pred_norm = model(image_tensor, scalar_tensor)[0]
            pred_norm_np = pred_norm.detach().cpu().numpy().tolist()

            pred_box_xyxy = normalized_box_to_xyxy(
                pred_norm_np,
                image_w=image_w,
                image_h=image_h,
            )

            if pred_box_xyxy is None or not is_valid_box_xyxy(pred_box_xyxy):
                output_bgr = clean_bgr.copy()
                insertion_status = "invalid_predicted_box"
            else:
                clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

                output_rgb = inserter.insert_sprite(
                    frame_rgb=clean_rgb,
                    sprite_path=SPRITE_PATH,
                    box_2d=pred_box_xyxy,
                    alpha=1.0,
                    horizontal_scale=1.1,
                    vertical_scale=1.15,
                    y_offset_ratio=0.0,
                    match_brightness=True,
                    add_shadow=True,
                    soften_edges=True,
                    motion_blur=False,
                    debug_box=False,
                )

                output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
                insertion_status = "inserted_using_predicted_box"

            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, output_bgr)

            # Ground-truth box for diagnostics only
            real_meta = find_real_meta_by_frame(real_sequence, frame_id)
            real_box = None
            if real_meta is not None:
                real_box = real_meta.get("adversary_box_2d", None)

            predicted_metadata.append({
                "frame": frame_id,
                "frame_name": frame_name,
                "clean_path": clean_path,
                "output_path": output_path,
                "pred_box_norm_cxcywh": pred_norm_np,
                "pred_box_2d": pred_box_xyxy,
                "real_box_2d_for_eval_only": real_box,
                "scalar_debug": scalar_debug,
                "insertion_status": insertion_status,
            })

            if idx % 20 == 0:
                preview = clean_bgr.copy()

                if real_box is not None:
                    preview = draw_box_bgr(
                        preview,
                        real_box,
                        color=(0, 0, 255),
                        label="real",
                    )

                if pred_box_xyxy is not None:
                    preview = draw_box_bgr(
                        preview,
                        pred_box_xyxy,
                        color=(0, 255, 0),
                        label="pred",
                    )

                # Add useful scalar debug text.
                cv2.rectangle(
                    preview,
                    (0, 0),
                    (preview.shape[1], 90),
                    (0, 0, 0),
                    -1,
                )

                debug_text_1 = (
                    f"frame={frame_id} "
                    f"dist={scalar_debug['distance_to_adversary_m']:.2f}m "
                    f"progress={scalar_debug['distance_progress']:.3f}"
                )

                debug_text_2 = (
                    f"ego_speed={scalar_debug['ego_speed_mps']:.2f}m/s "
                    f"ego_disp={scalar_debug['ego_forward_displacement_m']:.2f}m "
                    f"est_dist={scalar_debug['estimated_distance_from_displacement_m']:.2f}m"
                )

                cv2.putText(
                    preview,
                    debug_text_1,
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    preview,
                    debug_text_2,
                    (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                preview = cv2.resize(preview, (960, 540), interpolation=cv2.INTER_AREA)

                preview_path = os.path.join(
                    preview_dir,
                    f"preview_{frame_id:06d}.png",
                )

                cv2.imwrite(preview_path, preview)

            if idx % 20 == 0:
                print(
                    f"[ApplyBoxV2] Processed frame {idx}/{len(clean_paths)} "
                    f"| dist={scalar_debug['distance_to_adversary_m']:.2f}m "
                    f"| pred_box={pred_box_xyxy}"
                )

    out_metadata = {
        "model_version": "box_predictor_v2_distance_aware",
        "pair_run_dir": pair_run_dir,
        "checkpoint_path": checkpoint_path,
        "sprite_path": SPRITE_PATH,
        "device": DEVICE,
        "num_frames": len(predicted_metadata),
        "note": (
            "real_box_2d_for_eval_only is not used for prediction/insertion. "
            "Distance-aware scalar features are generated from pair-run metadata."
        ),
        "frames": predicted_metadata,
    }

    out_metadata_path = os.path.join(
        pair_run_dir,
        "metadata",
        "he_box_predicted_metadata.json",
    )

    save_json(out_metadata_path, out_metadata)

    print("[ApplyBoxV2] Done.")
    print("[ApplyBoxV2] Predicted HE frames:", output_dir)
    print("[ApplyBoxV2] Metadata:", out_metadata_path)
    print("[ApplyBoxV2] Preview:", preview_dir)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair_run_dir",
        type=str,
        default=None,
        help="Path to pair_run folder. If omitted, latest pair_run is used.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to box predictor checkpoint. If omitted, latest best checkpoint is used.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override device.",
    )

    args = parser.parse_args()

    global DEVICE

    if args.device is not None:
        DEVICE = args.device

    if args.pair_run_dir is None:
        pair_run_dir = find_latest_pair_run(PAIRED_BASE_DIR)
    else:
        pair_run_dir = args.pair_run_dir

    if args.checkpoint is None:
        checkpoint_path = find_latest_box_checkpoint(CHECKPOINT_BASE_DIR)
    else:
        checkpoint_path = args.checkpoint

    apply_box_predictor(
        pair_run_dir=pair_run_dir,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()