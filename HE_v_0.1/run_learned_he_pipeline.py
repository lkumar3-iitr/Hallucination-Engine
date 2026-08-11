import os
import cv2
import json
import glob
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from HallucinationEngine.insertion import ObjectInserter


# ============================================================
# Settings
# ============================================================

PAIRED_BASE_DIR = "paired_data"

SPRITE_PATH = "assets/sprites/carla_front_vehicle.png"

BOX_IMAGE_SIZE = 224
BOX_SCALAR_DIM = 12

REFINER_IMAGE_SIZE = 384
REFINER_CROP_EXPAND_FACTOR = 1.5
DIFF_THRESHOLD = 12

VIDEO_OUTPUT_NAME = "comparison_learned_he_pipeline.mp4"
VIDEO_FPS = 20
PANEL_SCALE = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Generic utilities
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


def save_bgr(path, img):
    cv2.imwrite(path, img)


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


def bgr_to_tensor_resized(img_bgr, size):
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    return tensor


def bgr_to_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    return tensor


def tensor_to_bgr(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    rgb = tensor.permute(1, 2, 0).numpy()
    rgb = (rgb * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def mask_to_tensor(mask_gray):
    if len(mask_gray.shape) == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)

    mask = mask_gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(mask).unsqueeze(0)
    return tensor


# ============================================================
# Box predictor model
# Must match train_box_predictor.py v2
# ============================================================

class BoxSmallCNNBackbone(nn.Module):
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
    def __init__(self, scalar_dim=BOX_SCALAR_DIM, hidden_dim=256):
        super().__init__()

        self.backbone = BoxSmallCNNBackbone(out_dim=hidden_dim)

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
# Residual refiner model
# Must match train_local_refiner_residual.py
# ============================================================

class RefinerConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualUNetRefiner(nn.Module):
    def __init__(self, in_ch=7, out_ch=3, base_ch=32, max_delta=0.25):
        super().__init__()

        self.max_delta = float(max_delta)

        self.enc1 = RefinerConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv2d(
            base_ch,
            base_ch * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc2 = RefinerConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv2d(
            base_ch * 2,
            base_ch * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.mid = RefinerConvBlock(base_ch * 4, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(
            base_ch * 4,
            base_ch * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec2 = RefinerConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(
            base_ch * 2,
            base_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec1 = RefinerConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        m = self.mid(d2)

        u2 = self.up2(m)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        residual = torch.tanh(self.out(d1)) * self.max_delta
        return residual


def apply_residual_tensor(rough, residual, mask):
    mask_dilated = F.max_pool2d(
        mask,
        kernel_size=21,
        stride=1,
        padding=10,
    )

    refined = rough + residual * mask_dilated
    refined = torch.clamp(refined, 0.0, 1.0)

    return refined


# ============================================================
# Box predictor helpers
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


def get_initial_distance(sequence_metadata):
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


def estimate_distance_progress(distance_to_adversary_m, initial_distance_m, frame_progress):
    if initial_distance_m is None or initial_distance_m <= 1e-6:
        return frame_progress

    if distance_to_adversary_m is None:
        return frame_progress

    progress = 1.0 - (float(distance_to_adversary_m) / float(initial_distance_m))
    progress = max(0.0, min(1.5, progress))

    return progress


def build_scalar_features_from_metadata(sequence_metadata, frame_id):
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

    if len(scalar_features) != BOX_SCALAR_DIM:
        raise RuntimeError(
            f"Expected {BOX_SCALAR_DIM} scalar features, got {len(scalar_features)}"
        )

    debug = {
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

    return torch.tensor(scalar_features, dtype=torch.float32), debug


def normalized_box_to_xyxy(box_norm, image_w, image_h):
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


# ============================================================
# Refiner crop/mask helpers
# ============================================================

def is_valid_box(box):
    if box is None:
        return False

    if not isinstance(box, list):
        return False

    if len(box) != 4:
        return False

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return False

    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return False

    return True


def square_expanded_crop_box(box, image_w, image_h, expand_factor):
    x1, y1, x2, y2 = [float(v) for v in box]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw = x2 - x1
    bh = y2 - y1

    side = max(bw, bh) * expand_factor
    side = max(side, 64.0)

    crop_x1 = int(round(cx - side / 2.0))
    crop_y1 = int(round(cy - side / 2.0))
    crop_x2 = int(round(cx + side / 2.0))
    crop_y2 = int(round(cy + side / 2.0))

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0

    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0

    if crop_x2 > image_w:
        shift = crop_x2 - image_w
        crop_x1 -= shift
        crop_x2 = image_w

    if crop_y2 > image_h:
        shift = crop_y2 - image_h
        crop_y1 -= shift
        crop_y2 = image_h

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(image_w, crop_x2)
    crop_y2 = min(image_h, crop_y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    return [crop_x1, crop_y1, crop_x2, crop_y2]


def crop_and_resize(img, crop_box, size=REFINER_IMAGE_SIZE):
    x1, y1, x2, y2 = crop_box

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def make_mask_from_rough_difference(clean_bgr, rough_bgr, box, crop_box):
    diff = cv2.absdiff(clean_bgr, rough_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    mask = (diff_gray > DIFF_THRESHOLD).astype(np.uint8) * 255

    h, w = mask.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in box]

    bw = x2 - x1
    bh = y2 - y1

    margin_x = int(0.25 * bw)
    margin_y = int(0.25 * bh)

    x1m = max(0, x1 - margin_x)
    y1m = max(0, y1 - margin_y)
    x2m = min(w - 1, x2 + margin_x)
    y2m = min(h - 1, y2 + margin_y)

    constraint = np.zeros_like(mask)
    constraint[y1m:y2m + 1, x1m:x2m + 1] = 255

    mask = cv2.bitwise_and(mask, constraint)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if np.sum(mask > 0) < 50:
        mask = np.zeros_like(mask)
        mask[y1:y2 + 1, x1:x2 + 1] = 255

    mask_crop = crop_and_resize(mask, crop_box, REFINER_IMAGE_SIZE)

    if mask_crop is None:
        return None

    if len(mask_crop.shape) == 3:
        mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_BGR2GRAY)

    mask_crop = (mask_crop > 30).astype(np.uint8) * 255

    return mask_crop


def paste_refined_crop_back(original_bgr, refined_crop_bgr, mask_crop_gray, crop_box):
    x1, y1, x2, y2 = crop_box

    crop_w = x2 - x1
    crop_h = y2 - y1

    refined_resized = cv2.resize(
        refined_crop_bgr,
        (crop_w, crop_h),
        interpolation=cv2.INTER_LINEAR,
    )

    mask_resized = cv2.resize(
        mask_crop_gray,
        (crop_w, crop_h),
        interpolation=cv2.INTER_LINEAR,
    )

    mask_resized = cv2.GaussianBlur(mask_resized, (7, 7), 0)

    alpha = mask_resized.astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_3 = np.dstack([alpha, alpha, alpha])

    output = original_bgr.copy()

    original_crop = output[y1:y2, x1:x2].astype(np.float32)
    refined_crop = refined_resized.astype(np.float32)

    blended = refined_crop * alpha_3 + original_crop * (1.0 - alpha_3)

    output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return output


# ============================================================
# Model loading
# ============================================================

def load_box_predictor(checkpoint_path):
    print("[Pipeline] Loading box predictor:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    train_meta = checkpoint.get("train_meta", {})

    hidden_dim = int(train_meta.get("hidden_dim", 256))
    scalar_dim = int(train_meta.get("scalar_dim", BOX_SCALAR_DIM))

    if scalar_dim != BOX_SCALAR_DIM:
        raise RuntimeError(
            f"Box checkpoint scalar_dim={scalar_dim}, expected {BOX_SCALAR_DIM}. "
            f"Use the v2 distance-aware checkpoint."
        )

    model = BoxPredictor(
        scalar_dim=BOX_SCALAR_DIM,
        hidden_dim=hidden_dim,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[Pipeline] Box predictor loaded.")
    return model, train_meta


def load_residual_refiner(checkpoint_path):
    print("[Pipeline] Loading residual refiner:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    train_meta = checkpoint.get("train_meta", {})

    base_channels = int(train_meta.get("base_channels", 32))
    max_delta = float(train_meta.get("max_residual_delta", 0.25))

    model = ResidualUNetRefiner(
        in_ch=7,
        out_ch=3,
        base_ch=base_channels,
        max_delta=max_delta,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[Pipeline] Residual refiner loaded.")
    return model, train_meta


# ============================================================
# Stage 1: apply box predictor
# ============================================================

def apply_box_predictor_stage(pair_run_dir, box_model, box_checkpoint_path):
    metadata_path = os.path.join(pair_run_dir, "metadata", "sequence_metadata.json")

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

    print("[Pipeline] Applying box predictor...")
    print("[Pipeline] Frames:", len(clean_paths))

    with torch.no_grad():
        for idx, clean_path in enumerate(clean_paths):
            frame_name = os.path.basename(clean_path)
            frame_id = get_frame_id_from_name(clean_path)

            if frame_id is None:
                continue

            clean_bgr = read_bgr(clean_path)
            image_h, image_w = clean_bgr.shape[:2]

            image_tensor = bgr_to_tensor_resized(clean_bgr, BOX_IMAGE_SIZE)

            scalar_tensor, scalar_debug = build_scalar_features_from_metadata(
                sequence_metadata=sequence_metadata,
                frame_id=frame_id,
            )

            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
            scalar_tensor = scalar_tensor.unsqueeze(0).to(DEVICE)

            pred_norm = box_model(image_tensor, scalar_tensor)[0]
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
                print(
                    f"[Pipeline] Box stage frame {idx}/{len(clean_paths)} "
                    f"dist={scalar_debug['distance_to_adversary_m']:.2f}m "
                    f"box={pred_box_xyxy}"
                )

    out_metadata = {
        "model_version": "box_predictor_v2_distance_aware",
        "pair_run_dir": pair_run_dir,
        "box_checkpoint_path": box_checkpoint_path,
        "sprite_path": SPRITE_PATH,
        "device": DEVICE,
        "num_frames": len(predicted_metadata),
        "frames": predicted_metadata,
    }

    out_metadata_path = os.path.join(
        pair_run_dir,
        "metadata",
        "he_box_predicted_metadata.json",
    )

    save_json(out_metadata_path, out_metadata)

    return out_metadata


# ============================================================
# Stage 2: apply residual refiner
# ============================================================

def find_box_pred_meta_by_frame(predicted_frames, frame_id):
    for item in predicted_frames:
        if int(item.get("frame", -1)) == int(frame_id):
            return item
    return None


def apply_residual_refiner_stage(pair_run_dir, refiner_model, refiner_checkpoint_path):
    sequence_metadata_path = os.path.join(pair_run_dir, "metadata", "sequence_metadata.json")
    box_pred_metadata_path = os.path.join(pair_run_dir, "metadata", "he_box_predicted_metadata.json")

    if not os.path.exists(sequence_metadata_path):
        raise RuntimeError(f"Missing sequence metadata: {sequence_metadata_path}")

    if not os.path.exists(box_pred_metadata_path):
        raise RuntimeError(f"Missing box predicted metadata: {box_pred_metadata_path}")

    sequence_metadata = load_json(sequence_metadata_path)
    box_pred_metadata = load_json(box_pred_metadata_path)

    real_sequence = sequence_metadata.get("real_sequence", [])
    predicted_frames = box_pred_metadata.get("frames", [])

    clean_dir = os.path.join(pair_run_dir, "clean")
    real_dir = os.path.join(pair_run_dir, "real_object")
    rough_dir = os.path.join(pair_run_dir, "he_box_predicted")

    refined_dir = os.path.join(pair_run_dir, "he_box_predicted_residual_refined")
    preview_dir = os.path.join(pair_run_dir, "he_box_predicted_residual_refined_preview")

    os.makedirs(refined_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    if not clean_paths:
        raise RuntimeError(f"No clean frames found: {clean_dir}")

    refined_metadata = []

    print("[Pipeline] Applying residual refiner...")
    print("[Pipeline] Frames:", len(clean_paths))

    with torch.no_grad():
        for idx, clean_path in enumerate(clean_paths):
            frame_name = os.path.basename(clean_path)
            frame_id = get_frame_id_from_name(clean_path)

            if frame_id is None:
                continue

            pred_meta = find_box_pred_meta_by_frame(predicted_frames, frame_id)

            if pred_meta is None:
                continue

            box = pred_meta.get("pred_box_2d", None)

            clean_bgr = read_bgr(clean_path)

            rough_path = os.path.join(rough_dir, frame_name)
            real_path = os.path.join(real_dir, frame_name)

            if not os.path.exists(rough_path):
                continue

            rough_bgr = read_bgr(rough_path)

            if os.path.exists(real_path):
                real_bgr = read_bgr(real_path)
            else:
                real_bgr = clean_bgr.copy()

            if not is_valid_box(box):
                save_bgr(os.path.join(refined_dir, frame_name), rough_bgr)

                refined_metadata.append({
                    "frame": frame_id,
                    "status": "no_valid_predicted_box",
                    "pred_box_2d": box,
                })

                continue

            h, w = clean_bgr.shape[:2]

            crop_box = square_expanded_crop_box(
                box=box,
                image_w=w,
                image_h=h,
                expand_factor=REFINER_CROP_EXPAND_FACTOR,
            )

            if crop_box is None:
                save_bgr(os.path.join(refined_dir, frame_name), rough_bgr)
                continue

            clean_crop = crop_and_resize(clean_bgr, crop_box, REFINER_IMAGE_SIZE)
            rough_crop = crop_and_resize(rough_bgr, crop_box, REFINER_IMAGE_SIZE)

            if clean_crop is None or rough_crop is None:
                save_bgr(os.path.join(refined_dir, frame_name), rough_bgr)
                continue

            mask_crop = make_mask_from_rough_difference(
                clean_bgr=clean_bgr,
                rough_bgr=rough_bgr,
                box=box,
                crop_box=crop_box,
            )

            if mask_crop is None:
                save_bgr(os.path.join(refined_dir, frame_name), rough_bgr)
                continue

            clean_tensor = bgr_to_tensor(clean_crop)
            rough_tensor = bgr_to_tensor(rough_crop)
            mask_tensor = mask_to_tensor(mask_crop)

            inp = torch.cat(
                [clean_tensor, rough_tensor, mask_tensor],
                dim=0,
            ).unsqueeze(0).to(DEVICE)

            rough_batch = rough_tensor.unsqueeze(0).to(DEVICE)
            mask_batch = mask_tensor.unsqueeze(0).to(DEVICE)

            residual = refiner_model(inp)

            refined_batch = apply_residual_tensor(
                rough=rough_batch,
                residual=residual,
                mask=mask_batch,
            )

            refined_crop_bgr = tensor_to_bgr(refined_batch[0])

            refined_full_bgr = paste_refined_crop_back(
                original_bgr=rough_bgr,
                refined_crop_bgr=refined_crop_bgr,
                mask_crop_gray=mask_crop,
                crop_box=crop_box,
            )

            refined_path = os.path.join(refined_dir, frame_name)
            save_bgr(refined_path, refined_full_bgr)

            real_meta = find_real_meta_by_frame(real_sequence, frame_id)
            real_box = None
            if real_meta is not None:
                real_box = real_meta.get("adversary_box_2d", None)

            refined_metadata.append({
                "frame": frame_id,
                "status": "residual_refined",
                "pred_box_2d": box,
                "real_box_2d_for_eval_only": real_box,
                "crop_box": crop_box,
                "refined_frame_path": refined_path,
            })

            if idx % 20 == 0:
                preview = create_preview_row(
                    clean_bgr=cv2.resize(clean_bgr, (320, 180)),
                    real_bgr=cv2.resize(real_bgr, (320, 180)),
                    rough_bgr=cv2.resize(rough_bgr, (320, 180)),
                    refined_bgr=cv2.resize(refined_full_bgr, (320, 180)),
                )

                preview_path = os.path.join(
                    preview_dir,
                    f"preview_{frame_id:06d}.png",
                )

                save_bgr(preview_path, preview)

            if idx % 20 == 0:
                print(f"[Pipeline] Refiner stage frame {idx}/{len(clean_paths)}")

    refined_meta = {
        "model_version": "residual_local_refiner_v1",
        "pair_run_dir": pair_run_dir,
        "residual_refiner_checkpoint_path": refiner_checkpoint_path,
        "box_predicted_metadata_path": box_pred_metadata_path,
        "device": DEVICE,
        "image_size": REFINER_IMAGE_SIZE,
        "crop_expand_factor": REFINER_CROP_EXPAND_FACTOR,
        "num_frames_processed": len(refined_metadata),
        "frames": refined_metadata,
    }

    refined_meta_path = os.path.join(
        pair_run_dir,
        "metadata",
        "he_box_predicted_residual_refined_metadata.json",
    )

    save_json(refined_meta_path, refined_meta)

    return refined_meta


def create_preview_row(clean_bgr, real_bgr, rough_bgr, refined_bgr):
    panels = [
        ("clean", clean_bgr),
        ("real", real_bgr),
        ("box_pred", rough_bgr),
        ("residual_refined", refined_bgr),
    ]

    labeled = []

    for label, img in panels:
        panel = img.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labeled.append(panel)

    return np.hstack(labeled)


# ============================================================
# Stage 3: video
# ============================================================

def read_frame(path, target_size=None):
    if path is None:
        return None

    frame = cv2.imread(path)

    if frame is None:
        return None

    if target_size is not None:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    return frame


def put_label(frame, label):
    output = frame.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 42),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        label,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def make_placeholder_like(reference_frame, label):
    placeholder = np.zeros_like(reference_frame)
    placeholder = put_label(placeholder, label)
    return placeholder


def make_pipeline_video(pair_run_dir, output_name=VIDEO_OUTPUT_NAME, fps=VIDEO_FPS):
    clean_dir = os.path.join(pair_run_dir, "clean")
    real_dir = os.path.join(pair_run_dir, "real_object")
    box_pred_dir = os.path.join(pair_run_dir, "he_box_predicted")
    residual_refined_dir = os.path.join(pair_run_dir, "he_box_predicted_residual_refined")

    required_dirs = {
        "clean": clean_dir,
        "real_object": real_dir,
        "he_box_predicted": box_pred_dir,
        "he_box_predicted_residual_refined": residual_refined_dir,
    }

    for name, folder in required_dirs.items():
        if not os.path.isdir(folder):
            raise RuntimeError(f"Missing folder '{name}': {folder}")

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    if not clean_paths:
        raise RuntimeError(f"No clean frames found in: {clean_dir}")

    first_frame = cv2.imread(clean_paths[0])

    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {clean_paths[0]}")

    h, w = first_frame.shape[:2]

    panel_w = int(w * PANEL_SCALE)
    panel_h = int(h * PANEL_SCALE)

    panel_size = (panel_w, panel_h)

    # 2x2 layout:
    # Clean            | Real CARLA Object
    # HE Box Predicted | HE Residual Refined
    output_width = panel_w * 2
    output_height = panel_h * 2

    output_path = os.path.join(pair_run_dir, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    frame_count = 0
    skipped = 0

    print("[Pipeline] Making 2x2 video:", output_path)
    print("[Pipeline] Panel size:", panel_size)
    print("[Pipeline] Video size:", (output_width, output_height))

    for clean_path in clean_paths:
        frame_name = os.path.basename(clean_path)

        real_path = os.path.join(real_dir, frame_name)
        box_pred_path = os.path.join(box_pred_dir, frame_name)
        residual_refined_path = os.path.join(residual_refined_dir, frame_name)

        clean = read_frame(clean_path, panel_size)

        if clean is None:
            skipped += 1
            continue

        real = read_frame(real_path, panel_size)
        box_pred = read_frame(box_pred_path, panel_size)
        residual_refined = read_frame(residual_refined_path, panel_size)

        if real is None:
            real = make_placeholder_like(clean, "Missing Real")

        if box_pred is None:
            box_pred = make_placeholder_like(clean, "Missing Box Pred")

        if residual_refined is None:
            residual_refined = make_placeholder_like(clean, "Missing Residual Refined")

        clean = put_label(clean, "Clean")
        real = put_label(real, "Real CARLA Object")
        box_pred = put_label(box_pred, "HE Box Predicted")
        residual_refined = put_label(residual_refined, "HE Residual Refined")

        top_row = np.hstack([
            clean,
            real,
        ])

        bottom_row = np.hstack([
            box_pred,
            residual_refined,
        ])

        comparison = np.vstack([
            top_row,
            bottom_row,
        ])

        writer.write(comparison)
        frame_count += 1

        if frame_count % 20 == 0:
            print(f"[Pipeline] Video written {frame_count} frames...")

    writer.release()

    print("[Pipeline] Video saved:", output_path)
    print("[Pipeline] Frames written:", frame_count)
    print("[Pipeline] Skipped:", skipped)

    return output_path
# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    pair_run_dir,
    box_checkpoint,
    refiner_checkpoint,
    output_video_name=VIDEO_OUTPUT_NAME,
    skip_box=False,
    skip_refiner=False,
    skip_video=False,
):
    box_model = None
    refiner_model = None

    if not skip_box:
        box_model, _ = load_box_predictor(box_checkpoint)
        apply_box_predictor_stage(
            pair_run_dir=pair_run_dir,
            box_model=box_model,
            box_checkpoint_path=box_checkpoint,
        )

    if not skip_refiner:
        refiner_model, _ = load_residual_refiner(refiner_checkpoint)
        apply_residual_refiner_stage(
            pair_run_dir=pair_run_dir,
            refiner_model=refiner_model,
            refiner_checkpoint_path=refiner_checkpoint,
        )

    video_path = None

    if not skip_video:
        video_path = make_pipeline_video(
            pair_run_dir=pair_run_dir,
            output_name=output_video_name,
            fps=VIDEO_FPS,
        )

    print("[Pipeline] Done.")
    print("[Pipeline] Pair run:", pair_run_dir)

    if video_path is not None:
        print("[Pipeline] Video:", video_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair_run_dir",
        type=str,
        default=None,
        help="Path to pair_run folder. If omitted, latest pair run is used.",
    )

    parser.add_argument(
        "--box_checkpoint",
        type=str,
        required=True,
        help="Path to v2 distance-aware box predictor checkpoint.",
    )

    parser.add_argument(
        "--refiner_checkpoint",
        type=str,
        required=True,
        help="Path to residual local refiner checkpoint.",
    )

    parser.add_argument(
        "--output_video_name",
        type=str,
        default=VIDEO_OUTPUT_NAME,
        help="Output video filename.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override device.",
    )

    parser.add_argument(
        "--skip_box",
        action="store_true",
        help="Skip box prediction stage and reuse existing he_box_predicted.",
    )

    parser.add_argument(
        "--skip_refiner",
        action="store_true",
        help="Skip residual refiner stage and reuse existing residual refined output.",
    )

    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip video creation.",
    )

    args = parser.parse_args()

    global DEVICE

    if args.device is not None:
        DEVICE = args.device

    if args.pair_run_dir is None:
        pair_run_dir = find_latest_pair_run(PAIRED_BASE_DIR)
    else:
        pair_run_dir = args.pair_run_dir

    run_pipeline(
        pair_run_dir=pair_run_dir,
        box_checkpoint=args.box_checkpoint,
        refiner_checkpoint=args.refiner_checkpoint,
        output_video_name=args.output_video_name,
        skip_box=args.skip_box,
        skip_refiner=args.skip_refiner,
        skip_video=args.skip_video,
    )


if __name__ == "__main__":
    main()