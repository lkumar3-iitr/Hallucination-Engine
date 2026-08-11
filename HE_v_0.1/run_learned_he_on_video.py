import os
import cv2
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from HallucinationEngine.insertion import ObjectInserter


# ============================================================
# Settings
# ============================================================

SPRITE_PATH_DEFAULT = "assets/sprites/carla_front_vehicle.png"

BOX_IMAGE_SIZE = 224
BOX_SCALAR_DIM = 12

REFINER_IMAGE_SIZE = 384
REFINER_CROP_EXPAND_FACTOR = 1.5
DIFF_THRESHOLD = 12

PANEL_SCALE = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def draw_box_bgr(img_bgr, box_xyxy, color=(0, 255, 0), label="pred"):
    output = img_bgr.copy()

    if box_xyxy is None:
        return output

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    cv2.putText(
        output,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    return output


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
# Load checkpoints
# ============================================================

def load_box_predictor(checkpoint_path):
    print("[VideoHE] Loading box predictor:", checkpoint_path)

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

    print("[VideoHE] Box predictor loaded.")
    return model, train_meta


def load_residual_refiner(checkpoint_path):
    print("[VideoHE] Loading residual refiner:", checkpoint_path)

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

    print("[VideoHE] Residual refiner loaded.")
    print("[VideoHE] Refiner base_channels:", base_channels)
    print("[VideoHE] Refiner max_delta:", max_delta)

    return model, train_meta


# ============================================================
# Box/scenario utilities
# ============================================================

def build_video_scalar_features(
    frame_id,
    total_frames,
    fps,
    initial_distance_m,
    ego_speed_mps,
    adversary_lateral_m,
    ego_throttle,
    ego_spawn_index,
    distance_override_m=None,
):
    """
    Build 12 scalar features expected by BoxPredictor v2.

    For a normal road video, we do not have CARLA metadata.
    So we estimate:
        current_distance = initial_distance - ego_speed * time

    This approximates the stationary wrong-way adversary scenario.
    """

    total_frames = max(1, int(total_frames))
    fps = max(1e-6, float(fps))

    frame_progress = 0.0
    if total_frames > 1:
        frame_progress = float(frame_id) / float(total_frames - 1)

    time_seconds = float(frame_id) / fps

    if distance_override_m is not None:
        distance_to_adversary_m = float(distance_override_m)
    else:
        distance_to_adversary_m = float(initial_distance_m) - float(ego_speed_mps) * time_seconds

    distance_to_adversary_m = max(1.0, distance_to_adversary_m)

    ego_forward_displacement_m = max(
        0.0,
        float(initial_distance_m) - distance_to_adversary_m,
    )

    estimated_distance_from_displacement_m = max(
        0.0,
        float(initial_distance_m) - ego_forward_displacement_m,
    )

    distance_progress = 1.0 - (
        distance_to_adversary_m / max(1e-6, float(initial_distance_m))
    )

    distance_progress = max(0.0, min(1.5, distance_progress))

    scalar_features = [
        frame_progress,
        float(initial_distance_m) / 60.0,                  # adversary_forward_m / 60
        float(adversary_lateral_m) / 3.0,                  # adversary_lateral_m / 3
        float(ego_throttle) / 1.0,                         # ego_throttle
        float(ego_spawn_index) / 300.0,                    # ego_spawn_index / 300
        time_seconds / 20.0,                               # time_seconds / 20
        float(initial_distance_m) / 60.0,                  # initial_distance_to_adversary_m / 60
        distance_to_adversary_m / 60.0,                    # distance_to_adversary_m / 60
        estimated_distance_from_displacement_m / 60.0,     # estimated_distance_from_displacement_m / 60
        distance_progress,                                 # distance_progress
        float(ego_speed_mps) / 20.0,                       # ego_speed_mps / 20
        ego_forward_displacement_m / 60.0,                 # ego_forward_displacement_m / 60
    ]

    debug = {
        "frame_id": frame_id,
        "frame_progress": frame_progress,
        "time_seconds": time_seconds,
        "initial_distance_m": float(initial_distance_m),
        "distance_to_adversary_m": distance_to_adversary_m,
        "estimated_distance_from_displacement_m": estimated_distance_from_displacement_m,
        "distance_progress": distance_progress,
        "ego_speed_mps": float(ego_speed_mps),
        "ego_forward_displacement_m": ego_forward_displacement_m,
        "adversary_lateral_m": float(adversary_lateral_m),
        "ego_throttle": float(ego_throttle),
        "ego_spawn_index": float(ego_spawn_index),
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


def is_valid_box(box):
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

def square_expanded_crop_box(box, image_w, image_h, expand_factor):
    x1, y1, x2, y2 = [float(v) for v in box]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw = x2 - x1
    bh = y2 - y1

    side = max(bw, bh) * float(expand_factor)
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


def crop_and_resize(img, crop_box, size):
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
# Video writers
# ============================================================

def open_video_writer(path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        path,
        fourcc,
        fps,
        (int(width), int(height)),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")

    return writer


def make_2x2_comparison(original, debug_box, he_box, he_refined):
    h, w = original.shape[:2]

    panel_w = int(w * PANEL_SCALE)
    panel_h = int(h * PANEL_SCALE)
    panel_size = (panel_w, panel_h)

    p1 = cv2.resize(original, panel_size, interpolation=cv2.INTER_AREA)
    p2 = cv2.resize(debug_box, panel_size, interpolation=cv2.INTER_AREA)
    p3 = cv2.resize(he_box, panel_size, interpolation=cv2.INTER_AREA)
    p4 = cv2.resize(he_refined, panel_size, interpolation=cv2.INTER_AREA)

    p1 = put_label(p1, "Original Video")
    p2 = put_label(p2, "Predicted Box Debug")
    p3 = put_label(p3, "HE Box Predicted")
    p4 = put_label(p4, "HE Residual Refined")

    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    grid = np.vstack([top, bottom])

    return grid


# ============================================================
# Main processing
# ============================================================

def run_he_on_video(args):
    global DEVICE

    if args.device is not None:
        DEVICE = args.device

    ensure_dir(args.output_dir)

    box_model, box_meta = load_box_predictor(args.box_checkpoint)
    refiner_model, refiner_meta = load_residual_refiner(args.refiner_checkpoint)

    inserter = ObjectInserter()

    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_fps <= 0:
        source_fps = args.fps

    fps = args.fps if args.fps is not None and args.fps > 0 else source_fps

    total_frames_for_progress = source_frame_count
    if args.max_frames is not None and args.max_frames > 0:
        total_frames_for_progress = min(total_frames_for_progress, args.max_frames)

    print("[VideoHE] Input video:", args.video_path)
    print("[VideoHE] Source FPS:", source_fps)
    print("[VideoHE] Output FPS:", fps)
    print("[VideoHE] Source size:", source_width, source_height)
    print("[VideoHE] Source frames:", source_frame_count)
    print("[VideoHE] Max frames:", args.max_frames)
    print("[VideoHE] Device:", DEVICE)

    out_box_path = os.path.join(args.output_dir, "he_box_predicted.mp4")
    out_refined_path = os.path.join(args.output_dir, "he_residual_refined.mp4")
    out_debug_path = os.path.join(args.output_dir, "he_predicted_box_debug.mp4")
    out_comparison_path = os.path.join(args.output_dir, "he_video_comparison_2x2.mp4")

    writer_box = open_video_writer(
        out_box_path,
        fps,
        source_width,
        source_height,
    )

    writer_refined = open_video_writer(
        out_refined_path,
        fps,
        source_width,
        source_height,
    )

    writer_debug = open_video_writer(
        out_debug_path,
        fps,
        source_width,
        source_height,
    )

    comparison_width = int(source_width * PANEL_SCALE) * 2
    comparison_height = int(source_height * PANEL_SCALE) * 2

    writer_comparison = open_video_writer(
        out_comparison_path,
        fps,
        comparison_width,
        comparison_height,
    )

    frame_metadata = []

    frame_id = 0
    processed = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()

            if not ret:
                break

            if args.max_frames is not None and processed >= args.max_frames:
                break

            if frame_id < args.start_frame:
                frame_id += 1
                continue

            h, w = frame_bgr.shape[:2]

            # ----------------------------------------------------
            # 1. Build scalar features for this real-video frame
            # ----------------------------------------------------
            scalar_tensor, scalar_debug = build_video_scalar_features(
                frame_id=processed,
                total_frames=total_frames_for_progress,
                fps=fps,
                initial_distance_m=args.initial_distance_m,
                ego_speed_mps=args.ego_speed_mps,
                adversary_lateral_m=args.adversary_lateral_m,
                ego_throttle=args.ego_throttle,
                ego_spawn_index=args.ego_spawn_index,
            )

            # ----------------------------------------------------
            # 2. Predict box
            # ----------------------------------------------------
            image_tensor = bgr_to_tensor_resized(
                frame_bgr,
                BOX_IMAGE_SIZE,
            )

            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
            scalar_tensor = scalar_tensor.unsqueeze(0).to(DEVICE)

            pred_norm = box_model(image_tensor, scalar_tensor)[0]
            pred_norm_np = pred_norm.detach().cpu().numpy().tolist()

            pred_box = normalized_box_to_xyxy(
                pred_norm_np,
                image_w=w,
                image_h=h,
            )

            # ----------------------------------------------------
            # 3. Insert sprite
            # ----------------------------------------------------
            if pred_box is None or not is_valid_box(pred_box):
                he_box_bgr = frame_bgr.copy()
                debug_bgr = frame_bgr.copy()
                refined_bgr = he_box_bgr.copy()
                status = "invalid_box"
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                he_box_rgb = inserter.insert_sprite(
                    frame_rgb=frame_rgb,
                    sprite_path=args.sprite_path,
                    box_2d=pred_box,
                    alpha=args.sprite_alpha,
                    horizontal_scale=args.sprite_horizontal_scale,
                    vertical_scale=args.sprite_vertical_scale,
                    y_offset_ratio=args.sprite_y_offset_ratio,
                    match_brightness=True,
                    add_shadow=True,
                    soften_edges=True,
                    motion_blur=False,
                    debug_box=False,
                )

                he_box_bgr = cv2.cvtColor(he_box_rgb, cv2.COLOR_RGB2BGR)
                debug_bgr = draw_box_bgr(frame_bgr, pred_box, color=(0, 255, 0), label="pred")

                # ------------------------------------------------
                # 4. Residual refiner
                # ------------------------------------------------
                crop_box = square_expanded_crop_box(
                    box=pred_box,
                    image_w=w,
                    image_h=h,
                    expand_factor=REFINER_CROP_EXPAND_FACTOR,
                )

                if crop_box is None:
                    refined_bgr = he_box_bgr.copy()
                    status = "inserted_no_refiner_crop"
                else:
                    clean_crop = crop_and_resize(
                        frame_bgr,
                        crop_box,
                        REFINER_IMAGE_SIZE,
                    )

                    rough_crop = crop_and_resize(
                        he_box_bgr,
                        crop_box,
                        REFINER_IMAGE_SIZE,
                    )

                    mask_crop = make_mask_from_rough_difference(
                        clean_bgr=frame_bgr,
                        rough_bgr=he_box_bgr,
                        box=pred_box,
                        crop_box=crop_box,
                    )

                    if clean_crop is None or rough_crop is None or mask_crop is None:
                        refined_bgr = he_box_bgr.copy()
                        status = "inserted_no_refiner_invalid_crop"
                    else:
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

                        refined_bgr = paste_refined_crop_back(
                            original_bgr=he_box_bgr,
                            refined_crop_bgr=refined_crop_bgr,
                            mask_crop_gray=mask_crop,
                            crop_box=crop_box,
                        )

                        status = "inserted_and_residual_refined"

            # ----------------------------------------------------
            # 5. Write videos
            # ----------------------------------------------------
            writer_box.write(he_box_bgr)
            writer_refined.write(refined_bgr)
            writer_debug.write(debug_bgr)

            comparison = make_2x2_comparison(
                original=frame_bgr,
                debug_box=debug_bgr,
                he_box=he_box_bgr,
                he_refined=refined_bgr,
            )

            writer_comparison.write(comparison)

            frame_metadata.append({
                "processed_index": processed,
                "source_frame_id": frame_id,
                "pred_box_norm_cxcywh": pred_norm_np if pred_box is not None else None,
                "pred_box_2d": pred_box,
                "scalar_debug": scalar_debug,
                "status": status,
            })

            if processed % 20 == 0:
                print(
                    f"[VideoHE] frame={processed} "
                    f"dist={scalar_debug['distance_to_adversary_m']:.2f}m "
                    f"progress={scalar_debug['distance_progress']:.3f} "
                    f"box={pred_box} "
                    f"status={status}"
                )

            processed += 1
            frame_id += 1

    cap.release()
    writer_box.release()
    writer_refined.release()
    writer_debug.release()
    writer_comparison.release()

    metadata = {
        "input_video": args.video_path,
        "output_dir": args.output_dir,
        "box_checkpoint": args.box_checkpoint,
        "refiner_checkpoint": args.refiner_checkpoint,
        "sprite_path": args.sprite_path,
        "device": DEVICE,
        "source_fps": source_fps,
        "output_fps": fps,
        "source_width": source_width,
        "source_height": source_height,
        "source_frame_count": source_frame_count,
        "processed_frames": processed,
        "scenario_assumptions": {
            "initial_distance_m": args.initial_distance_m,
            "ego_speed_mps": args.ego_speed_mps,
            "adversary_lateral_m": args.adversary_lateral_m,
            "ego_throttle": args.ego_throttle,
            "ego_spawn_index": args.ego_spawn_index,
            "sprite_alpha": args.sprite_alpha,
            "sprite_horizontal_scale": args.sprite_horizontal_scale,
            "sprite_vertical_scale": args.sprite_vertical_scale,
            "sprite_y_offset_ratio": args.sprite_y_offset_ratio,
        },
        "outputs": {
            "he_box_predicted": out_box_path,
            "he_residual_refined": out_refined_path,
            "he_predicted_box_debug": out_debug_path,
            "he_video_comparison_2x2": out_comparison_path,
        },
        "frames": frame_metadata,
        "note": (
            "This is zero-shot real-video HE inference. "
            "The models were trained on CARLA stationary wrong-way adversary data, "
            "so output should be interpreted as qualitative proof-of-concept."
        ),
    }

    metadata_path = os.path.join(args.output_dir, "he_video_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("[VideoHE] Done.")
    print("[VideoHE] Output box video:", out_box_path)
    print("[VideoHE] Output residual video:", out_refined_path)
    print("[VideoHE] Output debug video:", out_debug_path)
    print("[VideoHE] Output comparison video:", out_comparison_path)
    print("[VideoHE] Metadata:", metadata_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Input road video path.",
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
        "--output_dir",
        type=str,
        default="real_video_he_outputs",
        help="Directory to save output videos.",
    )

    parser.add_argument(
        "--sprite_path",
        type=str,
        default=SPRITE_PATH_DEFAULT,
        help="Sprite path used for object insertion.",
    )

    parser.add_argument(
        "--initial_distance_m",
        type=float,
        default=40.0,
        help="Assumed initial distance to hallucinated adversary.",
    )

    parser.add_argument(
        "--ego_speed_mps",
        type=float,
        default=4.0,
        help="Assumed ego speed in m/s for distance update.",
    )

    parser.add_argument(
        "--adversary_lateral_m",
        type=float,
        default=0.0,
        help="Assumed lateral offset of hallucinated adversary.",
    )

    parser.add_argument(
        "--ego_throttle",
        type=float,
        default=0.15,
        help="Approximate ego throttle scalar for BoxPred feature compatibility.",
    )

    parser.add_argument(
        "--ego_spawn_index",
        type=float,
        default=0.0,
        help="Dummy spawn index feature for real video.",
    )

    parser.add_argument(
        "--sprite_alpha",
        type=float,
        default=1.0,
        help="Sprite alpha.",
    )

    parser.add_argument(
        "--sprite_horizontal_scale",
        type=float,
        default=1.10,
        help="Sprite horizontal scale.",
    )

    parser.add_argument(
        "--sprite_vertical_scale",
        type=float,
        default=1.15,
        help="Sprite vertical scale.",
    )

    parser.add_argument(
        "--sprite_y_offset_ratio",
        type=float,
        default=0.0,
        help="Sprite y offset ratio.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS. If omitted, input FPS is used.",
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start processing from this input frame.",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override device.",
    )

    args = parser.parse_args()

    run_he_on_video(args)


if __name__ == "__main__":
    main()