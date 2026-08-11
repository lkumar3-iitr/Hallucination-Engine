import os
import cv2
import json
import glob
import argparse
import numpy as np

import torch
import torch.nn as nn


# ============================================================
# Settings
# ============================================================

PAIRED_BASE_DIR = "paired_data"
CHECKPOINT_BASE_DIR = "refiner_checkpoints"

IMAGE_SIZE = 256
CROP_EXPAND_FACTOR = 2.0

DIFF_THRESHOLD = 12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def find_latest_checkpoint(base_dir=CHECKPOINT_BASE_DIR):
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Checkpoint base not found: {base_dir}")

    ckpts = glob.glob(
        os.path.join(base_dir, "local_refiner_*", "checkpoints", "*.pth")
    )

    if not ckpts:
        raise RuntimeError(f"No checkpoint files found in: {base_dir}")

    ckpts = sorted(ckpts, key=os.path.getmtime, reverse=True)
    return ckpts[0]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_bgr(path):
    img = cv2.imread(path)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    return img


def save_bgr(path, img):
    cv2.imwrite(path, img)


def bgr_to_tensor(img_bgr):
    """
    BGR uint8 HWC -> RGB float tensor CHW in [0, 1]
    """

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)

    return tensor


def tensor_to_bgr(tensor):
    """
    RGB tensor CHW [0, 1] -> BGR uint8 HWC
    """

    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    rgb = tensor.permute(1, 2, 0).numpy()
    rgb = (rgb * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr


def mask_to_tensor(mask_gray):
    """
    Gray uint8 HxW -> 1xHxW tensor [0,1]
    """

    if len(mask_gray.shape) == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)

    mask = mask_gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(mask).unsqueeze(0)

    return tensor


# ============================================================
# Model definition
# Must match train_local_refiner.py
# ============================================================

class ConvBlock(nn.Module):
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


class TinyUNetRefiner(nn.Module):
    """
    Same architecture as train_local_refiner.py
    """

    def __init__(self, in_ch=7, out_ch=3, base_ch=32):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)

        self.enc2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)

        self.mid = ConvBlock(base_ch * 4, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

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

        out = torch.sigmoid(self.out(d1))
        return out


# ============================================================
# Crop and mask functions
# Same logic as dataset builder
# ============================================================

def is_valid_box(box):
    if box is None:
        return False

    if len(box) != 4:
        return False

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
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


def crop_and_resize(img, crop_box, size=IMAGE_SIZE):
    x1, y1, x2, y2 = crop_box

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    return crop


def make_mask_from_rough_difference(clean_bgr, rough_bgr, box, crop_box):
    """
    Create mask from difference between rough HE image and clean image,
    constrained near the real/adversary bbox.
    """

    diff = cv2.absdiff(clean_bgr, rough_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    mask = (diff_gray > DIFF_THRESHOLD).astype(np.uint8) * 255

    h, w = mask.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in box]

    bw = x2 - x1
    bh = y2 - y1

    margin_x = int(0.20 * bw)
    margin_y = int(0.20 * bh)

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

    mask_crop = crop_and_resize(mask, crop_box, IMAGE_SIZE)

    if mask_crop is None:
        return None

    if len(mask_crop.shape) == 3:
        mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_BGR2GRAY)

    mask_crop = (mask_crop > 30).astype(np.uint8) * 255

    return mask_crop


def paste_refined_crop_back(
    original_bgr,
    refined_crop_bgr,
    mask_crop_gray,
    crop_box,
):
    """
    Resize refined crop back to crop_box and alpha-blend into full frame.
    """

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

    # Soften mask for smoother paste.
    mask_resized = cv2.GaussianBlur(mask_resized, (9, 9), 0)
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
# Metadata helpers
# ============================================================

def find_real_meta_by_frame(real_sequence, frame_id):
    for item in real_sequence:
        if int(item.get("frame", -1)) == int(frame_id):
            return item

    return None


def create_preview_row(clean_bgr, real_bgr, rough_bgr, refined_bgr):
    """
    Side-by-side row:
        clean | real | rough | refined
    """

    panels = [
        ("clean", clean_bgr),
        ("real", real_bgr),
        ("rough", rough_bgr),
        ("refined", refined_bgr),
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
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labeled.append(panel)

    return np.hstack(labeled)


# ============================================================
# Load model
# ============================================================

def load_refiner_model(checkpoint_path):
    print("[ApplyRefiner] Loading checkpoint:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    train_meta = checkpoint.get("train_meta", {})
    base_channels = int(train_meta.get("base_channels", 32))

    model = TinyUNetRefiner(
        in_ch=7,
        out_ch=3,
        base_ch=base_channels,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[ApplyRefiner] Loaded model.")
    print("[ApplyRefiner] Base channels:", base_channels)
    print("[ApplyRefiner] Device:", DEVICE)

    return model, train_meta


# ============================================================
# Main apply function
# ============================================================

def apply_refiner_to_pair_run(pair_run_dir, checkpoint_path):
    model, train_meta = load_refiner_model(checkpoint_path)

    metadata_path = os.path.join(pair_run_dir, "metadata", "sequence_metadata.json")

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Missing sequence metadata: {metadata_path}")

    sequence_metadata = load_json(metadata_path)
    real_sequence = sequence_metadata.get("real_sequence", [])

    clean_dir = os.path.join(pair_run_dir, "clean")
    real_dir = os.path.join(pair_run_dir, "real_object")
    rough_dir = os.path.join(pair_run_dir, "he_oracle_box")

    refined_dir = os.path.join(pair_run_dir, "he_oracle_refined")
    preview_dir = os.path.join(pair_run_dir, "he_oracle_refined_preview")
    refined_meta_dir = os.path.join(pair_run_dir, "metadata")

    os.makedirs(refined_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    if not clean_paths:
        raise RuntimeError(f"No clean frames found: {clean_dir}")

    refined_metadata = []

    print("[ApplyRefiner] Pair run:", pair_run_dir)
    print("[ApplyRefiner] Frames:", len(clean_paths))
    print("[ApplyRefiner] Output:", refined_dir)

    with torch.no_grad():
        for idx, clean_path in enumerate(clean_paths):
            frame_name = os.path.basename(clean_path)
            frame_id = int(os.path.splitext(frame_name)[0].replace("frame_", ""))

            real_meta = find_real_meta_by_frame(real_sequence, frame_id)

            if real_meta is None:
                continue

            box = real_meta.get("adversary_box_2d", None)

            if not is_valid_box(box):
                # Save rough or clean fallback if no valid box.
                rough_fallback_path = os.path.join(rough_dir, frame_name)

                if os.path.exists(rough_fallback_path):
                    fallback = read_bgr(rough_fallback_path)
                else:
                    fallback = read_bgr(clean_path)

                save_bgr(os.path.join(refined_dir, frame_name), fallback)

                refined_metadata.append({
                    "frame": frame_id,
                    "status": "no_valid_box",
                    "box_2d": box,
                })
                continue

            rough_path = os.path.join(rough_dir, frame_name)
            real_path = os.path.join(real_dir, frame_name)

            if not os.path.exists(rough_path):
                continue

            if not os.path.exists(real_path):
                continue

            clean_bgr = read_bgr(clean_path)
            rough_bgr = read_bgr(rough_path)
            real_bgr = read_bgr(real_path)

            h, w = clean_bgr.shape[:2]

            crop_box = square_expanded_crop_box(
                box=box,
                image_w=w,
                image_h=h,
                expand_factor=CROP_EXPAND_FACTOR,
            )

            if crop_box is None:
                save_bgr(os.path.join(refined_dir, frame_name), rough_bgr)
                continue

            clean_crop = crop_and_resize(clean_bgr, crop_box, IMAGE_SIZE)
            rough_crop = crop_and_resize(rough_bgr, crop_box, IMAGE_SIZE)

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

            pred = model(inp)[0]

            refined_crop_bgr = tensor_to_bgr(pred)

            refined_full_bgr = paste_refined_crop_back(
                original_bgr=rough_bgr,
                refined_crop_bgr=refined_crop_bgr,
                mask_crop_gray=mask_crop,
                crop_box=crop_box,
            )

            refined_path = os.path.join(refined_dir, frame_name)
            save_bgr(refined_path, refined_full_bgr)

            refined_metadata.append({
                "frame": frame_id,
                "status": "refined",
                "box_2d": box,
                "crop_box": crop_box,
                "refined_frame_path": refined_path,
            })

            # Save a few preview rows.
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
                print(f"[ApplyRefiner] Processed frame {idx}/{len(clean_paths)}")

    refined_meta = {
        "pair_run_dir": pair_run_dir,
        "checkpoint_path": checkpoint_path,
        "device": DEVICE,
        "image_size": IMAGE_SIZE,
        "crop_expand_factor": CROP_EXPAND_FACTOR,
        "num_frames_processed": len(refined_metadata),
        "frames": refined_metadata,
    }

    refined_meta_path = os.path.join(
        refined_meta_dir,
        "he_oracle_refined_metadata.json",
    )

    with open(refined_meta_path, "w", encoding="utf-8") as f:
        json.dump(refined_meta, f, indent=4)

    print("[ApplyRefiner] Done.")
    print("[ApplyRefiner] Refined frames:", refined_dir)
    print("[ApplyRefiner] Metadata:", refined_meta_path)


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
        help="Path to local refiner checkpoint. If omitted, latest checkpoint is used.",
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
        checkpoint_path = find_latest_checkpoint(CHECKPOINT_BASE_DIR)
    else:
        checkpoint_path = args.checkpoint

    apply_refiner_to_pair_run(
        pair_run_dir=pair_run_dir,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()