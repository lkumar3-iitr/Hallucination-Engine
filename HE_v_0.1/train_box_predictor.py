import os
import csv
import json
import time
import glob
import argparse
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Settings
# ============================================================

BOX_DATASET_BASE = "box_dataset"
OUTPUT_BASE = "box_predictor_checkpoints"

IMAGE_SIZE = 224

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

SAVE_EVERY_EPOCHS = 5
PREVIEW_EVERY_EPOCHS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALAR_DIM = 12


# ============================================================
# Utilities
# ============================================================

def find_latest_box_run(base_dir=BOX_DATASET_BASE):
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Box dataset base not found: {base_dir}")

    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("box_run_")
        and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not runs:
        raise RuntimeError(f"No box_run folders found in: {base_dir}")

    runs = sorted(runs, key=os.path.getmtime, reverse=True)
    return runs[0]


def create_output_dir():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    out_dir = os.path.abspath(
        os.path.join(OUTPUT_BASE, f"box_predictor_v2_{timestamp}")
    )

    dirs = {
        "root": out_dir,
        "checkpoints": os.path.join(out_dir, "checkpoints"),
        "previews": os.path.join(out_dir, "previews"),
        "metadata": os.path.join(out_dir, "metadata"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"[TrainBoxV2] Created {name}: {path}")

    return dirs


def read_labels_csv(csv_path):
    rows = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


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


def bgr_to_tensor_resized(img_bgr, size=IMAGE_SIZE):
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)

    return tensor


def denormalize_box_norm(box_norm, image_w, image_h):
    """
    Convert normalized cx,cy,w,h to pixel x1,y1,x2,y2.
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

    return [x1, y1, x2, y2]


def draw_box(img_bgr, box_norm, color, label=None):
    output = img_bgr.copy()
    h, w = output.shape[:2]

    x1, y1, x2, y2 = denormalize_box_norm(box_norm, w, h)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    if label is not None:
        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def build_distance_aware_scalar_features(row):
    """
    Must match apply_box_predictor.py v2 later.

    12 features:
        0 frame_progress
        1 adversary_forward_m / 60
        2 adversary_lateral_m / 3
        3 ego_throttle
        4 ego_spawn_index / 300
        5 time_seconds / 20
        6 initial_distance_to_adversary_m / 60
        7 distance_to_adversary_m / 60
        8 estimated_distance_from_displacement_m / 60
        9 distance_progress
        10 ego_speed_mps / 20
        11 ego_forward_displacement_m / 60
    """

    frame_progress = safe_float(row.get("frame_progress", 0.0), 0.0)

    adversary_forward_m = safe_float(
        row.get("adversary_forward_m", 0.0),
        0.0,
    )

    adversary_lateral_m = safe_float(
        row.get("adversary_lateral_m", 0.0),
        0.0,
    )

    ego_throttle = safe_float(
        row.get("ego_throttle", 0.0),
        0.0,
    )

    ego_spawn_index = safe_float(
        row.get("ego_spawn_index", 0.0),
        0.0,
    )

    time_seconds = safe_float(
        row.get("time_seconds", 0.0),
        0.0,
    )

    initial_distance_to_adversary_m = safe_float(
        row.get("initial_distance_to_adversary_m", None),
        adversary_forward_m,
    )

    distance_to_adversary_m = safe_float(
        row.get("distance_to_adversary_m", None),
        initial_distance_to_adversary_m,
    )

    estimated_distance_from_displacement_m = safe_float(
        row.get("estimated_distance_from_displacement_m", None),
        distance_to_adversary_m,
    )

    distance_progress = safe_float(
        row.get("distance_progress", None),
        0.0,
    )

    ego_speed_mps = safe_float(
        row.get("ego_speed_mps", 0.0),
        0.0,
    )

    ego_forward_displacement_m = safe_float(
        row.get("ego_forward_displacement_m", 0.0),
        0.0,
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

    return scalar_features


# ============================================================
# Dataset
# ============================================================

class BoxPredictionDataset(Dataset):
    """
    Dataset for stationary wrong-way adversary box prediction.

    Input:
        clean RGB frame
        distance-aware scenario scalar vector

    Target:
        normalized box [cx, cy, w, h]
    """

    def __init__(self, box_run_dir):
        self.box_run_dir = box_run_dir
        self.labels_path = os.path.join(box_run_dir, "labels.csv")

        if not os.path.exists(self.labels_path):
            raise RuntimeError(f"Missing labels.csv: {self.labels_path}")

        rows = read_labels_csv(self.labels_path)

        if not rows:
            raise RuntimeError("No labels found.")

        self.samples = []

        for row in rows:
            image_path = row.get("image_path", "")

            if not os.path.exists(image_path):
                continue

            cx = safe_float(row.get("cx", None), None)
            cy = safe_float(row.get("cy", None), None)
            bw = safe_float(row.get("w", None), None)
            bh = safe_float(row.get("h", None), None)

            if cx is None or cy is None or bw is None or bh is None:
                continue

            if bw <= 0 or bh <= 0:
                continue

            scalar_features = build_distance_aware_scalar_features(row)

            if len(scalar_features) != SCALAR_DIM:
                raise RuntimeError(
                    f"Expected {SCALAR_DIM} scalar features, got {len(scalar_features)}"
                )

            self.samples.append({
                "image_path": image_path,
                "target_box": [cx, cy, bw, bh],
                "scalar_features": scalar_features,
                "frame_id": safe_int(row.get("frame_id", -1), -1),
                "sample_id": safe_int(row.get("sample_id", -1), -1),
                "source_run_name": row.get("source_run_name", ""),
                "distance_to_adversary_m": safe_float(row.get("distance_to_adversary_m", 0.0), 0.0),
                "ego_speed_mps": safe_float(row.get("ego_speed_mps", 0.0), 0.0),
            })

        if not self.samples:
            raise RuntimeError("No valid samples found after filtering.")

        print("[TrainBoxV2] Dataset:", box_run_dir)
        print("[TrainBoxV2] Valid samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img_bgr = cv2.imread(item["image_path"])

        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {item['image_path']}")

        image_tensor = bgr_to_tensor_resized(img_bgr, IMAGE_SIZE)

        scalar_tensor = torch.tensor(
            item["scalar_features"],
            dtype=torch.float32,
        )

        target_tensor = torch.tensor(
            item["target_box"],
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "scalars": scalar_tensor,
            "target": target_tensor,
            "image_path": item["image_path"],
            "frame_id": item["frame_id"],
            "sample_id": item["sample_id"],
            "source_run_name": item["source_run_name"],
            "distance_to_adversary_m": item["distance_to_adversary_m"],
            "ego_speed_mps": item["ego_speed_mps"],
        }


# ============================================================
# Model
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
# Loss and metrics
# ============================================================

def box_l1_loss(pred, target):
    """
    Weighted L1 loss for cx, cy, w, h.
    """

    weights = torch.tensor(
        [1.0, 1.0, 2.5, 2.5],
        dtype=torch.float32,
        device=pred.device,
    )

    loss = torch.abs(pred - target) * weights
    return loss.mean()


def size_growth_loss(pred, target):
    """
    Extra penalty on width/height because scale growth is critical.
    """

    pred_size = pred[:, 2:4]
    target_size = target[:, 2:4]

    return torch.abs(pred_size - target_size).mean()


def box_iou_norm(pred, target):
    pred_cx, pred_cy, pred_w, pred_h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tgt_cx, tgt_cy, tgt_w, tgt_h = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    pred_x1 = pred_cx - pred_w / 2.0
    pred_y1 = pred_cy - pred_h / 2.0
    pred_x2 = pred_cx + pred_w / 2.0
    pred_y2 = pred_cy + pred_h / 2.0

    tgt_x1 = tgt_cx - tgt_w / 2.0
    tgt_y1 = tgt_cy - tgt_h / 2.0
    tgt_x2 = tgt_cx + tgt_w / 2.0
    tgt_y2 = tgt_cy + tgt_h / 2.0

    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)

    inter = inter_w * inter_h

    pred_area = torch.clamp(pred_w, min=0.0) * torch.clamp(pred_h, min=0.0)
    tgt_area = torch.clamp(tgt_w, min=0.0) * torch.clamp(tgt_h, min=0.0)

    union = pred_area + tgt_area - inter + 1e-6
    iou = inter / union

    return iou.mean()


# ============================================================
# Preview
# ============================================================

def save_preview(batch, pred, out_path, max_items=8):
    """
    Save preview image:
        red = target box
        green = predicted box
    """

    n = min(pred.shape[0], max_items)

    panels = []

    for i in range(n):
        image_path = batch["image_path"][i]
        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)

        target_box = batch["target"][i].detach().cpu().numpy()
        pred_box = pred[i].detach().cpu().numpy()

        vis = img.copy()
        vis = draw_box(vis, target_box, color=(0, 0, 255), label="target")
        vis = draw_box(vis, pred_box, color=(0, 255, 0), label="pred")

        frame_id = batch["frame_id"][i].item() if torch.is_tensor(batch["frame_id"][i]) else batch["frame_id"][i]
        sample_id = batch["sample_id"][i].item() if torch.is_tensor(batch["sample_id"][i]) else batch["sample_id"][i]

        distance_m = batch["distance_to_adversary_m"][i]
        if torch.is_tensor(distance_m):
            distance_m = distance_m.item()

        ego_speed_mps = batch["ego_speed_mps"][i]
        if torch.is_tensor(ego_speed_mps):
            ego_speed_mps = ego_speed_mps.item()

        cv2.putText(
            vis,
            f"sample={sample_id} frame={frame_id} dist={distance_m:.1f}m speed={ego_speed_mps:.1f}mps",
            (10, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        panels.append(vis)

    if not panels:
        return

    rows = []
    cols = 2

    for i in range(0, len(panels), cols):
        row = panels[i:i + cols]

        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))

        rows.append(np.hstack(row))

    grid = np.vstack(rows)
    cv2.imwrite(out_path, grid)


# ============================================================
# Training / evaluation
# ============================================================

def evaluate(model, dataloader):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(DEVICE)
            scalars = batch["scalars"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            pred = model(image, scalars)

            loss = box_l1_loss(pred, target) + 0.5 * size_growth_loss(pred, target)
            iou = box_iou_norm(pred, target)

            total_loss += loss.item()
            total_iou += iou.item()
            count += 1

    return {
        "loss": total_loss / max(1, count),
        "iou": total_iou / max(1, count),
    }


def train(args):
    dataset_dir = args.dataset_dir

    if dataset_dir is None:
        dataset_dir = find_latest_box_run(BOX_DATASET_BASE)

    output_dirs = create_output_dir()

    dataset = BoxPredictionDataset(dataset_dir)

    val_count = max(1, int(len(dataset) * args.val_split))
    train_count = len(dataset) - val_count

    train_dataset, val_dataset = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    model = BoxPredictor(
        scalar_dim=SCALAR_DIM,
        hidden_dim=args.hidden_dim,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    train_meta = {
        "model_version": "box_predictor_v2_distance_aware",
        "dataset_dir": dataset_dir,
        "device": DEVICE,
        "image_size": IMAGE_SIZE,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_samples": len(dataset),
        "train_count": train_count,
        "val_count": val_count,
        "scalar_dim": SCALAR_DIM,
        "scalar_features": [
            "frame_progress",
            "adversary_forward_m/60",
            "adversary_lateral_m/3",
            "ego_throttle",
            "ego_spawn_index/300",
            "time_seconds/20",
            "initial_distance_to_adversary_m/60",
            "distance_to_adversary_m/60",
            "estimated_distance_from_displacement_m/60",
            "distance_progress",
            "ego_speed_mps/20",
            "ego_forward_displacement_m/60",
        ],
        "target": "normalized cx,cy,w,h",
    }

    meta_path = os.path.join(output_dirs["metadata"], "train_metadata.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=4)

    print("[TrainBoxV2] Device:", DEVICE)
    print("[TrainBoxV2] Metadata:", meta_path)

    best_val_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_iou = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            image = batch["image"].to(DEVICE)
            scalars = batch["scalars"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            pred = model(image, scalars)

            loss = box_l1_loss(pred, target) + 0.5 * size_growth_loss(pred, target)
            iou = box_iou_norm(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_iou += iou.item()

            if batch_idx % 10 == 0:
                print(
                    f"[TrainBoxV2] "
                    f"epoch {epoch}/{args.epochs} "
                    f"batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.5f} "
                    f"iou={iou.item():.3f}"
                )

        train_loss = epoch_loss / max(1, len(train_loader))
        train_iou = epoch_iou / max(1, len(train_loader))

        val_metrics = evaluate(model, val_loader)

        elapsed = time.time() - start_time

        print(
            f"[TrainBoxV2] Epoch {epoch} done | "
            f"train_loss={train_loss:.5f} "
            f"train_iou={train_iou:.3f} | "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_iou={val_metrics['iou']:.3f} | "
            f"time={elapsed:.1f}s"
        )

        if epoch % PREVIEW_EVERY_EPOCHS == 0:
            model.eval()

            with torch.no_grad():
                preview_batch = next(iter(val_loader))
                preview_image = preview_batch["image"].to(DEVICE)
                preview_scalars = preview_batch["scalars"].to(DEVICE)
                preview_pred = model(preview_image, preview_scalars).cpu()

            preview_path = os.path.join(
                output_dirs["previews"],
                f"preview_epoch_{epoch:03d}.png",
            )

            save_preview(
                batch=preview_batch,
                pred=preview_pred,
                out_path=preview_path,
            )

            print("[TrainBoxV2] Saved preview:", preview_path)

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]

            best_path = os.path.join(
                output_dirs["checkpoints"],
                "box_predictor_best.pth",
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_meta": train_meta,
                "val_metrics": val_metrics,
            }, best_path)

            print("[TrainBoxV2] Saved best checkpoint:", best_path)

        if epoch % SAVE_EVERY_EPOCHS == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                output_dirs["checkpoints"],
                f"box_predictor_epoch_{epoch:03d}.pth",
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_meta": train_meta,
                "val_metrics": val_metrics,
            }, ckpt_path)

            print("[TrainBoxV2] Saved checkpoint:", ckpt_path)

    print("[TrainBoxV2] Training complete.")
    print("[TrainBoxV2] Best val IoU:", best_val_iou)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to box_run folder. If omitted, latest box_run is used.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
    )

    args = parser.parse_args()

    global DEVICE

    if args.device is not None:
        DEVICE = args.device

    train(args)


if __name__ == "__main__":
    main()