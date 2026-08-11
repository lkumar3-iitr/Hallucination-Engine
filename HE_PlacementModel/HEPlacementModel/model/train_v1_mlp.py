import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class HEPlacementDataset(Dataset):
    def __init__(self, labels_path):
        self.rows = []

        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

        if len(self.rows) == 0:
            raise RuntimeError(f"No rows found in {labels_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        rel = row["relative_state"]
        cam = row["camera"]
        target = row["target"]

        width = float(cam["width"])
        height = float(cam["height"])

        # Input features.
        # Keep this fixed because runtime must use the same order.
        x = np.array(
            [
                float(rel["rel_x"]),
                float(rel["rel_z"]),
                float(rel["rel_y"]),
                float(rel["rel_yaw"]),
                float(cam["fx"]),
                float(cam["fy"]),
                float(cam["cx"]),
                float(cam["cy"]),
                width,
                height,
                float(cam["fov"]),
                float(cam["mount_x"]),
                float(cam["mount_y"]),
                float(cam["mount_z"]),
                float(cam["pitch"]),
                float(cam["yaw"]),
                float(cam["roll"]),
            ],
            dtype=np.float32,
        )

        visible = float(target["visible"])

        # Normalize bbox targets.
        y_bbox = np.array(
            [
                float(target["center_x"]) / width,
                float(target["bottom_y"]) / height,
                float(target["box_width"]) / width,
                float(target["box_height"]) / height,
            ],
            dtype=np.float32,
        )

        y_visible = np.array([visible], dtype=np.float32)

        return {
            "x": torch.from_numpy(x),
            "bbox": torch.from_numpy(y_bbox),
            "visible": torch.from_numpy(y_visible),
            "image_width": torch.tensor(width, dtype=torch.float32),
            "image_height": torch.tensor(height, dtype=torch.float32),
        }


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        xs = []
        for i in range(len(dataset)):
            xs.append(dataset[i]["x"].numpy())
        xs = np.stack(xs, axis=0)

        self.mean = xs.mean(axis=0).astype(np.float32)
        self.std = xs.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x):
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


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


def masked_smooth_l1(pred_bbox, gt_bbox, visible):
    """
    Only visible samples contribute to bbox regression loss.
    Invisible samples should not teach meaningless bbox coordinates.
    """
    visible_mask = visible.view(-1, 1)

    if visible_mask.sum() < 1:
        return torch.tensor(0.0, device=pred_bbox.device)

    loss = nn.functional.smooth_l1_loss(
        pred_bbox * visible_mask,
        gt_bbox * visible_mask,
        reduction="sum",
    )

    loss = loss / visible_mask.sum().clamp(min=1.0)

    return loss


def evaluate(model, loader, normalizer, device):
    model.eval()

    total = 0
    visible_correct = 0

    bbox_abs_errors_px = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            gt_bbox = batch["bbox"].to(device)
            gt_visible = batch["visible"].to(device)

            width = batch["image_width"].to(device).view(-1, 1)
            height = batch["image_height"].to(device).view(-1, 1)

            x_norm = normalizer.transform(x)
            pred_bbox, pred_visible_logit = model(x_norm)

            pred_visible = (torch.sigmoid(pred_visible_logit) >= 0.5).float()
            visible_correct += (pred_visible == gt_visible).float().sum().item()
            total += x.shape[0]

            visible_mask = gt_visible.view(-1) > 0.5

            if visible_mask.any():
                pred_px = pred_bbox[visible_mask].clone()
                gt_px = gt_bbox[visible_mask].clone()

                w = width[visible_mask]
                h = height[visible_mask]

                pred_px[:, 0] *= w[:, 0]
                pred_px[:, 1] *= h[:, 0]
                pred_px[:, 2] *= w[:, 0]
                pred_px[:, 3] *= h[:, 0]

                gt_px[:, 0] *= w[:, 0]
                gt_px[:, 1] *= h[:, 0]
                gt_px[:, 2] *= w[:, 0]
                gt_px[:, 3] *= h[:, 0]

                abs_err = torch.abs(pred_px - gt_px)
                bbox_abs_errors_px.append(abs_err.cpu())

    visibility_acc = visible_correct / max(total, 1)

    if bbox_abs_errors_px:
        bbox_abs_errors_px = torch.cat(bbox_abs_errors_px, dim=0)
        mae = bbox_abs_errors_px.mean(dim=0).numpy()
    else:
        mae = np.array([0, 0, 0, 0], dtype=np.float32)

    return {
        "visibility_acc": visibility_acc,
        "mae_center_x_px": float(mae[0]),
        "mae_bottom_y_px": float(mae[1]),
        "mae_box_width_px": float(mae[2]),
        "mae_box_height_px": float(mae[3]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="dataset/v1_straight/labels.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/v1_mlp")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    dataset = HEPlacementDataset(args.labels)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Fit normalizer on full dataset for simplicity in v1.
    # Later we can fit only on train split.
    normalizer = Normalizer()
    normalizer.fit(dataset)

    sample = dataset[0]
    input_dim = sample["x"].shape[0]

    model = PlacementMLP(input_dim=input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    best_score = float("inf")
    best_path = os.path.join(args.output_dir, "heplacement_v1_mlp_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_bbox_loss = 0.0
        total_vis_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            gt_bbox = batch["bbox"].to(device)
            gt_visible = batch["visible"].to(device)

            x_norm = normalizer.transform(x)

            pred_bbox, pred_visible_logit = model(x_norm)

            bbox_loss = masked_smooth_l1(pred_bbox, gt_bbox, gt_visible)
            vis_loss = bce_loss(pred_visible_logit, gt_visible)

            loss = bbox_loss + 0.5 * vis_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            total_vis_loss += vis_loss.item()
            num_batches += 1

        metrics = evaluate(model, val_loader, normalizer, device)

        score = (
            metrics["mae_center_x_px"]
            + metrics["mae_bottom_y_px"]
            + metrics["mae_box_width_px"]
            + metrics["mae_box_height_px"]
        )

        print(
            f"[EPOCH {epoch:03d}] "
            f"loss={total_loss / max(num_batches, 1):.5f} "
            f"bbox_loss={total_bbox_loss / max(num_batches, 1):.5f} "
            f"vis_loss={total_vis_loss / max(num_batches, 1):.5f} "
            f"val_vis_acc={metrics['visibility_acc']:.4f} "
            f"MAE_px=("
            f"cx={metrics['mae_center_x_px']:.2f}, "
            f"bottom={metrics['mae_bottom_y_px']:.2f}, "
            f"w={metrics['mae_box_width_px']:.2f}, "
            f"h={metrics['mae_box_height_px']:.2f})"
        )

        if score < best_score:
            best_score = score

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "input_order": [
                    "rel_x",
                    "rel_z",
                    "rel_y",
                    "rel_yaw",
                    "fx",
                    "fy",
                    "cx",
                    "cy",
                    "image_width",
                    "image_height",
                    "fov",
                    "mount_x",
                    "mount_y",
                    "mount_z",
                    "camera_pitch",
                    "camera_yaw",
                    "camera_roll",
                ],
                "normalizer": normalizer.to_dict(),
                "target_order": [
                    "center_x_norm",
                    "bottom_y_norm",
                    "box_width_norm",
                    "box_height_norm",
                ],
            }

            torch.save(checkpoint, best_path)

    print(f"[DONE] Best model saved to: {best_path}")


if __name__ == "__main__":
    main()