import os
import json
import glob
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Settings
# ============================================================

REFINER_DATASET_BASE = "refiner_dataset"
OUTPUT_BASE = "refiner_checkpoints"

IMAGE_SIZE = 384

BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

SAVE_EVERY_EPOCHS = 5
PREVIEW_EVERY_EPOCHS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Maximum correction applied to rough crop.
# 0.25 means residual can change pixel values by +/-25% in normalized [0,1] space.
MAX_RESIDUAL_DELTA = 0.25


# ============================================================
# Utilities
# ============================================================

def find_latest_refiner_run(base_dir=REFINER_DATASET_BASE):
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Refiner dataset base not found: {base_dir}")

    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if (
            d.startswith("refiner_run_")
            or d.startswith("refiner_combined_")
        )
        and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not runs:
        raise RuntimeError(f"No refiner dataset folders found in: {base_dir}")

    runs = sorted(runs, key=os.path.getmtime, reverse=True)
    return runs[0]


def create_output_dir():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    out_dir = os.path.abspath(
        os.path.join(OUTPUT_BASE, f"local_refiner_residual_{timestamp}")
    )

    dirs = {
        "root": out_dir,
        "checkpoints": os.path.join(out_dir, "checkpoints"),
        "previews": os.path.join(out_dir, "previews"),
        "metadata": os.path.join(out_dir, "metadata"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"[TrainResidualRefiner] Created {name}: {path}")

    return dirs


def read_bgr(path):
    img = cv2.imread(path)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    return img


def bgr_to_tensor(img_bgr):
    """
    BGR uint8 HWC -> RGB float tensor CHW in [0,1]
    """

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)

    return tensor


def mask_to_tensor(mask_bgr_or_gray):
    """
    mask uint8 -> 1xHxW float tensor in [0,1]
    """

    if len(mask_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(mask_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_bgr_or_gray

    gray = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray).unsqueeze(0)

    return tensor


def tensor_to_bgr(tensor):
    """
    RGB tensor CHW [0,1] -> BGR uint8 HWC
    """

    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    rgb = tensor.permute(1, 2, 0).numpy()
    rgb = (rgb * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr


# ============================================================
# Dataset
# ============================================================

class RefinerCropDataset(Dataset):
    """
    Dataset for residual local refiner.

    Input:
        clean_crop : 3 channels
        rough_crop : 3 channels
        mask_crop  : 1 channel

    Total input:
        7 channels

    Target:
        target_crop : 3 channels

    Model predicts:
        residual correction to rough_crop
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        self.clean_dir = os.path.join(dataset_dir, "clean_crop")
        self.rough_dir = os.path.join(dataset_dir, "rough_crop")
        self.target_dir = os.path.join(dataset_dir, "target_crop")
        self.mask_dir = os.path.join(dataset_dir, "mask_crop")

        for d in [self.clean_dir, self.rough_dir, self.target_dir, self.mask_dir]:
            if not os.path.isdir(d):
                raise RuntimeError(f"Missing dataset folder: {d}")

        clean_paths = sorted(glob.glob(os.path.join(self.clean_dir, "*.png")))

        if not clean_paths:
            raise RuntimeError(f"No samples found in: {self.clean_dir}")

        self.samples = []

        for clean_path in clean_paths:
            name = os.path.basename(clean_path)

            rough_path = os.path.join(self.rough_dir, name)
            target_path = os.path.join(self.target_dir, name)
            mask_path = os.path.join(self.mask_dir, name)

            if (
                os.path.exists(rough_path)
                and os.path.exists(target_path)
                and os.path.exists(mask_path)
            ):
                self.samples.append({
                    "name": name,
                    "clean": clean_path,
                    "rough": rough_path,
                    "target": target_path,
                    "mask": mask_path,
                })

        if not self.samples:
            raise RuntimeError("No valid matched samples found.")

        print("[TrainResidualRefiner] Dataset:", dataset_dir)
        print("[TrainResidualRefiner] Valid samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        clean_bgr = read_bgr(item["clean"])
        rough_bgr = read_bgr(item["rough"])
        target_bgr = read_bgr(item["target"])
        mask_img = read_bgr(item["mask"])

        clean = bgr_to_tensor(clean_bgr)
        rough = bgr_to_tensor(rough_bgr)
        target = bgr_to_tensor(target_bgr)
        mask = mask_to_tensor(mask_img)

        inp = torch.cat([clean, rough, mask], dim=0)

        return {
            "input": inp,
            "clean": clean,
            "rough": rough,
            "target": target,
            "mask": mask,
            "name": item["name"],
        }


# ============================================================
# Model
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


class ResidualUNetRefiner(nn.Module):
    """
    Small U-Net that predicts residual correction, not full RGB.

    Input:
        7 channels:
            clean RGB
            rough RGB
            mask

    Output:
        residual RGB correction in [-MAX_RESIDUAL_DELTA, +MAX_RESIDUAL_DELTA]
    """

    def __init__(self, in_ch=7, out_ch=3, base_ch=32, max_delta=MAX_RESIDUAL_DELTA):
        super().__init__()

        self.max_delta = float(max_delta)

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

        residual = torch.tanh(self.out(d1)) * self.max_delta

        return residual


# ============================================================
# Loss
# ============================================================

def image_gradient(x):
    """
    Simple image gradient for edge preservation.
    x: BxCxHxW
    """

    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

    return dx, dy


def gradient_loss(pred, target, mask):
    """
    Edge/detail loss mostly around object region.
    """

    mask_dilated = F.max_pool2d(
        mask,
        kernel_size=15,
        stride=1,
        padding=7,
    )

    pred_dx, pred_dy = image_gradient(pred)
    tgt_dx, tgt_dy = image_gradient(target)

    mask_dx = mask_dilated[:, :, :, 1:]
    mask_dy = mask_dilated[:, :, 1:, :]

    loss_dx = (torch.abs(pred_dx - tgt_dx) * (1.0 + 3.0 * mask_dx)).mean()
    loss_dy = (torch.abs(pred_dy - tgt_dy) * (1.0 + 3.0 * mask_dy)).mean()

    return loss_dx + loss_dy


def residual_refiner_loss(pred, target, rough, mask):
    """
    Loss for residual refiner.

    Important:
        - Object region should move toward real target.
        - Background should stay almost exactly rough.
        - Edges should remain sharp.
    """

    mask_dilated = F.max_pool2d(
        mask,
        kernel_size=21,
        stride=1,
        padding=10,
    )

    base_l1 = torch.abs(pred - target)

    # Stronger weight on object region.
    weight = 1.0 + 5.0 * mask_dilated
    weighted_l1 = (base_l1 * weight).mean()

    # Preserve background from rough crop.
    outside = 1.0 - mask_dilated
    background_loss = (torch.abs(pred - rough) * outside).mean()

    # Preserve sharpness/edges.
    edge_loss = gradient_loss(pred, target, mask)

    loss = weighted_l1 + 1.0 * background_loss + 0.25 * edge_loss

    return loss


def apply_residual(rough, residual, mask):
    """
    Apply residual only near object region.

    This is the key to preserving sharpness/background.
    """

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
# Preview
# ============================================================

def save_preview(batch, refined, residual, out_path, max_items=4):
    """
    Save preview grid:
        clean | rough | refined | target | residual
    """

    clean = batch["clean"]
    rough = batch["rough"]
    target = batch["target"]
    mask = batch["mask"]

    n = min(clean.shape[0], max_items)

    rows = []

    for i in range(n):
        clean_bgr = tensor_to_bgr(clean[i])
        rough_bgr = tensor_to_bgr(rough[i])
        refined_bgr = tensor_to_bgr(refined[i])
        target_bgr = tensor_to_bgr(target[i])

        # Visualize residual around neutral gray.
        res = residual[i].detach().cpu()
        res_vis = (res / (2.0 * MAX_RESIDUAL_DELTA)) + 0.5
        res_bgr = tensor_to_bgr(res_vis)

        mask_img = mask[i].detach().cpu().clamp(0.0, 1.0)
        mask_np = (mask_img.squeeze(0).numpy() * 255.0).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)

        panels = [
            ("clean", clean_bgr),
            ("rough", rough_bgr),
            ("refined", refined_bgr),
            ("target", target_bgr),
            ("residual", res_bgr),
            ("mask", mask_bgr),
        ]

        labeled = []

        for label, img in panels:
            panel = img.copy()
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            labeled.append(panel)

        row = np.hstack(labeled)
        rows.append(row)

    grid = np.vstack(rows)
    cv2.imwrite(out_path, grid)


# ============================================================
# Training
# ============================================================

def train(args):
    dataset_dir = args.dataset_dir

    if dataset_dir is None:
        dataset_dir = find_latest_refiner_run(REFINER_DATASET_BASE)

    output_dirs = create_output_dir()

    dataset = RefinerCropDataset(dataset_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    model = ResidualUNetRefiner(
        in_ch=7,
        out_ch=3,
        base_ch=args.base_channels,
        max_delta=args.max_delta,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    train_meta = {
        "model_version": "residual_local_refiner_v1",
        "dataset_dir": dataset_dir,
        "device": DEVICE,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "image_size": IMAGE_SIZE,
        "num_samples": len(dataset),
        "max_residual_delta": args.max_delta,
        "input": "clean RGB + rough RGB + mask",
        "output": "residual correction applied to rough crop near mask",
    }

    meta_path = os.path.join(output_dirs["metadata"], "train_metadata.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=4)

    print("[TrainResidualRefiner] Device:", DEVICE)
    print("[TrainResidualRefiner] Metadata:", meta_path)

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            inp = batch["input"].to(DEVICE)
            rough = batch["rough"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            residual = model(inp)
            refined = apply_residual(
                rough=rough,
                residual=residual,
                mask=mask,
            )

            loss = residual_refiner_loss(
                pred=refined,
                target=target,
                rough=rough,
                mask=mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"[TrainResidualRefiner] "
                    f"epoch {epoch}/{args.epochs} "
                    f"batch {batch_idx}/{len(dataloader)} "
                    f"loss {loss.item():.5f}"
                )

        avg_loss = epoch_loss / max(1, len(dataloader))
        elapsed = time.time() - start_time

        print(
            f"[TrainResidualRefiner] Epoch {epoch} done | "
            f"avg_loss={avg_loss:.5f} | "
            f"time={elapsed:.1f}s"
        )

        if epoch % PREVIEW_EVERY_EPOCHS == 0:
            model.eval()

            with torch.no_grad():
                preview_batch = next(iter(dataloader))

                preview_inp = preview_batch["input"].to(DEVICE)
                preview_rough = preview_batch["rough"].to(DEVICE)
                preview_mask = preview_batch["mask"].to(DEVICE)

                preview_residual = model(preview_inp)
                preview_refined = apply_residual(
                    rough=preview_rough,
                    residual=preview_residual,
                    mask=preview_mask,
                )

            preview_path = os.path.join(
                output_dirs["previews"],
                f"preview_epoch_{epoch:03d}.png",
            )

            save_preview(
                batch=preview_batch,
                refined=preview_refined.cpu(),
                residual=preview_residual.cpu(),
                out_path=preview_path,
            )

            print("[TrainResidualRefiner] Saved preview:", preview_path)

        if epoch % SAVE_EVERY_EPOCHS == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                output_dirs["checkpoints"],
                f"local_refiner_residual_epoch_{epoch:03d}.pth",
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_meta": train_meta,
            }, ckpt_path)

            print("[TrainResidualRefiner] Saved checkpoint:", ckpt_path)

    print("[TrainResidualRefiner] Training complete.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to refiner dataset folder. If omitted, latest refiner dataset is used.",
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
        "--base_channels",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max_delta",
        type=float,
        default=MAX_RESIDUAL_DELTA,
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