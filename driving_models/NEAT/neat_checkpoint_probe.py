"""
NEAT checkpoint compatibility probe.

Purpose:
1. Import the original NEAT architecture.
2. Construct the released AttentionField model.
3. Load the official encoder and decoder checkpoints.
4. Move the complete model to GPU.
5. Run a minimal three-camera encoder forward pass.

No CARLA dependency is required for this probe.
"""

from pathlib import Path
import sys

import numpy as np
import torch


# ============================================================
# Paths
# ============================================================

HE_ROOT = Path(__file__).resolve().parents[2]

NEAT_ROOT = (
    HE_ROOT
    / "external_models"
    / "neat"
)

CHECKPOINT_DIR = (
    NEAT_ROOT
    / "model_ckpt"
    / "neat"
)

sys.path.insert(
    0,
    str(NEAT_ROOT),
)


# ============================================================
# NEAT imports
# ============================================================

from neat.architectures import AttentionField
from neat.config import GlobalConfig


def main():

    print("=" * 72)
    print("NEAT CHECKPOINT PROBE")
    print("=" * 72)

    print()
    print("NEAT root:")
    print(NEAT_ROOT)

    print()
    print("Checkpoint:")
    print(CHECKPOINT_DIR)

    encoder_path = (
        CHECKPOINT_DIR
        / "best_encoder.pth"
    )

    decoder_path = (
        CHECKPOINT_DIR
        / "best_decoder.pth"
    )

    assert encoder_path.exists(), encoder_path
    assert decoder_path.exists(), decoder_path

    print()
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------------
    # Original NEAT configuration
    # --------------------------------------------------------

    config = GlobalConfig()

    print()
    print("NEAT configuration")
    print("-" * 72)

    print(
        "num_camera:",
        config.num_camera,
    )

    print(
        "seq_len:",
        config.seq_len,
    )

    print(
        "pred_len:",
        config.pred_len,
    )

    print(
        "crop:",
        config.crop,
    )

    # --------------------------------------------------------
    # Construct model
    # --------------------------------------------------------

    print()
    print("Constructing AttentionField...")

    net = AttentionField(
        config,
        device,
    )

    print("Architecture constructed.")

    # --------------------------------------------------------
    # Load official checkpoints
    # --------------------------------------------------------

    print()
    print("Loading encoder...")

    encoder_state = torch.load(
        encoder_path,
        map_location="cpu",
        weights_only=True,
    )

    encoder_result = (
        net.encoder.load_state_dict(
            encoder_state,
            strict=True,
        )
    )

    print(
        "encoder load:",
        encoder_result,
    )

    print()
    print("Loading decoder...")

    decoder_state = torch.load(
        decoder_path,
        map_location="cpu",
        weights_only=True,
    )

    decoder_result = (
        net.decoder.load_state_dict(
            decoder_state,
            strict=True,
        )
    )

    print(
        "decoder load:",
        decoder_result,
    )

    # --------------------------------------------------------
    # GPU + eval
    # --------------------------------------------------------

    net = net.to(device)
    net.eval()

    print()
    print("Model moved to GPU.")

    # --------------------------------------------------------
    # Minimal encoder forward pass
    #
    # Original agent:
    #   front, left, right
    #   each cropped to 256 x 256
    #
    # Input values are image-like [0, 255].
    # --------------------------------------------------------

    batch_size = 1

    images = []

    for _ in range(
        config.num_camera
    ):

        image = torch.zeros(
            (
                batch_size,
                3,
                config.crop,
                config.crop,
            ),
            dtype=torch.float32,
            device=device,
        )

        images.append(
            image
        )

    velocity = torch.tensor(
        [0.0],
        dtype=torch.float32,
        device=device,
    )

    print()
    print("Running encoder probe...")

    with torch.no_grad():

        encoding = net.encoder(
            images,
            velocity,
        )

    print(
        "encoding shape:",
        tuple(
            encoding.shape
        ),
    )

    print(
        "encoding finite:",
        bool(
            torch.isfinite(
                encoding
            ).all()
        ),
    )

    print()
    print("=" * 72)
    print("NEAT CHECKPOINT PROBE PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()