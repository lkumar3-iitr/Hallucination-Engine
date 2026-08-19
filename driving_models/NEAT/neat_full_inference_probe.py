"""
neat_full_inference_probe.py

Full model compatibility probe for the official NEAT checkpoint.

Tests:
    1. Encoder checkpoint
    2. Decoder checkpoint
    3. Three-camera forward pass
    4. Attention-field waypoint planning
    5. Original NEAT PID control

No CARLA server is required.
"""

from pathlib import Path
import sys

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
    print("NEAT FULL INFERENCE PROBE")
    print("=" * 72)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print()
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = GlobalConfig()

    print()
    print("Configuration")
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
        "tot_len:",
        config.tot_len,
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

    # --------------------------------------------------------
    # Load official weights
    # --------------------------------------------------------

    encoder_path = (
        CHECKPOINT_DIR
        / "best_encoder.pth"
    )

    decoder_path = (
        CHECKPOINT_DIR
        / "best_decoder.pth"
    )

    print("Loading encoder checkpoint...")

    encoder_state = torch.load(
        encoder_path,
        map_location="cpu",
        weights_only=True,
    )

    net.encoder.load_state_dict(
        encoder_state,
        strict=True,
    )

    print("Encoder loaded.")

    print("Loading decoder checkpoint...")

    decoder_state = torch.load(
        decoder_path,
        map_location="cpu",
        weights_only=True,
    )

    net.decoder.load_state_dict(
        decoder_state,
        strict=True,
    )

    print("Decoder loaded.")

    net = net.to(device)
    net.eval()

    # --------------------------------------------------------
    # Synthetic three-camera inputs
    #
    # Original NEAT agent sends image tensors in approximately
    # [0, 255] range after cropping.
    # --------------------------------------------------------

    torch.manual_seed(42)

    images = []

    for camera_idx in range(
        config.num_camera
    ):

        image = torch.rand(
            (
                1,
                3,
                config.crop,
                config.crop,
            ),
            dtype=torch.float32,
            device=device,
        )

        image *= 255.0

        images.append(
            image
        )

    # --------------------------------------------------------
    # Vehicle speed
    # --------------------------------------------------------

    velocity = torch.tensor(
        [4.0],
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Navigation target
    #
    # Original neat_agent.py produces target_point with shape:
    #
    #     (2, 1)
    #
    # x = lateral component
    # y = longitudinal component
    #
    # Straight target ~20 m forward.
    # --------------------------------------------------------

    target_point = torch.tensor(
        [
            [0.0],
            [20.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    print()
    print("Input")
    print("-" * 72)

    print(
        "camera tensors:",
        len(images),
    )

    print(
        "camera shape:",
        tuple(
            images[0].shape
        ),
    )

    print(
        "velocity:",
        float(
            velocity.item()
        ),
    )

    print(
        "target_point:",
        target_point
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .tolist(),
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    print()
    print("Running encoder...")

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

    # --------------------------------------------------------
    # Create planning grids
    # --------------------------------------------------------

    print()
    print("Creating planning grids...")

    plan_grid = net.create_plan_grid(
        config.plan_scale,
        config.plan_points,
        1,
    )

    light_grid = net.create_light_grid(
        config.light_x_steps,
        config.light_y_steps,
        1,
    )

    print(
        "plan grid shape:",
        tuple(
            plan_grid.shape
        ),
    )

    print(
        "light grid shape:",
        tuple(
            light_grid.shape
        ),
    )

    # --------------------------------------------------------
    # Attention-field planning
    # --------------------------------------------------------

    print()
    print("Running waypoint planner...")

    with torch.no_grad():

        predicted_waypoints, red_light_occ = (
            net.plan(
                target_point,
                encoding,
                plan_grid,
                light_grid,
                config.plan_points,
                config.plan_iters,
            )
        )

    print(
        "predicted waypoint shape:",
        tuple(
            predicted_waypoints.shape
        ),
    )

    print(
        "predicted waypoints:"
    )

    print(
        predicted_waypoints
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "waypoints finite:",
        bool(
            torch.isfinite(
                predicted_waypoints
            ).all()
        ),
    )

    if torch.is_tensor(
        red_light_occ
    ):

        red_light_value = float(
            red_light_occ
            .detach()
            .cpu()
            .item()
        )

    else:

        red_light_value = float(
            red_light_occ
        )

    print(
        "red_light_occ:",
        red_light_value,
    )

    # --------------------------------------------------------
    # Original agent discards the first waypoint because it
    # corresponds to the current timestep.
    # --------------------------------------------------------

    future_waypoints = (
        predicted_waypoints[
            :,
            config.seq_len:
        ]
    )

    print()
    print(
        "future waypoint shape:",
        tuple(
            future_waypoints.shape
        ),
    )

    # --------------------------------------------------------
    # Original NEAT PID controller
    # --------------------------------------------------------

    print()
    print("Running PID controller...")

    with torch.no_grad():

        (
            steer,
            throttle,
            brake,
            metadata,
        ) = net.control_pid(
            future_waypoints,
            velocity,
            target_point,
            red_light_occ,
        )

    print()
    print("Control")
    print("-" * 72)

    print(
        "steer:",
        float(steer),
    )

    print(
        "throttle:",
        float(throttle),
    )

    print(
        "brake:",
        bool(brake),
    )

    print()
    print("PID metadata")

    for key, value in metadata.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert torch.isfinite(
        encoding
    ).all()

    assert torch.isfinite(
        predicted_waypoints
    ).all()

    assert -1.0 <= float(steer) <= 1.0

    assert 0.0 <= float(throttle) <= 1.0

    print()
    print("=" * 72)
    print("NEAT FULL INFERENCE PROBE PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()