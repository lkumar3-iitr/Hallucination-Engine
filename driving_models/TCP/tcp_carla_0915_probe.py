"""
tcp_carla_0915_probe.py

First live CARLA 0.9.15 probe for pretrained TCP.

Purpose
-------
1. Spawn one ego vehicle in CARLA.
2. Attach TWO cameras:
      A. TCP-native camera:
         900x256, FOV 100, x=-1.5, z=2.0

      B. HE canonical camera:
         1280x720, FOV 90, x=1.5, z=1.6

3. Let CARLA autopilot move the ego vehicle.
4. Feed both images independently into exactly the same TCP network.
5. DO NOT apply TCP controls.
6. Log the predicted controls from both camera configurations.

This is only a camera-compatibility diagnostic.

CARLA continues controlling the vehicle through autopilot.
"""

import argparse
import csv
import math
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

import carla


# ============================================================
# Paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]

DEFAULT_TCP_ROOT = HE_ROOT / "external_models" / "TCP"
DEFAULT_CHECKPOINT = (
    DEFAULT_TCP_ROOT / "weights" / "TCP_state_dict.pt"
)


# ============================================================
# TCP utilities
# ============================================================

def import_tcp(tcp_root):
    tcp_root = Path(tcp_root).resolve()

    if not tcp_root.exists():
        raise FileNotFoundError(f"TCP root does not exist: {tcp_root}")

    if str(tcp_root) not in sys.path:
        sys.path.insert(0, str(tcp_root))

    from TCP.config import GlobalConfig
    from TCP.model import TCP

    return GlobalConfig, TCP


def load_tcp_model(tcp_root, checkpoint_path, device):
    GlobalConfig, TCP = import_tcp(tcp_root)

    config = GlobalConfig()

    net = TCP(config)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    result = net.load_state_dict(
        state_dict,
        strict=True,
    )

    # strict=True should already throw on mismatch.
    print("[TCP] state dict loaded")
    print("[TCP] missing    :", result.missing_keys)
    print("[TCP] unexpected :", result.unexpected_keys)

    net = net.to(device)
    net.eval()

    return net, config


# ============================================================
# Image conversion
# ============================================================

IMAGENET_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def carla_image_to_rgb(image):
    """
    CARLA raw_data is BGRA uint8.
    Return RGB HxWx3 uint8.
    """
    arr = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    arr = arr.reshape(
        (image.height, image.width, 4)
    )

    rgb = cv2.cvtColor(
        arr,
        cv2.COLOR_BGRA2RGB,
    )

    return rgb


def center_crop_to_aspect(image_rgb, target_w, target_h):
    """
    Preserve geometry/aspect ratio before resizing.

    Example:
        source 1280x720
        target 900x256

    We crop vertically rather than stretch the image.
    """
    h, w = image_rgb.shape[:2]

    source_aspect = w / float(h)
    target_aspect = target_w / float(target_h)

    if abs(source_aspect - target_aspect) < 1e-6:
        return image_rgb

    if source_aspect < target_aspect:
        # Image is too tall -> crop vertically.
        new_h = int(round(w / target_aspect))
        new_h = min(new_h, h)

        y0 = (h - new_h) // 2
        y1 = y0 + new_h

        return image_rgb[y0:y1, :, :]

    # Image is too wide -> crop horizontally.
    new_w = int(round(h * target_aspect))
    new_w = min(new_w, w)

    x0 = (w - new_w) // 2
    x1 = x0 + new_w

    return image_rgb[:, x0:x1, :]


def prepare_tcp_rgb_native(rgb):
    """
    Native TCP image should already be 900x256.
    """
    if rgb.shape[1] != 900 or rgb.shape[0] != 256:
        rgb = cv2.resize(
            rgb,
            (900, 256),
            interpolation=cv2.INTER_LINEAR,
        )

    return rgb


def prepare_tcp_rgb_from_he_camera(rgb):
    """
    HE canonical image:
        1280x720

    Convert to TCP's 900x256 tensor without geometric stretching:
        center crop to TCP aspect ratio
        resize to 900x256

    HE itself remains untouched.
    """
    cropped = center_crop_to_aspect(
        rgb,
        target_w=900,
        target_h=256,
    )

    resized = cv2.resize(
        cropped,
        (900, 256),
        interpolation=cv2.INTER_LINEAR,
    )

    return resized


def rgb_to_tcp_tensor(rgb, device):
    pil = Image.fromarray(rgb)

    tensor = IMAGENET_TRANSFORM(pil)
    tensor = tensor.unsqueeze(0)

    return tensor.to(
        device,
        dtype=torch.float32,
    )


# ============================================================
# TCP input state
# ============================================================

def get_vehicle_speed_mps(vehicle):
    vel = vehicle.get_velocity()

    return math.sqrt(
        vel.x * vel.x
        + vel.y * vel.y
        + vel.z * vel.z
    )


def make_tcp_state(speed_mps, device):
    """
    First probe uses a simple straight command.

    TCP state:
        normalized speed : 1
        target point     : 2
        command one-hot  : 6

        total            : 9

    RoadOption LANEFOLLOW has traditionally been command value 4.
    TCP subtracts 1, therefore one-hot index 3.

    Target point [10, 0] means approximately 10 m ahead
    in TCP's local navigation representation.
    """

    speed = torch.tensor(
        [[float(speed_mps) / 12.0]],
        dtype=torch.float32,
        device=device,
    )

    target_point = torch.tensor(
        [[10.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )

    command_one_hot = torch.zeros(
        (1, 6),
        dtype=torch.float32,
        device=device,
    )

    # LANEFOLLOW -> command 4 -> index 3
    command_one_hot[0, 3] = 1.0

    state = torch.cat(
        [
            speed,
            target_point,
            command_one_hot,
        ],
        dim=1,
    )

    return state, target_point


# ============================================================
# TCP inference
# ============================================================

@torch.no_grad()
def run_tcp(
    net,
    rgb,
    speed_mps,
    device,
):
    tensor = rgb_to_tcp_tensor(
        rgb,
        device,
    )

    state, target_point = make_tcp_state(
        speed_mps,
        device,
    )

    gt_velocity = torch.tensor(
        [speed_mps],
        dtype=torch.float32,
        device=device,
    )

    pred = net(
        tensor,
        state,
        target_point,
    )

    # Direct control branch.
    steer_ctrl, throttle_ctrl, brake_ctrl, _ = (
        net.process_action(
            pred,
            4,               # LANEFOLLOW
            gt_velocity,
            target_point,
        )
    )

    # Trajectory/PID branch.
    steer_traj, throttle_traj, brake_traj, _ = (
        net.control_pid(
            pred["pred_wp"].clone(),
            gt_velocity,
            target_point.clone(),
        )
    )

    if brake_traj < 0.05:
        brake_traj = 0.0

    if throttle_traj > brake_traj:
        brake_traj = 0.0

    # Same fusion used by original TCP agent.
    alpha = 0.3

    steer = np.clip(
        alpha * steer_ctrl
        + (1.0 - alpha) * steer_traj,
        -1.0,
        1.0,
    )

    throttle = np.clip(
        alpha * throttle_ctrl
        + (1.0 - alpha) * throttle_traj,
        0.0,
        0.75,
    )

    brake = np.clip(
        alpha * brake_ctrl
        + (1.0 - alpha) * brake_traj,
        0.0,
        1.0,
    )

    waypoints = (
        pred["pred_wp"][0]
        .detach()
        .cpu()
        .numpy()
    )

    pred_speed = float(
        pred["pred_speed"]
        .detach()
        .cpu()
        .reshape(-1)[0]
    )

    return {
        "steer": float(steer),
        "throttle": float(throttle),
        "brake": float(brake),

        "steer_ctrl": float(steer_ctrl),
        "throttle_ctrl": float(throttle_ctrl),
        "brake_ctrl": float(brake_ctrl),

        "steer_traj": float(steer_traj),
        "throttle_traj": float(throttle_traj),
        "brake_traj": float(brake_traj),

        "pred_speed": pred_speed,
        "waypoints": waypoints,
    }


# ============================================================
# CARLA utilities
# ============================================================

def get_sensor_frame(sensor_queue, target_frame):
    """
    Wait until we receive the image belonging to target_frame.
    """
    while True:
        data = sensor_queue.get(timeout=5.0)

        if data.frame == target_frame:
            return data

        if data.frame > target_frame:
            raise RuntimeError(
                f"Sensor skipped target frame {target_frame}; "
                f"received {data.frame}"
            )


def make_camera(
    world,
    ego,
    width,
    height,
    fov,
    x,
    y,
    z,
    pitch,
    yaw,
    roll,
):
    bp = world.get_blueprint_library().find(
        "sensor.camera.rgb"
    )

    bp.set_attribute(
        "image_size_x",
        str(width),
    )
    bp.set_attribute(
        "image_size_y",
        str(height),
    )
    bp.set_attribute(
        "fov",
        str(fov),
    )

    transform = carla.Transform(
        carla.Location(
            x=float(x),
            y=float(y),
            z=float(z),
        ),
        carla.Rotation(
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
        ),
    )

    return world.spawn_actor(
        bp,
        transform,
        attach_to=ego,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--town",
        default="Town10HD_Opt",
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--tcp-root",
        default=str(DEFAULT_TCP_ROOT),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--output",
        default=str(
            HE_ROOT
            / "driving_models"
            / "TCP"
            / "outputs"
            / "tcp_camera_probe.csv"
        ),
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable"
            )

        print(
            "[GPU]",
            torch.cuda.get_device_name(0),
        )

    tcp_root = Path(args.tcp_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()

    print("[TCP root]", tcp_root)
    print("[checkpoint]", checkpoint)

    net, config = load_tcp_model(
        tcp_root=tcp_root,
        checkpoint_path=checkpoint,
        device=device,
    )

    print("[TCP] pred_len:", config.pred_len)

    # --------------------------------------------------------
    # CARLA connection
    # --------------------------------------------------------

    client = carla.Client(
        args.host,
        args.port,
    )
    client.set_timeout(20.0)

    print("[CARLA] loading:", args.town)
    world = client.load_world(args.town)

    original_settings = world.get_settings()

    traffic_manager = client.get_trafficmanager(8000)

    ego = None
    camera_tcp = None
    camera_he = None

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        settings = world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            1.0 / float(args.fps)
        )

        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)

        # ----------------------------------------------------
        # Ego
        # ----------------------------------------------------

        blueprints = world.get_blueprint_library()

        ego_bp = blueprints.filter(
            "vehicle.tesla.model3"
        )[0]

        spawn_points = (
            world.get_map().get_spawn_points()
        )

        if not spawn_points:
            raise RuntimeError(
                "No spawn points available"
            )

        spawn_index = (
            args.spawn_index % len(spawn_points)
        )

        spawn_tf = spawn_points[spawn_index]

        ego = world.try_spawn_actor(
            ego_bp,
            spawn_tf,
        )

        if ego is None:
            raise RuntimeError(
                f"Could not spawn ego at index "
                f"{spawn_index}"
            )

        print(
            "[CARLA] ego spawned:",
            ego.id,
            "spawn index:",
            spawn_index,
        )

        # CARLA autopilot moves the car.
        # TCP controls are ONLY observed.
        ego.set_autopilot(
            True,
            traffic_manager.get_port(),
        )

        # ----------------------------------------------------
        # TCP-native camera
        # ----------------------------------------------------

        camera_tcp = make_camera(
            world=world,
            ego=ego,

            width=900,
            height=256,
            fov=100,

            x=-1.5,
            y=0.0,
            z=2.0,

            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        )

        # ----------------------------------------------------
        # HE canonical camera
        # ----------------------------------------------------

        camera_he = make_camera(
            world=world,
            ego=ego,

            width=1280,
            height=720,
            fov=90,

            x=1.5,
            y=0.0,
            z=1.6,

            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        )

        q_tcp = queue.Queue()
        q_he = queue.Queue()

        camera_tcp.listen(q_tcp.put)
        camera_he.listen(q_he.put)

        # Give sensors a few synchronous ticks.
        for _ in range(10):
            frame = world.tick()

            get_sensor_frame(
                q_tcp,
                frame,
            )
            get_sensor_frame(
                q_he,
                frame,
            )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_file = open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "probe_idx",
                "carla_frame",
                "speed_mps",

                "tcp_steer",
                "tcp_throttle",
                "tcp_brake",
                "tcp_pred_speed",

                "hecam_steer",
                "hecam_throttle",
                "hecam_brake",
                "hecam_pred_speed",

                "abs_steer_delta",
                "abs_throttle_delta",
                "abs_brake_delta",
            ],
        )

        writer.writeheader()

        print()
        print("=" * 78)
        print("TCP CARLA 0.9.15 CAMERA PROBE")
        print("=" * 78)
        print(
            "TCP native camera : "
            "900x256 FOV100 x=-1.5 z=2.0"
        )
        print(
            "HE camera         : "
            "1280x720 FOV90 x=1.5 z=1.6"
        )
        print(
            "TCP controls are NOT applied."
        )
        print(
            "CARLA autopilot controls the ego."
        )
        print("=" * 78)
        print()

        for probe_idx in range(args.frames):

            carla_frame = world.tick()

            tcp_image = get_sensor_frame(
                q_tcp,
                carla_frame,
            )

            he_image = get_sensor_frame(
                q_he,
                carla_frame,
            )

            rgb_tcp = carla_image_to_rgb(
                tcp_image
            )

            rgb_he = carla_image_to_rgb(
                he_image
            )

            rgb_tcp_input = (
                prepare_tcp_rgb_native(
                    rgb_tcp
                )
            )

            rgb_he_input = (
                prepare_tcp_rgb_from_he_camera(
                    rgb_he
                )
            )

            speed_mps = (
                get_vehicle_speed_mps(ego)
            )

            out_tcp = run_tcp(
                net=net,
                rgb=rgb_tcp_input,
                speed_mps=speed_mps,
                device=device,
            )

            out_he = run_tcp(
                net=net,
                rgb=rgb_he_input,
                speed_mps=speed_mps,
                device=device,
            )

            abs_steer_delta = abs(
                out_tcp["steer"]
                - out_he["steer"]
            )

            abs_throttle_delta = abs(
                out_tcp["throttle"]
                - out_he["throttle"]
            )

            abs_brake_delta = abs(
                out_tcp["brake"]
                - out_he["brake"]
            )

            writer.writerow({
                "probe_idx": probe_idx,
                "carla_frame": carla_frame,
                "speed_mps": speed_mps,

                "tcp_steer":
                    out_tcp["steer"],
                "tcp_throttle":
                    out_tcp["throttle"],
                "tcp_brake":
                    out_tcp["brake"],
                "tcp_pred_speed":
                    out_tcp["pred_speed"],

                "hecam_steer":
                    out_he["steer"],
                "hecam_throttle":
                    out_he["throttle"],
                "hecam_brake":
                    out_he["brake"],
                "hecam_pred_speed":
                    out_he["pred_speed"],

                "abs_steer_delta":
                    abs_steer_delta,
                "abs_throttle_delta":
                    abs_throttle_delta,
                "abs_brake_delta":
                    abs_brake_delta,
            })

            if (
                probe_idx < 10
                or probe_idx % 20 == 0
                or probe_idx == args.frames - 1
            ):
                print(
                    f"[{probe_idx:04d}] "
                    f"speed={speed_mps:5.2f} | "
                    f"TCP "
                    f"S={out_tcp['steer']:+.3f} "
                    f"T={out_tcp['throttle']:.3f} "
                    f"B={out_tcp['brake']:.3f} | "
                    f"HEcam "
                    f"S={out_he['steer']:+.3f} "
                    f"T={out_he['throttle']:.3f} "
                    f"B={out_he['brake']:.3f} | "
                    f"dS={abs_steer_delta:.3f}"
                )

        csv_file.close()

        print()
        print("=" * 78)
        print("PROBE COMPLETE")
        print("=" * 78)
        print("output:", output_path)
        print("=" * 78)

    finally:
        print("[cleanup]")

        if camera_tcp is not None:
            camera_tcp.stop()
            camera_tcp.destroy()

        if camera_he is not None:
            camera_he.stop()
            camera_he.destroy()

        if ego is not None:
            ego.destroy()

        traffic_manager.set_synchronous_mode(
            False
        )

        world.apply_settings(
            original_settings
        )


if __name__ == "__main__":
    main()