"""
aimmt_adapter_v1.py

Thin AIM-MT 2D adapter for the generic Hallucination Engine runtime.

Original model remains unchanged.

Native driving inputs:
    RGB:
        400x300
        FOV 100
        x=1.3, y=0.0, z=2.3
        yaw=0

    GNSS
    IMU
    vehicle speed

Navigation:
    original RoutePlanner(4.0, 50.0)

Model:
    aim_mt_2d
    checkpoint: model_ckpt/aim_mt_sem/best_model.pth
"""

from __future__ import annotations

import queue
import sys
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ============================================================
# Paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]

COMMON_DIR = (
    HE_ROOT
    / "driving_models"
    / "common"
)

NEAT_MODEL_DIR = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
)

DEFAULT_AIMMT_ROOT = (
    HE_ROOT
    / "external_models"
    / "neat"
)

DEFAULT_CHECKPOINT = (
    DEFAULT_AIMMT_ROOT
    / "model_ckpt"
    / "aim_mt_sem"
)

for path in (
    COMMON_DIR,
    NEAT_MODEL_DIR,
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


# ============================================================
# Generic adapter
# ============================================================

from driving_model_adapter_v1 import (
    CameraSpec,
    DrivingModelAdapter,
    ModelControl,
    ModelSensorFrame,
    ModelStepResult,
    validate_camera_rgb_keys,
)


# ============================================================
# Already validated common CARLA helpers
# ============================================================

from neat_carla_0915_closed_loop import (
    make_camera,
    make_gnss,
    make_imu,
    get_sensor_frame,
    build_neat_gps_plan,
    get_neat_navigation,
)


def command_label(command):
    name = getattr(command, "name", None)
    return str(name) if name is not None else str(command)


def command_value(command):
    value = getattr(command, "value", None)

    if value is None:
        return float("nan")

    try:
        return float(value)
    except Exception:
        return float("nan")


class AIMMTAdapterV1(DrivingModelAdapter):

    model_name = "aimmt"
    required_fps = 20.0

    @classmethod
    def camera_specs(cls):
        return (
            CameraSpec(
                name="rgb",
                width=400,
                height=300,
                fov_deg=100.0,
                x_m=1.3,
                y_m=0.0,
                z_m=2.3,
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
            ),
        )

    def __init__(self, args):
        super().__init__(args)

        self.device = None
        self.net = None
        self.config = None
        self.RoutePlanner = None
        self.scale_and_crop_image = None

        self.route_planner = None
        self.input_buffer = None

        self.camera = None
        self.gnss = None
        self.imu = None

        self.q_camera = None
        self.q_gnss = None
        self.q_imu = None

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    def load(self):

        device_name = getattr(
            self.args,
            "device",
            "cuda",
        )

        self.device = torch.device(
            device_name
        )

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "AIM-MT requested CUDA but CUDA is unavailable."
            )

        if self.device.type == "cuda":
            print(
                "[AIM-MT GPU]",
                torch.cuda.get_device_name(0),
            )

        aim_root = DEFAULT_AIMMT_ROOT.resolve()

        leaderboard_root = (
            aim_root
            / "leaderboard"
        )

        for path in (
            aim_root,
            leaderboard_root,
        ):
            text = str(path)

            if text not in sys.path:
                sys.path.insert(
                    0,
                    text,
                )
        # ----------------------------------------------------
        # Compatibility for older AIM-MT torchvision import.
        #
        # Original code imports:
        # torchvision.models.utils.load_state_dict_from_url
        #
        # Modern torchvision removed torchvision.models.utils.
        # Keep the original model source untouched and provide
        # the old import path from torch.hub.
        # ----------------------------------------------------

        try:
            import torchvision.models.utils

        except ModuleNotFoundError:

            from torch.hub import (
                load_state_dict_from_url
            )

            compat_module = types.ModuleType(
                "torchvision.models.utils"
            )

            compat_module.load_state_dict_from_url = (
                load_state_dict_from_url
            )

            sys.modules[
                "torchvision.models.utils"
            ] = compat_module

            print(
                "[AIM-MT compat] "
                "torchvision.models.utils -> torch.hub"
            )
        from aim_mt_2d.architectures import (
            MultiTaskImageNetwork
        )

        from aim_mt_2d.config import (
            GlobalConfig
        )

        from aim_mt_2d.data import (
            scale_and_crop_image
        )

        from team_code.planner import (
            RoutePlanner
        )

        self.RoutePlanner = RoutePlanner
        self.scale_and_crop_image = (
            scale_and_crop_image
        )

        self.config = GlobalConfig()

        if int(self.config.seq_len) != 1:
            raise RuntimeError(
                "AIM-MT adapter currently expects seq_len == 1."
            )

        self.net = MultiTaskImageNetwork(
            self.config,
            self.device,
        )

        checkpoint_value = (
            getattr(
                self.args,
                "checkpoint",
                None,
            )
            or
            DEFAULT_CHECKPOINT
        )

        checkpoint = Path(
            checkpoint_value
        ).resolve()

        if checkpoint.is_dir():
            checkpoint = (
                checkpoint
                / "best_model.pth"
            )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"AIM-MT checkpoint not found: {checkpoint}"
            )

        print(
            "[AIM-MT checkpoint]",
            checkpoint,
        )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        result = self.net.load_state_dict(
            state,
            strict=True,
        )

        print(
            "[AIM-MT state]",
            result,
        )

        self.net = self.net.to(
            self.device
        )

        self.net.eval()

        self.input_buffer = deque(
            maxlen=int(
                self.config.seq_len
            )
        )

        print(
            "[AIM-MT adapter] loaded"
        )

        print(
            "[AIM-MT adapter] "
            "camera: 400x300 FOV100 x=1.3 z=2.3"
        )

        print(
            "[AIM-MT adapter] "
            f"input crop: {self.config.input_resolution}"
        )

        print(
            "[AIM-MT adapter] "
            f"pred_len: {self.config.pred_len}"
        )

    # --------------------------------------------------------
    # Sensors + route
    # --------------------------------------------------------

    def setup(
        self,
        world,
        ego,
        carla_map,
        route,
    ):

        if self.net is None:
            raise RuntimeError(
                "AIMMTAdapterV1.load() "
                "must be called before setup()."
            )

        self.camera = make_camera(
            world,
            ego,
            yaw=0.0,
        )

        self.gnss = make_gnss(
            world,
            ego,
        )

        self.imu = make_imu(
            world,
            ego,
        )

        self.q_camera = queue.Queue()
        self.q_gnss = queue.Queue()
        self.q_imu = queue.Queue()

        self.camera.listen(
            self.q_camera.put
        )

        self.gnss.listen(
            self.q_gnss.put
        )

        self.imu.listen(
            self.q_imu.put
        )

        gps_plan = build_neat_gps_plan(
            carla_map,
            route,
        )

        self.route_planner = (
            self.RoutePlanner(
                4.0,
                50.0,
            )
        )

        self.route_planner.set_route(
            gps_plan,
            gps=True,
        )

        print(
            "[AIM-MT adapter] "
            "RoutePlanner(4.0, 50.0)"
        )

        return [
            self.camera,
            self.gnss,
            self.imu,
        ]

    # --------------------------------------------------------
    # Sensors
    # --------------------------------------------------------

    def read_sensor_frame(
        self,
        target_frame,
    ):

        camera = get_sensor_frame(
            self.q_camera,
            target_frame,
            "AIM-MT RGB",
        )

        gnss = get_sensor_frame(
            self.q_gnss,
            target_frame,
            "AIM-MT GNSS",
        )

        imu = get_sensor_frame(
            self.q_imu,
            target_frame,
            "AIM-MT IMU",
        )

        return ModelSensorFrame(
            carla_frame=int(
                target_frame
            ),
            cameras={
                "rgb": camera,
            },
            auxiliary={
                "gnss": gnss,
                "imu": imu,
            },
        )

    # --------------------------------------------------------
    # Native image preprocessing
    # --------------------------------------------------------

    def prepare_image(
        self,
        rgb,
    ):

        rgb = np.asarray(
            rgb,
            dtype=np.uint8,
        )

        chw = (
            self.scale_and_crop_image(
                Image.fromarray(rgb),
                scale=self.config.scale,
                crop=self.config.input_resolution,
            )
        )

        return (
            torch.from_numpy(
                np.asarray(chw)
            )
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=torch.float32,
            )
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        rgb_by_camera,
        sensor_frame,
        speed_mps,
        scenario_frame,
    ):

        validate_camera_rgb_keys(
            self,
            rgb_by_camera,
        )

        nav = get_neat_navigation(
            self.route_planner,
            sensor_frame.auxiliary["gnss"],
            sensor_frame.auxiliary["imu"],
        )

        target_xy = np.asarray(
            nav["target_point"],
            dtype=np.float32,
        )

        image = self.prepare_image(
            rgb_by_camera["rgb"]
        )

        bootstrap = (
            int(scenario_frame)
            <
            int(self.config.seq_len)
        )

        if bootstrap:

            self.input_buffer.append(
                image
            )

            steer = 0.0
            throttle = 0.0
            brake = 0.0

            native_metadata = {}

        else:

            if len(self.input_buffer) >= int(
                self.config.seq_len
            ):
                self.input_buffer.popleft()

            self.input_buffer.append(
                image
            )

            velocity = torch.tensor(
                [
                    float(speed_mps)
                ],
                dtype=torch.float32,
                device=self.device,
            )

            target_point = torch.tensor(
                [[
                    float(target_xy[0]),
                    float(target_xy[1]),
                ]],
                dtype=torch.float32,
                device=self.device,
            )

            encoding = []

            image_encoding = (
                self.net.image_encoder(
                    list(
                        self.input_buffer
                    )
                )[0]
            )

            encoding.append(
                image_encoding
            )

            predicted_waypoints = self.net(
                encoding,
                target_point,
            )

            (
                steer,
                throttle,
                brake,
                native_metadata,
            ) = self.net.control_pid(
                predicted_waypoints,
                velocity,
            )

            steer = float(steer)
            throttle = float(throttle)
            brake = float(brake)

            # Original AIM-MT leaderboard post-processing.
            if brake < 0.05:
                brake = 0.0

            if throttle > brake:
                brake = 0.0

        metadata = {
            "target_x":
                float(target_xy[0]),

            "target_y":
                float(target_xy[1]),

            "command_name":
                command_label(
                    nav["command"]
                ),

            "command_value":
                command_value(
                    nav["command"]
                ),

            "navigation_position":
                np.asarray(
                    nav["position"]
                )
                .astype(float)
                .tolist(),

            "desired_speed":
                float(
                    native_metadata.get(
                        "desired_speed",
                        float("nan"),
                    )
                ),

            "angle":
                float(
                    native_metadata.get(
                        "angle",
                        float("nan"),
                    )
                ),
        }

        return ModelStepResult(
            control=ModelControl(
                steer=float(steer),
                throttle=float(throttle),
                brake=float(brake),
            ),
            metadata=metadata,
            bootstrap=bool(
                bootstrap
            ),
        )

    def summary(self):

        out = super().summary()

        out.update({
            "native_cameras":
                "1x 400x300 FOV100 x=1.3 z=2.3 yaw=0",

            "navigation":
                "AIM-MT RoutePlanner(4.0, 50.0) + GNSS + IMU",

            "control":
                "AIM-MT predicted waypoints + native PID",

            "checkpoint_family":
                "aim_mt_sem",
        })

        if self.config is not None:
            out.update({
                "seq_len":
                    int(
                        self.config.seq_len
                    ),

                "pred_len":
                    int(
                        self.config.pred_len
                    ),

                "input_resolution":
                    int(
                        self.config.input_resolution
                    ),
            })

        return out


ADAPTER_CLASS = AIMMTAdapterV1