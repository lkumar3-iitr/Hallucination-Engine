"""
neat_adapter_v1.py

Thin NEAT adapter for the generic Hallucination Engine experiment runtime.

NEAT remains unchanged.

Native NEAT sensors:
    RGB front:
        400x300
        FOV 100
        x=1.3
        y=0.0
        z=2.3
        yaw=0

    RGB left:
        same geometry
        yaw=-60

    RGB right:
        same geometry
        yaw=+60

    GNSS
    IMU

The adapter reuses the already validated:
    load_neat()
    make_camera()
    make_gnss()
    make_imu()
    get_sensor_frame()
    build_neat_gps_plan()
    get_neat_navigation()
    run_neat()
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Repository paths
# ============================================================

THIS_FILE = Path(__file__).resolve()

HE_ROOT = THIS_FILE.parents[2]

COMMON_DIR = (
    HE_ROOT
    / "driving_models"
    / "common"
)

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(COMMON_DIR),
    )


# ============================================================
# Generic adapter contract
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
# Existing validated NEAT implementation
# ============================================================

from neat_carla_0915_closed_loop import (
    DEFAULT_NEAT_ROOT,
    DEFAULT_CHECKPOINT,

    load_neat,

    make_camera,
    make_gnss,
    make_imu,
    get_sensor_frame,

    build_neat_gps_plan,
    get_neat_navigation,

    run_neat,
)


# ============================================================
# Helpers
# ============================================================

def command_label(command):

    name = getattr(
        command,
        "name",
        None,
    )

    if name is not None:
        return str(name)

    return str(command)


def command_value(command):

    value = getattr(
        command,
        "value",
        None,
    )

    if value is None:
        return float("nan")

    try:
        return float(value)

    except Exception:
        return float("nan")


def scalar_float(
    value,
    default=float("nan"),
):

    if value is None:
        return float(default)

    if torch.is_tensor(value):

        try:

            return float(
                value
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )

        except Exception:

            return float(default)

    try:

        array = np.asarray(
            value
        )

        if array.size == 0:
            return float(default)

        return float(
            array.reshape(-1)[0]
        )

    except Exception:

        try:
            return float(value)

        except Exception:
            return float(default)


# ============================================================
# Bootstrap
# ============================================================

def make_bootstrap_result():

    return {
        "steer":
            0.0,

        "throttle":
            0.0,

        "brake":
            0.0,

        "red_light_occ":
            0,

        "metadata": {
            "desired_speed":
                float("nan"),

            "angle":
                float("nan"),

            "angle_last":
                float("nan"),

            "angle_target":
                float("nan"),

            "angle_final":
                float("nan"),
        },
    }


# ============================================================
# NEAT adapter
# ============================================================

class NEATAdapterV1(
    DrivingModelAdapter
):

    model_name = "neat"

    required_fps = 20.0

    # --------------------------------------------------------
    # Native camera layout
    # --------------------------------------------------------

    @classmethod
    def camera_specs(
        cls,
    ):

        return (
            CameraSpec(
                name="front",

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

            CameraSpec(
                name="left",

                width=400,
                height=300,
                fov_deg=100.0,

                x_m=1.3,
                y_m=0.0,
                z_m=2.3,

                pitch_deg=0.0,
                yaw_deg=-60.0,
                roll_deg=0.0,
            ),

            CameraSpec(
                name="right",

                width=400,
                height=300,
                fov_deg=100.0,

                x_m=1.3,
                y_m=0.0,
                z_m=2.3,

                pitch_deg=0.0,
                yaw_deg=60.0,
                roll_deg=0.0,
            ),
        )

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------

    def __init__(
        self,
        args,
    ):

        super().__init__(
            args
        )

        self.device = None

        self.net = None
        self.config = None

        self.RoutePlanner = None
        self.route_planner = None

        self.plan_grid = None
        self.light_grid = None

        self.camera_front = None
        self.camera_left = None
        self.camera_right = None

        self.gnss = None
        self.imu = None

        self.q_front = None
        self.q_left = None
        self.q_right = None
        self.q_gnss = None
        self.q_imu = None

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    def load(
        self,
    ):

        device_name = getattr(
            self.args,
            "device",
            "cuda",
        )

        self.device = torch.device(
            device_name
        )

        if (
            self.device.type
            == "cuda"
            and
            not torch.cuda.is_available()
        ):

            raise RuntimeError(
                "NEAT requested CUDA but CUDA is unavailable."
            )

        if self.device.type == "cuda":

            print(
                "[NEAT GPU]",
                torch.cuda.get_device_name(
                    0
                ),
            )

        neat_root_value = (
            getattr(
                self.args,
                "neat_root",
                None,
            )
            or
            DEFAULT_NEAT_ROOT
        )

        neat_root = Path(
            neat_root_value
        ).resolve()

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

        (
            self.net,
            self.config,
            self.RoutePlanner,
            self.plan_grid,
            self.light_grid,
        ) = load_neat(
            neat_root,
            checkpoint,
            self.device,
        )

        print(
            "[NEAT adapter] loaded"
        )

        print(
            "[NEAT adapter] cameras:",
            self.config.num_camera,
        )

        print(
            "[NEAT adapter] seq_len:",
            self.config.seq_len,
        )

        print(
            "[NEAT adapter] pred_len:",
            self.config.pred_len,
        )

    # --------------------------------------------------------
    # Native sensors + navigation
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
                "NEATAdapterV1.load() "
                "must be called before setup()."
            )

        # ----------------------------------------------------
        # Cameras
        # ----------------------------------------------------

        self.camera_front = (
            make_camera(
                world,
                ego,
                yaw=0.0,
            )
        )

        self.camera_left = (
            make_camera(
                world,
                ego,
                yaw=-60.0,
            )
        )

        self.camera_right = (
            make_camera(
                world,
                ego,
                yaw=60.0,
            )
        )

        # ----------------------------------------------------
        # Navigation sensors
        # ----------------------------------------------------

        self.gnss = (
            make_gnss(
                world,
                ego,
            )
        )

        self.imu = (
            make_imu(
                world,
                ego,
            )
        )

        # ----------------------------------------------------
        # Queues
        # ----------------------------------------------------

        self.q_front = (
            queue.Queue()
        )

        self.q_left = (
            queue.Queue()
        )

        self.q_right = (
            queue.Queue()
        )

        self.q_gnss = (
            queue.Queue()
        )

        self.q_imu = (
            queue.Queue()
        )

        self.camera_front.listen(
            self.q_front.put
        )

        self.camera_left.listen(
            self.q_left.put
        )

        self.camera_right.listen(
            self.q_right.put
        )

        self.gnss.listen(
            self.q_gnss.put
        )

        self.imu.listen(
            self.q_imu.put
        )

        # ----------------------------------------------------
        # Original NEAT route planner
        # ----------------------------------------------------

        gps_plan = (
            build_neat_gps_plan(
                carla_map,
                route,
            )
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
            "[NEAT adapter] "
            "RoutePlanner(4.0, 50.0)"
        )

        return [
            self.camera_front,
            self.camera_left,
            self.camera_right,
            self.gnss,
            self.imu,
        ]

    # --------------------------------------------------------
    # Exact synchronized sensor frame
    # --------------------------------------------------------

    def read_sensor_frame(
        self,
        target_frame,
    ):

        front = (
            get_sensor_frame(
                self.q_front,
                target_frame,
                "NEAT front",
            )
        )

        left = (
            get_sensor_frame(
                self.q_left,
                target_frame,
                "NEAT left",
            )
        )

        right = (
            get_sensor_frame(
                self.q_right,
                target_frame,
                "NEAT right",
            )
        )

        gnss = (
            get_sensor_frame(
                self.q_gnss,
                target_frame,
                "NEAT GNSS",
            )
        )

        imu = (
            get_sensor_frame(
                self.q_imu,
                target_frame,
                "NEAT IMU",
            )
        )

        return ModelSensorFrame(
            carla_frame=int(
                target_frame
            ),

            cameras={
                "front":
                    front,

                "left":
                    left,

                "right":
                    right,
            },

            auxiliary={
                "gnss":
                    gnss,

                "imu":
                    imu,
            },
        )

    # --------------------------------------------------------
    # One model step
    # --------------------------------------------------------

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

        gnss = (
            sensor_frame
            .auxiliary[
                "gnss"
            ]
        )

        imu = (
            sensor_frame
            .auxiliary[
                "imu"
            ]
        )

        # ----------------------------------------------------
        # Native NEAT navigation processing
        # ----------------------------------------------------

        nav = (
            get_neat_navigation(
                self.route_planner,
                gnss,
                imu,
            )
        )

        target_point = np.asarray(
            nav[
                "target_point"
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Native NEAT sequence bootstrap
        # ----------------------------------------------------

        bootstrap = (
            int(
                scenario_frame
            )
            <
            int(
                self.config.seq_len
            )
        )

        if bootstrap:

            result = (
                make_bootstrap_result()
            )

        else:

            result = (
                run_neat(
                    net=
                        self.net,

                    config=
                        self.config,

                    plan_grid=
                        self.plan_grid,

                    light_grid=
                        self.light_grid,

                    rgb_front=
                        rgb_by_camera[
                            "front"
                        ],

                    rgb_left=
                        rgb_by_camera[
                            "left"
                        ],

                    rgb_right=
                        rgb_by_camera[
                            "right"
                        ],

                    speed_mps=
                        float(
                            speed_mps
                        ),

                    target_xy=
                        target_point,

                    device=
                        self.device,
                )
            )

        # ----------------------------------------------------
        # Generic control result
        # ----------------------------------------------------

        control = (
            ModelControl(
                steer=float(
                    result[
                        "steer"
                    ]
                ),

                throttle=float(
                    result[
                        "throttle"
                    ]
                ),

                brake=float(
                    result[
                        "brake"
                    ]
                ),
            )
        )

        metadata = {}

        # Preserve result fields except huge waypoint arrays.
        for key, value in result.items():

            if key == "predicted_waypoints":
                continue

            if key == "metadata":
                continue

            metadata[key] = value

        native_metadata = (
            result.get(
                "metadata",
                {}
            )
        )

        metadata.update({
            "target_x":
                float(
                    target_point[0]
                ),

            "target_y":
                float(
                    target_point[1]
                ),

            "command_name":
                command_label(
                    nav[
                        "command"
                    ]
                ),

            "command_value":
                command_value(
                    nav[
                        "command"
                    ]
                ),

            "red_light_occ":
                int(
                    result.get(
                        "red_light_occ",
                        0,
                    )
                ),

            "desired_speed":
                scalar_float(
                    native_metadata.get(
                        "desired_speed"
                    )
                ),

            "angle":
                scalar_float(
                    native_metadata.get(
                        "angle"
                    )
                ),

            "angle_last":
                scalar_float(
                    native_metadata.get(
                        "angle_last"
                    )
                ),

            "angle_target":
                scalar_float(
                    native_metadata.get(
                        "angle_target"
                    )
                ),

            "angle_final":
                scalar_float(
                    native_metadata.get(
                        "angle_final"
                    )
                ),

            "navigation_position":
                (
                    np.asarray(
                        nav[
                            "position"
                        ]
                    )
                    .astype(
                        float
                    )
                    .tolist()
                ),
        })

        return ModelStepResult(
            control=
                control,

            metadata=
                metadata,

            bootstrap=
                bool(
                    bootstrap
                ),
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def summary(
        self,
    ):

        out = super().summary()

        out.update({
            "native_cameras":
                (
                    "3x 400x300 FOV100 "
                    "x=1.3 z=2.3 "
                    "yaw=0/-60/+60"
                ),

            "navigation":
                "NEAT RoutePlanner + GNSS + IMU",

            "control":
                "NEAT waypoint planner + native PID",
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

                "num_camera":
                    int(
                        self.config.num_camera
                    ),
            })

        return out