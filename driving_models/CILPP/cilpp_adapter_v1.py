"""
cilpp_adapter_v1.py

CIL++ adapter for the generic Hallucination Engine closed-loop runtime.

Official pretrained model:
    CILv2_multiview_attention
    Town12346_5
    checkpoint 40

Native sensors:
    rgb_left:
        300x300
        FOV 60
        x=0.0, y=0.0, z=2.0
        yaw=-60

    rgb_central:
        300x300
        FOV 60
        x=0.0, y=0.0, z=2.0
        yaw=0

    rgb_right:
        300x300
        FOV 60
        x=0.0, y=0.0, z=2.0
        yaw=+60

    GNSS
    IMU

Navigation:
    Original CIL++ Waypointer.tick_lb()
    6-class command encoding:
        1 LEFT
        2 RIGHT
        3 STRAIGHT
        4 LANEFOLLOW
        5 CHANGELANELEFT
        6 CHANGELANERIGHT

Control:
    network outputs:
        steer
        acceleration

    acceleration >= 0:
        throttle = acceleration
        brake = 0

    acceleration < 0:
        throttle = 0
        brake = abs(acceleration)
"""

from __future__ import annotations

import json
import os
import queue
import sys
from pathlib import Path

import carla
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


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

DEFAULT_CILPP_ROOT = (
    HE_ROOT
    / "external_models"
    / "CILPP"
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
# Helpers
# ============================================================

def get_sensor_frame(
    sensor_queue,
    target_frame,
    name,
    timeout=10.0,
):
    """
    Return exactly target_frame from a CARLA sensor queue.

    Old frames are discarded.

    A future frame indicates synchronization was lost.
    """

    while True:

        item = sensor_queue.get(
            timeout=timeout
        )

        frame = int(
            item.frame
        )

        if frame < int(target_frame):
            continue

        if frame > int(target_frame):

            raise RuntimeError(
                f"{name}: expected CARLA frame "
                f"{target_frame}, got future frame {frame}."
            )

        return item


def make_camera(
    world,
    ego,
    spec,
):

    bp = (
        world
        .get_blueprint_library()
        .find(
            "sensor.camera.rgb"
        )
    )

    bp.set_attribute(
        "image_size_x",
        str(
            int(spec.width)
        ),
    )

    bp.set_attribute(
        "image_size_y",
        str(
            int(spec.height)
        ),
    )

    bp.set_attribute(
        "fov",
        str(
            float(spec.fov_deg)
        ),
    )

    transform = carla.Transform(
        carla.Location(
            x=float(spec.x_m),
            y=float(spec.y_m),
            z=float(spec.z_m),
        ),
        carla.Rotation(
            pitch=float(spec.pitch_deg),
            yaw=float(spec.yaw_deg),
            roll=float(spec.roll_deg),
        ),
    )

    return world.spawn_actor(
        bp,
        transform,
        attach_to=ego,
    )


def make_gnss(
    world,
    ego,
):

    bp = (
        world
        .get_blueprint_library()
        .find(
            "sensor.other.gnss"
        )
    )

    return world.spawn_actor(
        bp,
        carla.Transform(),
        attach_to=ego,
    )


def make_imu(
    world,
    ego,
):

    bp = (
        world
        .get_blueprint_library()
        .find(
            "sensor.other.imu"
        )
    )

    return world.spawn_actor(
        bp,
        carla.Transform(),
        attach_to=ego,
    )


def gnss_to_array(
    measurement,
):

    return np.asarray(
        [
            float(measurement.latitude),
            float(measurement.longitude),
            float(measurement.altitude),
        ],
        dtype=np.float64,
    )


def imu_to_array(
    measurement,
):
    """
    Match the ordering expected by CIL++.

    Waypointer only relies on the final value:
        compass
    """

    return np.asarray(
        [
            float(
                measurement.accelerometer.x
            ),
            float(
                measurement.accelerometer.y
            ),
            float(
                measurement.accelerometer.z
            ),

            float(
                measurement.gyroscope.x
            ),
            float(
                measurement.gyroscope.y
            ),
            float(
                measurement.gyroscope.z
            ),

            float(
                measurement.compass
            ),
        ],
        dtype=np.float64,
    )


def carla_route_to_cilpp_route(
    route,
):
    """
    Generic runner route:
        [(carla.Waypoint, RoadOption), ...]

    CIL++ route:
        [(carla.Transform, RoadOption), ...]
    """

    out = []

    for point in route:

        if (
            not isinstance(
                point,
                (tuple, list),
            )
            or
            len(point) < 2
        ):

            raise TypeError(
                "Unexpected generic route item: "
                f"{point!r}"
            )

        waypoint_or_transform = (
            point[0]
        )

        road_option = (
            point[1]
        )

        transform = getattr(
            waypoint_or_transform,
            "transform",
            None,
        )

        if transform is None:

            if isinstance(
                waypoint_or_transform,
                carla.Transform,
            ):

                transform = (
                    waypoint_or_transform
                )

            else:

                raise TypeError(
                    "CIL++ expected route point "
                    "to contain a CARLA Waypoint "
                    "or Transform."
                )

        out.append(
            (
                transform,
                road_option,
            )
        )

    return out


def command_name(
    command,
):

    name = getattr(
        command,
        "name",
        None,
    )

    if name is not None:
        return str(name)

    return str(command)


def command_value(
    command,
):

    value = getattr(
        command,
        "value",
        command,
    )

    return int(
        value
    )


# ============================================================
# CIL++ adapter
# ============================================================

class CILPPAdapterV1(
    DrivingModelAdapter
):

    model_name = "cilpp"

    # Official CIL++ Town05 evaluation uses --fps=20.
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
                name="rgb_left",

                width=300,
                height=300,
                fov_deg=60.0,

                x_m=0.0,
                y_m=0.0,
                z_m=2.0,

                pitch_deg=0.0,
                yaw_deg=-60.0,
                roll_deg=0.0,
            ),

            CameraSpec(
                name="rgb_central",

                width=300,
                height=300,
                fov_deg=60.0,

                x_m=0.0,
                y_m=0.0,
                z_m=2.0,

                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
            ),

            CameraSpec(
                name="rgb_right",

                width=300,
                height=300,
                fov_deg=60.0,

                x_m=0.0,
                y_m=0.0,
                z_m=2.0,

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

        self.cilpp_root = None
        self.config_json = None
        self.yaml_path = None
        self.checkpoint_path = None

        self.g_conf = None
        self.model = None

        self.world = None
        self.ego = None

        self.waypointer = None

        self.cameras = {}

        self.gnss = None
        self.imu = None

        self.camera_queues = {}

        self.q_gnss = None
        self.q_imu = None

    # --------------------------------------------------------
    # Load checkpoint/model
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
                "CIL++ requested CUDA "
                "but CUDA is unavailable."
            )

        if self.device.type == "cuda":

            print(
                "[CIL++ GPU]",
                torch.cuda.get_device_name(
                    0
                ),
            )

        # ----------------------------------------------------
        # Root
        # ----------------------------------------------------

        cilpp_root_value = (
            getattr(
                self.args,
                "cilpp_root",
                None,
            )
            or
            os.environ.get(
                "HE_CILPP_ROOT"
            )
            or
            DEFAULT_CILPP_ROOT
        )

        self.cilpp_root = Path(
            cilpp_root_value
        ).resolve()

        if not self.cilpp_root.exists():

            raise FileNotFoundError(
                "CIL++ root does not exist:\n"
                f"{self.cilpp_root}"
            )

        # Make official CIL++ modules importable.
        root_text = str(
            self.cilpp_root
        )

        if root_text not in sys.path:
            sys.path.insert(
                0,
                root_text,
            )

        driving_root = (
            self.cilpp_root
            / "run_CARLA_driving"
        )

        driving_text = str(
            driving_root
        )

        if driving_text not in sys.path:
            sys.path.insert(
                0,
                driving_text,
            )

        # ----------------------------------------------------
        # Official pretrained selector
        # ----------------------------------------------------

        default_config = (
            self.cilpp_root
            / "_results"
            / "Ours"
            / "Town12346_5"
            / "config40.json"
        )

        checkpoint_arg = getattr(
            self.args,
            "checkpoint",
            None,
        )

        if checkpoint_arg:

            candidate = Path(
                checkpoint_arg
            ).resolve()

            if (
                candidate.suffix.lower()
                != ".json"
            ):

                raise ValueError(
                    "For CIL++, --checkpoint "
                    "must point to the official "
                    "configXX.json selector, e.g. "
                    "config40.json."
                )

            self.config_json = (
                candidate
            )

        else:

            self.config_json = (
                default_config
            )

        if not self.config_json.exists():

            raise FileNotFoundError(
                "CIL++ config selector "
                "not found:\n"
                f"{self.config_json}"
            )

        with self.config_json.open(
            "r",
            encoding="utf-8",
        ) as f:

            selector = json.load(
                f
            )

        experiment_dir = (
            self.config_json.parent
        )

        self.yaml_path = (
            experiment_dir
            / selector["yaml"]
        ).resolve()

        checkpoint_number = int(
            selector["checkpoint"]
        )

        # ----------------------------------------------------
        # Official CIL++ global configuration
        # ----------------------------------------------------

        from configs import (
            g_conf,
            merge_with_yaml,
            set_type_of_process,
        )

        g_conf.immutable(
            False
        )

        merge_with_yaml(
            str(
                self.yaml_path
            ),
            process_type="drive",
        )

        (
            self.cilpp_root
            / "_results"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        set_type_of_process(
            "drive",
            root=str(
                self.cilpp_root
            ),
        )

        self.g_conf = (
            g_conf
        )

        # ----------------------------------------------------
        # Construct official network
        # ----------------------------------------------------

        from network.models_console import (
            Models,
        )

        self.model = Models(
            self.g_conf.MODEL_TYPE,
            self.g_conf.MODEL_CONFIGURATION,
        )

        self.checkpoint_path = (
            experiment_dir
            / "checkpoints"
            / (
                f"{self.model.name}_"
                f"{checkpoint_number}.pth"
            )
        ).resolve()

        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                "CIL++ checkpoint "
                "not found:\n"
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if "model" not in checkpoint:

            raise KeyError(
                "Expected CIL++ "
                "checkpoint['model']."
            )

        load_result = (
            self.model
            .load_state_dict(
                checkpoint[
                    "model"
                ],
                strict=True,
            )
        )

        print(
            "[CIL++ load]",
            load_result,
        )

        self.model = (
            self.model
            .to(
                self.device
            )
        )

        self.model.eval()

        print(
            "[CIL++ adapter] loaded"
        )

        print(
            "[CIL++ adapter] model:",
            self.g_conf.MODEL_TYPE,
        )

        print(
            "[CIL++ adapter] cameras:",
            list(
                self.g_conf.DATA_USED
            ),
        )

        print(
            "[CIL++ adapter] command classes:",
            self.g_conf.DATA_COMMAND_CLASS_NUM,
        )

    # --------------------------------------------------------
    # Native sensors + route command system
    # --------------------------------------------------------

    def setup(
        self,
        world,
        ego,
        carla_map,
        route,
    ):

        if self.model is None:

            raise RuntimeError(
                "CILPPAdapterV1.load() "
                "must be called before setup()."
            )

        self.world = world
        self.ego = ego

        # ----------------------------------------------------
        # Native cameras
        # ----------------------------------------------------

        sensors = []

        for spec in self.camera_specs():

            camera = make_camera(
                world,
                ego,
                spec,
            )

            q = queue.Queue()

            camera.listen(
                q.put
            )

            self.cameras[
                spec.name
            ] = camera

            self.camera_queues[
                spec.name
            ] = q

            sensors.append(
                camera
            )

        # ----------------------------------------------------
        # GNSS + IMU
        # ----------------------------------------------------

        self.gnss = make_gnss(
            world,
            ego,
        )

        self.imu = make_imu(
            world,
            ego,
        )

        self.q_gnss = (
            queue.Queue()
        )

        self.q_imu = (
            queue.Queue()
        )

        self.gnss.listen(
            self.q_gnss.put
        )

        self.imu.listen(
            self.q_imu.put
        )

        sensors.extend(
            [
                self.gnss,
                self.imu,
            ]
        )

        # ----------------------------------------------------
        # Original CIL++ route representation
        # ----------------------------------------------------

        from driving.utils.route_manipulation import (
            _get_latlon_ref,
            downsample_route,
            location_route_to_gps,
        )

        from driving.utils.waypointer import (
            Waypointer,
        )

        global_route = (
            carla_route_to_cilpp_route(
                route
            )
        )

        lat_ref, lon_ref = (
            _get_latlon_ref(
                world
            )
        )

        full_gps_plan = (
            location_route_to_gps(
                global_route,
                lat_ref,
                lon_ref,
            )
        )

        # Match CILv2_agent.set_global_plan():
        #
        #   GPS plan is downsampled at 50 m,
        #   while Waypointer receives the full world route.
        ds_ids = (
            downsample_route(
                global_route,
                50,
            )
        )

        global_plan_gps = [
            full_gps_plan[i]
            for i in ds_ids
        ]

        self.waypointer = (
            Waypointer(
                world,
                global_plan_gps=
                    global_plan_gps,

                global_route=
                    global_route,
            )
        )

        print(
            "[CIL++ adapter] "
            "Waypointer.tick_lb()"
        )

        print(
            "[CIL++ adapter] "
            f"route points={len(global_route)} "
            f"gps checkpoints={len(global_plan_gps)}"
        )

        return sensors

    # --------------------------------------------------------
    # Exact synchronized sensor frame
    # --------------------------------------------------------

    def read_sensor_frame(
        self,
        target_frame,
    ):

        camera_frames = {}

        for name in self.camera_names():

            camera_frames[
                name
            ] = get_sensor_frame(
                self.camera_queues[
                    name
                ],
                target_frame,
                f"CIL++ {name}",
            )

        gnss = get_sensor_frame(
            self.q_gnss,
            target_frame,
            "CIL++ GNSS",
        )

        imu = get_sensor_frame(
            self.q_imu,
            target_frame,
            "CIL++ IMU",
        )

        return ModelSensorFrame(
            carla_frame=int(
                target_frame
            ),

            cameras=
                camera_frames,

            auxiliary={
                "gnss":
                    gnss,

                "imu":
                    imu,
            },
        )

    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    def preprocess_image(
        self,
        rgb,
    ):

        image = Image.fromarray(
            np.asarray(
                rgb,
                dtype=np.uint8,
            )
        )

        image = image.resize(
            (
                int(
                    self.g_conf
                    .IMAGE_SHAPE[2]
                ),
                int(
                    self.g_conf
                    .IMAGE_SHAPE[1]
                ),
            )
        ).convert(
            "RGB"
        )

        tensor = TF.to_tensor(
            image
        )

        tensor = TF.normalize(
            tensor,
            mean=
                self.g_conf
                .IMG_NORMALIZATION[
                    "mean"
                ],

            std=
                self.g_conf
                .IMG_NORMALIZATION[
                    "std"
                ],
        )

        return (
            tensor
            .unsqueeze(0)
            .to(
                device=
                    self.device,

                dtype=
                    torch.float32,
            )
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

        # ----------------------------------------------------
        # Native CIL++ image ordering
        # ----------------------------------------------------

        image_tensors = []

        for camera_name in (
            self.g_conf.DATA_USED
        ):

            if (
                camera_name
                not in rgb_by_camera
            ):

                raise KeyError(
                    "CIL++ expected camera "
                    f"{camera_name!r}."
                )

            image_tensors.append(
                self.preprocess_image(
                    rgb_by_camera[
                        camera_name
                    ]
                )
            )

        # Exact CIL++ nesting:
        #
        #   [[left, central, right]]
        norm_rgb = [
            image_tensors
        ]

        # ----------------------------------------------------
        # GNSS / IMU
        # ----------------------------------------------------

        gnss_data = (
            gnss_to_array(
                sensor_frame
                .auxiliary[
                    "gnss"
                ]
            )
        )

        imu_data = (
            imu_to_array(
                sensor_frame
                .auxiliary[
                    "imu"
                ]
            )
        )

        # ----------------------------------------------------
        # Original CIL++ 6-way route command
        # ----------------------------------------------------

        checkpoint = (
            self.waypointer
            .tick_lb(
                gnss_data,
                imu_data,
            )
        )

        command = (
            checkpoint[2]
        )

        cmd_value = (
            command_value(
                command
            )
        )

        if (
            int(
                self.g_conf
                .DATA_COMMAND_CLASS_NUM
            )
            != 6
        ):

            raise RuntimeError(
                "This adapter expects the "
                "Town12346_5 six-command "
                "CIL++ checkpoint."
            )

        from dataloaders.transforms import (
            encode_directions_6,
        )

        one_hot = (
            encode_directions_6(
                cmd_value
            )
        )

        direction = [
            torch.tensor(
                [
                    one_hot
                ],
                dtype=
                    torch.float32,

                device=
                    self.device,
            )
        ]

        # ----------------------------------------------------
        # Exact CIL++ speed normalization
        # ----------------------------------------------------

        speed_min = float(
            self.g_conf
            .DATA_NORMALIZATION[
                "speed"
            ][0]
        )

        speed_max = float(
            self.g_conf
            .DATA_NORMALIZATION[
                "speed"
            ][1]
        )

        normalized_speed = (
            abs(
                float(speed_mps)
                -
                speed_min
            )
            /
            (
                speed_max
                -
                speed_min
            )
        )

        norm_speed = [
            torch.tensor(
                [
                    [
                        normalized_speed
                    ]
                ],
                dtype=
                    torch.float32,

                device=
                    self.device,
            )
        ]

        # ----------------------------------------------------
        # CIL++ inference
        # ----------------------------------------------------

        with torch.no_grad():

            (
                actions_outputs,
                _,
                attention_weights,
            ) = (
                self.model
                .forward_eval(
                    norm_rgb,
                    direction,
                    norm_speed,
                )
            )

        raw = (
            actions_outputs
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        if np.asarray(
            raw
        ).size < 2:

            raise RuntimeError(
                "Unexpected CIL++ output: "
                f"{raw!r}"
            )

        steer = float(
            raw[0]
        )

        acceleration = float(
            raw[1]
        )

        # ----------------------------------------------------
        # Exact official control conversion
        # ----------------------------------------------------

        if acceleration >= 0.0:

            throttle = (
                acceleration
            )

            brake = 0.0

        else:

            throttle = 0.0

            brake = abs(
                acceleration
            )

        steer = float(
            np.clip(
                steer,
                -1.0,
                1.0,
            )
        )

        throttle = float(
            np.clip(
                throttle,
                0.0,
                1.0,
            )
        )

        brake = float(
            np.clip(
                brake,
                0.0,
                1.0,
            )
        )

        return ModelStepResult(
            control=
                ModelControl(
                    steer=
                        steer,

                    throttle=
                        throttle,

                    brake=
                        brake,
                ),

            metadata={
                "acceleration":
                    acceleration,

                "command_name":
                    command_name(
                        command
                    ),

                "command_value":
                    cmd_value,

                "speed_normalized":
                    normalized_speed,

                "attention_type":
                    str(
                        type(
                            attention_weights
                        ).__name__
                    ),
            },

            bootstrap=False,
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def summary(
        self,
    ):

        out = super().summary()

        out.update({
            "checkpoint":
                (
                    str(
                        self.checkpoint_path
                    )
                    if
                    self.checkpoint_path
                    is not None
                    else
                    None
                ),

            "native_cameras":
                (
                    "3x 300x300 FOV60 "
                    "x=0 z=2 "
                    "yaw=-60/0/+60"
                ),

            "navigation":
                (
                    "CIL++ Waypointer.tick_lb "
                    "+ GNSS + IMU "
                    "+ 6-way command"
                ),

            "control":
                (
                    "CIL++ direct "
                    "steer+acceleration"
                ),
        })

        return out

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def close(
        self,
    ):

        self.model = None
        self.waypointer = None

        if (
            self.device is not None
            and
            self.device.type == "cuda"
        ):

            torch.cuda.empty_cache()


# ============================================================
# Dynamic adapter discovery
# ============================================================

ADAPTER_CLASS = CILPPAdapterV1