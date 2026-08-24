"""
tcp_adapter_v1.py

Thin TCP adapter for the generic Hallucination Engine experiment runtime.

TCP remains unchanged.

Native TCP sensors:
    RGB:
        900x256
        FOV 100
        x=-1.5
        y=0.0
        z=2.0

    GNSS
    IMU

The adapter reuses the already validated:
    load_tcp_model()
    TCPFusionState
    run_tcp_route()
    TCP RoutePlanner navigation processing
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

THIS_FILE = Path(
    __file__
).resolve()

HE_ROOT = (
    THIS_FILE
    .parents[2]
)

COMMON_DIR = (
    HE_ROOT
    / "driving_models"
    / "common"
)

if str(
    COMMON_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            COMMON_DIR
        ),
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
# Existing validated TCP implementation
# ============================================================

from tcp_carla_0915_probe import (
    DEFAULT_TCP_ROOT,
    DEFAULT_CHECKPOINT,
    load_tcp_model,
    make_camera,
)


from tcp_carla_0915_route_probe import (
    TCPFusionState,
    command_name,
    command_value,
    run_tcp_route,
)


from tcp_carla_0915_closed_loop import (
    import_tcp_route_planner,
    make_gnss,
    make_imu,
    get_named_sensor_frame,
    build_tcp_gps_plan,
    get_tcp_navigation,
)


# ============================================================
# TCP adapter
# ============================================================

class TCPAdapterV1(
    DrivingModelAdapter
):

    model_name = "tcp"

    required_fps = 20.0

    # --------------------------------------------------------
    # Native TCP camera
    # --------------------------------------------------------

    @classmethod
    def camera_specs(
        cls,
    ):

        return (
            CameraSpec(
                name="front",

                width=900,
                height=256,
                fov_deg=100.0,

                x_m=-1.5,
                y_m=0.0,
                z_m=2.0,

                pitch_deg=0.0,
                yaw_deg=0.0,
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

        self.fusion = None

        self.camera_front = None
        self.gnss = None
        self.imu = None

        self.q_front = None
        self.q_gnss = None
        self.q_imu = None

    # --------------------------------------------------------
    # Load TCP
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
                "TCP requested CUDA but CUDA is unavailable."
            )

        if (
            self.device.type
            == "cuda"
        ):

            print(
                "[TCP GPU]",
                torch.cuda.get_device_name(
                    0
                ),
            )

        tcp_root_value = (
            getattr(
                self.args,
                "tcp_root",
                None,
            )
            or
            DEFAULT_TCP_ROOT
        )

        tcp_root = Path(
            tcp_root_value
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
        ) = load_tcp_model(
            tcp_root,
            checkpoint,
            self.device,
        )

        # Original TCP agent steering-history state.
        self.fusion = (
            TCPFusionState()
        )

        self.RoutePlanner = (
            import_tcp_route_planner(
                tcp_root
            )
        )

        print(
            "[TCP adapter] loaded"
        )

        print(
            "[TCP adapter] camera:",
            "900x256 FOV100 "
            "x=-1.5 y=0 z=2.0",
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
                "TCPAdapterV1.load() "
                "must be called before setup()."
            )

        spec = (
            self.camera_specs()[0]
        )

        # ----------------------------------------------------
        # Native TCP RGB camera
        # ----------------------------------------------------

        self.camera_front = (
            make_camera(
                world=world,
                ego=ego,

                width=spec.width,
                height=spec.height,
                fov=spec.fov_deg,

                x=spec.x_m,
                y=spec.y_m,
                z=spec.z_m,

                pitch=spec.pitch_deg,
                yaw=spec.yaw_deg,
                roll=spec.roll_deg,
            )
        )

        # ----------------------------------------------------
        # Native navigation sensors
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

        self.q_gnss = (
            queue.Queue()
        )

        self.q_imu = (
            queue.Queue()
        )

        self.camera_front.listen(
            self.q_front.put
        )

        self.gnss.listen(
            self.q_gnss.put
        )

        self.imu.listen(
            self.q_imu.put
        )

        # ----------------------------------------------------
        # Original TCP RoutePlanner
        # ----------------------------------------------------

        gps_plan = (
            build_tcp_gps_plan(
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
            "[TCP adapter] "
            "RoutePlanner(4.0, 50.0)"
        )

        return [
            self.camera_front,
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

        if (
            self.q_front is None
            or
            self.q_gnss is None
            or
            self.q_imu is None
        ):

            raise RuntimeError(
                "TCP sensors are not initialized."
            )

        front = (
            get_named_sensor_frame(
                self.q_front,
                target_frame,
                "TCP front",
            )
        )

        gnss = (
            get_named_sensor_frame(
                self.q_gnss,
                target_frame,
                "TCP GNSS",
            )
        )

        imu = (
            get_named_sensor_frame(
                self.q_imu,
                target_frame,
                "TCP IMU",
            )
        )

        return ModelSensorFrame(
            carla_frame=int(
                target_frame
            ),

            cameras={
                "front":
                    front,
            },

            auxiliary={
                "gnss":
                    gnss,

                "imu":
                    imu,
            },
        )

    # --------------------------------------------------------
    # TCP inference
    # --------------------------------------------------------

    def step(
        self,
        rgb_by_camera,
        sensor_frame,
        speed_mps,
        scenario_frame,
    ):

        del scenario_frame

        validate_camera_rgb_keys(
            self,
            rgb_by_camera,
        )

        if self.route_planner is None:

            raise RuntimeError(
                "TCP route planner is not initialized."
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
        # Original TCP navigation preprocessing
        # ----------------------------------------------------

        nav = (
            get_tcp_navigation(
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

        command = (
            nav[
                "command"
            ]
        )

        cmd_value = (
            command_value(
                command
            )
        )

        # ----------------------------------------------------
        # Existing validated TCP inference + control fusion
        # ----------------------------------------------------

        result = (
            run_tcp_route(
                net=
                    self.net,

                rgb=
                    rgb_by_camera[
                        "front"
                    ],

                speed_mps=
                    float(
                        speed_mps
                    ),

                target_xy=
                    target_point,

                cmd_value=
                    cmd_value,

                device=
                    self.device,

                fusion=
                    self.fusion,
            )
        )

        # ----------------------------------------------------
        # Standard generic control
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

        # ----------------------------------------------------
        # Preserve model-specific diagnostics
        # ----------------------------------------------------

        metadata = dict(
            result
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
                command_name(
                    command
                ),

            "command_value":
                int(
                    cmd_value
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
                False,
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def summary(
        self,
    ):

        out = super().summary()

        out.update({
            "native_camera":
                "900x256_fov100_x-1.5_z2.0",

            "navigation":
                "TCP RoutePlanner + GNSS + IMU",

            "control":
                "TCP direct-control + trajectory/PID fusion",
        })

        return out