"""
driving_model_adapter_v1.py

Generic model interface for Hallucination Engine experiments.

The generic experiment runner owns:
    - CARLA world / synchronous stepping
    - ego spawn + canonical initialization
    - route construction and common route metrics
    - ScenarioGenerator execution runtime
    - CARLA vs HE actor backend
    - HE multi-actor compositor
    - common logging
    - stopping criteria

A model adapter owns only:
    - native model sensors
    - checkpoint/model loading
    - native preprocessing
    - native navigation representation
    - inference
    - model-specific PID/control processing
    - model-specific diagnostics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple


# ============================================================
# Native camera specification
# ============================================================

@dataclass(frozen=True)
class CameraSpec:
    name: str

    width: int
    height: int
    fov_deg: float

    x_m: float
    y_m: float
    z_m: float

    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0


# ============================================================
# One synchronized sensor observation
# ============================================================

@dataclass
class ModelSensorFrame:
    carla_frame: int

    # Raw CARLA sensor objects.
    cameras: Dict[str, Any]

    # GNSS / IMU / speed sensor / etc.
    auxiliary: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Standard model output
# ============================================================

@dataclass(frozen=True)
class ModelControl:
    steer: float
    throttle: float
    brake: float


@dataclass
class ModelStepResult:
    control: ModelControl

    # Anything model-specific:
    # predicted speed, desired speed, waypoints,
    # red-light probability, fusion state, etc.
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    bootstrap: bool = False


# ============================================================
# Generic model adapter
# ============================================================

class DrivingModelAdapter(ABC):

    model_name: str = "unknown"

    required_fps: float = 20.0

    def __init__(
        self,
        args: Any,
    ):
        self.args = args

    # --------------------------------------------------------
    # Camera declaration
    # --------------------------------------------------------

    @classmethod
    @abstractmethod
    def camera_specs(
        cls,
    ) -> Tuple[CameraSpec, ...]:
        """
        Declare every native RGB camera used by the model.

        The generic runtime uses these specifications for:
            - CARLA camera creation
            - HE rendering
            - validation
            - debugging
        """
        raise NotImplementedError

    @classmethod
    def camera_names(
        cls,
    ) -> Tuple[str, ...]:

        return tuple(
            spec.name
            for spec
            in cls.camera_specs()
        )

    # --------------------------------------------------------
    # Model lifecycle
    # --------------------------------------------------------

    @abstractmethod
    def load(
        self,
    ) -> None:
        """
        Load checkpoint/config/model-only state.
        """
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        world: Any,
        ego: Any,
        carla_map: Any,
        route: Sequence[Any],
    ) -> Sequence[Any]:
        """
        Spawn/listen to native model sensors.

        Returns every spawned sensor actor so that the GENERIC
        runner can destroy them during cleanup.
        """
        raise NotImplementedError

    # --------------------------------------------------------
    # Synchronized observation
    # --------------------------------------------------------

    @abstractmethod
    def read_sensor_frame(
        self,
        target_frame: int,
    ) -> ModelSensorFrame:
        """
        Return exactly target_frame from every sensor.
        """
        raise NotImplementedError

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    @abstractmethod
    def step(
        self,
        rgb_by_camera: Mapping[str, Any],
        sensor_frame: ModelSensorFrame,
        speed_mps: float,
        scenario_frame: int,
    ) -> ModelStepResult:
        """
        Run one native model inference step.

        IMPORTANT:

        rgb_by_camera may contain either:

            real CARLA RGB
                OR
            HE-composited RGB.

        The model adapter MUST NOT care which condition generated it.
        """
        raise NotImplementedError

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "model":
                self.model_name,

            "required_fps":
                float(
                    self.required_fps
                ),

            "camera_names":
                list(
                    self.camera_names()
                ),

            "camera_count":
                len(
                    self.camera_specs()
                ),
        }

    # --------------------------------------------------------
    # Optional model-only cleanup
    # --------------------------------------------------------

    def close(
        self,
    ) -> None:
        return None


# ============================================================
# Contract validation
# ============================================================

def validate_camera_rgb_keys(
    adapter: DrivingModelAdapter,
    rgb_by_camera: Mapping[str, Any],
) -> None:

    expected = set(
        adapter.camera_names()
    )

    actual = set(
        rgb_by_camera.keys()
    )

    if actual == expected:
        return

    missing = sorted(
        expected - actual
    )

    extra = sorted(
        actual - expected
    )

    raise KeyError(
        f"{adapter.model_name}: RGB camera keys "
        f"do not match native sensor contract. "
        f"missing={missing}, extra={extra}"
    )