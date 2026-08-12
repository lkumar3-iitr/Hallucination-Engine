from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class ActorRole(str, Enum):
    EGO = "ego"
    ADVERSARY = "adversary"
    TRAFFIC = "traffic"
    PEDESTRIAN = "pedestrian"


class ActorType(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"


class ManeuverType(str, Enum):
    STATIC = "static"
    ONCOMING = "oncoming"
    CUT_IN = "cut_in"
    FOLLOWING = "following"
    CROSSING = "crossing"


class TrajectoryMode(str, Enum):
    RULE_BASED = "rule_based"
    OPTIMIZATION_BASED = "optimization_based"
    LEARNING_BASED = "learning_based"

class EgoMotionType(str, Enum):
    STATIC = "static"
    STRAIGHT = "straight"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"

class EgoMotionSpec(BaseModel):
    """
    Ego trajectory description before resolution.

    Coordinate convention:
    - x: forward
    - y: lateral, positive left
    - yaw_deg: positive means turning left
    """

    motion: EgoMotionType = EgoMotionType.STRAIGHT

    turn_start_s: float = 0.0
    turn_duration_s: float = 0.0
    turn_yaw_deg: float = 0.0

class CameraConfig(BaseModel):
    """Fixed camera convention used by HEPlacementModel v2."""

    image_width: int = 1280
    image_height: int = 720
    fov: float = 90.0

    camera_x: float = 1.5
    camera_y: float = 0.0
    camera_z: float = 1.6

    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

class CameraViewConfig(BaseModel):
    """Single camera view in an ego-mounted multi-camera rig.

    yaw_deg convention:
    - 0 deg: front
    - 90 deg: left
    - -90 deg: right
    - 180 deg: rear
    """

    camera_id: str
    image_width: int = 1280
    image_height: int = 720
    fov: float = 90.0

    camera_x: float = 1.5
    camera_y: float = 0.0
    camera_z: float = 1.6

    pitch: float = 0.0
    yaw_deg: float = 0.0
    roll: float = 0.0


class CameraRigConfig(BaseModel):
    """Optional future multi-camera / 360 camera rig.

    For current HE v1 we still use CameraConfig as the main fixed front camera.
    CameraRigConfig is for future BEV memory and multi-view rendering.
    """

    rig_id: str = "ego_camera_rig_v1"
    views: List[CameraViewConfig] = Field(default_factory=list)

    @staticmethod
    def default_four_view_360() -> "CameraRigConfig":
        return CameraRigConfig(
            rig_id="ego_four_view_360_v1",
            views=[
                CameraViewConfig(camera_id="front", yaw_deg=0.0),
                CameraViewConfig(camera_id="left", yaw_deg=90.0),
                CameraViewConfig(camera_id="right", yaw_deg=-90.0),
                CameraViewConfig(camera_id="rear", yaw_deg=180.0),
            ],
        )
class EgoConfig(BaseModel):
    """
    Ego initial state and requested motion in local BEV coordinates.

    Coordinate convention:
    - x: forward
    - y: lateral, positive left
    - yaw_deg=0: forward
    - positive yaw: left turn
    - negative yaw: right turn
    """

    initial_x_m: float = 0.0
    initial_y_m: float = 0.0
    initial_yaw_deg: float = 0.0
    speed_mps: float = 5.0

    motion: EgoMotionSpec = Field(default_factory=EgoMotionSpec)


class RoadConfig(BaseModel):
    lane_width_m: float = 3.5
    num_lanes_same_direction: int = 1
    num_lanes_opposite_direction: int = 1
    ego_lane_index: int = 0
    road_length_m: float = 120.0


class TrajectorySpec(BaseModel):
    mode: TrajectoryMode = TrajectoryMode.RULE_BASED
    maneuver: ManeuverType
    start_time_s: float = 0.0
    end_time_s: Optional[float] = None

    # Generic maneuver parameters. Keep flexible for v1.
    params: Dict[str, float | int | str | bool] = Field(default_factory=dict)


class ActorSpec(BaseModel):
    actor_id: str
    role: ActorRole = ActorRole.ADVERSARY
    actor_type: ActorType = ActorType.VEHICLE
    blueprint: str = "vehicle.generic"

    initial_x_m: float
    initial_y_m: float
    initial_yaw_deg: float = 0.0
    initial_speed_mps: float = 0.0

    dimensions_m: Tuple[float, float, float] = (4.5, 1.8, 1.6)  # length, width, height
    trajectory: TrajectorySpec


class ScenarioSpec(BaseModel):
    scenario_id: str
    description: str = ""
    duration_s: float = 8.0
    fps: int = 10

    # Current HEPlacementModel v2 front-camera convention.
    camera: CameraConfig = Field(default_factory=CameraConfig)

    # Optional future multi-camera / 360 rig.
    # Do not use this in current HE v1 unless needed.
    camera_rig: Optional[CameraRigConfig] = None

    ego: EgoConfig = Field(default_factory=EgoConfig)
    road: RoadConfig = Field(default_factory=RoadConfig)
    actors: List[ActorSpec]

    @model_validator(mode="after")
    def validate_duration_and_fps(self) -> "ScenarioSpec":
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.actors) == 0:
            raise ValueError("at least one actor is required")
        return self


class ResolvedEgoFrame(BaseModel):
    """
    Backend-independent resolved ego state.

    All coordinates are expressed in the scenario's local
    ego-initial BEV coordinate frame.
    """

    frame_idx: int
    t_s: float

    x_m: float
    y_m: float
    yaw_deg: float

    speed_mps: float

    vx_mps: float
    vy_mps: float


class ResolvedActorInfo(BaseModel):
    """
    Static actor information that does not need to be repeated
    in every frame.
    """

    actor_id: str
    role: ActorRole
    actor_type: ActorType
    blueprint: str

    dimensions_m: Tuple[float, float, float]


class ResolvedActorFrame(BaseModel):
    """
    Backend-independent physical actor state.

    IMPORTANT:
    No camera visibility, bbox, placement, sprite, or rendering
    information belongs here.
    """

    frame_idx: int
    t_s: float
    actor_id: str

    x_m: float
    y_m: float
    yaw_deg: float

    speed_mps: float

    vx_mps: float
    vy_mps: float


class ResolvedScenario(BaseModel):
    """
    Fully resolved physical scenario shared by all execution backends.

    Both CARLA and HE must consume this same representation.
    """

    scenario_id: str
    source_description: str

    duration_s: float
    fps: int

    # All resolved poses currently use the initial ego frame.
    coordinate_frame: Literal["ego_initial"] = "ego_initial"

    camera: CameraConfig
    camera_rig: Optional[CameraRigConfig] = None

    road: RoadConfig

    # Static actor properties.
    actors: List[ResolvedActorInfo]

    # Physical ego trajectory.
    ego_frames: List[ResolvedEgoFrame]

    # Physical trajectories of all non-ego actors.
    frames: List[ResolvedActorFrame]
