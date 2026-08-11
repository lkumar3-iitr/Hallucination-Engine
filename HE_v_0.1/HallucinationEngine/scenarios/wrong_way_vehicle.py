from .base_scenario import BaseScenario

from HallucinationEngine.placement.lane_aware_placement import LaneAwarePlacement
from HallucinationEngine.placement.projection_utils import (
    project_3d_box_to_2d,
)
from HallucinationEngine.motion.adversary_state import AdversaryState


class WrongWayVehicleScenario(BaseScenario):
    """
    Wrong-way hallucinated vehicle scenario.

    Version 3:
    - Maintains persistent adversary state.
    - Updates distance using motion state.
    - Uses lane-aware placement.
    - Projects full 3D cuboid into 2D image.
    """

    def __init__(self, config):
        super().__init__(config)

        self.placement = LaneAwarePlacement(config)
        self.adversary_state = None

    def update(self, metadata):
        frame_id = metadata.get("frame", 0)
        dt = metadata.get("dt", 1.0 / 20.0)

        if not self.active and self.should_start(frame_id):
            self.active = True
            self.start_frame = frame_id
            self.adversary_state = self._create_adversary_state()

        if not self.active:
            return None

        if not self.is_within_duration(frame_id):
            self.active = False

            if self.adversary_state is not None:
                self.adversary_state.active = False

            return None

        if self.adversary_state is None:
            self.adversary_state = self._create_adversary_state()

        # Step 1: update adversary motion state
        closing_speed_mps = self._compute_closing_speed(metadata)

        self.adversary_state.update_motion(
            dt=dt,
            closing_speed_mps=closing_speed_mps
        )

        # Step 2: choose corresponding road location
        placement = self.placement.choose_location_ahead(
            metadata=metadata,
            distance_ahead_m=self.adversary_state.distance_ahead_m
        )

        # Step 3: update adversary world pose
        self.adversary_state.update_placement(
            placement=placement,
            lateral_offset_m=self.config.lateral_offset_m
        )

        # Step 4: project to image
        if self.adversary_state.world_xyz is None:
            box_2d = list(self.config.default_box_2d)
            projection_status = "fallback_no_placement"
        else:
            box_2d = self._project_to_box(
                world_xyz=self.adversary_state.world_xyz,
                yaw=self.adversary_state.yaw,
                metadata=metadata
            )

            if box_2d is None:
                box_2d = list(self.config.default_box_2d)
                projection_status = "fallback_projection_failed"
            else:
                projection_status = "projected"

        # Step 5: build metadata
        hallucination = self.adversary_state.to_dict()

        hallucination.update({
            "box_2d": box_2d,
            "motion_type": "persistent_wrong_way_approach",
            "relative_speed_mps": closing_speed_mps,
            "ego_forward_speed_mps": self._compute_ego_forward_speed(metadata),
            "adversary_speed_mps": getattr(self.config, "adversary_speed_mps", self.config.relative_speed_mps),
            "render_mode": self.config.render_mode,
            "confidence": 1.0,
            "projection_status": projection_status,
        })

        return hallucination


    def _compute_closing_speed(self, metadata):
        """
        Compute ego-aware closing speed for wrong-way vehicle.

        wrong_way closing speed:
            ego forward speed + adversary speed

        If disabled, fallback to config.relative_speed_mps.
        """

        use_ego_speed = getattr(
            self.config,
            "use_ego_speed_for_relative_motion",
            False
        )

        if not use_ego_speed:
            return float(self.config.relative_speed_mps)

        ego_speed = self._compute_ego_forward_speed(metadata)

        adversary_speed = float(
            getattr(
                self.config,
                "adversary_speed_mps",
                self.config.relative_speed_mps
            )
        )

        closing_speed = ego_speed + adversary_speed

        max_closing_speed = float(
            getattr(self.config, "max_closing_speed_mps", 25.0)
        )

        closing_speed = max(0.0, min(closing_speed, max_closing_speed))

        return closing_speed


    def _compute_ego_forward_speed(self, metadata):
        """
        Compute ego speed along its forward direction.

        Uses:
            ego_transform.rotation.yaw
            ego_velocity.x/y/z

        Returns:
            forward speed in m/s.
        """

        ego_transform = metadata.get("ego_transform", None)
        ego_velocity = metadata.get("ego_velocity", None)

        if ego_transform is None or ego_velocity is None:
            return 0.0

        import math

        yaw = math.radians(ego_transform.rotation.yaw)

        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)

        vx = float(ego_velocity.x)
        vy = float(ego_velocity.y)

        forward_speed = vx * forward_x + vy * forward_y

    # If ego is reversing, do not let it create negative closing speed.
        return max(0.0, forward_speed)

    def _create_adversary_state(self):
        return AdversaryState(
            object_id=self.config.object_id,
            object_class=self.config.object_class,
            scenario_type="wrong_way_vehicle",
            initial_distance_m=self.config.initial_distance_m,
            min_distance_m=self.config.min_distance_m,
            approach_speed_mps=self.config.relative_speed_mps,
        )

    def _project_to_box(self, world_xyz, yaw, metadata):
        camera_transform = metadata.get("camera_transform", None)
        camera_intrinsics = metadata.get("camera_intrinsics", None)

        if camera_transform is None or camera_intrinsics is None:
            return None

        if world_xyz is None or yaw is None:
            return None

        vehicle_center_xyz = [
            world_xyz[0],
            world_xyz[1],
            world_xyz[2] + self.config.vehicle_height_m * 0.5
        ]

        box_2d = project_3d_box_to_2d(
            center_xyz=vehicle_center_xyz,
            yaw_degrees=yaw,
            camera_transform=camera_transform,
            camera_intrinsics=camera_intrinsics,
            length_m=self.config.vehicle_length_m,
            width_m=self.config.vehicle_width_m,
            height_m=self.config.vehicle_height_m,
            image_width=self.config.image_width,
            image_height=self.config.image_height
        )

        return box_2d
    
    