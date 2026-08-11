from .base_scenario import BaseScenario

from HallucinationEngine.placement.lane_aware_placement import LaneAwarePlacement
from HallucinationEngine.placement.projection_utils import project_3d_box_to_2d
from HallucinationEngine.motion.adversary_state import AdversaryState


class StoppedVehicleScenario(BaseScenario):
    """
    Hallucinated stopped vehicle ahead in ego lane.

    Correct behavior:
        - Select location once when scenario starts.
        - Keep world_xyz fixed.
        - Ego vehicle naturally approaches the stopped vehicle.
    """

    def __init__(self, config):
        super().__init__(config)

        self.placement = LaneAwarePlacement(config)
        self.adversary_state = None

        # Important: fixed world placement for stopped vehicle
        self.fixed_placement = None

    def update(self, metadata):
        frame_id = metadata.get("frame", 0)
        dt = metadata.get("dt", 1.0 / 20.0)

        if not self.active and self.should_start(frame_id):
            self.active = True
            self.start_frame = frame_id
            self.adversary_state = self._create_adversary_state()

            # Choose stopped vehicle location only once
            self.fixed_placement = self.placement.choose_location_ahead(
                metadata=metadata,
                distance_ahead_m=self.config.initial_distance_m
            )

            self.adversary_state.update_placement(
                placement=self.fixed_placement,
                lateral_offset_m=self.config.lateral_offset_m
            )

            # Stopped vehicle faces same route direction
            if self.fixed_placement is not None and self.fixed_placement.get("waypoint", None) is not None:
                wp = self.fixed_placement["waypoint"]
                self.adversary_state.yaw = wp.transform.rotation.yaw

        if not self.active:
            return None

        if not self.is_within_duration(frame_id):
            self.active = False

            if self.adversary_state is not None:
                self.adversary_state.active = False

            return None

        if self.adversary_state is None:
            self.adversary_state = self._create_adversary_state()

        # Do not move the stopped vehicle.
        # Only update age.
        self.adversary_state.age_frames += 1
        self.adversary_state.age_seconds += dt

        # Keep the same world placement fixed.
        if self.fixed_placement is not None:
            self.adversary_state.update_placement(
                placement=self.fixed_placement,
                lateral_offset_m=self.config.lateral_offset_m
            )

            if self.fixed_placement.get("waypoint", None) is not None:
                wp = self.fixed_placement["waypoint"]
                self.adversary_state.yaw = wp.transform.rotation.yaw

        # Update ego-relative distance for metadata only.
        self._update_distance_to_ego(metadata)

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

        hallucination = self.adversary_state.to_dict()

        hallucination.update({
            "box_2d": box_2d,
            "motion_type": "stopped_vehicle_fixed_world",
            "relative_speed_mps": 0.0,
            "render_mode": self.config.render_mode,
            "confidence": 1.0,
            "projection_status": projection_status,
        })

        return hallucination

    def _create_adversary_state(self):
        return AdversaryState(
            object_id=self.config.object_id,
            object_class=self.config.object_class,
            scenario_type="stopped_vehicle",
            initial_distance_m=self.config.initial_distance_m,
            min_distance_m=self.config.min_distance_m,
            approach_speed_mps=0.0,
        )

    def _update_distance_to_ego(self, metadata):
        """
        Compute actual distance from ego to fixed hallucinated vehicle.

        This is metadata only.
        It does not move the hallucinated vehicle.
        """

        ego_transform = metadata.get("ego_transform", None)

        if ego_transform is None:
            return

        if self.adversary_state is None:
            return

        if self.adversary_state.world_xyz is None:
            return

        ego_loc = ego_transform.location

        wx, wy, wz = self.adversary_state.world_xyz

        dx = wx - ego_loc.x
        dy = wy - ego_loc.y
        dz = wz - ego_loc.z

        dist = (dx * dx + dy * dy + dz * dz) ** 0.5

        self.adversary_state.distance_ahead_m = dist
        self.adversary_state.ego_local_xyz = [
            dist,
            self.config.lateral_offset_m,
            0.0
        ]

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

        return project_3d_box_to_2d(
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