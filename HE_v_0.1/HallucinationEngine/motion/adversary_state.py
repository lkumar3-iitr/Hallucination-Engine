class AdversaryState:
    """
    Persistent state for a hallucinated adversary.

    This object stores the virtual adversary across frames.
    It does not render anything by itself.
    """

    def __init__(
        self,
        object_id,
        object_class,
        scenario_type,
        initial_distance_m,
        min_distance_m,
        approach_speed_mps,
    ):
        self.object_id = object_id
        self.object_class = object_class
        self.scenario_type = scenario_type

        self.active = True

        # Ego-relative longitudinal distance along route
        self.distance_ahead_m = float(initial_distance_m)
        self.min_distance_m = float(min_distance_m)

        # Positive value means the adversary is approaching ego.
        self.approach_speed_mps = float(approach_speed_mps)

        # Updated every frame after lane placement
        self.world_xyz = None
        self.ego_local_xyz = None
        self.yaw = None
        self.lane_id = None
        self.road_id = None

        # Tracking information
        self.age_frames = 0
        self.age_seconds = 0.0

    def update_motion(self, dt, closing_speed_mps=None):
        """
        Move adversary closer to ego.

        Args:
            dt: simulation time step
            closing_speed_mps:
                If provided, use this as the actual closing speed.
                If None, fall back to self.approach_speed_mps.
        """

        dt = float(dt)

        if closing_speed_mps is None:
            speed = self.approach_speed_mps
        else:
            speed = float(closing_speed_mps)

        self.distance_ahead_m -= speed * dt

        if self.distance_ahead_m < self.min_distance_m:
            self.distance_ahead_m = self.min_distance_m

        self.age_frames += 1
        self.age_seconds += dt

    def update_placement(self, placement, lateral_offset_m):
        """
        Update world pose from lane-aware placement.
        """

        if placement is None:
            self.world_xyz = None
            self.yaw = None
            self.lane_id = None
            self.road_id = None
            self.ego_local_xyz = [
                self.distance_ahead_m,
                lateral_offset_m,
                0.0
            ]
            return

        self.world_xyz = placement.get("world_xyz", None)
        self.yaw = placement.get("yaw", None)
        self.lane_id = placement.get("lane_id", None)
        self.road_id = placement.get("road_id", None)

        self.ego_local_xyz = [
            self.distance_ahead_m,
            lateral_offset_m,
            0.0
        ]

    def to_dict(self):
        """
        Convert state to serializable metadata.
        """

        return {
            "object_id": self.object_id,
            "object_class": self.object_class,
            "scenario_type": self.scenario_type,
            "active": self.active,

            "distance_to_ego_m": self.distance_ahead_m,
            "approach_speed_mps": self.approach_speed_mps,
            "configured_approach_speed_mps": self.approach_speed_mps,
            "world_xyz": self.world_xyz,
            "ego_local_xyz": self.ego_local_xyz,
            "yaw": self.yaw,
            "lane_id": self.lane_id,
            "road_id": self.road_id,

            "age_frames": self.age_frames,
            "age_seconds": self.age_seconds,
        }