class ImageAdversaryTrack:
    """
    Image-space adversary track.

    This is camera-only.
    It does not know CARLA world coordinates.

    For wrong-way vehicle:
        object starts near the horizon
        moves downward
        grows larger
    """

    def __init__(
        self,
        object_id,
        scenario_type,
        start_center,
        target_center,
        duration_frames,
        initial_progress=0.0,
    ):
        self.object_id = object_id
        self.scenario_type = scenario_type

        self.start_u, self.start_v = start_center
        self.target_u, self.target_v = target_center

        self.duration_frames = max(1, int(duration_frames))

        self.progress = float(initial_progress)
        self.active = True

        self.age_frames = 0
        self.age_seconds = 0.0

        self.center_u = self.start_u
        self.center_v = self.start_v

        self.box_2d = None

    def update(self, dt):
        """
        Update image-space motion.
        """

        self.age_frames += 1
        self.age_seconds += float(dt)

        # Move from start to target over duration.
        self.progress = self.age_frames / float(self.duration_frames)
        self.progress = max(0.0, min(self.progress, 1.0))

        # Use eased progress so motion starts slowly and accelerates.
        p = self._ease_in(self.progress)

        self.center_u = self.start_u + p * (self.target_u - self.start_u)
        self.center_v = self.start_v + p * (self.target_v - self.start_v)

        if self.progress >= 1.0:
            # Keep active but saturated at target for now.
            self.center_u = self.target_u
            self.center_v = self.target_v

    def set_box(self, box_2d):
        self.box_2d = box_2d

    def to_dict(self):
        return {
            "object_id": self.object_id,
            "scenario_type": self.scenario_type,
            "active": self.active,

            "image_center": [
                float(self.center_u),
                float(self.center_v),
            ],
            "box_2d": self.box_2d,

            "progress": float(self.progress),
            "age_frames": int(self.age_frames),
            "age_seconds": float(self.age_seconds),

            "coordinate_mode": "image_space",
            "world_xyz": None,
            "yaw": None,
        }

    def _ease_in(self, x):
        """
        Simple nonlinear motion.

        This makes the approaching object grow/move faster later,
        similar to perspective approach.
        """

        x = max(0.0, min(float(x), 1.0))
        return x * x