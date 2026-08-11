class SimpleRoadModel:
    """
    Simple camera-only road model.

    This does not use CARLA world information.
    It only gives approximate image-space positions for object insertion.

    Assumption:
        front-facing dashcam image
        road center roughly near image center
        horizon/vanishing area near upper-middle frame
    """

    def __init__(
        self,
        image_width,
        image_height,
        road_center_x_ratio=0.5,
        horizon_y_ratio=0.45,
        spawn_y_ratio=0.48,
        target_y_ratio=0.82,
    ):
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        self.road_center_x_ratio = float(road_center_x_ratio)
        self.horizon_y_ratio = float(horizon_y_ratio)
        self.spawn_y_ratio = float(spawn_y_ratio)
        self.target_y_ratio = float(target_y_ratio)

    def get_road_center_x(self):
        return self.image_width * self.road_center_x_ratio

    def get_horizon_y(self):
        return self.image_height * self.horizon_y_ratio

    def get_spawn_point(self):
        """
        Initial location for far adversary.
        """

        u = self.image_width * self.road_center_x_ratio
        v = self.image_height * self.spawn_y_ratio

        return u, v

    def get_target_point(self):
        """
        Near location where approaching adversary becomes large.
        """

        u = self.image_width * self.road_center_x_ratio
        v = self.image_height * self.target_y_ratio

        return u, v

    def estimate_box_size_from_y(
        self,
        center_y,
        min_width=40,
        max_width=360,
        min_height=35,
        max_height=300,
    ):
        """
        Estimate apparent object size from vertical image position.

        Far near horizon:
            small box

        Near lower image:
            large box
        """

        horizon_y = self.get_horizon_y()
        bottom_y = self.image_height

        denom = max(1.0, bottom_y - horizon_y)

        depth_ratio = (center_y - horizon_y) / denom
        depth_ratio = max(0.0, min(depth_ratio, 1.0))

        # Nonlinear growth looks more natural than linear growth.
        scale = depth_ratio ** 1.4

        width = min_width + scale * (max_width - min_width)
        height = min_height + scale * (max_height - min_height)

        return width, height

    def make_box_from_center(self, center_x, center_y, width, height):
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0

        x1 = max(0, min(self.image_width - 1, x1))
        y1 = max(0, min(self.image_height - 1, y1))
        x2 = max(0, min(self.image_width - 1, x2))
        y2 = max(0, min(self.image_height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return [int(x1), int(y1), int(x2), int(y2)]