from HallucinationEngine.insertion import ObjectInserter


class SpriteRenderer:
    """
    Sprite renderer for HE.

    This renderer does not do blending itself anymore.
    It delegates object insertion to ObjectInserter.

    Benefit:
        - current CARLA HE uses ObjectInserter
        - future CameraOnlyHE will also use ObjectInserter
        - realism improvements are centralized
    """

    def __init__(self, config):
        self.config = config
        self.inserter = ObjectInserter()

    def render(self, frame, hallucination, metadata=None):
        """
        Args:
            frame: RGB image
            hallucination: metadata dictionary containing box_2d and scenario_type
            metadata: optional frame metadata

        Returns:
            RGB image with inserted object
        """

        if hallucination is None:
            return frame

        if not hallucination.get("active", False):
            return frame

        box = hallucination.get("box_2d", None)

        if box is None:
            return frame

        x1, y1, x2, y2 = box

        if x2 <= x1 or y2 <= y1:
            return frame

        sprite_path = self._select_sprite_path(hallucination)

        modified_frame = self.inserter.insert_sprite(
            frame_rgb=frame,
            sprite_path=sprite_path,
            box_2d=box,

            alpha=getattr(self.config, "sprite_alpha", 1.0),
            horizontal_scale=getattr(self.config, "sprite_horizontal_scale", 1.10),
            vertical_scale=getattr(self.config, "sprite_vertical_scale", 1.15),
            y_offset_ratio=getattr(self.config, "sprite_y_offset_ratio", 0.0),

            match_brightness=getattr(self.config, "match_brightness", True),
            add_shadow=getattr(self.config, "add_shadow", True),
            soften_edges=getattr(self.config, "soften_edges", True),
            motion_blur=getattr(self.config, "motion_blur", False),

            debug_box=getattr(self.config, "draw_sprite_debug_box", False),
            debug_color=getattr(self.config, "box_color", (255, 0, 0)),
        )

        return modified_frame

    def _select_sprite_path(self, hallucination):
        """
        Select sprite path based on scenario type.

        If scenario-specific paths are not configured,
        fall back to config.sprite_path.
        """

        scenario_type = hallucination.get("scenario_type", None)

        if scenario_type == "wrong_way_vehicle":
            return getattr(
                self.config,
                "wrong_way_sprite_path",
                getattr(self.config, "sprite_path", "assets/sprites/car_front.png")
            )

        if scenario_type == "stopped_vehicle":
            return getattr(
                self.config,
                "stopped_vehicle_sprite_path",
                getattr(self.config, "sprite_path", "assets/sprites/car_front.png")
            )

        return getattr(self.config, "sprite_path", "assets/sprites/car_front.png")