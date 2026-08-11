from he_placement_model import HEPlacementModel, build_camera_from_fov


def placement_to_sprite_rect(placement):
    """
    Convert HEPlacementModel output to compositor rectangle.

    Model output:
        center_x, bottom_y, box_width, box_height

    Compositor needs:
        top-left x, top-left y, width, height
    """
    center_x = placement["center_x"]
    bottom_y = placement["bottom_y"]
    box_width = placement["box_width"]
    box_height = placement["box_height"]

    return {
        "x": center_x - box_width / 2.0,
        "y": bottom_y - box_height,
        "w": box_width,
        "h": box_height,
        "center_x": center_x,
        "bottom_y": bottom_y,
        "visible": placement["visible"],
        "visible_prob": placement["visible_prob"],
    }


class HEPlacementRuntimeAdapter:
    def __init__(
        self,
        checkpoint_path,
        image_width=1280,
        image_height=720,
        fov=90.0,
        device="cpu",
    ):
        self.model = HEPlacementModel(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        self.camera = build_camera_from_fov(
            width=image_width,
            height=image_height,
            fov=fov,
        )

    def predict_sprite_rect(
        self,
        rel_x,
        rel_z,
        rel_yaw,
        rel_y=None,
        visible_threshold=0.5,
    ):
        relative_state = {
            "rel_x": float(rel_x),
            "rel_z": float(rel_z),
            "rel_yaw": float(rel_yaw),
        }

        if rel_y is not None:
            relative_state["rel_y"] = float(rel_y)

        placement = self.model.predict(
            relative_state=relative_state,
            camera=self.camera,
            visible_threshold=visible_threshold,
        )

        return placement_to_sprite_rect(placement)

    def predict_from_adversary_state(
        self,
        adversary_state,
        visible_threshold=0.5,
    ):
        """
        Expected adversary_state keys:
            rel_x
            rel_z
            rel_yaw

        Optional:
            rel_y
        """
        return self.predict_sprite_rect(
            rel_x=adversary_state["rel_x"],
            rel_z=adversary_state["rel_z"],
            rel_y=adversary_state.get("rel_y", None),
            rel_yaw=adversary_state["rel_yaw"],
            visible_threshold=visible_threshold,
        )