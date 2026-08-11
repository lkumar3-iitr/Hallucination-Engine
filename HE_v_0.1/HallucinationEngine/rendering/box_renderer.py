import cv2


class BoxRenderer:
    """
    First simple renderer.

    It draws a 2D bounding box on the RGB frame.
    This is only for validating the HE pipeline.
    """

    def __init__(self, config):
        self.config = config

    def render(self, frame, hallucination, metadata=None):
        """
        Args:
            frame: RGB image as numpy array
            hallucination: hallucination dictionary
            metadata: optional metadata

        Returns:
            modified frame
        """

        if hallucination is None:
            return frame

        if not hallucination.get("active", False):
            return frame

        box = hallucination.get("box_2d", None)
        if box is None:
            return frame

        x1, y1, x2, y2 = box

        modified = frame.copy()

        # If your CARLA frame is RGB and cv2 drawing is used,
        # color ordering visually may look swapped.
        # For now it is fine for debugging.
        color = self.config.box_color
        thickness = self.config.box_thickness

        cv2.rectangle(
            modified,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        label = hallucination.get("scenario_type", "HE object")
        distance = hallucination.get("distance_to_ego_m", None)

        if distance is not None:
            label = f"{label}: {distance:.1f}m"

        cv2.putText(
            modified,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

        return modified