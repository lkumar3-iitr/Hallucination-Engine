from HallucinationEngine.config import HEConfig
from HallucinationEngine.scenarios.scenario_manager import ScenarioManager
from HallucinationEngine.rendering.box_renderer import BoxRenderer
from HallucinationEngine.rendering.cuboid_renderer import CuboidRenderer
from HallucinationEngine.rendering.sprite_renderer import SpriteRenderer
class HallucinationEngine:
    """
    Main Hallucination Engine.

    Input:
        RGB frame from CARLA
        frame-level metadata

    Output:
        modified RGB frame
        updated metadata
    """

    def __init__(self, config=None):
        self.config = config if config is not None else HEConfig()

        self.scenario_manager = ScenarioManager(self.config)
        self.renderer = self._create_renderer()

    def process(self, frame, metadata):
        """
        Main HE processing function.

        Args:
            frame: RGB image from CARLA camera as numpy array
            metadata: dictionary with frame id, ego state, route, camera info, etc.

        Returns:
            modified_frame, updated_metadata
        """

        if not self.config.enabled:
            return frame, metadata

        hallucination = self.scenario_manager.update(metadata)

        modified_frame = self.renderer.render(
            frame=frame,
            hallucination=hallucination,
            metadata=metadata
        )

        updated_metadata = self._update_metadata(metadata, hallucination)

        return modified_frame, updated_metadata

    

    def _create_renderer(self):
        if self.config.render_mode == "box":
            return BoxRenderer(self.config)

        if self.config.render_mode == "cuboid":
            return CuboidRenderer(self.config)

        if self.config.render_mode == "sprite":
            return SpriteRenderer(self.config)

        raise ValueError(f"Unknown HE render mode: {self.config.render_mode}")

    def _update_metadata(self, metadata, hallucination):
        """
        Adds HE information to the original frame metadata.
        """

        updated = dict(metadata)

        if hallucination is None:
            updated["hallucination"] = {
                "active": False
            }
        else:
            updated["hallucination"] = hallucination

        return updated